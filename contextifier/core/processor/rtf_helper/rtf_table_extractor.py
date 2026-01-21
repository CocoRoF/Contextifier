# contextifier/core/processor/rtf_helper/rtf_table_extractor.py
"""
RTF Table Extractor

Extracts and parses tables from RTF content.
Includes RTFCellInfo and RTFTable data models.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

from contextifier.core.processor.rtf_helper.rtf_decoder import (
    decode_hex_escapes,
)
from contextifier.core.processor.rtf_helper.rtf_text_cleaner import (
    clean_rtf_text,
)
from contextifier.core.processor.rtf_helper.rtf_region_finder import (
    find_excluded_regions,
    is_in_excluded_region,
)

logger = logging.getLogger("contextify.rtf.table")


# =============================================================================
# Data Models
# =============================================================================

class RTFCellInfo(NamedTuple):
    """RTF cell information with merge info."""
    text: str              # Cell text content
    h_merge_first: bool    # Horizontal merge start (clmgf)
    h_merge_cont: bool     # Horizontal merge continue (clmrg)
    v_merge_first: bool    # Vertical merge start (clvmgf)
    v_merge_cont: bool     # Vertical merge continue (clvmrg)
    right_boundary: int    # Cell right boundary (twips)


@dataclass
class RTFTable:
    """RTF table structure with merge cell support."""
    rows: List[List[RTFCellInfo]] = field(default_factory=list)
    col_count: int = 0
    position: int = 0      # Start position in document
    end_position: int = 0  # End position in document
    
    def is_real_table(self) -> bool:
        """
        Determine if this is a real table.
        
        n rows x 1 column is considered a list, not a table.
        """
        if not self.rows:
            return False
        
        effective_cols = self._get_effective_col_count()
        return effective_cols >= 2
    
    def _get_effective_col_count(self) -> int:
        """Calculate effective column count (excluding empty columns)."""
        if not self.rows:
            return 0
        
        effective_counts = []
        for row in self.rows:
            non_empty_cells = []
            for i, cell in enumerate(row):
                if cell.h_merge_cont:
                    continue
                if cell.text.strip() or cell.v_merge_first:
                    non_empty_cells.append(i)
            
            if non_empty_cells:
                effective_counts.append(max(non_empty_cells) + 1)
        
        return max(effective_counts) if effective_counts else 0
    
    def to_html(self) -> str:
        """Convert table to HTML with merge cell support."""
        if not self.rows:
            return ""
        
        merge_info = self._calculate_merge_info()
        html_parts = ['<table border="1">']
        
        for row_idx, row in enumerate(self.rows):
            html_parts.append('<tr>')
            
            for col_idx, cell in enumerate(row):
                if col_idx < len(merge_info[row_idx]):
                    colspan, rowspan = merge_info[row_idx][col_idx]
                    
                    if colspan == 0 or rowspan == 0:
                        continue
                    
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    
                    attrs = []
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    
                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    html_parts.append(f'<td{attr_str}>{cell_text}</td>')
                else:
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    html_parts.append(f'<td>{cell_text}</td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        return '\n'.join(html_parts)
    
    def to_text_list(self) -> str:
        """
        Convert 1-column table to text list.
        
        - 1x1 table: Return cell content only (container table)
        - nx1 table: Return rows separated by blank lines
        """
        if not self.rows:
            return ""
        
        if len(self.rows) == 1 and len(self.rows[0]) == 1:
            return self.rows[0][0].text
        
        lines = []
        for row in self.rows:
            if row:
                cell_text = row[0].text
                if cell_text:
                    lines.append(cell_text)
        
        return '\n\n'.join(lines)
    
    def _calculate_merge_info(self) -> List[List[tuple]]:
        """Calculate colspan and rowspan for each cell."""
        if not self.rows:
            return []
        
        num_rows = len(self.rows)
        max_cols = max(len(row) for row in self.rows) if self.rows else 0
        
        if max_cols == 0:
            return []
        
        # Initialize with (1, 1) for all cells
        merge_info = [[(1, 1) for _ in range(max_cols)] for _ in range(num_rows)]
        
        # Process horizontal merges
        for row_idx, row in enumerate(self.rows):
            col_idx = 0
            while col_idx < len(row):
                cell = row[col_idx]
                
                if cell.h_merge_first:
                    colspan = 1
                    for next_col in range(col_idx + 1, len(row)):
                        if row[next_col].h_merge_cont:
                            colspan += 1
                            merge_info[row_idx][next_col] = (0, 0)
                        else:
                            break
                    merge_info[row_idx][col_idx] = (colspan, 1)
                
                col_idx += 1
        
        # Process vertical merges
        for col_idx in range(max_cols):
            row_idx = 0
            while row_idx < num_rows:
                if col_idx >= len(self.rows[row_idx]):
                    row_idx += 1
                    continue
                
                cell = self.rows[row_idx][col_idx]
                
                if cell.v_merge_first:
                    rowspan = 1
                    for next_row in range(row_idx + 1, num_rows):
                        if col_idx < len(self.rows[next_row]) and self.rows[next_row][col_idx].v_merge_cont:
                            rowspan += 1
                            merge_info[next_row][col_idx] = (0, 0)
                        else:
                            break
                    
                    current_colspan = merge_info[row_idx][col_idx][0]
                    merge_info[row_idx][col_idx] = (current_colspan, rowspan)
                    row_idx += rowspan
                elif cell.v_merge_cont:
                    merge_info[row_idx][col_idx] = (0, 0)
                    row_idx += 1
                else:
                    row_idx += 1
        
        return merge_info


# =============================================================================
# Table Extraction Functions
# =============================================================================

def extract_tables_with_positions(
    content: str,
    encoding: str = "cp949"
) -> Tuple[List[RTFTable], List[Tuple[int, int, RTFTable]]]:
    """
    Extract tables from RTF content with position information.
    
    RTF table structure:
    - \\trowd: Table row start (row definition)
    - \\cellxN: Cell boundary position
    - \\clmgf: Horizontal merge start
    - \\clmrg: Horizontal merge continue
    - \\clvmgf: Vertical merge start
    - \\clvmrg: Vertical merge continue
    - \\intbl: Paragraph in cell
    - \\cell: Cell end
    - \\row: Row end
    
    Args:
        content: RTF string content
        encoding: Encoding to use
        
    Returns:
        Tuple of (table list, table region list [(start, end, table), ...])
    """
    tables = []
    table_regions = []
    
    # Find excluded regions (header, footer, footnote, etc.)
    excluded_regions = find_excluded_regions(content)
    
    # Step 1: Find all \row positions
    row_positions = []
    for match in re.finditer(r'\\row(?![a-z])', content):
        row_positions.append(match.end())
    
    if not row_positions:
        return tables, table_regions
    
    # Step 2: Find \trowd before each \row
    all_rows = []
    for i, row_end in enumerate(row_positions):
        if i == 0:
            search_start = 0
        else:
            search_start = row_positions[i - 1]
        
        segment = content[search_start:row_end]
        trowd_match = re.search(r'\\trowd', segment)
        
        if trowd_match:
            row_start = search_start + trowd_match.start()
            
            # Skip rows in excluded regions
            if is_in_excluded_region(row_start, excluded_regions):
                logger.debug(f"Skipping table row at {row_start} (in header/footer/footnote)")
                continue
            
            row_text = content[row_start:row_end]
            all_rows.append((row_start, row_end, row_text))
    
    if not all_rows:
        return tables, table_regions
    
    # Group consecutive rows into tables
    table_groups = []
    current_table = []
    current_start = -1
    current_end = -1
    prev_end = -1
    
    for row_start, row_end, row_text in all_rows:
        # Rows within 150 chars are same table
        if prev_end == -1 or row_start - prev_end < 150:
            if current_start == -1:
                current_start = row_start
            current_table.append(row_text)
            current_end = row_end
        else:
            if current_table:
                table_groups.append((current_start, current_end, current_table))
            current_table = [row_text]
            current_start = row_start
            current_end = row_end
        prev_end = row_end
    
    if current_table:
        table_groups.append((current_start, current_end, current_table))
    
    logger.info(f"Found {len(table_groups)} table groups")
    
    # Parse each table group
    for start_pos, end_pos, table_rows in table_groups:
        table = _parse_table_with_merge(table_rows, encoding)
        if table and table.rows:
            table.position = start_pos
            table.end_position = end_pos
            tables.append(table)
            table_regions.append((start_pos, end_pos, table))
    
    logger.info(f"Extracted {len(tables)} tables")
    return tables, table_regions


def _parse_table_with_merge(rows: List[str], encoding: str = "cp949") -> Optional[RTFTable]:
    """
    Parse table rows to RTFTable object with merge support.
    
    Args:
        rows: Table row text list
        encoding: Encoding to use
        
    Returns:
        RTFTable object
    """
    table = RTFTable()
    
    for row_text in rows:
        cells = _extract_cells_with_merge(row_text, encoding)
        if cells:
            table.rows.append(cells)
            if len(cells) > table.col_count:
                table.col_count = len(cells)
    
    return table if table.rows else None


def _extract_cells_with_merge(row_text: str, encoding: str = "cp949") -> List[RTFCellInfo]:
    """
    Extract cell content and merge information from table row.
    
    Args:
        row_text: Table row RTF text
        encoding: Encoding to use
        
    Returns:
        List of RTFCellInfo
    """
    cells = []
    
    # Step 1: Parse cell definitions (attributes before cellx)
    cell_defs = []
    
    # Find first \cell that is not \cellx
    first_cell_idx = -1
    pos = 0
    while True:
        idx = row_text.find('\\cell', pos)
        if idx == -1:
            first_cell_idx = len(row_text)
            break
        if idx + 5 < len(row_text) and row_text[idx + 5] == 'x':
            pos = idx + 1
            continue
        first_cell_idx = idx
        break
    
    def_part = row_text[:first_cell_idx]
    
    current_def = {
        'h_merge_first': False,
        'h_merge_cont': False,
        'v_merge_first': False,
        'v_merge_cont': False,
        'right_boundary': 0
    }
    
    cell_def_pattern = r'\\cl(?:mgf|mrg|vmgf|vmrg)|\\cellx(-?\d+)'
    
    for match in re.finditer(cell_def_pattern, def_part):
        token = match.group()
        if token == '\\clmgf':
            current_def['h_merge_first'] = True
        elif token == '\\clmrg':
            current_def['h_merge_cont'] = True
        elif token == '\\clvmgf':
            current_def['v_merge_first'] = True
        elif token == '\\clvmrg':
            current_def['v_merge_cont'] = True
        elif token.startswith('\\cellx'):
            if match.group(1):
                current_def['right_boundary'] = int(match.group(1))
            cell_defs.append(current_def.copy())
            current_def = {
                'h_merge_first': False,
                'h_merge_cont': False,
                'v_merge_first': False,
                'v_merge_cont': False,
                'right_boundary': 0
            }
    
    # Step 2: Extract cell texts
    cell_texts = _extract_cell_texts(row_text, encoding)
    
    # Step 3: Match cell definitions with content
    for i, cell_text in enumerate(cell_texts):
        if i < len(cell_defs):
            cell_def = cell_defs[i]
        else:
            cell_def = {
                'h_merge_first': False,
                'h_merge_cont': False,
                'v_merge_first': False,
                'v_merge_cont': False,
                'right_boundary': 0
            }
        
        cells.append(RTFCellInfo(
            text=cell_text,
            h_merge_first=cell_def['h_merge_first'],
            h_merge_cont=cell_def['h_merge_cont'],
            v_merge_first=cell_def['v_merge_first'],
            v_merge_cont=cell_def['v_merge_cont'],
            right_boundary=cell_def['right_boundary']
        ))
    
    return cells


def _extract_cell_texts(row_text: str, encoding: str = "cp949") -> List[str]:
    """
    Extract cell texts from row.
    
    Args:
        row_text: Table row RTF text
        encoding: Encoding to use
        
    Returns:
        List of cell texts
    """
    cell_texts = []
    
    # Step 1: Find all \cell positions (not \cellx)
    cell_positions = []
    pos = 0
    while True:
        idx = row_text.find('\\cell', pos)
        if idx == -1:
            break
        next_pos = idx + 5
        if next_pos < len(row_text) and row_text[next_pos] == 'x':
            pos = idx + 1
            continue
        cell_positions.append(idx)
        pos = idx + 1
    
    if not cell_positions:
        return cell_texts
    
    # Step 2: Find last \cellx before first \cell
    first_cell_pos = cell_positions[0]
    def_part = row_text[:first_cell_pos]
    
    last_cellx_end = 0
    for match in re.finditer(r'\\cellx-?\d+', def_part):
        last_cellx_end = match.end()
    
    # Step 3: Extract each cell content
    prev_end = last_cellx_end
    for cell_end in cell_positions:
        cell_content = row_text[prev_end:cell_end]
        
        # RTF decoding and cleaning
        decoded = decode_hex_escapes(cell_content, encoding)
        clean = clean_rtf_text(decoded, encoding)
        cell_texts.append(clean)
        
        prev_end = cell_end + 5  # len('\\cell') = 5
    
    return cell_texts


__all__ = [
    'RTFCellInfo',
    'RTFTable',
    'extract_tables_with_positions',
]
