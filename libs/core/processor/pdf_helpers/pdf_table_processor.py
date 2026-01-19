# libs/core/processor/pdf_helpers/pdf_table_processor.py
"""
PDF Table Processing Module

Provides functions for table extraction, merging, annotation integration,
and HTML conversion from PDF documents.
"""
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from libs.core.processor.pdf_helpers.types import (
    TableDetectionStrategy,
    ElementType,
    PDFConfig,
    PageElement,
    PageBorderInfo,
    CellInfo,
)
from libs.core.processor.pdf_helpers.pdf_utils import (
    escape_html,
    get_text_lines_with_positions,
)
from libs.core.processor.pdf_helpers.pdf_table_detection import TableDetectionEngine
from libs.core.processor.pdf_helpers.pdf_cell_analysis import CellAnalysisEngine

logger = logging.getLogger("document-processor")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AnnotationInfo:
    """Annotation/footnote/endnote info."""
    text: str
    bbox: Tuple[float, float, float, float]
    type: str  # 'footnote', 'endnote', 'table_note'
    related_table_idx: Optional[int] = None


@dataclass
class TableInfo:
    """Final table info."""
    page_num: int
    table_idx: int
    bbox: Tuple[float, float, float, float]
    data: List[List[Optional[str]]]
    col_count: int
    row_count: int
    page_height: float
    cells_info: Optional[List[Dict]] = None
    annotations: Optional[List[AnnotationInfo]] = None
    detection_strategy: Optional[TableDetectionStrategy] = None
    confidence: float = 1.0


# ============================================================================
# Table Extraction
# ============================================================================

def extract_all_tables(
    doc,
    file_path: str,
    detect_page_border_func,
    is_table_likely_border_func
) -> Dict[int, List[PageElement]]:
    """
    Extracts tables from entire document.

    Strategy:
    1. Multi-strategy table detection
    2. Select best result based on confidence
    3. Cell analysis and merge cell processing
    4. Annotation integration
    5. Cross-page continuity handling

    Args:
        doc: PyMuPDF document object
        file_path: PDF file path
        detect_page_border_func: Function to detect page borders
        is_table_likely_border_func: Function to check if table is a border

    Returns:
        Dictionary mapping page numbers to list of table PageElements
    """
    tables_by_page: Dict[int, List[PageElement]] = {}
    all_table_infos: List[TableInfo] = []

    # Step 1: Detect tables on each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height

        # Detect page border
        border_info = detect_page_border_func(page)

        try:
            # Use table detection engine
            detection_engine = TableDetectionEngine(page, page_num, file_path)
            candidates = detection_engine.detect_tables()

            for idx, candidate in enumerate(candidates):
                # Check if overlaps with page border
                if border_info.has_border and is_table_likely_border_func(
                    candidate.bbox, border_info, page
                ):
                    logger.debug(f"[PDF] Skipping page border table: {candidate.bbox}")
                    continue

                # Convert cell info to dictionary
                cells_info = None
                if candidate.cells:
                    cells_info = [
                        {
                            'row': cell.row,
                            'col': cell.col,
                            'rowspan': cell.rowspan,
                            'colspan': cell.colspan,
                            'bbox': cell.bbox
                        }
                        for cell in candidate.cells
                    ]

                table_info = TableInfo(
                    page_num=page_num,
                    table_idx=idx,
                    bbox=candidate.bbox,
                    data=candidate.data,
                    col_count=candidate.col_count,
                    row_count=candidate.row_count,
                    page_height=page_height,
                    cells_info=cells_info,
                    detection_strategy=candidate.strategy,
                    confidence=candidate.confidence
                )

                all_table_infos.append(table_info)

        except Exception as e:
            logger.debug(f"[PDF] Error detecting tables on page {page_num}: {e}")
            continue

    # Step 2: Merge adjacent tables
    merged_tables = merge_adjacent_tables(all_table_infos)

    # Step 3: Find and insert annotations
    merged_tables = find_and_insert_annotations(doc, merged_tables)

    # Step 4: Handle table continuity
    processed_tables = process_table_continuity(merged_tables)

    # Step 5: HTML conversion and PageElement creation
    # Single-column tables as TEXT, 2+ columns as TABLE
    single_col_count = 0
    real_table_count = 0

    for table_info in processed_tables:
        try:
            page_num = table_info.page_num

            if page_num not in tables_by_page:
                tables_by_page[page_num] = []

            # Check if single-column table
            if is_single_column_table(table_info):
                # Single-column table: convert to text list as TEXT type
                text_content = convert_single_column_to_text(table_info)

                if text_content:
                    tables_by_page[page_num].append(PageElement(
                        element_type=ElementType.TEXT,
                        content=text_content,
                        bbox=table_info.bbox,
                        page_num=page_num
                    ))
                    single_col_count += 1
            else:
                # 2+ columns: convert to HTML table
                html_table = convert_table_to_html(table_info)

                if html_table:
                    tables_by_page[page_num].append(PageElement(
                        element_type=ElementType.TABLE,
                        content=html_table,
                        bbox=table_info.bbox,
                        page_num=page_num
                    ))
                    real_table_count += 1

        except Exception as e:
            logger.debug(f"[PDF] Error converting table to HTML: {e}")
            continue

    if single_col_count > 0:
        logger.info(f"[PDF] Converted {single_col_count} single-column tables to text")
    logger.info(f"[PDF] Extracted {real_table_count} tables from {len(tables_by_page)} pages")
    return tables_by_page


# ============================================================================
# Annotation Integration
# ============================================================================

def find_and_insert_annotations(doc, tables: List[TableInfo]) -> List[TableInfo]:
    """
    Finds and integrates annotations/footnotes/endnotes inside and after tables.

    Detection patterns:
    1. Rows starting with "Note)" etc. right after table
    2. Subheader rows inside table (e.g., (A), (B))
    3. Footnote/endnote markers (※, *, †, ‡, etc.)

    Args:
        doc: PyMuPDF document object
        tables: List of TableInfo

    Returns:
        Updated list of TableInfo with annotations
    """
    if not tables:
        return tables

    result = []
    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)

    for table in tables:
        tables_by_page[table.page_num].append(table)

    for page_num, page_tables in tables_by_page.items():
        page = doc[page_num]
        page_height = page.rect.height

        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])
        text_lines = get_text_lines_with_positions(page)

        for i, table in enumerate(sorted_tables):
            table_top = table.bbox[1]
            table_bottom = table.bbox[3]
            table_left = table.bbox[0]
            table_right = table.bbox[2]

            next_table_top = sorted_tables[i + 1].bbox[1] if i + 1 < len(sorted_tables) else page_height

            # 1. Find annotation rows right after table
            annotation_lines = []
            for line in text_lines:
                # Right below table, before next table
                if table_bottom - 3 <= line['y0'] <= table_bottom + PDFConfig.ANNOTATION_Y_MARGIN:
                    if line['x0'] >= table_left - 10 and line['x1'] <= table_right + 10:
                        if line['y0'] < next_table_top - 20:
                            # Check annotation pattern
                            for pattern in PDFConfig.ANNOTATION_PATTERNS:
                                if line['text'].startswith(pattern):
                                    annotation_lines.append(line)
                                    break

            if annotation_lines:
                table = add_annotation_to_table(table, annotation_lines, 'footer')
                logger.debug(f"[PDF] Added annotation to table on page {page_num + 1}")

            # 2. Find subheader rows (e.g., (A), (B)) - only when no subheader exists
            has_subheader = False
            if table.row_count >= 2 and table.data and len(table.data) >= 2:
                # Check if second row is subheader pattern
                second_row = table.data[1] if len(table.data) > 1 else []
                for cell in second_row:
                    if cell and ('(A)' in str(cell) or '(B)' in str(cell)):
                        has_subheader = True
                        break

            if not has_subheader and table.row_count >= 2 and table.data:
                row_height_estimate = (table_bottom - table_top) / table.row_count
                header_bottom_estimate = table_top + row_height_estimate
                second_row_top_estimate = table_top + row_height_estimate * 2

                subheader_lines = []
                for line in text_lines:
                    if header_bottom_estimate - 5 <= line['y0'] <= second_row_top_estimate - 5:
                        if line['x0'] >= table_left - 10 and line['x1'] <= table_right + 10:
                            # Check (A), (B) pattern
                            if '(A)' in line['text'] or '(B)' in line['text']:
                                subheader_lines.append(line)

                if subheader_lines:
                    table = add_annotation_to_table(table, subheader_lines, 'subheader')
                    logger.debug(f"[PDF] Added subheader to table on page {page_num + 1}")

            result.append(table)

    result.sort(key=lambda t: (t.page_num, t.bbox[1]))
    return result


def add_annotation_to_table(
    table: TableInfo,
    text_lines: List[Dict],
    position: str
) -> TableInfo:
    """
    Adds annotation rows to a table.

    Args:
        table: TableInfo object
        text_lines: List of text line dictionaries
        position: 'footer' or 'subheader'

    Returns:
        Updated TableInfo
    """
    if not text_lines:
        return table

    text_lines_sorted = sorted(text_lines, key=lambda l: l['x0'])

    table_width = table.bbox[2] - table.bbox[0]
    col_width = table_width / table.col_count if table.col_count > 0 else table_width

    new_row = [''] * table.col_count

    for line in text_lines_sorted:
        relative_x = line['x0'] - table.bbox[0]
        col_idx = min(int(relative_x / col_width), table.col_count - 1)
        col_idx = max(0, col_idx)

        if new_row[col_idx]:
            new_row[col_idx] += " " + line['text']
        else:
            new_row[col_idx] = line['text']

    non_empty_cols = sum(1 for c in new_row if c)
    if non_empty_cols == 1 and new_row[0]:
        combined_text = " ".join(line['text'] for line in text_lines_sorted)
        new_row = [combined_text] + [''] * (table.col_count - 1)

    new_data = list(table.data)

    # Update cell info
    new_cells_info = None
    if table.cells_info:
        new_cells_info = list(table.cells_info)
    else:
        new_cells_info = []

    if position == 'subheader':
        if len(new_data) > 0:
            new_data.insert(1, new_row)
            # Adjust existing cell info row indices (+1 for row >= 1)
            adjusted_cells = []
            for cell in new_cells_info:
                if cell['row'] >= 1:
                    adjusted_cell = dict(cell)
                    adjusted_cell['row'] = cell['row'] + 1
                    adjusted_cells.append(adjusted_cell)
                else:
                    adjusted_cells.append(cell)
            new_cells_info = adjusted_cells
            # Add cell info for new subheader row (each cell has colspan=1)
            for col_idx in range(table.col_count):
                new_cells_info.append({
                    'row': 1,
                    'col': col_idx,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })
        else:
            new_data.append(new_row)
    else:
        new_data.append(new_row)
        # Footer row cell info is handled in generate_html_from_cells

    all_y = [line['y0'] for line in text_lines] + [line['y1'] for line in text_lines]
    min_y = min(all_y)
    max_y = max(all_y)

    new_bbox = (
        table.bbox[0],
        min(table.bbox[1], min_y),
        table.bbox[2],
        max(table.bbox[3], max_y)
    )

    return TableInfo(
        page_num=table.page_num,
        table_idx=table.table_idx,
        bbox=new_bbox,
        data=new_data,
        col_count=table.col_count,
        row_count=len(new_data),
        page_height=table.page_height,
        cells_info=new_cells_info if new_cells_info else None,
        annotations=table.annotations,
        detection_strategy=table.detection_strategy,
        confidence=table.confidence
    )


# ============================================================================
# Table Merging
# ============================================================================

def merge_adjacent_tables(tables: List[TableInfo]) -> List[TableInfo]:
    """
    Merge adjacent tables.

    Args:
        tables: List of TableInfo

    Returns:
        Merged list of TableInfo
    """
    if not tables:
        return tables

    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)
    for table in tables:
        tables_by_page[table.page_num].append(table)

    merged_result = []

    for page_num, page_tables in tables_by_page.items():
        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])

        i = 0
        while i < len(sorted_tables):
            current = sorted_tables[i]

            merged = current
            while i + 1 < len(sorted_tables):
                next_table = sorted_tables[i + 1]

                if should_merge_tables(merged, next_table):
                    merged = do_merge_tables(merged, next_table)
                    i += 1
                    logger.debug(f"[PDF] Merged adjacent tables on page {page_num + 1}")
                else:
                    break

            merged_result.append(merged)
            i += 1

    merged_result.sort(key=lambda t: (t.page_num, t.bbox[1]))
    return merged_result


def should_merge_tables(t1: TableInfo, t2: TableInfo) -> bool:
    """
    Determine whether two tables should be merged.

    Args:
        t1: First table
        t2: Second table

    Returns:
        True if should merge, False otherwise
    """
    if t1.page_num != t2.page_num:
        return False

    y_gap = t2.bbox[1] - t1.bbox[3]
    if y_gap < 0 or y_gap > 30:
        return False

    x_overlap_start = max(t1.bbox[0], t2.bbox[0])
    x_overlap_end = min(t1.bbox[2], t2.bbox[2])
    x_overlap = max(0, x_overlap_end - x_overlap_start)

    t1_width = t1.bbox[2] - t1.bbox[0]
    t2_width = t2.bbox[2] - t2.bbox[0]

    overlap_ratio = x_overlap / max(t1_width, t2_width, 1)
    if overlap_ratio < 0.8:
        return False

    if t1.col_count == t2.col_count:
        return True
    if t1.row_count == 1 and t1.col_count < t2.col_count:
        return True

    return False


def do_merge_tables(t1: TableInfo, t2: TableInfo) -> TableInfo:
    """
    Perform table merging.

    Improvements:
    - Maintain basic cell info even without cells_info
    - Accurately adjust cell indices after merging

    Args:
        t1: First table
        t2: Second table

    Returns:
        Merged TableInfo
    """
    merged_bbox = (
        min(t1.bbox[0], t2.bbox[0]),
        t1.bbox[1],
        max(t1.bbox[2], t2.bbox[2]),
        t2.bbox[3]
    )

    merged_col_count = max(t1.col_count, t2.col_count)

    merged_data = []
    merged_cells = []

    # Process t1 data
    t1_row_count = len(t1.data)

    if t1.col_count < merged_col_count and t1.row_count == 1 and t1.data:
        # Handle colspan when header row has fewer columns
        extra_cols = merged_col_count - t1.col_count
        header_row = list(t1.data[0])

        new_header = []
        col_position = 0

        for orig_col_idx, value in enumerate(header_row):
            new_header.append(value)

            if orig_col_idx == 1 and extra_cols > 0:
                colspan = 1 + extra_cols
                merged_cells.append({
                    'row': 0,
                    'col': col_position,
                    'rowspan': 1,
                    'colspan': colspan,
                    'bbox': None
                })
                for _ in range(extra_cols):
                    new_header.append('')
                col_position += colspan
            else:
                merged_cells.append({
                    'row': 0,
                    'col': col_position,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })
                col_position += 1

        merged_data.append(new_header)
    else:
        # Process regular rows
        for row_idx, row in enumerate(t1.data):
            if len(row) < merged_col_count:
                adjusted_row = list(row) + [''] * (merged_col_count - len(row))
            else:
                adjusted_row = list(row)
            merged_data.append(adjusted_row)

        # Copy t1 cell info
        if t1.cells_info:
            merged_cells.extend(t1.cells_info)

    # Process t2 data
    row_offset = t1_row_count

    for row in t2.data:
        if len(row) < merged_col_count:
            adjusted_row = list(row) + [''] * (merged_col_count - len(row))
        else:
            adjusted_row = list(row)
        merged_data.append(adjusted_row)

    # Copy t2 cell info (with row offset applied)
    if t2.cells_info:
        for cell in t2.cells_info:
            adjusted_cell = dict(cell)
            adjusted_cell['row'] = cell.get('row', 0) + row_offset
            merged_cells.append(adjusted_cell)

    # If cell info is empty, set to None (handled by CellAnalysisEngine)
    final_cells_info = merged_cells if merged_cells else None

    return TableInfo(
        page_num=t1.page_num,
        table_idx=t1.table_idx,
        bbox=merged_bbox,
        data=merged_data,
        col_count=merged_col_count,
        row_count=len(merged_data),
        page_height=t1.page_height,
        cells_info=final_cells_info,
        detection_strategy=t1.detection_strategy,
        confidence=max(t1.confidence, t2.confidence)
    )


# ============================================================================
# Table Continuity Processing
# ============================================================================

def process_table_continuity(all_tables: List[TableInfo]) -> List[TableInfo]:
    """
    Handle table continuity across pages.

    Args:
        all_tables: List of all TableInfo

    Returns:
        Processed list of TableInfo
    """
    if not all_tables:
        return all_tables

    result = []
    last_category = None

    for i, table_info in enumerate(all_tables):
        table_info = TableInfo(
            page_num=table_info.page_num,
            table_idx=table_info.table_idx,
            bbox=table_info.bbox,
            data=copy.deepcopy(table_info.data),
            col_count=table_info.col_count,
            row_count=table_info.row_count,
            page_height=table_info.page_height,
            cells_info=table_info.cells_info,
            annotations=table_info.annotations,
            detection_strategy=table_info.detection_strategy,
            confidence=table_info.confidence
        )

        curr_data = table_info.data

        if i == 0:
            last_category = extract_last_category(curr_data)
            result.append(table_info)
            continue

        prev_table = all_tables[i - 1]

        is_continuation = (
            table_info.page_num > prev_table.page_num and
            prev_table.bbox[3] > prev_table.page_height * 0.7 and
            table_info.bbox[1] < table_info.page_height * 0.3 and
            table_info.col_count == prev_table.col_count
        )

        if is_continuation and last_category:
            for row in curr_data:
                if len(row) >= 2:
                    first_col = row[0]
                    second_col = row[1] if len(row) > 1 else ""

                    if (not first_col or not str(first_col).strip()) and second_col and str(second_col).strip():
                        row[0] = last_category
                    elif first_col and str(first_col).strip():
                        last_category = first_col
        else:
            new_last = extract_last_category(curr_data)
            if new_last:
                last_category = new_last

        result.append(table_info)

    return result


def extract_last_category(table_data: List[List[Optional[str]]]) -> Optional[str]:
    """
    Extract last category from table.

    Args:
        table_data: Table data

    Returns:
        Last category string or None
    """
    if not table_data:
        return None

    last_category = None

    for row in table_data:
        if len(row) >= 1 and row[0] and str(row[0]).strip():
            last_category = str(row[0]).strip()

    return last_category


# ============================================================================
# HTML Conversion
# ============================================================================

def is_single_column_table(table_info: TableInfo) -> bool:
    """
    Determines if a table has n rows × 1 column format.

    Tables with n rows × 1 column are often not actual tables,
    so converting them to a text list is more appropriate.

    Args:
        table_info: Table information

    Returns:
        True if single-column table, False otherwise
    """
    data = table_info.data

    if not data:
        return False

    # Calculate max columns across all rows
    max_cols = max(len(row) for row in data) if data else 0

    # Single column if max_cols is 1
    return max_cols == 1


def convert_single_column_to_text(table_info: TableInfo) -> str:
    """
    Converts a single-column table to a text list.

    Data with n rows × 1 column format is semantically more
    appropriate to express as structured text rather than a table.

    Args:
        table_info: Table information

    Returns:
        String in text list format
    """
    data = table_info.data

    if not data:
        return ""

    lines = []
    for row in data:
        if row and len(row) > 0:
            cell_text = str(row[0]).strip() if row[0] else ""
            if cell_text:
                lines.append(cell_text)

    return '\n'.join(lines)


def convert_table_to_html(table_info: TableInfo) -> str:
    """
    Converts a table to HTML.

    Improvements:
    1. Prioritize using PyMuPDF cell info
    2. Apply CellAnalysisEngine
    3. Accurate rowspan/colspan handling
    4. Full colspan for annotation rows
    5. Semantic HTML with accessibility considerations

    Args:
        table_info: Table information

    Returns:
        HTML string
    """
    data = table_info.data

    if not data:
        return ""

    num_rows = len(data)
    num_cols = max(len(row) for row in data) if data else 0

    if num_cols == 0:
        return ""

    # Perform cell analysis using CellAnalysisEngine
    cell_engine = CellAnalysisEngine(table_info, None)
    analyzed_cells = cell_engine.analyze()

    # Generate HTML from analyzed cell info
    return generate_html_from_cells(data, analyzed_cells, num_rows, num_cols)


def generate_html_from_cells(
    data: List[List[Optional[str]]],
    cells_info: List[Dict],
    num_rows: int,
    num_cols: int
) -> str:
    """
    Improved HTML generation.

    Improvements:
    - Process all cells even with incomplete cell info
    - Render empty cells correctly
    - Enhanced data range validation

    Args:
        data: Table data
        cells_info: Cell information list
        num_rows: Number of rows
        num_cols: Number of columns

    Returns:
        HTML string
    """
    # Create span_map: (row, col) -> {rowspan, colspan}
    span_map: Dict[Tuple[int, int], Dict] = {}

    for cell in cells_info:
        row = cell.get('row', 0)
        col = cell.get('col', 0)
        rowspan = max(1, cell.get('rowspan', 1))
        colspan = max(1, cell.get('colspan', 1))

        # Adjust to stay within data range
        if row >= num_rows or col >= num_cols:
            continue

        rowspan = min(rowspan, num_rows - row)
        colspan = min(colspan, num_cols - col)

        key = (row, col)
        span_map[key] = {
            'rowspan': rowspan,
            'colspan': colspan
        }

    # Create skip_set: positions covered by merged cells
    skip_set: Set[Tuple[int, int]] = set()

    for (row, col), spans in span_map.items():
        rowspan = spans['rowspan']
        colspan = spans['colspan']

        for r in range(row, min(row + rowspan, num_rows)):
            for c in range(col, min(col + colspan, num_cols)):
                if (r, c) != (row, col):
                    skip_set.add((r, c))

    # Detect annotation rows and apply full colspan
    for row_idx, row in enumerate(data):
        if not row:
            continue
        first_val = str(row[0]).strip() if row[0] else ""

        is_annotation = False
        for pattern in PDFConfig.ANNOTATION_PATTERNS:
            if first_val.startswith(pattern):
                is_annotation = True
                break

        if is_annotation:
            # Annotation row gets full colspan
            span_map[(row_idx, 0)] = {'rowspan': 1, 'colspan': num_cols}
            for col_idx in range(1, num_cols):
                skip_set.add((row_idx, col_idx))

    # Generate HTML
    html_parts = ["<table>"]

    for row_idx in range(num_rows):
        html_parts.append("  <tr>")

        row_data = data[row_idx] if row_idx < len(data) else []

        for col_idx in range(num_cols):
            # Check if this cell should be skipped
            if (row_idx, col_idx) in skip_set:
                continue

            # Extract cell content
            content = ""
            if col_idx < len(row_data):
                content = row_data[col_idx]
            content = escape_html(str(content).strip() if content else "")

            # Get span info (default to 1 if not found)
            spans = span_map.get((row_idx, col_idx), {'rowspan': 1, 'colspan': 1})
            attrs = []

            if spans['rowspan'] > 1:
                attrs.append(f'rowspan="{spans["rowspan"]}"')
            if spans['colspan'] > 1:
                attrs.append(f'colspan="{spans["colspan"]}"')

            attr_str = " " + " ".join(attrs) if attrs else ""

            # First row is treated as header
            tag = "th" if row_idx == 0 else "td"
            html_parts.append(f"    <{tag}{attr_str}>{content}</{tag}>")

        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)
