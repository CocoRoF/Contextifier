# contextifier/core/processor/docx_helper/docx_table_extractor.py
"""
DOCX Table Extractor

Extracts tables from DOCX documents using the BaseTableExtractor interface.
Converts DOCX table elements to TableData objects for further processing.

================================================================================
EXTRACTION APPROACH: Streaming Processing (요소 단위 실시간 처리) - APPROACH 2
================================================================================

DOCX uses the Streaming Processing approach because:
- Tables are explicit <w:tbl> XML elements
- Tables can be processed one-by-one during document traversal  
- Preserves natural document order
- Memory efficient for large documents

External Interface: extract_table(element, context) -> Optional[TableData]
- Called from docx_handler.py during body element traversal
- Each <w:tbl> element is passed to extract_table()
- Returns TableData or None
- ALL internal processing is encapsulated within this single method

================================================================================
APPROACH 2 Pure Implementation:
================================================================================
Per table_extractor.py structure, APPROACH 2 exposes ONLY extract_table().
All sub-functions are private and called only from within extract_table().

External (Public):
    extract_table(element, context) → Optional[TableData]

Internal (Private) - All called from extract_table():
    _estimate_column_count()    - Grid column count calculation
    _calculate_column_widths()  - Column width percentages
    _calculate_all_rowspans()   - vMerge rowspan calculation
    _extract_cell_text()        - Cell content extraction

================================================================================
Key Features:
- Full support for rowspan/colspan (vMerge/gridSpan)
- Column width calculation  
- Header row detection
- Nested table support (TODO)

OOXML Table Structure:
- w:tblGrid: Table grid column definitions
- w:tr: Table row
- w:tc: Table cell
- w:tcPr/w:gridSpan: colspan (horizontal merge)
- w:tcPr/w:vMerge val="restart": rowspan start
- w:tcPr/w:vMerge (no val): rowspan continue (merged cell)
"""
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.oxml.ns import qn

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableExtractorConfig,
)
from contextifier.core.processor.docx_helper.docx_constants import NAMESPACES

logger = logging.getLogger("document-processor")


class DOCXTableExtractor(BaseTableExtractor):
    """
    DOCX-specific table extractor implementation.
    
    Uses STREAMING PROCESSING approach (APPROACH 2 - 요소 단위 실시간 처리).
    
    Extracts tables from DOCX documents and converts them to TableData objects.
    Supports complex table structures including merged cells (rowspan/colspan).
    
    ============================================================================
    External Interface (Public):
    ============================================================================
        extract_table(element, context) -> Optional[TableData]
        
    This is the ONLY public method for table extraction.
    All other methods are private and called internally from extract_table().
    
    ============================================================================
    Usage:
    ============================================================================
        extractor = DOCXTableExtractor()
        
        # Streaming approach (APPROACH 2):
        for elem in doc.element.body:
            if elem.tag.endswith('tbl'):
                table_data = extractor.extract_table(elem, doc)
                if table_data:
                    process(table_data)
    """
    
    def __init__(self, config: Optional[TableExtractorConfig] = None):
        """Initialize the DOCX table extractor.
        
        Args:
            config: Table extraction configuration
        """
        super().__init__(config)
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format.
        
        Args:
            format_type: Format identifier
            
        Returns:
            True if format is 'docx'
        """
        return format_type.lower() == 'docx'
    
    # ==========================================================================
    # STREAMING PROCESSING - APPROACH 2 (요소 단위 실시간 처리)
    # ==========================================================================
    # 
    # DOCX는 APPROACH 2를 사용하므로 extract_table() 하나만 외부에 노출됨.
    # 모든 세부 함수는 extract_table() 내부에서만 호출됨.
    #
    # ==========================================================================
    
    def extract_table(
        self, 
        element: Any, 
        context: Any = None
    ) -> Optional[TableData]:
        """Extract a single table from a <w:tbl> XML element.
        
        ========================================================================
        [APPROACH 2 - STREAMING PROCESSING] - Single External Interface
        ========================================================================
        
        This is the ONLY public method for DOCX table extraction.
        Called from docx_handler.py during document body traversal.
        
        All internal processing (column calculation, rowspan detection, 
        cell extraction) is encapsulated within this method.
        
        Args:
            element: <w:tbl> XML element (lxml Element)
            context: Document object for additional context (optional)
            
        Returns:
            TableData object or None if extraction fails
            
        Example:
            for elem in doc.element.body:
                if etree.QName(elem).localname == 'tbl':
                    table_data = extractor.extract_table(elem, doc)
                    if table_data:
                        html = processor.format_table_as_html(table_data)
        """
        try:
            # ----------------------------------------------------------------
            # Step 1: Validate input and get row elements
            # ----------------------------------------------------------------
            rows_elem = element.findall('w:tr', NAMESPACES)
            if not rows_elem:
                return None
            
            num_rows = len(rows_elem)
            
            # ----------------------------------------------------------------
            # Step 2: Calculate column count and widths
            # ----------------------------------------------------------------
            num_cols = self._estimate_column_count(element, rows_elem)
            col_widths = self._calculate_column_widths(element, num_cols)
            
            # ----------------------------------------------------------------
            # Step 3: Calculate all rowspans and cell positions
            # ----------------------------------------------------------------
            rowspan_map, cell_grid_col = self._calculate_all_rowspans(
                element, rows_elem, num_rows
            )
            
            # ----------------------------------------------------------------
            # Step 4: Build TableCell grid
            # ----------------------------------------------------------------
            table_rows: List[List[TableCell]] = []
            
            for row_idx, row in enumerate(rows_elem):
                cells_elem = row.findall('w:tc', NAMESPACES)
                row_cells: List[TableCell] = []
                
                for cell_idx, cell in enumerate(cells_elem):
                    # Get cell properties
                    tcPr = cell.find('w:tcPr', NAMESPACES)
                    colspan = 1
                    is_vmerge_continue = False
                    
                    if tcPr is not None:
                        # Get colspan (gridSpan)
                        gs = tcPr.find('w:gridSpan', NAMESPACES)
                        if gs is not None:
                            try:
                                colspan = int(gs.get(qn('w:val'), 1))
                            except (ValueError, TypeError):
                                colspan = 1
                        
                        # Check vMerge status
                        vMerge = tcPr.find('w:vMerge', NAMESPACES)
                        if vMerge is not None:
                            val = vMerge.get(qn('w:val'))
                            if val != 'restart':
                                is_vmerge_continue = True
                    
                    # Skip cells that are merged (continue cells)
                    if is_vmerge_continue:
                        continue
                    
                    # Get grid column position
                    if cell_idx < len(cell_grid_col[row_idx]):
                        start_col, end_col = cell_grid_col[row_idx][cell_idx]
                    else:
                        start_col = cell_idx
                    
                    # Get rowspan from pre-calculated map
                    rowspan = rowspan_map.get((row_idx, start_col), 1)
                    
                    # Extract cell content
                    content = self._extract_cell_text(cell)
                    
                    # Create TableCell
                    table_cell = TableCell(
                        content=content,
                        row_span=rowspan,
                        col_span=colspan,
                        is_header=(row_idx == 0 and self.config.include_header_row),
                        row_index=row_idx,
                        col_index=start_col,
                        nested_table=None  # TODO: Handle nested tables if needed
                    )
                    row_cells.append(table_cell)
                
                if row_cells:
                    table_rows.append(row_cells)
            
            # ----------------------------------------------------------------
            # Step 5: Create and return TableData
            # ----------------------------------------------------------------
            actual_rows = len(table_rows)
            actual_cols = num_cols
            
            table_data = TableData(
                rows=table_rows,
                num_rows=actual_rows,
                num_cols=actual_cols,
                has_header=self.config.include_header_row and actual_rows > 0,
                start_offset=0,
                end_offset=0,
                source_format='docx',
                metadata={},
                col_widths_percent=col_widths
            )
            
            return table_data
            
        except Exception as e:
            self.logger.error(f"Error extracting table from element: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    # ==========================================================================
    # Private Helper Methods (Called internally from extract_table)
    # ==========================================================================
    
    def _estimate_column_count(
        self, 
        table_elem: Any, 
        rows: List[Any]
    ) -> int:
        """Estimate the number of columns in the table.
        
        Args:
            table_elem: Table XML element
            rows: List of row elements
            
        Returns:
            Number of columns
        """
        # Try to get from tblGrid first
        tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
        if tblGrid is not None:
            grid_cols = tblGrid.findall('w:gridCol', NAMESPACES)
            if grid_cols:
                return len(grid_cols)
        
        # Fallback: calculate from first row
        if not rows:
            return 0
        
        num_cols = 0
        for cell in rows[0].findall('w:tc', NAMESPACES):
            tcPr = cell.find('w:tcPr', NAMESPACES)
            colspan = 1
            if tcPr is not None:
                gs = tcPr.find('w:gridSpan', NAMESPACES)
                if gs is not None:
                    try:
                        colspan = int(gs.get(qn('w:val'), 1))
                    except (ValueError, TypeError):
                        colspan = 1
            num_cols += colspan
        
        return num_cols
    
    def _calculate_column_widths(
        self, 
        table_elem: Any, 
        num_cols: int
    ) -> List[float]:
        """Calculate column widths as percentages.
        
        Args:
            table_elem: Table XML element
            num_cols: Number of columns
            
        Returns:
            List of column widths as percentages
        """
        widths = []
        
        tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
        if tblGrid is not None:
            grid_cols = tblGrid.findall('w:gridCol', NAMESPACES)
            
            # Extract widths in twips
            raw_widths = []
            for col in grid_cols:
                w = col.get(qn('w:w'))
                if w:
                    try:
                        raw_widths.append(int(w))
                    except ValueError:
                        raw_widths.append(0)
                else:
                    raw_widths.append(0)
            
            # Convert to percentages
            total_width = sum(raw_widths)
            if total_width > 0:
                widths = [(w / total_width) * 100 for w in raw_widths]
        
        # Fallback: equal widths
        if not widths and num_cols > 0:
            widths = [100.0 / num_cols] * num_cols
        
        return widths
    
    def _calculate_all_rowspans(
        self, 
        table_elem: Any, 
        rows: List[Any], 
        num_rows: int
    ) -> Tuple[Dict[Tuple[int, int], int], List[List[Tuple[int, int]]]]:
        """Calculate rowspans for all cells with vMerge restart.
        
        Uses improved algorithm (v3) for accurate merge tracking:
        1. Collect all cell information
        2. Use merge_info matrix to track cell ownership
        3. Connect continue cells to restart cells above
        4. Calculate rowspan by counting owned cells below
        
        Args:
            table_elem: Table XML element
            rows: List of row elements
            num_rows: Number of rows
            
        Returns:
            Tuple of (rowspan_map, cell_grid_col)
            - rowspan_map: Dict[(row_idx, grid_col), rowspan]
            - cell_grid_col: List[List[(start_col, end_col)]]
        """
        rowspan_map: Dict[Tuple[int, int], int] = {}
        
        # Collect all cell info
        all_cells_info: List[List[Tuple[int, str]]] = []
        
        for row in rows:
            cells = row.findall('w:tc', NAMESPACES)
            row_info = []
            for cell in cells:
                tcPr = cell.find('w:tcPr', NAMESPACES)
                colspan = 1
                vmerge_status = 'none'
                
                if tcPr is not None:
                    gs = tcPr.find('w:gridSpan', NAMESPACES)
                    if gs is not None:
                        try:
                            colspan = int(gs.get(qn('w:val'), 1))
                        except (ValueError, TypeError):
                            colspan = 1
                    
                    vMerge = tcPr.find('w:vMerge', NAMESPACES)
                    if vMerge is not None:
                        val = vMerge.get(qn('w:val'))
                        vmerge_status = 'restart' if val == 'restart' else 'continue'
                
                row_info.append((colspan, vmerge_status))
            all_cells_info.append(row_info)
        
        # Step 1: Calculate grid column positions for all cells
        max_cols = 30
        cell_grid_col: List[List[Tuple[int, int]]] = []
        
        # merge_info[row][col] = (owner_row, owner_col, colspan)
        merge_info: List[List[Optional[Tuple[int, int, int]]]] = [
            [None] * max_cols for _ in range(num_rows)
        ]
        
        for row_idx, row_info in enumerate(all_cells_info):
            grid_col = 0
            row_grid_cols: List[Tuple[int, int]] = []
            
            for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
                # Skip already occupied columns (from vMerge above)
                while grid_col < max_cols and merge_info[row_idx][grid_col] is not None:
                    grid_col += 1
                
                # Expand if needed
                while grid_col + colspan > max_cols:
                    for r in range(num_rows):
                        merge_info[r].extend([None] * 10)
                    max_cols += 10
                
                start_col = grid_col
                end_col = grid_col + colspan - 1
                row_grid_cols.append((start_col, end_col))
                
                if vmerge_status == 'restart':
                    # Restart cell: mark current row only
                    for c in range(start_col, start_col + colspan):
                        merge_info[row_idx][c] = (row_idx, start_col, colspan)
                
                elif vmerge_status == 'continue':
                    # Continue cell: link to cell above
                    for prev_row in range(row_idx - 1, -1, -1):
                        if merge_info[prev_row][start_col] is not None:
                            owner = merge_info[prev_row][start_col]
                            for c in range(start_col, start_col + colspan):
                                merge_info[row_idx][c] = owner
                            break
                    else:
                        # Not found - set to current (edge case)
                        for c in range(start_col, start_col + colspan):
                            merge_info[row_idx][c] = (row_idx, start_col, colspan)
                else:
                    # Normal cell
                    for c in range(start_col, start_col + colspan):
                        merge_info[row_idx][c] = (row_idx, start_col, colspan)
                
                grid_col += colspan
            
            cell_grid_col.append(row_grid_cols)
        
        # Step 2: Calculate rowspans for restart cells
        for row_idx, row_info in enumerate(all_cells_info):
            for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
                if cell_idx >= len(cell_grid_col[row_idx]):
                    continue
                start_col, end_col = cell_grid_col[row_idx][cell_idx]
                
                if vmerge_status == 'restart':
                    # Count cells below with same owner
                    rowspan = 1
                    for next_row in range(row_idx + 1, num_rows):
                        if start_col < max_cols and merge_info[next_row][start_col] == (row_idx, start_col, colspan):
                            rowspan += 1
                        else:
                            break
                    rowspan_map[(row_idx, start_col)] = rowspan
                
                elif vmerge_status == 'none':
                    rowspan_map[(row_idx, start_col)] = 1
        
        return rowspan_map, cell_grid_col
    
    def _extract_cell_text(self, cell_elem: Any) -> str:
        """Extract text content from a cell element.
        
        Args:
            cell_elem: Cell XML element
            
        Returns:
            Cell text content
        """
        texts = []
        
        for p in cell_elem.findall('.//w:p', NAMESPACES):
            p_texts = []
            for t in p.findall('.//w:t', NAMESPACES):
                if t.text:
                    p_texts.append(t.text)
            if p_texts:
                texts.append(''.join(p_texts))
        
        return '\n'.join(texts)


# Factory function
def create_docx_table_extractor(
    config: Optional[TableExtractorConfig] = None
) -> DOCXTableExtractor:
    """Create a DOCX table extractor instance.
    
    Args:
        config: Table extraction configuration
        
    Returns:
        Configured DOCXTableExtractor instance
    """
    return DOCXTableExtractor(config)


__all__ = [
    'DOCXTableExtractor',
    'create_docx_table_extractor',
]
