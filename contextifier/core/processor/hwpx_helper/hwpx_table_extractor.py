# contextifier/core/processor/hwpx_helper/hwpx_table_extractor.py
"""
HWPX Table Extractor

Extracts tables from HWPX documents using the BaseTableExtractor interface.
Converts HWPX table elements to TableData objects for further processing.

================================================================================
EXTRACTION APPROACH: Streaming Processing (요소 단위 실시간 처리) - APPROACH 2
================================================================================

HWPX uses the Streaming Processing approach because:
- Tables are explicit <hp:tbl> XML elements
- Tables can be processed one-by-one during document traversal  
- Preserves natural document order
- Memory efficient for large documents

External Interface: extract_table(element, context) -> Optional[TableData]
- Called from hwpx_section.py during section element traversal
- Each <hp:tbl> element is passed to extract_table()
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
    _parse_cell_position()      - Extract cellAddr (row/col position)
    _parse_cell_span()          - Extract cellSpan (rowspan/colspan)
    _extract_cell_content()     - Cell content extraction (including nested tables)
    _build_cell_grid()          - Build grid from cells

================================================================================
Key Features:
- Full support for rowspan/colspan (hp:cellSpan)
- Grid-based cell positioning (hp:cellAddr)
- Nested table support (recursive processing)
- Container table detection (1x1 tables)

HWPX Table XML Structure:
- hp:tbl rowCnt="N" colCnt="M": Table element with row/col count attributes
- hp:tr: Table row
- hp:tc: Table cell
- hp:cellAddr colAddr="X" rowAddr="Y": Cell position in grid
- hp:cellSpan colSpan="N" rowSpan="M": Merge information
- hp:subList/hp:p/hp:run/hp:t: Cell text content
"""
import logging
import traceback
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableExtractorConfig,
)
from contextifier.core.processor.hwpx_helper.hwpx_constants import HWPX_NAMESPACES

logger = logging.getLogger("document-processor")


class HWPXTableExtractor(BaseTableExtractor):
    """
    HWPX-specific table extractor implementation.
    
    Uses STREAMING PROCESSING approach (APPROACH 2 - 요소 단위 실시간 처리).
    
    Extracts tables from HWPX documents and converts them to TableData objects.
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
        extractor = HWPXTableExtractor()
        
        # Streaming approach (APPROACH 2):
        for p in root.findall('hp:p', ns):
            for run in p.findall('hp:run', ns):
                table = run.find('hp:tbl', ns)
                if table is not None:
                    table_data = extractor.extract_table(table, ns)
                    if table_data:
                        process(table_data)
    """
    
    def __init__(self, config: Optional[TableExtractorConfig] = None):
        """Initialize the HWPX table extractor.
        
        Args:
            config: Table extraction configuration
        """
        super().__init__(config)
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format.
        
        Args:
            format_type: Format identifier
            
        Returns:
            True if format is 'hwpx'
        """
        return format_type.lower() == 'hwpx'
    
    # ==========================================================================
    # STREAMING PROCESSING - APPROACH 2 (요소 단위 실시간 처리)
    # ==========================================================================
    # 
    # HWPX는 APPROACH 2를 사용하므로 extract_table() 하나만 외부에 노출됨.
    # 모든 세부 함수는 extract_table() 내부에서만 호출됨.
    #
    # ==========================================================================
    
    def extract_table(
        self, 
        element: Any, 
        context: Any = None
    ) -> Optional[TableData]:
        """Extract a single table from a <hp:tbl> XML element.
        
        ========================================================================
        [APPROACH 2 - STREAMING PROCESSING] - Single External Interface
        ========================================================================
        
        This is the ONLY public method for HWPX table extraction.
        Called from hwpx_section.py during section parsing.
        
        All internal processing (grid building, span detection, 
        cell extraction) is encapsulated within this method.
        
        Args:
            element: <hp:tbl> XML element (ElementTree Element)
            context: Namespace dictionary for XML parsing (default: HWPX_NAMESPACES)
            
        Returns:
            TableData object or None if extraction fails
            
        Example:
            extractor = HWPXTableExtractor()
            for table in root.iter('{http://www.hancom.co.kr/hwpml/2011/paragraph}tbl'):
                table_data = extractor.extract_table(table, HWPX_NAMESPACES)
                if table_data:
                    html = processor.format_table_as_html(table_data)
        """
        try:
            # Use provided namespace or default
            ns = context if isinstance(context, dict) else HWPX_NAMESPACES
            
            # ----------------------------------------------------------------
            # Step 1: Get table dimensions from attributes
            # ----------------------------------------------------------------
            total_rows = int(element.get('rowCnt', 0))
            total_cols = int(element.get('colCnt', 0))
            
            # ----------------------------------------------------------------
            # Step 2: Build cell grid from table structure
            # ----------------------------------------------------------------
            grid, max_row, max_col = self._build_cell_grid(element, ns)
            
            # Update dimensions if not specified in attributes
            if total_rows == 0:
                total_rows = max_row + 1 if max_row >= 0 else 0
            if total_cols == 0:
                total_cols = max_col + 1 if max_col >= 0 else 0
            
            if not grid:
                return None
            
            # ----------------------------------------------------------------
            # Step 3: Build skip map for merged cells
            # ----------------------------------------------------------------
            skip_map = set()
            for (row_addr, col_addr), cell_info in grid.items():
                rowspan = cell_info['rowspan']
                colspan = cell_info['colspan']
                # Mark cells covered by merge (except the origin cell)
                for rs in range(rowspan):
                    for cs in range(colspan):
                        if rs == 0 and cs == 0:
                            continue
                        skip_map.add((row_addr + rs, col_addr + cs))
            
            # ----------------------------------------------------------------
            # Step 4: Build TableCell grid
            # ----------------------------------------------------------------
            table_rows: List[List[TableCell]] = []
            
            for r in range(total_rows):
                row_cells: List[TableCell] = []
                
                for c in range(total_cols):
                    # Skip merged cells
                    if (r, c) in skip_map:
                        continue
                    
                    if (r, c) in grid:
                        cell_info = grid[(r, c)]
                        content = cell_info['text']
                        rowspan = cell_info['rowspan']
                        colspan = cell_info['colspan']
                        
                        table_cell = TableCell(
                            content=content,
                            row_span=rowspan,
                            col_span=colspan,
                            is_header=(r == 0 and self.config.include_header_row),
                            row_index=r,
                            col_index=c,
                            nested_table=None  # Nested tables are embedded in content
                        )
                        row_cells.append(table_cell)
                    else:
                        # Empty cell (not in grid, not skipped)
                        table_cell = TableCell(
                            content="",
                            row_span=1,
                            col_span=1,
                            is_header=(r == 0 and self.config.include_header_row),
                            row_index=r,
                            col_index=c,
                            nested_table=None
                        )
                        row_cells.append(table_cell)
                
                if row_cells:
                    table_rows.append(row_cells)
            
            # ----------------------------------------------------------------
            # Step 5: Create and return TableData
            # ----------------------------------------------------------------
            actual_rows = len(table_rows)
            actual_cols = total_cols
            
            table_data = TableData(
                rows=table_rows,
                num_rows=actual_rows,
                num_cols=actual_cols,
                has_header=self.config.include_header_row and actual_rows > 0,
                start_offset=0,
                end_offset=0,
                source_format='hwpx',
                metadata={
                    'original_row_cnt': total_rows,
                    'original_col_cnt': total_cols,
                },
                col_widths_percent=[]  # HWPX doesn't typically specify column widths
            )
            
            return table_data
            
        except Exception as e:
            self.logger.error(f"Error extracting table from HWPX element: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    # ==========================================================================
    # Private Helper Methods (Called internally from extract_table)
    # ==========================================================================
    
    def _build_cell_grid(
        self, 
        table_elem: ET.Element, 
        ns: Dict[str, str]
    ) -> Tuple[Dict[Tuple[int, int], Dict], int, int]:
        """Build a grid of cells from the table element.
        
        Parses all cells and builds a dictionary mapping (row, col) positions
        to cell information including text, rowspan, and colspan.
        
        Args:
            table_elem: <hp:tbl> XML element
            ns: Namespace dictionary
            
        Returns:
            Tuple of (grid, max_row, max_col) where:
            - grid: Dict mapping (row_addr, col_addr) -> {text, rowspan, colspan}
            - max_row: Maximum row index found
            - max_col: Maximum column index found
        """
        grid = {}
        max_row = -1
        max_col = -1
        
        for tr in table_elem.findall('hp:tr', ns):
            for tc in tr.findall('hp:tc', ns):
                # Parse cell position
                row_addr, col_addr = self._parse_cell_position(tc, ns)
                
                # Parse cell span
                rowspan, colspan = self._parse_cell_span(tc, ns)
                
                # Extract cell content (including nested tables)
                cell_text = self._extract_cell_content(tc, ns)
                
                # Store in grid
                grid[(row_addr, col_addr)] = {
                    'text': cell_text,
                    'rowspan': rowspan,
                    'colspan': colspan,
                }
                
                max_row = max(max_row, row_addr)
                max_col = max(max_col, col_addr)
        
        return grid, max_row, max_col
    
    def _parse_cell_position(
        self, 
        tc: ET.Element, 
        ns: Dict[str, str]
    ) -> Tuple[int, int]:
        """Parse cell position from hp:cellAddr element.
        
        Args:
            tc: <hp:tc> cell element
            ns: Namespace dictionary
            
        Returns:
            Tuple of (row_addr, col_addr)
        """
        row_addr = 0
        col_addr = 0
        
        cell_addr = tc.find('hp:cellAddr', ns)
        if cell_addr is not None:
            try:
                col_addr = int(cell_addr.get('colAddr', 0))
            except (ValueError, TypeError):
                col_addr = 0
            try:
                row_addr = int(cell_addr.get('rowAddr', 0))
            except (ValueError, TypeError):
                row_addr = 0
        
        return row_addr, col_addr
    
    def _parse_cell_span(
        self, 
        tc: ET.Element, 
        ns: Dict[str, str]
    ) -> Tuple[int, int]:
        """Parse cell span from hp:cellSpan element.
        
        Args:
            tc: <hp:tc> cell element
            ns: Namespace dictionary
            
        Returns:
            Tuple of (rowspan, colspan)
        """
        rowspan = 1
        colspan = 1
        
        cell_span = tc.find('hp:cellSpan', ns)
        if cell_span is not None:
            try:
                colspan = int(cell_span.get('colSpan', 1))
            except (ValueError, TypeError):
                colspan = 1
            try:
                rowspan = int(cell_span.get('rowSpan', 1))
            except (ValueError, TypeError):
                rowspan = 1
        
        return rowspan, colspan
    
    def _extract_cell_content(
        self, 
        tc: ET.Element, 
        ns: Dict[str, str]
    ) -> str:
        """Extract cell content including text and nested tables.
        
        Recursively processes nested tables and returns them as embedded content.
        
        Args:
            tc: <hp:tc> cell element
            ns: Namespace dictionary
            
        Returns:
            Cell content as string (nested tables converted to text)
        """
        content_parts = []
        
        sublist = tc.find('hp:subList', ns)
        if sublist is not None:
            for p in sublist.findall('hp:p', ns):
                para_parts = []
                
                for run in p.findall('hp:run', ns):
                    # Extract text content
                    t = run.find('hp:t', ns)
                    if t is not None and t.text:
                        para_parts.append(t.text)
                    
                    # Handle nested table (recursive call)
                    nested_table = run.find('hp:tbl', ns)
                    if nested_table is not None:
                        nested_data = self.extract_table(nested_table, ns)
                        if nested_data:
                            # Convert nested table to simple text representation
                            nested_text = self._nested_table_to_text(nested_data)
                            if nested_text:
                                para_parts.append(nested_text)
                
                if para_parts:
                    content_parts.append("".join(para_parts))
        
        return " ".join(content_parts).strip()
    
    def _nested_table_to_text(self, table_data: TableData) -> str:
        """Convert a nested TableData to simple text representation.
        
        For nested tables, we convert to a simple text format to avoid
        deeply nested HTML structures.
        
        Args:
            table_data: TableData of the nested table
            
        Returns:
            Simple text representation of the table
        """
        if not table_data or not table_data.rows:
            return ""
        
        lines = []
        for row in table_data.rows:
            row_texts = [cell.content for cell in row if cell.content]
            if row_texts:
                lines.append(" | ".join(row_texts))
        
        return "\n".join(lines) if lines else ""


def create_hwpx_table_extractor(
    config: Optional[TableExtractorConfig] = None
) -> HWPXTableExtractor:
    """Factory function to create an HWPX table extractor.
    
    Args:
        config: Table extraction configuration
        
    Returns:
        Configured HWPXTableExtractor instance
    """
    return HWPXTableExtractor(config)
