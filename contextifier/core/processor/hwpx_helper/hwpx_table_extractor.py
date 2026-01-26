# contextifier/core/processor/hwpx_helper/hwpx_table_extractor.py
"""
HWPX Table Extractor - HWPX Format-Specific Table Extraction

Implements table extraction for HWPX files (ZIP/XML based Korean document format).
Follows BaseTableExtractor interface from table_extractor.py.

HWPX Table XML Structure:
- <hp:tbl rowCnt="N" colCnt="M">: 테이블 요소
- <hp:tr>: 테이블 행
- <hp:tc>: 테이블 셀
- <hp:cellAddr colAddr="X" rowAddr="Y"/>: 셀 위치 (그리드 좌표)
- <hp:cellSpan colSpan="N" rowSpan="M"/>: 병합 정보
- <hp:subList>: 셀 내용 컨테이너
- <hp:p><hp:run><hp:t>: 텍스트 내용

2-Pass Approach:
1. Pass 1: Detect table regions (hp:tbl elements in section XML)
2. Pass 2: Extract content from detected regions (TableData objects)

Usage:
    from contextifier.core.processor.hwpx_helper.hwpx_table_extractor import (
        HWPXTableExtractor,
    )

    extractor = HWPXTableExtractor()
    
    # From XML element
    table_data = extractor.extract_table_from_element(table_elem, namespaces)
    
    # Or using 2-pass approach
    regions = extractor.detect_table_regions(table_elements)
    for region in regions:
        table = extractor.extract_table_from_region(table_elements, region)
"""
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)
from contextifier.core.processor.hwpx_helper.hwpx_constants import HWPX_NAMESPACES

logger = logging.getLogger("document-processor")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HWPXTableExtractorConfig(TableExtractorConfig):
    """Configuration specific to HWPX table extraction.
    
    Attributes:
        skip_single_cell_tables: Whether to skip 1x1 tables (often layout containers)
        skip_single_column_tables: Whether to skip single column tables (often text blocks)
        extract_nested_tables: Whether to recursively extract nested tables
        flatten_container_tables: Whether to flatten 1x1/1-col tables to text
    """
    skip_single_cell_tables: bool = True
    skip_single_column_tables: bool = True
    extract_nested_tables: bool = True
    flatten_container_tables: bool = True


# ============================================================================
# Data Classes for Table Region Tracking
# ============================================================================

@dataclass
class HWPXTableRegionInfo:
    """Additional information for HWPX table region.
    
    Stores reference to the table element and parsed metadata
    for use in Pass 2 extraction.
    
    Attributes:
        table_element: Reference to the hp:tbl XML element
        row_count: Number of rows (from rowCnt attribute)
        col_count: Number of columns (from colCnt attribute)
        namespaces: XML namespaces dictionary
        has_nested_tables: Whether the table contains nested tables
    """
    table_element: Optional[ET.Element] = None
    row_count: int = 0
    col_count: int = 0
    namespaces: Dict[str, str] = field(default_factory=dict)
    has_nested_tables: bool = False


@dataclass
class HWPXCellInfo:
    """Parsed cell information from HWPX table.
    
    Attributes:
        row_addr: Row position in grid (0-based)
        col_addr: Column position in grid (0-based)
        row_span: Number of rows this cell spans
        col_span: Number of columns this cell spans
        content: Cell text content
        nested_tables: List of nested TableData objects
    """
    row_addr: int = 0
    col_addr: int = 0
    row_span: int = 1
    col_span: int = 1
    content: str = ""
    nested_tables: List[TableData] = field(default_factory=list)


# ============================================================================
# HWPX Table Extractor Class
# ============================================================================

class HWPXTableExtractor(BaseTableExtractor):
    """HWPX format-specific table extractor.
    
    Extracts tables from HWPX files by parsing XML structure.
    Implements BaseTableExtractor interface from table_extractor.py.
    
    HWPX files are ZIP archives containing XML files:
    - Contents/section0.xml, section1.xml, ... (document sections)
    - Each section contains hp:tbl elements for tables
    
    Supports:
    - Cell merges (rowspan/colspan via hp:cellSpan)
    - Grid-based cell positioning (hp:cellAddr)
    - Nested tables (recursive extraction)
    - 1x1 and single-column table flattening
    """
    
    def __init__(self, config: Optional[HWPXTableExtractorConfig] = None):
        """Initialize HWPX table extractor.
        
        Args:
            config: HWPX table extraction configuration
        """
        self._config = config or HWPXTableExtractorConfig()
        super().__init__(self._config)
        # Cache for region info (index -> HWPXTableRegionInfo)
        self._region_info_cache: Dict[int, HWPXTableRegionInfo] = {}
        # Default namespaces
        self._default_ns = HWPX_NAMESPACES.copy()
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() in ("hwpx",)
    
    # ========================================================================
    # BaseTableExtractor Interface Implementation
    # ========================================================================
    
    def detect_table_regions(
        self, 
        content: Any,
        namespaces: Optional[Dict[str, str]] = None
    ) -> List[TableRegion]:
        """Detect table regions in HWPX content.
        
        Pass 1: Find all hp:tbl elements and create TableRegion objects.
        
        Args:
            content: List of hp:tbl XML elements, or single element
            namespaces: XML namespaces (optional, uses defaults if not provided)
            
        Returns:
            List of TableRegion objects (start_offset = table index)
        """
        self._region_info_cache.clear()
        
        ns = namespaces or self._default_ns
        
        # Handle different content types
        table_elements = self._normalize_content(content)
        
        if not table_elements:
            return []
        
        regions = []
        
        for idx, table_elem in enumerate(table_elements):
            try:
                # Get table dimensions from attributes
                row_count = self._get_int_attr(table_elem, 'rowCnt', 0)
                col_count = self._get_int_attr(table_elem, 'colCnt', 0)
                
                # If attributes not present, count from actual elements
                if row_count == 0:
                    rows = table_elem.findall('hp:tr', ns)
                    row_count = len(rows)
                
                if col_count == 0:
                    # Count max cells in any row
                    for tr in table_elem.findall('hp:tr', ns):
                        cells = tr.findall('hp:tc', ns)
                        col_count = max(col_count, len(cells))
                
                # Check for nested tables
                has_nested = self._check_nested_tables(table_elem, ns)
                
                # Cache region info for Pass 2
                region_info = HWPXTableRegionInfo(
                    table_element=table_elem,
                    row_count=row_count,
                    col_count=col_count,
                    namespaces=ns,
                    has_nested_tables=has_nested,
                )
                self._region_info_cache[idx] = region_info
                
                # Calculate confidence
                confidence = self._calculate_confidence(row_count, col_count)
                
                # Create TableRegion
                region = TableRegion(
                    start_offset=idx,
                    end_offset=idx + 1,
                    row_count=row_count,
                    col_count=col_count,
                    confidence=confidence,
                )
                regions.append(region)
                
                self.logger.debug(
                    f"HWPX table region detected: idx={idx}, "
                    f"{row_count}x{col_count}, nested={has_nested}"
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to detect table region {idx}: {e}")
                continue
        
        return regions
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from HWPX table region.
        
        Pass 2: Parse the hp:tbl element to create TableData.
        
        Args:
            content: Original content (may be ignored if cached)
            region: TableRegion from detect_table_regions()
            
        Returns:
            TableData object or None if extraction fails
        """
        region_idx = region.start_offset
        region_info = self._region_info_cache.get(region_idx)
        
        if region_info is None:
            self.logger.warning(f"No cached region info for index {region_idx}")
            return None
        
        table_elem = region_info.table_element
        ns = region_info.namespaces
        
        if table_elem is None:
            return None
        
        try:
            return self._extract_table_data(table_elem, ns)
        except Exception as e:
            self.logger.error(f"Failed to extract table from region {region_idx}: {e}")
            return None
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def extract_table_from_element(
        self, 
        table_elem: ET.Element,
        namespaces: Optional[Dict[str, str]] = None
    ) -> Optional[TableData]:
        """Extract table directly from XML element.
        
        Convenience method that combines detect + extract in one call.
        
        Args:
            table_elem: hp:tbl XML element
            namespaces: XML namespaces
            
        Returns:
            TableData object or None
        """
        ns = namespaces or self._default_ns
        return self._extract_table_data(table_elem, ns)
    
    def extract_table_as_html(
        self, 
        table_elem: ET.Element,
        namespaces: Optional[Dict[str, str]] = None
    ) -> str:
        """Extract table and format as HTML string.
        
        Convenience method for backward compatibility with parse_hwpx_table().
        
        Args:
            table_elem: hp:tbl XML element
            namespaces: XML namespaces
            
        Returns:
            HTML table string or plain text for container tables
        """
        ns = namespaces or self._default_ns
        table_data = self._extract_table_data(table_elem, ns)
        
        if table_data is None:
            return ""
        
        # Check for container/single-column tables
        if self._config.flatten_container_tables:
            # 1x1 table -> return cell content only
            if table_data.num_rows == 1 and table_data.num_cols == 1:
                if table_data.rows and table_data.rows[0]:
                    return table_data.rows[0][0].content
                return ""
            
            # Single column table -> join with newlines
            if table_data.num_cols == 1:
                text_parts = []
                for row in table_data.rows:
                    if row and row[0].content:
                        text_parts.append(row[0].content)
                return "\n\n".join(text_parts)
        
        # Multi-column table -> format as HTML
        return self._format_table_as_html(table_data)
    
    # ========================================================================
    # Private Helper Methods - Table Extraction
    # ========================================================================
    
    def _extract_table_data(
        self, 
        table_elem: ET.Element,
        ns: Dict[str, str]
    ) -> Optional[TableData]:
        """Extract TableData from hp:tbl element.
        
        This is the core extraction logic.
        
        Args:
            table_elem: hp:tbl XML element
            ns: XML namespaces
            
        Returns:
            TableData object
        """
        try:
            # Get table dimensions
            row_count = self._get_int_attr(table_elem, 'rowCnt', 0)
            col_count = self._get_int_attr(table_elem, 'colCnt', 0)
            
            # Build cell grid
            grid: Dict[Tuple[int, int], HWPXCellInfo] = {}
            max_row = 0
            max_col = 0
            
            for tr in table_elem.findall('hp:tr', ns):
                for tc in tr.findall('hp:tc', ns):
                    cell_info = self._parse_cell(tc, ns)
                    grid[(cell_info.row_addr, cell_info.col_addr)] = cell_info
                    max_row = max(max_row, cell_info.row_addr)
                    max_col = max(max_col, cell_info.col_addr)
            
            # Use actual dimensions if attributes not present
            if row_count == 0:
                row_count = max_row + 1
            if col_count == 0:
                col_count = max_col + 1
            
            if not grid:
                return None
            
            # Build TableData from grid
            return self._build_table_data(grid, row_count, col_count)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract table data: {e}")
            return None
    
    def _parse_cell(
        self, 
        tc: ET.Element, 
        ns: Dict[str, str]
    ) -> HWPXCellInfo:
        """Parse a single hp:tc element.
        
        Extracts position, span info, and content from cell.
        
        Args:
            tc: hp:tc XML element
            ns: XML namespaces
            
        Returns:
            HWPXCellInfo object
        """
        # Get cell position from hp:cellAddr
        row_addr = 0
        col_addr = 0
        cell_addr = tc.find('hp:cellAddr', ns)
        if cell_addr is not None:
            col_addr = self._get_int_attr(cell_addr, 'colAddr', 0)
            row_addr = self._get_int_attr(cell_addr, 'rowAddr', 0)
        
        # Get span info from hp:cellSpan
        col_span = 1
        row_span = 1
        cell_span = tc.find('hp:cellSpan', ns)
        if cell_span is not None:
            col_span = self._get_int_attr(cell_span, 'colSpan', 1)
            row_span = self._get_int_attr(cell_span, 'rowSpan', 1)
        
        # Extract cell content
        content, nested_tables = self._extract_cell_content(tc, ns)
        
        return HWPXCellInfo(
            row_addr=row_addr,
            col_addr=col_addr,
            row_span=row_span,
            col_span=col_span,
            content=content,
            nested_tables=nested_tables,
        )
    
    def _extract_cell_content(
        self, 
        tc: ET.Element, 
        ns: Dict[str, str]
    ) -> Tuple[str, List[TableData]]:
        """Extract text content and nested tables from cell.
        
        HWPX cell structure:
        <hp:tc>
          <hp:subList>
            <hp:p>
              <hp:run>
                <hp:t>텍스트</hp:t>
                <hp:tbl>...</hp:tbl>  <!-- nested table -->
              </hp:run>
            </hp:p>
          </hp:subList>
        </hp:tc>
        
        Args:
            tc: hp:tc XML element
            ns: XML namespaces
            
        Returns:
            Tuple of (text_content, nested_tables_list)
        """
        content_parts = []
        nested_tables = []
        
        sublist = tc.find('hp:subList', ns)
        if sublist is None:
            return "", []
        
        for p in sublist.findall('hp:p', ns):
            para_parts = []
            
            for run in p.findall('hp:run', ns):
                # Extract text
                t = run.find('hp:t', ns)
                if t is not None and t.text:
                    para_parts.append(t.text)
                
                # Handle nested tables
                if self._config.extract_nested_tables:
                    nested_tbl = run.find('hp:tbl', ns)
                    if nested_tbl is not None:
                        nested_data = self._extract_table_data(nested_tbl, ns)
                        if nested_data is not None:
                            nested_tables.append(nested_data)
                            # Format nested table as HTML for content
                            nested_html = self._format_table_as_html(nested_data)
                            if nested_html:
                                para_parts.append(nested_html)
            
            if para_parts:
                content_parts.append("".join(para_parts))
        
        return " ".join(content_parts).strip(), nested_tables
    
    def _build_table_data(
        self, 
        grid: Dict[Tuple[int, int], HWPXCellInfo],
        row_count: int,
        col_count: int
    ) -> TableData:
        """Build TableData from parsed cell grid.
        
        Args:
            grid: Dictionary mapping (row, col) to HWPXCellInfo
            row_count: Total number of rows
            col_count: Total number of columns
            
        Returns:
            TableData object
        """
        # Build skip map for merged cells
        skip_map = set()
        
        for (row, col), cell_info in grid.items():
            for rs in range(cell_info.row_span):
                for cs in range(cell_info.col_span):
                    if rs == 0 and cs == 0:
                        continue
                    skip_map.add((row + rs, col + cs))
        
        # Build rows
        rows: List[List[TableCell]] = []
        
        for r in range(row_count):
            row_cells: List[TableCell] = []
            
            for c in range(col_count):
                if (r, c) in skip_map:
                    # Merged cell placeholder (span = 0)
                    row_cells.append(TableCell(
                        content="",
                        row_span=0,
                        col_span=0,
                        row_index=r,
                        col_index=c,
                    ))
                elif (r, c) in grid:
                    cell_info = grid[(r, c)]
                    nested = cell_info.nested_tables[0] if cell_info.nested_tables else None
                    row_cells.append(TableCell(
                        content=cell_info.content,
                        row_span=cell_info.row_span,
                        col_span=cell_info.col_span,
                        row_index=r,
                        col_index=c,
                        nested_table=nested,
                    ))
                else:
                    # Empty cell
                    row_cells.append(TableCell(
                        content="",
                        row_span=1,
                        col_span=1,
                        row_index=r,
                        col_index=c,
                    ))
            
            rows.append(row_cells)
        
        return TableData(
            rows=rows,
            num_rows=row_count,
            num_cols=col_count,
            has_header=False,  # HWPX doesn't explicitly mark headers
            source_format="hwpx",
        )
    
    # ========================================================================
    # Private Helper Methods - HTML Formatting
    # ========================================================================
    
    def _format_table_as_html(self, table_data: TableData) -> str:
        """Format TableData as HTML string.
        
        Args:
            table_data: TableData to format
            
        Returns:
            HTML table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        html_parts = ["<table border='1'>"]
        
        for row in table_data.rows:
            html_parts.append("<tr>")
            
            for cell in row:
                # Skip merged-away cells
                if cell.row_span == 0 or cell.col_span == 0:
                    continue
                
                attrs = []
                if cell.row_span > 1:
                    attrs.append(f"rowspan='{cell.row_span}'")
                if cell.col_span > 1:
                    attrs.append(f"colspan='{cell.col_span}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                content = self._escape_html(cell.content)
                
                html_parts.append(f"<td{attr_str}>{content}</td>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    # ========================================================================
    # Private Helper Methods - Utility
    # ========================================================================
    
    def _normalize_content(self, content: Any) -> List[ET.Element]:
        """Normalize content to list of table elements."""
        if content is None:
            return []
        
        if isinstance(content, ET.Element):
            return [content]
        
        if isinstance(content, list):
            return content
        
        return []
    
    def _get_int_attr(self, elem: ET.Element, attr: str, default: int = 0) -> int:
        """Get integer attribute from element."""
        try:
            return int(elem.get(attr, default))
        except (ValueError, TypeError):
            return default
    
    def _check_nested_tables(self, table_elem: ET.Element, ns: Dict[str, str]) -> bool:
        """Check if table contains nested tables."""
        # Look for hp:tbl inside hp:tc elements
        for tc in table_elem.findall('.//hp:tc', ns):
            for tbl in tc.findall('.//hp:tbl', ns):
                return True
        return False
    
    def _calculate_confidence(self, row_count: int, col_count: int) -> float:
        """Calculate confidence score for table detection."""
        if row_count == 0 or col_count == 0:
            return 0.0
        
        # Higher confidence for larger tables
        size_score = min(1.0, (row_count * col_count) / 20.0)
        
        # Bonus for multi-column tables
        if col_count >= 2:
            size_score = min(1.0, size_score + 0.3)
        
        return round(size_score, 2)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace("\n", "<br>")
        return text


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def parse_hwpx_table(table_node: ET.Element, ns: Dict[str, str]) -> str:
    """Parse HWPX table to HTML string.
    
    Backward compatible function that wraps HWPXTableExtractor.
    
    Args:
        table_node: hp:tbl XML element
        ns: XML namespaces dictionary
        
    Returns:
        HTML table string or plain text for container tables
    """
    extractor = HWPXTableExtractor()
    return extractor.extract_table_as_html(table_node, ns)


def extract_cell_content(tc: ET.Element, ns: Dict[str, str]) -> str:
    """Extract cell content from hp:tc element.
    
    Backward compatible function.
    
    Args:
        tc: hp:tc XML element
        ns: XML namespaces dictionary
        
    Returns:
        Cell text content
    """
    extractor = HWPXTableExtractor()
    content, _ = extractor._extract_cell_content(tc, ns)
    return content


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    "HWPXTableExtractorConfig",
    # Region Info
    "HWPXTableRegionInfo",
    "HWPXCellInfo",
    # Extractor
    "HWPXTableExtractor",
    # Backward compatibility
    "parse_hwpx_table",
    "extract_cell_content",
]
