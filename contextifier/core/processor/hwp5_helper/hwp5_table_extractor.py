# contextifier/core/processor/hwp5_helper/hwp5_table_extractor.py
"""
HWP 5.0 Table Extractor

Implements table extraction for HWP 5.0 OLE format files.
Follows BaseTableExtractor interface from table_extractor.py.

HWP 5.0 Table Structure:
- CTRL_HEADER (ctrl_id='tbl '): Table control header
- HWPTAG_TABLE: Table properties (rowCount, colCount)
- HWPTAG_LIST_HEADER: Cell information (position, spans)
- Cell content in child records or subsequent siblings

2-Pass Approach:
1. Pass 1: Detect table regions (CTRL_HEADER with 'tbl ' ID)
2. Pass 2: Extract content from detected regions (TableData objects)

Usage:
    from contextifier.core.processor.hwp5_helper.hwp5_table_extractor import (
        HWP5TableExtractor,
    )

    extractor = HWP5TableExtractor()
    
    # From record tree
    regions = extractor.detect_table_regions(root_record)
    for region in regions:
        table = extractor.extract_table_from_region(root_record, region)
"""
import struct
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)
from contextifier.core.processor.hwp5_helper.hwp5_constants import (
    HWPTAG_TABLE,
    HWPTAG_LIST_HEADER,
    HWPTAG_CTRL_HEADER,
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    CTRL_ID_TABLE,
)
from contextifier.core.processor.hwp5_helper.hwp5_record import HwpRecord

if TYPE_CHECKING:
    import olefile

logger = logging.getLogger("document-processor.HWP5")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HWP5TableExtractorConfig(TableExtractorConfig):
    """Configuration specific to HWP 5.0 table extraction.
    
    Attributes:
        skip_single_cell_tables: Whether to skip 1x1 tables (layout containers)
        skip_single_column_tables: Whether to skip single column tables
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
class HWP5TableRegionInfo:
    """Additional information for HWP 5.0 table region.
    
    Stores reference to the table record and parsed metadata
    for use in Pass 2 extraction.
    
    Attributes:
        ctrl_header_record: Reference to CTRL_HEADER record
        table_record: Reference to HWPTAG_TABLE record
        row_count: Number of rows
        col_count: Number of columns
        cell_records: List of HWPTAG_LIST_HEADER records (cells)
    """
    ctrl_header_record: Optional[HwpRecord] = None
    table_record: Optional[HwpRecord] = None
    row_count: int = 0
    col_count: int = 0
    cell_records: List[HwpRecord] = field(default_factory=list)


@dataclass
class HWP5CellInfo:
    """Parsed cell information from HWP 5.0 table.
    
    Attributes:
        row_addr: Row position (0-based)
        col_addr: Column position (0-based)
        row_span: Number of rows this cell spans
        col_span: Number of columns this cell spans
        content: Cell text content
        para_count: Number of paragraphs in cell
    """
    row_addr: int = 0
    col_addr: int = 0
    row_span: int = 1
    col_span: int = 1
    content: str = ""
    para_count: int = 0


# ============================================================================
# HWP 5.0 Table Extractor Class
# ============================================================================

class HWP5TableExtractor(BaseTableExtractor):
    """HWP 5.0 format-specific table extractor.
    
    Extracts tables from HWP 5.0 OLE files by parsing binary records.
    Implements BaseTableExtractor interface from table_extractor.py.
    
    HWP 5.0 files are OLE compound documents containing:
    - BodyText/Section0, Section1, ... (document sections)
    - Each section contains binary records
    - Tables are CTRL_HEADER records with 'tbl ' control ID
    
    Supports:
    - Cell merges (rowspan/colspan via LIST_HEADER payload)
    - Grid-based cell positioning
    - Nested tables (recursive extraction)
    - 1x1 and single-column table flattening
    """
    
    def __init__(self, config: Optional[HWP5TableExtractorConfig] = None):
        """Initialize HWP 5.0 table extractor.
        
        Args:
            config: HWP 5.0 table extraction configuration
        """
        self._config = config or HWP5TableExtractorConfig()
        super().__init__(self._config)
        # Cache for region info (index -> HWP5TableRegionInfo)
        self._region_info_cache: Dict[int, HWP5TableRegionInfo] = {}
        # Callback for extracting cell text
        self._traverse_callback: Optional[Callable] = None
        # OLE file reference
        self._ole: Optional["olefile.OleFileIO"] = None
        # BinData mapping
        self._bin_data_map: Optional[Dict] = None
        # Processed images set
        self._processed_images: Optional[Set[str]] = None
    
    def set_context(
        self,
        traverse_callback: Callable,
        ole: "olefile.OleFileIO" = None,
        bin_data_map: Dict = None,
        processed_images: Set[str] = None
    ) -> None:
        """Set context for table extraction.
        
        These references are needed to extract cell content including
        images and nested objects.
        
        Args:
            traverse_callback: Function to traverse record tree
            ole: OLE file object
            bin_data_map: BinData mapping information
            processed_images: Set of processed image paths
        """
        self._traverse_callback = traverse_callback
        self._ole = ole
        self._bin_data_map = bin_data_map
        self._processed_images = processed_images
    
    # ========================================================================
    # Pass 1: Region Detection
    # ========================================================================
    
    def detect_table_regions(
        self,
        content: Any,
        **kwargs
    ) -> List[TableRegion]:
        """Detect table regions in HWP record tree.
        
        Searches for CTRL_HEADER records with 'tbl ' control ID.
        
        Args:
            content: Root HwpRecord of section tree
            **kwargs: Additional options
            
        Returns:
            List of TableRegion objects for detected tables
        """
        regions = []
        self._region_info_cache.clear()
        
        if not isinstance(content, HwpRecord):
            return regions
        
        # Find all table CTRL_HEADER records
        table_controls = self._find_table_controls(content)
        
        for idx, ctrl_record in enumerate(table_controls):
            # Get table record
            table_rec = ctrl_record.find_first_child_by_tag(HWPTAG_TABLE)
            if not table_rec:
                continue
            
            # Parse table dimensions
            if len(table_rec.payload) < 8:
                continue
            
            row_count = struct.unpack('<H', table_rec.payload[4:6])[0]
            col_count = struct.unpack('<H', table_rec.payload[6:8])[0]
            
            # Get cell records
            cell_records = ctrl_record.find_children_by_tag(HWPTAG_LIST_HEADER)
            
            # Calculate confidence based on structure
            confidence = 1.0
            if row_count == 1 and col_count == 1:
                confidence = 0.3  # Likely container table
            elif col_count == 1:
                confidence = 0.5  # Single column table
            
            # Create region
            region = TableRegion(
                start_offset=idx,  # Use index as identifier
                end_offset=idx,
                row_count=row_count,
                col_count=col_count,
                confidence=confidence,
                metadata={
                    'cell_count': len(cell_records),
                    'is_container': row_count == 1 and col_count == 1,
                    'is_single_column': col_count == 1,
                }
            )
            regions.append(region)
            
            # Cache region info for Pass 2
            self._region_info_cache[idx] = HWP5TableRegionInfo(
                ctrl_header_record=ctrl_record,
                table_record=table_rec,
                row_count=row_count,
                col_count=col_count,
                cell_records=cell_records,
            )
        
        logger.debug(f"Detected {len(regions)} table regions in HWP 5.0 record tree")
        return regions
    
    def _find_table_controls(self, root: HwpRecord) -> List[HwpRecord]:
        """Find all table CTRL_HEADER records in tree.
        
        Args:
            root: Root record to search from
            
        Returns:
            List of CTRL_HEADER records with 'tbl ' control ID
        """
        results = []
        
        def search(record: HwpRecord):
            if record.tag_id == HWPTAG_CTRL_HEADER:
                # Check control ID
                if len(record.payload) >= 4:
                    ctrl_id = record.payload[:4][::-1]  # Reverse byte order
                    if ctrl_id == CTRL_ID_TABLE:
                        results.append(record)
            
            for child in record.children:
                search(child)
        
        search(root)
        return results
    
    # ========================================================================
    # Pass 2: Table Extraction
    # ========================================================================
    
    def extract_table_from_region(
        self,
        content: Any,
        region: TableRegion,
        **kwargs
    ) -> Optional[TableData]:
        """Extract table content from detected region.
        
        Args:
            content: Root HwpRecord (not used, info is cached)
            region: TableRegion from Pass 1
            **kwargs: Additional options
            
        Returns:
            TableData object with extracted content, or None
        """
        region_idx = region.start_offset
        
        if region_idx not in self._region_info_cache:
            logger.warning(f"Region info not found for index {region_idx}")
            return None
        
        info = self._region_info_cache[region_idx]
        
        # Build cell grid
        grid = self._build_cell_grid(info)
        
        if not grid:
            return None
        
        # Handle container tables (1x1)
        if info.row_count == 1 and info.col_count == 1:
            if self._config.flatten_container_tables:
                if (0, 0) in grid:
                    # Return as text, not table
                    return None
        
        # Handle single column tables
        if info.col_count == 1:
            if self._config.skip_single_column_tables:
                return None
        
        # Build TableData
        rows = []
        for r in range(info.row_count):
            cells = []
            for c in range(info.col_count):
                if (r, c) in grid:
                    cell_info = grid[(r, c)]
                    cell = TableCell(
                        content=cell_info.content,
                        row_span=cell_info.row_span,
                        col_span=cell_info.col_span,
                        is_header=(r == 0),  # First row as header
                    )
                    cells.append(cell)
                else:
                    # Empty cell or spanned
                    cells.append(TableCell(content=""))
            rows.append(cells)
        
        return TableData(
            rows=rows,
            headers=rows[0] if rows else [],
            row_count=info.row_count,
            col_count=info.col_count,
            has_header=True,
            metadata={
                'source': 'hwp5',
                'cell_count': len(info.cell_records),
            }
        )
    
    def _build_cell_grid(
        self,
        info: HWP5TableRegionInfo
    ) -> Dict[tuple, HWP5CellInfo]:
        """Build cell grid from LIST_HEADER records.
        
        Args:
            info: Table region info
            
        Returns:
            Dictionary: (row, col) -> HWP5CellInfo
        """
        grid = {}
        
        for cell_rec in info.cell_records:
            if len(cell_rec.payload) < 16:
                continue
            
            # Parse cell position and spans
            para_count = struct.unpack('<H', cell_rec.payload[0:2])[0]
            col_addr = struct.unpack('<H', cell_rec.payload[8:10])[0]
            row_addr = struct.unpack('<H', cell_rec.payload[10:12])[0]
            col_span = struct.unpack('<H', cell_rec.payload[12:14])[0]
            row_span = struct.unpack('<H', cell_rec.payload[14:16])[0]
            
            # Extract cell content
            content = self._extract_cell_content(cell_rec, para_count)
            
            grid[(row_addr, col_addr)] = HWP5CellInfo(
                row_addr=row_addr,
                col_addr=col_addr,
                row_span=row_span,
                col_span=col_span,
                content=content,
                para_count=para_count,
            )
        
        return grid
    
    def _extract_cell_content(
        self,
        cell_rec: HwpRecord,
        para_count: int
    ) -> str:
        """Extract text content from cell record.
        
        Args:
            cell_rec: LIST_HEADER record
            para_count: Number of paragraphs
            
        Returns:
            Cell text content
        """
        parts = []
        
        # Method 1: Use traverse callback if available
        if self._traverse_callback:
            if cell_rec.children:
                for child in cell_rec.children:
                    text = self._traverse_callback(
                        child,
                        self._ole,
                        self._bin_data_map,
                        self._processed_images
                    )
                    parts.append(text)
            else:
                # Content in subsequent siblings
                siblings = list(cell_rec.get_next_siblings(para_count))
                for sibling in siblings:
                    text = self._traverse_callback(
                        sibling,
                        self._ole,
                        self._bin_data_map,
                        self._processed_images
                    )
                    parts.append(text)
        
        # Method 2: Direct text extraction (fallback)
        else:
            text_records = cell_rec.find_descendants_by_tag(HWPTAG_PARA_TEXT)
            for text_rec in text_records:
                text = text_rec.get_text()
                if text:
                    parts.append(text.replace('\x0b', ''))
        
        return "".join(parts).strip()
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def extract_table_from_record(
        self,
        ctrl_header: HwpRecord,
        traverse_callback: Callable = None,
        ole: "olefile.OleFileIO" = None,
        bin_data_map: Dict = None,
        processed_images: Set[str] = None
    ) -> Optional[TableData]:
        """Extract table from CTRL_HEADER record directly.
        
        Convenience method for extracting a single table.
        
        Args:
            ctrl_header: CTRL_HEADER record with 'tbl ' control ID
            traverse_callback: Function to traverse record tree
            ole: OLE file object
            bin_data_map: BinData mapping
            processed_images: Processed image paths
            
        Returns:
            TableData object or None
        """
        # Set context
        if traverse_callback:
            self.set_context(traverse_callback, ole, bin_data_map, processed_images)
        
        # Create temporary region
        table_rec = ctrl_header.find_first_child_by_tag(HWPTAG_TABLE)
        if not table_rec or len(table_rec.payload) < 8:
            return None
        
        row_count = struct.unpack('<H', table_rec.payload[4:6])[0]
        col_count = struct.unpack('<H', table_rec.payload[6:8])[0]
        cell_records = ctrl_header.find_children_by_tag(HWPTAG_LIST_HEADER)
        
        info = HWP5TableRegionInfo(
            ctrl_header_record=ctrl_header,
            table_record=table_rec,
            row_count=row_count,
            col_count=col_count,
            cell_records=cell_records,
        )
        
        # Cache and extract
        self._region_info_cache[0] = info
        region = TableRegion(
            start_offset=0,
            row_count=row_count,
            col_count=col_count,
            confidence=1.0,
        )
        
        return self.extract_table_from_region(None, region)
    
    def extract_all_tables(
        self,
        root: HwpRecord,
        traverse_callback: Callable = None,
        ole: "olefile.OleFileIO" = None,
        bin_data_map: Dict = None,
        processed_images: Set[str] = None
    ) -> List[TableData]:
        """Extract all tables from record tree.
        
        Args:
            root: Root record of section
            traverse_callback: Function to traverse record tree
            ole: OLE file object
            bin_data_map: BinData mapping
            processed_images: Processed image paths
            
        Returns:
            List of TableData objects
        """
        # Set context
        if traverse_callback:
            self.set_context(traverse_callback, ole, bin_data_map, processed_images)
        
        tables = []
        
        # Detect regions
        regions = self.detect_table_regions(root)
        
        # Extract each table
        for region in regions:
            table = self.extract_table_from_region(root, region)
            if table:
                tables.append(table)
        
        return tables


__all__ = [
    'HWP5TableExtractor',
    'HWP5TableExtractorConfig',
    'HWP5TableRegionInfo',
    'HWP5CellInfo',
]
