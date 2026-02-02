# libs/core/functions/table_extractor.py
"""
Table Extractor - Abstract Interface for Table Extraction

Provides abstract base classes and data structures for table extraction.
Format-specific implementations should be placed in respective helper modules.

================================================================================
TABLE EXTRACTION ARCHITECTURE
================================================================================

This module defines the common interface for all format-specific table extractors.
There are TWO main extraction approaches supported:

┌─────────────────────────────────────────────────────────────────────────────┐
│ APPROACH 1: Batch Processing (전체 문서 일괄 처리)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Method: extract_tables(content) -> List[TableData]                          │
│                                                                             │
│ Description:                                                                │
│   - Extracts ALL tables from the entire document at once                    │
│   - Uses 2-Pass approach internally:                                        │
│     Pass 1: detect_table_regions() - Find table locations                   │
│     Pass 2: extract_table_from_region() - Extract from each region          │
│                                                                             │
│ Use Cases:                                                                  │
│   - PDF: Tables detected via layout analysis, extracted in batch            │
│   - Excel: All sheets processed together                                    │
│   - Scanned documents: OCR-based table detection                            │
│                                                                             │
│ Implemented By:                                                             │
│   - PDFTableExtractor (planned)                                             │
│   - ExcelTableExtractor (planned)                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ APPROACH 2: Streaming/Element Processing (요소 단위 실시간 처리)              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Method: extract_table(element, context) -> Optional[TableData]              │
│                                                                             │
│ Description:                                                                │
│   - Extracts a SINGLE table from an element/node                            │
│   - Called in real-time as document is traversed                            │
│   - More memory efficient for large documents                               │
│   - Preserves document order naturally                                      │
│                                                                             │
│ Use Cases:                                                                  │
│   - DOCX: Tables are explicit <w:tbl> elements                              │
│   - PPTX: Tables are shape elements in slides                               │
│   - HTML: Tables are <table> elements                                       │
│                                                                             │
│ Implemented By:                                                             │
│   - DOCXTableExtractor (contextifier.core.processor.docx_helper)            │
│   - PPTXTableExtractor (planned)                                            │
│   - HTMLTableExtractor (planned)                                            │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
IMPLEMENTATION STATUS BY FORMAT
================================================================================

| Format | Extractor Class      | Approach  | Status      | Location                    |
|--------|---------------------|-----------|-------------|------------------------------|
| DOCX   | DOCXTableExtractor  | Streaming | ✅ Complete | docx_helper/docx_table_extractor.py |
| DOC    | DOCTableExtractor   | Batch     | ❌ Planned  | doc_helper/                  |
| PDF    | PDFTableExtractor   | Batch     | ❌ Planned  | pdf_helper/                  |
| XLSX   | ExcelTableExtractor | Batch     | ❌ Planned  | excel_helper/                |
| PPTX   | PPTXTableExtractor  | Streaming | ❌ Planned  | pptx_helper/                 |
| HTML   | HTMLTableExtractor  | Streaming | ❌ Planned  | html_helper/                 |
| HWP    | HWPTableExtractor   | Batch     | ❌ Planned  | hwp_helper/                  |

================================================================================
MODULE COMPONENTS
================================================================================

- TableCell: Data class for table cell information
- TableData: Data class for complete table information  
- TableRegion: Data class for detected table regions (Batch approach)
- TableExtractorConfig: Configuration for extraction behavior
- BaseTableExtractor: Abstract base class for format-specific extractors
- NullTableExtractor: No-op extractor for unsupported formats

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Batch Processing (PDF, Excel)
    
    from contextifier.core.functions.table_extractor import BaseTableExtractor
    
    class PDFTableExtractor(BaseTableExtractor):
        def detect_table_regions(self, content):
            # Scan PDF for table-like regions
            return [TableRegion(...), ...]
        
        def extract_table_from_region(self, content, region):
            # Extract table from specific region
            return TableData(...)
        
        # Use inherited extract_tables() for batch processing
    
    extractor = PDFTableExtractor()
    tables = extractor.extract_tables(pdf_content)  # Returns List[TableData]

Example 2: Streaming Processing (DOCX, PPTX)
    
    from contextifier.core.functions.table_extractor import BaseTableExtractor
    
    class DOCXTableExtractor(BaseTableExtractor):
        def extract_table(self, element, context=None):
            # Extract single table from <w:tbl> element
            return TableData(...)  # or None if invalid
    
    extractor = DOCXTableExtractor()
    
    # Called during document traversal:
    for elem in doc.body:
        if is_table(elem):
            table = extractor.extract_table(elem, doc)  # Returns Optional[TableData]
            if table:
                process(table)
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("document-processor")


@dataclass
class TableCell:
    """Represents a single table cell.
    
    Attributes:
        content: Cell content (text)
        row_span: Number of rows this cell spans
        col_span: Number of columns this cell spans
        is_header: Whether this cell is a header cell
        row_index: Row position in the table
        col_index: Column position in the table
        nested_table: Nested table data if this cell contains a table
    """
    content: str = ""
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    row_index: int = 0
    col_index: int = 0
    nested_table: Optional['TableData'] = None


@dataclass
class TableData:
    """Data class for table information.
    
    Attributes:
        rows: 2D list of TableCell objects
        num_rows: Number of rows
        num_cols: Number of columns
        has_header: Whether the table has a header row
        start_offset: Byte offset where the table starts (for binary formats)
        end_offset: Byte offset where the table ends (for binary formats)
        source_format: Source format identifier (e.g., "doc", "docx", "xlsx")
        metadata: Additional metadata about the table
        col_widths_percent: Column widths as percentages (e.g., [25.0, 50.0, 25.0])
    """
    rows: List[List[TableCell]] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    has_header: bool = False
    start_offset: int = 0
    end_offset: int = 0
    source_format: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    col_widths_percent: List[float] = field(default_factory=list)
    
    def is_valid(self, min_rows: int = 2, min_cols: int = 2) -> bool:
        """Check if this table meets minimum requirements."""
        return self.num_rows >= min_rows and self.num_cols >= min_cols


@dataclass 
class TableRegion:
    """Represents a detected table region in the document.
    
    Used for 2-Pass table detection approach:
    - Pass 1: Detect table regions (TableRegion objects)
    - Pass 2: Extract content from regions (TableData objects)
    
    Attributes:
        start_offset: Start position in the document
        end_offset: End position in the document
        row_count: Estimated number of rows
        col_count: Estimated number of columns
        confidence: Confidence score (0.0 - 1.0)
        metadata: Additional metadata (optional)
    """
    start_offset: int = 0
    end_offset: int = 0
    row_count: int = 0
    col_count: int = 0
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def is_confident(self, threshold: float = 0.5) -> bool:
        """Check if this region detection is confident enough."""
        return self.confidence >= threshold


@dataclass
class TableExtractorConfig:
    """Configuration for table extraction.
    
    Attributes:
        min_rows: Minimum rows to consider as a table
        min_cols: Minimum columns to consider as a table
        confidence_threshold: Minimum confidence to accept a table region
        include_header_row: Whether to mark first row as header
    """
    min_rows: int = 2
    min_cols: int = 2
    confidence_threshold: float = 0.5
    include_header_row: bool = True


class BaseTableExtractor(ABC):
    """Abstract base class for format-specific table extractors.
    
    Each document format (DOC, DOCX, XLSX, etc.) should implement
    a subclass of BaseTableExtractor with format-specific logic.
    
    ============================================================================
    SUPPORTED EXTRACTION APPROACHES
    ============================================================================
    
    APPROACH 1: Batch Processing (전체 문서 일괄 처리)
    ------------------------------------------------
    Uses 2-Pass detection and extraction:
    - detect_table_regions(): Find all table locations in document
    - extract_table_from_region(): Extract table from each location
    - extract_tables(): Combines both passes (main entry point)
    
    Suitable for: PDF, DOC, Excel, HWP (where tables need detection)
    
    APPROACH 2: Streaming Processing (요소 단위 실시간 처리)
    -------------------------------------------------------
    Uses direct element extraction:
    - extract_table(): Extract single table from element/node
    
    Suitable for: DOCX, PPTX, HTML (where tables are explicit elements)
    
    ============================================================================
    IMPLEMENTATION GUIDE
    ============================================================================
    
    For Batch Processing (PDF, Excel, etc.):
        - Override detect_table_regions() - REQUIRED
        - Override extract_table_from_region() - REQUIRED
        - Use extract_tables() as main entry point
    
    For Streaming Processing (DOCX, PPTX, etc.):
        - Override extract_table() - REQUIRED
        - detect_table_regions() can return empty list
        - extract_table_from_region() can return None
        - Call extract_table() directly during document traversal
    
    ============================================================================
    """
    
    def __init__(self, config: Optional[TableExtractorConfig] = None):
        """Initialize the extractor.
        
        Args:
            config: Table extraction configuration
        """
        self.config = config or TableExtractorConfig()
        self.logger = logging.getLogger("document-processor")
    
    # ==========================================================================
    # APPROACH 1: Batch Processing Methods (PDF, DOC, Excel, HWP)
    # ==========================================================================
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in the document content.
        
        [BATCH PROCESSING - Pass 1]
        Scan document to find potential table locations.
        
        Override this method for formats that require table detection:
        - PDF: Layout analysis to find table-like structures
        - DOC: Binary format parsing for table markers
        - Excel: Sheet enumeration
        
        Args:
            content: Document content (bytes, str, or format-specific object)
            
        Returns:
            List of TableRegion objects representing detected table locations
            
        Note:
            For streaming formats (DOCX, PPTX), this can return empty list
            as tables are processed via extract_table() instead.
        """
        # Default implementation returns empty list
        # Override for batch processing formats
        return []
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected region.
        
        [BATCH PROCESSING - Pass 2]
        Extract actual table content from a specific region.
        
        Override this method for formats that use region-based extraction:
        - PDF: Extract from page coordinates
        - DOC: Extract from byte offsets
        - Excel: Extract from sheet/cell ranges
        
        Args:
            content: Document content (bytes, str, or format-specific object)
            region: TableRegion identifying where the table is
            
        Returns:
            TableData object or None if extraction fails
            
        Note:
            For streaming formats (DOCX, PPTX), this can return None
            as tables are processed via extract_table() instead.
        """
        # Default implementation returns None
        # Override for batch processing formats
        return None
    
    def extract_tables(self, content: Any) -> List[TableData]:
        """Extract all tables from document content using batch processing.
        
        [BATCH PROCESSING - Main Entry Point]
        Combines both passes for complete extraction:
        1. Detect all table regions
        2. Extract tables from each region
        
        Used by: PDF, DOC, Excel, HWP extractors
        
        Args:
            content: Document content
            
        Returns:
            List of TableData objects
        """
        tables = []
        
        # Pass 1: Detect regions
        regions = self.detect_table_regions(content)
        self.logger.debug(f"Detected {len(regions)} table regions")
        
        # Pass 2: Extract from each region
        for region in regions:
            if region.is_confident(self.config.confidence_threshold):
                table = self.extract_table_from_region(content, region)
                if table and table.is_valid(self.config.min_rows, self.config.min_cols):
                    tables.append(table)
        
        self.logger.debug(f"Extracted {len(tables)} valid tables")
        return tables
    
    # ==========================================================================
    # APPROACH 2: Streaming Processing Methods (DOCX, PPTX, HTML)
    # ==========================================================================
    
    def extract_table(
        self, 
        element: Any, 
        context: Any = None
    ) -> Optional[TableData]:
        """Extract a single table from an element/node.
        
        [STREAMING PROCESSING - Main Entry Point]
        Extract table data from a specific element during document traversal.
        Called in real-time as the document is being processed.
        
        Override this method for formats with explicit table elements:
        - DOCX: <w:tbl> XML element → TableData
        - PPTX: Table shape element → TableData  
        - HTML: <table> DOM element → TableData
        
        Used by:
        - DOCXTableExtractor: Extracts from <w:tbl> elements
        - PPTXTableExtractor: Extracts from slide table shapes (planned)
        - HTMLTableExtractor: Extracts from <table> elements (planned)
        
        Args:
            element: Table element/node (format-specific)
                - DOCX: lxml Element (<w:tbl>)
                - PPTX: Shape object
                - HTML: DOM Element
            context: Optional context object for additional information
                - DOCX: Document object
                - PPTX: Slide object
                - HTML: Parent document
            
        Returns:
            TableData object or None if extraction fails/invalid
            
        Example (DOCX):
            for elem in doc.body:
                if elem.tag.endswith('tbl'):
                    table_data = extractor.extract_table(elem, doc)
                    if table_data:
                        html = processor.format_table_as_html(table_data)
        """
        # Default implementation returns None
        # Override for streaming processing formats
        return None
    
    # ==========================================================================
    # Common Methods
    # ==========================================================================
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format.
        
        Args:
            format_type: Format identifier (e.g., "doc", "docx")
            
        Returns:
            True if format is supported
        """
        return False


class NullTableExtractor(BaseTableExtractor):
    """No-op table extractor for unsupported formats.
    
    Returns empty results for all operations.
    Used as a fallback when no format-specific extractor is available.
    """
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Return empty list (no table detection)."""
        return []
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Return None (no table extraction)."""
        return None
    
    def extract_tables(self, content: Any) -> List[TableData]:
        """Return empty list (no tables)."""
        return []
    
    def extract_table(
        self, 
        element: Any, 
        context: Any = None
    ) -> Optional[TableData]:
        """Return None (no table extraction)."""
        return None


# Default configuration
DEFAULT_EXTRACTOR_CONFIG = TableExtractorConfig()
