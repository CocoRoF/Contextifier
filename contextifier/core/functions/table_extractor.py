# libs/core/functions/table_extractor.py
"""
Table Extractor - Abstract Interface for Table Extraction

Provides abstract base classes and data structures for table extraction.
Format-specific implementations should be placed in respective helper modules.

Module Components:
- TableCell: Data class for table cell information
- TableData: Data class for complete table information
- TableRegion: Data class for detected table regions
- BaseTableExtractor: Abstract base class for format-specific extractors
- NullTableExtractor: No-op extractor for unsupported formats

Usage Example:
    from contextifier.core.functions.table_extractor import (
        BaseTableExtractor,
        TableData,
        TableRegion,
    )

    class DOCTableExtractor(BaseTableExtractor):
        def detect_table_regions(self, content):
            # DOC-specific implementation
            pass
        
        def extract_table_from_region(self, content, region):
            # DOC-specific implementation
            pass
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
    
    The extraction process follows a 2-Pass approach:
    1. detect_table_regions(): Identify where tables are in the document
    2. extract_table_from_region(): Extract actual table data from regions
    
    Implementation Guidelines:
        - Implement detect_table_regions() to find potential table locations
        - Implement extract_table_from_region() to parse table content
        - Override supports_format() to declare supported formats
        - Use config for validation parameters
    """
    
    def __init__(self, config: Optional[TableExtractorConfig] = None):
        """Initialize the extractor.
        
        Args:
            config: Table extraction configuration
        """
        self.config = config or TableExtractorConfig()
        self.logger = logging.getLogger("document-processor")
    
    @abstractmethod
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in the document content.
        
        Pass 1 of 2-Pass approach: Scan document to find potential table locations.
        
        Args:
            content: Document content (bytes, str, or format-specific object)
            
        Returns:
            List of TableRegion objects representing detected table locations
        """
        pass
    
    @abstractmethod
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected region.
        
        Pass 2 of 2-Pass approach: Extract actual table content from region.
        
        Args:
            content: Document content (bytes, str, or format-specific object)
            region: TableRegion identifying where the table is
            
        Returns:
            TableData object or None if extraction fails
        """
        pass
    
    def extract_tables(self, content: Any) -> List[TableData]:
        """Extract all tables from document content.
        
        Main entry point that combines both passes:
        1. Detect all table regions
        2. Extract tables from each region
        
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


# Default configuration
DEFAULT_EXTRACTOR_CONFIG = TableExtractorConfig()
