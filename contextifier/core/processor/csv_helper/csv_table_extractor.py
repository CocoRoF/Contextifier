# contextifier/core/processor/csv_helper/csv_table_extractor.py
"""
CSV Table Extractor - CSV Format-Specific Table Extraction

Implements table extraction for CSV/TSV files.
Follows BaseTableExtractor interface from table_extractor.py.

CSV Table Structure:
- CSV files are inherently tabular (entire file = single table)
- Parsed rows from csv module or simple split
- Merge detection based on empty cell patterns

2-Pass Approach (simplified for CSV):
1. Pass 1: Detect table region (entire CSV content = 1 region)
2. Pass 2: Extract content with merge analysis

Usage:
    from contextifier.core.processor.csv_helper.csv_table_extractor import (
        CSVTableExtractor,
    )

    extractor = CSVTableExtractor()
    
    # From parsed rows
    tables = extractor.extract_tables(rows)  # List[List[str]]
    
    # Or from content string
    tables = extractor.extract_tables_from_content(content, delimiter)
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)
from contextifier.core.processor.csv_helper.csv_parser import (
    parse_csv_content,
    detect_delimiter,
    detect_header,
)

logger = logging.getLogger("document-processor")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CSVTableExtractorConfig(TableExtractorConfig):
    """Configuration specific to CSV table extraction.
    
    Attributes:
        detect_merged_cells: Whether to detect merged cells from empty patterns
        treat_first_row_as_header: Whether to treat first row as header
        auto_detect_header: Whether to auto-detect header row
        skip_empty_rows: Whether to skip completely empty rows
        max_rows: Maximum rows to process
        max_cols: Maximum columns to process
    """
    detect_merged_cells: bool = True
    treat_first_row_as_header: bool = True
    auto_detect_header: bool = True
    skip_empty_rows: bool = True
    max_rows: int = 10000
    max_cols: int = 1000


# ============================================================================
# Data Classes for Table Region Tracking
# ============================================================================

@dataclass
class CSVTableRegionInfo:
    """Additional information for CSV table region.
    
    For CSV, the entire file is typically one table,
    so this stores the parsed rows and metadata.
    """
    rows: List[List[str]] = field(default_factory=list)  # Parsed row data
    has_header: bool = False                              # Header detection result
    delimiter: str = ","                                  # Detected delimiter
    row_count: int = 0                                    # Number of rows
    col_count: int = 0                                    # Number of columns
    has_merged_cells: bool = False                        # Merge cell detection result


# ============================================================================
# Merge Cell Analysis Classes
# ============================================================================

@dataclass
class CSVCellMergeInfo:
    """Merge information for a single CSV cell.
    
    Attributes:
        value: Cell content
        colspan: Horizontal merge span (1 = no merge)
        rowspan: Vertical merge span (1 = no merge)
        skip: Whether this cell should be skipped in rendering
    """
    value: str = ""
    colspan: int = 1
    rowspan: int = 1
    skip: bool = False


# ============================================================================
# CSV Table Extractor Class
# ============================================================================

class CSVTableExtractor(BaseTableExtractor):
    """CSV format-specific table extractor.
    
    Extracts tables from CSV/TSV files.
    Implements BaseTableExtractor interface.
    
    CSV files are inherently tabular, so the entire file content
    is treated as a single table region.
    
    Supports:
    - Merge cell detection from empty cell patterns
    - Header row detection (automatic or manual)
    - Multiple delimiter support (comma, tab, semicolon, pipe)
    """
    
    def __init__(self, config: Optional[CSVTableExtractorConfig] = None):
        """Initialize CSV table extractor.
        
        Args:
            config: CSV table extraction configuration
        """
        self._config = config or CSVTableExtractorConfig()
        super().__init__(self._config)
        # Cache for region info
        self._region_info_cache: Dict[int, CSVTableRegionInfo] = {}
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() in ("csv", "tsv", "txt")
    
    # ========================================================================
    # BaseTableExtractor Interface Implementation
    # ========================================================================
    
    def detect_table_regions(self, content: Any) -> List[TableRegion]:
        """Detect table regions in CSV content.
        
        For CSV, the entire content is one table region.
        
        Args:
            content: Parsed rows (List[List[str]]) or raw content string
            
        Returns:
            List containing single TableRegion (CSV = 1 table)
        """
        self._region_info_cache.clear()
        
        # Handle different content types
        rows, delimiter = self._parse_content(content)
        
        if not rows:
            return []
        
        # Calculate dimensions
        row_count = len(rows)
        col_count = max(len(row) for row in rows) if rows else 0
        
        # Detect header
        has_header = False
        if self._config.auto_detect_header:
            has_header = detect_header(rows)
        elif self._config.treat_first_row_as_header:
            has_header = True
        
        # Detect merged cells
        has_merged = False
        if self._config.detect_merged_cells:
            has_merged = self._has_merged_cells(rows)
        
        # Create region info (cache for Pass 2)
        region_info = CSVTableRegionInfo(
            rows=rows,
            has_header=has_header,
            delimiter=delimiter,
            row_count=row_count,
            col_count=col_count,
            has_merged_cells=has_merged,
        )
        self._region_info_cache[0] = region_info
        
        # Create single TableRegion
        region = TableRegion(
            start_offset=0,
            end_offset=1,
            row_count=row_count,
            col_count=col_count,
            confidence=self._calculate_confidence(rows),
        )
        
        self.logger.debug(f"CSV table detected: {row_count} rows x {col_count} cols, header={has_header}, merged={has_merged}")
        
        return [region]
    
    def extract_table_from_region(
        self, 
        content: Any, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from CSV region.
        
        Args:
            content: Parsed rows or raw content (may be ignored if cached)
            region: TableRegion (for CSV, typically index 0)
            
        Returns:
            TableData object or None if extraction fails
        """
        region_idx = region.start_offset
        region_info = self._region_info_cache.get(region_idx)
        
        if region_info is None:
            # If not cached, re-parse
            rows, _ = self._parse_content(content)
            if not rows:
                return None
            region_info = CSVTableRegionInfo(
                rows=rows,
                has_header=detect_header(rows) if self._config.auto_detect_header else self._config.treat_first_row_as_header,
                row_count=len(rows),
                col_count=max(len(row) for row in rows) if rows else 0,
            )
        
        rows = region_info.rows
        has_header = region_info.has_header
        
        try:
            # Analyze merge information
            merge_info = self._analyze_merge_info(rows) if self._config.detect_merged_cells else None
            
            # Build TableData
            table_rows: List[List[TableCell]] = []
            col_count = region_info.col_count
            
            for row_idx, row in enumerate(rows):
                row_cells: List[TableCell] = []
                
                for col_idx in range(col_count):
                    cell_value = row[col_idx].strip() if col_idx < len(row) else ""
                    
                    # Get merge info
                    colspan = 1
                    rowspan = 1
                    skip = False
                    
                    if merge_info and row_idx < len(merge_info) and col_idx < len(merge_info[row_idx]):
                        cell_merge = merge_info[row_idx][col_idx]
                        colspan = cell_merge.colspan
                        rowspan = cell_merge.rowspan
                        skip = cell_merge.skip
                    
                    # Skip cells that are part of a merge
                    if skip:
                        # Create cell with 0 span to indicate merged
                        table_cell = TableCell(
                            content="",
                            row_span=0,
                            col_span=0,
                            is_header=(row_idx == 0 and has_header),
                            row_index=row_idx,
                            col_index=col_idx,
                        )
                    else:
                        table_cell = TableCell(
                            content=cell_value,
                            row_span=rowspan,
                            col_span=colspan,
                            is_header=(row_idx == 0 and has_header),
                            row_index=row_idx,
                            col_index=col_idx,
                        )
                    
                    row_cells.append(table_cell)
                
                table_rows.append(row_cells)
            
            # Create TableData
            table_data = TableData(
                rows=table_rows,
                num_rows=len(table_rows),
                num_cols=col_count,
                has_header=has_header,
                source_format="csv",
                metadata={
                    "delimiter": region_info.delimiter,
                    "has_merged_cells": region_info.has_merged_cells,
                }
            )
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Error extracting CSV table: {e}")
            return None
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def extract_tables_from_content(
        self, 
        content: str, 
        delimiter: Optional[str] = None
    ) -> List[TableData]:
        """Extract tables from raw CSV content string.
        
        Convenience method that handles parsing internally.
        
        Args:
            content: Raw CSV content string
            delimiter: Delimiter character (None for auto-detect)
            
        Returns:
            List of TableData objects
        """
        if delimiter is None:
            delimiter = detect_delimiter(content)
        
        rows = parse_csv_content(content, delimiter)
        return self.extract_tables(rows)
    
    def extract_single_table(
        self, 
        rows: List[List[str]], 
        has_header: Optional[bool] = None
    ) -> Optional[TableData]:
        """Extract single table from parsed rows.
        
        Simplified method for direct table extraction.
        
        Args:
            rows: Parsed row data
            has_header: Header flag (None for auto-detect)
            
        Returns:
            TableData object or None
        """
        if not rows:
            return None
        
        regions = self.detect_table_regions(rows)
        if not regions:
            return None
        
        # Override header detection if specified
        if has_header is not None and 0 in self._region_info_cache:
            self._region_info_cache[0].has_header = has_header
        
        return self.extract_table_from_region(rows, regions[0])
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _parse_content(self, content: Any) -> Tuple[List[List[str]], str]:
        """Parse content to rows and delimiter.
        
        Args:
            content: Rows list or raw string
            
        Returns:
            Tuple of (rows, delimiter)
        """
        if isinstance(content, list):
            # Already parsed rows
            return content, ","
        elif isinstance(content, str):
            # Raw content string
            delimiter = detect_delimiter(content)
            rows = parse_csv_content(content, delimiter)
            return rows, delimiter
        else:
            self.logger.warning(f"Unsupported content type: {type(content)}")
            return [], ","
    
    def _has_merged_cells(self, rows: List[List[str]]) -> bool:
        """Check if CSV data has merged cell patterns.
        
        Merge detection criteria:
        - Consecutive empty cells exist
        - First column has empty cell with previous row having value (vertical merge)
        
        Args:
            rows: Parsed row data
            
        Returns:
            True if merge patterns detected
        """
        if not rows or len(rows) < 2:
            return False
        
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell_value = cell.strip() if cell else ""
                
                if not cell_value:
                    # First column empty (not first row) -> vertical merge
                    if row_idx > 0 and col_idx == 0:
                        return True
                    
                    # Previous cell not empty but current empty -> horizontal merge
                    if col_idx > 0:
                        prev_cell = row[col_idx - 1].strip() if col_idx - 1 < len(row) else ""
                        if prev_cell:
                            return True
        
        return False
    
    def _analyze_merge_info(self, rows: List[List[str]]) -> List[List[CSVCellMergeInfo]]:
        """Analyze merge cell information from empty cell patterns.
        
        Calculates colspan (horizontal) and rowspan (vertical) from empty cells.
        
        Args:
            rows: Parsed row data
            
        Returns:
            2D list of CSVCellMergeInfo for each cell
        """
        if not rows:
            return []
        
        row_count = len(rows)
        col_count = max(len(row) for row in rows) if rows else 0
        
        # Initialize merge info
        merge_info: List[List[CSVCellMergeInfo]] = []
        for row_idx, row in enumerate(rows):
            row_info = []
            for col_idx in range(col_count):
                cell_value = row[col_idx].strip() if col_idx < len(row) else ""
                row_info.append(CSVCellMergeInfo(
                    value=cell_value,
                    colspan=1,
                    rowspan=1,
                    skip=False,
                ))
            merge_info.append(row_info)
        
        # Pass 1: Calculate colspan (horizontal merge - right consecutive empty cells)
        for row_idx in range(row_count):
            col_idx = 0
            while col_idx < col_count:
                cell_info = merge_info[row_idx][col_idx]
                
                if cell_info.skip or not cell_info.value:
                    col_idx += 1
                    continue
                
                # Count consecutive empty cells to the right
                colspan = 1
                next_col = col_idx + 1
                while next_col < col_count:
                    next_cell = merge_info[row_idx][next_col]
                    if not next_cell.value and not next_cell.skip:
                        colspan += 1
                        next_cell.skip = True
                        next_col += 1
                    else:
                        break
                
                cell_info.colspan = colspan
                col_idx = next_col
        
        # Pass 2: Calculate rowspan (vertical merge - down consecutive empty cells)
        for col_idx in range(col_count):
            row_idx = 0
            while row_idx < row_count:
                cell_info = merge_info[row_idx][col_idx]
                
                if cell_info.skip or not cell_info.value:
                    row_idx += 1
                    continue
                
                # Count consecutive empty cells downward
                rowspan = 1
                next_row = row_idx + 1
                while next_row < row_count:
                    next_cell = merge_info[next_row][col_idx]
                    if not next_cell.value and not next_cell.skip:
                        rowspan += 1
                        next_cell.skip = True
                        next_row += 1
                    else:
                        break
                
                cell_info.rowspan = rowspan
                row_idx = next_row
        
        return merge_info
    
    def _calculate_confidence(self, rows: List[List[str]]) -> float:
        """Calculate confidence score for CSV table.
        
        CSV files are inherently tabular, so confidence is typically high.
        
        Args:
            rows: Parsed row data
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        if not rows:
            return 0.0
        
        # Base confidence for CSV
        confidence = 0.9
        
        # Check row consistency (all rows have similar column count)
        col_counts = [len(row) for row in rows]
        if len(set(col_counts)) == 1:
            confidence += 0.05  # Perfectly consistent
        elif max(col_counts) - min(col_counts) <= 2:
            confidence += 0.02  # Minor variation
        
        # Check data density
        total_cells = sum(len(row) for row in rows)
        non_empty_cells = sum(1 for row in rows for cell in row if cell.strip())
        if total_cells > 0:
            density = non_empty_cells / total_cells
            confidence += density * 0.05
        
        return min(confidence, 1.0)


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def has_merged_cells(rows: List[List[str]]) -> bool:
    """Check if CSV data has merged cell patterns.
    
    Backward compatible function.
    
    Args:
        rows: Parsed row data
        
    Returns:
        True if merge patterns detected
    """
    extractor = CSVTableExtractor()
    return extractor._has_merged_cells(rows)


def analyze_merge_info(rows: List[List[str]]) -> List[List[Dict[str, Any]]]:
    """Analyze merge cell information.
    
    Backward compatible function. Returns dict format for compatibility.
    
    Args:
        rows: Parsed row data
        
    Returns:
        2D list of merge info dicts
    """
    extractor = CSVTableExtractor()
    merge_info = extractor._analyze_merge_info(rows)
    
    # Convert to dict format for backward compatibility
    result = []
    for row_info in merge_info:
        row_dicts = []
        for cell_info in row_info:
            row_dicts.append({
                'value': cell_info.value,
                'colspan': cell_info.colspan,
                'rowspan': cell_info.rowspan,
                'skip': cell_info.skip,
            })
        result.append(row_dicts)
    
    return result


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    "CSVTableExtractorConfig",
    # Region Info
    "CSVTableRegionInfo",
    "CSVCellMergeInfo",
    # Extractor
    "CSVTableExtractor",
    # Backward compatibility
    "has_merged_cells",
    "analyze_merge_info",
]
