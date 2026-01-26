# contextifier/core/processor/ole_helper/ole_table_extractor.py
"""
OLE Table Extractor - OLE Format-Specific Table Extraction

Implements table extraction for legacy DOC files (OLE Compound Documents).
Uses 0x07 cell markers to detect and extract tables.
Now also parses sprmTDefTable for accurate merge information.

DOC Binary Format Table Structure:
- 0x07: Cell end marker
- 0x07 0x07: Row end marker (cell end + row end)
- Tables are stored as a sequence of cells in the WordDocument stream
- Cell merge info (colspan/rowspan) is in sprmTDefTable (0xD608) structures

2-Pass Approach:
1. Pass 1: Detect table regions by analyzing 0x07 marker patterns
2. Pass 2: Extract content from detected regions
3. Pass 3: Apply merge info from sprmTDefTable

Usage:
    from contextifier.core.processor.ole_helper.ole_table_extractor import (
        OLETableExtractor,
        extract_tables_from_word_stream,
    )

    extractor = OLETableExtractor()
    tables = extractor.extract_tables(word_stream_data)
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)
from contextifier.core.processor.ole_helper.ole_encoding import (
    OLEEncoder,
    OLEEncodingType,
    OLEEncodingInfo,
)
from contextifier.core.processor.ole_helper.ole_table_properties import (
    OLETablePropertiesParser,
    extract_table_merge_info,
)

logger = logging.getLogger("document-processor")


@dataclass
class CellMarkerInfo:
    """Information about a 0x07 cell marker.
    
    Attributes:
        offset: Byte offset of the marker
        is_row_end: Whether this marker indicates end of row (0x07 0x07)
    """
    offset: int
    is_row_end: bool = False


@dataclass
class OLETableExtractorConfig(TableExtractorConfig):
    """Configuration specific to OLE table extraction.
    
    Attributes:
        max_marker_gap: Maximum bytes between markers to consider same table
        min_cells_for_table: Minimum cell count to consider as table
    """
    max_marker_gap: int = 500
    min_cells_for_table: int = 4


class OLETableExtractor(BaseTableExtractor):
    """OLE format-specific table extractor.
    
    Extracts tables from DOC files by analyzing 0x07 cell markers
    in the WordDocument stream and parsing sprmTDefTable for merge info.
    
    Supports multiple encodings:
    - UTF-16LE (Far East documents): 0x07 0x00 cell marker
    - Single-byte (Western documents): 0x07 cell marker
    """
    
    def __init__(self, config: Optional[OLETableExtractorConfig] = None):
        """Initialize OLE table extractor.
        
        Args:
            config: OLE table extraction configuration
        """
        self._config = config or OLETableExtractorConfig()
        super().__init__(self._config)
        self._encoder = OLEEncoder()
        # Cache for merge info
        self._colspan_map: Dict[Tuple[int, int], int] = {}
        self._rowspan_map: Dict[Tuple[int, int], int] = {}
        self._merge_info_parsed = False
        # Encoding info (detected per document)
        self._encoding_info: Optional[OLEEncodingInfo] = None
    
    def _parse_merge_info(self, content: bytes) -> None:
        """Parse merge information from sprmTDefTable structures.
        
        Args:
            content: Raw bytes from WordDocument stream
        """
        if self._merge_info_parsed:
            return
        
        try:
            self._colspan_map, self._rowspan_map = extract_table_merge_info(content)
            self._merge_info_parsed = True
            logger.debug(
                f"Parsed merge info: colspan={self._colspan_map}, "
                f"rowspan={self._rowspan_map}"
            )
        except Exception as e:
            logger.warning(f"Failed to parse table merge info: {e}")
            self._merge_info_parsed = True  # Don't retry
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this extractor supports the given format."""
        return format_type.lower() in ("doc", "ole")
    
    def detect_table_regions(self, content: bytes) -> List[TableRegion]:
        """Detect table regions by finding 0x07 marker patterns.
        
        Pass 1: Scan the WordDocument stream for consecutive 0x07 patterns.
        Automatically detects encoding and uses appropriate marker pattern.
        
        Args:
            content: Raw bytes from WordDocument stream
            
        Returns:
            List of TableRegion objects
        """
        if not content or len(content) < 10:
            return []
        
        # Detect encoding first (this affects how we find cell markers)
        self._encoding_info = self._encoder._detect_encoding_from_fib(content)
        logger.debug(f"Detected encoding: {self._encoding_info.codepage}")
        
        # Parse FIB to get text area boundaries
        text_start, text_end = self._get_text_area_from_fib(content)
        
        # Find all 0x07 markers in text area only (encoding-aware)
        markers = self._find_cell_markers_in_text_area(content, text_start, text_end)
        
        if len(markers) < self._config.min_cells_for_table:
            return []
        
        # Group markers into potential table regions
        regions = self._group_markers_into_regions(markers, content)
        
        return regions
    
    def _get_text_area_from_fib(self, data: bytes) -> tuple:
        """Parse FIB (File Information Block) to get text area boundaries.
        
        DOC format stores text position info in FIB:
        - fcMin (offset 0x18): Start of text in file
        - ccpText (offset 0x4C): Character count for main document text
        
        For single-byte documents: text_end = fcMin + ccpText (1 byte per char)
        For UTF-16LE documents: text_end = fcMin + ccpText * 2 (2 bytes per char)
        
        Args:
            data: WordDocument stream data
            
        Returns:
            Tuple of (text_start_offset, text_end_offset)
        """
        import struct
        
        if len(data) < 0x60:  # FIB minimum size
            return 0, len(data)
        
        try:
            # Check DOC signature
            w_ident = struct.unpack('<H', data[0:2])[0]
            if w_ident != 0xA5EC:
                # Not a valid DOC - use heuristics
                return 0, len(data)
            
            # fcMin: Text start offset (offset 0x18)
            fc_min = struct.unpack('<I', data[0x18:0x1C])[0]
            
            # ccpText: Main document text character count (offset 0x4C)
            ccp_text = struct.unpack('<I', data[0x4C:0x50])[0]
            
            # Calculate text area based on encoding
            text_start = fc_min
            if self._encoding_info and self._encoding_info.bytes_per_char == 1:
                # Single-byte encoding (CP1252, etc.)
                text_end = fc_min + ccp_text
            else:
                # UTF-16LE (2 bytes per char) - default
                text_end = fc_min + (ccp_text * 2)
            
            # Sanity checks
            if text_start < 0x200:  # FIB is at least 512 bytes
                text_start = 0x200  # Start after FIB
            if text_end > len(data):
                text_end = len(data)
            
            return text_start, text_end
            
        except (struct.error, ValueError):
            return 0, len(data)
    
    def _find_cell_markers_in_text_area(
        self, 
        data: bytes, 
        text_start: int, 
        text_end: int
    ) -> List[CellMarkerInfo]:
        """Find 0x07 cell markers in the text area only.
        
        Encoding-aware marker detection:
        - UTF-16LE: Look for 0x07 0x00 pattern (2 bytes per char)
        - Single-byte: Look for 0x07 pattern (1 byte per char)
        
        Args:
            data: Raw byte data
            text_start: Start offset of text area
            text_end: End offset of text area
            
        Returns:
            List of CellMarkerInfo objects
        """
        markers = []
        i = text_start
        
        # Determine if using single-byte or double-byte encoding
        is_single_byte = (
            self._encoding_info and 
            self._encoding_info.encoding_type == OLEEncodingType.SINGLE_BYTE
        )
        
        if is_single_byte:
            # Single-byte encoding (CP1252, etc.): 0x07 is cell marker
            while i < min(text_end, len(data)):
                if data[i] == 0x07:
                    # Check if next byte is also 0x07 (row end)
                    is_row_end = (
                        i + 1 < len(data) and 
                        data[i + 1] == 0x07
                    )
                    
                    markers.append(CellMarkerInfo(
                        offset=i,
                        is_row_end=is_row_end
                    ))
                    
                    # Skip the second 0x07 if row end
                    if is_row_end:
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1  # Move by 1 byte (single-byte encoding)
        else:
            # UTF-16LE: 0x07 0x00 is cell marker
            while i < min(text_end, len(data) - 1):
                if data[i] == 0x07 and data[i + 1] == 0x00:
                    # Check if next UTF-16LE char is also 0x07 (row end)
                    is_row_end = (
                        i + 2 < len(data) - 1 and 
                        data[i + 2] == 0x07 and 
                        data[i + 3] == 0x00
                    )
                    
                    markers.append(CellMarkerInfo(
                        offset=i,
                        is_row_end=is_row_end
                    ))
                    
                    # Skip the second 0x07 0x00 if row end
                    if is_row_end:
                        i += 4
                    else:
                        i += 2
                else:
                    i += 2  # Move by UTF-16LE character (2 bytes)
        
        return markers
    
    def _find_cell_markers(self, data: bytes) -> List[CellMarkerInfo]:
        """Find all 0x07 cell markers in the data.
        
        Note: This method is kept for compatibility but should use
        _find_cell_markers_in_text_area for better accuracy.
        
        Args:
            data: Raw byte data
            
        Returns:
            List of CellMarkerInfo objects
        """
        # Use FIB-based method
        text_start, text_end = self._get_text_area_from_fib(data)
        return self._find_cell_markers_in_text_area(data, text_start, text_end)
    
    def _group_markers_into_regions(
        self, 
        markers: List[CellMarkerInfo], 
        data: bytes
    ) -> List[TableRegion]:
        """Group consecutive markers into table regions.
        
        Markers that are close together are likely part of a table.
        
        Args:
            markers: List of cell markers
            data: Raw byte data
            
        Returns:
            List of TableRegion objects
        """
        if not markers:
            return []
        
        regions = []
        current_group = [markers[0]]
        
        for i in range(1, len(markers)):
            marker = markers[i]
            prev_marker = markers[i - 1]
            gap = marker.offset - prev_marker.offset
            
            if gap <= self._config.max_marker_gap:
                current_group.append(marker)
            else:
                # Process current group
                region = self._analyze_marker_group(current_group, data)
                if region:
                    regions.append(region)
                current_group = [marker]
        
        # Process last group
        if current_group:
            region = self._analyze_marker_group(current_group, data)
            if region:
                regions.append(region)
        
        return regions
    
    def _analyze_marker_group(
        self, 
        markers: List[CellMarkerInfo], 
        data: bytes
    ) -> Optional[TableRegion]:
        """Analyze a group of markers to determine if it's a table.
        
        Args:
            markers: Group of consecutive markers
            data: Raw byte data
            
        Returns:
            TableRegion if valid table, None otherwise
        """
        if len(markers) < self._config.min_cells_for_table:
            return None
        
        # Count explicit row endings (0x07 0x07)
        explicit_row_ends = sum(1 for m in markers if m.is_row_end)
        
        # Estimate rows by gap patterns if no explicit row ends
        estimated_rows = self._estimate_rows_by_gaps(markers) if explicit_row_ends < 2 else explicit_row_ends
        
        row_count = max(explicit_row_ends, estimated_rows, 1)
        
        # Need minimum rows
        if row_count < self._config.min_rows:
            return None
        
        # Calculate columns per row
        total_cells = len(markers)
        avg_cols = total_cells / row_count
        
        if avg_cols < 1.5 or avg_cols > 50:
            return None
        
        # Calculate confidence
        confidence = self._calculate_confidence(markers, data, explicit_row_ends)
        
        return TableRegion(
            start_offset=markers[0].offset,
            end_offset=markers[-1].offset,
            row_count=row_count,
            col_count=int(round(avg_cols)),
            confidence=confidence
        )
    
    def _estimate_rows_by_gaps(self, markers: List[CellMarkerInfo]) -> int:
        """Estimate row count by analyzing gaps between markers.
        
        Large gaps typically indicate row boundaries.
        """
        if len(markers) < 2:
            return 1
        
        gaps = []
        for i in range(1, len(markers)):
            gap = markers[i].offset - markers[i-1].offset
            gaps.append(gap)
        
        if not gaps:
            return 1
        
        avg_gap = sum(gaps) / len(gaps)
        large_gap_threshold = avg_gap * 2
        row_boundaries = sum(1 for gap in gaps if gap > large_gap_threshold)
        
        return row_boundaries + 1
    
    def _calculate_confidence(
        self, 
        markers: List[CellMarkerInfo], 
        data: bytes,
        explicit_row_ends: int
    ) -> float:
        """Calculate confidence score for table detection."""
        confidence = 0.3
        
        # More markers = higher confidence
        if len(markers) >= 6:
            confidence += 0.2
        elif len(markers) >= 4:
            confidence += 0.1
        
        # Explicit row endings boost confidence significantly
        if explicit_row_ends >= 2:
            confidence += 0.3
        
        # Check if content looks like text
        text_like_count = 0
        for i in range(min(5, len(markers) - 1)):
            start = markers[i].offset + 1
            end = markers[i + 1].offset
            if end > start and self._looks_like_text(data[start:end]):
                text_like_count += 1
        
        if text_like_count >= 3:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _looks_like_text(self, segment: bytes) -> bool:
        """Check if segment looks like text content."""
        if len(segment) < 2:
            return False
        
        printable_count = 0
        for i in range(0, min(len(segment) - 1, 100), 2):
            low = segment[i]
            high = segment[i + 1] if i + 1 < len(segment) else 0
            
            # ASCII in UTF-16LE
            if 0x20 <= low <= 0x7E and high == 0x00:
                printable_count += 1
            # Korean
            elif 0xAC <= high <= 0xD7:
                printable_count += 1
        
        return printable_count >= 2
    
    def _calculate_column_widths(
        self,
        content: bytes,
        num_cols: int
    ) -> List[float]:
        """Calculate column widths as percentages from sprmTDefTable.
        
        Args:
            content: Document bytes
            num_cols: Number of columns in table
            
        Returns:
            List of column width percentages
        """
        try:
            parser = OLETablePropertiesParser(content)
            structure = parser.parse_table_structure()
            
            if not structure or not structure.rows:
                # Return equal widths if no structure info
                if num_cols > 0:
                    equal_width = 100.0 / num_cols
                    return [equal_width] * num_cols
                return []
            
            # Find the row with max columns (reference row)
            ref_row = None
            for row in structure.rows:
                if row.col_count == structure.max_cols:
                    ref_row = row
                    break
            
            if not ref_row or not ref_row.col_widths:
                # Return equal widths
                if num_cols > 0:
                    equal_width = 100.0 / num_cols
                    return [equal_width] * num_cols
                return []
            
            # Convert twips to percentages
            total_width = sum(ref_row.col_widths)
            if total_width == 0:
                # Return equal widths
                if num_cols > 0:
                    equal_width = 100.0 / num_cols
                    return [equal_width] * num_cols
                return []
            
            # Calculate percentages
            percentages = []
            for width_twips in ref_row.col_widths[:num_cols]:
                pct = (width_twips / total_width) * 100.0
                percentages.append(round(pct, 1))
            
            # Fill remaining columns if needed
            while len(percentages) < num_cols:
                percentages.append(round(100.0 / num_cols, 1))
            
            return percentages
            
        except Exception as e:
            self.logger.debug(f"Could not calculate column widths: {e}")
            # Return equal widths
            if num_cols > 0:
                equal_width = 100.0 / num_cols
                return [equal_width] * num_cols
            return []
    
    def extract_table_from_region(
        self, 
        content: bytes, 
        region: TableRegion
    ) -> Optional[TableData]:
        """Extract table data from a detected region.
        
        Pass 2: Parse content within the region to build TableData.
        Pass 3: Apply merge info from sprmTDefTable structures.
        
        Args:
            content: Raw bytes from WordDocument stream (full document)
            region: TableRegion identifying the table location
            
        Returns:
            TableData object or None
        """
        try:
            # Parse merge info if not already done
            self._parse_merge_info(content)
            
            # Find markers in the full content's text area
            text_start, text_end = self._get_text_area_from_fib(content)
            markers = self._find_cell_markers_in_text_area(content, text_start, text_end)
            
            # Filter markers within this region (with small padding)
            region_start = region.start_offset - 50
            region_end = region.end_offset + 50
            region_markers = [
                m for m in markers 
                if region_start <= m.offset <= region_end
            ]
            
            if len(region_markers) < self._config.min_cells_for_table:
                return None
            
            # Extract rows using the full content (not sliced)
            rows = self._extract_rows_from_markers_full(region_markers, content)
            
            if len(rows) < self._config.min_rows:
                return None
            
            # Build TableData with merge info from sprmTDefTable
            table_rows = self._build_table_rows_with_merge_info(rows)
            
            if not table_rows:
                return None
            
            max_cols = max(len(row) for row in table_rows) if table_rows else 0
            
            # Calculate column widths as percentages
            col_widths_percent = self._calculate_column_widths(content, max_cols)
            
            return TableData(
                rows=table_rows,
                num_rows=len(table_rows),
                num_cols=max_cols,
                has_header=self._config.include_header_row,
                start_offset=region.start_offset,
                end_offset=region.end_offset,
                source_format="doc",
                col_widths_percent=col_widths_percent,
            )
            
        except Exception as e:
            self.logger.warning(f"Error extracting table from region: {e}")
            return None
    
    def _extract_rows_from_markers_full(
        self, 
        markers: List[CellMarkerInfo], 
        data: bytes
    ) -> List[List[str]]:
        """Extract row contents from markers using full document data.
        
        This is similar to _extract_rows_from_markers but works with the
        full document data rather than a sliced region.
        Handles both UTF-16LE and single-byte encodings.
        """
        has_explicit_row_ends = any(m.is_row_end for m in markers)
        
        # Determine if single-byte encoding
        is_single_byte = (
            self._encoding_info and 
            self._encoding_info.encoding_type == OLEEncodingType.SINGLE_BYTE
        )
        
        if has_explicit_row_ends:
            return self._extract_rows_by_explicit_ends_full(markers, data, is_single_byte)
        else:
            return self._extract_rows_by_gap_analysis(markers, data, is_single_byte)
    
    def _extract_rows_by_explicit_ends_full(
        self,
        markers: List[CellMarkerInfo],
        data: bytes,
        is_single_byte: bool = False
    ) -> List[List[str]]:
        """Extract rows using explicit row endings with full document data.
        
        Encoding-aware extraction:
        - UTF-16LE: Cell marker is 0x07 0x00 (2 bytes), paragraph is 0x0D 0x00
        - Single-byte: Cell marker is 0x07 (1 byte), paragraph is 0x0D
        
        Args:
            markers: List of cell markers
            data: Full document data
            is_single_byte: Whether using single-byte encoding
            
        Returns:
            List of rows, each row is a list of cell text strings
        """
        rows = []
        current_row = []
        
        if not markers:
            return rows
        
        # Find text area
        text_start, _ = self._get_text_area_from_fib(data)
        
        # Find table start: look for the last paragraph marker (0x0D) before first cell
        first_marker = markers[0].offset
        table_start = first_marker  # Default: start at first marker
        
        if is_single_byte:
            # Single-byte: paragraph marker is 0x0D (1 byte)
            # Search backwards for 0x0D
            for j in range(first_marker - 1, max(text_start, first_marker - 100), -1):
                if data[j] == 0x0D:
                    table_start = j + 1  # Start after the paragraph marker
                    break
        else:
            # UTF-16LE: paragraph marker is 0x0D 0x00 (2 bytes)
            for j in range(first_marker - 2, max(text_start, first_marker - 200), -2):
                if j + 1 < len(data) and data[j] == 0x0D and data[j + 1] == 0x00:
                    table_start = j + 2  # Start after the paragraph marker
                    break
        
        prev_offset = table_start
        
        for marker in markers:
            cell_text = self._encoder.decode_cell_content(data, prev_offset, marker.offset)
            current_row.append(cell_text)
            
            if marker.is_row_end:
                if current_row:
                    rows.append(current_row)
                current_row = []
                if is_single_byte:
                    prev_offset = marker.offset + 2  # Skip both 0x07 0x07
                else:
                    prev_offset = marker.offset + 4  # Skip both 0x07 0x00 pairs
            else:
                if is_single_byte:
                    prev_offset = marker.offset + 1  # Skip single 0x07
                else:
                    prev_offset = marker.offset + 2  # Skip single 0x07 0x00
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _extract_rows_from_markers(
        self, 
        markers: List[CellMarkerInfo], 
        data: bytes
    ) -> List[List[str]]:
        """Extract row contents from markers.
        
        Uses explicit row endings if available, otherwise gap analysis.
        """
        has_explicit_row_ends = any(m.is_row_end for m in markers)
        
        # Determine if single-byte encoding
        is_single_byte = (
            self._encoding_info and 
            self._encoding_info.encoding_type == OLEEncodingType.SINGLE_BYTE
        )
        
        if has_explicit_row_ends:
            return self._extract_rows_by_explicit_ends(markers, data, is_single_byte)
        else:
            return self._extract_rows_by_gap_analysis(markers, data, is_single_byte)
    
    def _extract_rows_by_explicit_ends(
        self,
        markers: List[CellMarkerInfo],
        data: bytes,
        is_single_byte: bool = False
    ) -> List[List[str]]:
        """Extract rows using explicit row endings.
        
        Encoding-aware:
        - UTF-16LE: Cell marker 0x07 0x00 (2 bytes), Row end 0x07 0x00 0x07 0x00 (4 bytes)
        - Single-byte: Cell marker 0x07 (1 byte), Row end 0x07 0x07 (2 bytes)
        """
        rows = []
        current_row = []
        prev_offset = 0
        
        # Find text area start for proper cell content extraction
        text_start, _ = self._get_text_area_from_fib(data)
        if markers:
            # Find table start (last paragraph marker before first cell)
            first_marker = markers[0].offset
            table_start = first_marker  # Default to first marker
            
            if is_single_byte:
                # Single-byte: paragraph marker is 0x0D (1 byte)
                for j in range(first_marker - 1, max(text_start, first_marker - 100), -1):
                    if data[j] == 0x0D:
                        table_start = j + 1
                        break
            else:
                # UTF-16LE: paragraph marker is 0x0D 0x00 (2 bytes)
                for j in range(first_marker - 2, max(text_start, first_marker - 200), -2):
                    if j + 1 < len(data) and data[j] == 0x0D and data[j + 1] == 0x00:
                        table_start = j + 2
                        break
            prev_offset = table_start
        
        for marker in markers:
            cell_text = self._encoder.decode_cell_content(data, prev_offset, marker.offset)
            current_row.append(cell_text)
            
            if marker.is_row_end:
                if current_row:
                    rows.append(current_row)
                current_row = []
                if is_single_byte:
                    prev_offset = marker.offset + 2  # Skip both 0x07 0x07 (2 bytes)
                else:
                    prev_offset = marker.offset + 4  # Skip both 0x07 0x00 pairs (4 bytes)
            else:
                if is_single_byte:
                    prev_offset = marker.offset + 1  # Skip single 0x07 (1 byte)
                else:
                    prev_offset = marker.offset + 2  # Skip single 0x07 0x00 (2 bytes)
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _extract_rows_by_gap_analysis(
        self,
        markers: List[CellMarkerInfo],
        data: bytes,
        is_single_byte: bool = False
    ) -> List[List[str]]:
        """Extract rows by analyzing gaps between markers.
        
        Args:
            markers: List of cell markers
            data: Document data
            is_single_byte: Whether using single-byte encoding
        """
        if len(markers) < 2:
            return []
        
        # Calculate gaps
        gaps = []
        for i in range(1, len(markers)):
            gap = markers[i].offset - markers[i-1].offset
            gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        row_boundary_threshold = avg_gap * 1.5
        
        rows = []
        current_row = []
        
        # Find table start
        text_start, _ = self._get_text_area_from_fib(data)
        first_marker = markers[0].offset if markers else text_start
        table_start = first_marker
        
        if is_single_byte:
            for j in range(first_marker - 1, max(text_start, first_marker - 100), -1):
                if data[j] == 0x0D:
                    table_start = j + 1
                    break
        else:
            for j in range(first_marker - 2, max(text_start, first_marker - 200), -2):
                if j + 1 < len(data) and data[j] == 0x0D and data[j + 1] == 0x00:
                    table_start = j + 2
                    break
        
        prev_offset = table_start
        
        for i, marker in enumerate(markers):
            cell_text = self._encoder.decode_cell_content(data, prev_offset, marker.offset)
            current_row.append(cell_text)
            
            # Check if next gap is a row boundary
            if i < len(gaps) and gaps[i] > row_boundary_threshold:
                if current_row:
                    rows.append(current_row)
                current_row = []
            
            if is_single_byte:
                prev_offset = marker.offset + 1  # Skip 0x07 (1 byte)
            else:
                prev_offset = marker.offset + 2  # Skip 0x07 0x00 (2 bytes)
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _build_table_rows_with_merge_info(
        self, 
        raw_rows: List[List[str]]
    ) -> List[List[TableCell]]:
        """Build TableCell objects using merge info from sprmTDefTable.
        
        Uses colspan/rowspan information parsed from the DOC binary format.
        
        The mapping between 0x07 marker rows and sprmTDefTable rows:
        - sprmTDefTable defines logical structure (columns, merges)
        - 0x07 markers define physical cells in text stream
        - Empty rows in 0x07 data are rowspan continuations
        - Rows with fewer cells than logical columns have colspan
        
        Args:
            raw_rows: 2D list of cell text content from 0x07 markers
            
        Returns:
            2D list of TableCell objects with accurate merge info
        """
        if not raw_rows:
            return []
        
        # If no merge info available, fall back to heuristic method
        if not self._colspan_map and not self._rowspan_map:
            return self._build_table_rows(raw_rows)
        
        table_rows: List[List[TableCell]] = []
        
        # Map raw row index to sprm row index
        # Empty rows (1 cell, empty content) are rowspan continuations
        sprm_row_idx = 0
        raw_to_sprm_map: Dict[int, int] = {}
        
        for raw_idx, raw_row in enumerate(raw_rows):
            # Check if this is an empty row (rowspan continuation)
            is_empty_row = (
                len(raw_row) == 1 and 
                not raw_row[0].strip()
            )
            
            if is_empty_row:
                # Empty row is part of previous row's rowspan
                # Map to same sprm row as previous non-empty row
                if raw_idx > 0 and (raw_idx - 1) in raw_to_sprm_map:
                    # This is a continuation, skip it in output
                    raw_to_sprm_map[raw_idx] = -1  # Mark as skip
            else:
                raw_to_sprm_map[raw_idx] = sprm_row_idx
                sprm_row_idx += 1
        
        # Build output rows
        for raw_idx, raw_row in enumerate(raw_rows):
            mapped_sprm_idx = raw_to_sprm_map.get(raw_idx, raw_idx)
            
            # Skip empty continuation rows
            if mapped_sprm_idx == -1:
                continue
            
            is_empty_row = (
                len(raw_row) == 1 and 
                not raw_row[0].strip()
            )
            if is_empty_row:
                continue
            
            table_row: List[TableCell] = []
            
            for cell_idx, content in enumerate(raw_row):
                # Get merge info using sprm row index
                colspan = self._colspan_map.get((mapped_sprm_idx, cell_idx), 1)
                rowspan = self._rowspan_map.get((mapped_sprm_idx, cell_idx), 1)
                
                # Determine if this is a header cell
                is_header = (mapped_sprm_idx == 0 and self._config.include_header_row)
                
                cell = TableCell(
                    content=content,
                    row_span=rowspan,
                    col_span=colspan,
                    row_index=len(table_rows),
                    col_index=cell_idx,
                    is_header=is_header,
                )
                table_row.append(cell)
            
            table_rows.append(table_row)
        
        return table_rows
    
    def _build_table_rows(self, raw_rows: List[List[str]]) -> List[List[TableCell]]:
        """Build TableCell objects from raw row data with heuristic merge detection.
        
        DOC 파일은 병합 정보가 바이너리에 있어 직접 파싱하기 어려우므로,
        휴리스틱 기반으로 colspan/rowspan을 추론합니다:
        
        1. 첫 번째 행의 셀 수 = 논리적 열 수
        2. 셀 수가 적은 행은 colspan이 있다고 추론
        3. 빈 셀 또는 이전 행과 동일한 내용의 셀 = rowspan 후보
        
        Args:
            raw_rows: 2D list of cell text content
            
        Returns:
            2D list of TableCell objects with merge info
        """
        if not raw_rows:
            return []
        
        # 논리적 열 수 결정 (첫 번째 행 기준)
        logical_col_count = max(len(row) for row in raw_rows)
        
        # 휴리스틱: 대부분의 행이 같은 셀 수를 가지면 그 수가 논리적 열 수
        cell_counts = [len(row) for row in raw_rows if row]
        if cell_counts:
            from collections import Counter
            count_freq = Counter(cell_counts)
            most_common_count = count_freq.most_common(1)[0][0]
            if most_common_count > 1:
                logical_col_count = most_common_count
        
        # 셀 병합 정보 추론
        merge_info = self._infer_cell_merges(raw_rows, logical_col_count)
        
        # TableCell 객체로 변환
        table_rows = self._create_table_cells_with_merges(raw_rows, merge_info, logical_col_count)
        
        return table_rows
    
    def _infer_cell_merges(
        self, 
        raw_rows: List[List[str]], 
        logical_col_count: int
    ) -> List[List[dict]]:
        """Infer cell merge information heuristically.
        
        Returns a grid where each cell has:
        - colspan: estimated column span
        - rowspan: estimated row span
        - is_merged_away: True if cell is part of another cell's span
        - content: original content
        
        Args:
            raw_rows: 2D list of cell text content
            logical_col_count: Expected number of columns
            
        Returns:
            2D list of merge info dictionaries
        """
        num_rows = len(raw_rows)
        
        # 초기화: 모든 셀을 colspan=1, rowspan=1로 시작
        merge_grid = []
        for row_idx, row in enumerate(raw_rows):
            row_info = []
            actual_cells = len(row)
            
            # 셀 수가 적으면 colspan 추론
            if actual_cells < logical_col_count and actual_cells > 0:
                # 균등하게 colspan 분배
                colspan_per_cell = logical_col_count // actual_cells
                remainder = logical_col_count % actual_cells
                
                col_pos = 0
                for cell_idx, content in enumerate(row):
                    # 남은 칸은 첫 번째 셀들에 분배
                    span = colspan_per_cell + (1 if cell_idx < remainder else 0)
                    row_info.append({
                        'content': content,
                        'colspan': span,
                        'rowspan': 1,
                        'is_merged_away': False,
                        'grid_col': col_pos
                    })
                    col_pos += span
            else:
                # 셀 수가 맞거나 초과하면 colspan=1
                for cell_idx, content in enumerate(row):
                    row_info.append({
                        'content': content,
                        'colspan': 1,
                        'rowspan': 1,
                        'is_merged_away': False,
                        'grid_col': cell_idx
                    })
            
            merge_grid.append(row_info)
        
        # Rowspan 추론: 빈 셀이 있고 위 셀에 내용이 있으면 rowspan
        self._infer_rowspans(merge_grid, num_rows, logical_col_count)
        
        return merge_grid
    
    def _infer_rowspans(
        self, 
        merge_grid: List[List[dict]], 
        num_rows: int,
        logical_col_count: int
    ) -> None:
        """Infer rowspans from empty cells or rows.
        
        DOC 포맷의 제한으로 정확한 병합 정보를 얻을 수 없으므로 휴리스틱 사용:
        
        1. 행이 1-2셀만 있고 모두 빈 경우 → "빈 행"으로 식별
        2. 빈 행 다음에 내용 있는 행이 있으면 → 첫 번째 열만 rowspan
        3. 빈 행이 마지막이면 → 모든 열이 rowspan
        
        Args:
            merge_grid: 병합 정보 그리드 (수정됨)
            num_rows: 총 행 수
            logical_col_count: 논리적 열 수
        """
        # 1단계: 완전 빈 행 식별
        empty_rows = set()
        for row_idx, row_cells in enumerate(merge_grid):
            if len(row_cells) <= 2:  # 1-2개의 셀만 있는 행
                all_empty = all(not cell['content'].strip() for cell in row_cells)
                if all_empty:
                    empty_rows.add(row_idx)
        
        # 2단계: 각 빈 행에 대해 처리
        for empty_row_idx in sorted(empty_rows):
            if empty_row_idx == 0:
                continue
            
            # 이 빈 행의 모든 셀을 병합된 것으로 표시
            for cell in merge_grid[empty_row_idx]:
                cell['is_merged_away'] = True
            
            # 다음 행 확인 (빈 행이 아닌 첫 번째 행)
            next_content_row = None
            for next_row_idx in range(empty_row_idx + 1, num_rows):
                if next_row_idx not in empty_rows:
                    next_content_row = next_row_idx
                    break
            
            # 빈 행 위의 행 찾기 (빈 행이 아닌 마지막 행)
            prev_row_idx = empty_row_idx - 1
            while prev_row_idx in empty_rows and prev_row_idx > 0:
                prev_row_idx -= 1
            
            if prev_row_idx in empty_rows:
                continue
            
            prev_row_cells = merge_grid[prev_row_idx]
            
            # 다음 행이 있고 셀 수가 있으면 → 첫 번째 열만 rowspan
            if next_content_row is not None:
                # 첫 번째 열만 rowspan 적용
                if prev_row_cells and prev_row_cells[0]['rowspan'] == 1:
                    # 연속된 빈 행 수 계산
                    span_count = 0
                    check_row = prev_row_idx + 1
                    while check_row in empty_rows and check_row < num_rows:
                        span_count += 1
                        check_row += 1
                    if span_count > 0:
                        # rowspan = 빈 행 수 + 1 (자기 자신 포함)
                        prev_row_cells[0]['rowspan'] = span_count + 1
            else:
                # 다음 행이 없으면 (마지막 빈 행) → 위 행의 모든 셀에 rowspan
                for cell in prev_row_cells:
                    if cell['rowspan'] == 1:
                        # 연속된 빈 행 수 계산
                        span_count = 0
                        check_row = prev_row_idx + 1
                        while check_row in empty_rows and check_row < num_rows:
                            span_count += 1
                            check_row += 1
                        if span_count > 0:
                            cell['rowspan'] = span_count + 1
    
    def _create_table_cells_with_merges(
        self,
        raw_rows: List[List[str]],
        merge_info: List[List[dict]],
        logical_col_count: int
    ) -> List[List[TableCell]]:
        """Create TableCell objects with merge information.
        
        Args:
            raw_rows: Original raw row data
            merge_info: Merge information grid
            logical_col_count: Expected column count
            
        Returns:
            2D list of TableCell objects
        """
        table_rows = []
        
        for row_idx, row_info in enumerate(merge_info):
            table_row = []
            
            for cell_info in row_info:
                if cell_info['is_merged_away']:
                    # 병합된 셀은 건너뜀 (HTML에서 렌더링하지 않음)
                    continue
                
                cell = TableCell(
                    content=cell_info['content'],
                    row_span=cell_info['rowspan'],
                    col_span=cell_info['colspan'],
                    row_index=row_idx,
                    col_index=cell_info['grid_col'],
                    is_header=(row_idx == 0 and self._config.include_header_row)
                )
                table_row.append(cell)
            
            # 행이 비어있어도 추가 (빈 행 유지)
            table_rows.append(table_row)
        
        return table_rows
    
    def extract_content_with_tables(self, content: bytes) -> str:
        """Extract full content with tables converted to HTML.
        
        Main method for DOC processing. Detects tables and extracts
        them as HTML while preserving non-table content as plain text.
        
        Note: This method delegates to OLETableProcessor for actual processing.
        Kept here for backward compatibility.
        
        Args:
            content: Raw bytes from WordDocument stream
            
        Returns:
            Extracted text with tables in HTML format
        """
        from contextifier.core.processor.ole_helper.ole_table_processor import (
            OLETableProcessor,
        )
        processor = OLETableProcessor()
        return processor.extract_content_with_tables(content)


# Backward compatibility aliases
DOCTableExtractor = OLETableExtractor
DOCTableExtractorConfig = OLETableExtractorConfig


def detect_tables_in_word_stream(data: bytes) -> List[TableRegion]:
    """Convenience function to detect tables in WordDocument stream.
    
    Args:
        data: Raw bytes from WordDocument stream
        
    Returns:
        List of TableRegion objects
    """
    extractor = OLETableExtractor()
    return extractor.detect_table_regions(data)


def extract_tables_from_word_stream(
    data: bytes, 
    config: Optional[OLETableExtractorConfig] = None
) -> List[TableData]:
    """Convenience function to extract tables from WordDocument stream.
    
    Args:
        data: Raw bytes from WordDocument stream
        config: Optional extraction config
        
    Returns:
        List of TableData objects
    """
    extractor = OLETableExtractor(config)
    return extractor.extract_tables(data)


def extract_content_with_tables_from_word_stream(
    data: bytes,
    config: Optional[OLETableExtractorConfig] = None
) -> str:
    """Convenience function to extract content with tables as HTML.
    
    Note: This function now delegates to OLETableProcessor.
    
    Args:
        data: Raw bytes from WordDocument stream
        config: Optional extraction config
        
    Returns:
        Text content with tables in HTML format
    """
    from contextifier.core.processor.ole_helper.ole_table_processor import (
        OLETableProcessor,
    )
    processor = OLETableProcessor()
    return processor.extract_content_with_tables(data)
