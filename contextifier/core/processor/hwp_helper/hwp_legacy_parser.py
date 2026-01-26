# contextifier/core/processor/hwp_helper/hwp_legacy_parser.py
"""
HWP Legacy Format Parser (HWP 2.0/3.0)

Parser for legacy HWP file formats that are not OLE-based.
HWP 2.0/3.0 files store text in little-endian 2-byte format:
- ASCII characters: low byte is ASCII, high byte is 0x00
- Korean characters: Johab encoding with bytes swapped (little-endian)

File Structure (HWP 2.0/3.0):
- Header: Variable (starts with "HWP Document File Vx.xx")
- Document sections separated by 'ddddd' markers
- Font table sections (contain 'pX@' pattern - to be skipped)
- Text data blocks (little-endian Johab encoded)
- Table cells are stored as individual sections

Table Processing:
- Uses HWPLegacyTableExtractor for table detection (Pass 1)
- Uses HWPLegacyTableProcessor for HTML conversion (Pass 2)
"""

import re
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

from contextifier.core.processor.hwp_helper.hwp_table_extractor import (
    HWPLegacyTableExtractor,
    HWPLegacyTableExtractorConfig,
    HWPLegacySection,
)
from contextifier.core.processor.hwp_helper.hwp_table_processor import (
    HWPLegacyTableProcessor,
    HWPLegacyTableProcessorConfig,
)

logger = logging.getLogger("document-processor")


# HWP signatures
HWP2_SIGNATURE = b'HWP Document File V2.00 '
HWP3_SIGNATURE = b'HWP Document File V3.00 '

# Record separator pattern in HWP 2.0/3.0
RECORD_SEPARATOR = b'ddddd'

# Font table marker pattern (to be skipped)
FONT_TABLE_MARKER = b'pX@'


@dataclass
class HWPSection:
    """Represents a section in HWP 2.0/3.0 file."""
    index: int
    section_type: str  # 'TEXT_CELL', 'FONT_DEF', 'STRUCT_MARKER', 'OTHER'
    text: str
    raw_data: bytes
    length: int


@dataclass
class HWPTable:
    """Represents a detected table in HWP 2.0/3.0."""
    cells: List[str]
    num_cols: int
    num_rows: int
    start_index: int
    end_index: int


@dataclass
class HWP2ParseResult:
    """Result of HWP 2.0/3.0 parsing."""
    text: str = ""
    version: str = ""
    is_supported: bool = True
    error_message: str = ""
    korean_char_count: int = 0
    total_char_count: int = 0
    tables: List[HWPTable] = field(default_factory=list)


@dataclass
class HWP2Config:
    """Configuration for HWP 2.0/3.0 parser."""
    min_korean_chars: int = 1
    skip_formatting_markers: bool = True
    clean_output: bool = True
    detect_tables: bool = True  # Enable table detection


class HWP2Parser:
    """
    Parser for HWP 2.0/3.0 legacy format files.
    
    HWP 2.0/3.0 stores text in little-endian 2-byte format:
    - ASCII: [char_code, 0x00]
    - Korean (Johab): [low_byte, high_byte] (swapped from standard Johab)
    
    Text sections are separated by 'ddddd' markers.
    Font definition sections contain 'pX@' and should be skipped.
    
    Table Processing:
    - Uses HWPLegacyTableExtractor for table detection (Pass 1)
    - Uses HWPLegacyTableProcessor for HTML conversion (Pass 2)
    """
    
    def __init__(self, config: Optional[HWP2Config] = None):
        """
        Initialize HWP 2.0/3.0 parser.
        
        Args:
            config: Parser configuration
        """
        self.config = config or HWP2Config()
        self.logger = logger
        
        # Initialize table extractor and processor
        self._table_extractor = HWPLegacyTableExtractor(
            HWPLegacyTableExtractorConfig(
                min_cells_for_table=3,
                detect_header_patterns=True,
            )
        )
        self._table_processor = HWPLegacyTableProcessor(
            HWPLegacyTableProcessorConfig()
        )
    
    @property
    def table_extractor(self) -> HWPLegacyTableExtractor:
        """Get the table extractor instance."""
        return self._table_extractor
    
    @property
    def table_processor(self) -> HWPLegacyTableProcessor:
        """Get the table processor instance."""
        return self._table_processor
    
    def detect_version(self, data: bytes) -> Tuple[str, bool]:
        """
        Detect HWP legacy file version.
        
        Args:
            data: Raw file data
            
        Returns:
            Tuple of (version string, is_supported)
        """
        if len(data) < 30:
            return ("Unknown", False)
        
        header = data[:32]
        
        # Check for HWP 2.0
        if header.startswith(HWP2_SIGNATURE):
            return ("2.0", True)
        
        # Check for HWP 3.0 - also supported now
        if header.startswith(HWP3_SIGNATURE):
            return ("3.0", True)
        
        # Generic HWP document file
        if b'HWP Document File' in header:
            try:
                version_match = re.search(rb'V(\d+\.\d+)', header)
                if version_match:
                    version = version_match.group(1).decode('ascii')
                    # Support both 2.x and 3.x
                    if version.startswith('2.') or version.startswith('3.'):
                        return (version, True)
            except:
                pass
            return ("Unknown Legacy", False)
        
        return ("Unknown", False)
    
    def parse(self, data: bytes) -> HWP2ParseResult:
        """
        Parse HWP 2.0/3.0 file and extract text.
        
        Args:
            data: Raw file data
            
        Returns:
            HWP2ParseResult with extracted text
        """
        result = HWP2ParseResult()
        
        # Detect version
        version, is_supported = self.detect_version(data)
        result.version = version
        
        if not is_supported:
            result.is_supported = False
            result.error_message = f"HWP {version} format is not supported"
            return result
        
        try:
            # Parse all sections first
            sections = self._parse_all_sections(data)
            
            # Detect tables if enabled
            tables = []
            if self.config.detect_tables:
                tables = self._detect_tables(sections)
                result.tables = tables
            
            # Extract text with table HTML conversion
            text = self._build_document_text(sections, tables)
            
            # Clean up the text
            if self.config.clean_output:
                text = self._clean_text(text)
            
            result.text = text
            result.korean_char_count = sum(1 for c in text if '가' <= c <= '힣')
            result.total_char_count = len(text)
            
            # Check if we got enough Korean content
            if result.korean_char_count < self.config.min_korean_chars:
                self.logger.warning(
                    f"HWP {version}: Only {result.korean_char_count} Korean chars extracted"
                )
            
        except Exception as e:
            result.is_supported = False
            result.error_message = f"Error parsing HWP {version}: {str(e)}"
            self.logger.error(result.error_message)
        
        return result
    
    def _parse_all_sections(self, data: bytes) -> List[HWPSection]:
        """
        Parse all sections from the file.
        
        Args:
            data: Raw file data
            
        Returns:
            List of HWPSection objects
        """
        # Find all section separators
        separators = []
        pos = 0
        while True:
            pos = data.find(RECORD_SEPARATOR, pos)
            if pos == -1:
                break
            separators.append(pos)
            pos += len(RECORD_SEPARATOR)
        
        if not separators:
            return []
        
        sections = []
        
        for idx in range(len(separators) - 1):
            start = separators[idx] + len(RECORD_SEPARATOR)
            end = separators[idx + 1]
            chunk = data[start:end]
            
            # Classify section type
            section_type = self._classify_section(chunk)
            
            # Get text content
            text = ""
            if section_type not in ('FONT_DEF', 'STRUCT_MARKER'):
                offset = 0
                while offset < min(20, len(chunk)) and chunk[offset] == 0:
                    offset += 1
                text = self._decode_hwp2_text(chunk[offset:]).strip()
                
                # Clean font patterns from text
                text = re.sub(r"#'\*\.[^\n]*", '', text)
                text = re.sub(r'\]?[a-z]{4,10}uy', '', text)
                text = text.strip()
            
            sections.append(HWPSection(
                index=idx,
                section_type=section_type,
                text=text,
                raw_data=chunk,
                length=len(chunk)
            ))
        
        return sections
    
    def _classify_section(self, chunk: bytes) -> str:
        """
        Classify a section by its content type.
        
        Args:
            chunk: Raw section data
            
        Returns:
            Section type string
        """
        if FONT_TABLE_MARKER in chunk[:30]:
            return 'FONT_DEF'
        elif chunk.count(b'\x01') > 5 and len(chunk) < 70:
            return 'STRUCT_MARKER'
        elif b'\x0d\x00' in chunk and len(chunk) > 25:
            return 'TEXT_CELL'
        else:
            return 'OTHER'
    
    def _detect_tables(self, sections: List[HWPSection]) -> List[HWPTable]:
        """
        Detect tables from sections using HWPLegacyTableExtractor.
        
        Delegates to the table extractor for 2-pass table detection:
        - Pass 1: Detect table regions
        - Pass 2: Extract table structure
        
        Args:
            sections: List of parsed sections
            
        Returns:
            List of detected HWPTable objects
        """
        tables = []
        
        # Use table extractor for detection (Pass 1)
        regions = self._table_extractor.detect_table_regions(sections)
        
        for region in regions:
            # Extract table data (Pass 2)
            table_data = self._table_extractor.extract_table_from_region(sections, region)
            
            if table_data and table_data.num_rows >= 1 and table_data.num_cols >= 2:
                # Convert TableData to HWPTable for backward compatibility
                cells = []
                for row in table_data.rows:
                    for cell in row:
                        cells.append(cell.content)
                
                table = HWPTable(
                    cells=cells,
                    num_cols=table_data.num_cols,
                    num_rows=table_data.num_rows,
                    start_index=region.start_offset,
                    end_index=region.end_offset,
                )
                tables.append(table)
        
        return tables
    
    def _build_document_text(self, sections: List[HWPSection], tables: List[HWPTable]) -> str:
        """
        Build the final document text with tables converted to HTML.
        
        Uses HWPLegacyTableProcessor for table HTML rendering.
        
        Args:
            sections: All parsed sections
            tables: Detected tables
            
        Returns:
            Document text with HTML tables
        """
        # Create a set of section indices that belong to tables
        table_indices = set()
        for table in tables:
            for i in range(table.start_index, table.end_index + 1):
                table_indices.add(i)
        
        result_parts = []
        tables_added = set()
        
        for section in sections:
            # Skip font definitions and structure markers
            if section.section_type in ('FONT_DEF', 'STRUCT_MARKER'):
                continue
            
            # Skip empty sections
            if not section.text:
                continue
            
            # Check if this section belongs to a table
            if section.index in table_indices:
                # Find which table this belongs to
                for table in tables:
                    if (table.start_index <= section.index <= table.end_index 
                        and table.start_index not in tables_added):
                        # Add table HTML using table processor
                        table_html = self._render_table_html(table)
                        result_parts.append(table_html)
                        tables_added.add(table.start_index)
                        break
            else:
                # Regular text
                result_parts.append(section.text)
        
        return '\n'.join(result_parts)
    
    def _render_table_html(self, table: HWPTable) -> str:
        """
        Render a table as HTML using HWPLegacyTableProcessor.
        
        Args:
            table: HWPTable object
            
        Returns:
            HTML table string
        """
        # Convert HWPTable to TableData format
        from contextifier.core.functions.table_extractor import TableCell, TableData
        
        cells = table.cells
        num_cols = table.num_cols
        
        rows = []
        for row_idx in range(table.num_rows):
            row_cells = []
            for col_idx in range(num_cols):
                cell_idx = row_idx * num_cols + col_idx
                if cell_idx < len(cells):
                    cell = TableCell(
                        content=cells[cell_idx],
                        row_span=1,
                        col_span=1,
                        is_header=(row_idx == 0),
                    )
                else:
                    cell = TableCell(content="")
                row_cells.append(cell)
            rows.append(row_cells)
        
        table_data = TableData(
            rows=rows,
            num_rows=table.num_rows,
            num_cols=num_cols,
            has_header=True,
            metadata={'source': 'hwp_legacy'}
        )
        
        # Use table processor for HTML rendering
        return self._table_processor.format_table_as_html(table_data)
    
    def _decode_hwp2_text(self, chunk: bytes) -> str:
        """
        Decode HWP 2.0/3.0 text from little-endian 2-byte format.
        
        Format:
        - ASCII: [char_code, 0x00]
        - Korean (Johab): [low_byte, high_byte] where high_byte >= 0x84
        
        Args:
            chunk: Raw bytes to decode
            
        Returns:
            Decoded text string
        """
        result = []
        i = 0
        
        while i < len(chunk) - 1:
            low = chunk[i]
            high = chunk[i + 1]
            
            # ASCII character (high byte is 0x00)
            if high == 0x00:
                if 0x20 <= low < 0x7F:
                    result.append(chr(low))
                elif low == 0x0D:
                    result.append('\n')
                elif low == 0x0A:
                    pass  # Skip LF after CR
                elif low == 0x09:
                    result.append(' ')
            
            # Korean character - Johab encoding (bytes are swapped)
            # Johab first byte range: 0x84-0xD3
            elif 0x84 <= high <= 0xD3:
                try:
                    # Swap back to big-endian for Johab decoding
                    char = bytes([high, low]).decode('johab')
                    result.append(char)
                except:
                    pass
            
            # Extended range (special characters, Hanja, etc.)
            elif high >= 0x80:
                try:
                    char = bytes([high, low]).decode('johab')
                    result.append(char)
                except:
                    try:
                        char = bytes([high, low]).decode('cp949')
                        result.append(char)
                    except:
                        pass
            
            i += 2
        
        return ''.join(result)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove font table patterns that got through
        # Pattern: #'*.26:>BFJ...aeimquy or ]aeimquy or similar sequences
        text = re.sub(r"#'\*\.[^\n]*\n?", '', text)
        text = re.sub(r'\]?[a-z]{4,10}uy\n?', '', text)  # Remove ...aeimquy patterns
        text = re.sub(r'pX@\([^)]*\)[^\n]*', '', text)
        text = re.sub(r'pX@[^\n]{0,50}', '', text)
        
        # Remove pattern noise
        text = re.sub(r'[G]{2,}cccc', '', text)
        text = re.sub(r'xi`mHq0uy\^*', '', text)
        text = re.sub(r'NR[^\s]*Y\]', '', text)
        
        # Remove isolated single characters/symbols on lines
        text = re.sub(r'^[a-zA-Z,;:!.&*#()0-9\]\[]{1,5}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Clean up resulting multiple newlines again
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


def parse_hwp_legacy(data: bytes, config: Optional[HWP2Config] = None) -> HWP2ParseResult:
    """
    Parse HWP legacy format file.
    
    Convenience function for parsing HWP 2.0/3.0 files.
    
    Args:
        data: Raw file data
        config: Optional parser configuration
        
    Returns:
        HWP2ParseResult with extracted text
    """
    parser = HWP2Parser(config)
    return parser.parse(data)


def is_hwp_legacy_format(data: bytes) -> bool:
    """
    Check if data is an HWP legacy format file.
    
    Args:
        data: Raw file data
        
    Returns:
        True if this is an HWP legacy format file
    """
    if len(data) < 30:
        return False
    return b'HWP Document File' in data[:30]


def get_hwp_legacy_version(data: bytes) -> str:
    """
    Get the version of an HWP legacy format file.
    
    Args:
        data: Raw file data
        
    Returns:
        Version string or "Unknown"
    """
    parser = HWP2Parser()
    version, _ = parser.detect_version(data)
    return version
