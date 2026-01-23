# chunking_helper/constants.py
"""
Chunking Module Constants - Definition of constants, patterns, and dataclasses for chunking

This module defines all constants and data structures used throughout the chunking system.
"""
import logging
from dataclasses import dataclass
from typing import List
from langchain_text_splitters import Language

logger = logging.getLogger("document-processor")


# ============================================================================
# Code Language Mapping
# ============================================================================

LANGCHAIN_CODE_LANGUAGE_MAP = {
    'py': Language.PYTHON, 'js': Language.JS, 'ts': Language.TS,
    'java': Language.JAVA, 'cpp': Language.CPP, 'c': Language.CPP,
    'cs': Language.CSHARP, 'go': Language.GO, 'rs': Language.RUST,
    'php': Language.PHP, 'rb': Language.RUBY, 'swift': Language.SWIFT,
    'kt': Language.KOTLIN, 'scala': Language.SCALA,
    'html': Language.HTML, 'jsx': Language.JS, 'tsx': Language.TS,
}


# ============================================================================
# Protected Region Patterns (Blocks that should not be split during chunking)
# ============================================================================

# HTML table - Protect all <table> tags (regardless of attributes)
HTML_TABLE_PATTERN = r'<table[^>]*>.*?</table>'

# Chart block - Always protected (cannot be chunked under any condition)
# Default format: [chart]...[/chart] - can be customized via ChartProcessor
CHART_BLOCK_PATTERN = r'\[chart\].*?\[/chart\]'

# Textbox block - Always protected (cannot be chunked under any condition)
TEXTBOX_BLOCK_PATTERN = r'\[textbox\].*?\[/textbox\]'

# Image tag - Always protected (cannot be chunked under any condition)
# Format: [image:path], [Image: {path}], [image : path] etc. (case-insensitive, whitespace allowed, {} wrapping allowed)
IMAGE_TAG_PATTERN = r'\[(?i:image)\s*:\s*\{?[^\]\}]+\}?\]'

# Page/Slide/Sheet tag patterns - Always protected (NEVER overlap)
# Default formats from PageTagProcessor
PAGE_TAG_PATTERN = r'\[Page Number:\s*\d+\]'
SLIDE_TAG_PATTERN = r'\[Slide Number:\s*\d+\]'
SHEET_TAG_PATTERN = r'\[Sheet:\s*[^\]]+\]'

# OCR variants of page/slide tags
PAGE_TAG_OCR_PATTERN = r'\[Page Number:\s*\d+\s*\(OCR(?:\+Ref)?\)\]'
SLIDE_TAG_OCR_PATTERN = r'\[Slide Number:\s*\d+\s*\(OCR(?:\+Ref)?\)\]'

# Document metadata block - Always protected (NEVER overlap)
# Default format: <Document-Metadata>...</Document-Metadata> - can be customized via MetadataFormatter
METADATA_BLOCK_PATTERN = r'<Document-Metadata>.*?</Document-Metadata>'

# Data analysis block - Always protected
DATA_ANALYSIS_PATTERN = r'\[(?:Data Analysis|데이터 분석)\].*?\[/(?:Data Analysis|데이터 분석)\]'

# Markdown table patterns
# Complete Markdown table pattern (rows starting with |, including header separator |---|---|)
MARKDOWN_TABLE_PATTERN = r'(?:^|\n)(\|[^\n]+\|\n\|[-:|\s]+\|\n(?:\|[^\n]+\|(?:\n|$))+)'

# Markdown table individual row pattern (for row-level protection)
MARKDOWN_TABLE_ROW_PATTERN = r'\|[^\n]+\|'

# Markdown table header separator pattern (|---|---| or |:---:|---| etc.)
MARKDOWN_TABLE_SEPARATOR_PATTERN = r'^\|[\s\-:]+\|[\s\-:|]*$'

# Markdown table header detection (first row followed by separator)
MARKDOWN_TABLE_HEADER_PATTERN = r'^(\|[^\n]+\|\n)(\|[-:|\s]+\|)'


# ============================================================================
# Table Chunking Related Constants
# ============================================================================

# Table wrapping overhead (table tags, line breaks, etc.)
TABLE_WRAPPER_OVERHEAD = 30  # <table border='1'>\n</table>

# Minimum overhead per row (<tr>\n</tr>)
ROW_OVERHEAD = 12

# Overhead per cell (<td></td> or <th></th>)
CELL_OVERHEAD = 10

# Chunk index metadata overhead
CHUNK_INDEX_OVERHEAD = 30  # [Table chunk 1/10]\n

# Tables larger than this are subject to splitting
TABLE_SIZE_THRESHOLD_MULTIPLIER = 1.2  # 1.2x of chunk_size

# Table-based file types (CSV, TSV, Excel)
TABLE_BASED_FILE_TYPES = {'csv', 'tsv', 'xlsx', 'xls'}


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class TableRow:
    """Table row data (HTML or Markdown)"""
    html: str  # Raw content (HTML or Markdown)
    is_header: bool
    cell_count: int
    char_length: int


@dataclass
class ParsedTable:
    """Parsed table information (HTML)"""
    header_rows: List[TableRow]  # Header rows
    data_rows: List[TableRow]    # Data rows
    total_cols: int              # Total columns
    original_html: str           # Original HTML
    header_html: str             # Header HTML (for reuse)
    header_size: int             # Header size (characters)


@dataclass
class ParsedMarkdownTable:
    """Parsed Markdown table information"""
    header_row: str              # Header row (first row with column names)
    separator_row: str           # Separator row (|---|---|)
    data_rows: List[str]         # Data rows
    total_cols: int              # Total columns
    original_text: str           # Original Markdown text
    header_text: str             # Header + separator for reuse
    header_size: int             # Header size (characters)
