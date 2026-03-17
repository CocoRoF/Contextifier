# contextifier_new/chunking/constants.py
"""
Chunking Module Constants

Defines regex patterns, dataclasses, and thresholds used across the chunking
subsystem. All tag patterns are kept consistent with TagService defaults.

Ported and cleaned from contextifier/chunking/constants.py with a cleaner
structure and explicit type annotations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet, List

if TYPE_CHECKING:
    from contextifier.config import ProcessingConfig


# ============================================================================
# Code Language Mapping
# ============================================================================

# Extension → langchain Language enum name (string-based to avoid hard dep)
CODE_LANGUAGE_MAP: dict[str, str] = {
    "py": "PYTHON",
    "js": "JS",
    "ts": "TS",
    "java": "JAVA",
    "cpp": "CPP",
    "c": "CPP",
    "cs": "CSHARP",
    "go": "GO",
    "rs": "RUST",
    "php": "PHP",
    "rb": "RUBY",
    "swift": "SWIFT",
    "kt": "KOTLIN",
    "scala": "SCALA",
    "html": "HTML",
    "jsx": "JS",
    "tsx": "TS",
}


# ============================================================================
# Protected Region Patterns (blocks that must NEVER be split)
# ============================================================================
#
# Tag-dependent patterns (page, slide, sheet, image, chart, metadata) are
# built at runtime from ProcessingConfig via build_protected_patterns().
# This ensures they stay in sync when users customize tag formats.
#
# Only format-structural patterns that don't depend on TagConfig are
# defined as module-level constants.
# ============================================================================

# HTML table (with any attributes) — config-independent
HTML_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)

# Textbox block — [textbox]...[/textbox] — config-independent (no TagConfig entry)
TEXTBOX_BLOCK_PATTERN = re.compile(r"\[textbox\].*?\[/textbox\]", re.DOTALL | re.IGNORECASE)

# Markdown tables — config-independent
MARKDOWN_TABLE_PATTERN = re.compile(
    r"(?:^|\n)(\|[^\n]+\|\n\|[-:|\s]+\|\n(?:\|[^\n]+\|(?:\n|$))+)"
)
MARKDOWN_TABLE_ROW_PATTERN = re.compile(r"\|[^\n]+\|")
MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[\s\-:]+\|[\s\-:|]*$", re.MULTILINE)
MARKDOWN_TABLE_HEADER_PATTERN = re.compile(r"^(\|[^\n]+\|\n)(\|[-:|\s]+\|)")


# ============================================================================
# Pattern Builder Functions — Config-Driven
# ============================================================================

def _build_block_pattern(
    open_tag: str, close_tag: str, flags: int = re.DOTALL,
) -> re.Pattern[str]:
    """Build regex for block-level tags like [chart]...[/chart]."""
    return re.compile(
        rf"{re.escape(open_tag)}.*?{re.escape(close_tag)}", flags,
    )


def _build_inline_pattern(prefix: str, suffix: str) -> re.Pattern[str]:
    """Build regex for inline tags like [Image:path] or [Sheet: name]."""
    return re.compile(rf"{re.escape(prefix)}.+?{re.escape(suffix)}")


def _build_number_tag_pattern(prefix: str, suffix: str) -> re.Pattern[str]:
    """Build regex for number-based tags with optional OCR annotation."""
    return re.compile(
        rf"{re.escape(prefix)}\d+(?:\s*\(OCR(?:\+Ref)?\))?{re.escape(suffix)}"
    )


def build_protected_patterns(config: "ProcessingConfig") -> list[re.Pattern[str]]:
    """
    Build all protected region patterns from ProcessingConfig.

    Tag-dependent patterns are derived from ``config.tags`` so they stay
    in sync when tag formats are customized.  Config-independent patterns
    (HTML tables, textboxes) are included as module-level constants.

    Args:
        config: Processing configuration with tag settings.

    Returns:
        List of compiled regex patterns in priority order.
    """
    tags = config.tags
    return [
        # Block-level protected regions
        _build_block_pattern(tags.chart_prefix, tags.chart_suffix, re.DOTALL | re.IGNORECASE),
        TEXTBOX_BLOCK_PATTERN,
        HTML_TABLE_PATTERN,
        _build_block_pattern(tags.metadata_prefix, tags.metadata_suffix, re.DOTALL),
        # Inline protected tags
        _build_inline_pattern(tags.image_prefix, tags.image_suffix),
        _build_number_tag_pattern(tags.page_prefix, tags.page_suffix),
        _build_number_tag_pattern(tags.slide_prefix, tags.slide_suffix),
        _build_inline_pattern(tags.sheet_prefix, tags.sheet_suffix),
    ]


def build_image_capture_pattern(config: "ProcessingConfig") -> re.Pattern[str]:
    """
    Build regex for image tags with a capture group for the path.

    Used by chunking strategies that need to identify image tag content.

    Args:
        config: Processing configuration with tag settings.

    Returns:
        Compiled regex with one capture group for the image path.
    """
    tags = config.tags
    return re.compile(
        rf"{re.escape(tags.image_prefix)}\s*(.+?)\s*{re.escape(tags.image_suffix)}"
    )


# ============================================================================
# Table Chunking Thresholds
# ============================================================================

TABLE_WRAPPER_OVERHEAD: int = 30       # <table border='1'>\n</table>
ROW_OVERHEAD: int = 12                 # <tr>\n</tr>
CELL_OVERHEAD: int = 10                # <td></td> or <th></th>
CHUNK_INDEX_OVERHEAD: int = 30         # [Table chunk 1/10]\n
TABLE_SIZE_THRESHOLD_MULTIPLIER: float = 1.2  # 1.2× of chunk_size

# Extensions that are inherently table-based
TABLE_EXTENSIONS: FrozenSet[str] = frozenset({"csv", "tsv", "xlsx", "xls"})


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True)
class TableRow:
    """A single table row (HTML or Markdown)."""

    html: str
    is_header: bool
    cell_count: int
    char_length: int


@dataclass(frozen=True)
class ParsedTable:
    """Parsed HTML table."""

    header_rows: List[TableRow]
    data_rows: List[TableRow]
    total_cols: int
    original_html: str
    header_html: str
    header_size: int


@dataclass(frozen=True)
class ParsedMarkdownTable:
    """Parsed Markdown table."""

    header_row: str
    separator_row: str
    data_rows: List[str]
    total_cols: int
    original_text: str
    header_text: str
    header_size: int


__all__ = [
    # Config-independent patterns
    "HTML_TABLE_PATTERN",
    "TEXTBOX_BLOCK_PATTERN",
    "MARKDOWN_TABLE_PATTERN",
    "MARKDOWN_TABLE_ROW_PATTERN",
    "MARKDOWN_TABLE_SEPARATOR_PATTERN",
    "MARKDOWN_TABLE_HEADER_PATTERN",
    # Config-driven pattern builders
    "build_protected_patterns",
    "build_image_capture_pattern",
    # Thresholds
    "TABLE_WRAPPER_OVERHEAD",
    "ROW_OVERHEAD",
    "CELL_OVERHEAD",
    "CHUNK_INDEX_OVERHEAD",
    "TABLE_SIZE_THRESHOLD_MULTIPLIER",
    "TABLE_EXTENSIONS",
    # Language mapping
    "CODE_LANGUAGE_MAP",
    # Dataclasses
    "TableRow",
    "ParsedTable",
    "ParsedMarkdownTable",
]
