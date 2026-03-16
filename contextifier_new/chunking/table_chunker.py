# contextifier_new/chunking/table_chunker.py
"""
Table Chunker — Row-Level Splitting for HTML & Markdown Tables

Splits large tables exceeding chunk_size into multiple chunks while:
- Preserving header rows in every chunk
- Never cutting in the middle of a row
- Adding chunk index metadata (``[Table Chunk N/M]``)
- Using zero overlap between table chunks (intentional)

Migrated from old contextifier.chunking.table_chunker with:
- Config-aware constants
- Unified HTML + Markdown API
- Proper rowspan adjustment for complex tables
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Tuple

from contextifier_new.chunking.constants import (
    TABLE_WRAPPER_OVERHEAD,
    ROW_OVERHEAD,
    CHUNK_INDEX_OVERHEAD,
    ParsedTable,
    ParsedMarkdownTable,
    TableRow,
)
from contextifier_new.chunking.table_parser import (
    parse_html_table,
    extract_cell_spans,
    has_complex_spans,
    parse_markdown_table,
    is_markdown_table,
)

logger = logging.getLogger("contextifier.chunking.table_chunker")


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Table Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_html_table(
    table_html: str,
    chunk_size: int,
    context_prefix: str = "",
) -> List[str]:
    """
    Split an HTML table that exceeds *chunk_size* into row-level chunks.

    Each chunk:
    - Starts with the original header rows (restored)
    - Contains contiguous data rows fitting in *chunk_size*
    - Is wrapped in ``<table>...</table>``
    - Includes ``[Table Chunk N/M]`` annotation if >1 chunk

    Args:
        table_html: Complete ``<table>...</table>`` string.
        chunk_size: Target maximum characters per chunk.
        context_prefix: Optional prefix (sheet tag, etc.) prepended to each chunk.

    Returns:
        List of chunk strings.
    """
    parsed = parse_html_table(table_html)

    if not parsed.data_rows:
        return [f"{context_prefix}{table_html}".strip()] if table_html else []

    # If entire table fits → return as-is
    total_size = len(table_html) + len(context_prefix)
    if total_size <= chunk_size:
        return [f"{context_prefix}{table_html}".strip()]

    # Complex rowspan handling
    if has_complex_spans(table_html):
        return _split_table_preserving_rowspan(parsed, chunk_size, context_prefix)

    return _split_table_simple(parsed, chunk_size, context_prefix)


def _split_table_simple(
    parsed: ParsedTable,
    chunk_size: int,
    context_prefix: str,
) -> List[str]:
    """Split table by rows without rowspan complications."""
    overhead = len(context_prefix) + TABLE_WRAPPER_OVERHEAD + CHUNK_INDEX_OVERHEAD
    available = chunk_size - overhead - parsed.header_size

    if available <= 0:
        logger.warning("Chunk size too small for table headers; returning single chunk")
        return [f"{context_prefix}{parsed.original_html}".strip()]

    # Partition rows into groups
    groups: List[List[TableRow]] = []
    current_group: List[TableRow] = []
    current_size = 0

    for row in parsed.data_rows:
        if current_group and current_size + row.char_length > available:
            groups.append(current_group)
            current_group = [row]
            current_size = row.char_length
        else:
            current_group.append(row)
            current_size += row.char_length

    if current_group:
        groups.append(current_group)

    total_chunks = len(groups)
    if total_chunks <= 1:
        return [f"{context_prefix}{parsed.original_html}".strip()]

    chunks: List[str] = []
    for idx, group in enumerate(groups, start=1):
        body_rows = "\n".join(r.html for r in group)
        header_part = f"\n{parsed.header_html}" if parsed.header_html else ""
        table_str = f"<table>{header_part}\n{body_rows}\n</table>"
        index_tag = f"[Table Chunk {idx}/{total_chunks}]"
        chunk = f"{context_prefix}{index_tag}\n{table_str}".strip()
        chunks.append(chunk)

    return chunks


def _split_table_preserving_rowspan(
    parsed: ParsedTable,
    chunk_size: int,
    context_prefix: str,
) -> List[str]:
    """
    Split a table with complex rowspans.

    For simplicity and correctness, when a table has active rowspans
    at a split boundary, we adjust the rowspan values down to reflect
    the rows remaining in that chunk.
    """
    groups: List[List[TableRow]] = []
    overhead = len(context_prefix) + TABLE_WRAPPER_OVERHEAD + CHUNK_INDEX_OVERHEAD
    available = chunk_size - overhead - parsed.header_size

    if available <= 0:
        return [f"{context_prefix}{parsed.original_html}".strip()]

    current_group: List[TableRow] = []
    current_size = 0

    for row in parsed.data_rows:
        if current_group and current_size + row.char_length > available:
            groups.append(current_group)
            current_group = [row]
            current_size = row.char_length
        else:
            current_group.append(row)
            current_size += row.char_length

    if current_group:
        groups.append(current_group)

    total_chunks = len(groups)
    if total_chunks <= 1:
        return [f"{context_prefix}{parsed.original_html}".strip()]

    chunks: List[str] = []
    for idx, group in enumerate(groups, start=1):
        adjusted_rows = [_adjust_row_rowspan(r, len(group), i) for i, r in enumerate(group)]
        body_rows = "\n".join(adjusted_rows)
        header_part = f"\n{parsed.header_html}" if parsed.header_html else ""
        table_str = f"<table>{header_part}\n{body_rows}\n</table>"
        index_tag = f"[Table Chunk {idx}/{total_chunks}]"
        chunk = f"{context_prefix}{index_tag}\n{table_str}".strip()
        chunks.append(chunk)

    return chunks


def _adjust_row_rowspan(row: TableRow, group_size: int, row_idx: int) -> str:
    """Reduce rowspan values that would exceed the remaining rows in this chunk."""
    remaining = group_size - row_idx
    html = row.html

    def _clamp_rowspan(match: re.Match) -> str:
        val = int(match.group(1))
        clamped = min(val, remaining)
        if clamped <= 1:
            return ""  # Remove rowspan="1"
        return f'rowspan="{clamped}"'

    return re.sub(r'rowspan\s*=\s*["\']?(\d+)["\']?', _clamp_rowspan, html, flags=re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown Table Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_markdown_table(
    table_text: str,
    chunk_size: int,
    context_prefix: str = "",
) -> List[str]:
    """
    Split a Markdown pipe-table into row-level chunks.

    Each chunk restores the header + separator rows.

    Args:
        table_text: Complete Markdown table string.
        chunk_size: Target maximum characters per chunk.
        context_prefix: Optional prefix prepended to each chunk.

    Returns:
        List of chunk strings.
    """
    parsed = parse_markdown_table(table_text)
    if parsed is None:
        return [f"{context_prefix}{table_text}".strip()] if table_text else []

    if not parsed.data_rows:
        return [f"{context_prefix}{table_text}".strip()]

    total_size = len(table_text) + len(context_prefix)
    if total_size <= chunk_size:
        return [f"{context_prefix}{table_text}".strip()]

    overhead = len(context_prefix) + CHUNK_INDEX_OVERHEAD
    available = chunk_size - overhead - parsed.header_size - 2  # 2 for newlines

    if available <= 0:
        return [f"{context_prefix}{table_text}".strip()]

    groups: List[List[str]] = []
    current_group: List[str] = []
    current_size = 0

    for row in parsed.data_rows:
        row_len = len(row) + 1  # +1 for newline
        if current_group and current_size + row_len > available:
            groups.append(current_group)
            current_group = [row]
            current_size = row_len
        else:
            current_group.append(row)
            current_size += row_len

    if current_group:
        groups.append(current_group)

    total_chunks = len(groups)
    if total_chunks <= 1:
        return [f"{context_prefix}{table_text}".strip()]

    chunks: List[str] = []
    for idx, group in enumerate(groups, start=1):
        data_part = "\n".join(group)
        table_str = f"{parsed.header_text}\n{data_part}"
        index_tag = f"[Table Chunk {idx}/{total_chunks}]"
        chunk = f"{context_prefix}{index_tag}\n{table_str}".strip()
        chunks.append(chunk)

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_large_table(
    table_text: str,
    chunk_size: int,
    context_prefix: str = "",
) -> List[str]:
    """
    Auto-detect table format (HTML or Markdown) and split into chunks.

    Args:
        table_text: Table content (HTML or Markdown).
        chunk_size: Target maximum characters per chunk.
        context_prefix: Optional prefix prepended to each chunk.

    Returns:
        List of chunk strings.
    """
    if is_markdown_table(table_text):
        return chunk_markdown_table(table_text, chunk_size, context_prefix)
    else:
        return chunk_html_table(table_text, chunk_size, context_prefix)


__all__ = [
    "chunk_html_table",
    "chunk_markdown_table",
    "chunk_large_table",
]
