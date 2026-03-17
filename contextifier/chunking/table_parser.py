# contextifier_new/chunking/table_parser.py
"""
Table Parser — HTML & Markdown Table Structure Analysis

Parses HTML and Markdown tables into structured representations
(ParsedTable / ParsedMarkdownTable) suitable for row-level chunking.

This module is shared by TableChunkingStrategy and ProtectedChunkingStrategy.

Migrated from old contextifier.chunking.table_parser with:
- Unified API surface
- Type-safe dataclasses from constants.py
- No external dependencies beyond stdlib re
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Tuple

from contextifier.chunking.constants import (
    ParsedTable,
    ParsedMarkdownTable,
    TableRow,
)

logger = logging.getLogger("contextifier.chunking.table_parser")


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Table Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_html_table(html: str) -> ParsedTable:
    """
    Parse an HTML table string into a structured ParsedTable.

    Separates header rows (``<th>`` or ``<thead>``) from data rows,
    computes column count, and preserves original HTML for fallback.

    Args:
        html: Complete ``<table>...</table>`` HTML string.

    Returns:
        ParsedTable with header_rows, data_rows, header_html, etc.
    """
    header_rows: List[TableRow] = []
    data_rows: List[TableRow] = []

    # Extract rows from <thead> section
    thead_match = re.search(r"<thead[^>]*>(.*?)</thead>", html, re.DOTALL | re.IGNORECASE)
    thead_rows_html: List[str] = []
    if thead_match:
        thead_rows_html = re.findall(r"<tr[^>]*>.*?</tr>", thead_match.group(1), re.DOTALL | re.IGNORECASE)

    # Extract ALL <tr> rows from the full table
    all_rows_html = re.findall(r"<tr[^>]*>.*?</tr>", html, re.DOTALL | re.IGNORECASE)

    for row_html in all_rows_html:
        cells = re.findall(r"<t[hd][^>]*>.*?</t[hd]>", row_html, re.DOTALL | re.IGNORECASE)
        cell_count = len(cells)
        char_length = len(row_html)

        has_th = bool(re.search(r"<th[\s>]", row_html, re.IGNORECASE))
        is_in_thead = row_html in thead_rows_html
        is_header = is_in_thead or has_th

        table_row = TableRow(
            html=row_html,
            is_header=is_header,
            cell_count=cell_count,
            char_length=char_length,
        )

        if is_header:
            header_rows.append(table_row)
        else:
            data_rows.append(table_row)

    # Determine total columns (from header or max data row)
    total_cols = 0
    if header_rows:
        total_cols = max(r.cell_count for r in header_rows)
    elif data_rows:
        total_cols = max(r.cell_count for r in data_rows)

    # Build header HTML string
    header_html = ""
    if header_rows:
        header_lines = [r.html for r in header_rows]
        header_html = "\n".join(header_lines)

    header_size = len(header_html) if header_html else 0

    return ParsedTable(
        header_rows=header_rows,
        data_rows=data_rows,
        total_cols=total_cols,
        original_html=html,
        header_html=header_html,
        header_size=header_size,
    )


def extract_cell_spans(row_html: str) -> List[Tuple[int, int]]:
    """
    Extract ``(rowspan, colspan)`` for each cell in a row.

    Returns:
        List of (rowspan, colspan) tuples, one per cell.
    """
    cells = re.findall(r"<t[hd][^>]*>", row_html, re.IGNORECASE)
    spans: List[Tuple[int, int]] = []
    for cell_open in cells:
        rs_match = re.search(r'rowspan\s*=\s*["\']?(\d+)', cell_open, re.IGNORECASE)
        cs_match = re.search(r'colspan\s*=\s*["\']?(\d+)', cell_open, re.IGNORECASE)
        rowspan = int(rs_match.group(1)) if rs_match else 1
        colspan = int(cs_match.group(1)) if cs_match else 1
        spans.append((rowspan, colspan))
    return spans


def has_complex_spans(html: str) -> bool:
    """
    Check if an HTML table has any ``rowspan > 1`` or ``colspan > 1``.
    """
    return bool(
        re.search(r'(?:rowspan|colspan)\s*=\s*["\']?[2-9]', html, re.IGNORECASE)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown Table Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_markdown_table(text: str) -> Optional[ParsedMarkdownTable]:
    """
    Parse a Markdown pipe-table into a structured ParsedMarkdownTable.

    Expected format:
        | Col1 | Col2 |
        |------|------|
        | A    | B    |

    Returns:
        ParsedMarkdownTable or None if the text is not a valid table.
    """
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return None

    # First line = header
    header_row = lines[0].strip()
    if not header_row.startswith("|"):
        return None

    # Second line = separator
    separator_row = lines[1].strip()
    if not re.match(r"^\|[\s\-:]+(\|[\s\-:]+)*\|$", separator_row):
        return None

    # Remaining = data rows
    data_rows: List[str] = []
    for line in lines[2:]:
        stripped = line.strip()
        if stripped and stripped.startswith("|"):
            data_rows.append(stripped)

    # Count columns from separator
    total_cols = separator_row.count("|") - 1

    header_text = f"{header_row}\n{separator_row}"
    header_size = len(header_text)

    return ParsedMarkdownTable(
        header_row=header_row,
        separator_row=separator_row,
        data_rows=data_rows,
        total_cols=total_cols,
        original_text=text,
        header_text=header_text,
        header_size=header_size,
    )


def is_markdown_table(text: str) -> bool:
    """
    Quick check whether text looks like a Markdown pipe-table.
    """
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return False
    has_pipes = any(line.strip().startswith("|") for line in lines)
    has_separator = any("---" in line and "|" in line for line in lines)
    return has_pipes and has_separator


__all__ = [
    "parse_html_table",
    "extract_cell_spans",
    "has_complex_spans",
    "parse_markdown_table",
    "is_markdown_table",
]
