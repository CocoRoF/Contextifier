# contextifier/chunking/table_parser.py
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    thead_match = re.search(
        r"<thead[^>]*>(.*?)</thead>", html, re.DOTALL | re.IGNORECASE
    )
    thead_rows_html: List[str] = []
    if thead_match:
        thead_rows_html = re.findall(
            r"<tr[^>]*>.*?</tr>", thead_match.group(1), re.DOTALL | re.IGNORECASE
        )

    # Extract ALL <tr> rows from the full table
    all_rows_html = re.findall(r"<tr[^>]*>.*?</tr>", html, re.DOTALL | re.IGNORECASE)

    for row_html in all_rows_html:
        cells = re.findall(
            r"<t[hd][^>]*>.*?</t[hd]>", row_html, re.DOTALL | re.IGNORECASE
        )
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


# ═══════════════════════════════════════════════════════════════════════════════
# Row-span carry-over — cells that outlive a chunk boundary
# ═══════════════════════════════════════════════════════════════════════════════

_CELL_PATTERN = re.compile(
    r"<(?P<tag>t[hd])(?P<attrs>[^>]*)>(?P<content>.*?)</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)
_ROWSPAN_ATTR_PATTERN = re.compile(r"\s*rowspan\s*=\s*[\"\']?\d+[\"\']?", re.IGNORECASE)


@dataclass(frozen=True)
class SpanningCell:
    """A cell whose ``rowspan`` carries it into following rows.

    Holds everything needed to write the cell again at the top of a
    continuation chunk: where it sits, how wide it is, and its markup.
    """

    col: int
    rowspan: int
    colspan: int
    tag: str
    attrs: str  # original attributes with rowspan removed
    content: str

    def render(self, remaining_rows: int) -> str:
        """Markup for re-issuing the cell with an adjusted row span."""
        attrs = self.attrs.strip()
        parts = [self.tag]
        if attrs:
            parts.append(attrs)
        if remaining_rows > 1:
            parts.append(f'rowspan="{remaining_rows}"')
        return f"<{' '.join(parts)}>{self.content}</{self.tag}>"


@dataclass(frozen=True)
class RowCell:
    """One cell as written in a row, with its resolved grid position."""

    col: int
    rowspan: int
    colspan: int
    html: str


def extract_row_cells(row_html: str, occupied: Optional[set] = None) -> List[RowCell]:
    """
    Resolve the grid position of every cell written in a row.

    A row that sits under a ``rowspan`` from an earlier row omits the covered
    cell entirely, so the first cell it writes is not necessarily in column 0.
    *occupied* carries the columns already taken by those spans; without it the
    positions are only correct for a row no span reaches into.

    Args:
        row_html: The ``<tr>…</tr>`` markup.
        occupied: Column indices covered by spans from earlier rows.

    Returns:
        The row's cells in written order, each with its visual column.
    """
    taken = occupied or set()
    cells: List[RowCell] = []
    col = 0

    for match in _CELL_PATTERN.finditer(row_html):
        while col in taken:
            col += 1
        attrs = match.group("attrs")
        rowspan = _int_attr(attrs, "rowspan")
        colspan = _int_attr(attrs, "colspan")
        cells.append(
            RowCell(col=col, rowspan=rowspan, colspan=colspan, html=match.group(0))
        )
        col += colspan

    return cells


def _int_attr(attrs: str, name: str) -> int:
    match = re.search(rf'{name}\s*=\s*["\']?(\d+)', attrs, re.IGNORECASE)
    if not match:
        return 1
    try:
        return max(1, int(match.group(1)))
    except ValueError:
        return 1


def compute_carried_cells(
    data_rows: List[TableRow],
) -> List[Dict[int, Tuple[SpanningCell, int]]]:
    """
    For every row, the spanning cells that cover it but are written earlier.

    A chunk that begins at such a row has to write those cells again, or the
    chunk loses their content and every row in it is short a column.

    Args:
        data_rows: The table's data rows, in order.

    Returns:
        One entry per row: ``{column: (cell, rows_still_covered)}`` where the
        count includes the row itself. Empty for a row nothing reaches into.
    """
    carried: List[Dict[int, Tuple[SpanningCell, int]]] = []
    active: Dict[int, Tuple[SpanningCell, int]] = {}

    for row_index, row in enumerate(data_rows):
        if row_index > 0:
            for col in list(active):
                cell, remaining = active[col]
                remaining -= 1
                if remaining <= 0:
                    del active[col]
                else:
                    active[col] = (cell, remaining)

        # Snapshot before this row's own cells join: exactly the spans that
        # started earlier and still reach this row.
        carried.append(dict(active))

        occupied = {
            column
            for col, (cell, _) in active.items()
            for column in range(col, col + cell.colspan)
        }

        for written in extract_row_cells(row.html, occupied):
            if written.rowspan <= 1:
                continue
            match = _CELL_PATTERN.match(written.html)
            if match is None:  # pragma: no cover — html came from the matcher
                continue
            cell = SpanningCell(
                col=written.col,
                rowspan=written.rowspan,
                colspan=written.colspan,
                tag=match.group("tag").lower(),
                attrs=_ROWSPAN_ATTR_PATTERN.sub("", match.group("attrs")),
                content=match.group("content"),
            )
            active[written.col] = (cell, written.rowspan)

    return carried


def reissue_carried_cells(
    row_html: str,
    carried: Dict[int, Tuple[SpanningCell, int]],
) -> str:
    """
    Write the carried spanning cells back into a row that opens a chunk.

    Each cell is placed at the column it occupies, so a category column stays
    on the left and a mid-table span stays in the middle. The row's own cells
    keep their order and markup.

    Args:
        row_html: Markup of the row that begins the chunk.
        carried: ``{column: (cell, rows_still_covered)}`` from
            :func:`compute_carried_cells`.

    Returns:
        The row with the carried cells restored.
    """
    if not carried:
        return row_html

    occupied = {
        column
        for col, (cell, _) in carried.items()
        for column in range(col, col + cell.colspan)
    }
    own_cells = extract_row_cells(row_html, occupied)

    pieces: List[str] = []
    pending = sorted(carried.items())
    pending_index = 0

    def flush_up_to(limit: Optional[int]) -> None:
        nonlocal pending_index
        while pending_index < len(pending):
            col, (cell, remaining) = pending[pending_index]
            if limit is not None and col > limit:
                return
            pieces.append(cell.render(remaining))
            pending_index += 1

    for written in own_cells:
        flush_up_to(written.col)
        pieces.append(written.html)
    flush_up_to(None)

    open_match = re.match(r"\s*<tr[^>]*>", row_html, re.IGNORECASE)
    open_tag = open_match.group(0).strip() if open_match else "<tr>"
    return f"{open_tag}{''.join(pieces)}</tr>"


__all__ = [
    "parse_html_table",
    "SpanningCell",
    "RowCell",
    "extract_row_cells",
    "compute_carried_cells",
    "reissue_carried_cells",
    "extract_cell_spans",
    "has_complex_spans",
    "parse_markdown_table",
    "is_markdown_table",
]
