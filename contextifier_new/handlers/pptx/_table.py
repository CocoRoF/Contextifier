"""
Table extraction for PPTX slides.

Handles python-pptx Table objects with merge detection
(``is_merge_origin`` / ``is_spanned`` + XML fallbacks).

Public API:
- ``is_simple_table(table)``  — True if table is a layout table (1-row or 1-col)
- ``extract_table(table)``    → ``TableData``
- ``extract_simple_text(table)`` → plain text for simple/layout tables
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier_new.types import TableCell, TableData

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public helpers
# ═══════════════════════════════════════════════════════════════════════════════

def is_simple_table(table: Any) -> bool:
    """
    Decide whether a table is a "simple" layout table that should
    be rendered as plain text rather than HTML.

    Simple = 1-row or 1-column table (often used as text boxes).
    """
    try:
        num_rows = len(table.rows)
        num_cols = len(table.columns)
        return num_rows == 1 or num_cols == 1
    except Exception:
        return False


def extract_simple_text(table: Any) -> str:
    """
    Extract text from a simple/layout table as plain text.

    Rows are separated by newlines, cells within a row by spaces.
    """
    try:
        lines: list[str] = []
        for row in table.rows:
            parts: list[str] = []
            for cell in row.cells:
                text = cell.text.strip() if cell.text else ""
                if text:
                    parts.append(text)
            if parts:
                lines.append(" ".join(parts))
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""


def extract_table(table: Any) -> TableData:
    """
    Extract a full table with merge detection → ``TableData``.

    Handles rowspan/colspan via python-pptx's built-in merge
    attributes (``is_merge_origin``, ``is_spanned``, ``span_height``,
    ``span_width``) with XML-level fallbacks.
    """
    try:
        num_rows = len(table.rows)
        num_cols = len(table.columns)
    except Exception:
        return TableData()

    if num_rows == 0 or num_cols == 0:
        return TableData()

    # Phase 1: Collect merge info into a 2-D grid
    # Each cell is either a dict (origin) or "skip" (spanned)
    grid: List[List[Any]] = [[None] * num_cols for _ in range(num_rows)]

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if grid[row_idx][col_idx] == "skip":
                continue

            cell = table.cell(row_idx, col_idx)
            merge = _get_merge_info(cell, table, row_idx, col_idx, num_rows, num_cols)

            rowspan = merge["rowspan"]
            colspan = merge["colspan"]

            # Mark spanned cells
            for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                for c in range(col_idx, min(col_idx + colspan, num_cols)):
                    if r == row_idx and c == col_idx:
                        grid[r][c] = {
                            "rowspan": rowspan,
                            "colspan": colspan,
                            "text": cell.text.strip() if cell.text else "",
                        }
                    else:
                        grid[r][c] = "skip"

    # Phase 2: Build TableData rows
    rows: List[List[TableCell]] = []
    for row_idx in range(num_rows):
        row_cells: List[TableCell] = []
        for col_idx in range(num_cols):
            info = grid[row_idx][col_idx]
            if info == "skip":
                continue
            if info is None:
                # Fallback — shouldn't happen normally
                cell = table.cell(row_idx, col_idx)
                info = {
                    "rowspan": 1,
                    "colspan": 1,
                    "text": cell.text.strip() if cell.text else "",
                }
            row_cells.append(
                TableCell(
                    content=info["text"],
                    row_span=info["rowspan"],
                    col_span=info["colspan"],
                    is_header=(row_idx == 0),
                    row_index=row_idx,
                    col_index=col_idx,
                )
            )
        rows.append(row_cells)

    return TableData(
        rows=rows,
        num_rows=num_rows,
        num_cols=num_cols,
        has_header=True,  # PPT convention: first row is header
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Merge detection
# ═══════════════════════════════════════════════════════════════════════════════

def _get_merge_info(
    cell: Any,
    table: Any,
    row_idx: int,
    col_idx: int,
    num_rows: int,
    num_cols: int,
) -> Dict[str, int]:
    """
    Determine rowspan/colspan for a cell.

    Strategy (in priority order):
    1. python-pptx built-in attributes (``is_merge_origin``, ``span_*``)
    2. XML attributes (``gridSpan``, ``rowSpan``)
    3. Cell-reference comparison (same ``_tc`` object → merged)
    """
    rowspan = 1
    colspan = 1

    try:
        # Method 1: Built-in python-pptx merge API
        if getattr(cell, "is_merge_origin", False):
            if hasattr(cell, "span_height"):
                rowspan = cell.span_height
            if hasattr(cell, "span_width"):
                colspan = cell.span_width
            return {"rowspan": rowspan, "colspan": colspan}

        # Spanned (already covered by another origin)
        if getattr(cell, "is_spanned", False):
            return {"rowspan": 0, "colspan": 0}

        # Method 2: XML attributes
        tc = cell._tc
        gs = tc.get("gridSpan")
        if gs:
            colspan = int(gs)
        rs = tc.get("rowSpan")
        if rs:
            rowspan = int(rs)

        # Method 3: Reference comparison fallback
        if colspan == 1:
            colspan = _detect_colspan_by_ref(table, row_idx, col_idx, num_cols)
        if rowspan == 1:
            rowspan = _detect_rowspan_by_ref(table, row_idx, col_idx, num_rows)

    except Exception as exc:
        logger.debug("Error getting merge info at [%d,%d]: %s", row_idx, col_idx, exc)

    return {"rowspan": rowspan, "colspan": colspan}


def _detect_colspan_by_ref(
    table: Any, row_idx: int, col_idx: int, num_cols: int
) -> int:
    """Detect colspan by comparing cell internal ``_tc`` references."""
    colspan = 1
    try:
        current = table.cell(row_idx, col_idx)._tc
        for c in range(col_idx + 1, num_cols):
            if table.cell(row_idx, c)._tc is current:
                colspan += 1
            else:
                break
    except Exception:
        pass
    return colspan


def _detect_rowspan_by_ref(
    table: Any, row_idx: int, col_idx: int, num_rows: int
) -> int:
    """Detect rowspan by comparing cell internal ``_tc`` references."""
    rowspan = 1
    try:
        current = table.cell(row_idx, col_idx)._tc
        for r in range(row_idx + 1, num_rows):
            if table.cell(r, col_idx)._tc is current:
                rowspan += 1
            else:
                break
    except Exception:
        pass
    return rowspan


__all__ = ["is_simple_table", "extract_simple_text", "extract_table"]
