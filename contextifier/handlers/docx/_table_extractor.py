"""
DOCX table extraction — ``<w:tbl>`` → ``TableData``.

Parses OOXML table elements with full support for:
- Column widths (from ``<w:tblGrid>``)
- Row spans (``<w:vMerge>``)
- Column spans (``<w:gridSpan>``)
- Cell text extraction (paragraphs → joined text)
- Header row detection (first row or explicit ``<w:tblHeader>``)

The extraction is "streaming" in the sense that it operates on a
single ``<w:tbl>`` lxml element passed by the ContentExtractor,
rather than reparsing the entire DOCX.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from contextifier.types import TableCell, TableData
from contextifier.handlers.docx._constants import NAMESPACES
from contextifier.handlers.docx._paragraph import (
    iter_block_elements,
    local_name,
    process_paragraph,
)

logger = logging.getLogger(__name__)

# ── Cached qualified names ────────────────────────────────────────────────

_W = NAMESPACES["w"]
_QN_TBL_GRID = f"{{{_W}}}tblGrid"
_QN_GRID_COL = f"{{{_W}}}gridCol"
_QN_TR = f"{{{_W}}}tr"
_QN_TC = f"{{{_W}}}tc"
_QN_TC_PR = f"{{{_W}}}tcPr"
_QN_GRID_SPAN = f"{{{_W}}}gridSpan"
_QN_V_MERGE = f"{{{_W}}}vMerge"
_QN_P = f"{{{_W}}}p"
_QN_R = f"{{{_W}}}r"
_QN_T = f"{{{_W}}}t"
_QN_TBL_HEADER = f"{{{_W}}}tblHeader"
_QN_TR_PR = f"{{{_W}}}trPr"
_QN_W = f"{{{_W}}}w"
_QN_VAL = f"{{{_W}}}val"


def extract_table(
    table_element: Any,
    *,
    resolve_image: Optional[Callable[[Any], str]] = None,
) -> Optional[TableData]:
    """
    Extract a ``TableData`` from a ``<w:tbl>`` lxml element.

    Args:
        table_element: lxml element for ``<w:tbl>``.

    Returns:
        ``TableData`` or ``None`` if the table is empty.
    """
    try:
        # 1. Column widths
        col_widths = _calculate_column_widths(table_element)
        num_cols = (
            len(col_widths) if col_widths else _estimate_column_count(table_element)
        )

        if num_cols == 0:
            return None

        # 2. Parse all rows → raw cells
        raw_rows: List[List[_RawCell]] = []
        for tr_elem in table_element.iterchildren(_QN_TR):
            cells = _parse_row(tr_elem, num_cols, resolve_image)
            if cells:
                raw_rows.append(cells)

        if not raw_rows:
            return None

        num_rows = len(raw_rows)

        # 3. Calculate row spans (vMerge analysis)
        row_spans = _calculate_rowspans(raw_rows, num_rows, num_cols)

        # 4. Build TableData rows
        table_rows: List[List[TableCell]] = []
        for row_idx, raw_row in enumerate(raw_rows):
            row_cells: List[TableCell] = []
            col_idx = 0
            for raw_cell in raw_row:
                # Skip "continue" cells from vMerge
                actual_col = col_idx
                col_span = raw_cell.col_span

                rs = row_spans.get((row_idx, actual_col), 1)

                # If this is a vMerge continuation, still include as empty
                if raw_cell.v_merge_continue:
                    col_idx += col_span
                    continue

                cell = TableCell(
                    nested_table=raw_cell.nested_table,
                    content=raw_cell.text,
                    row_span=rs,
                    col_span=col_span,
                    is_header=(row_idx == 0),
                    row_index=row_idx,
                    col_index=actual_col,
                )
                row_cells.append(cell)
                col_idx += col_span

            if row_cells:
                table_rows.append(row_cells)

        if not table_rows:
            return None

        # 5. Calculate percentage widths
        col_widths_pct: Optional[List[float]] = None
        if col_widths:
            total = sum(col_widths)
            if total > 0:
                col_widths_pct = [round(w / total * 100, 1) for w in col_widths]

        # 6. Detect header
        has_header = _detect_header(raw_rows)

        return TableData(
            rows=table_rows,
            num_rows=len(table_rows),
            num_cols=num_cols,
            has_header=has_header,
            col_widths_percent=col_widths_pct,
        )

    except Exception as exc:
        logger.warning("Failed to extract DOCX table: %s", exc)
        return None


# ── Internal data structures ──────────────────────────────────────────────


class _RawCell:
    """Temporary cell representation during parsing."""

    __slots__ = (
        "text",
        "col_span",
        "v_merge_restart",
        "v_merge_continue",
        "nested_table",
    )

    def __init__(
        self,
        text: str = "",
        col_span: int = 1,
        v_merge_restart: bool = False,
        v_merge_continue: bool = False,
        nested_table: Optional[TableData] = None,
    ) -> None:
        self.text = text
        self.col_span = col_span
        self.v_merge_restart = v_merge_restart
        self.v_merge_continue = v_merge_continue
        self.nested_table = nested_table


# ── Row parsing ───────────────────────────────────────────────────────────


def _parse_row(
    tr_element: Any,
    expected_cols: int,
    resolve_image: Optional[Callable[[Any], str]] = None,
) -> List[_RawCell]:
    """Parse a ``<w:tr>`` into a list of ``_RawCell``."""
    cells: List[_RawCell] = []

    for tc_elem in tr_element.iterchildren(_QN_TC):
        text, nested = _extract_cell_content(tc_elem, resolve_image)
        col_span = 1
        v_merge_restart = False
        v_merge_continue = False

        tc_pr = tc_elem.find(_QN_TC_PR)
        if tc_pr is not None:
            # gridSpan
            grid_span_elem = tc_pr.find(_QN_GRID_SPAN)
            if grid_span_elem is not None:
                try:
                    col_span = int(grid_span_elem.get(_QN_VAL, "1"))
                except (ValueError, TypeError):
                    col_span = 1

            # vMerge
            v_merge_elem = tc_pr.find(_QN_V_MERGE)
            if v_merge_elem is not None:
                val = v_merge_elem.get(_QN_VAL, "")
                if val == "restart":
                    v_merge_restart = True
                else:
                    # Empty val or "continue" means this cell continues a merge
                    v_merge_continue = True

        cells.append(
            _RawCell(
                text=text,
                col_span=col_span,
                v_merge_restart=v_merge_restart,
                v_merge_continue=v_merge_continue,
                nested_table=nested,
            )
        )

    return cells


# ── Cell text extraction ──────────────────────────────────────────────────


def _extract_cell_content(
    tc_element: Any,
    resolve_image: Optional[Callable[[Any], str]] = None,
) -> Tuple[str, Optional[TableData]]:
    """
    Read one ``<w:tc>``: its text, any image it holds, and a nested table.

    Three things used to be lost here. Cell paragraphs were read for ``w:t``
    only, so a cell holding a picture instead of typed text — a pasted
    screenshot, which is how scanned figures usually enter a document — came
    out empty, with no image tag anywhere for OCR to pick up later. Content
    controls inside a cell were skipped along with their text. And a nested
    table, the usual way a form expresses a sub-grid, was dropped entirely
    because only direct ``w:p`` children were visited.

    Args:
        tc_element: The ``<w:tc>`` element.
        resolve_image: Optional callback turning a drawing/pict descriptor
            into an image tag. Without it the cell behaves as text-only.

    Returns:
        ``(text, nested_table)``.
    """
    pieces: List[str] = []
    nested: Optional[TableData] = None

    for block in iter_block_elements(tc_element):
        name = local_name(block)

        if name == "p":
            text, drawings, picts, _ = process_paragraph(block)
            line = text.strip()

            if resolve_image is not None:
                tags: List[str] = []
                for drawing in drawings:
                    tag = resolve_image(drawing)
                    if tag:
                        tags.append(tag)
                for pict in picts:
                    tag = resolve_image(pict)
                    if tag:
                        tags.append(tag)
                if tags:
                    # Keep any caption the cell also carries: an image is
                    # content, not a substitute for the text beside it.
                    line = " ".join(part for part in [line, *tags] if part)

            if line:
                pieces.append(line)

        elif name == "tbl" and nested is None:
            nested = extract_table(block, resolve_image=resolve_image)

    return "\n".join(pieces), nested


# ── Column width calculation ──────────────────────────────────────────────


def _calculate_column_widths(table_element: Any) -> List[int]:
    """
    Calculate column widths from ``<w:tblGrid>/<w:gridCol>``.

    Returns list of widths in twips. Empty list if no grid is defined.
    """
    tbl_grid = table_element.find(_QN_TBL_GRID)
    if tbl_grid is None:
        return []

    widths: List[int] = []
    for grid_col in tbl_grid.iterchildren(_QN_GRID_COL):
        w_val = grid_col.get(_QN_W, "0")
        try:
            widths.append(int(w_val))
        except (ValueError, TypeError):
            widths.append(0)

    return widths


def _estimate_column_count(table_element: Any) -> int:
    """
    Estimate column count from the first row when tblGrid is absent.
    """
    for tr in table_element.iterchildren(_QN_TR):
        count = 0
        for tc in tr.iterchildren(_QN_TC):
            tc_pr = tc.find(_QN_TC_PR)
            if tc_pr is not None:
                gs = tc_pr.find(_QN_GRID_SPAN)
                if gs is not None:
                    try:
                        count += int(gs.get(_QN_VAL, "1"))
                    except (ValueError, TypeError):
                        count += 1
                else:
                    count += 1
            else:
                count += 1
        return count
    return 0


# ── Row span (vMerge) calculation ─────────────────────────────────────────


def _calculate_rowspans(
    raw_rows: List[List[_RawCell]],
    num_rows: int,
    num_cols: int,
) -> Dict[Tuple[int, int], int]:
    """
    Calculate actual row_span values for cells with vMerge.

    Returns a dict of ``(row_idx, col_idx) → row_span`` for cells
    that start a vertical merge (row_span > 1).
    """
    # Build a column-index map for each row
    # (since gridSpan means one _RawCell can cover multiple columns)
    col_map: List[List[Tuple[int, _RawCell]]] = []  # row → [(col_start, cell), ...]

    for raw_row in raw_rows:
        entries: List[Tuple[int, _RawCell]] = []
        col_idx = 0
        for cell in raw_row:
            entries.append((col_idx, cell))
            col_idx += cell.col_span
        col_map.append(entries)

    # For each column, walk down the rows to find merge groups
    rowspans: Dict[Tuple[int, int], int] = {}

    # Track which columns we've already processed at which rows
    for col in range(num_cols):
        row = 0
        while row < num_rows:
            cell = _find_cell_at_col(col_map[row], col)
            if cell is not None and cell.v_merge_restart:
                # Start of a vertical merge — count how many rows it spans
                merge_start = row
                col_start = _find_col_start(col_map[row], col)
                span = 1
                r = row + 1
                while r < num_rows:
                    next_cell = _find_cell_at_col(col_map[r], col)
                    if next_cell is not None and next_cell.v_merge_continue:
                        span += 1
                        r += 1
                    else:
                        break
                if span > 1:
                    rowspans[(merge_start, col_start)] = span
                row = r
            else:
                row += 1

    return rowspans


def _find_cell_at_col(
    entries: List[Tuple[int, _RawCell]], target_col: int
) -> Optional[_RawCell]:
    """Find the _RawCell that covers the given column index."""
    for col_start, cell in entries:
        if col_start <= target_col < col_start + cell.col_span:
            return cell
    return None


def _find_col_start(entries: List[Tuple[int, _RawCell]], target_col: int) -> int:
    """Find the starting column index of the cell covering target_col."""
    for col_start, cell in entries:
        if col_start <= target_col < col_start + cell.col_span:
            return col_start
    return target_col


# ── Header detection ──────────────────────────────────────────────────────


def _detect_header(raw_rows: List[List[_RawCell]]) -> bool:
    """
    Detect if the table has a header row.

    Returns True if at least one row exists (assuming first row is header).
    A more sophisticated check could inspect ``<w:tblHeader/>`` in trPr.
    """
    return len(raw_rows) > 1


__all__ = ["extract_table"]
