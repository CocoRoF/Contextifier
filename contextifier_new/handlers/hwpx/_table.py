# contextifier_new/handlers/hwpx/_table.py
"""
HWPX table parsing utilities.

Converts ``<hp:tbl>`` XML elements into HTML (or plain-text for
simple tables), matching the v1.0 strategy:

- 1×1 table → transparent container (return cell text)
- single-column table → join rows with ``\\n\\n``
- multi-column table → full ``<table>`` HTML with rowspan/colspan
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import xml.etree.ElementTree as ET

from contextifier_new.handlers.hwpx._constants import HWPX_NAMESPACES

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_hwpx_table(
    tbl_elem: ET.Element,
    ns: Optional[Dict[str, str]] = None,
) -> str:
    """
    Parse an ``<hp:tbl>`` element and return HTML or plain text.

    Args:
        tbl_elem: The ``<hp:tbl>`` XML element.
        ns: XML namespace dictionary (defaults to ``HWPX_NAMESPACES``).

    Returns:
        Rendered table content (HTML for multi-column, plain text for
        single-column or 1×1 tables).  Empty string on failure.
    """
    if ns is None:
        ns = HWPX_NAMESPACES

    try:
        total_rows = int(tbl_elem.get("rowCnt", 0))
        total_cols = int(tbl_elem.get("colCnt", 0))

        grid, max_row, max_col = _build_grid(tbl_elem, ns)
        if not grid:
            return ""

        # Adjust dimensions if unspecified in attributes
        if total_rows == 0:
            total_rows = max_row + 1
        if total_cols == 0:
            total_cols = max_col + 1

        # ── 1×1 container table ──────────────────────────────────────
        if total_rows == 1 and total_cols == 1:
            cell = grid.get((0, 0))
            return cell["text"] if cell else ""

        # ── single-column table ──────────────────────────────────────
        if total_cols == 1:
            parts: List[str] = []
            for r in range(total_rows):
                cell = grid.get((r, 0))
                if cell:
                    text = cell["text"].strip()
                    if text:
                        parts.append(text)
            return "\n\n".join(parts)

        # ── multi-column → HTML ──────────────────────────────────────
        return _render_html(grid, total_rows, total_cols)

    except Exception as exc:
        logger.warning("Failed to parse HWPX table: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Grid Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_grid(
    tbl_elem: ET.Element,
    ns: Dict[str, str],
) -> Tuple[Dict[Tuple[int, int], Dict], int, int]:
    """
    Walk ``<hp:tr>/<hp:tc>`` and build a ``(row, col) → info`` dict.

    Each cell info has keys: ``text``, ``rowspan``, ``colspan``.

    Returns:
        (grid dict, max_row, max_col)
    """
    grid: Dict[Tuple[int, int], Dict] = {}
    max_row = -1
    max_col = -1

    for tr in tbl_elem.findall("hp:tr", ns):
        for tc in tr.findall("hp:tc", ns):
            row_addr, col_addr = _parse_cell_position(tc, ns)
            rowspan, colspan = _parse_cell_span(tc, ns)
            text = _extract_cell_text(tc, ns)

            grid[(row_addr, col_addr)] = {
                "text": text,
                "rowspan": rowspan,
                "colspan": colspan,
            }
            max_row = max(max_row, row_addr)
            max_col = max(max_col, col_addr)

    return grid, max_row, max_col


def _parse_cell_position(tc: ET.Element, ns: Dict[str, str]) -> Tuple[int, int]:
    """Return ``(row_addr, col_addr)`` from ``<hp:cellAddr>``."""
    cell_addr = tc.find("hp:cellAddr", ns)
    if cell_addr is None:
        return 0, 0
    try:
        row = int(cell_addr.get("rowAddr", 0))
    except (ValueError, TypeError):
        row = 0
    try:
        col = int(cell_addr.get("colAddr", 0))
    except (ValueError, TypeError):
        col = 0
    return row, col


def _parse_cell_span(tc: ET.Element, ns: Dict[str, str]) -> Tuple[int, int]:
    """Return ``(rowspan, colspan)`` from ``<hp:cellSpan>``."""
    cell_span = tc.find("hp:cellSpan", ns)
    if cell_span is None:
        return 1, 1
    try:
        rowspan = int(cell_span.get("rowSpan", 1))
    except (ValueError, TypeError):
        rowspan = 1
    try:
        colspan = int(cell_span.get("colSpan", 1))
    except (ValueError, TypeError):
        colspan = 1
    return max(1, rowspan), max(1, colspan)


def _extract_cell_text(tc: ET.Element, ns: Dict[str, str]) -> str:
    """
    Extract text content from a ``<hp:tc>`` cell.

    Walks ``<hp:subList>/<hp:p>/<hp:run>/<hp:t>`` collecting text.
    """
    parts: List[str] = []

    sublist = tc.find("hp:subList", ns)
    target = sublist if sublist is not None else tc

    for p in target.findall("hp:p", ns):
        para_parts: List[str] = []
        for run in p.findall("hp:run", ns):
            t = run.find("hp:t", ns)
            if t is not None and t.text:
                para_parts.append(t.text)
        if para_parts:
            parts.append("".join(para_parts))

    return " ".join(parts).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def _render_html(
    grid: Dict[Tuple[int, int], Dict],
    total_rows: int,
    total_cols: int,
) -> str:
    """
    Render the grid as an HTML ``<table>``.

    Respects rowspan / colspan and produces a skip-map so merged cells
    are not duplicated.
    """
    # Build skip-map for merged cells
    skip: Set[Tuple[int, int]] = set()
    for (r, c), info in grid.items():
        for rs in range(info["rowspan"]):
            for cs in range(info["colspan"]):
                if rs == 0 and cs == 0:
                    continue
                skip.add((r + rs, c + cs))

    rows_html: List[str] = []
    for r in range(total_rows):
        cells_html: List[str] = []
        for c in range(total_cols):
            if (r, c) in skip:
                continue
            info = grid.get((r, c))
            if info is None:
                cells_html.append("<td></td>")
                continue

            attrs: List[str] = []
            if info["rowspan"] > 1:
                attrs.append(f'rowspan="{info["rowspan"]}"')
            if info["colspan"] > 1:
                attrs.append(f'colspan="{info["colspan"]}"')

            attr_str = (" " + " ".join(attrs)) if attrs else ""
            text = info["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            cells_html.append(f"<td{attr_str}>{text}</td>")

        if cells_html:
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

    if not rows_html:
        return ""

    return "<table>" + "".join(rows_html) + "</table>"


__all__ = ["parse_hwpx_table"]
