# contextifier/handlers/hwp/_table.py
"""
HWP 5.0 table parsing utilities.

Given a CTRL_HEADER record whose ctrl_id is ``tbl `` (table), this
module builds a cell grid from the child LIST_HEADER records and
renders it to either plain text (for 1-column tables) or HTML (for
multi-column tables).

Public API:
    parse_table(ctrl_header, traverse_cb, ...) -> str
    render_table_html(grid, rows, cols)        -> str
"""

from __future__ import annotations

import struct
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from contextifier.handlers.hwp._constants import (
    HWPTAG_LIST_HEADER,
    HWPTAG_TABLE,
)
from contextifier.handlers.hwp._record import HwpRecord

logger = logging.getLogger(__name__)


def parse_table(
    ctrl_header: HwpRecord,
    traverse_cb: Callable[..., str],
    ole: Any = None,
    bin_data_map: Optional[Dict] = None,
    processed_images: Optional[Set[str]] = None,
) -> str:
    """
    Parse a HWP table control and render it.

    - 1×1 table → return cell text (container wrapper)
    - 1-column, multi-row → newline-separated cell texts
    - multi-column → HTML ``<table>``

    Args:
        ctrl_header: The CTRL_HEADER record that wraps the table.
        traverse_cb: Callback to extract text from child records.
        ole: Open OLE container (for image extraction).
        bin_data_map: BinData mapping from DocInfo.
        processed_images: Already-processed image paths.

    Returns:
        Rendered table string (text or HTML).
    """
    try:
        table_rec = next(
            (c for c in ctrl_header.children if c.tag_id == HWPTAG_TABLE),
            None,
        )
        if table_rec is None or len(table_rec.payload) < 8:
            return ""

        row_cnt = struct.unpack_from("<H", table_rec.payload, 4)[0]
        col_cnt = struct.unpack_from("<H", table_rec.payload, 6)[0]

        grid = _build_grid(
            ctrl_header, traverse_cb, ole, bin_data_map, processed_images
        )

        # 1×1 → container, return inner text directly
        if row_cnt == 1 and col_cnt == 1:
            cell = grid.get((0, 0))
            return cell["text"] if cell else ""

        # Single-column table → line-separated
        if col_cnt == 1:
            parts: List[str] = []
            for r in range(row_cnt):
                cell = grid.get((r, 0))
                if cell and cell["text"]:
                    parts.append(cell["text"])
            return "\n\n".join(parts) if parts else ""

        # Multi-column → HTML
        return render_table_html(grid, row_cnt, col_cnt)

    except Exception as exc:
        logger.warning("Failed to parse HWP table: %s", exc)
        return "[Table Extraction Failed]"


# ── Internal helpers ──────────────────────────────────────────────────────


def _build_grid(
    ctrl_header: HwpRecord,
    traverse_cb: Callable[..., str],
    ole: Any,
    bin_data_map: Optional[Dict],
    processed_images: Optional[Set[str]],
) -> Dict:
    """
    Collect cell content from LIST_HEADER children.

    Returns ``{(row, col): {"text", "rowspan", "colspan"}}``
    """
    grid: Dict = {}
    cells = [c for c in ctrl_header.children if c.tag_id == HWPTAG_LIST_HEADER]

    for cell in cells:
        if len(cell.payload) < 16:
            continue

        para_count = struct.unpack_from("<H", cell.payload, 0)[0]
        col_idx = struct.unpack_from("<H", cell.payload, 8)[0]
        row_idx = struct.unpack_from("<H", cell.payload, 10)[0]
        col_span = struct.unpack_from("<H", cell.payload, 12)[0]
        row_span = struct.unpack_from("<H", cell.payload, 14)[0]

        text_parts: List[str] = []
        if cell.children:
            for child in cell.children:
                t = traverse_cb(child, ole, bin_data_map, processed_images)
                text_parts.append(t)
        else:
            for sib in cell.get_next_siblings(para_count):
                t = traverse_cb(sib, ole, bin_data_map, processed_images)
                text_parts.append(t)

        grid[(row_idx, col_idx)] = {
            "text": "".join(text_parts).strip(),
            "rowspan": row_span,
            "colspan": col_span,
        }

    return grid


def render_table_html(
    grid: Dict,
    row_cnt: int,
    col_cnt: int,
) -> str:
    """Render a cell grid to an HTML ``<table>``."""
    parts: List[str] = ["<table border='1'>"]
    skip: set = set()

    for r in range(row_cnt):
        parts.append("<tr>")
        for c in range(col_cnt):
            if (r, c) in skip:
                continue

            cell = grid.get((r, c))
            if cell:
                rs = cell["rowspan"]
                cs = cell["colspan"]
                attr = ""
                if rs > 1:
                    attr += f" rowspan='{rs}'"
                if cs > 1:
                    attr += f" colspan='{cs}'"
                parts.append(f"<td{attr}>{cell['text']}</td>")

                for dr in range(rs):
                    for dc in range(cs):
                        if dr == 0 and dc == 0:
                            continue
                        skip.add((r + dr, c + dc))
            else:
                parts.append("<td></td>")
        parts.append("</tr>")

    parts.append("</table>")
    return "\n".join(parts)


__all__ = ["parse_table", "render_table_html"]
