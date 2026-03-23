# contextifier/handlers/xls/_table.py
"""
Table conversion utilities for XLS (xlrd) sheets.

Converts a LayoutRange region of an xlrd Sheet into:
- Markdown (if no merged cells in region)
- HTML (if merged cells present)
- TableData (structured representation)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Set, Tuple

from contextifier.types import TableData, TableCell

from contextifier.handlers.xls._layout import LayoutRange, layout_detect_range

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────────


def convert_sheet_to_text(
    sheet: object,
    book: object,
    region: Optional[LayoutRange] = None,
) -> str:
    """Auto-select Markdown or HTML based on merged-cell presence."""
    if region is None:
        region = layout_detect_range(sheet)
        if region is None:
            return ""

    if _has_merged_in_region(sheet, region):
        return convert_region_to_html(sheet, book, region)
    return convert_region_to_markdown(sheet, book, region)


def convert_region_to_table(
    sheet: object,
    book: object,
    region: LayoutRange,
) -> Optional[TableData]:
    """Convert a region to a structured TableData object."""
    merged = _get_merged_in_region(sheet, region)
    skip: Set[Tuple[int, int]] = set()
    for rlo, rhi, clo, chi in merged:
        for r in range(rlo, rhi):
            for c in range(clo, chi):
                if (r, c) != (rlo, clo):
                    skip.add((r, c))

    rows: List[List[TableCell]] = []
    has_content = False

    for r1 in range(region.min_row, region.max_row + 1):
        r0 = r1 - 1
        row_cells: List[TableCell] = []
        for c1 in range(region.min_col, region.max_col + 1):
            c0 = c1 - 1
            if (r0, c0) in skip:
                continue
            val = _format_cell(sheet, book, r0, c0)
            rs, cs = 1, 1
            for rlo, rhi, clo, chi in merged:
                if r0 == rlo and c0 == clo:
                    rs = rhi - rlo
                    cs = chi - clo
                    break
            if val:
                has_content = True
            row_cells.append(
                TableCell(
                    content=val,
                    row_span=rs,
                    col_span=cs,
                    is_header=(r1 == region.min_row),
                    row_index=r1 - region.min_row,
                    col_index=c1 - region.min_col,
                )
            )
        if row_cells:
            rows.append(row_cells)

    if not has_content:
        return None

    return TableData(
        rows=rows,
        num_rows=len(rows),
        num_cols=region.cols,
        has_header=True,
    )


def convert_region_to_markdown(
    sheet: object,
    book: object,
    region: LayoutRange,
) -> str:
    """Render a region as a Markdown pipe table."""
    parts: List[str] = []
    row_count = 0

    for r1 in range(region.min_row, region.max_row + 1):
        r0 = r1 - 1
        cells: List[str] = []
        has_content = False
        for c1 in range(region.min_col, region.max_col + 1):
            c0 = c1 - 1
            val = _format_cell(sheet, book, r0, c0)
            if val:
                has_content = True
            val = val.replace("|", "\\|").replace("\n", " ")
            cells.append(val)

        if not has_content:
            continue

        parts.append("| " + " | ".join(cells) + " |")
        row_count += 1
        if row_count == 1:
            parts.append("| " + " | ".join(["---"] * len(cells)) + " |")

    return "\n".join(parts)


def convert_region_to_html(
    sheet: object,
    book: object,
    region: LayoutRange,
) -> str:
    """Render a region as an HTML table with rowspan/colspan."""
    merged = _get_merged_in_region(sheet, region)
    # Build merge info keyed by (r0, c0)
    merge_info: dict[Tuple[int, int], Tuple[int, int]] = {}
    skip: Set[Tuple[int, int]] = set()
    for rlo, rhi, clo, chi in merged:
        merge_info[(rlo, clo)] = (rhi - rlo, chi - clo)
        for r in range(rlo, rhi):
            for c in range(clo, chi):
                if (r, c) != (rlo, clo):
                    skip.add((r, c))

    parts = ["<table>"]
    has_data = False

    for r1 in range(region.min_row, region.max_row + 1):
        r0 = r1 - 1
        row_parts = ["<tr>"]
        for c1 in range(region.min_col, region.max_col + 1):
            c0 = c1 - 1
            if (r0, c0) in skip:
                continue
            val = _format_cell(sheet, book, r0, c0)
            if val:
                has_data = True
            val = _html_escape(val)
            tag = "th" if r1 == region.min_row else "td"
            attrs: List[str] = []
            if (r0, c0) in merge_info:
                rs, cs = merge_info[(r0, c0)]
                if rs > 1:
                    attrs.append(f"rowspan='{rs}'")
                if cs > 1:
                    attrs.append(f"colspan='{cs}'")
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            row_parts.append(f"<{tag}{attr_str}>{val}</{tag}>")
        row_parts.append("</tr>")
        parts.append("".join(row_parts))

    parts.append("</table>")
    return "\n".join(parts) if has_data else ""


# ── Internal helpers ─────────────────────────────────────────────────────────


def _has_merged_in_region(sheet: object, region: LayoutRange) -> bool:
    """Check whether any merged cell overlaps the region."""
    for rlo, rhi, clo, chi in getattr(sheet, "merged_cells", []):
        mr_min, mr_max = rlo + 1, rhi
        mc_min, mc_max = clo + 1, chi
        if mr_min <= region.max_row and mr_max >= region.min_row and mc_min <= region.max_col and mc_max >= region.min_col:
            return True
    return False


def _get_merged_in_region(
    sheet: object,
    region: LayoutRange,
) -> List[Tuple[int, int, int, int]]:
    """Return list of (rlo, rhi, clo, chi) — 0-based, half-open — overlapping the region."""
    result: List[Tuple[int, int, int, int]] = []
    for rlo, rhi, clo, chi in getattr(sheet, "merged_cells", []):
        mr_min, mr_max = rlo + 1, rhi
        mc_min, mc_max = clo + 1, chi
        if mr_min <= region.max_row and mr_max >= region.min_row and mc_min <= region.max_col and mc_max >= region.min_col:
            result.append((rlo, rhi, clo, chi))
    return result


def _format_cell(sheet: object, book: object, r0: int, c0: int) -> str:
    """Format an xlrd cell value to string (0-based indices)."""
    try:
        import xlrd

        val = sheet.cell_value(r0, c0)  # type: ignore[attr-defined]
        ctype = sheet.cell_type(r0, c0)  # type: ignore[attr-defined]

        if val is None or ctype == xlrd.XL_CELL_EMPTY:
            return ""
        if ctype == xlrd.XL_CELL_NUMBER:
            return str(int(val)) if val == int(val) else str(val)
        if ctype == xlrd.XL_CELL_DATE:
            try:
                dt = xlrd.xldate_as_tuple(val, book.datemode)  # type: ignore[attr-defined]
                return f"{dt[0]:04d}-{dt[1]:02d}-{dt[2]:02d}"
            except Exception:
                return str(val)
        if ctype == xlrd.XL_CELL_BOOLEAN:
            return "TRUE" if val else "FALSE"
        return str(val).strip()
    except Exception:
        return ""


def _html_escape(text: str) -> str:
    if not text:
        return ""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    return text


__all__ = [
    "convert_sheet_to_text",
    "convert_region_to_table",
    "convert_region_to_markdown",
    "convert_region_to_html",
]
