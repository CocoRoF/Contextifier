"""
XLSX table conversion utilities.

Converts openpyxl worksheet regions into text (Markdown or HTML).
The format is chosen automatically:
- **HTML** if the region contains merged cells (for rowspan/colspan)
- **Markdown** otherwise (simpler, more readable)
"""

from __future__ import annotations

import html as html_module
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier.handlers.xlsx._layout import LayoutRange
from contextifier.types import TableData, TableCell

logger = logging.getLogger(__name__)


def convert_region_to_table(
    ws: object,
    region: LayoutRange,
) -> Optional[TableData]:
    """
    Convert a worksheet region to a ``TableData`` object.

    Automatically selects HTML or Markdown format based on whether
    the region contains merged cells.

    Args:
        ws: openpyxl Worksheet.
        region: The cell region to convert.

    Returns:
        TableData or None if the region is empty.
    """
    merged = _get_merged_cells_in_region(ws, region)
    anchor_values = _merge_anchor_values(ws, region)

    rows: List[List[TableCell]] = []
    skip_cells: Set[Tuple[int, int]] = set()

    for row_idx in range(region.min_row, region.max_row + 1):
        row_cells: List[TableCell] = []
        for col_idx in range(region.min_col, region.max_col + 1):
            if (row_idx, col_idx) in skip_cells:
                continue

            cell = ws.cell(row=row_idx, column=col_idx)
            raw = anchor_values.get((row_idx, col_idx), cell.value)
            value = _format_cell_value(raw)

            row_span = 1
            col_span = 1

            # Check if this cell starts a merge
            merge_key = (row_idx, col_idx)
            if merge_key in merged:
                row_span, col_span = merged[merge_key]
                # Mark spanned cells to skip
                for dr in range(row_span):
                    for dc in range(col_span):
                        if dr == 0 and dc == 0:
                            continue
                        skip_cells.add((row_idx + dr, col_idx + dc))

            is_header = row_idx == region.min_row

            row_cells.append(
                TableCell(
                    content=value,
                    row_span=row_span,
                    col_span=col_span,
                    is_header=is_header,
                    row_index=row_idx - region.min_row,
                    col_index=col_idx - region.min_col,
                )
            )

        if row_cells:
            rows.append(row_cells)

    if not rows:
        return None

    # Filter out completely empty tables
    has_content = any(cell.content.strip() for row in rows for cell in row)
    if not has_content:
        return None

    return TableData(
        rows=rows,
        num_rows=len(rows),
        num_cols=region.cols,
        has_header=True,
    )


def convert_region_to_markdown(
    ws: object,
    region: LayoutRange,
    merged_outside: Optional[Dict[Tuple[int, int], str]] = None,
) -> str:
    """
    Convert a worksheet region to a Markdown table.

    Used when there are no merged cells in the region.
    """
    anchor_values = _merge_anchor_values(ws, region)
    lines: List[str] = []
    col_count = region.cols

    for row_idx in range(region.min_row, region.max_row + 1):
        cells: List[str] = []
        for col_idx in range(region.min_col, region.max_col + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            raw = anchor_values.get((row_idx, col_idx), cell.value)
            value = _format_cell_value(raw)

            # Check if merged cell value comes from outside the region
            if not value and merged_outside:
                key = (row_idx, col_idx)
                if key in merged_outside:
                    value = merged_outside[key]

            # Escape pipes for Markdown
            value = value.replace("|", "\\|")
            # Replace newlines with space
            value = value.replace("\n", " ")
            cells.append(value)

        line = "| " + " | ".join(cells) + " |"
        lines.append(line)

        # Add separator after first row (header)
        if row_idx == region.min_row:
            sep = "| " + " | ".join(["---"] * col_count) + " |"
            lines.append(sep)

    return "\n".join(lines)


def convert_region_to_html(
    ws: object,
    region: LayoutRange,
) -> str:
    """
    Convert a worksheet region to an HTML table.

    Used when the region contains merged cells for proper
    rowspan/colspan representation.
    """
    merged = _get_merged_cells_in_region(ws, region)
    anchor_values = _merge_anchor_values(ws, region)
    skip_cells: Set[Tuple[int, int]] = set()

    rows_html: List[str] = []

    for row_idx in range(region.min_row, region.max_row + 1):
        cells_html: List[str] = []
        is_header = row_idx == region.min_row

        for col_idx in range(region.min_col, region.max_col + 1):
            if (row_idx, col_idx) in skip_cells:
                continue

            cell = ws.cell(row=row_idx, column=col_idx)
            raw = anchor_values.get((row_idx, col_idx), cell.value)
            value = _format_cell_value(raw)
            value = _html_escape(value)

            tag = "th" if is_header else "td"
            attrs = ""

            merge_key = (row_idx, col_idx)
            if merge_key in merged:
                row_span, col_span = merged[merge_key]
                if row_span > 1:
                    attrs += f' rowspan="{row_span}"'
                if col_span > 1:
                    attrs += f' colspan="{col_span}"'
                for dr in range(row_span):
                    for dc in range(col_span):
                        if dr == 0 and dc == 0:
                            continue
                        skip_cells.add((row_idx + dr, col_idx + dc))

            cells_html.append(f"<{tag}{attrs}>{value}</{tag}>")

        if cells_html:
            rows_html.append("<tr>" + "".join(cells_html) + "</tr>")

    if not rows_html:
        return ""

    return "<table>\n" + "\n".join(rows_html) + "\n</table>"


def convert_region_to_plain_text(
    ws: object,
    region: LayoutRange,
) -> str:
    """
    Render a region that is too small to be a table as plain text.

    A single cell, a lone row or a lone column has no row/column relationship
    to preserve. Writing it as a table wraps one value in a header separator
    and, because tables are protected from splitting, hands it a chunk of its
    own. The values are kept — one line per row, cells joined by ``" | "`` —
    so nothing is dropped.
    """
    anchor_values = _merge_anchor_values(ws, region)
    lines: List[str] = []

    for row_idx in range(region.min_row, region.max_row + 1):
        values: List[str] = []
        for col_idx in range(region.min_col, region.max_col + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            raw = anchor_values.get((row_idx, col_idx), cell.value)
            text = _format_cell_value(raw).replace("\n", " ").strip()
            if text:
                values.append(text)
        if values:
            lines.append(" | ".join(values))

    return "\n".join(lines)


def convert_sheet_to_text(
    ws: object,
    region: LayoutRange,
) -> str:
    """
    Convert a worksheet region to text, auto-selecting format.

    - plain text if the region is too small to have a grid
    - HTML if merged cells present
    - Markdown otherwise
    """
    if not region.is_table_like():
        return convert_region_to_plain_text(ws, region)

    merged = _get_merged_cells_in_region(ws, region)
    if merged:
        return convert_region_to_html(ws, region)
    else:
        return convert_region_to_markdown(ws, region)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _get_merged_cells_in_region(
    ws: object,
    region: LayoutRange,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Merged cells as they appear *within* a region, clipped to it.

    A merge is anchored at its top-left cell, and only that cell holds the
    value. When a sheet is split into several regions, a merge can cover a
    region while being anchored in an earlier one — the region then shows a
    blank column where the label should be, and the label is lost. Equally, a
    merge anchored inside the region can reach past its last row, leaving a
    span that points at rows the table does not contain.

    Both cases are handled by clipping: the merge is reported at the first
    cell of the region it actually covers, with the span it has inside the
    region.

    Returns:
        ``{(row, col): (row_span, col_span)}`` for cells the region renders.
    """
    merged: Dict[Tuple[int, int], Tuple[int, int]] = {}

    try:
        for merge_range in ws.merged_cells.ranges:
            start_row, start_col = merge_range.min_row, merge_range.min_col
            end_row, end_col = merge_range.max_row, merge_range.max_col

            # Overlap with the region, in region coordinates.
            first_row = max(start_row, region.min_row)
            last_row = min(end_row, region.max_row)
            first_col = max(start_col, region.min_col)
            last_col = min(end_col, region.max_col)
            if first_row > last_row or first_col > last_col:
                continue

            merged[(first_row, first_col)] = (
                last_row - first_row + 1,
                last_col - first_col + 1,
            )
    except Exception as exc:
        logger.debug("Failed to read merged cells: %s", exc)

    return merged


def _merge_anchor_values(
    ws: object,
    region: LayoutRange,
) -> Dict[Tuple[int, int], Any]:
    """
    Values to display for merges whose anchor lies outside *region*.

    Only the anchor cell of a merge holds the value, so a region that sees the
    tail of a merge reads an empty cell. This maps the region's first covered
    cell to the anchor's value so the label is not lost.
    """
    values: Dict[Tuple[int, int], Any] = {}

    try:
        for merge_range in ws.merged_cells.ranges:
            start_row, start_col = merge_range.min_row, merge_range.min_col
            if region.contains(start_row, start_col):
                continue  # anchor is inside; the normal read finds the value

            first_row = max(start_row, region.min_row)
            last_row = min(merge_range.max_row, region.max_row)
            first_col = max(start_col, region.min_col)
            last_col = min(merge_range.max_col, region.max_col)
            if first_row > last_row or first_col > last_col:
                continue

            value = ws.cell(row=start_row, column=start_col).value
            if value is not None:
                values[(first_row, first_col)] = value
    except Exception as exc:
        logger.debug("Failed to resolve merge anchors: %s", exc)

    return values


def _format_cell_value(value: Any) -> str:
    """Convert a cell value to a clean string."""
    if value is None:
        return ""
    if isinstance(value, float):
        # Format integers without decimal
        if value == int(value):
            return str(int(value))
        return str(value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value).strip()


def _html_escape(text: str) -> str:
    """HTML-escape text and convert newlines to <br>."""
    text = html_module.escape(text)
    text = text.replace("\n", "<br>")
    return text


__all__ = [
    "convert_region_to_table",
    "convert_region_to_markdown",
    "convert_region_to_html",
    "convert_sheet_to_text",
]
