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

from contextifier_new.handlers.xlsx._layout import LayoutRange
from contextifier_new.types import TableData, TableCell

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

    rows: List[List[TableCell]] = []
    skip_cells: Set[Tuple[int, int]] = set()

    for row_idx in range(region.min_row, region.max_row + 1):
        row_cells: List[TableCell] = []
        for col_idx in range(region.min_col, region.max_col + 1):
            if (row_idx, col_idx) in skip_cells:
                continue

            cell = ws.cell(row=row_idx, column=col_idx)
            value = _format_cell_value(cell.value)

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

            is_header = (row_idx == region.min_row)

            row_cells.append(TableCell(
                content=value,
                row_span=row_span,
                col_span=col_span,
                is_header=is_header,
                row_index=row_idx - region.min_row,
                col_index=col_idx - region.min_col,
            ))

        if row_cells:
            rows.append(row_cells)

    if not rows:
        return None

    # Filter out completely empty tables
    has_content = any(
        cell.content.strip()
        for row in rows
        for cell in row
    )
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
    lines: List[str] = []
    col_count = region.cols

    for row_idx in range(region.min_row, region.max_row + 1):
        cells: List[str] = []
        for col_idx in range(region.min_col, region.max_col + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            value = _format_cell_value(cell.value)

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
    skip_cells: Set[Tuple[int, int]] = set()

    rows_html: List[str] = []

    for row_idx in range(region.min_row, region.max_row + 1):
        cells_html: List[str] = []
        is_header = (row_idx == region.min_row)

        for col_idx in range(region.min_col, region.max_col + 1):
            if (row_idx, col_idx) in skip_cells:
                continue

            cell = ws.cell(row=row_idx, column=col_idx)
            value = _format_cell_value(cell.value)
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


def convert_sheet_to_text(
    ws: object,
    region: LayoutRange,
) -> str:
    """
    Convert a worksheet region to text, auto-selecting format.

    - HTML if merged cells present
    - Markdown otherwise
    """
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
    Find merged cells that start within the region.

    Returns dict of (start_row, start_col) → (row_span, col_span).
    """
    merged: Dict[Tuple[int, int], Tuple[int, int]] = {}

    try:
        for merge_range in ws.merged_cells.ranges:
            start_row = merge_range.min_row
            start_col = merge_range.min_col
            end_row = merge_range.max_row
            end_col = merge_range.max_col

            # Check if merge starts within our region
            if region.contains(start_row, start_col):
                row_span = end_row - start_row + 1
                col_span = end_col - start_col + 1
                merged[(start_row, start_col)] = (row_span, col_span)
    except Exception:
        pass

    return merged


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
