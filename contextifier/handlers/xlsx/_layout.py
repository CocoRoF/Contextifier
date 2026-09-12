"""
XLSX layout detection and object grouping.

Implements BFS-based region detection for Excel worksheets:
1. ``LayoutRange`` — represents a rectangular cell region (1-based)
2. ``layout_detect_range()`` — find the data-containing bounds of a sheet
3. ``object_detect()`` — group cells into bordered and value regions

This module ports the v1 ``excel_layout_detector.py`` logic to the v2
architecture while cleaning up the code structure.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from contextifier.handlers.xlsx._constants import (
    MIN_TABLE_COLS,
    MIN_TABLE_ROWS,
    SCAN_COL_LIMIT,
    SCAN_ROW_LIMIT,
)

logger = logging.getLogger(__name__)


@dataclass
class LayoutRange:
    """A rectangular region in a worksheet (1-based coordinates)."""

    min_row: int
    max_row: int
    min_col: int
    max_col: int

    @property
    def rows(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def cols(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def cell_count(self) -> int:
        return self.rows * self.cols

    def is_table_like(
        self,
        *,
        min_rows: int = MIN_TABLE_ROWS,
        min_cols: int = MIN_TABLE_COLS,
    ) -> bool:
        """
        Whether the region has enough of a grid to be worth calling a table.

        A single cell, a lone row or a lone column carries no row/column
        relationship — rendering one as a table produces a header separator
        around one value and, because tables are protected from splitting,
        hands it a whole chunk of its own. The threshold matches the minimum
        the DOCX, HWPX and PDF table detectors already enforce.
        """
        return self.rows >= min_rows and self.cols >= min_cols

    def is_adjacent(self, other: "LayoutRange", *, tolerance: int = 1) -> bool:
        """Check if two ranges are adjacent (within tolerance rows/cols)."""
        row_gap = max(
            0, max(self.min_row, other.min_row) - min(self.max_row, other.max_row) - 1
        )
        col_gap = max(
            0, max(self.min_col, other.min_col) - min(self.max_col, other.max_col) - 1
        )
        return row_gap <= tolerance and col_gap <= tolerance

    def merge_with(self, other: "LayoutRange") -> "LayoutRange":
        """Return a new range encompassing both ranges."""
        return LayoutRange(
            min_row=min(self.min_row, other.min_row),
            max_row=max(self.max_row, other.max_row),
            min_col=min(self.min_col, other.min_col),
            max_col=max(self.max_col, other.max_col),
        )

    def overlaps(self, other: "LayoutRange") -> bool:
        """Check if two ranges overlap."""
        return not (
            self.max_row < other.min_row
            or other.max_row < self.min_row
            or self.max_col < other.min_col
            or other.max_col < self.min_col
        )

    def contains(self, row: int, col: int) -> bool:
        """Check if a cell is within this range."""
        return (
            self.min_row <= row <= self.max_row and self.min_col <= col <= self.max_col
        )


def sheet_extent(ws: object) -> tuple:
    """
    The sheet's addressable extent, clamped to a sanity bound.

    openpyxl reports ``max_row``/``max_column`` from the sheet dimension, which
    can be inflated by stray formatting but is never smaller than the data.
    Scanning a fixed window instead silently truncates every sheet past it —
    a 1,500-row export lost a third of its rows with nothing in the output to
    say so.

    Returns:
        ``(max_row, max_col)``, both at least 1.
    """
    try:
        max_row = int(getattr(ws, "max_row", 0) or 0)
        max_col = int(getattr(ws, "max_column", 0) or 0)
    except (TypeError, ValueError):
        return 1, 1

    if max_row > SCAN_ROW_LIMIT:
        logger.warning(
            "Sheet reports %d rows; scanning the first %d",
            max_row,
            SCAN_ROW_LIMIT,
        )
        max_row = SCAN_ROW_LIMIT
    if max_col > SCAN_COL_LIMIT:
        logger.warning(
            "Sheet reports %d columns; scanning the first %d",
            max_col,
            SCAN_COL_LIMIT,
        )
        max_col = SCAN_COL_LIMIT

    return max(1, max_row), max(1, max_col)


def layout_detect_range(ws: object) -> Optional[LayoutRange]:
    """
    Detect the data-containing rectangular region of a worksheet.

    Args:
        ws: openpyxl Worksheet object.

    Returns:
        LayoutRange or None if the sheet is empty.
    """
    max_row, max_col = sheet_extent(ws)

    min_row = None
    max_row_seen = None
    min_col = None
    max_col_seen = None

    try:
        # iter_rows() streams the used range; ws.cell() in a nested loop
        # materialises a cell object per coordinate and is far slower on the
        # large sheets this now has to handle.
        for row_idx, row in enumerate(
            ws.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col),
            start=1,
        ):
            for col_idx, cell in enumerate(row, start=1):
                if cell.value is None:
                    continue
                if min_row is None or row_idx < min_row:
                    min_row = row_idx
                if max_row_seen is None or row_idx > max_row_seen:
                    max_row_seen = row_idx
                if min_col is None or col_idx < min_col:
                    min_col = col_idx
                if max_col_seen is None or col_idx > max_col_seen:
                    max_col_seen = col_idx
    except Exception as exc:
        logger.debug("Error during layout detection: %s", exc)

    if min_row is None:
        return None

    return LayoutRange(
        min_row=min_row,
        max_row=max_row_seen,
        min_col=min_col,
        max_col=max_col_seen,
    )


def object_detect(
    ws: object,
    layout: Optional[LayoutRange] = None,
) -> List[LayoutRange]:
    """
    Detect table-like regions in a worksheet using BFS grouping.

    Algorithm:
    1. Find bordered cells → BFS group into regions
    2. Find value-only cells (excluding bordered) → BFS group
    3. Merge adjacent regions
    4. Sort top→bottom, left→right

    Args:
        ws: openpyxl Worksheet object.
        layout: Optional overall layout bounds (if None, auto-detect).

    Returns:
        List of LayoutRange objects representing table regions.
    """
    if layout is None:
        layout = layout_detect_range(ws)
    if layout is None:
        return []

    # Phase 1: Find bordered cells
    bordered_cells: Set[Tuple[int, int]] = set()
    value_cells: Set[Tuple[int, int]] = set()

    try:
        for row_idx, row in enumerate(
            ws.iter_rows(
                min_row=layout.min_row,
                max_row=layout.max_row,
                min_col=layout.min_col,
                max_col=layout.max_col,
            ),
            start=layout.min_row,
        ):
            for col_idx, cell in enumerate(row, start=layout.min_col):
                if _has_border(cell):
                    bordered_cells.add((row_idx, col_idx))
                elif cell.value is not None:
                    value_cells.add((row_idx, col_idx))
    except Exception as exc:
        logger.debug("Error during region detection: %s", exc)

    # A merged range is one cell as far as the reader is concerned, but only
    # its anchor holds a value — the rest read as empty and break the region
    # apart, so a table with a merged category column is reported as several
    # unrelated blocks and the column is lost from all but the first. Treating
    # the whole range as occupied keeps it together.
    value_cells |= _merged_range_cells(ws, layout)
    value_cells -= bordered_cells

    # Phase 2: BFS group bordered cells into regions
    regions: List[LayoutRange] = []
    if bordered_cells:
        bordered_regions = _bfs_group(bordered_cells, layout)
        regions.extend(bordered_regions)

    # Phase 3: BFS group value cells (excluding bordered)
    if value_cells:
        value_regions = _bfs_group(value_cells, layout)
        regions.extend(value_regions)

    if not regions:
        # Fallback: treat entire layout as one region
        return [layout]

    # Phase 4: Merge adjacent regions
    regions = _merge_adjacent_regions(regions)

    # Sort: top→bottom, then left→right
    regions.sort(key=lambda r: (r.min_row, r.min_col))

    return regions


def _merged_range_cells(ws: object, layout: LayoutRange) -> Set[Tuple[int, int]]:
    """Every cell of each non-empty merged range that intersects *layout*."""
    cells: Set[Tuple[int, int]] = set()
    try:
        for merge_range in ws.merged_cells.ranges:
            if ws.cell(row=merge_range.min_row, column=merge_range.min_col).value is None:
                continue
            first_row = max(merge_range.min_row, layout.min_row)
            last_row = min(merge_range.max_row, layout.max_row)
            first_col = max(merge_range.min_col, layout.min_col)
            last_col = min(merge_range.max_col, layout.max_col)
            if first_row > last_row or first_col > last_col:
                continue
            for row in range(first_row, last_row + 1):
                for col in range(first_col, last_col + 1):
                    cells.add((row, col))
    except Exception as exc:
        logger.debug("Failed to expand merged ranges: %s", exc)
    return cells


def _has_border(cell: object) -> bool:
    """Check if a cell has any non-none border style."""
    try:
        border = cell.border
        if border is None:
            return False
        for side_name in ("left", "right", "top", "bottom"):
            side = getattr(border, side_name, None)
            if side is not None:
                style = getattr(side, "style", None)
                if style is not None and style != "none":
                    return True
    except Exception:
        pass
    return False


def _bfs_group(
    cells: Set[Tuple[int, int]],
    layout: LayoutRange,
) -> List[LayoutRange]:
    """Group adjacent cells into rectangular regions using BFS."""
    visited: Set[Tuple[int, int]] = set()
    regions: List[LayoutRange] = []

    for cell in sorted(cells):
        if cell in visited:
            continue

        # BFS to find connected component
        queue = deque([cell])
        visited.add(cell)
        component: List[Tuple[int, int]] = []

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (
                    neighbor in cells
                    and neighbor not in visited
                    and layout.contains(nr, nc)
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Convert component to bounding rectangle
        if component:
            rows = [p[0] for p in component]
            cols = [p[1] for p in component]
            region = LayoutRange(
                min_row=min(rows),
                max_row=max(rows),
                min_col=min(cols),
                max_col=max(cols),
            )
            regions.append(region)

    return regions


def _merge_adjacent_regions(regions: List[LayoutRange]) -> List[LayoutRange]:
    """Iteratively merge adjacent or overlapping regions."""
    if len(regions) <= 1:
        return regions

    changed = True
    while changed:
        changed = False
        merged: List[LayoutRange] = []
        used: Set[int] = set()

        for i in range(len(regions)):
            if i in used:
                continue
            current = regions[i]
            for j in range(i + 1, len(regions)):
                if j in used:
                    continue
                if current.is_adjacent(regions[j]) or current.overlaps(regions[j]):
                    current = current.merge_with(regions[j])
                    used.add(j)
                    changed = True
            merged.append(current)
            used.add(i)

        regions = merged

    return regions


__all__ = [
    "LayoutRange",
    "layout_detect_range",
    "object_detect",
]
