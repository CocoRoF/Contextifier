# contextifier/handlers/xls/_layout.py
"""
Layout detection for XLS (xlrd) sheets.

Re-uses the same LayoutRange dataclass and BFS-based object detection
logic as the XLSX handler, adapted for xlrd's 0-based API.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from contextifier.handlers.xls._constants import MAX_SCAN_COLS, MAX_SCAN_ROWS

logger = logging.getLogger(__name__)


@dataclass
class LayoutRange:
    """Bounding rectangle (1-based, inclusive)."""

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

    def contains(self, row: int, col: int) -> bool:
        return self.min_row <= row <= self.max_row and self.min_col <= col <= self.max_col

    def overlaps(self, other: LayoutRange) -> bool:
        return not (
            self.max_row < other.min_row
            or self.min_row > other.max_row
            or self.max_col < other.min_col
            or self.min_col > other.max_col
        )

    def is_adjacent(self, other: LayoutRange, tolerance: int = 1) -> bool:
        v_gap = max(other.min_row - self.max_row, self.min_row - other.max_row)
        h_gap = max(other.min_col - self.max_col, self.min_col - other.max_col)
        row_overlap = self.min_row <= other.max_row and other.min_row <= self.max_row
        col_overlap = self.min_col <= other.max_col and other.min_col <= self.max_col
        if col_overlap and 0 < v_gap <= tolerance:
            return True
        if row_overlap and 0 < h_gap <= tolerance:
            return True
        return self.overlaps(other)

    def merge_with(self, other: LayoutRange) -> LayoutRange:
        return LayoutRange(
            min_row=min(self.min_row, other.min_row),
            max_row=max(self.max_row, other.max_row),
            min_col=min(self.min_col, other.min_col),
            max_col=max(self.max_col, other.max_col),
        )


# ── Layout detection ─────────────────────────────────────────────────────────


def layout_detect_range(sheet: object) -> Optional[LayoutRange]:
    """
    Detect the bounding rectangle of non-empty cells in an xlrd Sheet.

    Uses 0-based xlrd API internally but returns 1-based LayoutRange.
    """
    try:
        if sheet.nrows == 0 or sheet.ncols == 0:  # type: ignore[attr-defined]
            return None

        max_r = min(sheet.nrows, MAX_SCAN_ROWS)  # type: ignore[attr-defined]
        max_c = min(sheet.ncols, MAX_SCAN_COLS)  # type: ignore[attr-defined]

        min_row = max_row = min_col = max_col = None

        for r in range(max_r):
            for c in range(max_c):
                try:
                    val = sheet.cell_value(r, c)  # type: ignore[attr-defined]
                except Exception:
                    continue
                if val is None or str(val).strip() == "":
                    continue
                r1, c1 = r + 1, c + 1  # 1-based
                if min_row is None or r1 < min_row:
                    min_row = r1
                if max_row is None or r1 > max_row:
                    max_row = r1
                if min_col is None or c1 < min_col:
                    min_col = c1
                if max_col is None or c1 > max_col:
                    max_col = c1

        if min_row is None:
            return None

        return LayoutRange(
            min_row=min_row,
            max_row=max_row,  # type: ignore[arg-type]
            min_col=min_col,  # type: ignore[arg-type]
            max_col=max_col,  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.debug("layout_detect_range failed: %s", exc)
        return None


# ── Object detection ─────────────────────────────────────────────────────────


def object_detect(sheet: object, book: object = None, layout: Optional[LayoutRange] = None) -> List[LayoutRange]:
    """
    Detect individual data regions in an xlrd sheet.

    Algorithm:
    1. Find bordered cells (if book has formatting info)
    2. Find value-only cells outside bordered regions
    3. BFS-group both sets independently
    4. Merge adjacent regions
    5. Sort top→bottom, left→right
    """
    try:
        if layout is None:
            layout = layout_detect_range(sheet)
            if layout is None:
                return []

        bordered = _detect_bordered_regions(sheet, book, layout) if book is not None else []
        value_regions = _detect_value_regions(sheet, layout, bordered)

        all_regions = bordered + value_regions
        if not all_regions:
            return []

        merged = _merge_adjacent_regions(all_regions)
        return sorted(merged, key=lambda r: (r.min_row, r.min_col))
    except Exception as exc:
        logger.debug("object_detect failed: %s", exc)
        return []


# ── Border detection ─────────────────────────────────────────────────────────


def _has_border(sheet: object, book: object, row0: int, col0: int) -> bool:
    """Check if a cell has a visible border (0-based indices)."""
    try:
        xf_idx = sheet.cell_xf_index(row0, col0)  # type: ignore[attr-defined]
        xf = book.xf_list[xf_idx]  # type: ignore[attr-defined]
        for attr in ("top_line_style", "bottom_line_style", "left_line_style", "right_line_style"):
            val = getattr(xf.border, attr, 0)
            if val and val > 0:
                return True
    except Exception:
        pass
    return False


def _detect_bordered_regions(sheet: object, book: object, layout: LayoutRange) -> List[LayoutRange]:
    bordered: Set[Tuple[int, int]] = set()
    for r1 in range(layout.min_row, layout.max_row + 1):
        for c1 in range(layout.min_col, layout.max_col + 1):
            if _has_border(sheet, book, r1 - 1, c1 - 1):
                bordered.add((r1, c1))
    if not bordered:
        return []
    return _bfs_group(bordered)


# ── Value detection ──────────────────────────────────────────────────────────


def _detect_value_regions(sheet: object, layout: LayoutRange, exclude: List[LayoutRange]) -> List[LayoutRange]:
    def _excluded(r: int, c: int) -> bool:
        return any(reg.contains(r, c) for reg in exclude)

    cells: Set[Tuple[int, int]] = set()
    for r1 in range(layout.min_row, layout.max_row + 1):
        for c1 in range(layout.min_col, layout.max_col + 1):
            if _excluded(r1, c1):
                continue
            try:
                val = sheet.cell_value(r1 - 1, c1 - 1)  # type: ignore[attr-defined]
            except Exception:
                continue
            if val is not None and str(val).strip():
                cells.add((r1, c1))
    if not cells:
        return []
    return _bfs_group(cells)


# ── BFS grouping ─────────────────────────────────────────────────────────────


def _bfs_group(cells: Set[Tuple[int, int]]) -> List[LayoutRange]:
    visited: Set[Tuple[int, int]] = set()
    regions: List[LayoutRange] = []

    for start in sorted(cells):
        if start in visited:
            continue
        group: Set[Tuple[int, int]] = set()
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            group.add(cur)
            r, c = cur
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cells and (nr, nc) not in visited:
                    queue.append((nr, nc))
        if group:
            min_r = min(r for r, _ in group)
            max_r = max(r for r, _ in group)
            min_c = min(c for _, c in group)
            max_c = max(c for _, c in group)
            regions.append(LayoutRange(min_row=min_r, max_row=max_r, min_col=min_c, max_col=max_c))

    return regions


# ── Region merging ───────────────────────────────────────────────────────────


def _merge_adjacent_regions(regions: List[LayoutRange]) -> List[LayoutRange]:
    if len(regions) <= 1:
        return list(regions)

    current = list(regions)
    changed = True
    while changed:
        changed = False
        result: List[LayoutRange] = []
        used: Set[int] = set()
        for i, a in enumerate(current):
            if i in used:
                continue
            merged = a
            for j, b in enumerate(current):
                if j <= i or j in used:
                    continue
                if merged.is_adjacent(b):
                    merged = merged.merge_with(b)
                    used.add(j)
                    changed = True
            result.append(merged)
            used.add(i)
        current = result

    return current


__all__ = [
    "LayoutRange",
    "layout_detect_range",
    "object_detect",
]
