# contextifier_new/handlers/pdf_plus/_line_analysis.py
"""
PDF Plus — Line Analysis Engine.

Extracts and analyses lines from PDF drawings to build grid structures
used for line-based table detection.

Pipeline:
  extract_all_lines → classify → merge_double_lines → build_grid → reconstruct_border
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    GridInfo,
    LineInfo,
    LineThickness,
    PdfPlusConfig,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class LineAnalysisEngine:
    """
    Extracts straight lines from PyMuPDF drawing primitives, classifies their
    thickness, merges double-line borders, and optionally constructs a grid
    structure suitable for table detection.
    """

    def __init__(self, page: Any, page_width: float, page_height: float) -> None:
        self.page = page
        self.page_width = page_width
        self.page_height = page_height

        self.all_lines: list[LineInfo] = []
        self.h_lines: list[LineInfo] = []
        self.v_lines: list[LineInfo] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> Tuple[List[LineInfo], List[LineInfo]]:
        """Run the analysis pipeline and return *(h_lines, v_lines)*."""
        self._extract_all_lines()
        self._classify_lines()
        self._merge_double_lines()
        return self.h_lines, self.v_lines

    def build_grid(self, tolerance: float | None = None) -> Optional[GridInfo]:
        """Reconstruct a grid from detected lines."""
        if tolerance is None:
            tolerance = CFG.LINE_MERGE_TOLERANCE

        if not self.h_lines and not self.v_lines:
            return None

        h_positions = self._cluster_positions(
            [ln.y0 for ln in self.h_lines], tolerance
        )
        v_positions = self._cluster_positions(
            [ln.x0 for ln in self.v_lines], tolerance
        )

        if len(h_positions) < 2 or len(v_positions) < 2:
            return None

        x0, x1 = min(v_positions), max(v_positions)
        y0, y1 = min(h_positions), max(h_positions)
        is_complete = self._check_border_completeness(h_positions, v_positions)

        return GridInfo(
            h_lines=sorted(h_positions),
            v_lines=sorted(v_positions),
            bbox=(x0, y0, x1, y1),
            is_complete=is_complete,
            reconstructed=False,
        )

    def reconstruct_incomplete_border(self, grid: GridInfo) -> GridInfo:
        """
        Attempts to close a 3-sided grid by extending a 4th border.
        """
        if grid.is_complete:
            return grid

        h_lines = list(grid.h_lines)
        v_lines = list(grid.v_lines)

        y_min, y_max = min(h_lines), max(h_lines)
        x_min, x_max = min(v_lines), max(v_lines)
        reconstructed = False

        if not any(abs(y - y_min) < CFG.LINE_MERGE_TOLERANCE for y in h_lines) and len(h_lines) >= 2:
            h_lines.insert(0, y_min - CFG.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        if not any(abs(y - y_max) < CFG.LINE_MERGE_TOLERANCE for y in h_lines) and len(h_lines) >= 2:
            h_lines.append(y_max + CFG.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        if not any(abs(x - x_min) < CFG.LINE_MERGE_TOLERANCE for x in v_lines) and len(v_lines) >= 2:
            v_lines.insert(0, x_min - CFG.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        if not any(abs(x - x_max) < CFG.LINE_MERGE_TOLERANCE for x in v_lines) and len(v_lines) >= 2:
            v_lines.append(x_max + CFG.BORDER_EXTENSION_MARGIN)
            reconstructed = True

        if not reconstructed:
            return grid

        return GridInfo(
            h_lines=sorted(h_lines),
            v_lines=sorted(v_lines),
            bbox=(min(v_lines), min(h_lines), max(v_lines), max(h_lines)),
            is_complete=True,
            reconstructed=True,
        )

    # ------------------------------------------------------------------
    # Line extraction
    # ------------------------------------------------------------------

    def _extract_all_lines(self) -> None:
        drawings = self.page.get_drawings()
        if not drawings:
            return
        for d in drawings:
            rect = d.get("rect")
            if rect is None:
                continue
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            w, h = abs(x1 - x0), abs(y1 - y0)
            stroke_width = d.get("width", 1.0) or 1.0

            is_h = h <= max(3.0, stroke_width * 2) and w > 10
            is_v = w <= max(3.0, stroke_width * 2) and h > 10

            if not (is_h or is_v):
                for item in d.get("items", []):
                    if item[0] == "l":
                        self._add_line_from_points(item[1], item[2], stroke_width)
                continue

            self.all_lines.append(
                LineInfo(
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    thickness=stroke_width,
                    thickness_class=self._classify_thickness(stroke_width),
                    is_horizontal=is_h,
                    is_vertical=is_v,
                )
            )

    def _add_line_from_points(self, p1: Any, p2: Any, stroke_width: float) -> None:
        x0, y0 = p1.x, p1.y
        x1, y1 = p2.x, p2.y
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        is_h = dy < 3 and dx > 10
        is_v = dx < 3 and dy > 10
        if not (is_h or is_v):
            return
        self.all_lines.append(
            LineInfo(
                x0=min(x0, x1), y0=min(y0, y1),
                x1=max(x0, x1), y1=max(y0, y1),
                thickness=stroke_width,
                thickness_class=self._classify_thickness(stroke_width),
                is_horizontal=is_h, is_vertical=is_v,
            )
        )

    # ------------------------------------------------------------------
    # Classification & merging
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_thickness(t: float) -> LineThickness:
        if t < CFG.THIN_LINE_THRESHOLD:
            return LineThickness.THIN
        if t > CFG.THICK_LINE_THRESHOLD:
            return LineThickness.THICK
        return LineThickness.NORMAL

    def _classify_lines(self) -> None:
        for ln in self.all_lines:
            if ln.is_horizontal:
                self.h_lines.append(ln)
            elif ln.is_vertical:
                self.v_lines.append(ln)

    def _merge_double_lines(self) -> None:
        self.h_lines = self._merge_parallel(self.h_lines, is_horizontal=True)
        self.v_lines = self._merge_parallel(self.v_lines, is_horizontal=False)

    def _merge_parallel(self, lines: List[LineInfo], *, is_horizontal: bool) -> List[LineInfo]:
        if len(lines) < 2:
            return lines
        if is_horizontal:
            sorted_lines = sorted(lines, key=lambda l: (l.y0, l.x0))
        else:
            sorted_lines = sorted(lines, key=lambda l: (l.x0, l.y0))

        merged: list[LineInfo] = []
        used: set[int] = set()
        for i, l1 in enumerate(sorted_lines):
            if i in used:
                continue
            cur = l1
            for j in range(i + 1, len(sorted_lines)):
                if j in used:
                    continue
                l2 = sorted_lines[j]
                if self._is_double(cur, l2, is_horizontal):
                    cur = self._merge_two(cur, l2, is_horizontal)
                    used.add(j)
            merged.append(cur)
            used.add(i)
        return merged

    def _is_double(self, a: LineInfo, b: LineInfo, is_h: bool) -> bool:
        if is_h:
            if abs(a.y0 - b.y0) > CFG.DOUBLE_LINE_GAP:
                return False
            overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            return overlap > min(self._length(a), self._length(b)) * 0.5
        else:
            if abs(a.x0 - b.x0) > CFG.DOUBLE_LINE_GAP:
                return False
            overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
            return overlap > min(self._length(a), self._length(b)) * 0.5

    @staticmethod
    def _merge_two(a: LineInfo, b: LineInfo, is_h: bool) -> LineInfo:
        thicker = a if a.thickness >= b.thickness else b
        if is_h:
            avg_y = (a.y0 + b.y0) / 2
            return LineInfo(
                x0=min(a.x0, b.x0), y0=avg_y,
                x1=max(a.x1, b.x1), y1=avg_y,
                thickness=max(a.thickness, b.thickness),
                thickness_class=thicker.thickness_class,
                is_horizontal=True, is_vertical=False,
            )
        avg_x = (a.x0 + b.x0) / 2
        return LineInfo(
            x0=avg_x, y0=min(a.y0, b.y0),
            x1=avg_x, y1=max(a.y1, b.y1),
            thickness=max(a.thickness, b.thickness),
            thickness_class=thicker.thickness_class,
            is_horizontal=False, is_vertical=True,
        )

    @staticmethod
    def _length(ln: LineInfo) -> float:
        return math.sqrt((ln.x1 - ln.x0) ** 2 + (ln.y1 - ln.y0) ** 2)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_positions(positions: List[float], tolerance: float) -> List[float]:
        if not positions:
            return []
        sorted_pos = sorted(positions)
        clusters: list[list[float]] = [[sorted_pos[0]]]
        for p in sorted_pos[1:]:
            if p - clusters[-1][-1] <= tolerance:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [sum(c) / len(c) for c in clusters]

    def _check_border_completeness(
        self, h_positions: List[float], v_positions: List[float]
    ) -> bool:
        if len(h_positions) < 2 or len(v_positions) < 2:
            return False
        y_min, y_max = min(h_positions), max(h_positions)
        x_min, x_max = min(v_positions), max(v_positions)
        tol = CFG.LINE_MERGE_TOLERANCE
        has_top = any(ln.y0 <= y_min + tol for ln in self.h_lines)
        has_bot = any(ln.y0 >= y_max - tol for ln in self.h_lines)
        has_left = any(ln.x0 <= x_min + tol for ln in self.v_lines)
        has_right = any(ln.x0 >= x_max - tol for ln in self.v_lines)
        return all([has_top, has_bot, has_left, has_right])


__all__ = ["LineAnalysisEngine"]
