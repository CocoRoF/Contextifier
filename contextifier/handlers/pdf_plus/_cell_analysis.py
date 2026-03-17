# contextifier_new/handlers/pdf_plus/_cell_analysis.py
"""
PDF Plus — Cell Analysis Engine.

Calculates accurate *rowspan / colspan* from physical cell bboxes
(PyMuPDF ``table.cells``) or, when those are unavailable, reconstructs
span information from the data grid.

Algorithm priority:
  1. Use pre-computed spans from ``TableDetectionEngine`` if valid.
  2. Build a grid from per-cell bboxes and compute spans geometrically.
  3. Validate / enhance existing spans with text-position heuristics.
  4. Fall back to 1×1 cells for every data cell.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier.handlers.pdf_plus._types import CellInfo, PdfPlusConfig

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class CellAnalysisEngine:
    """Compute merged-cell spans for a single table."""

    GRID_TOLERANCE: float = CFG.CELL_GRID_TOLERANCE
    OVERLAP_THRESHOLD: float = CFG.CELL_OVERLAP_THRESHOLD

    def __init__(self, data: List[List], cells_info: List[CellInfo], bbox: Optional[tuple] = None, page: Any = None) -> None:
        self.data = data or []
        self.cells_info = cells_info or []
        self.bbox = bbox
        self.page = page
        # Grid-line caches
        self._h_grid: list[float] = []
        self._v_grid: list[float] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def analyze(self) -> List[Dict]:
        """
        Returns a list of ``{row, col, rowspan, colspan, bbox}`` dicts.
        """
        nr = len(self.data)
        nc = max((len(r) for r in self.data), default=0)
        if not nr or not nc:
            return []

        # 1. Pre-computed valid spans → just validate
        if self.cells_info and self._has_valid_spans():
            result = self._use_existing(nr, nc)
            if result:
                return result

        # 2. Cells with bboxes → geometric approach
        ci_dicts = self._cells_as_dicts()
        if ci_dicts and any(c.get("bbox") for c in ci_dicts):
            result = self._analyze_with_bbox(ci_dicts, nr, nc)
            if result:
                return result

        # 3. Validate existing cells (non-bbox)
        if ci_dicts:
            result = self._validate_and_enhance(ci_dicts, nr, nc)
            if result:
                return result

        # 4. Default 1×1
        return self._default_cells(nr, nc)

    # ------------------------------------------------------------------
    # Strategy 1: trust existing spans
    # ------------------------------------------------------------------

    def _has_valid_spans(self) -> bool:
        if not self.cells_info:
            return False
        has_span = any(
            (c.rowspan > 1 or c.colspan > 1) for c in self.cells_info
        )
        has_pos = all(
            c.row is not None and c.col is not None for c in self.cells_info
        )
        return has_span or has_pos

    def _use_existing(self, nr: int, nc: int) -> List[Dict]:
        out: list[dict] = []
        covered: set[tuple[int, int]] = set()
        for c in self.cells_info:
            r, co = c.row, c.col
            if r >= nr or co >= nc:
                continue
            rs = min(max(1, c.rowspan), nr - r)
            cs = min(max(1, c.colspan), nc - co)
            if (r, co) in covered:
                continue
            out.append({"row": r, "col": co, "rowspan": rs, "colspan": cs, "bbox": c.bbox})
            for ri in range(r, r + rs):
                for ci in range(co, co + cs):
                    covered.add((ri, ci))
        # Fill uncovered
        for ri in range(nr):
            for ci in range(nc):
                if (ri, ci) not in covered:
                    out.append({"row": ri, "col": ci, "rowspan": 1, "colspan": 1, "bbox": None})
        return out

    # ------------------------------------------------------------------
    # Strategy 2: bbox-based grid analysis
    # ------------------------------------------------------------------

    def _cells_as_dicts(self) -> List[Dict]:
        return [
            {"row": c.row, "col": c.col, "rowspan": c.rowspan, "colspan": c.colspan, "bbox": c.bbox}
            for c in self.cells_info
        ]

    def _analyze_with_bbox(self, cells: list[dict], nr: int, nc: int) -> list[dict]:
        h_set: set[float] = set()
        v_set: set[float] = set()
        for c in cells:
            bb = c.get("bbox")
            if bb and len(bb) >= 4:
                h_set.add(round(bb[1], 1))
                h_set.add(round(bb[3], 1))
                v_set.add(round(bb[0], 1))
                v_set.add(round(bb[2], 1))
        if len(h_set) < 2 or len(v_set) < 2:
            return []
        self._h_grid = self._cluster(list(h_set))
        self._v_grid = self._cluster(list(v_set))
        gr = len(self._h_grid) - 1
        gc = len(self._v_grid) - 1
        if gr < 1 or gc < 1:
            return []

        out: list[dict] = []
        covered: set[tuple[int, int]] = set()
        for c in cells:
            bb = c.get("bbox")
            if not bb:
                continue
            rs_idx = self._grid_index(bb[1], self._h_grid)
            re_idx = self._grid_index(bb[3], self._h_grid)
            cs_idx = self._grid_index(bb[0], self._v_grid)
            ce_idx = self._grid_index(bb[2], self._v_grid)
            if rs_idx is None or cs_idx is None:
                rs_idx, cs_idx = c["row"], c["col"]
                re_idx = rs_idx + c.get("rowspan", 1)
                ce_idx = cs_idx + c.get("colspan", 1)
            else:
                if re_idx is None or re_idx <= rs_idx:
                    re_idx = rs_idx + 1
                if ce_idx is None or ce_idx <= cs_idx:
                    ce_idx = cs_idx + 1
            dr = min(rs_idx, nr - 1)
            dc = min(cs_idx, nc - 1)
            rspan = min(max(1, re_idx - rs_idx), nr - dr)
            cspan = min(max(1, ce_idx - cs_idx), nc - dc)
            if (dr, dc) in covered:
                continue
            out.append({"row": dr, "col": dc, "rowspan": rspan, "colspan": cspan, "bbox": bb})
            for ri in range(dr, min(dr + rspan, nr)):
                for ci in range(dc, min(dc + cspan, nc)):
                    covered.add((ri, ci))

        for ri in range(nr):
            for ci in range(nc):
                if (ri, ci) not in covered:
                    out.append({"row": ri, "col": ci, "rowspan": 1, "colspan": 1, "bbox": None})
        return out

    # ------------------------------------------------------------------
    # Strategy 3: validate & enhance
    # ------------------------------------------------------------------

    def _validate_and_enhance(self, cells: list[dict], nr: int, nc: int) -> list[dict]:
        out: list[dict] = []
        covered: set[tuple[int, int]] = set()
        for c in cells:
            r, co = c.get("row", 0), c.get("col", 0)
            if r >= nr or co >= nc:
                continue
            rs = min(c.get("rowspan", 1), nr - r)
            cs = min(c.get("colspan", 1), nc - co)
            if (r, co) in covered:
                continue
            out.append({"row": r, "col": co, "rowspan": max(1, rs), "colspan": max(1, cs), "bbox": c.get("bbox")})
            for ri in range(r, min(r + rs, nr)):
                for ci in range(co, min(co + cs, nc)):
                    covered.add((ri, ci))
        for ri in range(nr):
            for ci in range(nc):
                if (ri, ci) not in covered:
                    out.append({"row": ri, "col": ci, "rowspan": 1, "colspan": 1, "bbox": None})
        return out

    # ------------------------------------------------------------------
    # Strategy 4: defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_cells(nr: int, nc: int) -> list[dict]:
        return [
            {"row": r, "col": c, "rowspan": 1, "colspan": 1, "bbox": None}
            for r in range(nr) for c in range(nc)
        ]

    # ------------------------------------------------------------------
    # Grid utilities
    # ------------------------------------------------------------------

    def _cluster(self, values: list[float]) -> list[float]:
        if not values:
            return []
        sv = sorted(values)
        clusters: list[list[float]] = [[sv[0]]]
        for v in sv[1:]:
            if v - clusters[-1][-1] <= self.GRID_TOLERANCE:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [sum(c) / len(c) for c in clusters]

    def _grid_index(self, val: float, grid: list[float]) -> Optional[int]:
        tol = self.GRID_TOLERANCE
        for i, g in enumerate(grid):
            if abs(val - g) <= tol:
                return i
        if grid:
            best = min(range(len(grid)), key=lambda i: abs(grid[i] - val))
            if abs(grid[best] - val) <= tol * 2:
                return best
        return None


__all__ = ["CellAnalysisEngine"]
