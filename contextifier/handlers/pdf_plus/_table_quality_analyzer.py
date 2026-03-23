# contextifier/handlers/pdf_plus/_table_quality_analyzer.py
"""
PDF Plus — Table Quality Analyzer.

Determines whether a table region can be **text-extracted** or requires
**image conversion** (block-image rendering).

Quality grades (from :class:`TableQuality`):
  EXCELLENT → must text-extract
  GOOD      → text-extract recommended
  MODERATE  → attempt text-extract, fall back to image
  POOR      → image conversion recommended
  UNPROCESSABLE → must use image

The analyser evaluates four dimensions (each 0–1 score):
  1. Border completeness  (weight 30 %)
  2. Grid regularity       (weight 30 %)
  3. Cell structure         (weight 20 %)
  4. Element simplicity     (weight 20 %)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    PdfPlusConfig,
    TableQuality,
    TableQualityResult,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class TableQualityAnalyzer:
    """Assess table regions for text-extraction feasibility."""

    def __init__(
        self,
        page: Any,
        page_num: int = 0,
    ) -> None:
        self.page = page
        self.page_num = page_num
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height
        self._drawings: Optional[list[dict]] = None
        self._text_dict: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def analyze_table(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> TableQualityResult:
        """Analyse a single table region."""
        drawings = self._get_region_drawings(bbox)
        issues: list[str] = []

        border_sc, bi = self._border_completeness(bbox, drawings)
        issues.extend(bi)
        grid_sc, gi = self._grid_regularity(bbox, drawings)
        issues.extend(gi)
        cell_sc, ci = self._cell_structure(bbox, drawings)
        issues.extend(ci)
        simple_sc, si = self._element_simplicity(drawings)
        issues.extend(si)

        total = (
            border_sc * CFG.QUALITY_WEIGHT_BORDER
            + grid_sc * CFG.QUALITY_WEIGHT_GRID
            + cell_sc * CFG.QUALITY_WEIGHT_CELL
            + simple_sc * CFG.QUALITY_WEIGHT_SIMPLE
        )

        quality = self._grade(total)
        text_ok = quality in (
            TableQuality.EXCELLENT,
            TableQuality.GOOD,
            TableQuality.MODERATE,
        )

        logger.debug(
            "[TQAnalyzer] %s quality=%s total=%.2f  border=%.2f grid=%.2f cell=%.2f simple=%.2f",
            bbox, quality.name, total, border_sc, grid_sc, cell_sc, simple_sc,
        )
        return TableQualityResult(
            bbox=bbox,
            quality=quality,
            score=total,
            border_completeness=border_sc,
            grid_regularity=grid_sc,
            cell_structure=cell_sc,
            no_complex_elements=simple_sc,
            text_extractable=text_ok,
            issues=issues,
        )

    def analyze_page_tables(self) -> Dict[str, Any]:
        """
        Analyse all table-candidate regions on the page.

        Returns a dict with *table_candidates*, *has_processable_tables*, *summary*.
        """
        table_regions = self._find_table_regions()
        results: list[dict] = []
        for bbox in table_regions:
            qr = self.analyze_table(bbox)
            results.append({
                "bbox": bbox,
                "quality": qr.quality,
                "score": qr.score,
                "is_processable": qr.text_extractable,
                "issues": qr.issues,
            })
        has_ok = any(r["is_processable"] for r in results)
        return {
            "table_candidates": results,
            "has_processable_tables": has_ok,
            "summary": {
                "total_candidates": len(results),
                "processable": sum(1 for r in results if r["is_processable"]),
                "unprocessable": sum(1 for r in results if not r["is_processable"]),
            },
        }

    # ------------------------------------------------------------------
    # Dimension analysers
    # ------------------------------------------------------------------

    def _border_completeness(
        self, bbox: tuple, drawings: list[dict]
    ) -> Tuple[float, list[str]]:
        issues: list[str] = []
        lines = self._extract_lines(drawings)
        if not lines:
            return 0.0, ["no_border_lines"]
        tol = CFG.QUALITY_BORDER_TOLERANCE
        x0, y0, x1, y1 = bbox
        sides: dict[str, bool] = {"top": False, "bottom": False, "left": False, "right": False}
        for ln in lines:
            if ln["is_horizontal"]:
                if abs(ln["y1"] - y0) <= tol:
                    sides["top"] = True
                elif abs(ln["y1"] - y1) <= tol:
                    sides["bottom"] = True
            if ln["is_vertical"]:
                if abs(ln["x1"] - x0) <= tol:
                    sides["left"] = True
                elif abs(ln["x1"] - x1) <= tol:
                    sides["right"] = True
        found = sum(sides.values())
        if found < 4:
            missing = [k for k, v in sides.items() if not v]
            issues.append(f"missing_borders({','.join(missing)})")
        return found / 4.0, issues

    def _grid_regularity(
        self, bbox: tuple, drawings: list[dict]
    ) -> Tuple[float, list[str]]:
        issues: list[str] = []
        lines = self._extract_lines(drawings)
        if not lines:
            return 0.0, ["no_grid_lines"]
        orth_cnt = sum(1 for l in lines if l["is_horizontal"] or l["is_vertical"])
        orth_ratio = orth_cnt / len(lines)
        if orth_ratio < CFG.QUALITY_MIN_ORTHOGONAL_RATIO:
            issues.append(f"non_orth({(1 - orth_ratio) * 100:.0f}%)")
        h_align = self._alignment([l["y1"] for l in lines if l["is_horizontal"]])
        v_align = self._alignment([l["x1"] for l in lines if l["is_vertical"]])
        alignment = (h_align + v_align) / 2
        if alignment < 0.8:
            issues.append("misaligned_grid")
        return orth_ratio * 0.6 + alignment * 0.4, issues

    def _cell_structure(
        self, bbox: tuple, drawings: list[dict]
    ) -> Tuple[float, list[str]]:
        issues: list[str] = []
        lines = self._extract_lines(drawings)
        h_sorted = sorted((l for l in lines if l["is_horizontal"]), key=lambda l: l["y1"])
        v_sorted = sorted((l for l in lines if l["is_vertical"]), key=lambda l: l["x1"])
        if len(h_sorted) < 2 or len(v_sorted) < 2:
            return 0.5, ["insufficient_lines"]
        heights = [h_sorted[i + 1]["y1"] - h_sorted[i]["y1"] for i in range(len(h_sorted) - 1) if h_sorted[i + 1]["y1"] > h_sorted[i]["y1"]]
        widths = [v_sorted[i + 1]["x1"] - v_sorted[i]["x1"] for i in range(len(v_sorted) - 1) if v_sorted[i + 1]["x1"] > v_sorted[i]["x1"]]
        tiny = sum(1 for h in heights if h < CFG.QUALITY_MIN_CELL_SIZE) + sum(1 for w in widths if w < CFG.QUALITY_MIN_CELL_SIZE)
        total_dims = len(heights) + len(widths)
        extreme_ratio = 0
        for h in heights:
            for w in widths:
                if h > 0 and w > 0 and max(h / w, w / h) > CFG.QUALITY_MAX_CELL_ASPECT_RATIO:
                    extreme_ratio += 1
        sc = 1.0
        if tiny and total_dims and tiny / total_dims > 0.1:
            issues.append("tiny_cells")
            sc -= 0.2
        if extreme_ratio:
            issues.append("extreme_aspect")
            sc -= 0.2
        return max(0.0, sc), issues

    def _element_simplicity(self, drawings: list[dict]) -> Tuple[float, list[str]]:
        if not drawings:
            return 1.0, []
        issues: list[str] = []
        curve_c = diag_c = fill_c = total_i = 0
        for d in drawings:
            items = d.get("items", [])
            total_i += len(items)
            for i in items:
                if i[0] == "c":
                    curve_c += 1
                elif i[0] == "l":
                    p1, p2 = i[1], i[2]
                    if abs(p2.x - p1.x) > CFG.QUALITY_LINE_ANGLE_TOLERANCE and abs(p2.y - p1.y) > CFG.QUALITY_LINE_ANGLE_TOLERANCE:
                        diag_c += 1
            if d.get("fill"):
                fill_c += 1
        ti = max(1, total_i)
        nd = max(1, len(drawings))
        cr, dr, fr = curve_c / ti, diag_c / ti, fill_c / nd
        if cr > CFG.QUALITY_MAX_CURVE_RATIO:
            issues.append(f"curves({cr * 100:.0f}%)")
        if dr > CFG.QUALITY_MAX_DIAGONAL_RATIO:
            issues.append(f"diagonals({dr * 100:.0f}%)")
        if fr > 0.5:
            issues.append("heavy_fills")
        sc = 1.0 - min(0.3, cr * 3) - min(0.3, dr * 3) - min(0.2, fr * 0.4)
        return max(0.0, sc), issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grade(self, total: float) -> TableQuality:
        if total >= CFG.QUALITY_EXCELLENT_THRESHOLD:
            return TableQuality.EXCELLENT
        if total >= CFG.QUALITY_GOOD_THRESHOLD:
            return TableQuality.GOOD
        if total >= CFG.QUALITY_MODERATE_THRESHOLD:
            return TableQuality.MODERATE
        if total >= CFG.QUALITY_POOR_THRESHOLD:
            return TableQuality.POOR
        return TableQuality.UNPROCESSABLE

    def _get_region_drawings(self, bbox: tuple) -> list[dict]:
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return [
            d for d in self._drawings
            if d.get("rect") and self._overlaps(bbox, tuple(d["rect"]))
        ]

    def _find_table_regions(self) -> list[tuple]:
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        lines = self._extract_lines(self._drawings)
        if not lines:
            return []
        # quick bbox of all lines
        xs = [l["x1"] for l in lines] + [l["x2"] for l in lines]
        ys = [l["y1"] for l in lines] + [l["y2"] for l in lines]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return [bbox] if w > 100 and h > 50 else []

    @staticmethod
    def _extract_lines(drawings: list[dict]) -> list[dict]:
        result: list[dict] = []
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
                    result.append({
                        "x1": min(x1, x2), "y1": min(y1, y2),
                        "x2": max(x1, x2), "y2": max(y1, y2),
                        "is_horizontal": abs(y2 - y1) <= 2.0,
                        "is_vertical": abs(x2 - x1) <= 2.0,
                        "length": ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5,
                    })
                elif item[0] == "re":
                    r = item[1]
                    x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
                    for a, b, c, dd, ih, iv in [
                        (x0, y0, x1, y0, True, False),
                        (x0, y1, x1, y1, True, False),
                        (x0, y0, x0, y1, False, True),
                        (x1, y0, x1, y1, False, True),
                    ]:
                        result.append({
                            "x1": a, "y1": b, "x2": c, "y2": dd,
                            "is_horizontal": ih, "is_vertical": iv,
                            "length": abs(c - a) + abs(dd - b),
                        })
        return result

    @staticmethod
    def _alignment(positions: list[float]) -> float:
        if len(positions) < 2:
            return 1.0
        sp = sorted(positions)
        clusters: list[list[float]] = [[sp[0]]]
        for p in sp[1:]:
            if p - clusters[-1][-1] <= 3.0:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        aligned = sum(len(c) for c in clusters if len(c) > 1)
        return aligned / len(positions) if positions else 1.0

    @staticmethod
    def _overlaps(a: tuple, b: tuple) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


__all__ = ["TableQualityAnalyzer"]
