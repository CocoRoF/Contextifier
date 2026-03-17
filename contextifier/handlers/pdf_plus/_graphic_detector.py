# contextifier_new/handlers/pdf_plus/_graphic_detector.py
"""
PDF Plus — Graphic Region Detector.

Detects vector-graphic regions (charts, diagrams, icons) in PDF pages so they
can be **excluded** from table-detection candidates.

Approach:
  1. Cluster nearby drawing primitives into regions.
  2. Score each region on 8 criteria (curve ratio, fill count, colour
     diversity, chart pattern, rectangle-only penalty, …).
  3. Regions scoring ≥ 0.5 are flagged as graphics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    GraphicRegionInfo,
    PdfPlusConfig,
)
from contextifier.handlers.pdf_plus._utils import calculate_overlap_ratio

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class GraphicRegionDetector:
    """Detect graphical (non-table) regions on a single PDF page."""

    _MERGE_MARGIN: float = 20.0  # px distance to merge neighbouring regions

    def __init__(self, page: Any, page_num: int) -> None:
        self.page = page
        self.page_num = page_num
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height
        self.graphic_regions: list[GraphicRegionInfo] = []
        self._drawings_cache: Optional[list[dict]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> List[GraphicRegionInfo]:
        """Run detection and return confirmed graphic regions."""
        drawings = self._get_drawings()
        if not drawings:
            return []

        raw_regions = self._cluster_drawings(drawings)
        for r in raw_regions:
            self._analyze_region(r)

        self.graphic_regions = [r for r in raw_regions if r.is_graphic]
        logger.debug(
            "[GraphicDetector] Page %d: %d graphic regions",
            self.page_num + 1,
            len(self.graphic_regions),
        )
        return self.graphic_regions

    def is_bbox_in_graphic_region(
        self,
        bbox: Tuple[float, float, float, float],
        threshold: float = 0.3,
    ) -> bool:
        """Return *True* if *bbox* overlaps a graphic region by ≥ *threshold*."""
        return any(
            calculate_overlap_ratio(bbox, g.bbox) >= threshold
            for g in self.graphic_regions
        )

    # ------------------------------------------------------------------
    # Drawing clustering
    # ------------------------------------------------------------------

    def _get_drawings(self) -> list[dict]:
        if self._drawings_cache is None:
            self._drawings_cache = self.page.get_drawings()
        return self._drawings_cache

    def _cluster_drawings(self, drawings: list[dict]) -> list[GraphicRegionInfo]:
        raw: list[dict] = []
        for d in drawings:
            rect = d.get("rect")
            if rect is None or rect.is_empty or rect.is_infinite:
                continue
            items = d.get("items", [])
            fill = d.get("fill")
            stroke = d.get("color")

            rd: Dict[str, Any] = {
                "bbox": tuple(rect),
                "curve_count": sum(1 for i in items if i[0] == "c"),
                "line_count": sum(1 for i in items if i[0] == "l"),
                "rect_count": sum(1 for i in items if i[0] == "re"),
                "fill_count": 1 if fill else 0,
                "colors": set(),
            }
            if fill:
                rd["colors"].add(tuple(fill) if isinstance(fill, (list, tuple)) else fill)
            if stroke:
                rd["colors"].add(tuple(stroke) if isinstance(stroke, (list, tuple)) else stroke)

            merged = False
            for existing in raw:
                if self._should_merge(existing["bbox"], rd["bbox"]):
                    self._merge_data(existing, rd)
                    merged = True
                    break
            if not merged:
                raw.append(rd)

        raw = self._iterative_merge(raw)

        return [
            GraphicRegionInfo(
                bbox=r["bbox"],
                curve_count=r["curve_count"],
                line_count=r["line_count"],
                rect_count=r["rect_count"],
                fill_count=r["fill_count"],
                color_count=len(r["colors"]),
                is_graphic=False,
                confidence=0.0,
            )
            for r in raw
        ]

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    def _should_merge(self, b1: tuple, b2: tuple) -> bool:
        m = self._MERGE_MARGIN
        return (
            b1[0] - m <= b2[2]
            and b1[2] + m >= b2[0]
            and b1[1] - m <= b2[3]
            and b1[3] + m >= b2[1]
        )

    @staticmethod
    def _merge_data(target: dict, source: dict) -> None:
        target["bbox"] = (
            min(target["bbox"][0], source["bbox"][0]),
            min(target["bbox"][1], source["bbox"][1]),
            max(target["bbox"][2], source["bbox"][2]),
            max(target["bbox"][3], source["bbox"][3]),
        )
        target["curve_count"] += source["curve_count"]
        target["line_count"] += source["line_count"]
        target["rect_count"] += source["rect_count"]
        target["fill_count"] += source["fill_count"]
        target["colors"].update(source["colors"])

    def _iterative_merge(self, regions: list[dict], max_iter: int = 5) -> list[dict]:
        for _ in range(max_iter):
            changed = False
            new: list[dict] = []
            used: set[int] = set()
            for i, r1 in enumerate(regions):
                if i in used:
                    continue
                cur = {**r1, "colors": set(r1["colors"])}
                for j, r2 in enumerate(regions):
                    if j <= i or j in used:
                        continue
                    if self._should_merge(cur["bbox"], r2["bbox"]):
                        self._merge_data(cur, r2)
                        used.add(j)
                        changed = True
                new.append(cur)
            regions = new
            if not changed:
                break
        return regions

    # ------------------------------------------------------------------
    # Region scoring (8-point system)
    # ------------------------------------------------------------------

    def _analyze_region(self, region: GraphicRegionInfo) -> None:
        total = region.curve_count + region.line_count + region.rect_count
        if total == 0:
            region.is_graphic = False
            region.confidence = 0.0
            return

        score = 0.0
        reasons: list[str] = []

        # 1. Curve ratio
        cratio = region.curve_count / total
        if cratio >= CFG.GRAPHIC_CURVE_RATIO_THRESHOLD:
            score += 0.4
            reasons.append(f"curve_ratio={cratio:.2f}")

        # 2. Minimum curve count
        if region.curve_count >= CFG.GRAPHIC_MIN_CURVE_COUNT:
            score += 0.2
            reasons.append(f"curves={region.curve_count}")

        # 3. Fill ratio (rough)
        fill_ratio = region.fill_count / max(1, total // 10)
        if fill_ratio >= CFG.GRAPHIC_FILL_RATIO_THRESHOLD:
            score += 0.2
            reasons.append(f"fills={region.fill_count}")

        # 4. Colour diversity
        if region.color_count >= CFG.GRAPHIC_COLOR_VARIETY_THRESHOLD:
            score += 0.2
            reasons.append(f"colors={region.color_count}")

        # 5. Chart pattern (curves + fills)
        if region.curve_count >= 5 and region.fill_count >= 3:
            score += 0.3
            reasons.append(f"chart_pattern(c={region.curve_count},f={region.fill_count})")

        # 6. Only-rectangles penalty (table cells, not graphics)
        if region.rect_count >= 5 and region.curve_count == 0 and region.line_count == 0:
            if region.color_count >= 3:
                score += 0.2
                reasons.append(f"colored_rects({region.rect_count})")
            else:
                score -= 0.3
                reasons.append("likely_table_cells")

        # 7. Full-page background exclusion
        bw = region.bbox[2] - region.bbox[0]
        bh = region.bbox[3] - region.bbox[1]
        if bw > self.page_width * 0.9 and bh > self.page_height * 0.9:
            score = 0.0
            reasons = ["page_background"]

        # 8. Tiny region penalty
        if bw * bh < 500:
            score *= 0.5

        region.confidence = max(0.0, min(1.0, score))
        region.is_graphic = score >= 0.5
        region.reason = ", ".join(reasons) if reasons else "not_graphic"


__all__ = ["GraphicRegionDetector"]
