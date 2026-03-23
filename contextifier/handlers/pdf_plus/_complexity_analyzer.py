# contextifier/handlers/pdf_plus/_complexity_analyzer.py
"""
PDF Plus — Page Complexity Analyzer.

Examines drawing density, image density, text quality, and layout
complexity of a PDF page and maps the result to a
``ProcessingStrategy``:

============  ========  ====================================
Level         Score     Strategy
============  ========  ====================================
SIMPLE        < 0.35    TEXT_EXTRACTION
MODERATE      0.35–0.65 HYBRID (text + partial OCR)
COMPLEX       0.65–0.90 HYBRID (heavy text, selective block)
EXTREME       ≥ 0.90    FULL_PAGE_OCR (with smart blocks)
============  ========  ====================================

Weighted formula (4 dimensions):
    ``0.30 × drawing + 0.20 × image + 0.25 × text_badness + 0.25 × layout``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    ComplexityLevel,
    PageComplexity,
    PdfPlusConfig,
    ProcessingStrategy,
    RegionComplexity,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class ComplexityAnalyzer:
    """
    Analyse a single page and return ``PageComplexity``.

    Usage::

        analyzer = ComplexityAnalyzer(page, page_num)
        complexity = analyzer.analyze()
        strategy = complexity.recommended_strategy
    """

    def __init__(self, page: Any, page_num: int) -> None:
        self.page = page
        self.page_num = page_num
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height
        self.page_area: float = self.page_width * self.page_height

        # lazy caches
        self._drawings: Optional[List[Dict]] = None
        self._text_dict: Optional[Dict] = None
        self._images: Optional[list] = None

    # ─────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────

    def analyze(self) -> PageComplexity:
        drawings = self._get_drawings()
        text_dict = self._get_text_dict()
        images = self._get_images()

        text_blocks = [
            b for b in text_dict.get("blocks", []) if b.get("type") == 0
        ]

        column_count = self._count_columns(text_blocks)
        d_score = self._drawing_score(drawings)
        i_score = self._image_score(images)
        t_quality = self._text_quality(text_blocks)
        l_score = self._layout_score(column_count, text_blocks)

        overall = self._weighted_score(d_score, i_score, t_quality, l_score)
        level = self._level(overall)

        # region-wise
        regions = self._region_analysis(drawings, text_blocks, images)
        complex_regions = [
            r.bbox for r in regions
            if r.complexity_level in (ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME)
        ]

        strategy = self._strategy(level, overall, t_quality, complex_regions)

        result = PageComplexity(
            page_num=self.page_num,
            page_size=(self.page_width, self.page_height),
            overall_complexity=level,
            overall_score=overall,
            regions=regions,
            complex_regions=complex_regions,
            total_drawings=len(drawings),
            total_images=len(images),
            total_text_blocks=len(text_blocks),
            column_count=column_count,
            recommended_strategy=strategy,
        )
        logger.debug(
            "[ComplexityAnalyzer] Page %d: %s (%.2f) → %s  cols=%d",
            self.page_num + 1, level.name, overall, strategy.name, column_count,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Dimension scores
    # ─────────────────────────────────────────────────────────────────────

    def _drawing_score(self, drawings: List[Dict]) -> float:
        if not drawings:
            return 0.0
        items_total = 0
        curves = 0
        fills = 0
        for d in drawings:
            items = d.get("items", [])
            items_total += len(items)
            curves += sum(1 for it in items if it[0] == "c")
            if d.get("fill"):
                fills += 1
        density = items_total / (self.page_area / 1000) if self.page_area else 0.0
        curve_r = curves / max(1, items_total)
        fill_r = fills / max(1, len(drawings))

        if density >= CFG.DRAWING_DENSITY_EXTREME:
            base = 1.0
        elif density >= CFG.DRAWING_DENSITY_COMPLEX:
            base = 0.7
        elif density >= CFG.DRAWING_DENSITY_MODERATE:
            base = 0.4
        else:
            base = density / max(0.001, CFG.DRAWING_DENSITY_MODERATE) * 0.4
        return min(1.0, base + curve_r * 0.2 + fill_r * 0.1)

    def _image_score(self, images: list) -> float:
        if not images:
            return 0.0
        density = len(images) / (self.page_area / 10000) if self.page_area else 0
        if density >= CFG.IMAGE_DENSITY_EXTREME:
            return 1.0
        if density >= CFG.IMAGE_DENSITY_COMPLEX:
            return 0.7
        if density >= CFG.IMAGE_DENSITY_MODERATE:
            return 0.4
        return density / max(0.001, CFG.IMAGE_DENSITY_MODERATE) * 0.4

    def _text_quality(self, text_blocks: List[Dict]) -> float:
        """Return 0.0 (broken) → 1.0 (clean)."""
        if not text_blocks:
            return 1.0
        total = 0
        bad = 0
        for block in text_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    total += len(text)
                    for ch in text:
                        cp = ord(ch)
                        if CFG.PUA_START <= cp <= CFG.PUA_END:
                            bad += 1
                        elif CFG.PUA_SUPP_START <= cp <= CFG.PUA_SUPP_END:
                            bad += 1
        return 1.0 - bad / max(1, total)

    def _layout_score(self, col_count: int, text_blocks: List[Dict]) -> float:
        if col_count >= CFG.COLUMN_COUNT_EXTREME:
            score = 0.95
        elif col_count >= CFG.COLUMN_COUNT_COMPLEX:
            score = 0.75
        elif col_count >= CFG.COLUMN_COUNT_MODERATE:
            score = 0.50
        elif col_count >= 2:
            score = 0.30
        else:
            score = 0.0

        # additional multi-column evidence from Y-position overlap
        if text_blocks:
            ys = [b.get("bbox", (0, 0, 0, 0))[1] for b in text_blocks]
            unique_y = len({int(y / 10) for y in ys})
            if unique_y < len(text_blocks) * 0.5 and len(text_blocks) > 5:
                score = max(score, 0.6)
        return min(1.0, score)

    # ─────────────────────────────────────────────────────────────────────
    # Weighted overall
    # ─────────────────────────────────────────────────────────────────────

    def _weighted_score(
        self, drawing: float, image: float, text_quality: float, layout: float,
    ) -> float:
        if layout >= 0.95:
            return 0.90  # cap — other factors needed for EXTREME
        text_bad = 1.0 - text_quality
        return min(
            1.0,
            drawing * CFG.WEIGHT_DRAWING
            + image * CFG.WEIGHT_IMAGE
            + text_bad * CFG.WEIGHT_TEXT
            + layout * CFG.WEIGHT_LAYOUT,
        )

    @staticmethod
    def _level(score: float) -> ComplexityLevel:
        if score >= CFG.COMPLEXITY_EXTREME:
            return ComplexityLevel.EXTREME
        if score >= CFG.COMPLEXITY_COMPLEX:
            return ComplexityLevel.COMPLEX
        if score >= CFG.COMPLEXITY_MODERATE:
            return ComplexityLevel.MODERATE
        return ComplexityLevel.SIMPLE

    # ─────────────────────────────────────────────────────────────────────
    # Region-wise analysis
    # ─────────────────────────────────────────────────────────────────────

    def _region_analysis(
        self,
        drawings: List[Dict],
        text_blocks: List[Dict],
        images: list,
    ) -> List[RegionComplexity]:
        grid = CFG.REGION_GRID_SIZE
        regions: List[RegionComplexity] = []
        for y in range(0, int(self.page_height), grid):
            for x in range(0, int(self.page_width), grid):
                x0, y0 = float(x), float(y)
                x1 = min(x0 + grid, self.page_width)
                y1 = min(y0 + grid, self.page_height)
                bbox = (x0, y0, x1, y1)
                area = (x1 - x0) * (y1 - y0)

                rd = [
                    d for d in drawings
                    if d.get("rect") and self._overlaps(bbox, tuple(d["rect"]))
                ]
                rt = [
                    b for b in text_blocks
                    if self._overlaps(bbox, b.get("bbox", (0, 0, 0, 0)))
                ]

                dd = len(rd) / (area / 1000) if area else 0
                tq = self._text_quality(rt)
                score = min(1.0, dd / 3.0 + (1.0 - tq) * 0.5)

                if score >= 0.7:
                    lev = ComplexityLevel.COMPLEX
                    strat = ProcessingStrategy.BLOCK_IMAGE_OCR
                elif score >= 0.4:
                    lev = ComplexityLevel.MODERATE
                    strat = (
                        ProcessingStrategy.HYBRID
                        if tq < CFG.TEXT_QUALITY_POOR
                        else ProcessingStrategy.TEXT_EXTRACTION
                    )
                else:
                    lev = ComplexityLevel.SIMPLE
                    strat = ProcessingStrategy.TEXT_EXTRACTION

                regions.append(RegionComplexity(
                    bbox=bbox,
                    complexity_level=lev,
                    complexity_score=score,
                    drawing_density=dd,
                    text_quality=tq,
                    recommended_strategy=strat,
                ))
        return regions

    # ─────────────────────────────────────────────────────────────────────
    # Strategy mapping
    # ─────────────────────────────────────────────────────────────────────

    def _strategy(
        self,
        level: ComplexityLevel,
        score: float,
        text_quality: float,
        complex_regions: List[Tuple[float, float, float, float]],
    ) -> ProcessingStrategy:
        # very bad text → full OCR
        if text_quality < 0.4:
            return ProcessingStrategy.FULL_PAGE_OCR

        # extreme + poor text → full OCR
        if level == ComplexityLevel.EXTREME and text_quality < 0.6:
            return ProcessingStrategy.FULL_PAGE_OCR

        # large complex area + poor text → full OCR
        if complex_regions:
            total_area = sum(
                (r[2] - r[0]) * (r[3] - r[1]) for r in complex_regions
            )
            if total_area / max(1, self.page_area) > 0.5 and text_quality < 0.7:
                return ProcessingStrategy.FULL_PAGE_OCR

        if level == ComplexityLevel.COMPLEX:
            return ProcessingStrategy.HYBRID

        if level == ComplexityLevel.MODERATE:
            return ProcessingStrategy.HYBRID

        return ProcessingStrategy.TEXT_EXTRACTION

    # ─────────────────────────────────────────────────────────────────────
    # Column counting
    # ─────────────────────────────────────────────────────────────────────

    def _count_columns(self, text_blocks: List[Dict]) -> int:
        if not text_blocks:
            return 1
        xs = sorted(b.get("bbox", (0, 0, 0, 0))[0] for b in text_blocks)
        if not xs:
            return 1
        tol = CFG.COLUMN_CLUSTER_TOLERANCE
        cols: List[List[float]] = []
        cur = [xs[0]]
        for x in xs[1:]:
            if x - cur[-1] < tol:
                cur.append(x)
            else:
                cols.append(cur)
                cur = [x]
        cols.append(cur)
        return len(cols)

    # ─────────────────────────────────────────────────────────────────────
    # Caching & helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_drawings(self) -> List[Dict]:
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings

    def _get_text_dict(self) -> Dict:
        if self._text_dict is None:
            self._text_dict = self.page.get_text("dict", sort=True)
        return self._text_dict

    def _get_images(self) -> list:
        if self._images is None:
            self._images = self.page.get_images()
        return self._images

    @staticmethod
    def _overlaps(a: Tuple, b: Tuple) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


__all__ = ["ComplexityAnalyzer"]
