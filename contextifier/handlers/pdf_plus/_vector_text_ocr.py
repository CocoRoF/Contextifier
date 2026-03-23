# contextifier/handlers/pdf_plus/_vector_text_ocr.py
"""
PDF Plus — Vector / Outlined Text OCR Engine.

Some PDFs embed text as vector outlines (paths) rather than as a text
layer.  This module detects such regions by looking for clusters of
small filled paths with no extractable text, then renders those regions
to images and OCRs them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    PdfPlusConfig,
    VectorTextRegion,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class VectorTextOCREngine:
    """Detect outlined text → render → OCR."""

    def __init__(self, page: Any, page_num: int = 0) -> None:
        self.page = page
        self.page_num = page_num
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect_and_ocr(self) -> List[VectorTextRegion]:
        """
        Detect vector-text regions and OCR them.

        Returns a list of :class:`VectorTextRegion` with ``ocr_text``
        populated.
        """
        regions = self._detect_regions()
        if not regions:
            return []

        for region in regions:
            region.ocr_text = self._ocr_region(region.bbox)

        # Filter out empty results
        return [r for r in regions if r.ocr_text and r.ocr_text.strip()]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_regions(self) -> List[VectorTextRegion]:
        """
        Find clusters of small filled paths that lack extractable text.
        """
        drawings = self.page.get_drawings()
        if not drawings:
            return []

        # Collect small filled shapes
        small_filled: list[tuple] = []
        for d in drawings:
            rect = d.get("rect")
            if rect is None:
                continue
            w = rect.x1 - rect.x0
            h = rect.y1 - rect.y0
            area = w * h
            # Small shapes (<400 pt²) with fill → possibly outlined glyphs
            if area < CFG.VECTOR_TEXT_MAX_GLYPH_AREA and d.get("fill"):
                small_filled.append(tuple(rect))

        if len(small_filled) < CFG.VECTOR_TEXT_MIN_GLYPH_CLUSTER:
            return []

        # Cluster nearby shapes
        clusters = self._cluster_bboxes(small_filled)

        regions: list[VectorTextRegion] = []
        for cluster in clusters:
            if len(cluster) < CFG.VECTOR_TEXT_MIN_GLYPH_CLUSTER:
                continue
            # Compute region bbox
            x0 = min(b[0] for b in cluster)
            y0 = min(b[1] for b in cluster)
            x1 = max(b[2] for b in cluster)
            y1 = max(b[3] for b in cluster)
            region_bbox = (x0, y0, x1, y1)

            # Check if there's already extractable text in this region
            text = self.page.get_text("text", clip=region_bbox) or ""
            if text.strip():
                continue  # already has real text

            regions.append(VectorTextRegion(
                bbox=region_bbox,
                glyph_count=len(cluster),
            ))

        return regions

    def _cluster_bboxes(
        self, bboxes: list[tuple], margin: float = 10.0
    ) -> list[list[tuple]]:
        """Cluster bboxes that are within *margin* of each other."""
        clusters: list[list[tuple]] = []
        used: set[int] = set()

        for i, b1 in enumerate(bboxes):
            if i in used:
                continue
            cluster = [b1]
            used.add(i)
            queue = [b1]
            while queue:
                cur = queue.pop()
                for j, b2 in enumerate(bboxes):
                    if j in used:
                        continue
                    if self._nearby(cur, b2, margin):
                        cluster.append(b2)
                        used.add(j)
                        queue.append(b2)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _nearby(a: tuple, b: tuple, m: float) -> bool:
        return not (
            a[2] + m < b[0]
            or b[2] + m < a[0]
            or a[3] + m < b[1]
            or b[3] + m < a[1]
        )

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_region(self, bbox: Tuple[float, float, float, float]) -> str:
        """Render the region as an image and OCR it."""
        try:
            import fitz
            import pytesseract
            from PIL import Image
            from io import BytesIO

            clip = fitz.Rect(bbox)
            mat = fitz.Matrix(CFG.BLOCK_IMAGE_DPI / 72, CFG.BLOCK_IMAGE_DPI / 72)
            pix = self.page.get_pixmap(matrix=mat, clip=clip)
            img = Image.open(BytesIO(pix.tobytes("png")))
            text: str = pytesseract.image_to_string(img, lang=CFG.OCR_LANGUAGE)
            return text.strip()
        except ImportError:
            logger.debug("[VectorOCR] pytesseract/Pillow not installed")
            return ""
        except Exception as exc:
            logger.debug("[VectorOCR] OCR failed for %s: %s", bbox, exc)
            return ""


__all__ = ["VectorTextOCREngine"]
