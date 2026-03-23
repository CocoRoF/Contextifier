# contextifier/handlers/pdf_plus/_text_quality_analyzer.py
"""
PDF Plus — Text Quality Analyzer & OCR Fallback.

Three cooperating components:

* :class:`TextQualityAnalyzer`
    – Per-page text quality evaluation (PUA detection, garbled-text
      heuristics).

* :class:`PageOCRFallbackEngine`
    – When text quality is below threshold, render the page as an image
      and OCR it via ``pytesseract`` (``kor+eng``).

* :class:`QualityAwareTextExtractor`
    – Convenience wrapper: extract text from a page, automatically
      falling back to OCR if the measured quality is poor.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    PdfPlusConfig,
    TextQualityResult,
    PageTextAnalysis,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


# ======================================================================
# Text Quality Analyzer
# ======================================================================

class TextQualityAnalyzer:
    """Measure extractable-text quality on a single PDF page."""

    PUA_RANGES: list[tuple[int, int]] = CFG.PUA_RANGES

    def __init__(self, page: Any, page_num: int = 0) -> None:
        self.page = page
        self.page_num = page_num

    def analyze(self) -> TextQualityResult:
        """Return a :class:`TextQualityResult`."""
        raw = self.page.get_text("text") or ""
        total_chars = len(raw)
        if total_chars == 0:
            return TextQualityResult(
                quality_score=0.0,
                total_chars=0,
                pua_chars=0,
                garbled_ratio=0.0,
                needs_ocr=True,
                details="empty_page",
            )

        pua = self._count_pua(raw)
        garbled_ratio = pua / total_chars if total_chars else 0.0
        quality = 1.0 - garbled_ratio
        needs_ocr = quality < CFG.OCR_QUALITY_THRESHOLD

        details_parts: list[str] = []
        if pua:
            details_parts.append(f"pua={pua}")
        if garbled_ratio > 0.1:
            details_parts.append(f"garbled={garbled_ratio:.2%}")

        return TextQualityResult(
            quality_score=quality,
            total_chars=total_chars,
            pua_chars=pua,
            garbled_ratio=garbled_ratio,
            needs_ocr=needs_ocr,
            details=", ".join(details_parts) if details_parts else "ok",
        )

    def _count_pua(self, text: str) -> int:
        count = 0
        for ch in text:
            cp = ord(ch)
            for lo, hi in self.PUA_RANGES:
                if lo <= cp <= hi:
                    count += 1
                    break
        return count


# ======================================================================
# Page OCR Fallback
# ======================================================================

class PageOCRFallbackEngine:
    """Render page → image → pytesseract OCR."""

    OCR_LANG: str = CFG.OCR_LANGUAGE
    OCR_DPI: int = CFG.BLOCK_IMAGE_DPI  # reuse block-image resolution

    def __init__(self, page: Any, page_num: int = 0) -> None:
        self.page = page
        self.page_num = page_num

    def ocr(self) -> str:
        """Return OCR text for the full page, or empty string on failure."""
        try:
            import pytesseract
            from PIL import Image
            from io import BytesIO

            mat = self.page.get_pixmap(dpi=self.OCR_DPI)
            img = Image.open(BytesIO(mat.tobytes("png")))
            text: str = pytesseract.image_to_string(img, lang=self.OCR_LANG)
            return text.strip()
        except ImportError:
            logger.debug("[OCRFallback] pytesseract / Pillow not installed")
            return ""
        except Exception as exc:
            logger.warning("[OCRFallback] page %d OCR failed: %s", self.page_num + 1, exc)
            return ""


# ======================================================================
# Quality-Aware Text Extractor
# ======================================================================

class QualityAwareTextExtractor:
    """
    Extract page text, automatically switching to OCR when quality is
    below ``OCR_QUALITY_THRESHOLD``.
    """

    def __init__(self, page: Any, page_num: int = 0) -> None:
        self.page = page
        self.page_num = page_num
        self._quality: Optional[TextQualityResult] = None

    @property
    def quality_result(self) -> TextQualityResult:
        if self._quality is None:
            self._quality = TextQualityAnalyzer(self.page, self.page_num).analyze()
        return self._quality

    def extract(self) -> PageTextAnalysis:
        """
        Returns a :class:`PageTextAnalysis` with the best available text
        and quality metadata.
        """
        qr = self.quality_result
        if qr.needs_ocr:
            ocr_text = PageOCRFallbackEngine(self.page, self.page_num).ocr()
            if ocr_text:
                return PageTextAnalysis(
                    text=ocr_text,
                    quality=qr,
                    used_ocr=True,
                )
        # Direct extraction
        text = self.page.get_text("text") or ""
        return PageTextAnalysis(
            text=text.strip(),
            quality=qr,
            used_ocr=False,
        )


__all__ = [
    "TextQualityAnalyzer",
    "PageOCRFallbackEngine",
    "QualityAwareTextExtractor",
]
