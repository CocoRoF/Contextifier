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
        cjk_compat = self._count_cjk_compat(raw)
        fragmented = is_fragmented_text(raw)

        cjk_ratio = cjk_compat / total_chars
        garbled_ratio = (pua + cjk_compat) / total_chars
        quality = 1.0 - garbled_ratio

        needs_ocr = (
            quality < CFG.OCR_QUALITY_THRESHOLD
            or cjk_ratio >= CFG.CJK_COMPAT_RATIO_THRESHOLD
            or fragmented
        )

        details_parts: list[str] = []
        if pua:
            details_parts.append(f"pua={pua}")
        if cjk_compat:
            details_parts.append(f"cjk_compat={cjk_compat}")
        if garbled_ratio > 0.1:
            details_parts.append(f"garbled={garbled_ratio:.2%}")
        if fragmented:
            details_parts.append("fragmented")

        return TextQualityResult(
            quality_score=quality,
            total_chars=total_chars,
            pua_chars=pua,
            garbled_ratio=garbled_ratio,
            needs_ocr=needs_ocr,
            details=", ".join(details_parts) if details_parts else "ok",
            cjk_compat_chars=cjk_compat,
            is_fragmented=fragmented,
        )

    @staticmethod
    def _count_cjk_compat(text: str) -> int:
        """
        Count characters from the CJK Compatibility block.

        When a Word document is exported to PDF with a font the exporter
        cannot map, punctuation is liable to come out as squared unit symbols
        from this block — parentheses as ㏙/㏚, a range dash as ㏊.

        These are counted, never substituted. The block holds legitimate unit
        abbreviations used throughout Korean and Japanese technical writing,
        and the neighbouring CJK Extension A block that doc2chunk also
        rewrites is ordinary Hanja: mapping U+3711 (㜑) to an arrow, as its
        table does, corrupts any document that genuinely uses the character.
        A high density is evidence the page needs re-reading, which OCR does
        without having to guess what each character was supposed to be.
        """
        lo, hi = CFG.CJK_COMPAT_RANGE
        return sum(1 for ch in text if lo <= ord(ch) <= hi)

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
# Fragmented text
# ======================================================================


def is_fragmented_text(text: str) -> bool:
    """
    Report whether a page's text layer has collapsed to one character a line.

    Some producers — Word text boxes and vertical-text frames in particular —
    export each glyph as its own text object, and the extractor then has no
    grounds to join them, so ``현재 시장`` arrives as five lines of one
    character.

    The test is the share of lines holding almost nothing, not the average
    line length: a table of contents, a bulleted list or a column of figures
    has a low average and is perfectly intact. Rewriting one of those would be
    worse than leaving a genuinely fragmented page alone.
    """
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    if len(lines) < CFG.FRAGMENT_MIN_LINES:
        return False

    tiny = sum(1 for line in lines if len(line) <= CFG.FRAGMENT_LINE_MAX_CHARS)
    return tiny / len(lines) > CFG.FRAGMENT_LINE_RATIO


class FragmentedTextReconstructor:
    """
    Rebuild lines from character positions when the text layer is fragmented.

    Works from ``rawdict``, which reports a bounding box per glyph: characters
    sharing a baseline are one line, and their order along that line is their
    order on the page. No guessing is involved — the geometry the PDF already
    carries is what reassembles the text.
    """

    def __init__(
        self,
        page: Any,
        page_num: int = 0,
        *,
        y_tolerance: float = CFG.FRAGMENT_Y_TOLERANCE,
        exclude_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> None:
        self.page = page
        self.page_num = page_num
        self.y_tolerance = y_tolerance
        self.exclude_bboxes = exclude_bboxes or []

    def reconstruct(self) -> str:
        """Return the rebuilt text, or ``""`` when nothing could be read."""
        try:
            chars = self._collect_chars()
        except Exception as exc:
            logger.warning(
                "[Reconstruct] page %d failed: %s", self.page_num + 1, exc
            )
            return ""

        if not chars:
            return ""

        lines = self._group_into_lines(chars)
        text = "\n".join(lines)
        logger.info(
            "[Reconstruct] page %d: %d characters → %d lines",
            self.page_num + 1,
            len(chars),
            len(lines),
        )
        return text

    # ── internals ─────────────────────────────────────────────────────

    def _collect_chars(self) -> List[Tuple[float, float, str]]:
        """``(baseline_y, x, character)`` for every glyph outside the exclusions."""
        raw = self.page.get_text("rawdict") or {}
        chars: List[Tuple[float, float, str]] = []

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    for char in span.get("chars", []):
                        bbox = char.get("bbox")
                        text = char.get("c", "")
                        # Spaces are kept: they separate words inside a line.
                        # Lines that end up blank are dropped when rendered.
                        if not bbox or not text:
                            continue
                        if self._is_excluded(bbox):
                            continue
                        # Use the vertical centre: a descender would otherwise
                        # place its glyph on a different line from its
                        # neighbours.
                        chars.append(((bbox[1] + bbox[3]) / 2, bbox[0], text))

        return chars

    def _is_excluded(self, bbox: Tuple[float, float, float, float]) -> bool:
        for ex in self.exclude_bboxes:
            if (
                bbox[0] >= ex[0]
                and bbox[1] >= ex[1]
                and bbox[2] <= ex[2]
                and bbox[3] <= ex[3]
            ):
                return True
        return False

    def _group_into_lines(self, chars: List[Tuple[float, float, str]]) -> List[str]:
        """Group glyphs onto shared baselines, then order each line by x."""
        lines: List[List[Tuple[float, str]]] = []
        baselines: List[float] = []

        for y, x, text in sorted(chars, key=lambda c: (c[0], c[1])):
            for index, baseline in enumerate(baselines):
                if abs(y - baseline) <= self.y_tolerance:
                    lines[index].append((x, text))
                    break
            else:
                baselines.append(y)
                lines.append([(x, text)])

        rendered: List[str] = []
        for line in lines:
            line.sort(key=lambda item: item[0])
            joined = "".join(text for _, text in line).strip()
            if joined:
                rendered.append(joined)
        return rendered


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
            logger.warning(
                "[OCRFallback] page %d OCR failed: %s", self.page_num + 1, exc
            )
            return ""


# ======================================================================
# Quality-Aware Text Extractor
# ======================================================================


class QualityAwareTextExtractor:
    """
    Extract page text, automatically switching to OCR when quality is
    below ``OCR_QUALITY_THRESHOLD``.
    """

    def __init__(
        self,
        page: Any,
        page_num: int = 0,
        *,
        exclude_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> None:
        self.page = page
        self.page_num = page_num
        self.exclude_bboxes = exclude_bboxes or []
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

        # A fragmented page has all its characters; only the line grouping was
        # lost. Rebuilding from glyph positions recovers the text exactly,
        # where OCR would re-read it and can only be as good as the render.
        if qr.is_fragmented:
            rebuilt = FragmentedTextReconstructor(
                self.page,
                self.page_num,
                exclude_bboxes=self.exclude_bboxes,
            ).reconstruct()
            if rebuilt.strip():
                return PageTextAnalysis(text=rebuilt, quality=qr, used_ocr=False)

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
    "FragmentedTextReconstructor",
    "is_fragmented_text",
    "PageOCRFallbackEngine",
    "QualityAwareTextExtractor",
]
