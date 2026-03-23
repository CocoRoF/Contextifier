# contextifier/handlers/pdf_plus/_text_extractor.py
"""
PDF Plus — Text Block Extractor.

Extracts **positioned** text blocks from a PDF page, excluding regions
already covered by detected tables.  Falls back to quality-aware
extraction (including OCR) when the native text layer is poor.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

from contextifier.handlers.pdf_plus._types import (
    ElementType,
    PageElement,
    PdfPlusConfig,
)
from contextifier.handlers.pdf_plus._utils import (
    escape_html,
    is_inside_any_bbox,
)
from contextifier.handlers.pdf_plus._text_quality_analyzer import (
    QualityAwareTextExtractor,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


def extract_text_blocks(
    page: Any,
    page_num: int,
    table_bboxes: List[Tuple[float, float, float, float]],
    *,
    overlap_threshold: float = 0.5,
) -> List[PageElement]:
    """
    Extract text blocks from *page*, skipping those that overlap with
    *table_bboxes*.

    Returns a list of :class:`PageElement` (type ``TEXT``), each
    carrying its y/x position for later element merging.
    """
    elements: list[PageElement] = []

    # 1. Try structured extraction (get_text("dict"))
    pd = page.get_text("dict", sort=True)
    blocks = pd.get("blocks", [])

    text_found = False
    for blk in blocks:
        if blk.get("type") != 0:
            continue  # skip images in dict output
        bb = blk.get("bbox", (0, 0, 0, 0))

        # Skip blocks inside table regions
        if is_inside_any_bbox(bb, table_bboxes, threshold=overlap_threshold):
            continue

        # Assemble text from spans
        lines: list[str] = []
        for ln in blk.get("lines", []):
            spans_text = "".join(sp.get("text", "") for sp in ln.get("spans", []))
            if spans_text.strip():
                lines.append(spans_text.strip())
        text = "\n".join(lines)

        if not text.strip():
            continue
        text_found = True
        elements.append(PageElement(
            element_type=ElementType.TEXT,
            content=text,
            bbox=bb,
            page_num=page_num,
        ))

    # 2. Fallback: quality-aware extraction if structured gave nothing
    if not text_found:
        qa = QualityAwareTextExtractor(page, page_num)
        result = qa.extract()
        if result.text.strip():
            elements.append(PageElement(
                element_type=ElementType.TEXT,
                content=result.text,
                bbox=(0, 0, page.rect.width, page.rect.height),
                page_num=page_num,
            ))
            if result.used_ocr:
                logger.debug(
                    "[TextExtractor] page %d: used OCR fallback (quality=%.2f)",
                    page_num + 1,
                    result.quality.quality_score,
                )

    return elements


__all__ = ["extract_text_blocks"]
