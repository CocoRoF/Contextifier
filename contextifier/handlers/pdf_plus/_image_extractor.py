# contextifier_new/handlers/pdf_plus/_image_extractor.py
"""
PDF Plus — PDF Image Extractor.

Extracts embedded raster images from a single PDF page, filtering by
size and table-overlap.  Images are saved through the ``image_service``
(if provided) or returned as raw PNG bytes.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    ElementType,
    PageElement,
    PdfPlusConfig,
)
from contextifier.handlers.pdf_plus._utils import (
    find_image_position,
    is_inside_any_bbox,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


def extract_images(
    page: Any,
    page_num: int,
    table_bboxes: List[Tuple[float, float, float, float]],
    *,
    image_service: Any = None,
    overlap_threshold: float = 0.7,
) -> List[PageElement]:
    """
    Extract raster images from *page*.

    Filters:
      - minimum width/height: 50 px
      - minimum area: 2500 px²
      - overlap with *table_bboxes* ≥ *overlap_threshold* → skip

    Each qualifying image is saved via *image_service* (if available),
    producing an ``<image>`` tag in the returned :class:`PageElement`.
    """
    import fitz  # PyMuPDF

    elements: list[PageElement] = []
    img_list = page.get_images(full=True)
    if not img_list:
        return elements

    doc = page.parent

    for img_idx, img_info in enumerate(img_list):
        xref = img_info[0]
        try:
            base_img = doc.extract_image(xref)
            if not base_img:
                continue
            width = base_img.get("width", 0)
            height = base_img.get("height", 0)

            # Size filter
            if width < CFG.IMAGE_MIN_SIZE or height < CFG.IMAGE_MIN_SIZE:
                continue
            if width * height < CFG.IMAGE_MIN_AREA:
                continue

            # Find bbox on page
            bbox = find_image_position(page, xref)

            # Table overlap filter
            if bbox and is_inside_any_bbox(bbox, table_bboxes, threshold=overlap_threshold):
                continue

            # Convert to PNG
            png_data = _to_png(doc, xref, base_img)
            if not png_data:
                continue

            # Save through image service or build placeholder
            if image_service:
                try:
                    tag = image_service.save_and_tag(
                        png_data,
                        page_num=page_num,
                        image_index=img_idx,
                    )
                except Exception as exc:
                    logger.debug("[ImgExtractor] save_and_tag failed: %s", exc)
                    tag = f"[Image: page {page_num + 1}, index {img_idx}]"
            else:
                tag = f"[Image: page {page_num + 1}, index {img_idx}]"

            y_pos = bbox[1] if bbox else 0.0
            x_pos = bbox[0] if bbox else 0.0
            elements.append(PageElement(
                element_type=ElementType.IMAGE,
                content=tag,
                bbox=bbox or (0, 0, float(width), float(height)),
                page_num=page_num,
            ))
        except Exception as exc:
            logger.debug(
                "[ImgExtractor] page %d xref %d error: %s", page_num + 1, xref, exc,
            )

    return elements


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _to_png(doc: Any, xref: int, base_img: dict) -> Optional[bytes]:
    """Convert extracted image data to PNG bytes."""
    import fitz

    try:
        pix = fitz.Pixmap(doc, xref)
        # CMYK → RGB
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        elif pix.n == 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix.tobytes("png")
    except Exception:
        # Fallback: use raw image data if already PNG/JPEG
        img_data = base_img.get("image")
        ext = base_img.get("ext", "")
        if img_data and ext in ("png", "jpeg", "jpg"):
            return img_data
        return None


__all__ = ["extract_images"]
