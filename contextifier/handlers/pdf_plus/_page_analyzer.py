# contextifier_new/handlers/pdf_plus/_page_analyzer.py
"""
PDF Plus — page-level analysis helpers.

* ``detect_page_border()``  — detect decorative page-border frames
* ``is_table_likely_border()`` — reject "tables" that are actually borders
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from contextifier.handlers.pdf_plus._types import (
    PageBorderInfo,
    PdfPlusConfig,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


def detect_page_border(page: Any) -> PageBorderInfo:
    """
    Detect whether *page* has a decorative border frame.

    Thin lines (≤ ``PAGE_BORDER_LINE_MAX_SIZE`` pt) spanning > 85 %
    of the page near the edges are considered border lines.
    """
    info = PageBorderInfo()
    try:
        pw, ph = page.rect.width, page.rect.height
        margin_x = pw * CFG.PAGE_BORDER_MARGIN_RATIO
        margin_y = ph * CFG.PAGE_BORDER_MARGIN_RATIO
        drawings = page.get_drawings()

        for d in drawings:
            rect = d.get("rect")
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            w = x1 - x0
            h = y1 - y0

            # Must be very thin — one dimension tiny
            is_h_line = h <= CFG.PAGE_BORDER_LINE_MAX_SIZE and w >= pw * CFG.PAGE_SPANNING_RATIO
            is_v_line = w <= CFG.PAGE_BORDER_LINE_MAX_SIZE and h >= ph * CFG.PAGE_SPANNING_RATIO

            if is_h_line:
                if y0 < margin_y:
                    info.border_lines["top"] = True
                elif y1 > ph - margin_y:
                    info.border_lines["bottom"] = True
            if is_v_line:
                if x0 < margin_x:
                    info.border_lines["left"] = True
                elif x1 > pw - margin_x:
                    info.border_lines["right"] = True

        if any(info.border_lines.values()):
            info.has_border = True
            # Compute the border bbox from detected sides
            bx0 = 0.0 if info.border_lines.get("left") else margin_x
            by0 = 0.0 if info.border_lines.get("top") else margin_y
            bx1 = pw if info.border_lines.get("right") else pw - margin_x
            by1 = ph if info.border_lines.get("bottom") else ph - margin_y
            info.border_bbox = (bx0, by0, bx1, by1)

    except Exception as exc:
        logger.debug("Border detection failed: %s", exc)

    return info


def is_table_likely_border(
    table_bbox: Tuple[float, float, float, float],
    border_info: PageBorderInfo,
    page: Any,
) -> bool:
    """
    Return ``True`` if *table_bbox* covers >85 % of both page width and height,
    meaning it's really a page border rather than a real table.
    """
    if not border_info.has_border:
        return False
    pw, ph = page.rect.width, page.rect.height
    tw = table_bbox[2] - table_bbox[0]
    th = table_bbox[3] - table_bbox[1]
    return (tw / pw > CFG.PAGE_SPANNING_RATIO and th / ph > CFG.PAGE_SPANNING_RATIO)


__all__ = ["detect_page_border", "is_table_likely_border"]
