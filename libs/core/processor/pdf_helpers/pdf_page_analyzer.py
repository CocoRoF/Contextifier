# libs/core/processor/pdf_helpers/pdf_page_analyzer.py
"""
PDF Page Analysis Module

Provides functions for analyzing PDF page structure including border detection.
"""
import logging
from typing import List, Tuple

from libs.core.processor.pdf_helpers.types import (
    PDFConfig,
    PageElement,
    PageBorderInfo,
)

logger = logging.getLogger("document-processor")


def detect_page_border(page) -> PageBorderInfo:
    """
    Detects page borders (decorative).

    Improvements:
    1. Detect thin lines as well
    2. Handle double lines
    3. More accurate border identification

    Args:
        page: PyMuPDF page object

    Returns:
        PageBorderInfo object
    """
    result = PageBorderInfo()

    drawings = page.get_drawings()
    if not drawings:
        return result

    page_width = page.rect.width
    page_height = page.rect.height

    edge_margin = min(page_width, page_height) * PDFConfig.PAGE_BORDER_MARGIN
    page_spanning_ratio = PDFConfig.PAGE_SPANNING_RATIO

    border_lines = {
        'top': False,
        'bottom': False,
        'left': False,
        'right': False
    }

    for drawing in drawings:
        rect = drawing.get('rect')
        if not rect:
            continue

        w = rect.width
        h = rect.height

        # Detect thin lines as well (relaxed thickness limit)
        # Horizontal line (small height, large width)
        if h <= 10 and w > page_width * page_spanning_ratio:
            if rect.y0 < edge_margin:
                border_lines['top'] = True
            elif rect.y1 > page_height - edge_margin:
                border_lines['bottom'] = True

        # Vertical line (small width, large height)
        if w <= 10 and h > page_height * page_spanning_ratio:
            if rect.x0 < edge_margin:
                border_lines['left'] = True
            elif rect.x1 > page_width - edge_margin:
                border_lines['right'] = True

    # If all 4 sides present, it's a page border
    if all(border_lines.values()):
        result.has_border = True
        result.border_bbox = (edge_margin, edge_margin, page_width - edge_margin, page_height - edge_margin)
        result.border_lines = border_lines

    return result


def is_table_likely_border(
    table_bbox: Tuple[float, float, float, float],
    border_info: PageBorderInfo,
    page
) -> bool:
    """
    Check if a table is likely a page border.

    Args:
        table_bbox: Table bounding box
        border_info: Page border information
        page: PyMuPDF page object

    Returns:
        True if table is likely a border, False otherwise
    """
    if not border_info.has_border or not border_info.border_bbox:
        return False

    page_width = page.rect.width
    page_height = page.rect.height

    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]

    if table_width > page_width * 0.85 and table_height > page_height * 0.85:
        return True

    return False
