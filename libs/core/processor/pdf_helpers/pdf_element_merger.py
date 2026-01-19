# libs/core/processor/pdf_helpers/pdf_element_merger.py
"""
PDF Element Merger Module

Provides functions for merging and sorting page elements.
"""
import logging
from typing import List

from libs.core.processor.pdf_helpers.types import (
    ElementType,
    PageElement,
)

logger = logging.getLogger("document-processor")


def merge_page_elements(elements: List[PageElement]) -> str:
    """
    Merge page elements sorted by position.

    Args:
        elements: List of PageElement

    Returns:
        Merged text content
    """
    if not elements:
        return ""

    sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))

    text_parts = []

    for element in sorted_elements:
        content = element.content.strip()
        if not content:
            continue

        if element.element_type == ElementType.TABLE:
            text_parts.append(f"\n{content}\n")
        else:
            text_parts.append(content)

    return "\n".join(text_parts)
