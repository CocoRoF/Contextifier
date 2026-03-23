# contextifier/handlers/pdf_plus/_element_merger.py
"""
PDF Plus — position-aware element merger.

Collects heterogeneous ``PageElement`` objects (text, table, image, annotation)
and merges them into a single page string, sorted by (y, x).
"""

from __future__ import annotations

from typing import List

from contextifier.handlers.pdf_plus._types import PageElement


def merge_page_elements(elements: List[PageElement], *, page_gap: str = "\n\n") -> str:
    """
    Sort *elements* by reading order (top-to-bottom, left-to-right) and
    concatenate their ``content`` with ``page_gap`` between them.

    The ``PageElement.__lt__`` comparator handles the sort key.
    """
    if not elements:
        return ""

    sorted_els = sorted(elements)

    parts: list[str] = []
    for el in sorted_els:
        text = el.content.strip()
        if text:
            parts.append(text)

    return page_gap.join(parts)


__all__ = ["merge_page_elements"]
