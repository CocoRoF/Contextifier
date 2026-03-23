# contextifier/handlers/pdf_plus/_utils.py
"""
PDF Plus — shared utility functions.

Small, pure helpers used across multiple pdf_plus modules.
"""

from __future__ import annotations

import html as _html
from typing import List, Optional, Tuple


def escape_html(text: str) -> str:
    """Escape ``&``, ``<``, ``>``, ``"`` for safe HTML embedding."""
    return _html.escape(text, quote=True)


def calculate_overlap_ratio(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> float:
    """Return the fraction of *bbox1*'s area that overlaps *bbox2*."""
    ix0 = max(bbox1[0], bbox2[0])
    iy0 = max(bbox1[1], bbox2[1])
    ix1 = min(bbox1[2], bbox2[2])
    iy1 = min(bbox1[3], bbox2[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    return inter / area if area > 0 else 0.0


def is_inside_any_bbox(
    bbox: Tuple[float, float, float, float],
    bbox_list: List[Tuple[float, float, float, float]],
    threshold: float = 0.5,
) -> bool:
    """True if *bbox* overlaps ≥ *threshold* with any box in *bbox_list*."""
    return any(calculate_overlap_ratio(bbox, b) >= threshold for b in bbox_list)


def bbox_overlaps(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> bool:
    """Simple boolean: do the two boxes overlap at all?"""
    return not (
        bbox1[2] <= bbox2[0]
        or bbox1[0] >= bbox2[2]
        or bbox1[3] <= bbox2[1]
        or bbox1[1] >= bbox2[3]
    )


def find_image_position(
    page,  # fitz.Page
    xref: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Return the bbox of image *xref* on *page*, or ``None``."""
    try:
        for info in page.get_image_info(xrefs=True):
            if info.get("xref") == xref:
                b = info.get("bbox")
                if b:
                    return tuple(b)  # type: ignore[return-value]
    except Exception:
        pass
    return None


def get_text_lines_with_positions(
    page,  # fitz.Page
) -> List[dict]:
    """
    Return ``[{"bbox": (...), "text": "..."}]`` for every text line on *page*.
    """
    result: list[dict] = []
    try:
        td = page.get_text("dict", sort=True)
        for block in td.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                if text.strip():
                    result.append({"bbox": tuple(line["bbox"]), "text": text})
    except Exception:
        pass
    return result


__all__ = [
    "escape_html",
    "calculate_overlap_ratio",
    "is_inside_any_bbox",
    "bbox_overlaps",
    "find_image_position",
    "get_text_lines_with_positions",
]
