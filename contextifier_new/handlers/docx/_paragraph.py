"""
DOCX paragraph / run-level processing.

Handles the low-level traversal of ``<w:p>`` elements:

- Iterator over ``<w:r>`` (run) child elements
- Text extraction from ``<w:t>`` elements inside runs
- Drawing detection: ``<w:drawing>`` → image / chart / diagram
- Legacy VML detection: ``<w:pict>`` → image
- Page break detection: ``<w:br w:type="page">`` or ``<w:lastRenderedPageBreak>``

This module does NOT know about ImageService or ChartService —
it returns descriptors that the ContentExtractor interprets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, List, Optional, Tuple

from lxml import etree

from contextifier_new.handlers.docx._constants import NAMESPACES, ElementType

logger = logging.getLogger(__name__)

# ── Qualified names (cached for performance) ──────────────────────────────

_W = NAMESPACES["w"]
_WP = NAMESPACES["wp"]
_A = NAMESPACES["a"]
_PIC = NAMESPACES["pic"]
_R = NAMESPACES["r"]
_V = NAMESPACES["v"]

_QN_R = f"{{{_W}}}r"
_QN_T = f"{{{_W}}}t"
_QN_BR = f"{{{_W}}}br"
_QN_LAST_PAGE_BREAK = f"{{{_W}}}lastRenderedPageBreak"
_QN_DRAWING = f"{{{_W}}}drawing"
_QN_PICT = f"{{{_W}}}pict"
_QN_HYPERLINK = f"{{{_W}}}hyperlink"
_QN_INLINE = f"{{{_WP}}}inline"
_QN_ANCHOR = f"{{{_WP}}}anchor"
_QN_GRAPHIC = f"{{{_A}}}graphic"
_QN_GRAPHIC_DATA = f"{{{_A}}}graphicData"
_QN_BLIP = f"{{{_A}}}blip"
_QN_IMAGEDATA = f"{{{_V}}}imagedata"


# ── Drawing descriptor ────────────────────────────────────────────────────

@unique
class DrawingKind(str, Enum):
    """Kind of drawing element detected."""
    IMAGE = "image"
    CHART = "chart"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


@dataclass
class DrawingInfo:
    """Information about a drawing element found in a run."""
    kind: DrawingKind
    rel_id: Optional[str] = None      # r:embed or r:link for images
    uri: Optional[str] = None         # graphicData URI
    graphic_data: Any = None          # lxml element (for diagrams)


@dataclass
class PictInfo:
    """Information about a legacy VML pict element in a run."""
    rel_id: Optional[str] = None


# ── Run element descriptors ───────────────────────────────────────────────

@dataclass
class RunContent:
    """Content extracted from a single run (or hyperlink child)."""
    text: str = ""
    drawings: List[DrawingInfo] = None  # type: ignore[assignment]
    picts: List[PictInfo] = None        # type: ignore[assignment]
    has_page_break: bool = False

    def __post_init__(self) -> None:
        if self.drawings is None:
            self.drawings = []
        if self.picts is None:
            self.picts = []


# ── Paragraph processing ─────────────────────────────────────────────────

def process_paragraph(paragraph_element: Any) -> Tuple[str, List[DrawingInfo], List[PictInfo], bool]:
    """
    Process a ``<w:p>`` element and extract its content.

    Args:
        paragraph_element: lxml element for ``<w:p>``.

    Returns:
        Tuple of:
        - text: Concatenated text from all runs
        - drawings: List of DrawingInfo for images/charts/diagrams
        - picts: List of PictInfo for legacy VML images
        - has_page_break: True if a page break was detected
    """
    full_text_parts: List[str] = []
    all_drawings: List[DrawingInfo] = []
    all_picts: List[PictInfo] = []
    has_page_break = False

    for child in paragraph_element:
        tag = _local_name(child)

        if child.tag == _QN_R:
            rc = _process_run(child)
            if rc.text:
                full_text_parts.append(rc.text)
            all_drawings.extend(rc.drawings)
            all_picts.extend(rc.picts)
            if rc.has_page_break:
                has_page_break = True

        elif child.tag == _QN_HYPERLINK:
            # Process runs inside hyperlink
            for sub in child:
                if sub.tag == _QN_R:
                    rc = _process_run(sub)
                    if rc.text:
                        full_text_parts.append(rc.text)
                    all_drawings.extend(rc.drawings)
                    all_picts.extend(rc.picts)
                    if rc.has_page_break:
                        has_page_break = True

    text = "".join(full_text_parts)
    return text, all_drawings, all_picts, has_page_break


def has_page_break(paragraph_element: Any) -> bool:
    """
    Check if a paragraph contains a page break.

    Checks for:
    - ``<w:br w:type="page"/>``
    - ``<w:lastRenderedPageBreak/>``
    """
    for br_elem in paragraph_element.iter(_QN_BR):
        br_type = br_elem.get(f"{{{_W}}}type", "")
        if br_type == "page":
            return True

    for _ in paragraph_element.iter(_QN_LAST_PAGE_BREAK):
        return True

    return False


# ── Run processing (internal) ─────────────────────────────────────────────

def _process_run(run_element: Any) -> RunContent:
    """
    Process a single ``<w:r>`` element.

    Extracts text (``<w:t>``), drawings (``<w:drawing>``),
    pict elements (``<w:pict>``), and page breaks.
    """
    rc = RunContent()

    for child in run_element:
        if child.tag == _QN_T:
            # Text element
            if child.text:
                rc.text += child.text

        elif child.tag == _QN_BR:
            # Break element
            br_type = child.get(f"{{{_W}}}type", "")
            if br_type == "page":
                rc.has_page_break = True
            else:
                rc.text += "\n"

        elif child.tag == _QN_LAST_PAGE_BREAK:
            rc.has_page_break = True

        elif child.tag == _QN_DRAWING:
            drawing_info = _process_drawing(child)
            if drawing_info is not None:
                rc.drawings.append(drawing_info)

        elif child.tag == _QN_PICT:
            pict_info = _process_pict(child)
            if pict_info is not None:
                rc.picts.append(pict_info)

    return rc


def _process_drawing(drawing_element: Any) -> Optional[DrawingInfo]:
    """
    Analyze a ``<w:drawing>`` element.

    Checks for inline or anchor container, then inspects graphic data URI
    to classify as image, chart, or diagram.
    """
    # Find inline or anchor container
    container = drawing_element.find(_QN_INLINE)
    if container is None:
        container = drawing_element.find(_QN_ANCHOR)
    if container is None:
        return None

    # Find graphic → graphicData
    graphic = container.find(f".//{_QN_GRAPHIC}")
    if graphic is None:
        return None

    graphic_data = graphic.find(_QN_GRAPHIC_DATA)
    if graphic_data is None:
        return None

    uri = graphic_data.get("uri", "")

    # Image
    if "picture" in uri.lower():
        rel_id = _find_blip_rel_id(graphic_data)
        return DrawingInfo(
            kind=DrawingKind.IMAGE,
            rel_id=rel_id,
            uri=uri,
            graphic_data=graphic_data,
        )

    # Chart
    if "chart" in uri.lower():
        return DrawingInfo(
            kind=DrawingKind.CHART,
            uri=uri,
            graphic_data=graphic_data,
        )

    # Diagram
    if "diagram" in uri.lower():
        return DrawingInfo(
            kind=DrawingKind.DIAGRAM,
            uri=uri,
            graphic_data=graphic_data,
        )

    return DrawingInfo(kind=DrawingKind.UNKNOWN, uri=uri)


def _process_pict(pict_element: Any) -> Optional[PictInfo]:
    """
    Analyze a legacy VML ``<w:pict>`` element.

    Looks for ``<v:imagedata r:id="rIdN"/>`` to find the image relationship.
    """
    # Look for VML imagedata
    for imagedata in pict_element.iter(f"{{{_V}}}imagedata"):
        rel_id = imagedata.get(f"{{{_R}}}id")
        if rel_id:
            return PictInfo(rel_id=rel_id)

    # Also check for shapes that might have images
    for shape in pict_element.iter(f"{{{_V}}}shape"):
        for imagedata in shape.iter(f"{{{_V}}}imagedata"):
            rel_id = imagedata.get(f"{{{_R}}}id")
            if rel_id:
                return PictInfo(rel_id=rel_id)

    return None


def _find_blip_rel_id(graphic_data: Any) -> Optional[str]:
    """
    Find the relationship ID from a blip element inside graphic data.

    Looks for ``<a:blip r:embed="rIdN"/>`` or ``<a:blip r:link="rIdN"/>``.
    """
    for blip in graphic_data.iter(_QN_BLIP):
        r_embed = blip.get(f"{{{_R}}}embed")
        if r_embed:
            return r_embed
        r_link = blip.get(f"{{{_R}}}link")
        if r_link:
            return r_link
    return None


def extract_diagram_text(graphic_data: Any) -> str:
    """
    Extract text content from a diagram's graphic data.

    Diagrams (SmartArt) store their text in ``<a:t>`` elements
    within the DrawingML tree.

    Args:
        graphic_data: The ``<a:graphicData>`` lxml element.

    Returns:
        Formatted diagram text string.
    """
    texts: List[str] = []
    for t_elem in graphic_data.iter(f"{{{_A}}}t"):
        if t_elem.text and t_elem.text.strip():
            texts.append(t_elem.text.strip())

    if texts:
        return f"[Diagram: {' / '.join(texts)}]"
    return "[Diagram]"


# ── Utilities ─────────────────────────────────────────────────────────────

def _local_name(element: Any) -> str:
    """Get the local name of an lxml element (without namespace)."""
    tag = element.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return str(tag)


__all__ = [
    "process_paragraph",
    "has_page_break",
    "extract_diagram_text",
    "DrawingInfo",
    "DrawingKind",
    "PictInfo",
    "RunContent",
]
