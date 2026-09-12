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
from typing import Any, Iterator, List, Optional, Tuple


from contextifier.handlers.docx._constants import NAMESPACES

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

_MC = NAMESPACES["mc"]
_QN_ALTERNATE_CONTENT = f"{{{_MC}}}AlternateContent"
_QN_MC_CHOICE = f"{{{_MC}}}Choice"
_QN_MC_FALLBACK = f"{{{_MC}}}Fallback"
_QN_P = f"{{{_W}}}p"
_QN_TXBX_CONTENT = f"{{{_W}}}txbxContent"
_QN_SDT = f"{{{_W}}}sdt"
_QN_SDT_CONTENT = f"{{{_W}}}sdtContent"
_QN_TAB = f"{{{_W}}}tab"


# ── Block-level traversal ─────────────────────────────────────────────────


def local_name(element: Any) -> str:
    """Tag name of an element without its namespace."""
    tag = element.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag if isinstance(tag, str) else ""


def iter_block_elements(container: Any) -> Iterator[Any]:
    """
    Yield the block-level children of *container* in document order,
    descending through content controls.

    A ``<w:sdt>`` (Structured Document Tag) is a wrapper, not content: tables
    of contents, bibliographies, cover-page fields and anything a user inserted
    as a content control put their real paragraphs and tables inside
    ``<w:sdtContent>``. A walk that only recognises ``w:p`` and ``w:tbl`` skips
    the wrapper and loses everything it holds, so the wrapper is unwrapped here
    — recursively, because content controls nest.
    """
    for child in container:
        if not isinstance(child.tag, str):
            continue  # comments / processing instructions
        if local_name(child) != "sdt":
            yield child
            continue
        for sdt_child in child:
            if local_name(sdt_child) == "sdtContent":
                yield from iter_block_elements(sdt_child)



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
    rel_id: Optional[str] = None  # r:embed or r:link for images
    uri: Optional[str] = None  # graphicData URI
    graphic_data: Any = None  # lxml element (for diagrams)


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
    picts: List[PictInfo] = None  # type: ignore[assignment]
    has_page_break: bool = False

    def __post_init__(self) -> None:
        if self.drawings is None:
            self.drawings = []
        if self.picts is None:
            self.picts = []


# ── Paragraph processing ─────────────────────────────────────────────────


def process_paragraph(
    paragraph_element: Any,
) -> Tuple[str, List[DrawingInfo], List[PictInfo], bool]:
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

    def absorb(rc: RunContent) -> None:
        nonlocal has_page_break
        if rc.text:
            full_text_parts.append(rc.text)
        all_drawings.extend(rc.drawings)
        all_picts.extend(rc.picts)
        if rc.has_page_break:
            has_page_break = True

    def visit(node: Any) -> None:
        """Walk the direct children of a paragraph-level container."""
        for child in node:
            if child.tag == _QN_R:
                absorb(_process_run(child))
            elif child.tag == _QN_HYPERLINK:
                visit(child)
            elif child.tag == _QN_SDT:
                # Inline content control: the runs live under sdtContent.
                for sdt_child in child:
                    if sdt_child.tag == _QN_SDT_CONTENT:
                        visit(sdt_child)
            elif child.tag == _QN_ALTERNATE_CONTENT:
                # Some producers wrap an anchored shape at paragraph level
                # rather than inside a run.
                rc = RunContent()
                for resolved in resolve_alternate_content(child):
                    if resolved.tag == _QN_R:
                        absorb(_process_run(resolved))
                    else:
                        _process_run_child(resolved, rc)
                absorb(rc)

    visit(paragraph_element)

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
        _process_run_child(child, rc)
    return rc


def _process_run_child(child: Any, rc: RunContent) -> None:
    """
    Fold one child of a run into *rc*.

    Split out of :func:`_process_run` so that a compatibility branch resolved
    from ``<mc:AlternateContent>`` goes through exactly the same handling as a
    child written directly in the run.
    """
    tag = child.tag

    if tag == _QN_T:
        if child.text:
            rc.text += child.text

    elif tag == _QN_BR:
        br_type = child.get(f"{{{_W}}}type", "")
        if br_type == "page":
            rc.has_page_break = True
        else:
            rc.text += "\n"

    elif tag == _QN_LAST_PAGE_BREAK:
        rc.has_page_break = True

    elif tag == _QN_DRAWING:
        drawing_info = _process_drawing(child)
        if drawing_info is not None:
            rc.drawings.append(drawing_info)
        # A shape carries its own text; the drawing descriptor only describes
        # the picture/chart/diagram, so the text box has to be read here.
        shape_text = _extract_shape_text(child)
        if shape_text:
            rc.text += shape_text

    elif tag == _QN_PICT:
        pict_info = _process_pict(child)
        if pict_info is not None:
            rc.picts.append(pict_info)
        shape_text = _extract_shape_text(child)
        if shape_text:
            rc.text += shape_text

    elif tag == _QN_ALTERNATE_CONTENT:
        for resolved in resolve_alternate_content(child):
            _process_run_child(resolved, rc)


def resolve_alternate_content(element: Any) -> List[Any]:
    """
    Pick ONE branch of an ``<mc:AlternateContent>`` element.

    Word writes a shape twice — a DrawingML rendering under ``<mc:Choice>``
    and an equivalent legacy VML rendering under ``<mc:Fallback>``. Reading
    both duplicates every character the shape contains, so a consumer must
    commit to one. ``Choice`` is preferred (it is what a modern Word renders);
    ``Fallback`` is the insurance for producers that ship only VML.

    Returns:
        The children of the chosen branch, or ``[]`` when neither is usable.
    """
    for branch_tag in (_QN_MC_CHOICE, _QN_MC_FALLBACK):
        for branch in element.iterchildren(branch_tag):
            children = list(branch)
            if children:
                return children
    return []


def _collect_text(element: Any) -> str:
    """
    Concatenate the text of an element subtree in document order.

    Unlike a bare ``iter(w:t)`` this resolves nested ``<mc:AlternateContent>``
    to a single branch, so a shape nested inside another shape contributes its
    text once rather than twice. Paragraph boundaries become newlines.
    """
    parts: List[str] = []

    def walk(node: Any) -> None:
        for child in node:
            tag = child.tag
            if tag == _QN_T:
                if child.text:
                    parts.append(child.text)
            elif tag == _QN_TAB:
                parts.append("\t")
            elif tag == _QN_BR or tag == _QN_P:
                # A paragraph or break inside a shape separates lines.
                if parts and not parts[-1].endswith("\n"):
                    parts.append("\n")
                walk(child)
            elif tag == _QN_ALTERNATE_CONTENT:
                for resolved in resolve_alternate_content(child):
                    walk(resolved)
            elif isinstance(tag, str):
                walk(child)

    walk(element)
    return "".join(parts).strip()


def _extract_shape_text(element: Any) -> str:
    """
    Text held inside a shape's text box (``<w:txbxContent>``).

    Covers both renderings: DrawingML shapes store it under
    ``wps:wsp/wps:txbx/w:txbxContent`` and VML shapes under
    ``v:shape/v:textbox/w:txbxContent``. Callers pass the ``<w:drawing>`` or
    ``<w:pict>`` element; the caller has already committed to one
    compatibility branch, so no de-duplication is needed here.
    """
    texts: List[str] = []
    for txbx in element.iter(_QN_TXBX_CONTENT):
        text = _collect_text(txbx)
        if text:
            texts.append(text)
    return "\n".join(texts)


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
