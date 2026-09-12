# contextifier/handlers/hwpx/_section.py
"""
HWPX section XML parser.

Parses a single ``Contents/section*.xml`` file, traversing the XML tree
to extract text, tables, inline images, and charts in document order.

XML element hierarchy::

    <hs:sec>
      <hp:p>                    ← paragraph
        <hp:run>                ← run
          <hp:t>text</hp:t>    ← inline text
          <hp:tbl .../>        ← table (rowCnt, colCnt)
          <hp:ctrl>            ← control (image / chart / etc.)
            <hc:pic>           ← picture
              <hc:img binaryItemIDRef="..."/>
            </hc:pic>
          </hp:ctrl>
          <hp:pic>             ← direct picture (variant)
            <hc:img binaryItemIDRef="..."/>
          </hp:pic>
        </hp:run>
        <hp:switch>            ← conditional content
          <hp:case>
            <hp:chart chartIDRef="..."/>
          </hp:case>
        </hp:switch>
      </hp:p>
    </hs:sec>
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from contextifier.handlers.hwpx._constants import (
    CHART_PREFIXES,
    HWPX_NAMESPACES,
    OOXML_CHART_NS,
    CHART_TYPE_MAP,
)
from contextifier.handlers._ooxml import extract_chart_title
from contextifier.handlers.hwpx._table import parse_hwpx_table

if TYPE_CHECKING:
    from contextifier.services.image_service import ImageService
    from contextifier.services.chart_service import ChartService

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HwpxSupplementary:
    """Header / footer / note text collected while walking a section.

    These parts are anchored to the page, not to a position in the running
    text, so splicing them into the body puts a page header in the middle of a
    sentence. The caller collects them here and renders them as their own
    block, matching how the DOCX handler reports ``[Headers]`` / ``[Footers]``.
    """

    headers: List[str] = field(default_factory=list)
    footers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class _Ctx:
    """Everything the recursive walk needs, threaded as one object."""

    zf: zipfile.ZipFile
    bin_item_map: Dict[str, str]
    ns: Dict[str, str]
    image_service: Optional["ImageService"]
    chart_service: Optional["ChartService"]
    processed_images: Set[str]
    supplementary: Optional[HwpxSupplementary]


# Shapes that can carry a text box. Hangul has many shape elements and the
# list grows between versions, so the walk treats "has an <hp:drawText>"
# as the real test and uses this set only to know what to recurse into.
_SHAPE_TAGS = frozenset(
    {
        "container",
        "rect",
        "ellipse",
        "polygon",
        "arc",
        "curve",
        "line",
        "connectLine",
        "textart",
        "shapeObject",
    }
)

# Elements whose content is rendered by a dedicated handler. A paragraph
# inside one of these belongs to that handler, not to the section body.
_OWNED_CONTAINER_TAGS = frozenset({"tbl", "drawText", "ctrl", "subList", "switch"})

# Control children that are page furniture rather than running text.
_SUPPLEMENTARY_TAGS = {
    "header": "headers",
    "footer": "footers",
    "footNote": "notes",
    "endNote": "notes",
}


def _tag_of(element: ET.Element) -> str:
    """Local tag name without the namespace URI."""
    tag = element.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag if isinstance(tag, str) else ""


def parse_hwpx_section(
    section_xml: bytes,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    *,
    image_service: Optional["ImageService"] = None,
    chart_service: Optional["ChartService"] = None,
    processed_images: Optional[Set[str]] = None,
    supplementary: Optional[HwpxSupplementary] = None,
) -> str:
    """
    Parse a single HWPX section XML and return its body text.

    Tables are rendered inline (HTML or plain-text depending on shape),
    images are saved via *image_service* and replaced by image tags,
    charts are formatted as text blocks, and shape text boxes are rendered
    where the shape sits in the text.

    Only paragraphs that belong to the section body are walked. Table cells,
    shape text boxes and header/footer parts store their content as ``hp:p``
    as well, so a blanket ``.//hp:p`` search reports each of them a second
    time — table content came out twice and shape text landed at the end of
    the section rather than in place.

    Args:
        section_xml: Raw XML bytes of the section.
        zf: The parent HWPX ZipFile (for reading BinData images / charts).
        bin_item_map: Mapping from ``binaryItemIDRef`` to ZIP path.
        image_service: Optional — for saving embedded images.
        chart_service: Optional — for formatting chart data.
        processed_images: A set that accumulates processed image paths
                          (to avoid duplicates across sections).
        supplementary: Optional collector for header/footer/note text. When
                       omitted, that text is emitted inline at the position of
                       its control so that nothing is silently dropped.

    Returns:
        Extracted text for the section.
    """
    if processed_images is None:
        processed_images = set()

    try:
        root = ET.fromstring(section_xml)
    except ET.ParseError as exc:
        logger.warning("Failed to parse HWPX section XML: %s", exc)
        return ""

    ctx = _Ctx(
        zf=zf,
        bin_item_map=bin_item_map,
        ns=HWPX_NAMESPACES,
        image_service=image_service,
        chart_service=chart_service,
        processed_images=processed_images,
        supplementary=supplementary,
    )

    parts: List[str] = []
    for para in _iter_body_paragraphs(root):
        para_text = _process_paragraph(para, ctx)
        if para_text and para_text.strip():
            parts.append(para_text)

    return "\n".join(parts)


def _iter_body_paragraphs(node: ET.Element):
    """
    Yield the paragraphs that belong to the section body.

    Descends through structural wrappers but stops at any container that
    renders its own content, so a cell or text-box paragraph is reached only
    by the renderer that owns it.
    """
    for child in node:
        tag = _tag_of(child)
        if tag == "p":
            yield child
        elif tag in _OWNED_CONTAINER_TAGS:
            continue
        else:
            yield from _iter_body_paragraphs(child)


# ═══════════════════════════════════════════════════════════════════════════════
# Paragraph Processing
# ═══════════════════════════════════════════════════════════════════════════════


def _process_paragraph(para: ET.Element, ctx: _Ctx) -> str:
    """Process a single ``<hp:p>`` paragraph, returning its text."""
    parts: List[str] = []
    for child in para:
        piece = _process_node(child, ctx)
        if piece:
            parts.append(piece)
    return "".join(parts)


def _process_node(node: ET.Element, ctx: _Ctx) -> str:
    """
    Render one element of a paragraph/run/control in document order.

    Every container that can hold content routes through here, so a table in a
    shape, a shape in a control and a control in a run all behave the same.
    """
    tag = _tag_of(node)

    if tag == "t":
        return node.text or ""

    if tag == "run":
        return "".join(_process_node(child, ctx) for child in node)

    if tag == "tbl":
        table_text = parse_hwpx_table(
            node,
            ctx.ns,
            render_cell=lambda tc: _render_cell(tc, ctx),
        )
        return f"\n{table_text}\n" if table_text else ""

    if tag == "switch":
        return _process_switch(node, ctx)

    if tag == "ctrl":
        return _process_ctrl(node, ctx)

    if tag in ("pic",):
        return _process_picture_element(
            node,
            ctx.zf,
            ctx.bin_item_map,
            ctx.ns,
            image_service=ctx.image_service,
            processed_images=ctx.processed_images,
        )

    if tag == "chart":
        return _process_chart_ref(node, ctx.zf, chart_service=ctx.chart_service)

    if tag == "drawText":
        return _process_sublists(node, ctx)

    if tag == "subList":
        return _process_sublist(node, ctx)

    if tag == "p":
        return _process_paragraph(node, ctx)

    if tag in _SHAPE_TAGS:
        # A shape is a container: it may nest further shapes and it may carry
        # a text box. Recursing over its children covers both.
        return "\n".join(
            piece for piece in (_process_node(child, ctx) for child in node) if piece
        )

    # Unknown element — descend only if it actually holds a text box, so an
    # unrecognised shape from a newer Hangul release still yields its text.
    if node.find("hp:drawText", ctx.ns) is not None:
        return _process_sublists(node, ctx)

    return ""


def _render_cell(tc: ET.Element, ctx: _Ctx) -> str:
    """
    Render one ``<hp:tc>`` with the same walker the body uses.

    A cell is not limited to text: a scanned figure pasted into a form is a
    picture with no text beside it, and reading cells for ``hp:t`` alone left
    that cell blank with no image tag anywhere — nothing for the OCR pass to
    pick up. Routing the cell through the ordinary walker also brings nested
    tables and shape text along.
    """
    pieces = [
        _process_sublist(sublist, ctx) for sublist in tc.findall("hp:subList", ctx.ns)
    ]
    rendered = " ".join(piece.strip() for piece in pieces if piece.strip())
    return rendered.strip()


def _process_sublist(sublist: ET.Element, ctx: _Ctx) -> str:
    """Render the paragraphs of one ``<hp:subList>``."""
    pieces = [_process_paragraph(p, ctx) for p in sublist.findall("hp:p", ctx.ns)]
    return "\n".join(piece for piece in pieces if piece.strip())


def _iter_owned_sublists(node: ET.Element):
    """
    Yield the ``<hp:subList>`` elements that belong to *node* itself.

    Descends through nested shapes and text-box wrappers but never enters a
    sub-list it has already yielded, nor a table or control — those render
    their own cells, and walking into them would report a cell's text a second
    time as if it were shape text.
    """
    for child in node:
        tag = _tag_of(child)
        if tag == "subList":
            yield child
        elif tag in ("tbl", "ctrl", "switch"):
            continue
        else:
            yield from _iter_owned_sublists(child)


def _process_sublists(node: ET.Element, ctx: _Ctx) -> str:
    """Render the text boxes belonging to *node* (a shape or ``drawText``)."""
    pieces: List[str] = []
    for sublist in _iter_owned_sublists(node):
        text = _process_sublist(sublist, ctx)
        if text.strip():
            pieces.append(text)
    return "\n".join(pieces)


# ═══════════════════════════════════════════════════════════════════════════════
# Control / Switch Processing
# ═══════════════════════════════════════════════════════════════════════════════


def _process_ctrl(ctrl: ET.Element, ctx: _Ctx) -> str:
    """
    Process ``<hp:ctrl>``.

    A control holds anything anchored rather than inline: pictures, tables,
    shapes, and the page furniture (header, footer, footnote, endnote). The
    furniture is diverted to the supplementary collector when the caller
    supplied one; everything else renders in place.
    """
    parts: List[str] = []

    for child in ctrl:
        tag = _tag_of(child)

        bucket = _SUPPLEMENTARY_TAGS.get(tag)
        if bucket is not None:
            text = _process_sublists(child, ctx) or _process_sublist_fallback(child, ctx)
            if not text.strip():
                continue
            if ctx.supplementary is not None:
                getattr(ctx.supplementary, bucket).append(text.strip())
            else:
                parts.append(text)
            continue

        # hc:pic and hp:pic differ only by namespace.
        if tag == "pic":
            piece = _process_picture_element(
                child,
                ctx.zf,
                ctx.bin_item_map,
                ctx.ns,
                image_service=ctx.image_service,
                processed_images=ctx.processed_images,
            )
        else:
            piece = _process_node(child, ctx)

        if piece:
            parts.append(piece)

    return "".join(parts)


def _process_sublist_fallback(node: ET.Element, ctx: _Ctx) -> str:
    """Header/footer variants that hold paragraphs without a ``subList``."""
    pieces = [
        _process_paragraph(para, ctx) for para in _iter_body_paragraphs(node)
    ]
    return "\n".join(piece for piece in pieces if piece.strip())


def _process_switch(switch: ET.Element, ctx: _Ctx) -> str:
    """
    Process ``<hp:switch>`` — iterate cases for charts / nested content.

    The ``<hp:case>`` elements may contain ``<hp:chart>`` or paragraphs.
    """
    parts: List[str] = []
    for case in switch.findall("hp:case", ctx.ns):
        for child in case:
            piece = _process_node(child, ctx)
            if piece:
                parts.append(piece)
    return "".join(parts)


def _process_picture_element(
    pic: ET.Element,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    ns: Dict[str, str],
    *,
    image_service: Optional["ImageService"],
    processed_images: Set[str],
) -> str:
    """
    Extract image from a picture element (``<hc:pic>`` or ``<hp:pic>``).

    Looks for ``<hc:img binaryItemIDRef="...">`` to resolve the image
    path via *bin_item_map*, then reads it from the ZIP.
    """
    if image_service is None:
        return ""

    try:
        # Find <hc:img> child
        img_elem = pic.find("hc:img", ns)
        bin_item_id: Optional[str] = None

        if img_elem is not None:
            bin_item_id = img_elem.get("binaryItemIDRef")
        else:
            # Fallback: direct attribute
            bin_item_id = pic.get("BinItem") or pic.get("binaryItemIDRef")

        if not bin_item_id or bin_item_id not in bin_item_map:
            return ""

        img_path = bin_item_map[bin_item_id]

        # Resolve path inside the ZIP
        full_path = _resolve_zip_path(zf, img_path)
        if not full_path:
            return ""

        # Skip if already processed
        if full_path in processed_images:
            return ""

        with zf.open(full_path) as f:
            image_data = f.read()

        tag = image_service.save_and_tag(image_data)
        if tag:
            processed_images.add(full_path)
            return f"\n{tag}\n"

    except Exception as exc:
        logger.warning("Failed to process HWPX image: %s", exc)

    return ""


def _resolve_zip_path(zf: zipfile.ZipFile, href: str) -> Optional[str]:
    """
    Resolve an image href to an actual entry in the ZIP.

    Tries the path as-is, then with a ``Contents/`` prefix.
    """
    namelist = zf.namelist()
    if href in namelist:
        return href
    prefixed = f"Contents/{href}"
    if prefixed in namelist:
        return prefixed
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Chart Processing
# ═══════════════════════════════════════════════════════════════════════════════


def _process_chart_ref(
    chart_elem: ET.Element,
    zf: zipfile.ZipFile,
    *,
    chart_service: Optional["ChartService"],
) -> str:
    """
    Process a ``<hp:chart chartIDRef="...">`` element.

    Reads the referenced chart XML from the ZIP, parses it as
    OOXML chart data, and formats it as a text block via *chart_service*.
    """
    chart_id_ref = chart_elem.get("chartIDRef")
    if not chart_id_ref:
        return ""

    # Find the chart file in the ZIP
    chart_path = _resolve_chart_path(zf, chart_id_ref)
    if not chart_path:
        return ""

    try:
        with zf.open(chart_path) as f:
            chart_xml = f.read()

        chart_data = _parse_ooxml_chart(chart_xml)
        if not chart_data:
            return ""

        # Format via chart_service if available
        if chart_service is not None:
            return chart_service.format_chart(chart_data)

        # Fallback: simple text
        return _format_chart_simple(chart_data)

    except Exception as exc:
        logger.warning("Failed to process HWPX chart %s: %s", chart_id_ref, exc)
        return ""


def _resolve_chart_path(
    zf: zipfile.ZipFile,
    chart_id_ref: str,
) -> Optional[str]:
    """Resolve a chartIDRef to an actual ZIP entry."""
    namelist = zf.namelist()

    # Direct match
    if chart_id_ref in namelist:
        return chart_id_ref

    # Try common prefixes
    for prefix in CHART_PREFIXES:
        candidate = prefix + chart_id_ref
        if candidate in namelist:
            return candidate

    # Try Contents/ prefix
    if f"Contents/{chart_id_ref}" in namelist:
        return f"Contents/{chart_id_ref}"

    return None


def _parse_ooxml_chart(chart_xml: bytes) -> Optional[Dict]:
    """
    Parse OOXML chart XML and return a simple dict with chart info.

    Returns:
        ``{"type": str, "title": str|None, "categories": list, "series": list}``
        or ``None`` if parsing fails.
    """
    try:
        root = ET.fromstring(chart_xml)
    except ET.ParseError:
        return None

    ns = OOXML_CHART_NS

    # Find <c:chart>
    chart = root.find(".//c:chart", ns)
    if chart is None:
        # Maybe root itself is <c:chart>
        if root.tag.endswith("}chart") or root.tag == "chart":
            chart = root
        else:
            return None

    title = _extract_chart_title(chart, ns)
    chart_type, categories, series = _extract_chart_plot(chart, ns)

    if not series:
        return None

    return {
        "type": chart_type,
        "title": title,
        "categories": categories,
        "series": series,
    }


def _extract_chart_title(chart: ET.Element, ns: Dict[str, str]) -> Optional[str]:
    """Extract chart title from ``<c:title>``."""
    return extract_chart_title(chart, ns_c=ns["c"], ns_a=ns["a"])


def _extract_chart_plot(
    chart: ET.Element,
    ns: Dict[str, str],
) -> tuple:
    """Extract chart type, categories, and series data."""
    plot_area = chart.find(".//c:plotArea", ns)
    if plot_area is None:
        return "Chart", [], []

    chart_type = "Chart"
    categories: List[str] = []
    series: List[Dict] = []

    for tag_name, type_name in CHART_TYPE_MAP.items():
        elem = plot_area.find(f".//c:{tag_name}", ns)
        if elem is not None:
            chart_type = type_name
            categories, series = _extract_series(elem, ns)
            break

    return chart_type, categories, series


def _extract_series(
    chart_type_elem: ET.Element,
    ns: Dict[str, str],
) -> tuple:
    """Extract categories and series from a chart-type element."""
    categories: List[str] = []
    series: List[Dict] = []
    cats_done = False

    for idx, ser_elem in enumerate(chart_type_elem.findall(".//c:ser", ns)):
        name = f"Series {idx + 1}"
        tx = ser_elem.find(".//c:tx//c:v", ns)
        if tx is not None and tx.text:
            name = tx.text.strip()

        if not cats_done:
            cat = ser_elem.find(".//c:cat", ns)
            if cat is not None:
                str_cache = cat.find(".//c:strCache", ns)
                if str_cache is not None:
                    for pt in str_cache.findall(".//c:pt", ns):
                        v = pt.find("c:v", ns)
                        if v is not None and v.text:
                            categories.append(v.text.strip())
            cats_done = True

        values: List[float] = []
        val = ser_elem.find(".//c:val", ns)
        if val is not None:
            num_cache = val.find(".//c:numCache", ns)
            if num_cache is not None:
                for pt in num_cache.findall(".//c:pt", ns):
                    v = pt.find("c:v", ns)
                    if v is not None and v.text:
                        try:
                            values.append(float(v.text))
                        except ValueError:
                            values.append(0.0)

        if values:
            series.append({"name": name, "values": values})

    return categories, series


def _format_chart_simple(chart_data: Dict) -> str:
    """Fallback chart formatting when no chart_service is available."""
    lines: List[str] = []
    title = chart_data.get("title") or chart_data.get("type", "Chart")
    lines.append(f"\n[Chart: {title}]")
    for s in chart_data.get("series", []):
        vals = ", ".join(str(v) for v in s.get("values", []))
        lines.append(f"  {s['name']}: {vals}")
    return "\n".join(lines) + "\n"


__all__ = ["parse_hwpx_section", "HwpxSupplementary"]
