# contextifier_new/handlers/hwpx/_section.py
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
import os
import xml.etree.ElementTree as ET
import zipfile
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from contextifier_new.handlers.hwpx._constants import (
    BINDATA_PREFIX,
    CHART_PREFIXES,
    HWPX_NAMESPACES,
    OOXML_CHART_NS,
    CHART_TYPE_MAP,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from contextifier_new.handlers.hwpx._table import parse_hwpx_table

if TYPE_CHECKING:
    from contextifier_new.services.image_service import ImageService
    from contextifier_new.services.chart_service import ChartService

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def parse_hwpx_section(
    section_xml: bytes,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    *,
    image_service: Optional["ImageService"] = None,
    chart_service: Optional["ChartService"] = None,
    processed_images: Optional[Set[str]] = None,
) -> str:
    """
    Parse a single HWPX section XML and return its text.

    Tables are rendered inline (HTML or plain-text depending on shape),
    images are saved via *image_service* and replaced by image tags,
    and charts are formatted as text blocks.

    Args:
        section_xml: Raw XML bytes of the section.
        zf: The parent HWPX ZipFile (for reading BinData images / charts).
        bin_item_map: Mapping from ``binaryItemIDRef`` to ZIP path.
        image_service: Optional — for saving embedded images.
        chart_service: Optional — for formatting chart data.
        processed_images: A set that accumulates processed image paths
                          (to avoid duplicates across sections).

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

    ns = HWPX_NAMESPACES
    parts: List[str] = []

    # Walk all <hp:p> paragraphs under any parent (hs:sec, or root itself)
    paragraphs = root.findall(".//hp:p", ns)
    for para in paragraphs:
        para_text = _process_paragraph(
            para, zf, bin_item_map, ns,
            image_service=image_service,
            chart_service=chart_service,
            processed_images=processed_images,
        )
        if para_text and para_text.strip():
            parts.append(para_text)

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Paragraph Processing
# ═══════════════════════════════════════════════════════════════════════════════

def _process_paragraph(
    para: ET.Element,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    ns: Dict[str, str],
    *,
    image_service: Optional["ImageService"],
    chart_service: Optional["ChartService"],
    processed_images: Set[str],
) -> str:
    """Process a single ``<hp:p>`` paragraph, returning its text."""
    parts: List[str] = []

    for child in para:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "run":
            run_text = _process_run(
                child, zf, bin_item_map, ns,
                image_service=image_service,
                processed_images=processed_images,
            )
            if run_text:
                parts.append(run_text)

        elif tag == "tbl":
            table_text = parse_hwpx_table(child, ns)
            if table_text:
                parts.append(f"\n{table_text}\n")

        elif tag == "switch":
            switch_text = _process_switch(
                child, zf, bin_item_map, ns,
                chart_service=chart_service,
                image_service=image_service,
                processed_images=processed_images,
            )
            if switch_text:
                parts.append(switch_text)

        elif tag == "ctrl":
            ctrl_text = _process_ctrl(
                child, zf, bin_item_map, ns,
                image_service=image_service,
                processed_images=processed_images,
            )
            if ctrl_text:
                parts.append(ctrl_text)

        elif tag == "pic":
            img_text = _process_picture_element(
                child, zf, bin_item_map, ns,
                image_service=image_service,
                processed_images=processed_images,
            )
            if img_text:
                parts.append(img_text)

        elif tag == "chart":
            chart_text = _process_chart_ref(
                child, zf,
                chart_service=chart_service,
            )
            if chart_text:
                parts.append(chart_text)

    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Run / Control / Switch Processing
# ═══════════════════════════════════════════════════════════════════════════════

def _process_run(
    run: ET.Element,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    ns: Dict[str, str],
    *,
    image_service: Optional["ImageService"],
    processed_images: Set[str],
) -> str:
    """Process a single ``<hp:run>`` element."""
    parts: List[str] = []

    for child in run:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "t" and child.text:
            parts.append(child.text)

        elif tag == "tbl":
            table_text = parse_hwpx_table(child, ns)
            if table_text:
                parts.append(f"\n{table_text}\n")

        elif tag == "ctrl":
            ctrl_text = _process_ctrl(
                child, zf, bin_item_map, ns,
                image_service=image_service,
                processed_images=processed_images,
            )
            if ctrl_text:
                parts.append(ctrl_text)

        elif tag == "pic":
            img_text = _process_picture_element(
                child, zf, bin_item_map, ns,
                image_service=image_service,
                processed_images=processed_images,
            )
            if img_text:
                parts.append(img_text)

    return "".join(parts)


def _process_ctrl(
    ctrl: ET.Element,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    ns: Dict[str, str],
    *,
    image_service: Optional["ImageService"],
    processed_images: Set[str],
) -> str:
    """Process ``<hp:ctrl>`` — look for ``<hc:pic>`` children."""
    parts: List[str] = []
    # hc:pic is in the 'hc' namespace
    for pic in ctrl.findall("hc:pic", ns):
        img_text = _process_picture_element(
            pic, zf, bin_item_map, ns,
            image_service=image_service,
            processed_images=processed_images,
        )
        if img_text:
            parts.append(img_text)
    # Also direct hp:pic
    for pic in ctrl.findall("hp:pic", ns):
        img_text = _process_picture_element(
            pic, zf, bin_item_map, ns,
            image_service=image_service,
            processed_images=processed_images,
        )
        if img_text:
            parts.append(img_text)
    return "".join(parts)


def _process_switch(
    switch: ET.Element,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    ns: Dict[str, str],
    *,
    chart_service: Optional["ChartService"],
    image_service: Optional["ImageService"],
    processed_images: Set[str],
) -> str:
    """
    Process ``<hp:switch>`` — iterate cases for charts / nested content.

    The ``<hp:case>`` elements may contain ``<hp:chart>`` or paragraphs.
    """
    parts: List[str] = []

    for case in switch.findall("hp:case", ns):
        for child in case:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "chart":
                chart_text = _process_chart_ref(
                    child, zf, chart_service=chart_service,
                )
                if chart_text:
                    parts.append(chart_text)

            elif tag == "p":
                para_text = _process_paragraph(
                    child, zf, bin_item_map, ns,
                    image_service=image_service,
                    chart_service=chart_service,
                    processed_images=processed_images,
                )
                if para_text:
                    parts.append(para_text)

            elif tag == "pic":
                img_text = _process_picture_element(
                    child, zf, bin_item_map, ns,
                    image_service=image_service,
                    processed_images=processed_images,
                )
                if img_text:
                    parts.append(img_text)

    # Also try <hp:default>
    for default in switch.findall("hp:default", ns):
        for child in default:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "p":
                para_text = _process_paragraph(
                    child, zf, bin_item_map, ns,
                    image_service=image_service,
                    chart_service=chart_service,
                    processed_images=processed_images,
                )
                if para_text:
                    parts.append(para_text)

    return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Image Processing
# ═══════════════════════════════════════════════════════════════════════════════

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

        tag = image_service.save_image(image_data)
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
    t = chart.find(".//c:title//c:tx//c:rich//a:t", ns)
    if t is not None and t.text:
        return t.text.strip()
    return None


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


__all__ = ["parse_hwpx_section"]
