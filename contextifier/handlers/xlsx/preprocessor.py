"""
XlsxPreprocessor — Stage 2: Workbook → PreprocessedData.

Wraps the openpyxl Workbook and pre-extracts file-level resources:
- Chart data from ``xl/charts/chart*.xml`` inside the ZIP
- Image data from ``xl/media/*`` inside the ZIP
- Textbox content from ``xl/drawings/drawing*.xml``
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.errors import PreprocessingError

from contextifier.handlers.xlsx._constants import (
    OOXML_NS,
    CHART_TYPE_MAP,
    SUPPORTED_IMAGE_EXTENSIONS,
    UNSUPPORTED_IMAGE_EXTENSIONS,
)
from contextifier.handlers.xlsx.converter import XlsxConvertedData

logger = logging.getLogger(__name__)


class XlsxPreprocessor(BasePreprocessor):
    """
    Preprocessor for XLSX files.

    Pre-extracts charts, images, and textboxes from the ZIP archive
    so downstream extractors don't need to re-open the file.
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """
        Wrap the openpyxl Workbook and pre-extract resources.

        Args:
            converted_data: ``XlsxConvertedData`` from the Converter
                (contains both Workbook and original file bytes).

        Returns:
            PreprocessedData with:
            - ``content``: the Workbook object
            - ``resources["charts"]``: list of chart dicts from ZIP XML
            - ``resources["images"]``: dict of filename → image bytes from xl/media/
            - ``resources["textboxes"]``: dict of sheet_name → list of text strings
            - ``properties``: sheet_count, sheet_names, etc.
        """
        import openpyxl

        if converted_data is None:
            raise PreprocessingError(
                "Received None as converted data",
                stage="preprocess",
                handler="xlsx",
            )

        # Unwrap the converted data
        if isinstance(converted_data, XlsxConvertedData):
            wb = converted_data.workbook
            file_data = converted_data.file_data
        elif isinstance(converted_data, openpyxl.Workbook):
            wb = converted_data
            file_data = b""
        else:
            raise PreprocessingError(
                f"Expected XlsxConvertedData or Workbook, got {type(converted_data).__name__}",
                stage="preprocess",
                handler="xlsx",
            )

        sheet_names = wb.sheetnames
        sheet_count = len(sheet_names)

        # Pre-extract resources from the ZIP archive
        charts: List[dict] = []
        images: Dict[str, bytes] = {}
        textboxes: Dict[str, List[str]] = {}

        if file_data:
            try:
                charts = _extract_charts_from_zip(file_data)
            except Exception as exc:
                logger.debug("Failed to pre-extract charts: %s", exc)

            try:
                images = _extract_images_from_zip(file_data)
            except Exception as exc:
                logger.debug("Failed to pre-extract images: %s", exc)

            try:
                textboxes = _extract_textboxes_from_zip(file_data)
            except Exception as exc:
                logger.debug("Failed to pre-extract textboxes: %s", exc)

        return PreprocessedData(
            content=wb,
            raw_content=wb,
            encoding="utf-8",
            resources={
                "charts": charts,
                "images": images,
                "textboxes": textboxes,
            },
            properties={
                "sheet_count": sheet_count,
                "sheet_names": sheet_names,
                "chart_count": len(charts),
                "image_count": len(images),
            },
        )

    def get_format_name(self) -> str:
        return "xlsx"


# ═══════════════════════════════════════════════════════════════════════════════
# ZIP-level extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_charts_from_zip(file_data: bytes) -> List[dict]:
    """
    Extract all chart data from ``xl/charts/chart*.xml`` in the ZIP.

    Returns a list of chart dicts with keys:
        chart_type, title, categories, series
    """
    charts: List[dict] = []

    try:
        with zipfile.ZipFile(io.BytesIO(file_data)) as zf:
            chart_files = sorted(
                n for n in zf.namelist()
                if n.startswith("xl/charts/chart") and n.endswith(".xml")
            )
            for chart_file in chart_files:
                try:
                    xml_data = zf.read(chart_file)
                    chart = _parse_chart_xml(xml_data)
                    if chart:
                        charts.append(chart)
                except Exception as exc:
                    logger.debug("Failed to parse chart %s: %s", chart_file, exc)
    except zipfile.BadZipFile:
        pass

    return charts


def _parse_chart_xml(xml_data: bytes) -> Optional[dict]:
    """
    Parse a single OOXML chart XML file into a chart dict.

    Returns:
        dict with chart_type, title, categories, series or None on failure.
    """
    # Handle BOM
    if xml_data.startswith(b"\xef\xbb\xbf"):
        xml_data = xml_data[3:]

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return None

    ns_c = OOXML_NS["c"]
    ns_a = OOXML_NS["a"]

    # Find <c:chart> element
    chart_el = root.find(f"{{{ns_c}}}chart")
    if chart_el is None:
        chart_el = root

    # Extract title
    title = _extract_chart_title(chart_el, ns_c, ns_a)

    # Find plot area
    plot_area = chart_el.find(f"{{{ns_c}}}plotArea")
    if plot_area is None:
        return None

    # Detect chart type and extract series
    chart_type = None
    series_list: List[dict] = []
    categories: List[str] = []

    for tag_local, display_name in CHART_TYPE_MAP.items():
        chart_type_el = plot_area.find(f"{{{ns_c}}}{tag_local}")
        if chart_type_el is not None:
            chart_type = display_name
            series_list, categories = _extract_series(chart_type_el, ns_c, tag_local)
            break

    if chart_type is None:
        return None

    return {
        "chart_type": chart_type,
        "title": title or "",
        "categories": categories,
        "series": series_list,
    }


def _extract_chart_title(chart_el: ET.Element, ns_c: str, ns_a: str) -> Optional[str]:
    """Extract chart title from c:title element."""
    title_el = chart_el.find(f"{{{ns_c}}}title")
    if title_el is None:
        return None

    # Try rich text: c:title/c:tx/c:rich/a:p/a:r/a:t
    rich = title_el.find(f"{{{ns_c}}}tx/{{{ns_c}}}rich")
    if rich is not None:
        parts = []
        for t_el in rich.iter(f"{{{ns_a}}}t"):
            if t_el.text:
                parts.append(t_el.text)
        if parts:
            return " ".join(parts).strip()

    # Try string reference: c:title/c:tx/c:strRef/c:strCache/c:pt/c:v
    str_cache = title_el.find(
        f"{{{ns_c}}}tx/{{{ns_c}}}strRef/{{{ns_c}}}strCache"
    )
    if str_cache is not None:
        pt = str_cache.find(f"{{{ns_c}}}pt/{{{ns_c}}}v")
        if pt is not None and pt.text:
            return pt.text.strip()

    return None


def _extract_series(
    chart_type_el: ET.Element,
    ns_c: str,
    chart_tag: str,
) -> tuple:
    """
    Extract series data and categories from a chart type element.

    Returns:
        (series_list, categories) where series_list is a list of
        dicts with 'name' and 'values' keys.
    """
    series_list: List[dict] = []
    categories: List[str] = []
    is_scatter = chart_tag in ("scatterChart", "bubbleChart")

    for ser in chart_type_el.findall(f"{{{ns_c}}}ser"):
        # Series name
        name = None
        tx = ser.find(f"{{{ns_c}}}tx")
        if tx is not None:
            # Try strRef
            str_cache = tx.find(f"{{{ns_c}}}strRef/{{{ns_c}}}strCache")
            if str_cache is not None:
                pt = str_cache.find(f"{{{ns_c}}}pt/{{{ns_c}}}v")
                if pt is not None and pt.text:
                    name = pt.text
            # Try direct value
            if name is None:
                v_el = tx.find(f"{{{ns_c}}}v")
                if v_el is not None and v_el.text:
                    name = v_el.text

        # Categories (from first series only)
        if not categories:
            cat_el = ser.find(f"{{{ns_c}}}cat")
            if cat_el is not None:
                categories = _extract_cache_values(cat_el, ns_c)

        # Values
        val_tag = f"{{{ns_c}}}yVal" if is_scatter else f"{{{ns_c}}}val"
        val_el = ser.find(val_tag)
        values: List[Any] = []
        if val_el is not None:
            values = _extract_cache_values(val_el, ns_c, numeric=True)

        series_list.append({"name": name, "values": values})

    return series_list, categories


def _extract_cache_values(
    parent: ET.Element,
    ns_c: str,
    *,
    numeric: bool = False,
) -> list:
    """Extract cached values from strCache or numCache."""
    # Try strCache first
    cache = parent.find(f"{{{ns_c}}}strCache")
    if cache is None:
        cache = parent.find(f"{{{ns_c}}}numCache")
        if cache is not None:
            numeric = True

    if cache is None:
        return []

    values = []
    for pt in cache.findall(f"{{{ns_c}}}pt"):
        v_el = pt.find(f"{{{ns_c}}}v")
        if v_el is not None and v_el.text:
            if numeric:
                try:
                    values.append(float(v_el.text))
                except ValueError:
                    values.append(v_el.text)
            else:
                values.append(v_el.text)
        else:
            values.append("")

    return values


def _extract_images_from_zip(file_data: bytes) -> Dict[str, bytes]:
    """
    Extract image files from ``xl/media/`` in the ZIP.

    Returns dict of filename → image bytes, filtered to supported formats.
    """
    images: Dict[str, bytes] = {}

    try:
        with zipfile.ZipFile(io.BytesIO(file_data)) as zf:
            for name in zf.namelist():
                if not name.startswith("xl/media/"):
                    continue
                ext = os.path.splitext(name)[1].lower()
                if ext in UNSUPPORTED_IMAGE_EXTENSIONS:
                    continue
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    try:
                        images[name] = zf.read(name)
                    except Exception:
                        pass
    except zipfile.BadZipFile:
        pass

    return images


def _extract_textboxes_from_zip(file_data: bytes) -> Dict[str, List[str]]:
    """
    Extract textbox content from drawing XML files.

    Builds a sheet→drawing mapping via workbook.xml.rels and sheet*.xml.rels,
    then extracts text from ``<xdr:sp>/<xdr:txBody>/<a:p>/<a:r>/<a:t>`` elements.

    Returns dict of sheet_name → list of textbox strings.
    """
    ns_xdr = OOXML_NS["xdr"]
    ns_a = OOXML_NS["a"]
    ns_r = OOXML_NS["r"]

    textboxes: Dict[str, List[str]] = {}

    try:
        with zipfile.ZipFile(io.BytesIO(file_data)) as zf:
            # Build sheet index → drawing file mapping
            sheet_drawing_map = _build_sheet_drawing_map(zf)

            for sheet_name, drawing_path in sheet_drawing_map.items():
                if drawing_path not in zf.namelist():
                    continue
                try:
                    drawing_xml = zf.read(drawing_path)
                    texts = _parse_drawing_textboxes(drawing_xml, ns_xdr, ns_a)
                    if texts:
                        textboxes[sheet_name] = texts
                except Exception as exc:
                    logger.debug("Failed to parse textboxes from %s: %s", drawing_path, exc)
    except zipfile.BadZipFile:
        pass

    return textboxes


def _build_sheet_drawing_map(zf: zipfile.ZipFile) -> Dict[str, str]:
    """
    Map sheet names to drawing XML file paths via OOXML relationships.

    Follows: workbook.xml → sheet names + rIds → sheet*.xml → sheet*.xml.rels → drawing*.xml
    """
    ns_r = OOXML_NS["r"]
    ns_ss = OOXML_NS["ss"]
    ns_pkg = OOXML_NS["pkg"]
    mapping: Dict[str, str] = {}

    try:
        # Read workbook.xml for sheet names and rIds
        if "xl/workbook.xml" not in zf.namelist():
            return mapping
        wb_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = wb_xml.findall(f"{{{ns_ss}}}sheets/{{{ns_ss}}}sheet")

        # Read workbook.xml.rels for rId → sheet file mapping
        if "xl/_rels/workbook.xml.rels" not in zf.namelist():
            return mapping
        wb_rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

        rid_to_target: Dict[str, str] = {}
        for rel in wb_rels.findall(f"{{{ns_pkg}}}Relationship"):
            rid = rel.get("Id", "")
            target = rel.get("Target", "")
            if rid and target:
                rid_to_target[rid] = target

        for sheet in sheets:
            sheet_name = sheet.get("name", "")
            rid = sheet.get(f"{{{ns_r}}}id", "")
            if not sheet_name or not rid:
                continue

            target = rid_to_target.get(rid, "")
            if not target:
                continue

            # Resolve relative path: target is relative to xl/
            sheet_path = f"xl/{target}" if not target.startswith("/") else target.lstrip("/")
            rels_path = sheet_path.replace(
                os.path.basename(sheet_path),
                f"_rels/{os.path.basename(sheet_path)}.rels",
            )

            if rels_path not in zf.namelist():
                continue

            try:
                sheet_rels = ET.fromstring(zf.read(rels_path))
                for rel in sheet_rels.findall(f"{{{ns_pkg}}}Relationship"):
                    rel_target = rel.get("Target", "")
                    rel_type = rel.get("Type", "")
                    if "drawing" in rel_type.lower() and rel_target:
                        drawing_path = f"xl/{rel_target}" if not rel_target.startswith("/") else rel_target.lstrip("/")
                        # Normalize path (remove ../)
                        drawing_path = os.path.normpath(drawing_path).replace("\\", "/")
                        mapping[sheet_name] = drawing_path
                        break
            except Exception:
                pass

    except Exception as exc:
        logger.debug("Failed to build sheet-drawing map: %s", exc)

    return mapping


def _parse_drawing_textboxes(
    drawing_xml: bytes,
    ns_xdr: str,
    ns_a: str,
) -> List[str]:
    """Extract text from textbox shapes in a drawing XML."""
    texts: List[str] = []

    try:
        root = ET.fromstring(drawing_xml)
    except ET.ParseError:
        return texts

    # Find all shape elements: xdr:sp
    for sp in root.iter(f"{{{ns_xdr}}}sp"):
        tx_body = sp.find(f"{{{ns_xdr}}}txBody")
        if tx_body is None:
            tx_body = sp.find(f"{{{ns_a}}}txBody")
        if tx_body is None:
            continue

        parts: List[str] = []
        for p in tx_body.findall(f"{{{ns_a}}}p"):
            line_parts: List[str] = []
            for r in p.findall(f"{{{ns_a}}}r"):
                t = r.find(f"{{{ns_a}}}t")
                if t is not None and t.text:
                    line_parts.append(t.text)
            if line_parts:
                parts.append("".join(line_parts))

        text = "\n".join(parts).strip()
        if text:
            texts.append(text)

    return texts


__all__ = ["XlsxPreprocessor"]
