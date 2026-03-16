# tests/test_hwpx_handler.py
"""
Comprehensive tests for the HWPX handler pipeline.

Covers:
- _constants (namespace dicts, paths, magic bytes)
- _table (grid building, HTML rendering, 1×1 / single-col / multi-col)
- _section (paragraph/run/image/chart/table processing)
- converter (ZIP validation, open, close)
- preprocessor (bin_item_map, section discovery)
- metadata_extractor (header.xml, version.xml, manifest.xml)
- content_extractor (multi-section, remaining images)
- handler (wiring, full pipeline)
"""

from __future__ import annotations

import io
import struct
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Imports under test
# ═══════════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.hwpx._constants import (
    BINDATA_PREFIX,
    CHART_PREFIXES,
    CHART_TYPE_MAP,
    HEADER_FILE_PATHS,
    HEADER_PATH,
    HPF_PATH,
    HWPX_NAMESPACES,
    MANIFEST_PATH,
    OOXML_CHART_NS,
    OPF_NAMESPACES,
    SECTION_PREFIX,
    SUPPORTED_IMAGE_EXTENSIONS,
    VERSION_PATH,
    ZIP_MAGIC,
)
from contextifier_new.handlers.hwpx._table import (
    parse_hwpx_table,
    _build_grid,
    _extract_cell_text,
    _parse_cell_position,
    _parse_cell_span,
    _render_html,
)
from contextifier_new.handlers.hwpx._section import (
    parse_hwpx_section,
    _process_paragraph,
    _process_run,
    _process_picture_element,
    _resolve_zip_path,
    _parse_ooxml_chart,
)
from contextifier_new.handlers.hwpx.converter import (
    HwpxConverter,
    HwpxConvertedData,
)
from contextifier_new.handlers.hwpx.preprocessor import (
    HwpxPreprocessor,
    parse_bin_item_map,
    find_section_paths,
)
from contextifier_new.handlers.hwpx.metadata_extractor import (
    HwpxMetadataExtractor,
)
from contextifier_new.handlers.hwpx.content_extractor import (
    HwpxContentExtractor,
)
from contextifier_new.handlers.hwpx.handler import HWPXHandler
from contextifier_new.types import (
    DocumentMetadata,
    ExtractionResult,
    PreprocessedData,
)
from contextifier_new.config import ProcessingConfig
from contextifier_new.errors import ConversionError


# ═══════════════════════════════════════════════════════════════════════════════
# Test Helpers
# ═══════════════════════════════════════════════════════════════════════════════

NS = HWPX_NAMESPACES

# Shorthand namespace URIs
_HP = HWPX_NAMESPACES["hp"]
_HC = HWPX_NAMESPACES["hc"]
_HH = HWPX_NAMESPACES["hh"]
_HS = HWPX_NAMESPACES["hs"]


def _make_zip(files: Dict[str, bytes | str]) -> bytes:
    """Create a ZIP archive in memory with the given files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(name, content)
    return buf.getvalue()


def _make_section_xml(paragraphs: list[str]) -> str:
    """Build a minimal HWPX section XML with text paragraphs."""
    paras = ""
    for text in paragraphs:
        paras += (
            f'<hp:p xmlns:hp="{_HP}">'
            f'<hp:run><hp:t>{text}</hp:t></hp:run>'
            f'</hp:p>'
        )
    return (
        f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" '
        f'xmlns:hc="{_HC}" xmlns:hh="{_HH}">'
        f'{paras}'
        f'</hs:sec>'
    )


def _make_table_xml(rows: list[list[str]], row_cnt: int = 0, col_cnt: int = 0) -> str:
    """Build a minimal HWPX table XML string."""
    if row_cnt == 0:
        row_cnt = len(rows)
    if col_cnt == 0:
        col_cnt = len(rows[0]) if rows else 0

    trs = ""
    for r, row in enumerate(rows):
        tcs = ""
        for c, text in enumerate(row):
            tcs += (
                f'<hp:tc xmlns:hp="{_HP}">'
                f'<hp:cellAddr colAddr="{c}" rowAddr="{r}"/>'
                f'<hp:cellSpan colSpan="1" rowSpan="1"/>'
                f'<hp:subList><hp:p><hp:run><hp:t>{text}</hp:t></hp:run></hp:p></hp:subList>'
                f'</hp:tc>'
            )
        trs += f'<hp:tr xmlns:hp="{_HP}">{tcs}</hp:tr>'

    return (
        f'<hp:tbl xmlns:hp="{_HP}" rowCnt="{row_cnt}" colCnt="{col_cnt}">'
        f'{trs}'
        f'</hp:tbl>'
    )


def _make_opf_manifest(items: Dict[str, str]) -> str:
    """Build OPF content.hpf XML."""
    entries = ""
    for item_id, href in items.items():
        entries += f'<opf:item id="{item_id}" href="{href}"/>'
    return (
        f'<opf:package xmlns:opf="{OPF_NAMESPACES["opf"]}">'
        f'<opf:manifest>{entries}</opf:manifest>'
        f'</opf:package>'
    )


def _make_header_xml(props: Dict[str, str]) -> str:
    """Build a minimal Contents/header.xml."""
    children = ""
    for tag, text in props.items():
        children += f'<hh:{tag} xmlns:hh="{_HH}">{text}</hh:{tag}>'
    return (
        f'<hh:head xmlns:hh="{_HH}">'
        f'<hh:docInfo>{children}</hh:docInfo>'
        f'</hh:head>'
    )


def _make_version_xml(text: str = "", **attrs) -> str:
    """Build a version.xml."""
    attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f'<version {attr_str}>{text}</version>'


def _make_file_context(data: bytes) -> dict:
    """Create a minimal FileContext dict."""
    return {
        "file_path": "/tmp/test.hwpx",
        "file_name": "test.hwpx",
        "file_extension": "hwpx",
        "file_category": "document",
        "file_data": data,
        "file_stream": io.BytesIO(data),
        "file_size": len(data),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_zip_magic(self):
        assert ZIP_MAGIC == b"PK\x03\x04"

    def test_hwpx_namespaces_has_hp_hc_hh(self):
        assert "hp" in HWPX_NAMESPACES
        assert "hc" in HWPX_NAMESPACES
        assert "hh" in HWPX_NAMESPACES

    def test_opf_namespaces(self):
        assert "opf" in OPF_NAMESPACES

    def test_standard_paths(self):
        assert HPF_PATH == "Contents/content.hpf"
        assert HEADER_PATH == "Contents/header.xml"
        assert VERSION_PATH == "version.xml"

    def test_supported_image_extensions(self):
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS

    def test_chart_type_map(self):
        assert "barChart" in CHART_TYPE_MAP
        assert "pieChart" in CHART_TYPE_MAP

    def test_header_file_paths(self):
        assert HEADER_PATH in HEADER_FILE_PATHS

    def test_section_prefix(self):
        assert SECTION_PREFIX == "Contents/section"

    def test_bindata_prefix(self):
        assert BINDATA_PREFIX == "BinData/"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Table Parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestTable:
    def test_1x1_container(self):
        xml = _make_table_xml([["Hello World"]], 1, 1)
        result = parse_hwpx_table(ET.fromstring(xml))
        assert result == "Hello World"

    def test_single_column(self):
        xml = _make_table_xml([["Alpha"], ["Beta"], ["Gamma"]], 3, 1)
        result = parse_hwpx_table(ET.fromstring(xml))
        assert "Alpha" in result
        assert "Beta" in result
        assert "Gamma" in result
        assert "\n\n" in result

    def test_multi_column_html(self):
        xml = _make_table_xml([["A", "B"], ["C", "D"]], 2, 2)
        result = parse_hwpx_table(ET.fromstring(xml))
        assert "<table>" in result
        assert "<tr>" in result
        assert "<td>A</td>" in result
        assert "<td>D</td>" in result

    def test_empty_table(self):
        xml = f'<hp:tbl xmlns:hp="{_HP}" rowCnt="0" colCnt="0"></hp:tbl>'
        result = parse_hwpx_table(ET.fromstring(xml))
        assert result == ""

    def test_html_escaping(self):
        """Verify that cell text with special chars is escaped in HTML output."""
        # Build the element programmatically to avoid XML escaping issues
        tbl = ET.Element(f"{{{_HP}}}tbl", attrib={"rowCnt": "1", "colCnt": "2"})
        tr = ET.SubElement(tbl, f"{{{_HP}}}tr")
        for c, text in enumerate(["A<B", "C>D"]):
            tc = ET.SubElement(tr, f"{{{_HP}}}tc")
            addr = ET.SubElement(tc, f"{{{_HP}}}cellAddr", attrib={"colAddr": str(c), "rowAddr": "0"})
            span = ET.SubElement(tc, f"{{{_HP}}}cellSpan", attrib={"colSpan": "1", "rowSpan": "1"})
            sub = ET.SubElement(tc, f"{{{_HP}}}subList")
            p = ET.SubElement(sub, f"{{{_HP}}}p")
            run = ET.SubElement(p, f"{{{_HP}}}run")
            t = ET.SubElement(run, f"{{{_HP}}}t")
            t.text = text

        result = parse_hwpx_table(tbl)
        assert "&lt;" in result
        assert "&gt;" in result

    def test_rowspan_colspan(self):
        """Table with a merged cell spanning 2 columns."""
        xml = (
            f'<hp:tbl xmlns:hp="{_HP}" rowCnt="2" colCnt="2">'
            f'<hp:tr>'
            f'<hp:tc>'
            f'<hp:cellAddr colAddr="0" rowAddr="0"/>'
            f'<hp:cellSpan colSpan="2" rowSpan="1"/>'
            f'<hp:subList><hp:p><hp:run><hp:t>Merged</hp:t></hp:run></hp:p></hp:subList>'
            f'</hp:tc>'
            f'</hp:tr>'
            f'<hp:tr>'
            f'<hp:tc>'
            f'<hp:cellAddr colAddr="0" rowAddr="1"/>'
            f'<hp:cellSpan colSpan="1" rowSpan="1"/>'
            f'<hp:subList><hp:p><hp:run><hp:t>X</hp:t></hp:run></hp:p></hp:subList>'
            f'</hp:tc>'
            f'<hp:tc>'
            f'<hp:cellAddr colAddr="1" rowAddr="1"/>'
            f'<hp:cellSpan colSpan="1" rowSpan="1"/>'
            f'<hp:subList><hp:p><hp:run><hp:t>Y</hp:t></hp:run></hp:p></hp:subList>'
            f'</hp:tc>'
            f'</hp:tr>'
            f'</hp:tbl>'
        )
        result = parse_hwpx_table(ET.fromstring(xml))
        assert 'colspan="2"' in result
        assert "Merged" in result
        assert "X" in result
        assert "Y" in result

    def test_cell_position_parsing(self):
        xml = (
            f'<hp:tc xmlns:hp="{_HP}">'
            f'<hp:cellAddr colAddr="3" rowAddr="2"/>'
            f'</hp:tc>'
        )
        tc = ET.fromstring(xml)
        row, col = _parse_cell_position(tc, NS)
        assert row == 2
        assert col == 3

    def test_cell_span_defaults(self):
        xml = f'<hp:tc xmlns:hp="{_HP}"></hp:tc>'
        tc = ET.fromstring(xml)
        rowspan, colspan = _parse_cell_span(tc, NS)
        assert rowspan == 1
        assert colspan == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Section Parser
# ═══════════════════════════════════════════════════════════════════════════════

class TestSection:
    def test_simple_text(self):
        section = _make_section_xml(["Hello", "World"])
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({})), "r")
        result = parse_hwpx_section(section.encode(), zf, {})
        assert "Hello" in result
        assert "World" in result
        zf.close()

    def test_empty_section(self):
        xml = f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}"></hs:sec>'
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({})), "r")
        result = parse_hwpx_section(xml.encode(), zf, {})
        assert result.strip() == ""
        zf.close()

    def test_invalid_xml(self):
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({})), "r")
        result = parse_hwpx_section(b"not xml!!!", zf, {})
        assert result == ""
        zf.close()

    def test_inline_table(self):
        """Section with an inline table inside a paragraph."""
        table_xml = (
            f'<hp:tbl rowCnt="1" colCnt="2">'
            f'<hp:tr>'
            f'<hp:tc><hp:cellAddr colAddr="0" rowAddr="0"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
            f'<hp:subList><hp:p><hp:run><hp:t>A</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
            f'<hp:tc><hp:cellAddr colAddr="1" rowAddr="0"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
            f'<hp:subList><hp:p><hp:run><hp:t>B</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
            f'</hp:tr>'
            f'</hp:tbl>'
        )
        xml = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run><hp:t>Before</hp:t></hp:run>{table_xml}</hp:p>'
            f'</hs:sec>'
        )
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({})), "r")
        result = parse_hwpx_section(xml.encode(), zf, {})
        assert "Before" in result
        assert "<table>" in result
        zf.close()

    def test_image_extraction(self):
        """Section with an inline image via binaryItemIDRef."""
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG
        bin_item_map = {"image1": "BinData/image1.png"}
        files = {"BinData/image1.png": image_data}

        xml = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run>'
            f'<hp:ctrl><hc:pic><hc:img binaryItemIDRef="image1"/></hc:pic></hp:ctrl>'
            f'</hp:run></hp:p>'
            f'</hs:sec>'
        )

        zip_data = _make_zip(files)
        zf = zipfile.ZipFile(io.BytesIO(zip_data), "r")
        mock_img_service = MagicMock()
        mock_img_service.save_image.return_value = "[Image:test.png]"

        result = parse_hwpx_section(
            xml.encode(), zf, bin_item_map, image_service=mock_img_service,
        )
        assert "[Image:test.png]" in result
        mock_img_service.save_image.assert_called_once()
        zf.close()

    def test_image_dedup(self):
        """Same image referenced twice should be processed only once."""
        image_data = b"\x89PNG" + b"\x00" * 50
        bin_item_map = {"img1": "BinData/img1.png"}
        files = {"BinData/img1.png": image_data}

        xml = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run>'
            f'<hp:ctrl><hc:pic><hc:img binaryItemIDRef="img1"/></hc:pic></hp:ctrl>'
            f'</hp:run></hp:p>'
            f'<hp:p><hp:run>'
            f'<hp:ctrl><hc:pic><hc:img binaryItemIDRef="img1"/></hc:pic></hp:ctrl>'
            f'</hp:run></hp:p>'
            f'</hs:sec>'
        )

        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        mock_img = MagicMock()
        mock_img.save_image.return_value = "[Image:dup.png]"

        result = parse_hwpx_section(
            xml.encode(), zf, bin_item_map, image_service=mock_img,
        )
        # save_image called only once due to dedup
        assert mock_img.save_image.call_count == 1
        zf.close()

    def test_missing_image_no_crash(self):
        """Image ref pointing to non-existent file should not crash."""
        bin_item_map = {"img1": "BinData/missing.png"}
        xml = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run>'
            f'<hp:ctrl><hc:pic><hc:img binaryItemIDRef="img1"/></hc:pic></hp:ctrl>'
            f'</hp:run></hp:p>'
            f'</hs:sec>'
        )
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({})), "r")
        mock_img = MagicMock()
        result = parse_hwpx_section(
            xml.encode(), zf, bin_item_map, image_service=mock_img,
        )
        mock_img.save_image.assert_not_called()
        zf.close()

    def test_direct_hp_pic(self):
        """Image via direct <hp:pic> (variant syntax)."""
        image_data = b"\x89PNG" + b"\x00" * 50
        bin_item_map = {"img2": "BinData/img2.png"}
        files = {"BinData/img2.png": image_data}

        xml = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run>'
            f'<hp:pic><hc:img binaryItemIDRef="img2"/></hp:pic>'
            f'</hp:run></hp:p>'
            f'</hs:sec>'
        )
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        mock_img = MagicMock()
        mock_img.save_image.return_value = "[Image:pic2.png]"

        result = parse_hwpx_section(
            xml.encode(), zf, bin_item_map, image_service=mock_img,
        )
        assert "[Image:pic2.png]" in result
        zf.close()

    def test_resolve_zip_path_with_prefix(self):
        """Image href resolved with Contents/ prefix."""
        files = {"Contents/BinData/img.png": b"PNG"}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        assert _resolve_zip_path(zf, "BinData/img.png") == "Contents/BinData/img.png"
        zf.close()

    def test_resolve_zip_path_direct(self):
        files = {"BinData/img.png": b"PNG"}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        assert _resolve_zip_path(zf, "BinData/img.png") == "BinData/img.png"
        zf.close()

    def test_resolve_zip_path_missing(self):
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({})), "r")
        assert _resolve_zip_path(zf, "nonexistent.png") is None
        zf.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Chart Parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestChart:
    def _make_chart_xml(self) -> bytes:
        """Create a minimal OOXML bar chart."""
        ns_c = OOXML_CHART_NS["c"]
        ns_a = OOXML_CHART_NS["a"]
        return (
            f'<c:chartSpace xmlns:c="{ns_c}" xmlns:a="{ns_a}">'
            f'<c:chart>'
            f'<c:title><c:tx><c:rich><a:p><a:r><a:t>Sales</a:t></a:r></a:p></c:rich></c:tx></c:title>'
            f'<c:plotArea>'
            f'<c:barChart>'
            f'<c:ser>'
            f'<c:tx><c:strRef><c:strCache><c:ptCount val="1"/><c:pt idx="0"><c:v>Revenue</c:v></c:pt></c:strCache></c:strRef></c:tx>'
            f'<c:cat><c:strRef><c:strCache><c:ptCount val="2"/><c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt></c:strCache></c:strRef></c:cat>'
            f'<c:val><c:numRef><c:numCache><c:ptCount val="2"/><c:pt idx="0"><c:v>100</c:v></c:pt><c:pt idx="1"><c:v>200</c:v></c:pt></c:numCache></c:numRef></c:val>'
            f'</c:ser>'
            f'</c:barChart>'
            f'</c:plotArea>'
            f'</c:chart>'
            f'</c:chartSpace>'
        ).encode()

    def test_parse_ooxml_chart(self):
        data = self._make_chart_xml()
        result = _parse_ooxml_chart(data)
        assert result is not None
        assert result["type"] == "Bar Chart"
        assert result["title"] == "Sales"
        assert "Q1" in result["categories"]
        assert len(result["series"]) == 1
        assert result["series"][0]["name"] == "Revenue"
        assert result["series"][0]["values"] == [100.0, 200.0]

    def test_parse_ooxml_chart_invalid_xml(self):
        result = _parse_ooxml_chart(b"not xml!!!")
        assert result is None

    def test_parse_ooxml_chart_no_chart_elem(self):
        result = _parse_ooxml_chart(b"<root/>")
        assert result is None

    def test_chart_in_section(self):
        """Chart referenced from section with chartIDRef."""
        chart_xml = self._make_chart_xml()
        files = {"Chart/chart1.xml": chart_xml}

        xml = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p>'
            f'<hp:switch>'
            f'<hp:case>'
            f'<hp:chart chartIDRef="Chart/chart1.xml"/>'
            f'</hp:case>'
            f'</hp:switch>'
            f'</hp:p>'
            f'</hs:sec>'
        )

        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        result = parse_hwpx_section(xml.encode(), zf, {})
        assert "Sales" in result or "Chart" in result
        zf.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Converter
# ═══════════════════════════════════════════════════════════════════════════════

class TestConverter:
    def test_convert_valid_zip(self):
        zip_data = _make_zip({"test.txt": b"hello"})
        fc = _make_file_context(zip_data)
        converter = HwpxConverter()
        result = converter.convert(fc)
        assert isinstance(result, HwpxConvertedData)
        assert isinstance(result.zf, zipfile.ZipFile)
        assert result.file_data == zip_data
        result.zf.close()

    def test_convert_empty_data(self):
        fc = _make_file_context(b"")
        converter = HwpxConverter()
        with pytest.raises(ConversionError):
            converter.convert(fc)

    def test_convert_bad_zip(self):
        fc = _make_file_context(b"not a zip file")
        converter = HwpxConverter()
        with pytest.raises(ConversionError):
            converter.convert(fc)

    def test_validate_valid(self):
        zip_data = _make_zip({"x": b""})
        fc = _make_file_context(zip_data)
        converter = HwpxConverter()
        assert converter.validate(fc) is True

    def test_validate_invalid(self):
        fc = _make_file_context(b"\xd0\xcf\x11\xe0")  # OLE magic
        converter = HwpxConverter()
        assert converter.validate(fc) is False

    def test_validate_too_short(self):
        fc = _make_file_context(b"PK")
        converter = HwpxConverter()
        assert converter.validate(fc) is False

    def test_close(self):
        zip_data = _make_zip({"t": b""})
        fc = _make_file_context(zip_data)
        converter = HwpxConverter()
        result = converter.convert(fc)
        converter.close(result)
        # After close, ZipFile should be closed

    def test_close_non_hwpx(self):
        converter = HwpxConverter()
        mock_obj = MagicMock()
        converter.close(mock_obj)
        mock_obj.close.assert_called_once()

    def test_get_format_name(self):
        assert HwpxConverter().get_format_name() == "hwpx"

    def test_namedtuple_fields(self):
        assert HwpxConvertedData._fields == ("zf", "file_data")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Preprocessor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    def test_preprocess_basic(self):
        opf = _make_opf_manifest({"img1": "BinData/img1.png"})
        section = _make_section_xml(["hello"])
        files = {
            HPF_PATH: opf,
            "Contents/section0.xml": section,
        }
        zip_data = _make_zip(files)
        zf = zipfile.ZipFile(io.BytesIO(zip_data), "r")

        prep = HwpxPreprocessor()
        result = prep.preprocess(HwpxConvertedData(zf=zf, file_data=zip_data))

        assert isinstance(result, PreprocessedData)
        assert result.content is zf
        assert result.resources["bin_item_map"] == {"img1": "BinData/img1.png"}
        assert result.properties["section_count"] == 1
        assert len(result.properties["section_paths"]) == 1
        zf.close()

    def test_preprocess_no_opf(self):
        """Preprocessor works even without content.hpf."""
        files = {"Contents/section0.xml": _make_section_xml(["hi"])}
        zip_data = _make_zip(files)
        zf = zipfile.ZipFile(io.BytesIO(zip_data), "r")

        prep = HwpxPreprocessor()
        result = prep.preprocess(HwpxConvertedData(zf=zf, file_data=zip_data))
        assert result.resources["bin_item_map"] == {}
        zf.close()

    def test_preprocess_bare_zipfile(self):
        """Preprocessor accepts a bare ZipFile (not HwpxConvertedData)."""
        files = {"Contents/section0.xml": _make_section_xml(["x"])}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")

        prep = HwpxPreprocessor()
        result = prep.preprocess(zf)
        assert result.properties["section_count"] == 1
        zf.close()

    def test_preprocess_none_raises(self):
        prep = HwpxPreprocessor()
        with pytest.raises(Exception):
            prep.preprocess(None)

    def test_get_format_name(self):
        assert HwpxPreprocessor().get_format_name() == "hwpx"


class TestBinItemMap:
    def test_parse_basic(self):
        opf = _make_opf_manifest({"img1": "BinData/img1.png", "img2": "BinData/img2.jpg"})
        files = {HPF_PATH: opf}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        result = parse_bin_item_map(zf)
        assert result == {"img1": "BinData/img1.png", "img2": "BinData/img2.jpg"}
        zf.close()

    def test_parse_missing_hpf(self):
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({})), "r")
        assert parse_bin_item_map(zf) == {}
        zf.close()

    def test_parse_malformed_xml(self):
        files = {HPF_PATH: b"not xml"}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        assert parse_bin_item_map(zf) == {}
        zf.close()


class TestFindSections:
    def test_sort_order(self):
        files = {
            "Contents/section2.xml": b"",
            "Contents/section0.xml": b"",
            "Contents/section10.xml": b"",
            "Contents/section1.xml": b"",
        }
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        result = find_section_paths(zf)
        assert result == [
            "Contents/section0.xml",
            "Contents/section1.xml",
            "Contents/section2.xml",
            "Contents/section10.xml",
        ]
        zf.close()

    def test_no_sections(self):
        zf = zipfile.ZipFile(io.BytesIO(_make_zip({"other.xml": b""})), "r")
        assert find_section_paths(zf) == []
        zf.close()

    def test_case_insensitive(self):
        files = {"Contents/Section0.xml": b""}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        result = find_section_paths(zf)
        assert len(result) == 1
        zf.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Metadata Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetadataExtractor:
    def test_extract_from_header(self):
        header = _make_header_xml({"title": "My Doc", "author": "Alice"})
        files = {
            HEADER_PATH: header,
            "Contents/section0.xml": b"<hs:sec/>",
        }
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")

        ext = HwpxMetadataExtractor()
        meta = ext.extract(zf)
        assert meta.title == "My Doc"
        assert meta.author == "Alice"
        assert meta.page_count == 1
        zf.close()

    def test_extract_version(self):
        version = _make_version_xml("5.1", odt="5.1.0.3")
        files = {VERSION_PATH: version}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")

        ext = HwpxMetadataExtractor()
        meta = ext.extract(zf)
        assert meta.custom.get("version") == "5.1"
        assert meta.custom.get("version_odt") == "5.1.0.3"
        zf.close()

    def test_extract_manifest_media_type(self):
        manifest = (
            '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
            '<manifest:file-entry full-path="/" media-type="application/hwp+zip"/>'
            '</manifest:manifest>'
        )
        files = {MANIFEST_PATH: manifest}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")

        ext = HwpxMetadataExtractor()
        meta = ext.extract(zf)
        assert meta.custom.get("media_type") == "application/hwp+zip"
        zf.close()

    def test_extract_none_source(self):
        ext = HwpxMetadataExtractor()
        meta = ext.extract(None)
        assert meta.is_empty()

    def test_extract_preprocessed_data(self):
        """MetadataExtractor unwraps PreprocessedData.content."""
        header = _make_header_xml({"subject": "Test Subject"})
        files = {HEADER_PATH: header}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")

        pp = PreprocessedData(content=zf)
        ext = HwpxMetadataExtractor()
        meta = ext.extract(pp)
        assert meta.subject == "Test Subject"
        zf.close()

    def test_get_format_name(self):
        assert HwpxMetadataExtractor().get_format_name() == "hwpx"

    def test_empty_header(self):
        """Header.xml with empty docInfo."""
        header = f'<hh:head xmlns:hh="{_HH}"><hh:docInfo/></hh:head>'
        files = {HEADER_PATH: header}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        ext = HwpxMetadataExtractor()
        meta = ext.extract(zf)
        assert meta.title is None
        zf.close()

    def test_multiple_sections_page_count(self):
        files = {
            "Contents/section0.xml": b"<x/>",
            "Contents/section1.xml": b"<x/>",
            "Contents/section2.xml": b"<x/>",
        }
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        ext = HwpxMetadataExtractor()
        meta = ext.extract(zf)
        assert meta.page_count == 3
        zf.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Content Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestContentExtractor:
    def _make_preprocessed(self, files: Dict[str, str | bytes]) -> tuple:
        """Create PreprocessedData from files dict. Returns (pp, zf)."""
        zip_data = _make_zip(files)
        zf = zipfile.ZipFile(io.BytesIO(zip_data), "r")
        sections = find_section_paths(zf)
        opf_content = files.get(HPF_PATH, b"")
        bin_item_map = {}
        if opf_content:
            bin_item_map = parse_bin_item_map(zf)

        pp = PreprocessedData(
            content=zf,
            raw_content=zip_data,
            resources={"file_data": zip_data, "bin_item_map": bin_item_map},
            properties={"section_count": len(sections), "section_paths": sections},
        )
        return pp, zf

    def test_extract_text_basic(self):
        section = _make_section_xml(["Hello HWPX"])
        files = {"Contents/section0.xml": section}
        pp, zf = self._make_preprocessed(files)

        ext = HwpxContentExtractor()
        text = ext.extract_text(pp)
        assert "Hello HWPX" in text
        zf.close()

    def test_extract_text_multi_section(self):
        files = {
            "Contents/section0.xml": _make_section_xml(["Section Zero"]),
            "Contents/section1.xml": _make_section_xml(["Section One"]),
        }
        pp, zf = self._make_preprocessed(files)

        ext = HwpxContentExtractor()
        text = ext.extract_text(pp)
        assert "Section Zero" in text
        assert "Section One" in text
        zf.close()

    def test_extract_text_with_images(self):
        opf = _make_opf_manifest({"img1": "BinData/img1.png"})
        image_data = b"\x89PNG" + b"\x00" * 50
        section = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run>'
            f'<hp:ctrl><hc:pic><hc:img binaryItemIDRef="img1"/></hc:pic></hp:ctrl>'
            f'</hp:run></hp:p>'
            f'</hs:sec>'
        )
        files = {
            HPF_PATH: opf,
            "Contents/section0.xml": section,
            "BinData/img1.png": image_data,
        }
        pp, zf = self._make_preprocessed(files)

        mock_img = MagicMock()
        mock_img.save_image.return_value = "[Image:hwpx_img.png]"

        ext = HwpxContentExtractor(image_service=mock_img)
        text = ext.extract_text(pp)
        assert "[Image:hwpx_img.png]" in text
        zf.close()

    def test_remaining_images(self):
        """BinData images not referenced inline should be appended."""
        opf = _make_opf_manifest({})
        image_data = b"\x89PNG" + b"\x00" * 50
        section = _make_section_xml(["Hello"])
        files = {
            HPF_PATH: opf,
            "Contents/section0.xml": section,
            "BinData/extra.png": image_data,
        }
        pp, zf = self._make_preprocessed(files)

        mock_img = MagicMock()
        mock_img.save_image.return_value = "[Image:extra.png]"

        ext = HwpxContentExtractor(image_service=mock_img)
        text = ext.extract_text(pp)
        assert "[Image:extra.png]" in text
        zf.close()

    def test_no_zipfile(self):
        pp = PreprocessedData(content=None, resources={}, properties={})
        ext = HwpxContentExtractor()
        assert ext.extract_text(pp) == ""

    def test_extract_tables_empty(self):
        pp = PreprocessedData(content=None, resources={}, properties={})
        ext = HwpxContentExtractor()
        assert ext.extract_tables(pp) == []

    def test_extract_charts_empty(self):
        pp = PreprocessedData(content=None, resources={}, properties={})
        ext = HwpxContentExtractor()
        assert ext.extract_charts(pp) == []

    def test_get_format_name(self):
        assert HwpxContentExtractor().get_format_name() == "hwpx"

    def test_extract_all_integration(self):
        """Test extract_all() orchestration."""
        section = _make_section_xml(["Full pipeline"])
        files = {"Contents/section0.xml": section}
        pp, zf = self._make_preprocessed(files)

        ext = HwpxContentExtractor()
        result = ext.extract_all(pp)
        assert isinstance(result, ExtractionResult)
        assert "Full pipeline" in result.text
        zf.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Handler
# ═══════════════════════════════════════════════════════════════════════════════

class TestHandler:
    def _make_handler(self, **kwargs):
        config = ProcessingConfig()
        return HWPXHandler(config, **kwargs)

    def test_supported_extensions(self):
        h = self._make_handler()
        assert h.supported_extensions == frozenset({"hwpx"})

    def test_handler_name(self):
        h = self._make_handler()
        assert h.handler_name == "HWPX Handler"

    def test_pipeline_components_created(self):
        h = self._make_handler()
        assert isinstance(h._converter, HwpxConverter)
        assert isinstance(h._preprocessor, HwpxPreprocessor)
        assert isinstance(h._metadata_extractor, HwpxMetadataExtractor)
        assert isinstance(h._content_extractor, HwpxContentExtractor)

    def test_process_basic(self):
        header = _make_header_xml({"title": "Test HWPX"})
        section = _make_section_xml(["Hello from HWPX handler"])
        opf = _make_opf_manifest({})
        files = {
            HEADER_PATH: header,
            HPF_PATH: opf,
            "Contents/section0.xml": section,
        }
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = self._make_handler()
        result = h.process(fc)
        assert isinstance(result, ExtractionResult)
        assert "Hello from HWPX handler" in result.text

    def test_process_invalid_zip(self):
        fc = _make_file_context(b"not a zip")
        h = self._make_handler()
        with pytest.raises(Exception):
            h.process(fc)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Full Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_text_with_table(self):
        table_xml = (
            f'<hp:tbl rowCnt="1" colCnt="2">'
            f'<hp:tr>'
            f'<hp:tc><hp:cellAddr colAddr="0" rowAddr="0"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
            f'<hp:subList><hp:p><hp:run><hp:t>Col1</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
            f'<hp:tc><hp:cellAddr colAddr="1" rowAddr="0"/><hp:cellSpan colSpan="1" rowSpan="1"/>'
            f'<hp:subList><hp:p><hp:run><hp:t>Col2</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
            f'</hp:tr>'
            f'</hp:tbl>'
        )
        section = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run><hp:t>Before table</hp:t></hp:run>{table_xml}</hp:p>'
            f'<hp:p><hp:run><hp:t>After table</hp:t></hp:run></hp:p>'
            f'</hs:sec>'
        )
        files = {"Contents/section0.xml": section}
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert "Before table" in result.text
        assert "After table" in result.text
        assert "<table>" in result.text

    def test_metadata_in_result(self):
        header = _make_header_xml({"title": "Pipeline Title", "author": "Bob"})
        section = _make_section_xml(["Body text"])
        files = {
            HEADER_PATH: header,
            "Contents/section0.xml": section,
        }
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert result.metadata is not None
        assert result.metadata.title == "Pipeline Title"
        assert result.metadata.author == "Bob"

    def test_image_in_pipeline(self):
        opf = _make_opf_manifest({"pic1": "BinData/pic1.png"})
        image_data = b"\x89PNG" + b"\x00" * 50

        section = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run>'
            f'<hp:ctrl><hc:pic><hc:img binaryItemIDRef="pic1"/></hc:pic></hp:ctrl>'
            f'</hp:run></hp:p>'
            f'</hs:sec>'
        )
        files = {
            HPF_PATH: opf,
            "Contents/section0.xml": section,
            "BinData/pic1.png": image_data,
        }
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        mock_img = MagicMock()
        mock_img.save_image.return_value = "[Image:pipeline.png]"

        h = HWPXHandler(ProcessingConfig(), image_service=mock_img)
        result = h.process(fc)
        assert "[Image:pipeline.png]" in result.text

    def test_multi_section_pipeline(self):
        files = {
            "Contents/section0.xml": _make_section_xml(["Page 1 content"]),
            "Contents/section1.xml": _make_section_xml(["Page 2 content"]),
            "Contents/section2.xml": _make_section_xml(["Page 3 content"]),
        }
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert "Page 1 content" in result.text
        assert "Page 2 content" in result.text
        assert "Page 3 content" in result.text

    def test_chart_in_pipeline(self):
        ns_c = OOXML_CHART_NS["c"]
        ns_a = OOXML_CHART_NS["a"]
        chart_xml = (
            f'<c:chartSpace xmlns:c="{ns_c}" xmlns:a="{ns_a}">'
            f'<c:chart>'
            f'<c:title><c:tx><c:rich><a:p><a:r><a:t>Q Report</a:t></a:r></a:p></c:rich></c:tx></c:title>'
            f'<c:plotArea>'
            f'<c:pieChart>'
            f'<c:ser>'
            f'<c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Share</c:v></c:pt></c:strCache></c:strRef></c:tx>'
            f'<c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>50</c:v></c:pt></c:numCache></c:numRef></c:val>'
            f'</c:ser>'
            f'</c:pieChart>'
            f'</c:plotArea>'
            f'</c:chart>'
            f'</c:chartSpace>'
        )
        section = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p>'
            f'<hp:switch><hp:case>'
            f'<hp:chart chartIDRef="Chart/chart1.xml"/>'
            f'</hp:case></hp:switch>'
            f'</hp:p>'
            f'</hs:sec>'
        )
        files = {
            "Contents/section0.xml": section,
            "Chart/chart1.xml": chart_xml,
        }
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert "Q Report" in result.text or "Chart" in result.text


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_zip(self):
        """ZIP with no sections produces empty text."""
        files = {"mimetype": b"application/hwp+zip"}
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert isinstance(result, ExtractionResult)

    def test_corrupt_section_xml(self):
        """Corrupt XML in a section should not crash the handler."""
        files = {"Contents/section0.xml": b"<invalid xml!!!"}
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert isinstance(result, ExtractionResult)

    def test_section_with_no_text(self):
        """Section with structure but no text content."""
        xml = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}">'
            f'<hp:p><hp:run></hp:run></hp:p>'
            f'</hs:sec>'
        )
        files = {"Contents/section0.xml": xml}
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert isinstance(result, ExtractionResult)

    def test_image_without_service(self):
        """Image elements with no image_service should be silently skipped."""
        opf = _make_opf_manifest({"img1": "BinData/img1.png"})
        section = (
            f'<hs:sec xmlns:hs="{_HS}" xmlns:hp="{_HP}" xmlns:hc="{_HC}">'
            f'<hp:p><hp:run>'
            f'<hp:ctrl><hc:pic><hc:img binaryItemIDRef="img1"/></hc:pic></hp:ctrl>'
            f'</hp:run></hp:p>'
            f'</hs:sec>'
        )
        files = {
            HPF_PATH: opf,
            "Contents/section0.xml": section,
            "BinData/img1.png": b"\x89PNG" + b"\x00" * 50,
        }
        zip_data = _make_zip(files)
        fc = _make_file_context(zip_data)

        # No image_service
        h = HWPXHandler(ProcessingConfig())
        result = h.process(fc)
        assert "[Image:" not in result.text

    def test_bin_item_map_item_without_id(self):
        """OPF item without id should be skipped."""
        opf = (
            f'<opf:package xmlns:opf="{OPF_NAMESPACES["opf"]}">'
            f'<opf:manifest>'
            f'<opf:item href="BinData/noId.png"/>'
            f'<opf:item id="good" href="BinData/good.png"/>'
            f'</opf:manifest>'
            f'</opf:package>'
        )
        files = {HPF_PATH: opf}
        zf = zipfile.ZipFile(io.BytesIO(_make_zip(files)), "r")
        result = parse_bin_item_map(zf)
        assert "good" in result
        assert len(result) == 1
        zf.close()

    def test_remaining_non_image_skipped(self):
        """Non-image files in BinData/ should NOT be processed as images."""
        section = _make_section_xml(["Text"])
        files = {
            "Contents/section0.xml": section,
            "BinData/data.txt": b"not an image",
            "BinData/real.png": b"\x89PNG" + b"\x00" * 50,
        }
        pp_data = _make_zip(files)
        zf = zipfile.ZipFile(io.BytesIO(pp_data), "r")

        pp = PreprocessedData(
            content=zf,
            resources={"bin_item_map": {}},
            properties={"section_paths": find_section_paths(zf)},
        )

        mock_img = MagicMock()
        mock_img.save_image.return_value = "[Image:x]"

        ext = HwpxContentExtractor(image_service=mock_img)
        text = ext.extract_text(pp)
        # Only .png should be processed, not .txt
        assert mock_img.save_image.call_count == 1
        zf.close()

    def test_unsupported_image_ext(self):
        """Files with unsupported extensions in BinData/ should be skipped."""
        section = _make_section_xml(["Text"])
        files = {
            "Contents/section0.xml": section,
            "BinData/file.wmf": b"\x00" * 100,
        }
        pp_data = _make_zip(files)
        zf = zipfile.ZipFile(io.BytesIO(pp_data), "r")

        pp = PreprocessedData(
            content=zf,
            resources={"bin_item_map": {}},
            properties={"section_paths": find_section_paths(zf)},
        )

        mock_img = MagicMock()
        ext = HwpxContentExtractor(image_service=mock_img)
        ext.extract_text(pp)
        mock_img.save_image.assert_not_called()
        zf.close()
