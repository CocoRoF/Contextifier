# tests/unit/handlers/test_hwpx_charts.py
"""
P2-2: HWPX chart extraction verification tests.

Verifies that HWPX chart XML is correctly parsed inline during
section parsing, and that the content extractor's architectural
choice (charts inline in extract_text, extract_charts returns [])
is working as designed.
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from unittest import mock

import pytest

from contextifier.handlers.hwpx._section import (
    _parse_ooxml_chart,
    _process_chart_ref,
    _format_chart_simple,
    parse_hwpx_section,
)
from contextifier.handlers.hwpx._constants import OOXML_CHART_NS


# ═══════════════════════════════════════════════════════════════════════════
# OOXML chart XML parsing
# ═══════════════════════════════════════════════════════════════════════════


def _make_chart_xml(
    chart_type: str = "barChart",
    title: str = "Revenue",
    categories: list[str] | None = None,
    series: list[dict] | None = None,
) -> bytes:
    """Build a minimal OOXML chart XML for testing."""
    cats = categories or ["Q1", "Q2", "Q3"]
    sers = series or [{"name": "Sales", "values": [100.0, 200.0, 300.0]}]

    cat_pts = "".join(
        f'<c:pt idx="{i}"><c:v>{c}</c:v></c:pt>'
        for i, c in enumerate(cats)
    )
    ser_parts = []
    for idx, s in enumerate(sers):
        val_pts = "".join(
            f'<c:pt idx="{j}"><c:v>{v}</c:v></c:pt>'
            for j, v in enumerate(s["values"])
        )
        ser_parts.append(
            f'<c:ser>'
            f'<c:idx val="{idx}"/>'
            f'<c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>{s["name"]}</c:v></c:pt></c:strCache></c:strRef></c:tx>'
            f'<c:cat><c:strCache>{cat_pts}</c:strCache></c:cat>'
            f'<c:val><c:numCache>{val_pts}</c:numCache></c:val>'
            f'</c:ser>'
        )

    return (
        f'<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"'
        f'              xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f'<c:chart>'
        f'<c:title><c:tx><c:rich><a:p><a:r><a:t>{title}</a:t></a:r></a:p></c:rich></c:tx></c:title>'
        f'<c:plotArea><c:{chart_type}>{"".join(ser_parts)}</c:{chart_type}></c:plotArea>'
        f'</c:chart>'
        f'</c:chartSpace>'
    ).encode("utf-8")


class TestParseOoxmlChart:
    """Tests for _parse_ooxml_chart()."""

    def test_bar_chart(self):
        xml_bytes = _make_chart_xml("barChart", "Revenue", ["Q1", "Q2"], [
            {"name": "Sales", "values": [100.0, 200.0]},
        ])
        result = _parse_ooxml_chart(xml_bytes)
        assert result is not None
        assert result["type"] == "Bar Chart"
        assert result["title"] == "Revenue"
        assert result["categories"] == ["Q1", "Q2"]
        assert len(result["series"]) == 1
        assert result["series"][0]["name"] == "Sales"
        assert result["series"][0]["values"] == [100.0, 200.0]

    def test_line_chart(self):
        xml_bytes = _make_chart_xml("lineChart", "Trend")
        result = _parse_ooxml_chart(xml_bytes)
        assert result is not None
        assert result["type"] == "Line Chart"
        assert result["title"] == "Trend"

    def test_pie_chart(self):
        xml_bytes = _make_chart_xml("pieChart", "Share")
        result = _parse_ooxml_chart(xml_bytes)
        assert result is not None
        assert result["type"] == "Pie Chart"

    def test_invalid_xml_returns_none(self):
        result = _parse_ooxml_chart(b"not valid xml")
        assert result is None

    def test_empty_series_returns_none(self):
        """Chart with no series data returns None."""
        xml_bytes = (
            b'<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">'
            b'<c:chart><c:plotArea><c:barChart></c:barChart></c:plotArea></c:chart>'
            b'</c:chartSpace>'
        )
        result = _parse_ooxml_chart(xml_bytes)
        assert result is None

    def test_multiple_series(self):
        xml_bytes = _make_chart_xml("barChart", "Multi", ["A", "B"], [
            {"name": "Series1", "values": [10.0, 20.0]},
            {"name": "Series2", "values": [30.0, 40.0]},
        ])
        result = _parse_ooxml_chart(xml_bytes)
        assert result is not None
        assert len(result["series"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Chart formatting
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatChartSimple:
    """Tests for _format_chart_simple()."""

    def test_basic_format(self):
        data = {
            "type": "Bar Chart",
            "title": "Sales",
            "series": [{"name": "Revenue", "values": [100, 200, 300]}],
        }
        result = _format_chart_simple(data)
        assert "[Chart: Sales]" in result
        assert "Revenue: 100, 200, 300" in result

    def test_no_title_uses_type(self):
        data = {
            "type": "Pie Chart",
            "title": None,
            "series": [{"name": "Data", "values": [50]}],
        }
        result = _format_chart_simple(data)
        assert "[Chart: Pie Chart]" in result


# ═══════════════════════════════════════════════════════════════════════════
# Inline chart via section parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestInlineChartProcessing:
    """Verify that charts are processed inline during section parsing."""

    def test_chart_processed_inline_in_section(self):
        """Chart referenced via <hp:chart chartIDRef="..."> is extracted."""
        chart_xml = _make_chart_xml("barChart", "InlineTest", ["X"], [
            {"name": "Y", "values": [42.0]},
        ])

        # Build a minimal HWPX section XML with a chart reference
        section_xml = (
            '<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"'
            '        xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">'
            '<hp:p>'
            '<hp:run><hp:t>Before chart</hp:t></hp:run>'
            '<hp:chart chartIDRef="chart1.xml"/>'
            '<hp:run><hp:t>After chart</hp:t></hp:run>'
            '</hp:p>'
            '</hs:sec>'
        ).encode("utf-8")

        # Create a mock ZIP with the chart file
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf_out:
            zf_out.writestr("Chart/chart1.xml", chart_xml)

        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zf_in:
            result = parse_hwpx_section(
                section_xml, zf_in, {},
                chart_service=None,
            )

        assert "Before chart" in result
        assert "After chart" in result
        assert "[Chart: InlineTest]" in result
        assert "Y: 42.0" in result

    def test_extract_charts_returns_empty(self):
        """HwpxContentExtractor.extract_charts() returns [] by design."""
        from contextifier.handlers.hwpx.content_extractor import HwpxContentExtractor
        from contextifier.types import PreprocessedData

        extractor = HwpxContentExtractor()
        preprocessed = PreprocessedData(content=None, raw_content=b"")
        assert extractor.extract_charts(preprocessed) == []
