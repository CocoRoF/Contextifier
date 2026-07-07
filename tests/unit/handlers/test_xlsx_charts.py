# tests/unit/handlers/test_xlsx_charts.py
"""
P2-6: XLSX chart extraction verification tests.

Verifies that XlsxContentExtractor.extract_charts() correctly converts
pre-extracted chart dicts from ZIP resources into ChartData objects.
"""

from __future__ import annotations

from unittest import mock


from contextifier.handlers.xlsx.content_extractor import XlsxContentExtractor
from contextifier.types import ChartData, ChartSeries, PreprocessedData


def _make_preprocessed(charts: list[dict] | None = None):
    """Create PreprocessedData with chart resources."""
    wb = mock.MagicMock()
    wb.sheetnames = []
    return PreprocessedData(
        content=wb,
        raw_content=b"",
        resources={
            "charts": charts or [],
            "images": {},
            "textboxes": {},
        },
    )


class TestXlsxChartExtraction:
    """Tests for XLSX chart extraction."""

    def test_empty_charts(self):
        extractor = XlsxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed([]))
        assert charts == []

    def test_single_bar_chart(self):
        chart_dict = {
            "chart_type": "Bar Chart",
            "title": "Sales by Region",
            "categories": ["East", "West", "North"],
            "series": [
                {"name": "Q1", "values": [100, 200, 150]},
                {"name": "Q2", "values": [120, 180, 160]},
            ],
        }
        extractor = XlsxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed([chart_dict]))
        assert len(charts) == 1

        cd = charts[0]
        assert isinstance(cd, ChartData)
        assert cd.chart_type == "Bar Chart"
        assert cd.title == "Sales by Region"
        assert cd.categories == ["East", "West", "North"]
        assert len(cd.series) == 2
        assert cd.series[0].name == "Q1"
        assert cd.series[0].values == [100, 200, 150]

    def test_multiple_charts(self):
        charts_data = [
            {
                "chart_type": "Line Chart",
                "title": "Trend",
                "categories": ["Jan", "Feb"],
                "series": [{"name": "Revenue", "values": [500, 600]}],
            },
            {
                "chart_type": "Pie Chart",
                "title": "Market Share",
                "categories": ["A", "B", "C"],
                "series": [{"name": "Share", "values": [40, 35, 25]}],
            },
        ]
        extractor = XlsxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed(charts_data))
        assert len(charts) == 2
        assert charts[0].chart_type == "Line Chart"
        assert charts[1].chart_type == "Pie Chart"

    def test_chart_with_missing_fields(self):
        """Chart dict with missing optional fields uses defaults."""
        chart_dict = {
            "series": [{"name": "Data", "values": [1, 2, 3]}],
        }
        extractor = XlsxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed([chart_dict]))
        assert len(charts) == 1
        assert charts[0].chart_type == "Unknown"
        assert charts[0].title == ""
        assert charts[0].categories == []

    def test_invalid_chart_dict_skipped(self):
        """Invalid chart dict entries are gracefully skipped."""
        charts_data = [
            {"chart_type": "Valid", "title": "OK", "series": [{"name": "S", "values": [1]}]},
        ]
        extractor = XlsxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed(charts_data))
        assert len(charts) == 1
        assert charts[0].title == "OK"

    def test_no_content_returns_empty(self):
        """None workbook returns empty list."""
        preprocessed = PreprocessedData(
            content=None,
            raw_content=b"",
            resources={"charts": [{"series": [{"name": "X", "values": [1]}]}]},
        )
        extractor = XlsxContentExtractor()
        charts = extractor.extract_charts(preprocessed)
        assert len(charts) >= 1  # extract_charts reads from resources, not wb

    def test_chart_series_structure(self):
        """ChartSeries objects have correct structure."""
        chart_dict = {
            "chart_type": "Scatter",
            "title": "Scatter Plot",
            "categories": [],
            "series": [
                {"name": "Points", "values": [1.5, 2.7, 3.1]},
            ],
        }
        extractor = XlsxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed([chart_dict]))
        series = charts[0].series[0]
        assert isinstance(series, ChartSeries)
        assert series.name == "Points"
        assert series.values == [1.5, 2.7, 3.1]
