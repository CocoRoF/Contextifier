# tests/unit/services/test_chart_service.py
"""P3-2: Unit tests for ChartService — formatting, OOXML type mapping, pattern matching."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from contextifier.config import ProcessingConfig
from contextifier.services.chart_service import ChartService
from contextifier.types import ChartData, ChartSeries


@pytest.fixture()
def chart_service() -> ChartService:
    config = ProcessingConfig()
    tag_svc = MagicMock()
    tag_svc.create_chart_open_tag.return_value = "[chart]"
    tag_svc.create_chart_close_tag.return_value = "[/chart]"
    return ChartService(config, tag_service=tag_svc)


@pytest.fixture()
def chart_service_text() -> ChartService:
    """ChartService with HTML table disabled."""
    config = ProcessingConfig().with_charts(use_html_table=False)
    return ChartService(config)


# ── format_chart ─────────────────────────────────────────────────────────

class TestFormatChart:
    def test_basic_bar_chart(self, chart_service: ChartService) -> None:
        data = ChartData(
            chart_type="barChart",
            title="Sales",
            categories=["Q1", "Q2"],
            series=[ChartSeries(name="Revenue", values=[100, 200])],
        )
        result = chart_service.format_chart(data)
        assert "[chart]" in result
        assert "[/chart]" in result
        assert "Chart Type: Bar Chart" in result
        assert "Title: Sales" in result
        assert "<table>" in result

    def test_empty_chart_returns_empty(self, chart_service: ChartService) -> None:
        data = ChartData(chart_type=None, title=None, categories=[], series=[])
        result = chart_service.format_chart(data)
        assert result == ""

    def test_chart_with_raw_content(self, chart_service: ChartService) -> None:
        data = ChartData(raw_content="Custom chart text")
        result = chart_service.format_chart(data)
        assert "Custom chart text" in result

    def test_chart_no_type(self, chart_service: ChartService) -> None:
        """Chart type omitted when None."""
        data = ChartData(
            title="Untitled",
            categories=["A"],
            series=[ChartSeries(name="S1", values=[1])],
        )
        result = chart_service.format_chart(data)
        assert "Chart Type:" not in result
        assert "Title: Untitled" in result

    def test_chart_no_title(self, chart_service: ChartService) -> None:
        data = ChartData(
            chart_type="lineChart",
            categories=["A"],
            series=[ChartSeries(name="S1", values=[1])],
        )
        result = chart_service.format_chart(data)
        assert "Title:" not in result
        assert "Chart Type: Line Chart" in result

    def test_text_table_format(self, chart_service_text: ChartService) -> None:
        data = ChartData(
            chart_type="pieChart",
            categories=["A", "B"],
            series=[ChartSeries(name="Values", values=[10, 20])],
        )
        result = chart_service_text.format_chart(data)
        assert "<table>" not in result
        assert "A: 10" in result
        assert "B: 20" in result

    def test_html_table_multiple_series(self, chart_service: ChartService) -> None:
        data = ChartData(
            categories=["X", "Y"],
            series=[
                ChartSeries(name="S1", values=[1, 2]),
                ChartSeries(name="S2", values=[3, 4]),
            ],
        )
        result = chart_service.format_chart(data)
        assert "<th>S1</th>" in result
        assert "<th>S2</th>" in result
        assert "<td>X</td>" in result


# ── format_chart_fallback ────────────────────────────────────────────────

class TestFormatChartFallback:
    def test_fallback_with_type_and_title(self, chart_service: ChartService) -> None:
        result = chart_service.format_chart_fallback(
            chart_type="barChart", title="My Chart",
        )
        assert "Bar Chart" in result
        assert "My Chart" in result

    def test_fallback_with_message(self, chart_service: ChartService) -> None:
        result = chart_service.format_chart_fallback(message="Chart skipped")
        assert "Chart skipped" in result

    def test_fallback_default_message(self, chart_service: ChartService) -> None:
        result = chart_service.format_chart_fallback()
        assert "could not be extracted" in result


# ── OOXML type mapping ───────────────────────────────────────────────────

class TestChartTypeName:
    @pytest.mark.parametrize("ooxml_type,expected", [
        ("barChart", "Bar Chart"),
        ("lineChart", "Line Chart"),
        ("pieChart", "Pie Chart"),
        ("scatterChart", "Scatter Chart"),
        ("doughnutChart", "Doughnut Chart"),
    ])
    def test_known_types(self, chart_service: ChartService, ooxml_type: str, expected: str) -> None:
        assert chart_service.get_chart_type_name(ooxml_type) == expected

    def test_unknown_type_passthrough(self, chart_service: ChartService) -> None:
        assert chart_service.get_chart_type_name("customChart") == "customChart"


# ── Pattern matching ─────────────────────────────────────────────────────

class TestChartPatterns:
    def test_has_chart_blocks(self, chart_service: ChartService) -> None:
        text = "before [chart]\nchart data\n[/chart] after"
        assert chart_service.has_chart_blocks(text) is True

    def test_no_chart_blocks(self, chart_service: ChartService) -> None:
        assert chart_service.has_chart_blocks("plain text") is False

    def test_find_chart_blocks(self, chart_service: ChartService) -> None:
        text = "pre [chart]\nblock1\n[/chart] mid [chart]\nblock2\n[/chart] end"
        blocks = chart_service.find_chart_blocks(text)
        assert len(blocks) == 2
        assert "block1" in blocks[0][2]
        assert "block2" in blocks[1][2]


# ── Tag service integration ──────────────────────────────────────────────

class TestTagFallback:
    def test_uses_config_tags_without_tag_service(self) -> None:
        config = ProcessingConfig()
        svc = ChartService(config, tag_service=None)
        data = ChartData(chart_type="barChart", categories=["A"], series=[ChartSeries(name="S", values=[1])])
        result = svc.format_chart(data)
        assert result.startswith("[chart]")
        assert result.endswith("[/chart]")
