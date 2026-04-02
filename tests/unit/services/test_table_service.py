# tests/unit/services/test_table_service.py
"""Unit tests for TableService."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig, TableConfig
from contextifier.services.table_service import TableService
from contextifier.types import OutputFormat, TableData, TableCell


def _simple_table() -> TableData:
    """2×2 table with basic text content."""
    return TableData(
        rows=[
            [TableCell(content="A", is_header=True), TableCell(content="B", is_header=True)],
            [TableCell(content="1"), TableCell(content="2")],
        ],
        num_cols=2,
    )


def _table_with_special_chars() -> TableData:
    """Table with HTML-sensitive characters."""
    return TableData(
        rows=[
            [TableCell(content="<script>alert('xss')</script>"), TableCell(content="a & b")],
        ],
        num_cols=2,
    )


class TestFormatAsHtml:
    def test_basic(self) -> None:
        svc = TableService(ProcessingConfig())
        result = svc.format_as_html(_simple_table())
        assert "<table>" in result
        assert "<th>" in result
        assert "A" in result

    def test_empty_table(self) -> None:
        svc = TableService(ProcessingConfig())
        assert svc.format_as_html(TableData(rows=[], num_cols=0)) == ""

    def test_escapes_html_special_chars(self) -> None:
        """P0-1: format_as_html() must escape HTML-sensitive content."""
        svc = TableService(ProcessingConfig())
        result = svc.format_as_html(_table_with_special_chars())
        assert "&lt;script&gt;" in result
        assert "&amp;" in result
        assert "<script>alert" not in result


class TestFormatAsMarkdown:
    def test_basic(self) -> None:
        svc = TableService(ProcessingConfig())
        result = svc.format_as_markdown(_simple_table())
        assert "| A | B |" in result
        assert "| --- |" in result
        assert "| 1 | 2 |" in result


class TestFormatAsText:
    def test_basic(self) -> None:
        svc = TableService(ProcessingConfig())
        result = svc.format_as_text(_simple_table())
        assert "A\tB" in result
        assert "1\t2" in result


class TestFormatAsHtmlSimple:
    """Tests for the static fallback method."""

    def test_escapes_html(self) -> None:
        result = TableService.format_as_html_simple(_table_with_special_chars())
        assert "&lt;script&gt;" in result
        assert "&amp;" in result
        assert "<script>" not in result

    def test_empty(self) -> None:
        assert TableService.format_as_html_simple(TableData(rows=[], num_cols=0)) == ""

    def test_newlines_become_br(self) -> None:
        table = TableData(
            rows=[[TableCell(content="line1\nline2")]],
            num_cols=1,
        )
        result = TableService.format_as_html_simple(table)
        assert "<br>" in result

    def test_rowspan_colspan(self) -> None:
        table = TableData(
            rows=[[TableCell(content="merged", row_span=2, col_span=3)]],
            num_cols=3,
        )
        result = TableService.format_as_html_simple(table)
        assert "rowspan='2'" in result
        assert "colspan='3'" in result
