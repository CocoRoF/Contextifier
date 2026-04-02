# tests/unit/handlers/test_ppt_tables_charts.py
"""
Tests for PPT table detection and chart OLE scanning.

Covers:
- Tab-separated text → table detection
- No-tab text → empty tables
- OLE chart storage detection
- Edge cases (empty streams, missing OLE)
"""

from __future__ import annotations

import struct
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from contextifier.handlers.ppt.content_extractor import (
    PptContentExtractor,
    _detect_ole_charts,
    _detect_tabular_text,
    _rows_to_table_data,
)
from contextifier.types import PreprocessedData, TableCell


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — build synthetic PPT text records
# ═════════════════════════════════════════════════════════════════════════════

_TEXT_CHARS_ATOM = 0x0FA8  # Unicode text record


def _make_text_record(text: str, offset: int = 0) -> bytes:
    """Build a single TextCharsAtom record."""
    text_bytes = text.encode("utf-16-le")
    rec_ver_inst = 0x0000  # version=0, instance=0
    header = struct.pack("<HHI", rec_ver_inst, _TEXT_CHARS_ATOM, len(text_bytes))
    return header + text_bytes


def _make_pp_stream(texts: list[str]) -> bytes:
    """Build a minimal PowerPoint Document stream from text list."""
    parts: list[bytes] = []
    for t in texts:
        parts.append(_make_text_record(t))
    return b"".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# Table detection tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTabularTextDetection:
    """Tab-separated text pattern detection."""

    def test_tab_separated_rows(self):
        records: List[Tuple[int, str]] = [
            (0, "Name\tAge\tCity"),
            (100, "Alice\t25\tSeoul"),
            (200, "Bob\t30\tBusan"),
        ]
        tables = _detect_tabular_text(records)
        assert len(tables) == 1
        assert tables[0].num_rows == 3
        assert tables[0].num_cols == 3
        assert tables[0].rows[0][0].content == "Name"
        assert tables[0].rows[1][2].content == "Seoul"

    def test_no_tabs_no_tables(self):
        records: List[Tuple[int, str]] = [
            (0, "Just plain text"),
            (100, "No tabs here"),
        ]
        tables = _detect_tabular_text(records)
        assert tables == []

    def test_single_tab_row_ignored(self):
        """A single row with tabs is not enough for a table."""
        records: List[Tuple[int, str]] = [
            (0, "Single\tRow"),
        ]
        tables = _detect_tabular_text(records)
        assert tables == []

    def test_mixed_tab_and_plain(self):
        """Tab-separated block surrounded by plain text."""
        records: List[Tuple[int, str]] = [
            (0, "Paragraph text"),
            (100, "Col1\tCol2"),
            (200, "Val1\tVal2"),
            (300, "Another paragraph"),
        ]
        tables = _detect_tabular_text(records)
        assert len(tables) == 1
        assert tables[0].num_rows == 2

    def test_multiple_tables(self):
        records: List[Tuple[int, str]] = [
            (0, "A\tB"),
            (100, "C\tD"),
            (200, "Plain text"),
            (300, "E\tF\tG"),
            (400, "H\tI\tJ"),
        ]
        tables = _detect_tabular_text(records)
        assert len(tables) == 2


class TestRowsToTableData:
    """Convert raw rows to TableData."""

    def test_basic(self):
        rows = [["A", "B"], ["C", "D"]]
        td = _rows_to_table_data(rows)
        assert td.num_rows == 2
        assert td.num_cols == 2
        assert td.rows[0][0].content == "A"
        assert td.rows[1][1].content == "D"

    def test_ragged_rows(self):
        rows = [["A", "B", "C"], ["D", "E"]]
        td = _rows_to_table_data(rows)
        assert td.num_cols == 3  # max col count


# ═════════════════════════════════════════════════════════════════════════════
# Chart OLE detection tests
# ═════════════════════════════════════════════════════════════════════════════

class TestOleChartDetection:
    """OLE directory chart scanning."""

    def test_chart_storage_detected(self):
        ole = MagicMock()
        ole.listdir.return_value = [
            ["PowerPoint Document"],
            ["Pictures"],
            ["Object 1", "CONTENTS"],
            ["Object 1", "Microsoft Graph"],
        ]
        charts = _detect_ole_charts(ole)
        # Should detect at least one chart (from "Microsoft Graph" keyword)
        assert len(charts) >= 1
        assert any("chart" in c.chart_type.lower() or "ole" in c.chart_type.lower() for c in charts)

    def test_no_chart_storages(self):
        ole = MagicMock()
        ole.listdir.return_value = [
            ["PowerPoint Document"],
            ["Pictures"],
        ]
        charts = _detect_ole_charts(ole)
        assert charts == []

    def test_ole_error_returns_empty(self):
        ole = MagicMock()
        ole.listdir.side_effect = Exception("OLE error")
        charts = _detect_ole_charts(ole)
        assert charts == []

    def test_none_ole(self):
        """None OLE object returns empty."""
        charts = _detect_ole_charts(None)
        assert charts == []


# ═════════════════════════════════════════════════════════════════════════════
# Integration: extract_tables via PptContentExtractor
# ═════════════════════════════════════════════════════════════════════════════

class TestPptContentExtractorTables:
    """End-to-end table extraction through PptContentExtractor."""

    def test_extract_tables_from_tab_text(self):
        pp_stream = _make_pp_stream(["Col1\tCol2", "Val1\tVal2"])
        preprocessed = PreprocessedData(
            content=None,
            raw_content=None,
            resources={"pp_stream": pp_stream, "image_streams": []},
        )
        extractor = PptContentExtractor()
        tables = extractor.extract_tables(preprocessed)
        assert len(tables) == 1

    def test_extract_tables_no_stream(self):
        preprocessed = PreprocessedData(
            content=None,
            raw_content=None,
            resources={},
        )
        extractor = PptContentExtractor()
        tables = extractor.extract_tables(preprocessed)
        assert tables == []
