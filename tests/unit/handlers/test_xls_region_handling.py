# tests/unit/handlers/test_xls_region_handling.py
"""
XLS (BIFF) region detection and rendering.

Exercised through lightweight stand-ins for the xlrd objects: writing .xls
fixtures needs a writer library that is not a dependency, and the behaviour
under test is the module's own arithmetic, not xlrd's parsing.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from contextifier.handlers.xls._layout import LayoutRange, layout_detect_range
from contextifier.handlers.xls._table import convert_sheet_to_text


class FakeSheet:
    """Minimal stand-in for an ``xlrd`` Sheet."""

    def __init__(
        self,
        grid: List[List[object]],
        merged: List[Tuple[int, int, int, int]] | None = None,
    ) -> None:
        self._grid = grid
        self.nrows = len(grid)
        self.ncols = max((len(r) for r in grid), default=0)
        self.merged_cells = merged or []

    def cell_value(self, r: int, c: int) -> object:
        try:
            return self._grid[r][c]
        except IndexError:
            return ""

    def cell_type(self, r: int, c: int) -> int:
        return 1  # xlrd XL_CELL_TEXT


class FakeBook:
    datemode = 0


class TestSheetExtent:
    def test_rows_past_the_old_scan_budget_are_read(self) -> None:
        grid = [[""] for _ in range(1500)]
        grid[0] = ["top"]
        grid[1499] = ["ROW-1500-MARKER"]
        layout = layout_detect_range(FakeSheet(grid))

        assert layout is not None
        assert layout.max_row == 1500


class TestMinimumTableShape:
    def test_lone_cell_is_plain_text(self) -> None:
        sheet = FakeSheet([["only value"]])
        out = convert_sheet_to_text(sheet, FakeBook(), LayoutRange(1, 1, 1, 1))

        assert out.strip() == "only value"
        assert "|" not in out
        assert "<table>" not in out

    def test_two_by_two_is_a_table(self) -> None:
        sheet = FakeSheet([["a", "b"], ["c", "d"]])
        out = convert_sheet_to_text(sheet, FakeBook(), LayoutRange(1, 2, 1, 2))

        assert "| a | b |" in out


class TestMergeClipping:
    def test_anchor_above_the_region_still_shows_its_label(self) -> None:
        grid = [["SECTION", "first"], ["", "second"], ["", "third"]]
        # A1:A3 merged — anchor on row 0, region starts at row 2.
        sheet = FakeSheet(grid, merged=[(0, 3, 0, 1)])
        out = convert_sheet_to_text(sheet, FakeBook(), LayoutRange(2, 3, 1, 2))

        assert "SECTION" in out, "the covering merge lost its label:\n" + out

    def test_span_is_clipped_to_the_region(self) -> None:
        import re

        grid = [["SECTION", "first"], ["", "second"], ["", "third"]]
        sheet = FakeSheet(grid, merged=[(0, 3, 0, 1)])
        out = convert_sheet_to_text(sheet, FakeBook(), LayoutRange(2, 3, 1, 2))

        row_count = len(re.findall(r"<tr[^>]*>", out))
        for value in re.findall(r"rowspan=['\"]?(\d+)", out):
            assert int(value) <= row_count, out
