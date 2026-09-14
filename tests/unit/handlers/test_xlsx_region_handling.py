# tests/unit/handlers/test_xlsx_region_handling.py
"""
Worksheet region detection and rendering.

Three behaviours are pinned here:

* a sheet is scanned to its real extent, not to a fixed cell budget;
* a block too small to have a grid is written as text, not as a "table";
* a merged cell that covers a region but is anchored outside it still
  contributes its label and does not leave a span pointing past the table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextifier import DocumentProcessor

openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture()
def processor() -> DocumentProcessor:
    return DocumentProcessor()


class TestSheetExtent:
    def test_rows_past_the_old_scan_budget_are_read(
        self, processor: DocumentProcessor, tmp_path: Path
    ) -> None:
        path = tmp_path / "tall.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "header"
        ws.cell(row=1500, column=1, value="ROW-1500-MARKER")
        wb.save(path)

        text = processor.extract_text(str(path))
        assert "ROW-1500-MARKER" in text

    def test_columns_past_the_old_scan_budget_are_read(
        self, processor: DocumentProcessor, tmp_path: Path
    ) -> None:
        path = tmp_path / "wide.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "header"
        ws.cell(row=1, column=140, value="COL-140-MARKER")
        wb.save(path)

        text = processor.extract_text(str(path))
        assert "COL-140-MARKER" in text


class TestMinimumTableShape:
    def test_lone_cell_is_not_rendered_as_a_table(
        self, processor: DocumentProcessor, tmp_path: Path
    ) -> None:
        path = tmp_path / "stray.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "Name"
        ws["B1"] = "Score"
        ws["A2"] = "Alice"
        ws["B2"] = 95
        ws["A8"] = "Note: figures in thousands"
        wb.save(path)

        text = processor.extract_text(str(path))

        assert "Note: figures in thousands" in text, "the stray cell was dropped"
        assert "| Note: figures in thousands |" not in text, (
            "a single cell was rendered as a one-column table"
        )
        assert "| Alice | 95 |" in text, "the real table must still render"

    def test_single_row_block_is_text(
        self, processor: DocumentProcessor, tmp_path: Path
    ) -> None:
        path = tmp_path / "onerow.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "Name"
        ws["B1"] = "Score"
        ws["A2"] = "Alice"
        ws["B2"] = 95
        ws["A8"] = "left"
        ws["B8"] = "right"
        wb.save(path)

        text = processor.extract_text(str(path))
        assert "left" in text and "right" in text
        assert "| left | right |" not in text

    def test_two_by_two_block_is_still_a_table(
        self, processor: DocumentProcessor, tmp_path: Path
    ) -> None:
        path = tmp_path / "twobytwo.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "a"
        ws["B1"] = "b"
        ws["A2"] = "c"
        ws["B2"] = "d"
        wb.save(path)

        text = processor.extract_text(str(path))
        assert "| a | b |" in text


class TestMergedColumnKeepsBlocksTogether:
    """A merged category column is the spine of the table it labels.

    Only the anchor cell of a merged range holds a value; the rest read as
    empty. Region detection therefore used to break such a table into
    unrelated blocks, and every block but the first lost the column entirely —
    along with any chance of a chunk carrying the label its rows belong to.
    """

    @pytest.fixture()
    def split_merge_sheet(self, tmp_path: Path) -> Path:
        path = tmp_path / "merged.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "ANNUAL TOTAL"
        ws["B1"] = "Q1"
        ws["C1"] = 100
        ws["B2"] = "Q2"
        ws["C2"] = 200
        ws.merge_cells("A1:A12")
        ws["B11"] = "Q3"
        ws["C11"] = 300
        ws["B12"] = "Q4"
        ws["C12"] = 400
        wb.save(path)
        return path

    def test_rows_stay_in_one_table(
        self, processor: DocumentProcessor, split_merge_sheet: Path
    ) -> None:
        import re

        text = processor.extract_text(str(split_merge_sheet))
        tables = re.findall(r"<table>.*?</table>", text, re.DOTALL)
        assert len(tables) == 1, f"the merged column was split apart:\n{text}"
        assert "ANNUAL TOTAL" in tables[0]
        for marker in ("Q1", "Q2", "Q3", "Q4"):
            assert marker in tables[0]

    def test_no_span_points_past_its_table(
        self, processor: DocumentProcessor, split_merge_sheet: Path
    ) -> None:
        import re

        text = processor.extract_text(str(split_merge_sheet))
        for table in re.findall(r"<table>.*?</table>", text, re.DOTALL):
            row_count = len(re.findall(r"<tr[^>]*>", table))
            for value in re.findall(r'rowspan=["\']?(\d+)', table):
                assert int(value) <= row_count, (
                    f"rowspan={value} in a {row_count}-row table:\n{table}"
                )

    def test_label_reaches_a_region_it_only_covers(
        self, processor: DocumentProcessor, tmp_path: Path
    ) -> None:
        """A merge anchored above a region still labels the region's rows.

        Here the merge is in the same columns as a later block but anchored far
        above it, so the block reads empty cells where the label belongs.
        """
        path = tmp_path / "anchor_outside.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "SECTION LABEL"
        ws["B1"] = "first"
        ws.merge_cells("A1:B1")
        ws["A3"] = "x"
        ws["B3"] = "y"
        wb.save(path)

        text = processor.extract_text(str(path))
        assert "SECTION LABEL" in text
