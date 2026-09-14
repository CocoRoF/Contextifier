# tests/unit/handlers/test_html_disguised_as_excel.py
"""
Spreadsheets that are really HTML.

Report portals — Korean government sites in particular — serve an HTML table
with a ``.xls`` or ``.xlsx`` filename. Excel opens them, so users treat them as
spreadsheets, but xlrd and openpyxl cannot: the whole document fails to
convert and nothing is extracted.

Detection is unambiguous rather than heuristic: a real XLS begins with the OLE
compound-file magic and a real XLSX with the ZIP magic, so neither can begin
with markup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextifier import DocumentProcessor
from contextifier.handlers.html._detect import looks_like_html

_GOV_EXPORT = """<html xmlns:o="urn:schemas-microsoft-com:office:office">
<head><meta charset="utf-8"><title>Quarterly Report</title></head>
<body>
<table border="1"><thead><tr><th>Item</th><th>2025</th><th>2026</th></tr></thead></table>
<table border="1"><tbody>
<tr><td>Revenue</td><td>1,200</td><td>1,450</td></tr>
<tr><td>Operating profit</td><td>180</td><td>210</td></tr>
</tbody></table>
</body></html>
"""


class TestDetection:
    @pytest.mark.parametrize(
        "payload",
        [
            b"<html><body><table></table></body></html>",
            b"<!DOCTYPE html>\n<html></html>",
            b"\xef\xbb\xbf<html></html>",  # UTF-8 BOM
            b"\n\n  <HTML></HTML>",  # leading whitespace, upper case
            b'<?xml version="1.0"?><html><body></body></html>',
            b"<table><tr><td>bare table</td></tr></table>",
        ],
    )
    def test_markup_is_detected(self, payload: bytes) -> None:
        assert looks_like_html(payload) is True

    @pytest.mark.parametrize(
        "payload",
        [
            b"PK\x03\x04" + b"\x00" * 40,  # XLSX
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 40,  # XLS (OLE)
            b"Name,Age\nAlice,30\n",  # CSV
            b"",
            b"short",
        ],
    )
    def test_real_spreadsheets_are_not_flagged(self, payload: bytes) -> None:
        assert looks_like_html(payload) is False

    def test_excel_2003_xml_is_not_html(self) -> None:
        """SpreadsheetML also opens with `<?xml` but is a real workbook."""
        payload = (
            b'<?xml version="1.0"?>\n'
            b'<?mso-application progid="Excel.Sheet"?>\n'
            b'<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet">'
            b"<Worksheet ss:Name=\"Sheet1\"/></Workbook>"
        )
        assert looks_like_html(payload) is False


class TestExtraction:
    @pytest.mark.parametrize("suffix", [".xls", ".xlsx"])
    def test_html_export_extracts_instead_of_failing(
        self, tmp_path: Path, suffix: str
    ) -> None:
        path = tmp_path / f"report{suffix}"
        path.write_text(_GOV_EXPORT, encoding="utf-8")

        text = DocumentProcessor().extract_text(str(path))

        assert "Revenue" in text
        assert "1,450" in text
        assert "Operating profit" in text

    def test_real_xlsx_is_unaffected(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")

        path = tmp_path / "real.xlsx"
        wb = openpyxl.Workbook()
        wb.active.title = "Data"
        wb.active.append(["Name", "Score"])
        wb.active.append(["Alice", 95])
        wb.save(path)

        text = DocumentProcessor().extract_text(str(path))
        assert "[Sheet: Data]" in text
        assert "Alice" in text


class TestSplitTableRepair:
    """Header-in-one-table / data-in-the-next is a common export shape."""

    def test_header_and_body_tables_are_merged(self, tmp_path: Path) -> None:
        path = tmp_path / "report.xls"
        path.write_text(_GOV_EXPORT, encoding="utf-8")

        text = DocumentProcessor().extract_text(str(path))

        assert text.count("<table>") == 1, (
            "the heading row and the data rows must end up in one table:\n" + text
        )
        header_pos = text.index("Item")
        assert header_pos < text.index("Revenue")

    def test_unrelated_adjacent_tables_are_left_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "two.html"
        path.write_text(
            "<html><body>"
            "<table><tr><td>first table</td></tr></table>"
            "<table><tr><td>second table</td></tr></table>"
            "</body></html>",
            encoding="utf-8",
        )
        text = DocumentProcessor().extract_text(str(path))
        assert text.count("<table>") == 2

    def test_column_count_mismatch_prevents_merging(self, tmp_path: Path) -> None:
        path = tmp_path / "mismatch.html"
        path.write_text(
            "<html><body>"
            "<table><tr><th>a</th><th>b</th></tr></table>"
            "<table><tr><td>1</td><td>2</td><td>3</td></tr></table>"
            "</body></html>",
            encoding="utf-8",
        )
        text = DocumentProcessor().extract_text(str(path))
        assert text.count("<table>") == 2

    def test_text_between_tables_prevents_merging(self, tmp_path: Path) -> None:
        path = tmp_path / "separated.html"
        path.write_text(
            "<html><body>"
            "<table><tr><th>a</th></tr></table>"
            "<p>An explanatory paragraph.</p>"
            "<table><tr><td>1</td></tr></table>"
            "</body></html>",
            encoding="utf-8",
        )
        text = DocumentProcessor().extract_text(str(path))
        assert text.count("<table>") == 2

    def test_a_normal_table_is_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "normal.html"
        path.write_text(
            "<html><body><table>"
            "<tr><th>Name</th><th>Score</th></tr>"
            "<tr><td>Alice</td><td>95</td></tr>"
            "</table></body></html>",
            encoding="utf-8",
        )
        text = DocumentProcessor().extract_text(str(path))
        assert text.count("<table>") == 1
        assert "Alice" in text and "Name" in text
