#!/usr/bin/env python3
"""
End-to-end tests for CSVHandler and TSVHandler.

Tests the full 5-stage pipeline:
  Convert → Preprocess → MetadataExtract → ContentExtract → Postprocess

Covers:
 1. Basic CSV (comma-separated, UTF-8)
 2. TSV (tab-separated, UTF-8)
 3. BOM-prefixed CSV (UTF-8-sig)
 4. Korean CP949-encoded CSV
 5. Header detection (with and without header)
 6. Semicolon-delimited CSV
 7. Metadata extraction (encoding, delimiter, row/col count, header, columns)
 8. Table output in HTML and Markdown modes
 9. Merged cell detection (empty cell patterns → colspan/rowspan)
10. Empty file error handling
11. Edge cases (single column, single row, large column count)
"""

from __future__ import annotations

import sys
import os
import traceback

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from contextifier_new.config import ProcessingConfig, TableConfig
from contextifier_new.types import FileContext, OutputFormat, ExtractionResult, TableData
from contextifier_new.services.table_service import TableService
from contextifier_new.services.metadata_service import MetadataService
from contextifier_new.services.tag_service import TagService
from contextifier_new.handlers.csv.handler import CSVHandler
from contextifier_new.handlers.tsv.handler import TSVHandler
from contextifier_new.handlers.csv.converter import CsvConverter, CsvConvertedData
from contextifier_new.handlers.csv.preprocessor import (
    CsvPreprocessor,
    CsvParsedData,
    _detect_delimiter,
    _detect_header,
    _is_numeric,
)
from contextifier_new.handlers.csv.metadata_extractor import CsvMetadataExtractor
from contextifier_new.handlers.csv.content_extractor import (
    CsvContentExtractor,
    _has_merged_cells,
    _build_table_data,
)
from contextifier_new.errors import ConversionError


# ── Helpers ──────────────────────────────────────────────────────────────

passed = 0
failed = 0


def make_ctx(data: bytes, ext: str = "csv", name: str = "test.csv") -> FileContext:
    """Create a minimal FileContext for testing."""
    return {
        "file_data": data,
        "file_name": name,
        "file_extension": ext,
    }


def make_handler(ext: str = "csv", output_format: OutputFormat = OutputFormat.HTML):
    """Create a handler with services for the given extension."""
    tables_cfg = TableConfig(output_format=output_format)
    config = ProcessingConfig(tables=tables_cfg)
    table_service = TableService(config)
    metadata_service = MetadataService(config)
    tag_service = TagService(config)

    if ext == "tsv":
        return TSVHandler(
            config,
            table_service=table_service,
            metadata_service=metadata_service,
            tag_service=tag_service,
        )
    return CSVHandler(
        config,
        table_service=table_service,
        metadata_service=metadata_service,
        tag_service=tag_service,
    )


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Basic CSV pipeline ────────────────────────────────────────────────

def test_basic_csv_pipeline():
    print("\n═══ Test 1: Basic CSV Pipeline ═══")
    csv_text = "Name,Age,City\nAlice,30,Seoul\nBob,25,Busan\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("result is ExtractionResult", isinstance(result, ExtractionResult))
    check("text is not empty", bool(result.text))
    check("has tables", len(result.tables) > 0)
    check("metadata not None", result.metadata is not None)
    check("text contains <table>", "<table>" in result.text)
    check("text contains Alice", "Alice" in result.text)
    check("text contains Seoul", "Seoul" in result.text)
    check("metadata has encoding", "encoding" in (result.metadata.custom or {}))
    check("metadata has delimiter", "delimiter" in (result.metadata.custom or {}))
    check("metadata row_count=3", result.metadata.custom.get("row_count") == 3)
    check("metadata col_count=3", result.metadata.custom.get("col_count") == 3)
    check("metadata has_header=Yes", result.metadata.custom.get("has_header") == "Yes")
    check("metadata columns contain Name", "Name" in result.metadata.custom.get("columns", ""))


# ── 2. TSV pipeline ─────────────────────────────────────────────────────

def test_tsv_pipeline():
    print("\n═══ Test 2: TSV Pipeline ═══")
    tsv_text = "Name\tAge\tCity\nAlice\t30\tSeoul\nBob\t25\tBusan\n"
    ctx = make_ctx(tsv_text.encode("utf-8"), ext="tsv", name="test.tsv")
    handler = make_handler("tsv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("result is ExtractionResult", isinstance(result, ExtractionResult))
    check("text contains <table>", "<table>" in result.text)
    check("text contains Alice", "Alice" in result.text)
    check("metadata delimiter is Tab", "Tab" in result.metadata.custom.get("delimiter", ""))
    check("metadata row_count=3", result.metadata.custom.get("row_count") == 3)


# ── 3. BOM detection ────────────────────────────────────────────────────

def test_bom_csv():
    print("\n═══ Test 3: UTF-8 BOM CSV ═══")
    csv_text = "이름,나이,도시\n김철수,30,서울\n이영희,25,부산\n"
    bom_data = b"\xef\xbb\xbf" + csv_text.encode("utf-8")
    ctx = make_ctx(bom_data, ext="csv", name="korean_bom.csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("text contains 김철수", "김철수" in result.text)
    check("text contains 서울", "서울" in result.text)
    check("encoding is utf-8-sig", result.metadata.custom.get("encoding") == "utf-8-sig")


# ── 4. Korean CP949 encoding ────────────────────────────────────────────

def test_cp949_csv():
    print("\n═══ Test 4: CP949 Encoded CSV ═══")
    csv_text = "이름,나이,도시\n김철수,30,서울\n이영희,25,부산\n"
    cp949_data = csv_text.encode("cp949")
    ctx = make_ctx(cp949_data, ext="csv", name="korean_cp949.csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("text contains 김철수", "김철수" in result.text)
    check("encoding is cp949", result.metadata.custom.get("encoding") == "cp949")


# ── 5. No header detection ──────────────────────────────────────────────

def test_no_header_csv():
    print("\n═══ Test 5: CSV Without Header ═══")
    # First row also has numbers → not a header
    csv_text = "100,200,300\n400,500,600\n700,800,900\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("has_header=No", result.metadata.custom.get("has_header") == "No")
    check("no 'columns' in metadata", "columns" not in result.metadata.custom)


# ── 6. Semicolon delimiter ──────────────────────────────────────────────

def test_semicolon_csv():
    print("\n═══ Test 6: Semicolon-Delimited CSV ═══")
    csv_text = "Name;Age;City\nAlice;30;Seoul\nBob;25;Busan\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("text contains Alice", "Alice" in result.text)
    check("delimiter is Semicolon", "Semicolon" in result.metadata.custom.get("delimiter", ""))


# ── 7. Markdown output format ───────────────────────────────────────────

def test_markdown_output():
    print("\n═══ Test 7: Markdown Output ═══")
    csv_text = "Name,Age,City\nAlice,30,Seoul\nBob,25,Busan\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.MARKDOWN)

    result = handler.process(ctx)

    check("no <table> in Markdown mode", "<table>" not in result.text)
    check("has pipe character |", "|" in result.text)
    check("text contains Alice", "Alice" in result.text)


# ── 8. Merged cell detection ────────────────────────────────────────────

def test_merged_cells():
    print("\n═══ Test 8: Merged Cell Detection ═══")
    # Pattern: Category column spans 2 rows (empty cell below non-empty)
    csv_text = (
        "Category,Product,Price\n"
        "Fruit,Apple,100\n"
        ",Banana,200\n"   # <-- empty first col = vertical merge
        "Veggie,Carrot,150\n"
        ",Potato,120\n"   # <-- empty first col = vertical merge
    )
    rows = [row.split(",") for row in csv_text.strip().split("\n")]

    check("_has_merged_cells detects merge", _has_merged_cells(rows))

    table_data = _build_table_data(rows, has_header=True)
    check("table_data has rows", len(table_data.rows) > 0)

    # Check that the first non-header row's first cell has rowspan > 1
    # Row 0: header. Row 1: Fruit (rowspan=2). Row 2: (empty, skipped).
    fruit_row = table_data.rows[1]
    fruit_cell = fruit_row[0]
    check("Fruit cell has rowspan 2", fruit_cell.row_span == 2,
          f"got row_span={fruit_cell.row_span}")
    check("Fruit cell content is 'Fruit'", fruit_cell.content == "Fruit")

    # Test HTML output — should include rowspan attribute
    config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.HTML))
    ts = TableService(config)
    html = ts.format_table(table_data)
    check("HTML has rowspan attribute", 'rowspan="2"' in html)


# ── 9. Empty file error ─────────────────────────────────────────────────

def test_empty_file():
    print("\n═══ Test 9: Empty File Error ═══")
    ctx = make_ctx(b"", ext="csv")
    handler = make_handler("csv")

    try:
        handler.process(ctx)
        check("should raise error", False, "No exception raised for empty file")
    except Exception as e:
        check("raises ConversionError", "ConversionError" in type(e).__name__ or "empty" in str(e).lower(),
              f"got {type(e).__name__}: {e}")


# ── 10. Unit tests for _detect_delimiter ─────────────────────────────────

def test_delimiter_detection():
    print("\n═══ Test 10: Delimiter Detection ═══")
    check("comma detected", _detect_delimiter("a,b,c\n1,2,3\n") == ",")
    check("tab detected", _detect_delimiter("a\tb\tc\n1\t2\t3\n") == "\t")
    check("semicolon detected", _detect_delimiter("a;b;c\n1;2;3\n") == ";")
    check("pipe detected", _detect_delimiter("a|b|c\n1|2|3\n") == "|")


# ── 11. Unit tests for _is_numeric ───────────────────────────────────────

def test_is_numeric():
    print("\n═══ Test 11: Numeric Detection ═══")
    check("123 is numeric", _is_numeric("123"))
    check("-45.6 is numeric", _is_numeric("-45.6"))
    check("1,234,567 is numeric", _is_numeric("1,234,567"))
    check("12.5% is numeric", _is_numeric("12.5%"))
    check("$99.99 is numeric", _is_numeric("$99.99"))
    check("₩10,000 is numeric", _is_numeric("₩10,000"))
    check("'hello' not numeric", not _is_numeric("hello"))
    check("empty not numeric", not _is_numeric(""))


# ── 12. Unit tests for _detect_header ────────────────────────────────────

def test_header_detection():
    print("\n═══ Test 12: Header Detection ═══")
    rows_with_header = [["Name", "Age", "City"], ["Alice", "30", "Seoul"]]
    rows_no_header = [["100", "200", "300"], ["400", "500", "600"]]
    rows_single = [["Only one row"]]

    check("detects header", _detect_header(rows_with_header))
    check("no header for numeric first row", not _detect_header(rows_no_header))
    check("no header for single row", not _detect_header(rows_single))


# ── 13. Single column CSV ───────────────────────────────────────────────

def test_single_column():
    print("\n═══ Test 13: Single Column CSV ═══")
    csv_text = "Name\nAlice\nBob\nCharlie\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("col_count=1", result.metadata.custom.get("col_count") == 1)
    check("text contains Alice", "Alice" in result.text)


# ── 14. Large number of columns ─────────────────────────────────────────

def test_many_columns():
    print("\n═══ Test 14: Many Columns CSV ═══")
    headers = [f"col{i}" for i in range(20)]
    values = [str(i) for i in range(20)]
    csv_text = ",".join(headers) + "\n" + ",".join(values) + "\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("col_count=20", result.metadata.custom.get("col_count") == 20)
    # Metadata should show first 10 columns + "(+10 more)"
    col_meta = result.metadata.custom.get("columns", "")
    check("columns shows +10 more", "+10 more" in col_meta,
          f"got: {col_meta}")


# ── 15. Converter unit test ─────────────────────────────────────────────

def test_converter_unit():
    print("\n═══ Test 15: CsvConverter Unit ═══")
    converter = CsvConverter()
    ctx = make_ctx("hello,world\n1,2\n".encode("utf-8"), ext="csv")

    result = converter.convert(ctx)
    check("returns CsvConvertedData", isinstance(result, CsvConvertedData))
    check("text decoded", "hello,world" in result.text)
    check("encoding is utf-8", result.encoding == "utf-8")
    check("extension is csv", result.file_extension == "csv")


# ── 16. Preprocessor unit test ──────────────────────────────────────────

def test_preprocessor_unit():
    print("\n═══ Test 16: CsvPreprocessor Unit ═══")
    preprocessor = CsvPreprocessor(default_delimiter=None)
    converted = CsvConvertedData(
        text="Name,Age\nAlice,30\nBob,25\n",
        encoding="utf-8",
        file_extension="csv",
    )

    result = preprocessor.preprocess(converted)
    parsed = result.content

    check("content is CsvParsedData", isinstance(parsed, CsvParsedData))
    check("row_count=3", parsed.row_count == 3)
    check("has_header=True", parsed.has_header is True)
    check("delimiter=comma", parsed.delimiter == ",")
    check("first row is header", parsed.rows[0] == ["Name", "Age"])


# ── 17. TSV preprocessor with forced delimiter ──────────────────────────

def test_tsv_preprocessor():
    print("\n═══ Test 17: TSV Preprocessor (Forced Delimiter) ═══")
    preprocessor = CsvPreprocessor(default_delimiter="\t")
    # Data with BOTH commas and tabs — comma appears more, but tab should win
    converted = CsvConvertedData(
        text="Name\tAge,Country\nAlice\t30,Korea\n",
        encoding="utf-8",
        file_extension="tsv",
    )

    result = preprocessor.preprocess(converted)
    parsed = result.content

    check("delimiter is tab", parsed.delimiter == "\t")
    check("first cell is Name", parsed.rows[0][0] == "Name")
    # With tab delimiter, "Age,Country" should be a single cell
    check("second cell contains comma", "Age,Country" in parsed.rows[0][1])


# ── 18. ContentExtractor with no merge ──────────────────────────────────

def test_content_extractor_no_merge():
    print("\n═══ Test 18: ContentExtractor (Simple Table) ═══")
    config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.HTML))
    ts = TableService(config)
    extractor = CsvContentExtractor(table_service=ts)

    from contextifier_new.types import PreprocessedData
    parsed = CsvParsedData(
        rows=[["Name", "Age"], ["Alice", "30"], ["Bob", "25"]],
        has_header=True,
        delimiter=",",
        encoding="utf-8",
        row_count=3,
        col_count=2,
    )
    preprocessed = PreprocessedData(content=parsed)

    text = extractor.extract_text(preprocessed)
    tables = extractor.extract_tables(preprocessed)

    check("text contains <table>", "<table>" in text)
    check("text contains <th>Name</th>", "<th>Name</th>" in text)
    check("text contains <td>Alice</td>", "<td>Alice</td>" in text)
    check("returns 1 TableData", len(tables) == 1)
    check("TableData has 3 rows", tables[0].num_rows == 3)
    check("TableData has 2 cols", tables[0].num_cols == 2)


# ── 19. No merged cells for simple data ─────────────────────────────────

def test_no_merged_cells():
    print("\n═══ Test 19: No Merged Cells Detection ═══")
    rows = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"]]
    check("simple data has no merged cells", not _has_merged_cells(rows))


# ── 20. extract_text convenience method ─────────────────────────────────

def test_extract_text_convenience():
    print("\n═══ Test 20: extract_text() Convenience ═══")
    csv_text = "Name,Age\nAlice,30\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.HTML)

    text = handler.extract_text(ctx)

    check("returns string", isinstance(text, str))
    check("contains Alice", "Alice" in text)
    check("contains metadata block", "Document-Metadata" in text or "encoding" in text)


# ── 21. Metadata block in postprocessed output ──────────────────────────

def test_metadata_in_output():
    print("\n═══ Test 21: Metadata Block in Output ═══")
    csv_text = "Name,Age\nAlice,30\n"
    ctx = make_ctx(csv_text.encode("utf-8"), ext="csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)
    text = result.text

    check("output has metadata prefix", "<Document-Metadata>" in text)
    check("output has encoding info", "encoding" in text)
    check("output has delimiter info", "delimiter" in text)
    check("output has row_count", "row_count" in text)
    check("output has <table>", "<table>" in text)


# ── 22. UTF-16 BOM CSV ──────────────────────────────────────────────────

def test_utf16_bom():
    print("\n═══ Test 22: UTF-16 LE BOM CSV ═══")
    csv_text = "Name,Age\nAlice,30\n"
    utf16_data = b"\xff\xfe" + csv_text.encode("utf-16-le")
    ctx = make_ctx(utf16_data, ext="csv", name="utf16.csv")
    handler = make_handler("csv", OutputFormat.HTML)

    result = handler.process(ctx)

    check("text contains Alice", "Alice" in result.text)
    check("encoding is utf-16-le", result.metadata.custom.get("encoding") == "utf-16-le")


# ═══════════════════════════════════════════════════════════════════════════
# Run All Tests
# ═══════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        test_basic_csv_pipeline,
        test_tsv_pipeline,
        test_bom_csv,
        test_cp949_csv,
        test_no_header_csv,
        test_semicolon_csv,
        test_markdown_output,
        test_merged_cells,
        test_empty_file,
        test_delimiter_detection,
        test_is_numeric,
        test_header_detection,
        test_single_column,
        test_many_columns,
        test_converter_unit,
        test_preprocessor_unit,
        test_tsv_preprocessor,
        test_content_extractor_no_merge,
        test_no_merged_cells,
        test_extract_text_convenience,
        test_metadata_in_output,
        test_utf16_bom,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            global failed
            failed += 1
            print(f"  [ERROR] {test_fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
