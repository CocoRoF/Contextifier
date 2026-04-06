# tests/unit/handlers/test_csv_streaming.py
"""
P4-1: Tests for CSV large-file streaming and max_rows option.

Verifies:
1. CsvPreprocessor respects max_rows limit
2. Truncated flag is set when rows exceed limit
3. format_options["csv"]["max_rows"] flows through CSVHandler
4. format_options["tsv"]["max_rows"] flows through TSVHandler
5. Default MAX_ROWS constant is used when no option specified
6. Fallback parser (_parse_csv_simple) also respects max_rows
"""

from __future__ import annotations

import unittest

from contextifier.handlers.csv.preprocessor import (
    CsvPreprocessor,
    CsvParsedData,
    MAX_ROWS,
    _parse_csv_content,
    _parse_csv_simple,
)
from contextifier.config import ProcessingConfig


class TestParseMaxRows(unittest.TestCase):
    """Low-level _parse_csv_content with max_rows."""

    def _make_csv(self, n: int) -> str:
        """Generate CSV text with *n* data rows (no header)."""
        return "\n".join(f"a{i},b{i},c{i}" for i in range(n))

    def test_no_truncation_under_limit(self):
        text = self._make_csv(5)
        rows, truncated = _parse_csv_content(text, ",", max_rows=10)
        self.assertEqual(len(rows), 5)
        self.assertFalse(truncated)

    def test_truncation_at_limit(self):
        text = self._make_csv(20)
        rows, truncated = _parse_csv_content(text, ",", max_rows=10)
        self.assertEqual(len(rows), 10)
        self.assertTrue(truncated)

    def test_exact_limit(self):
        text = self._make_csv(10)
        rows, truncated = _parse_csv_content(text, ",", max_rows=10)
        self.assertEqual(len(rows), 10)
        self.assertFalse(truncated)

    def test_default_uses_max_rows_constant(self):
        """Default max_rows should be the module-level MAX_ROWS."""
        text = self._make_csv(5)
        rows, truncated = _parse_csv_content(text, ",")
        self.assertEqual(len(rows), 5)
        self.assertFalse(truncated)


class TestParseSimpleMaxRows(unittest.TestCase):
    """Low-level _parse_csv_simple with max_rows."""

    def _make_csv(self, n: int) -> str:
        return "\n".join(f"a{i},b{i},c{i}" for i in range(n))

    def test_truncation(self):
        text = self._make_csv(15)
        rows, truncated = _parse_csv_simple(text, ",", max_rows=8)
        self.assertEqual(len(rows), 8)
        self.assertTrue(truncated)

    def test_no_truncation(self):
        text = self._make_csv(5)
        rows, truncated = _parse_csv_simple(text, ",", max_rows=10)
        self.assertEqual(len(rows), 5)
        self.assertFalse(truncated)


class TestCsvPreprocessorMaxRows(unittest.TestCase):
    """CsvPreprocessor with max_rows parameter."""

    def _make_csv(self, n: int) -> str:
        header = "col_a,col_b,col_c"
        data = "\n".join(f"{i},val_{i},{i*10}" for i in range(n))
        return f"{header}\n{data}"

    def test_default_max_rows(self):
        """Without max_rows, uses module constant."""
        pp = CsvPreprocessor()
        self.assertEqual(pp._max_rows, MAX_ROWS)

    def test_custom_max_rows(self):
        """Custom max_rows is stored correctly."""
        pp = CsvPreprocessor(max_rows=50)
        self.assertEqual(pp._max_rows, 50)

    def test_preprocess_truncates(self):
        """Preprocessor limits rows and sets truncated flag."""
        pp = CsvPreprocessor(max_rows=5)
        text = self._make_csv(20)  # header + 20 data rows = 21 lines
        result = pp.preprocess(text)
        parsed: CsvParsedData = result.content
        self.assertLessEqual(parsed.row_count, 5)
        self.assertTrue(parsed.truncated)
        self.assertTrue(result.properties["truncated"])

    def test_preprocess_no_truncation(self):
        """Under limit — truncated=False."""
        pp = CsvPreprocessor(max_rows=100)
        text = self._make_csv(10)
        result = pp.preprocess(text)
        parsed: CsvParsedData = result.content
        self.assertEqual(parsed.row_count, 11)  # header + 10 data
        self.assertFalse(parsed.truncated)
        self.assertFalse(result.properties["truncated"])


class TestCsvHandlerMaxRowsOption(unittest.TestCase):
    """CSVHandler reads max_rows from format_options."""

    def test_csv_handler_passes_max_rows(self):
        from contextifier.handlers.csv.handler import CSVHandler
        config = ProcessingConfig(
            format_options={"csv": {"max_rows": 500}}
        )
        handler = CSVHandler(config=config)
        pp = handler.create_preprocessor()
        self.assertEqual(pp._max_rows, 500)

    def test_csv_handler_default_max_rows(self):
        from contextifier.handlers.csv.handler import CSVHandler
        handler = CSVHandler(config=ProcessingConfig())
        pp = handler.create_preprocessor()
        self.assertEqual(pp._max_rows, MAX_ROWS)


class TestTsvHandlerMaxRowsOption(unittest.TestCase):
    """TSVHandler reads max_rows from format_options."""

    def test_tsv_handler_passes_max_rows(self):
        from contextifier.handlers.tsv.handler import TSVHandler
        config = ProcessingConfig(
            format_options={"tsv": {"max_rows": 200}}
        )
        handler = TSVHandler(config=config)
        pp = handler.create_preprocessor()
        self.assertEqual(pp._max_rows, 200)


class TestCsvParsedDataTruncated(unittest.TestCase):
    """CsvParsedData.truncated field."""

    def test_default_false(self):
        pd = CsvParsedData(
            rows=[], has_header=False, delimiter=",",
            encoding="utf-8", row_count=0, col_count=0,
        )
        self.assertFalse(pd.truncated)

    def test_explicit_true(self):
        pd = CsvParsedData(
            rows=[], has_header=False, delimiter=",",
            encoding="utf-8", row_count=0, col_count=0,
            truncated=True,
        )
        self.assertTrue(pd.truncated)


if __name__ == "__main__":
    unittest.main()
