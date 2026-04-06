# tests/unit/handlers/test_csv_delimiter_confidence.py
"""Tests for CSV delimiter confidence scoring (P6-6)."""

from __future__ import annotations

import pytest

from contextifier.handlers.csv.preprocessor import (
    CsvParsedData,
    CsvPreprocessor,
    _detect_delimiter,
)


# ═══════════════════════════════════════════════════════════════════════════
# _detect_delimiter — confidence
# ═══════════════════════════════════════════════════════════════════════════


class TestDelimiterConfidence:
    """Tests for delimiter confidence score returned by _detect_delimiter."""

    def test_comma_perfect_consistency(self):
        """Perfectly consistent comma-separated data → high confidence."""
        content = "a,b,c\n1,2,3\n4,5,6\n7,8,9"
        delim, confidence = _detect_delimiter(content)
        assert delim == ","
        assert confidence >= 0.8

    def test_semicolon_perfect_consistency(self):
        """Perfectly consistent semicolon-separated data."""
        content = "a;b;c\n1;2;3\n4;5;6"
        delim, confidence = _detect_delimiter(content)
        assert delim == ";"
        assert confidence >= 0.8

    def test_pipe_perfect_consistency(self):
        """Perfectly consistent pipe-separated data."""
        content = "a|b|c\n1|2|3\n4|5|6"
        delim, confidence = _detect_delimiter(content)
        assert delim == "|"
        assert confidence >= 0.8

    def test_tab_perfect_consistency(self):
        """Perfectly consistent tab-separated data."""
        content = "a\tb\tc\n1\t2\t3\n4\t5\t6"
        delim, confidence = _detect_delimiter(content)
        assert delim == "\t"
        assert confidence >= 0.8

    def test_inconsistent_data_lower_confidence(self):
        """Rows with different delimiter counts → lower confidence."""
        # Line 1 has 2 commas, line 2 has 1, line 3 has 3
        content = "a,b,c\n1,2\n4,5,6,7"
        delim, confidence = _detect_delimiter(content)
        assert delim == ","
        assert confidence < 1.0

    def test_empty_content_zero_confidence(self):
        """Empty content → comma fallback with 0 confidence."""
        delim, confidence = _detect_delimiter("")
        assert delim == ","
        assert confidence == 0.0

    def test_no_delimiter_found(self):
        """Single-column data → comma fallback with 0 confidence."""
        content = "hello\nworld\nfoo"
        delim, confidence = _detect_delimiter(content)
        assert delim == ","
        assert confidence == 0.0

    def test_confidence_is_float_between_0_and_1(self):
        """Confidence is always in [0.0, 1.0] range."""
        test_data = [
            "a,b\n1,2",
            "a;b;c\n1;2;3",
            "hello",
            "",
            "a,b,c\n1,2\n3,4,5,6,7",
        ]
        for content in test_data:
            _, confidence = _detect_delimiter(content)
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range for content: {content!r}"

    def test_returns_tuple(self):
        """_detect_delimiter returns a 2-tuple (str, float)."""
        result = _detect_delimiter("a,b,c\n1,2,3")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)


# ═══════════════════════════════════════════════════════════════════════════
# CsvParsedData — delimiter_confidence field
# ═══════════════════════════════════════════════════════════════════════════


class TestCsvParsedDataConfidence:
    """Tests for CsvParsedData.delimiter_confidence field."""

    def test_default_confidence_is_1(self):
        """Default delimiter_confidence is 1.0."""
        parsed = CsvParsedData(
            rows=[["a", "b"]],
            has_header=False,
            delimiter=",",
            encoding="utf-8",
            row_count=1,
            col_count=2,
        )
        assert parsed.delimiter_confidence == 1.0

    def test_custom_confidence(self):
        """delimiter_confidence can be set explicitly."""
        parsed = CsvParsedData(
            rows=[["a", "b"]],
            has_header=False,
            delimiter=",",
            encoding="utf-8",
            row_count=1,
            col_count=2,
            delimiter_confidence=0.75,
        )
        assert parsed.delimiter_confidence == 0.75


# ═══════════════════════════════════════════════════════════════════════════
# CsvPreprocessor — confidence in properties
# ═══════════════════════════════════════════════════════════════════════════


class TestPreprocessorConfidenceOutput:
    """Tests for confidence in PreprocessedData.properties."""

    def test_confidence_in_properties(self):
        """Preprocessor includes delimiter_confidence in properties."""
        preprocessor = CsvPreprocessor()
        result = preprocessor.preprocess("a,b,c\n1,2,3\n4,5,6")
        assert "delimiter_confidence" in result.properties
        assert isinstance(result.properties["delimiter_confidence"], float)
        assert 0.0 <= result.properties["delimiter_confidence"] <= 1.0

    def test_forced_delimiter_has_full_confidence(self):
        """Forced delimiter (e.g. TSV) → confidence = 1.0."""
        preprocessor = CsvPreprocessor(default_delimiter="\t")
        result = preprocessor.preprocess("a\tb\n1\t2")
        assert result.properties["delimiter_confidence"] == 1.0

    def test_confidence_in_parsed_data(self):
        """CsvParsedData in content has confidence field."""
        preprocessor = CsvPreprocessor()
        result = preprocessor.preprocess("a;b;c\n1;2;3")
        parsed = result.content
        assert hasattr(parsed, "delimiter_confidence")
        assert isinstance(parsed.delimiter_confidence, float)
