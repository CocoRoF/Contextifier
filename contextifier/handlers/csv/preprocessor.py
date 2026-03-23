# contextifier/handlers/csv/preprocessor.py
"""
CsvPreprocessor — Stage 2: Decoded text → Parsed CSV data

Takes the CsvConvertedData from Stage 1 and produces a
PreprocessedData containing structured CSV information:

1. Delimiter detection (csv.Sniffer + manual heuristic, or forced)
2. CSV parsing via csv.reader (with simple-split fallback)
3. Header detection heuristic
4. Row/column count computation

The parsed result is stored as a CsvParsedData NamedTuple in
PreprocessedData.content, providing type-safe access for
downstream stages (MetadataExtractor and ContentExtractor).

Delimiter handling:
- CSVHandler: delimiter=None → auto-detect
- TSVHandler: delimiter='\\t' → forced, skip detection

v1.0 logic ported:
- detect_delimiter: csv.Sniffer → manual consistency scoring
- parse_csv_content: csv.reader with MAX_ROWS/MAX_COLS limits
- parse_csv_simple: simple split fallback on csv.Error
- detect_header: first row all text + second row has numbers + uniqueness
- is_numeric: multi-pattern numeric detection (int, float, thousands, %, currencies)
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any, List, NamedTuple, Optional

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.handlers.csv.converter import CsvConvertedData


# ── Processing limits ────────────────────────────────────────────────────

MAX_ROWS: int = 100_000
"""Maximum rows to process (memory protection)."""

MAX_COLS: int = 1_000
"""Maximum columns to process."""

DELIMITER_CANDIDATES: List[str] = [",", "\t", ";", "|"]
"""Delimiters to consider during auto-detection."""


# ── Parsed data structure ────────────────────────────────────────────────

class CsvParsedData(NamedTuple):
    """
    Structured output of the CsvPreprocessor.

    Stored in PreprocessedData.content and consumed by
    CsvMetadataExtractor and CsvContentExtractor.
    """
    rows: List[List[str]]
    has_header: bool
    delimiter: str
    encoding: str
    row_count: int
    col_count: int


# ── Preprocessor ─────────────────────────────────────────────────────────

class CsvPreprocessor(BasePreprocessor):
    """
    Preprocessor for CSV and TSV files.

    Parses decoded text into structured row/column data using
    Python's csv module, with automatic delimiter and header detection.

    Args:
        default_delimiter: If provided, skip delimiter auto-detection
            and use this delimiter. TSVHandler passes ``'\\t'`` here.
    """

    def __init__(self, default_delimiter: Optional[str] = None) -> None:
        super().__init__()
        self._default_delimiter = default_delimiter

    def preprocess(
        self,
        converted_data: Any,
        **kwargs: Any,
    ) -> PreprocessedData:
        """
        Parse CSV text into structured rows.

        Accepts either:
        - CsvConvertedData (from CsvConverter) — preferred
        - Plain str — for testing or standalone use

        Args:
            converted_data: Output from CsvConverter.
            **kwargs: Optional ``delimiter`` override.

        Returns:
            PreprocessedData with CsvParsedData in ``content``.
        """
        text, encoding, file_extension = self._unpack(converted_data)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Determine delimiter
        delimiter = (
            kwargs.get("delimiter")
            or self._default_delimiter
        )
        if delimiter is None:
            delimiter = _detect_delimiter(text)

        # Parse CSV rows
        rows = _parse_csv_content(text, delimiter)

        # Detect header
        has_header = _detect_header(rows)

        # Compute dimensions
        row_count = len(rows)
        col_count = max((len(r) for r in rows), default=0)

        parsed = CsvParsedData(
            rows=rows,
            has_header=has_header,
            delimiter=delimiter,
            encoding=encoding,
            row_count=row_count,
            col_count=col_count,
        )

        return PreprocessedData(
            content=parsed,
            raw_content=text,
            encoding=encoding,
            resources={},
            properties={
                "file_extension": file_extension,
                "delimiter": delimiter,
                "has_header": has_header,
                "row_count": row_count,
                "col_count": col_count,
            },
        )

    def get_format_name(self) -> str:
        return "csv"

    def validate(self, data: Any) -> bool:
        """Accept CsvConvertedData or str."""
        if isinstance(data, str):
            return True
        if isinstance(data, tuple) and hasattr(data, "text"):
            return True
        return data is not None

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(converted_data: Any) -> tuple[str, str, str]:
        """Extract text + metadata from converter output."""
        if isinstance(converted_data, CsvConvertedData):
            return (
                converted_data.text,
                converted_data.encoding,
                converted_data.file_extension,
            )
        if isinstance(converted_data, str):
            return (converted_data, "utf-8", "")
        return (str(converted_data) if converted_data else "", "utf-8", "")


# ═══════════════════════════════════════════════════════════════════════════
# CSV Parsing Functions (module-level for testability)
# ═══════════════════════════════════════════════════════════════════════════

_logger = logging.getLogger("contextifier.csv.preprocessor")


def _detect_delimiter(content: str) -> str:
    """
    Auto-detect the CSV delimiter.

    Strategy:
    1. Use csv.Sniffer on the first 20 lines.
    2. If Sniffer fails, score each candidate delimiter by
       counting per-line consistency (same count on every line).

    Args:
        content: Decoded CSV text.

    Returns:
        Detected delimiter character. Defaults to ``','``.
    """
    sample_lines = content.split("\n")[:20]
    sample = "\n".join(sample_lines)

    # Phase 1: csv.Sniffer
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        pass

    # Phase 2: Manual consistency scoring
    best_delimiter = ","
    best_score = 0.0

    non_empty_lines = [line for line in sample_lines if line.strip()]
    if not non_empty_lines:
        return ","

    for delim in DELIMITER_CANDIDATES:
        counts = [line.count(delim) for line in non_empty_lines]
        if not counts or max(counts) == 0:
            continue

        # Perfect consistency: same count on every line → high bonus
        if len(set(counts)) == 1 and counts[0] > 0:
            score = float(counts[0]) * 10.0
        else:
            score = sum(counts) / len(counts)

        if score > best_score:
            best_score = score
            best_delimiter = delim

    return best_delimiter


def _parse_csv_content(content: str, delimiter: str) -> List[List[str]]:
    """
    Parse CSV content into rows using csv.reader.

    Falls back to simple line-splitting on csv.Error.
    Limits output to MAX_ROWS rows and MAX_COLS columns.
    Skips completely empty rows.

    Args:
        content: Decoded CSV text (LF-normalized).
        delimiter: Column delimiter.

    Returns:
        List of rows, each row a list of cell strings.
    """
    try:
        reader = csv.reader(
            io.StringIO(content),
            delimiter=delimiter,
            quotechar='"',
            doublequote=True,
            skipinitialspace=True,
        )

        rows: List[List[str]] = []
        for i, row in enumerate(reader):
            if i >= MAX_ROWS:
                _logger.warning("CSV row limit reached: %d", MAX_ROWS)
                break
            if len(row) > MAX_COLS:
                row = row[:MAX_COLS]
            # Skip fully empty rows
            if any(cell.strip() for cell in row):
                rows.append(row)
        return rows

    except csv.Error as e:
        _logger.warning("csv.reader failed (%s), falling back to simple split", e)
        return _parse_csv_simple(content, delimiter)


def _parse_csv_simple(content: str, delimiter: str) -> List[List[str]]:
    """
    Simple fallback CSV parser using str.split().

    Used when csv.reader raises csv.Error (malformed quoting, etc.).

    Args:
        content: Decoded CSV text.
        delimiter: Column delimiter.

    Returns:
        List of rows.
    """
    rows: List[List[str]] = []
    for i, line in enumerate(content.split("\n")):
        if i >= MAX_ROWS:
            break
        line = line.strip()
        if not line:
            continue
        cells = line.split(delimiter)
        if len(cells) > MAX_COLS:
            cells = cells[:MAX_COLS]
        rows.append(cells)
    return rows


def _detect_header(rows: List[List[str]]) -> bool:
    """
    Heuristic: detect whether the first row is a header.

    Criteria (any combination signals a header):
    1. All cells in the first row are non-numeric.
    2. At least one cell in the second row IS numeric.
    3. First row cells are all unique.

    Args:
        rows: Parsed row data.

    Returns:
        True if the first row appears to be a header.
    """
    if len(rows) < 2:
        return False

    first_row = rows[0]
    second_row = rows[1]

    first_all_text = all(
        not _is_numeric(cell) for cell in first_row if cell.strip()
    )
    second_has_numbers = any(
        _is_numeric(cell) for cell in second_row if cell.strip()
    )
    first_unique = len(set(first_row)) == len(first_row)

    return first_all_text and (second_has_numbers or first_unique)


# Compiled numeric patterns for performance
_NUMERIC_PATTERNS = [
    re.compile(r"^-?\d+$"),                         # Integer
    re.compile(r"^-?\d+\.\d+$"),                    # Float
    re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$"),   # Thousands separators
    re.compile(r"^-?\d+(\.\d+)?%$"),                # Percentage
    re.compile(r"^\$-?\d+(\.\d+)?$"),               # USD
    re.compile(r"^₩-?\d+(,\d{3})*$"),               # KRW
]


def _is_numeric(value: str) -> bool:
    """
    Check if a string represents a numeric value.

    Supports: integers, floats, thousands-separated numbers,
    percentages, USD ($), KRW (₩).

    Args:
        value: Cell content to check.

    Returns:
        True if the value looks numeric.
    """
    if not value or not value.strip():
        return False
    value = value.strip()
    return any(p.match(value) for p in _NUMERIC_PATTERNS)


__all__ = [
    "CsvPreprocessor",
    "CsvParsedData",
    "MAX_ROWS",
    "MAX_COLS",
    "DELIMITER_CANDIDATES",
]
