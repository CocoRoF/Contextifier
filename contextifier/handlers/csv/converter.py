# contextifier/handlers/csv/converter.py
"""
CsvConverter — Stage 1: Binary bytes → Decoded string

Converts raw binary CSV/TSV data into a decoded Python string
with BOM-aware encoding detection.

CSV files commonly have UTF-8 BOM (from Excel export) or may use
UTF-16 with BOM. This converter checks for BOMs first, then falls
through to the standard encoding candidate list.

Returns a CsvConvertedData NamedTuple that carries:
- text: the decoded string
- encoding: the encoding that succeeded
- file_extension: from FileContext (csv or tsv)

Reused by both CSVHandler and TSVHandler since encoding detection
is format-independent (delimiter is handled in the Preprocessor stage).
"""

from __future__ import annotations

import logging
from typing import Any, List, NamedTuple, Optional

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError


class CsvConvertedData(NamedTuple):
    """
    Output of the CsvConverter.

    Carries decoded text and metadata needed by downstream stages.
    """
    text: str
    encoding: str
    file_extension: str


# BOM detection table.
# Sorted by longest BOM first to avoid false positives
# (e.g., UTF-32-LE starts with the same bytes as UTF-16-LE).
_BOM_TABLE: List[tuple[bytes, str]] = [
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xef\xbb\xbf",     "utf-8-sig"),
    (b"\xff\xfe",          "utf-16-le"),
    (b"\xfe\xff",          "utf-16-be"),
]

# Default encoding priority for CSV/TSV files.
# utf-8-sig handles BOM-prefixed UTF-8.
# cp949/euc-kr cover Korean Windows/legacy environments.
# iso-8859-1 (= latin-1) is a lossless 8-bit fallback.
DEFAULT_ENCODINGS: List[str] = [
    "utf-8",
    "utf-8-sig",
    "cp949",
    "euc-kr",
    "iso-8859-1",
    "latin-1",
]


class CsvConverter(BaseConverter):
    """
    BOM-aware converter for CSV and TSV files.

    Decodes binary data to string in two phases:
    1. Check for BOM — if found, use the corresponding encoding.
    2. If no BOM, try each candidate encoding in order.
    3. Fallback: UTF-8 with ``errors='replace'``.

    Encoding list can be customized via constructor or per-call kwargs.
    """

    def __init__(self, encodings: Optional[List[str]] = None) -> None:
        super().__init__()
        self._encodings = encodings or list(DEFAULT_ENCODINGS)

    def convert(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> CsvConvertedData:
        """
        Decode binary CSV/TSV data to a string.

        Args:
            file_context: Standardized file input with binary data.
            **kwargs: Optional ``encodings`` list to prepend.

        Returns:
            CsvConvertedData with decoded text and metadata.

        Raises:
            ConversionError: If file data is missing or empty.
        """
        file_data: bytes = file_context.get("file_data", b"")
        file_ext: str = file_context.get("file_extension", "")

        if not file_data:
            raise ConversionError(
                "Empty file data — nothing to decode",
                stage="convert",
                handler="CsvConverter",
            )

        # Phase 1: BOM detection
        for bom_bytes, bom_encoding in _BOM_TABLE:
            if file_data.startswith(bom_bytes):
                try:
                    text = file_data.decode(bom_encoding)
                    self._logger.debug(
                        "BOM detected: %s, decoded %d bytes",
                        bom_encoding, len(file_data),
                    )
                    return CsvConvertedData(
                        text=text,
                        encoding=bom_encoding,
                        file_extension=file_ext,
                    )
                except (UnicodeDecodeError, LookupError):
                    self._logger.debug("BOM %s detected but decode failed", bom_encoding)

        # Phase 2: Try encoding candidates
        extra: List[str] = kwargs.get("encodings", [])
        all_encodings = _deduplicate(extra + self._encodings)

        for enc in all_encodings:
            try:
                text = file_data.decode(enc)
                self._logger.debug(
                    "Decoded %d bytes with %s encoding", len(file_data), enc
                )
                return CsvConvertedData(
                    text=text,
                    encoding=enc,
                    file_extension=file_ext,
                )
            except (UnicodeDecodeError, LookupError):
                continue

        # Phase 3: Fallback with replacement characters
        self._logger.warning(
            "All encodings failed for %s, falling back to utf-8 with errors='replace'",
            file_context.get("file_name", "unknown"),
        )
        text = file_data.decode("utf-8", errors="replace")
        return CsvConvertedData(
            text=text,
            encoding="utf-8",
            file_extension=file_ext,
        )

    def get_format_name(self) -> str:
        return "csv"

    def validate(self, file_context: FileContext) -> bool:
        """Validate that file data exists and is non-empty."""
        return len(file_context.get("file_data", b"")) > 0


def _deduplicate(encodings: List[str]) -> List[str]:
    """Remove duplicate encodings while preserving order."""
    seen: set[str] = set()
    result: List[str] = []
    for enc in encodings:
        key = enc.lower().replace("-", "").replace("_", "")
        if key not in seen:
            seen.add(key)
            result.append(enc)
    return result


__all__ = ["CsvConverter", "CsvConvertedData", "DEFAULT_ENCODINGS"]
