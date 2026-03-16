# contextifier_new/handlers/csv/metadata_extractor.py
"""
CsvMetadataExtractor — Stage 3: CSV structure metadata

Takes CsvParsedData (from preprocessed.content) and produces
a DocumentMetadata with CSV-specific structural information
stored in the ``custom`` dict.

Custom metadata fields:
- encoding: Detected file encoding
- delimiter: Detected/forced delimiter name
- row_count: Number of data rows
- col_count: Number of columns
- has_header: Whether a header row was detected
- columns: Column names (if header exists, max 10 shown)

Unlike document formats (PDF, DOCX) that have embedded metadata
properties, CSV files only have structural information.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.types import DocumentMetadata
from contextifier_new.handlers.csv.preprocessor import CsvParsedData


# Delimiter display names (human-readable)
_DELIMITER_NAMES: Dict[str, str] = {
    ",":  "Comma (,)",
    "\t": "Tab (\\t)",
    ";":  "Semicolon (;)",
    "|":  "Pipe (|)",
}

# Maximum number of column names to include in metadata
_MAX_COLUMNS_SHOWN: int = 10


class CsvMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for CSV and TSV files.

    Extracts structural information (row/column counts, encoding,
    delimiter, header info) into DocumentMetadata.custom.
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract CSV structural metadata.

        Args:
            source: CsvParsedData from PreprocessedData.content.
                    Also accepts None (returns empty metadata).

        Returns:
            DocumentMetadata with custom fields populated.
        """
        if not isinstance(source, CsvParsedData):
            self._logger.warning(
                "CsvMetadataExtractor: expected CsvParsedData, got %s",
                type(source).__name__,
            )
            return DocumentMetadata()

        custom: Dict[str, Any] = {}

        # Encoding
        custom["encoding"] = source.encoding

        # Delimiter (human-readable)
        custom["delimiter"] = _DELIMITER_NAMES.get(
            source.delimiter, repr(source.delimiter)
        )

        # Dimensions
        custom["row_count"] = source.row_count
        custom["col_count"] = source.col_count

        # Header
        custom["has_header"] = "Yes" if source.has_header else "No"

        # Column names (if header exists)
        if source.has_header and source.rows:
            headers = [h.strip() for h in source.rows[0] if h.strip()]
            if headers:
                shown = headers[:_MAX_COLUMNS_SHOWN]
                col_text = ", ".join(shown)
                remaining = len(headers) - _MAX_COLUMNS_SHOWN
                if remaining > 0:
                    col_text += f" (+{remaining} more)"
                custom["columns"] = col_text

        return DocumentMetadata(custom=custom)

    def get_format_name(self) -> str:
        return "csv"


__all__ = ["CsvMetadataExtractor"]
