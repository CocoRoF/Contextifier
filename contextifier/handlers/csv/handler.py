# contextifier/handlers/csv/handler.py
"""
CSVHandler — Handler for CSV (Comma-Separated Values) files (.csv ONLY).

CSV uses comma as the default delimiter. TSV (tab-separated) is handled
by a separate TSVHandler that reuses the same pipeline components with
a forced tab delimiter.

Pipeline:
    Stage 1 (Convert):     Raw bytes → decoded text (BOM + encoding detection)
    Stage 2 (Preprocess):  Delimiter detection, CSV parsing, header detection
    Stage 3 (Metadata):    Row/col count, columns, encoding, delimiter info
    Stage 4 (Content):     Parsed rows → TableData → formatted table string
    Stage 5 (Postprocess): Metadata block + tag assembly

Old issues resolved:
- Extra parameters on extract_text() — now uses config.format_options
- CSV and TSV no longer share a handler (different delimiter logic)
- Table formatting was inline — now delegated to TableService
- Metadata was custom-formatted — now uses DocumentMetadata.custom + MetadataService
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.csv.converter import CsvConverter
from contextifier.handlers.csv.preprocessor import CsvPreprocessor
from contextifier.handlers.csv.metadata_extractor import CsvMetadataExtractor
from contextifier.handlers.csv.content_extractor import CsvContentExtractor


class CSVHandler(BaseHandler):
    """
    Handler for CSV files (.csv only).

    Delimiter is auto-detected (csv.Sniffer + heuristic scoring).
    Encoding customization via ``config.format_options``:

        config = ProcessingConfig(
            format_options={"csv": {"encodings": ["shift_jis", "utf-8"]}}
        )
    """

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"csv"})

    @property
    def handler_name(self) -> str:
        return "CSV Handler"

    def create_converter(self) -> BaseConverter:
        """BOM-aware converter with configurable encoding list."""
        csv_opts = self._config.format_options.get("csv", {})
        encodings = csv_opts.get("encodings", None)
        return CsvConverter(encodings=encodings)

    def create_preprocessor(self) -> BasePreprocessor:
        """
        Preprocessor with auto delimiter detection.

        Detects comma, tab, semicolon, or pipe — returns CsvParsedData
        with parsed rows, header flag, and structural metadata.
        """
        return CsvPreprocessor(default_delimiter=None)

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        """Extracts CSV structural info into DocumentMetadata.custom."""
        return CsvMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        """Formats parsed rows as a table via TableService."""
        return CsvContentExtractor(table_service=self._table_service)

    def create_postprocessor(self) -> BasePostprocessor:
        """Standard postprocessor for metadata block assembly."""
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["CSVHandler"]
