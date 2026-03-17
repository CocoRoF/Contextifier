# contextifier_new/handlers/tsv/handler.py
"""
TSVHandler — Handler for TSV (Tab-Separated Values) files (.tsv ONLY).

TSV uses a fixed tab delimiter. All pipeline components are shared
with CSVHandler — the only difference is the forced ``\\t`` delimiter
passed to CsvPreprocessor, which skips auto-detection.

Pipeline (same 5-stage model as CSV):
    Stage 1 (Convert):     Raw bytes → decoded text (BOM + encoding detection)
    Stage 2 (Preprocess):  Fixed tab delimiter, parse, header detection
    Stage 3 (Metadata):    Row/col count, columns, encoding, delimiter info
    Stage 4 (Content):     Parsed rows → TableData → formatted table string
    Stage 5 (Postprocess): Metadata block + tag assembly

Why a separate handler instead of a flag on CSVHandler:
- One-extension-per-handler rule for proper registry behaviour
- TSV data rarely has the quoting ambiguity issues that CSV has
- Different default encoding assumptions possible via format_options["tsv"]
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

# Reuse CSV pipeline components — TSV differs only in delimiter
from contextifier.handlers.csv.converter import CsvConverter
from contextifier.handlers.csv.preprocessor import CsvPreprocessor
from contextifier.handlers.csv.metadata_extractor import CsvMetadataExtractor
from contextifier.handlers.csv.content_extractor import CsvContentExtractor


class TSVHandler(BaseHandler):
    """
    Handler for TSV files (.tsv only).

    Uses the same pipeline as CSVHandler but forces tab as the
    delimiter, bypassing auto-detection.

    Encoding customization via ``config.format_options``:

        config = ProcessingConfig(
            format_options={"tsv": {"encodings": ["shift_jis", "utf-8"]}}
        )
    """

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"tsv"})

    @property
    def handler_name(self) -> str:
        return "TSV Handler"

    def create_converter(self) -> BaseConverter:
        """BOM-aware converter with configurable encoding list."""
        tsv_opts = self._config.format_options.get("tsv", {})
        encodings = tsv_opts.get("encodings", None)
        return CsvConverter(encodings=encodings)

    def create_preprocessor(self) -> BasePreprocessor:
        """
        Preprocessor with forced tab delimiter.

        Skips csv.Sniffer detection — always uses ``\\t``.
        """
        return CsvPreprocessor(default_delimiter="\t")

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        """Extracts TSV structural info into DocumentMetadata.custom."""
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


__all__ = ["TSVHandler"]
