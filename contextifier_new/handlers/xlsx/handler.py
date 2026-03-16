# contextifier_new/handlers/xlsx/handler.py
"""
XLSXHandler — Handler for modern Excel XLSX spreadsheets (.xlsx ONLY).

XLSX is an OOXML (Office Open XML) ZIP-based format parsed with openpyxl.
This is fundamentally different from legacy .xls (BIFF binary) which
requires xlrd or LibreOffice conversion.

Pipeline:
    Convert:  Raw bytes → openpyxl Workbook (data_only=True)
    Preprocess: Wrap Workbook, pre-extract charts/images/textboxes from ZIP
    Metadata: OOXML core properties → DocumentMetadata
    Content:  Per-sheet layout detection → table conversion (MD/HTML),
              chart extraction, image extraction, textbox extraction
    Postprocess: Assemble with sheet tags and metadata block
"""

from __future__ import annotations

from typing import Any, FrozenSet

from contextifier_new.handlers.base import BaseHandler
from contextifier_new.pipeline.converter import BaseConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier_new.handlers.xlsx.converter import XlsxConverter
from contextifier_new.handlers.xlsx.preprocessor import XlsxPreprocessor
from contextifier_new.handlers.xlsx.metadata_extractor import XlsxMetadataExtractor
from contextifier_new.handlers.xlsx.content_extractor import XlsxContentExtractor


class XLSXHandler(BaseHandler):
    """Handler for modern Excel files (.xlsx only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"xlsx"})

    @property
    def handler_name(self) -> str:
        return "XLSX Handler"

    def create_converter(self) -> BaseConverter:
        return XlsxConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return XlsxPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return XlsxMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return XlsxContentExtractor(
            image_service=self._image_service,
            tag_service=self._tag_service,
            chart_service=self._chart_service,
            table_service=self._table_service,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["XLSXHandler"]
