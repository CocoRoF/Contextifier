# contextifier_new/handlers/xls/handler.py
"""
XLSHandler — Handler for legacy Excel XLS spreadsheets (.xls ONLY).

XLS is a BIFF (Binary Interchange File Format) binary format.
Requires xlrd for reading and olefile for OLE metadata.

Pipeline:
    Convert:  Raw bytes → xlrd Book (via xlrd.open_workbook)
    Preprocess: Record sheet info
    Metadata: OLE SummaryInformation + xlrd user_name
    Content:  Sheet data as Markdown/HTML tables (no chart/image extraction)
    Postprocess: Assemble with sheet tags and metadata block

Delegation:
    If the incoming .xls file is actually a ZIP (XLSX), delegate to 'xlsx'.
"""

from __future__ import annotations

from typing import Any, FrozenSet, Optional

from contextifier_new.handlers.base import BaseHandler
from contextifier_new.pipeline.converter import BaseConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor
from contextifier_new.types import ExtractionResult, FileContext

from contextifier_new.handlers.xls._constants import ZIP_MAGIC
from contextifier_new.handlers.xls.converter import XlsConverter
from contextifier_new.handlers.xls.preprocessor import XlsPreprocessor
from contextifier_new.handlers.xls.metadata_extractor import XlsMetadataExtractor
from contextifier_new.handlers.xls.content_extractor import XlsContentExtractor


class XLSHandler(BaseHandler):
    """Handler for legacy Excel files (.xls only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"xls"})

    @property
    def handler_name(self) -> str:
        return "XLS Handler"

    # ── Delegation ───────────────────────────────────────────────────────

    def _check_delegation(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> Optional[ExtractionResult]:
        """If the .xls file is actually XLSX (ZIP), delegate."""
        data: bytes = file_context.get("file_data", b"")
        if data and len(data) >= 4 and data[:4] == ZIP_MAGIC:
            self._logger.info("XLS file is actually XLSX (ZIP magic detected)")
            return self._delegate_to(
                "xlsx",
                file_context,
                include_metadata=kwargs.get("include_metadata", True),
                **{k: v for k, v in kwargs.items() if k != "include_metadata"},
            )
        return None

    # ── Pipeline stages ──────────────────────────────────────────────────

    def create_converter(self) -> BaseConverter:
        return XlsConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return XlsPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return XlsMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return XlsContentExtractor(
            tag_service=self._tag_service,
            table_service=self._table_service,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["XLSHandler"]
