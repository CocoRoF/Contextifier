# contextifier_new/handlers/hwp/handler.py
"""
HWPHandler — Handler for Hangul Word Processor 5.0 (HWP) documents.

Pipeline:
    Convert:  Raw bytes → OLE2 compound file (olefile)
    Preprocess: Parse DocInfo for BinData mapping, detect compression
    Metadata: OLE metadata + HwpSummaryInformation stream
    Content:  Record-tree traversal → text, tables (HTML), images
    Postprocess: Assemble with page tags and metadata block

Delegation:
    ZIP-magic files (.hwp that is actually HWPX) → delegate to 'hwpx'
    HWP 3.0 format → reject with informational message
"""

from __future__ import annotations

from typing import Any, FrozenSet, Optional

from contextifier.handlers.base import BaseHandler
from contextifier.types import ExtractionResult, FileContext
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.errors import ConversionError
from contextifier.handlers.hwp.converter import HwpConverter
from contextifier.handlers.hwp.preprocessor import HwpPreprocessor
from contextifier.handlers.hwp.metadata_extractor import HwpMetadataExtractor
from contextifier.handlers.hwp.content_extractor import HwpContentExtractor

_ZIP_MAGIC = b"PK\x03\x04"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_HWP30_MAGIC = b"HWP Document File"


class HWPHandler(BaseHandler):
    """Handler for HWP files (.hwp)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"hwp"})

    @property
    def handler_name(self) -> str:
        return "HWP Handler"

    # ── Delegation ────────────────────────────────────────────────────

    def _check_delegation(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> Optional[ExtractionResult]:
        data: bytes = file_context.get("file_data", b"")
        if not data:
            return None

        # ZIP magic → HWPX
        if data[:4] == _ZIP_MAGIC:
            return self._delegate_to("hwpx", file_context, **kwargs)

        # HWP 3.0 format → not supported
        if data[:17] == _HWP30_MAGIC:
            raise ConversionError(
                "HWP 3.0 format is not supported. "
                "Please convert to HWP 5.0 (.hwp) or HWPX (.hwpx) format.",
                stage="convert",
                handler=self.handler_name,
            )

        return None

    # ── Pipeline factory methods ──────────────────────────────────────

    def create_converter(self) -> BaseConverter:
        return HwpConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return HwpPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return HwpMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return HwpContentExtractor(
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


__all__ = ["HWPHandler"]
