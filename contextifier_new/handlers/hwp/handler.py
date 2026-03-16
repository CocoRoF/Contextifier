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

from contextifier_new.handlers.base import BaseHandler
from contextifier_new.types import ExtractionResult, FileContext
from contextifier_new.pipeline.converter import BaseConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier_new.handlers.hwp.converter import HwpConverter
from contextifier_new.handlers.hwp.preprocessor import HwpPreprocessor
from contextifier_new.handlers.hwp.metadata_extractor import HwpMetadataExtractor
from contextifier_new.handlers.hwp.content_extractor import HwpContentExtractor

_ZIP_MAGIC = b"PK\x03\x04"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


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
