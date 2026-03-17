# contextifier_new/handlers/hwpx/handler.py
"""
HWPXHandler — Unified handler for Hangul HWPX (ZIP-based XML) documents.

Pipeline:
    Convert:  Raw bytes → ZIP archive (zipfile.ZipFile)
    Preprocess: Parse OPF manifest, discover sections, build bin_item_map
    Metadata: Author, title, creation date from header.xml / version.xml
    Content:  Text from section XML, tables, embedded images, charts
    Postprocess: Assemble with page tags and metadata block
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.hwpx.converter import HwpxConverter
from contextifier.handlers.hwpx.preprocessor import HwpxPreprocessor
from contextifier.handlers.hwpx.metadata_extractor import HwpxMetadataExtractor
from contextifier.handlers.hwpx.content_extractor import HwpxContentExtractor


class HWPXHandler(BaseHandler):
    """Handler for HWPX files (.hwpx)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"hwpx"})

    @property
    def handler_name(self) -> str:
        return "HWPX Handler"

    def create_converter(self) -> BaseConverter:
        return HwpxConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return HwpxPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return HwpxMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return HwpxContentExtractor(
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


__all__ = ["HWPXHandler"]
