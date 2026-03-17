# contextifier_new/handlers/pptx/handler.py
"""
PPTXHandler — Handler for modern PowerPoint PPTX documents (.pptx ONLY).

PPTX is an OOXML (Office Open XML) ZIP-based format that can be parsed
directly with python-pptx. This is fundamentally different from legacy
.ppt (OLE2/CFBF binary) which requires LibreOffice conversion.

Pipeline:
    Convert:  Raw bytes → python-pptx Presentation (ZIP + OOXML validation)
    Preprocess: Wrap Presentation, compute slide stats, pre-extract charts
    Metadata: core_properties → DocumentMetadata (title, author, dates, slide count)
    Content:  Slide iteration, shape dispatch (table/picture/chart/text/group),
              position-based sorting, slide notes
    Postprocess: Assemble with slide tags and metadata block
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier_new.handlers.base import BaseHandler
from contextifier_new.pipeline.converter import BaseConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier_new.handlers.pptx.converter import PptxConverter
from contextifier_new.handlers.pptx.preprocessor import PptxPreprocessor
from contextifier_new.handlers.pptx.metadata_extractor import PptxMetadataExtractor
from contextifier_new.handlers.pptx.content_extractor import PptxContentExtractor


class PPTXHandler(BaseHandler):
    """Handler for modern PowerPoint files (.pptx only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"pptx"})

    @property
    def handler_name(self) -> str:
        return "PPTX Handler"

    def create_converter(self) -> BaseConverter:
        return PptxConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return PptxPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return PptxMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return PptxContentExtractor(
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


__all__ = ["PPTXHandler"]
