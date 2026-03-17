# contextifier_new/handlers/docx/handler.py
"""
DOCXHandler — Handler for Microsoft Word DOCX documents (.docx ONLY).

Pipeline:
    Convert:  Raw bytes → python-docx Document (ZIP + OOXML validation)
    Preprocess: Wrap Document, compute stats, pre-extract charts from ZIP
    Metadata: core_properties → DocumentMetadata (title, author, dates)
    Content:  Body traversal → paragraphs, tables (vMerge/gridSpan),
              images (relationship-based), charts (OOXML DrawingML),
              diagrams, page breaks
    Postprocess: Assemble with page tags and metadata block

Old issues resolved:
- Chart formatting duplicated — now uses ChartService
- Image processor created without standard config args — fixed
- Dual metadata approach eliminated
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier_new.handlers.base import BaseHandler
from contextifier_new.pipeline.converter import BaseConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier_new.handlers.docx.converter import DocxConverter
from contextifier_new.handlers.docx.preprocessor import DocxPreprocessor
from contextifier_new.handlers.docx.metadata_extractor import DocxMetadataExtractor
from contextifier_new.handlers.docx.content_extractor import DocxContentExtractor


class DOCXHandler(BaseHandler):
    """Handler for DOCX files (.docx)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"docx"})

    @property
    def handler_name(self) -> str:
        return "DOCX Handler"

    def create_converter(self) -> BaseConverter:
        return DocxConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return DocxPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return DocxMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return DocxContentExtractor(
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


__all__ = ["DOCXHandler"]
