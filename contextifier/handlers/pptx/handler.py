# contextifier/handlers/pptx/handler.py
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

from typing import Any, FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.types import FileContext
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.pptx.converter import PptxConverter
from contextifier.handlers.pptx.preprocessor import PptxPreprocessor
from contextifier.handlers.pptx.metadata_extractor import PptxMetadataExtractor
from contextifier.handlers.pptx.content_extractor import PptxContentExtractor


class PPTXHandler(BaseHandler):
    """Handler for modern PowerPoint files (.pptx only)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"pptx"})

    @property
    def handler_name(self) -> str:
        return "PPTX Handler"

    def extract_text_fast(self, file_context: FileContext, **kwargs: Any) -> str:
        """Shape and table text from each slide; no images, charts or notes."""
        import io

        try:
            import pptx

            presentation = pptx.Presentation(
                io.BytesIO(file_context.get("file_data", b""))
            )
        except Exception as exc:
            self._logger.debug("Fast PPTX path unavailable (%s); using pipeline", exc)
            return super().extract_text_fast(file_context, **kwargs)

        lines = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    for paragraph in shape.text_frame.paragraphs:
                        text = "".join(run.text or "" for run in paragraph.runs).strip()
                        if text:
                            lines.append(text)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        for cell in row.cells:
                            text = (cell.text or "").strip()
                            if text:
                                lines.append(text)
        return "\n".join(lines)

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
            config=self._config,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["PPTXHandler"]
