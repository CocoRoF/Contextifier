# contextifier_new/handlers/ppt/handler.py
"""
PPTHandler — Handler for legacy PowerPoint PPT documents (.ppt ONLY).

PPT is an OLE2/CFBF binary format. Files with .ppt extension can be:

1. Genuine OLE2 PPT (binary PowerPoint format) — most common
2. Misnamed PPTX (ZIP/OOXML) saved with .ppt extension

This handler uses the DELEGATION pattern from DOCHandler to:
    .ppt file arrives
        ├── Starts with PK (ZIP magic) → delegate to PPTX handler
        └── OLE2 signature (D0CF11E0)  → process as genuine PPT

Pipeline (for genuine OLE2 PPT):
    Convert:  Raw bytes → olefile OLE2 object
    Preprocess: Extract PowerPoint Document stream, Pictures stream
    Metadata: OLE2 SummaryInformation → DocumentMetadata
    Content:  Binary stream parsing for text records (TextBytesAtom,
              TextCharsAtom), heuristic slide grouping, image extraction
    Postprocess: Assemble with slide tags and metadata block

Limitations (genuine PPT):
- Tables and charts are not extractable from binary PPT without LibreOffice
- Text extraction is heuristic (record-level parsing)
- Image extraction depends on Pictures stream availability
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

from contextifier_new.handlers.ppt._constants import ZIP_MAGIC, OLE2_MAGIC
from contextifier_new.handlers.ppt.converter import PptConverter
from contextifier_new.handlers.ppt.preprocessor import PptPreprocessor
from contextifier_new.handlers.ppt.metadata_extractor import PptMetadataExtractor
from contextifier_new.handlers.ppt.content_extractor import PptContentExtractor


class PPTHandler(BaseHandler):
    """
    Handler for legacy PowerPoint files (.ppt only).

    Uses format detection + delegation for polymorphic .ppt files.
    Only processes genuine OLE2 PPT files through its own pipeline;
    misnamed PPTX files are delegated to PPTXHandler.
    """

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"ppt"})

    @property
    def handler_name(self) -> str:
        return "PPT Handler"

    # ── Delegation: detect actual format ──────────────────────────────────

    def _check_delegation(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> Optional[ExtractionResult]:
        """
        Detect the actual format of a .ppt file and delegate if needed.

        Checks magic bytes:
        - ZIP magic (PK) → delegate to 'pptx' handler
        - OLE2 magic     → return None (process as genuine PPT)
        """
        data = file_context.get("file_data", b"")
        if not data or len(data) < 8:
            return None

        # Check for ZIP/OOXML (misnamed PPTX)
        if data[:4] == ZIP_MAGIC:
            self._logger.info("PPT file is actually PPTX (ZIP magic detected)")
            return self._delegate_to(
                "pptx", file_context,
                include_metadata=kwargs.get("include_metadata", True),
                **{k: v for k, v in kwargs.items() if k != "include_metadata"},
            )

        return None

    # ── Pipeline components ───────────────────────────────────────────────

    def create_converter(self) -> BaseConverter:
        return PptConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return PptPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return PptMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return PptContentExtractor(
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


__all__ = ["PPTHandler"]
