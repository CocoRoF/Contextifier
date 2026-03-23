# contextifier/handlers/rtf/handler.py
"""
RTFHandler — Unified handler for Rich Text Format documents.

Pipeline:
    Stage 1 (Convert):     Validate RTF magic, detect encoding, wrap bytes
    Stage 2 (Preprocess):  Extract binary images, remove binary data, decode
    Stage 3 (Metadata):    Parse \\info group (title, author, dates)
    Stage 4 (Content):     Clean control codes, extract tables + inline text,
                           save images via ImageService
    Stage 5 (Postprocess): Assemble with metadata block

RTF is a binary-text hybrid format:
- Header contains font/color/style tables encoded as ASCII control words
- Body mixes control words with text content
- Images are embedded as hex data (\\pict) or raw binary (\\binN)
- Tables use \\trowd/\\cell/\\row with merge flags
- Metadata lives in the \\info group
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.rtf.converter import RtfConverter
from contextifier.handlers.rtf.preprocessor import RtfPreprocessor
from contextifier.handlers.rtf.metadata_extractor import RtfMetadataExtractor
from contextifier.handlers.rtf.content_extractor import RtfContentExtractor


class RTFHandler(BaseHandler):
    """
    Handler for RTF files (.rtf).

    Processes Rich Text Format documents through a 5-stage pipeline
    with full support for:
    - Korean / CJK text (\\ansicpg, hex escapes, multi-byte decoding)
    - Embedded images (\\pict hex + \\bin binary)
    - Tables with merge support (\\clmgf/\\clmrg/\\clvmgf/\\clvmrg)
    - Document metadata (\\info group)
    - Header/footer/footnote exclusion
    - striprtf fallback for degraded RTF content
    """

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"rtf"})

    @property
    def handler_name(self) -> str:
        return "RTF Handler"

    def create_converter(self) -> BaseConverter:
        return RtfConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return RtfPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return RtfMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return RtfContentExtractor(
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


__all__ = ["RTFHandler"]
