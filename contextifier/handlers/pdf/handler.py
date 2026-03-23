# contextifier/handlers/pdf/handler.py
"""
PDFHandler — Unified handler for PDF documents.

This is the **only** handler registered for the ``.pdf`` extension.
It reads the ``mode`` option from ``config.get_format_option("pdf", "mode")``
and delegates to one of two content-extractor implementations:

    ``"plus"``  (default) → ``PdfPlusContentExtractor``
    ``"default"``         → ``PdfDefaultContentExtractor``

Converter, preprocessor, metadata extractor, and postprocessor are
shared between both modes.

Usage::

    # default (plus) mode
    handler = PDFHandler(config)
    result  = handler.process(file_context)

    # explicit default mode
    handler = PDFHandler(config.with_format_option("pdf", mode="default"))
    result  = handler.process(file_context)
"""

from __future__ import annotations

import logging
from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.pdf._constants import (
    PDF_FORMAT_OPTION_KEY,
    PDF_MODE_DEFAULT,
    PDF_MODE_OPTION,
    PDF_MODE_PLUS,
)
from contextifier.handlers.pdf.converter import PdfConverter
from contextifier.handlers.pdf.preprocessor import PdfPreprocessor
from contextifier.handlers.pdf.metadata_extractor import PdfMetadataExtractor

logger = logging.getLogger(__name__)


class PDFHandler(BaseHandler):
    """Handler for PDF files (.pdf)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"pdf"})

    @property
    def handler_name(self) -> str:
        return "PDF Handler"

    # ── pipeline factories ───────────────────────────────────────────────

    def create_converter(self) -> BaseConverter:
        return PdfConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return PdfPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return PdfMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        mode = self._config.get_format_option(
            PDF_FORMAT_OPTION_KEY, PDF_MODE_OPTION, PDF_MODE_PLUS,
        )
        logger.debug("[PDFHandler] PDF mode = %s", mode)

        if mode == PDF_MODE_DEFAULT:
            from contextifier.handlers.pdf_default import (
                PdfDefaultContentExtractor,
            )
            return PdfDefaultContentExtractor(
                image_service=self._image_service,
                tag_service=self._tag_service,
                table_service=self._table_service,
            )

        # default → plus mode
        from contextifier.handlers.pdf_plus import (
            PdfPlusContentExtractor,
        )
        return PdfPlusContentExtractor(
            image_service=self._image_service,
            tag_service=self._tag_service,
            table_service=self._table_service,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )

__all__ = ["PDFHandler"]
