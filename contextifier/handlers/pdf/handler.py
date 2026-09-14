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
from typing import Any, FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.types import FileContext
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
    PDF_VALID_MODES,
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

    def extract_text_fast(self, file_context: FileContext, **kwargs: Any) -> str:
        """Page text straight from the text layer.

        Skips table detection, layout and complexity analysis, image
        extraction, OCR and chart parsing — everything a keyword scan has no
        use for, and everything that makes PDF the slowest format here.
        """
        data = file_context.get("file_data", b"")
        try:
            import pymupdf

            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            self._logger.debug("Fast PDF path unavailable (%s); using pipeline", exc)
            return super().extract_text_fast(file_context, **kwargs)

        try:
            pages = []
            for index in range(doc.page_count):
                try:
                    text = doc[index].get_text("text") or ""
                except Exception as exc:
                    self._logger.debug("Page %d unreadable: %s", index + 1, exc)
                    continue
                if text.strip():
                    pages.append(text)
            return "\n\n".join(pages)
        finally:
            doc.close()

    def create_converter(self) -> BaseConverter:
        return PdfConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return PdfPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return PdfMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        mode = self._config.get_format_option(
            PDF_FORMAT_OPTION_KEY,
            PDF_MODE_OPTION,
            PDF_MODE_PLUS,
        )
        logger.debug("[PDFHandler] PDF mode = %s", mode)

        if mode not in PDF_VALID_MODES:
            from contextifier.errors import ConfigurationError

            raise ConfigurationError(
                f"Invalid PDF mode '{mode}'. Valid options: {sorted(PDF_VALID_MODES)}",
                context={"mode": mode},
            )

        if mode == PDF_MODE_DEFAULT:
            from contextifier.handlers.pdf_default import (
                PdfDefaultContentExtractor,
            )

            return PdfDefaultContentExtractor(
                image_service=self._image_service,
                tag_service=self._tag_service,
                table_service=self._table_service,
                config=self._config,
            )

        # default → plus mode
        from contextifier.handlers.pdf_plus import (
            PdfPlusContentExtractor,
        )

        return PdfPlusContentExtractor(
            image_service=self._image_service,
            tag_service=self._tag_service,
            table_service=self._table_service,
            config=self._config,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["PDFHandler"]
