# contextifier_new/handlers/pdf/converter.py
"""
PdfConverter — Stage 1: raw bytes → fitz.Document

Shared by both pdf_default and pdf_plus modes.
Validates the ``%PDF`` magic header and opens the file
via PyMuPDF (``fitz``).
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, NamedTuple

import fitz  # PyMuPDF

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError
from contextifier.handlers.pdf._constants import PDF_MAGIC

logger = logging.getLogger(__name__)


class PdfConvertedData(NamedTuple):
    """Result of PDF conversion: a fitz.Document + the original bytes."""
    doc: Any          # fitz.Document (typed as Any to avoid import issues in tests)
    file_data: bytes


class PdfConverter(BaseConverter):
    """
    Open a PDF via PyMuPDF.

    Validates the ``%PDF`` header, then opens the binary data as a
    ``fitz.Document``.  The document **must** be closed via ``close()``.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> PdfConvertedData:
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data",
                stage="convert",
                handler="pdf",
            )

        if not file_data[:4].startswith(PDF_MAGIC):
            raise ConversionError(
                "Not a valid PDF (missing %PDF header)",
                stage="convert",
                handler="pdf",
            )

        try:
            doc = fitz.open(stream=BytesIO(file_data), filetype="pdf")
        except Exception as exc:
            raise ConversionError(
                f"Failed to open PDF: {exc}",
                stage="convert",
                handler="pdf",
            ) from exc

        return PdfConvertedData(doc=doc, file_data=file_data)

    def get_format_name(self) -> str:
        return "pdf"

    def validate(self, file_context: FileContext) -> bool:
        data = file_context.get("file_data", b"")
        return len(data) >= 4 and data[:4].startswith(PDF_MAGIC)

    def close(self, converted_object: Any) -> None:
        """Close the fitz.Document to release resources."""
        if isinstance(converted_object, PdfConvertedData):
            try:
                converted_object.doc.close()
            except Exception:
                pass
        elif hasattr(converted_object, "close"):
            try:
                converted_object.close()
            except Exception:
                pass


__all__ = ["PdfConverter", "PdfConvertedData"]
