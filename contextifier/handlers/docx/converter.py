"""
DocxConverter — Stage 1: Raw bytes → python-docx Document.

Validates the input as a valid DOCX file (ZIP with [Content_Types].xml),
then opens it as a ``python_docx.Document`` object via a ``BytesIO``
stream.

Validation checks:
1. ZIP magic bytes (PK\\x03\\x04)
2. Presence of ``[Content_Types].xml`` inside the ZIP archive
3. Successful creation of a python-docx ``Document`` object
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any

from docx import Document

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError

from contextifier.handlers.docx._constants import ZIP_MAGIC, CONTENT_TYPES_PATH

logger = logging.getLogger(__name__)


class DocxConverter(BaseConverter):
    """
    Converter for DOCX (Office Open XML) files.

    Opens binary data as a ``python_docx.Document`` via ``BytesIO``.
    The Document object stays open after ``convert()`` returns and
    is closed by ``close()`` when the handler pipeline is done.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> Document:
        """
        Open DOCX binary data as a python-docx ``Document``.

        Args:
            file_context: Standardized file input.

        Returns:
            ``python_docx.Document`` object.

        Raises:
            ConversionError: If the file is not a valid DOCX.
        """
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data",
                stage="convert",
                handler="docx",
            )

        stream = io.BytesIO(file_data)

        try:
            doc = Document(stream)
        except Exception as exc:
            raise ConversionError(
                f"Failed to open DOCX document: {exc}",
                stage="convert",
                handler="docx",
                cause=exc,
            ) from exc

        return doc

    def get_format_name(self) -> str:
        return "docx"

    def validate(self, file_context: FileContext) -> bool:
        """
        Verify that the binary data looks like a valid DOCX:
        1. Starts with ZIP magic bytes
        2. Contains ``[Content_Types].xml``
        """
        data: bytes = file_context.get("file_data", b"")
        if len(data) < 4:
            return False

        # ZIP magic check
        if data[:4] != ZIP_MAGIC:
            return False

        # Content-Types check
        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                return CONTENT_TYPES_PATH in zf.namelist()
        except (zipfile.BadZipFile, Exception):
            return False

    def close(self, converted_object: Any) -> None:
        """
        Close the Document object (no-op for python-docx, but we
        call super() for any generic close() support).
        """
        # python-docx Document objects don't have an explicit close()
        # but we attempt it anyway for completeness
        if hasattr(converted_object, "close") and callable(converted_object.close):
            try:
                converted_object.close()
            except Exception:
                pass


__all__ = ["DocxConverter"]
