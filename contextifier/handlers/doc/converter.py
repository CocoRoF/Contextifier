# contextifier/handlers/doc/converter.py
"""
DocConverter — Stage 1: Raw bytes → OLE2 object

Validates that the input is a genuine OLE2 Compound Binary file,
opens it with ``olefile.OleFileIO`` and wraps the result in a
``DocConvertedData`` named-tuple for the downstream pipeline.

IMPORTANT: .doc files that are actually RTF, HTML, or misnamed DOCX
are intercepted in ``DOCHandler._check_delegation()`` BEFORE this
converter runs.  By the time we reach here the data is guaranteed
(within reason) to be genuine OLE2.
"""

from __future__ import annotations

import io
import logging
from typing import Any, NamedTuple

import olefile

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError

from contextifier.handlers.doc._constants import OLE2_MAGIC

logger = logging.getLogger(__name__)


class DocConvertedData(NamedTuple):
    """Result of the DOC conversion stage."""
    ole: olefile.OleFileIO          # Opened OLE2 compound file
    file_extension: str             # Original extension (always "doc")


class DocConverter(BaseConverter):
    """
    Converter for genuine OLE2 DOC files.

    Opens the binary data as an ``olefile.OleFileIO`` object.
    The OLE object remains open after ``convert()`` returns; it will
    be closed by ``close()`` when the handler pipeline is done.
    """

    # ── BaseConverter abstract methods ────────────────────────────────────

    def convert(self, file_context: FileContext, **kwargs: Any) -> DocConvertedData:
        """
        Open OLE2 compound file.

        Args:
            file_context: Standardized file input.

        Returns:
            DocConvertedData containing the opened OLE2 object.

        Raises:
            ConversionError: If the data is not a valid OLE2 file.
        """
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data",
                stage="convert",
                handler="doc",
            )

        try:
            ole = olefile.OleFileIO(io.BytesIO(file_data))
        except Exception as exc:
            raise ConversionError(
                f"Failed to open OLE2 compound file: {exc}",
                stage="convert",
                handler="doc",
                cause=exc,
            ) from exc

        ext = file_context.get("file_extension", "doc")
        return DocConvertedData(ole=ole, file_extension=ext)

    def get_format_name(self) -> str:
        return "doc"

    def validate(self, file_context: FileContext) -> bool:
        """
        Check whether the binary data starts with the OLE2 signature.

        This is called before ``_check_delegation()`` as a fast pre-check.
        Files that fail validation will be caught by delegation logic anyway.
        """
        data: bytes = file_context.get("file_data", b"")
        if len(data) < len(OLE2_MAGIC):
            return False
        return data[: len(OLE2_MAGIC)] == OLE2_MAGIC

    def close(self, converted_object: Any) -> None:
        """
        Close the OLE2 object held inside *converted_object*.

        Accepts either a ``DocConvertedData`` namedtuple or a raw
        ``olefile.OleFileIO`` instance.
        """
        ole = None
        if isinstance(converted_object, DocConvertedData):
            ole = converted_object.ole
        elif isinstance(converted_object, olefile.OleFileIO):
            ole = converted_object
        elif hasattr(converted_object, "close"):
            try:
                converted_object.close()
            except Exception:
                pass
            return

        if ole is not None:
            try:
                ole.close()
            except Exception:
                pass


__all__ = ["DocConverter", "DocConvertedData"]
