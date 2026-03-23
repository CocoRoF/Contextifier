"""
PptxConverter — Stage 1: Raw bytes → python-pptx Presentation.

Validates that binary data is a valid PPTX (ZIP with OOXML content types),
then creates a ``pptx.Presentation`` object.
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError

from contextifier.handlers.pptx._constants import ZIP_MAGIC

logger = logging.getLogger(__name__)

# Expected entry in a valid PPTX ZIP
_CONTENT_TYPES = "[Content_Types].xml"


class PptxConverter(BaseConverter):
    """
    Converter for PPTX files.

    Validates ZIP magic bytes and Content_Types.xml presence,
    then opens a ``pptx.Presentation`` from the byte stream.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> Any:
        """
        Convert raw PPTX bytes to a ``pptx.Presentation``.

        Returns:
            ``pptx.Presentation`` object.

        Raises:
            ConversionError: If the data is not a valid PPTX file.
        """
        from pptx import Presentation

        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data",
                stage="convert",
                handler="pptx",
            )

        if not self.validate(file_context):
            raise ConversionError(
                "Invalid PPTX file (failed ZIP/OOXML validation)",
                stage="convert",
                handler="pptx",
            )

        try:
            stream = io.BytesIO(file_data)
            return Presentation(stream)
        except Exception as exc:
            raise ConversionError(
                f"Failed to open PPTX: {exc}",
                stage="convert",
                handler="pptx",
            ) from exc

    def get_format_name(self) -> str:
        return "pptx"

    def validate(self, file_context: FileContext) -> bool:
        """
        Validate that the data is a valid PPTX.

        Checks:
        1. ZIP magic bytes (``PK\\x03\\x04``)
        2. ``[Content_Types].xml`` in the ZIP archive
        """
        file_data: bytes = file_context.get("file_data", b"")
        if len(file_data) < 4:
            return False

        # Check ZIP magic
        if file_data[:4] != ZIP_MAGIC:
            return False

        # Verify Content_Types.xml exists
        try:
            with zipfile.ZipFile(io.BytesIO(file_data), "r") as zf:
                if _CONTENT_TYPES not in zf.namelist():
                    return False
        except zipfile.BadZipFile:
            return False

        return True

    def close(self, converted_object: Any) -> None:
        """Release resources (python-pptx Presentation has no explicit close)."""
        pass


__all__ = ["PptxConverter"]
