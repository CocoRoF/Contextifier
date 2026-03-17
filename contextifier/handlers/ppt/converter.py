"""
PptConverter — Stage 1: Raw bytes → OLE2 object.

Validates that the input is a genuine OLE2 Compound Binary file
(after delegation has intercepted any misnamed PPTX/RTF files),
opens it with ``olefile.OleFileIO``, and wraps the result in a
``PptConvertedData`` named-tuple.

IMPORTANT: .ppt files that are actually PPTX (ZIP) are intercepted
in ``PPTHandler._check_delegation()`` BEFORE this converter runs.
"""

from __future__ import annotations

import io
import logging
from typing import Any, NamedTuple

import olefile

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError

from contextifier.handlers.ppt._constants import OLE2_MAGIC

logger = logging.getLogger(__name__)


class PptConvertedData(NamedTuple):
    """Result of the PPT conversion stage."""
    ole: olefile.OleFileIO          # Opened OLE2 compound file
    file_extension: str             # Original extension (always "ppt")


class PptConverter(BaseConverter):
    """
    Converter for genuine OLE2 PPT files.

    Opens the binary data as an ``olefile.OleFileIO`` object.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> PptConvertedData:
        """
        Open OLE2 compound file.

        Returns:
            PptConvertedData containing the opened OLE2 object.

        Raises:
            ConversionError: If the data is not a valid OLE2 file.
        """
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data",
                stage="convert",
                handler="ppt",
            )

        try:
            ole = olefile.OleFileIO(io.BytesIO(file_data))
        except Exception as exc:
            raise ConversionError(
                f"Failed to open OLE2 compound file: {exc}",
                stage="convert",
                handler="ppt",
                cause=exc,
            ) from exc

        ext = file_context.get("file_extension", "ppt")
        return PptConvertedData(ole=ole, file_extension=ext)

    def get_format_name(self) -> str:
        return "ppt"

    def validate(self, file_context: FileContext) -> bool:
        """Check whether the binary data starts with the OLE2 signature."""
        data: bytes = file_context.get("file_data", b"")
        if len(data) < len(OLE2_MAGIC):
            return False
        return data[: len(OLE2_MAGIC)] == OLE2_MAGIC

    def close(self, converted_object: Any) -> None:
        """Close the OLE2 object."""
        ole = None
        if isinstance(converted_object, PptConvertedData):
            ole = converted_object.ole
        elif isinstance(converted_object, olefile.OleFileIO):
            ole = converted_object

        if ole is not None:
            try:
                ole.close()
            except Exception:
                pass


__all__ = ["PptConverter", "PptConvertedData"]
