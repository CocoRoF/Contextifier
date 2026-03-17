# contextifier_new/handlers/hwp/converter.py
"""
HwpConverter — Stage 1: raw bytes → OLE2 object

Validates that the binary data starts with the OLE2 magic signature,
then opens it with ``olefile.OleFileIO``.  The result is a
``HwpConvertedData`` named-tuple consumed by later stages.

Files that start with ZIP magic (PK) should be intercepted by
``HWPHandler._check_delegation()`` BEFORE this converter runs.
"""

from __future__ import annotations

import io
import logging
from typing import Any, NamedTuple

import olefile

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError
from contextifier.handlers.hwp._constants import OLE2_MAGIC

logger = logging.getLogger(__name__)


class HwpConvertedData(NamedTuple):
    """Result of the HWP conversion stage."""
    ole: olefile.OleFileIO   # Opened OLE compound file
    file_data: bytes         # Original raw bytes (for OLE metadata parsing)


class HwpConverter(BaseConverter):
    """
    Open HWP5 OLE compound files.

    The OLE object remains open after ``convert()`` returns; the
    handler framework calls ``close()`` when the pipeline finishes.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> HwpConvertedData:
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data", stage="convert", handler="hwp",
            )

        try:
            ole = olefile.OleFileIO(io.BytesIO(file_data))
        except Exception as exc:
            raise ConversionError(
                f"Failed to open HWP OLE compound file: {exc}",
                stage="convert",
                handler="hwp",
                cause=exc,
            ) from exc

        return HwpConvertedData(ole=ole, file_data=file_data)

    def get_format_name(self) -> str:
        return "hwp"

    def validate(self, file_context: FileContext) -> bool:
        data = file_context.get("file_data", b"")
        if len(data) < 8:
            return False
        return data[:8] == OLE2_MAGIC

    def close(self, converted_object: Any) -> None:
        if isinstance(converted_object, HwpConvertedData):
            try:
                converted_object.ole.close()
            except Exception:
                pass
        elif hasattr(converted_object, "close"):
            try:
                converted_object.close()
            except Exception:
                pass


__all__ = ["HwpConverter", "HwpConvertedData"]
