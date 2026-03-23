# contextifier/handlers/hwpx/converter.py
"""
HwpxConverter — Stage 1: raw bytes → ZipFile object

Validates that the binary data starts with ZIP magic (``PK\\x03\\x04``),
then opens it as a ``zipfile.ZipFile``.  The result is a
``HwpxConvertedData`` named-tuple consumed by later stages.
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any, NamedTuple

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError
from contextifier.handlers.hwpx._constants import ZIP_MAGIC

logger = logging.getLogger(__name__)


class HwpxConvertedData(NamedTuple):
    """Result of the HWPX conversion stage."""
    zf: zipfile.ZipFile      # Opened ZIP archive
    file_data: bytes          # Original raw bytes


class HwpxConverter(BaseConverter):
    """
    Open HWPX ZIP archives.

    The ZipFile object remains open after ``convert()`` returns; the
    handler framework calls ``close()`` when the pipeline finishes.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> HwpxConvertedData:
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data", stage="convert", handler="hwpx",
            )

        try:
            zf = zipfile.ZipFile(io.BytesIO(file_data), "r")
        except zipfile.BadZipFile as exc:
            raise ConversionError(
                f"Invalid HWPX ZIP archive: {exc}",
                stage="convert",
                handler="hwpx",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise ConversionError(
                f"Failed to open HWPX archive: {exc}",
                stage="convert",
                handler="hwpx",
                cause=exc,
            ) from exc

        return HwpxConvertedData(zf=zf, file_data=file_data)

    def get_format_name(self) -> str:
        return "hwpx"

    def validate(self, file_context: FileContext) -> bool:
        data = file_context.get("file_data", b"")
        if len(data) < 4:
            return False
        return data[:4] == ZIP_MAGIC

    def close(self, converted_object: Any) -> None:
        if isinstance(converted_object, HwpxConvertedData):
            try:
                converted_object.zf.close()
            except Exception:
                pass
        elif hasattr(converted_object, "close"):
            try:
                converted_object.close()
            except Exception:
                pass


__all__ = ["HwpxConverter", "HwpxConvertedData"]
