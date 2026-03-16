# contextifier_new/handlers/image/converter.py
"""
ImageConverter — Stage 1: raw bytes → validated image data

Validates that the binary data looks like a known image format,
then wraps it in ``ImageConvertedData`` for downstream stages.
Non-magic-validated formats (SVG, ICO, HEIC) are passed through
on trust based on file extension.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple, Optional

from contextifier_new.pipeline.converter import BaseConverter
from contextifier_new.types import FileContext
from contextifier_new.errors import ConversionError
from contextifier_new.handlers.image._constants import (
    MAGIC_VALIDATED_EXTENSIONS,
    detect_image_format,
)

logger = logging.getLogger(__name__)


class ImageConvertedData(NamedTuple):
    """Result of the image conversion stage."""
    image_data: bytes         # Raw image bytes (unchanged)
    detected_format: Optional[str]   # Detected format from magic bytes (or None)


class ImageConverter(BaseConverter):
    """
    Validate and wrap standalone image file data.

    For formats with known magic bytes (JPEG, PNG, GIF, BMP, WebP, TIFF),
    the binary data is validated.  For others (SVG, ICO, HEIC, etc.) the
    data is accepted on trust.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> ImageConvertedData:
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data", stage="convert", handler="image",
            )

        detected = detect_image_format(file_data)
        ext = file_context.get("file_extension", "").lower()

        # If the extension is one we can validate and magic didn't match,
        # that's a warning but we still proceed (the file may be corrupt
        # but we let the content stage decide).
        if ext in MAGIC_VALIDATED_EXTENSIONS and detected is None:
            logger.warning(
                "Image magic bytes not recognised for extension '%s'", ext,
            )

        return ImageConvertedData(image_data=file_data, detected_format=detected)

    def get_format_name(self) -> str:
        return "image"

    def validate(self, file_context: FileContext) -> bool:
        data = file_context.get("file_data", b"")
        return len(data) > 0

    def close(self, converted_object: Any) -> None:
        # Images are raw bytes — nothing to close.
        pass


__all__ = ["ImageConverter", "ImageConvertedData"]
