# contextifier_new/handlers/image/preprocessor.py
"""
ImagePreprocessor — Stage 2: detect format, wrap in PreprocessedData

The preprocessor detects the image format from magic bytes and file
extension, then stores metadata in ``properties`` for downstream use.
"""

from __future__ import annotations

import logging
from typing import Any

from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.types import PreprocessedData
from contextifier_new.handlers.image.converter import ImageConvertedData
from contextifier_new.handlers.image._constants import detect_image_format

logger = logging.getLogger(__name__)


class ImagePreprocessor(BasePreprocessor):
    """Detect image format and wrap in *PreprocessedData*."""

    def preprocess(
        self,
        converted_data: Any,
        **kwargs: Any,
    ) -> PreprocessedData:
        if isinstance(converted_data, ImageConvertedData):
            image_data = converted_data.image_data
            detected_format = converted_data.detected_format
        else:
            # Fallback – accept raw bytes directly
            image_data = converted_data if isinstance(converted_data, bytes) else b""
            detected_format = detect_image_format(image_data) if image_data else None

        # Extension may be passed via kwargs (from handler) or inferred
        ext = kwargs.get("file_extension", "")
        # Use detected magic-based format if available; fall back to extension
        fmt = detected_format or ext or "unknown"

        return PreprocessedData(
            content=image_data,
            raw_content=image_data,
            encoding="binary",
            resources={},
            properties={
                "detected_format": fmt,
                "file_extension": ext,
                "file_size": len(image_data),
            },
        )

    def get_format_name(self) -> str:
        return "image"


__all__ = ["ImagePreprocessor"]
