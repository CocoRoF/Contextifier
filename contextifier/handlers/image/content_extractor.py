# contextifier/handlers/image/content_extractor.py
"""
ImageContentExtractor — Stage 4: save the image and return a tag

For standalone image files the "text extraction" is simply:
1. Save the image via ``image_service.save_and_tag()``
2. Return the image tag as the text content

If no *image_service* is available, a minimal placeholder tag is
returned so downstream processing always has something.

**OCR integration path**:
    OCR is an **external** concern — the caller (e.g. DocumentProcessor)
    is responsible for feeding OCR output back if needed.

    When ``ocr_processing=True`` is set on ``DocumentProcessor.extract_text()``:
    1. This handler outputs ``[Image:filename.jpg]`` tags
    2. ``DocumentProcessor`` post-processing invokes ``OCRProcessor``
    3. ``OCRProcessor`` finds ``[Image:...]`` tags → resolves file paths
    4. OCR engine converts images to text → tags are replaced

    No code changes are needed in this handler to support OCR.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import PreprocessedData

logger = logging.getLogger(__name__)


class ImageContentExtractor(BaseContentExtractor):
    """
    Content extractor for standalone image files.

    Saves the image via *image_service* and returns the resulting
    tag (e.g. ``[Image:filename.jpg]``) as the extracted text.
    """

    # ── extract_text (required) ───────────────────────────────────────────

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        image_data: bytes = (
            preprocessed.content if isinstance(preprocessed.content, bytes) else b""
        )

        if not image_data:
            return ""

        props = preprocessed.properties or {}
        ext = props.get("file_extension", "")
        fmt = props.get("detected_format", ext or "image")

        # Determine a custom name based on keyword args if provided
        custom_name: Optional[str] = kwargs.get("file_name")

        # Try saving via ImageService first
        if self._image_service is not None:
            tag = self._image_service.save_and_tag(
                image_data,
                custom_name=custom_name,
            )
            if tag:
                return tag

        # Fallback: return a simple placeholder tag
        label = custom_name or f"image.{fmt}" if fmt else "image"
        return f"[Image:{label}]"

    # ── extract_images (return the saved path) ────────────────────────────

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> list[str]:
        """Return list of saved image paths (already processed during extract_text)."""
        if self._image_service is not None:
            return list(self._image_service.get_processed_paths())
        return []

    # ── Format name ───────────────────────────────────────────────────────

    def get_format_name(self) -> str:
        return "image"


__all__ = ["ImageContentExtractor"]
