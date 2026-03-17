# contextifier_new/handlers/image/handler.py
"""
ImageFileHandler — Unified handler for standalone image files.

Pipeline:
    Convert:  Raw bytes → validated magic-byte image data
    Preprocess: Detect format, wrap in PreprocessedData
    Metadata: Minimal metadata (format, file size, page_count=1)
    Content:  Save image via ImageService, return image tag
    Postprocess: Default postprocessor (metadata header etc.)
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.image.converter import ImageConverter
from contextifier.handlers.image.preprocessor import ImagePreprocessor
from contextifier.handlers.image.metadata_extractor import ImageMetadataExtractor
from contextifier.handlers.image.content_extractor import ImageContentExtractor
from contextifier.handlers.image._constants import IMAGE_EXTENSIONS


class ImageFileHandler(BaseHandler):
    """Handler for standalone image files."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return IMAGE_EXTENSIONS

    @property
    def handler_name(self) -> str:
        return "Image File Handler"

    def create_converter(self) -> BaseConverter:
        return ImageConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return ImagePreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return ImageMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return ImageContentExtractor(
            image_service=self._image_service,
            tag_service=self._tag_service,
            chart_service=self._chart_service,
            table_service=self._table_service,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["ImageFileHandler"]
