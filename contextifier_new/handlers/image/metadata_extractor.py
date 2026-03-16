# contextifier_new/handlers/image/metadata_extractor.py
"""
ImageMetadataExtractor — Stage 3: extract minimal metadata from image

Images carry very little textual metadata.  This extractor returns just
the detected format, file size, and file name.
"""

from __future__ import annotations

import logging
from typing import Any

from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.types import DocumentMetadata, PreprocessedData

logger = logging.getLogger(__name__)


class ImageMetadataExtractor(BaseMetadataExtractor):
    """Return minimal metadata for a standalone image file."""

    def extract(self, content: Any, **kwargs: Any) -> DocumentMetadata:
        meta = DocumentMetadata()

        # *content* is the raw image bytes (BaseHandler passes preprocessed.content)
        if isinstance(content, bytes):
            meta.page_count = 1
        elif isinstance(content, PreprocessedData):
            props = content.properties or {}
            meta.page_count = 1
            meta.custom = {
                "format": props.get("detected_format", "unknown"),
                "file_size": props.get("file_size", 0),
            }
        else:
            meta.page_count = 1

        return meta

    def get_format_name(self) -> str:
        return "image"


__all__ = ["ImageMetadataExtractor"]
