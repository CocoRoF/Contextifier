# contextifier/handlers/html/handler.py
"""
HtmlHandler — Handler for HTML files (.html, .htm, .xhtml).

Parses HTML with BeautifulSoup and extracts structured text,
preserving headings, lists, tables, and links in an AI-friendly
format.  Also extracts embedded base64 images and <meta> metadata.

Pipeline:
    Stage 1  HtmlConverter       — Validate & decode bytes → str
    Stage 2  HtmlPreprocessor    — Parse with BS4, strip scripts/styles
    Stage 3  HtmlMetadataExtractor — Extract <title>, <meta> tags
    Stage 4  HtmlContentExtractor  — Structured HTML → clean text
    Stage 5  DefaultPostprocessor   — Standard assembly
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.html.converter import HtmlConverter
from contextifier.handlers.html.preprocessor import HtmlPreprocessor
from contextifier.handlers.html.metadata_extractor import HtmlMetadataExtractor
from contextifier.handlers.html.content_extractor import HtmlContentExtractor


class HtmlHandler(BaseHandler):
    """
    Handler for HTML/HTM/XHTML files.

    Uses BeautifulSoup for robust parsing.  Extracts structure-aware
    text, metadata from ``<meta>`` tags, tables, and base64 images.
    """

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"html", "htm", "xhtml"})

    @property
    def handler_name(self) -> str:
        return "HTML Handler"

    def create_converter(self) -> BaseConverter:
        return HtmlConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return HtmlPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return HtmlMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return HtmlContentExtractor(
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
