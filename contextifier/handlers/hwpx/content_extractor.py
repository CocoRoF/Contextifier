# contextifier_new/handlers/hwpx/content_extractor.py
"""
HwpxContentExtractor — Stage 4: ZipFile → text / tables / images

Iterates ``Contents/section*.xml`` entries (sorted numerically),
delegates each to ``parse_hwpx_section()`` which walks the XML tree
and returns text with inline table HTML and image tags.

After processing sections, any remaining un-processed BinData images
are appended at the end.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import zipfile

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    ChartData,
    ExtractionResult,
    PreprocessedData,
    TableData,
)
from contextifier.handlers.hwpx._constants import (
    BINDATA_PREFIX,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from contextifier.handlers.hwpx._section import parse_hwpx_section
from contextifier.handlers.hwpx.preprocessor import find_section_paths

if TYPE_CHECKING:
    from contextifier.services.image_service import ImageService
    from contextifier.services.tag_service import TagService
    from contextifier.services.chart_service import ChartService
    from contextifier.services.table_service import TableService

logger = logging.getLogger(__name__)


class HwpxContentExtractor(BaseContentExtractor):
    """
    Extract text, tables, images, and charts from an HWPX archive.
    """

    # ── BaseContentExtractor interface ────────────────────────────────

    def extract_text(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> str:
        zf, bin_item_map = self._unwrap(preprocessed)
        if zf is None:
            return ""

        section_paths = preprocessed.properties.get("section_paths") or find_section_paths(zf)
        processed_images: Set[str] = set()
        parts: List[str] = []

        for section_path in section_paths:
            try:
                with zf.open(section_path) as f:
                    section_xml = f.read()

                section_text = parse_hwpx_section(
                    section_xml,
                    zf,
                    bin_item_map,
                    image_service=self._image_service,
                    chart_service=self._chart_service,
                    processed_images=processed_images,
                )
                if section_text and section_text.strip():
                    parts.append(section_text)

            except Exception as exc:
                logger.warning("Error parsing section %s: %s", section_path, exc)

        # Process remaining BinData images not referenced inline
        remaining = self._process_remaining_images(
            zf, bin_item_map, processed_images,
        )
        if remaining:
            parts.append(remaining)

        return "\n".join(parts)

    def extract_tables(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[TableData]:
        # Tables are rendered inline as HTML/text during extract_text
        return []

    def extract_images(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[str]:
        # Images are embedded as tags during extract_text
        return []

    def extract_charts(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[ChartData]:
        # Charts are rendered inline during extract_text
        return []

    def get_format_name(self) -> str:
        return "hwpx"

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _unwrap(preprocessed: PreprocessedData) -> tuple:
        """Return ``(ZipFile, bin_item_map)`` from PreprocessedData."""
        zf = preprocessed.content
        if not isinstance(zf, zipfile.ZipFile):
            return None, {}

        resources = preprocessed.resources or {}
        bin_item_map = resources.get("bin_item_map", {})
        return zf, bin_item_map

    def _process_remaining_images(
        self,
        zf: zipfile.ZipFile,
        bin_item_map: Dict[str, str],
        processed_images: Set[str],
    ) -> str:
        """
        Process BinData images that were NOT referenced inline.

        Returns image tags joined by newlines, or empty string.
        """
        if self._image_service is None:
            return ""

        tags: List[str] = []

        # Scan BinData/ entries
        for name in sorted(zf.namelist()):
            if not name.startswith(BINDATA_PREFIX):
                continue
            if name in processed_images:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            try:
                with zf.open(name) as f:
                    data = f.read()
                tag = self._image_service.save_image(data)
                if tag:
                    processed_images.add(name)
                    tags.append(tag)
            except Exception as exc:
                logger.debug("Failed to process remaining image %s: %s", name, exc)

        return "\n".join(tags) if tags else ""


__all__ = ["HwpxContentExtractor"]
