# contextifier_new/handlers/hwpx/preprocessor.py
"""
HwpxPreprocessor — Stage 2: ZipFile → PreprocessedData

Parses the OPF manifest (``content.hpf``) to build the
``bin_item_map`` (binItemId → file path in ZIP), counts sections,
and stores everything in ``PreprocessedData.resources / .properties``.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.errors import PreprocessingError
from contextifier.handlers.hwpx._constants import (
    HPF_PATH,
    OPF_NAMESPACES,
    SECTION_PREFIX,
)
from contextifier.handlers.hwpx.converter import HwpxConvertedData

logger = logging.getLogger(__name__)


def parse_bin_item_map(zf: zipfile.ZipFile) -> Dict[str, str]:
    """
    Parse ``Contents/content.hpf`` (OPF manifest) to build the
    bin-item-id → ZIP path mapping.

    OPF structure::

        <opf:package>
          <opf:manifest>
            <opf:item id="image3" href="BinData/image3.png" />
          </opf:manifest>
        </opf:package>

    Returns:
        Dict mapping item id (e.g. ``"image3"``) to href
        (e.g. ``"BinData/image3.png"``).
    """
    bin_item_map: Dict[str, str] = {}
    try:
        if HPF_PATH not in zf.namelist():
            return bin_item_map

        with zf.open(HPF_PATH) as f:
            root = ET.fromstring(f.read())

        for item in root.findall(".//opf:item", OPF_NAMESPACES):
            item_id = item.get("id")
            href = item.get("href")
            if item_id and href:
                bin_item_map[item_id] = href

    except Exception as exc:
        logger.warning("Failed to parse content.hpf: %s", exc)

    return bin_item_map


def find_section_paths(zf: zipfile.ZipFile) -> List[str]:
    """
    Find all ``Contents/section*.xml`` entries sorted numerically.

    Returns:
        Sorted list of section paths inside the ZIP.
    """
    pattern = re.compile(r"^Contents/section(\d+)\.xml$", re.IGNORECASE)
    sections: List[tuple[int, str]] = []

    for name in zf.namelist():
        m = pattern.match(name)
        if m:
            sections.append((int(m.group(1)), name))

    sections.sort(key=lambda t: t[0])
    return [path for _, path in sections]


class HwpxPreprocessor(BasePreprocessor):
    """
    Preprocess an HWPX ZIP archive.

    Extracts:
    - ``bin_item_map`` from the OPF manifest
    - section paths for the content extractor
    - basic archive statistics
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        zf, file_data = self._unwrap(converted_data)
        if zf is None:
            raise PreprocessingError(
                "No ZipFile available for HWPX preprocessing",
                stage="preprocess",
                handler="hwpx",
            )

        bin_item_map = parse_bin_item_map(zf)
        sections = find_section_paths(zf)

        return PreprocessedData(
            content=zf,                     # ZipFile for downstream stages
            raw_content=file_data,          # Original bytes
            encoding="utf-8",
            resources={
                "file_data": file_data,
                "bin_item_map": bin_item_map,
            },
            properties={
                "section_count": len(sections),
                "section_paths": sections,
                "total_entries": len(zf.namelist()),
            },
        )

    def get_format_name(self) -> str:
        return "hwpx"

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(source: Any) -> tuple:
        """Return ``(ZipFile, bytes)`` from various input shapes."""
        if isinstance(source, HwpxConvertedData):
            return source.zf, source.file_data
        if isinstance(source, zipfile.ZipFile):
            return source, b""
        return None, b""


__all__ = ["HwpxPreprocessor", "parse_bin_item_map", "find_section_paths"]
