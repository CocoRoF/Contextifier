# contextifier_new/handlers/hwpx/metadata_extractor.py
"""
HwpxMetadataExtractor — Stage 3: ZipFile → DocumentMetadata

Reads metadata from three HWPX sources:
1. ``Contents/header.xml`` — ``<hh:docInfo>`` children
2. ``version.xml`` — version attributes (stored in ``custom``)
3. ``META-INF/manifest.xml`` — media-type (stored in ``custom``)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, Optional

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata, PreprocessedData
from contextifier.handlers.hwpx._constants import (
    HEADER_FILE_PATHS,
    HWPX_NAMESPACES,
    MANIFEST_NAMESPACES,
    MANIFEST_PATH,
    VERSION_PATH,
)

logger = logging.getLogger(__name__)


class HwpxMetadataExtractor(BaseMetadataExtractor):
    """
    Extract metadata from HWPX archives.

    The ``extract()`` method accepts:
    - A ``zipfile.ZipFile`` directly (from ``preprocessed.content``)
    - A ``PreprocessedData`` wrapper
    """

    def extract(self, source: Any) -> DocumentMetadata:
        zf = self._unwrap(source)
        if zf is None:
            return DocumentMetadata()

        raw: Dict[str, Any] = {}

        try:
            self._read_header(zf, raw)
            self._read_version(zf, raw)
            self._read_manifest(zf, raw)
        except Exception as exc:
            logger.warning("HWPX metadata extraction error: %s", exc)

        # Map standard fields
        standard_keys = {
            "title", "subject", "author", "keywords", "comments",
            "last_saved_by", "create_time", "last_saved_time",
        }
        custom = {k: v for k, v in raw.items() if k not in standard_keys}

        meta = DocumentMetadata(
            title=_str_or_none(raw.get("title")),
            subject=_str_or_none(raw.get("subject")),
            author=_str_or_none(raw.get("author")),
            keywords=_str_or_none(raw.get("keywords")),
            comments=_str_or_none(raw.get("comments")),
            last_saved_by=_str_or_none(raw.get("last_saved_by")),
            custom=custom if custom else {},
        )

        # Page count heuristic: number of sections
        section_count = self._count_sections(zf)
        if section_count > 0:
            meta.page_count = section_count

        return meta

    def get_format_name(self) -> str:
        return "hwpx"

    # ── Internal readers ──────────────────────────────────────────────

    def _read_header(self, zf: zipfile.ZipFile, raw: Dict[str, Any]) -> None:
        """Parse ``Contents/header.xml`` → ``<hh:docInfo>`` children."""
        for header_path in HEADER_FILE_PATHS:
            if header_path not in zf.namelist():
                continue
            try:
                with zf.open(header_path) as f:
                    root = ET.fromstring(f.read())

                doc_info = root.find(".//hh:docInfo", HWPX_NAMESPACES)
                if doc_info is not None:
                    for prop in doc_info:
                        tag = prop.tag.split("}")[-1] if "}" in prop.tag else prop.tag
                        if prop.text and prop.text.strip():
                            raw[tag.lower()] = prop.text.strip()
                return  # success — stop trying alternative paths
            except Exception as exc:
                logger.debug("Failed to read %s: %s", header_path, exc)

    def _read_version(self, zf: zipfile.ZipFile, raw: Dict[str, Any]) -> None:
        """Parse ``version.xml`` for version attributes."""
        if VERSION_PATH not in zf.namelist():
            return
        try:
            with zf.open(VERSION_PATH) as f:
                root = ET.fromstring(f.read())
            if root.text and root.text.strip():
                raw["version"] = root.text.strip()
            for attr in root.attrib:
                raw[f"version_{attr}"] = root.get(attr)
        except Exception as exc:
            logger.debug("Failed to read version.xml: %s", exc)

    def _read_manifest(self, zf: zipfile.ZipFile, raw: Dict[str, Any]) -> None:
        """Parse ``META-INF/manifest.xml`` for MIME-type info."""
        if MANIFEST_PATH not in zf.namelist():
            return
        try:
            with zf.open(MANIFEST_PATH) as f:
                root = ET.fromstring(f.read())

            for child in root:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if tag == "file-entry":
                    full_path = (
                        child.get("full-path")
                        or child.get(
                            "{urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}full-path",
                            "",
                        )
                    )
                    if full_path == "/":
                        media_type = (
                            child.get("media-type")
                            or child.get(
                                "{urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}media-type",
                                "",
                            )
                        )
                        if media_type:
                            raw["media_type"] = media_type
        except Exception as exc:
            logger.debug("Failed to read manifest.xml: %s", exc)

    @staticmethod
    def _count_sections(zf: zipfile.ZipFile) -> int:
        """Count ``Contents/section*.xml`` entries."""
        count = 0
        for name in zf.namelist():
            lower = name.lower()
            if lower.startswith("contents/section") and lower.endswith(".xml"):
                count += 1
        return count

    # ── Unwrap helper ─────────────────────────────────────────────────

    @staticmethod
    def _unwrap(source: Any) -> Optional[zipfile.ZipFile]:
        """Extract a ZipFile from various input shapes."""
        if isinstance(source, zipfile.ZipFile):
            return source
        if isinstance(source, PreprocessedData):
            content = source.content
            if isinstance(content, zipfile.ZipFile):
                return content
        return None


def _str_or_none(val: Any) -> Optional[str]:
    """Return stripped string or ``None``."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


__all__ = ["HwpxMetadataExtractor"]
