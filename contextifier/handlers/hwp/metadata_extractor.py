# contextifier/handlers/hwp/metadata_extractor.py
"""
HwpMetadataExtractor — Stage 3: OLE → DocumentMetadata

Extracts metadata from HWP files via two methods:
1. Standard OLE metadata (``olefile.get_metadata()``)
2. HWP-specific ``\\x05HwpSummaryInformation`` stream (property-set
   format with cp949 / UTF-16LE strings)

The HWP-specific stream takes priority when both are available.
"""

from __future__ import annotations

import struct
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import olefile

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata
from contextifier.handlers.hwp._constants import STREAM_HWP_SUMMARY

logger = logging.getLogger(__name__)


class HwpMetadataExtractor(BaseMetadataExtractor):
    """
    Extract metadata from an ``olefile.OleFileIO`` HWP container.
    """

    def extract(self, source: Any) -> DocumentMetadata:
        ole = self._unwrap(source)
        if ole is None:
            return DocumentMetadata()

        meta_dict: Dict[str, Any] = {}

        # 1. Standard OLE metadata
        try:
            om = ole.get_metadata()
            if om:
                for field, attr in (
                    ("title", "title"),
                    ("subject", "subject"),
                    ("author", "author"),
                    ("keywords", "keywords"),
                    ("comments", "comments"),
                    ("last_saved_by", "last_saved_by"),
                    ("create_time", "create_time"),
                    ("last_saved_time", "last_saved_time"),
                ):
                    val = getattr(om, attr, None)
                    if val:
                        meta_dict[field] = val
        except Exception as exc:
            logger.debug("OLE metadata extraction failed: %s", exc)

        # 2. HwpSummaryInformation stream (overwrites OLE values)
        try:
            if ole.exists(STREAM_HWP_SUMMARY):
                data = ole.openstream(STREAM_HWP_SUMMARY).read()
                for k, v in _parse_hwp_summary(data).items():
                    if v:
                        meta_dict[k] = v
        except Exception as exc:
            logger.debug("HwpSummaryInformation parsing failed: %s", exc)

        # Count sections as pages
        section_count = sum(
            1 for e in ole.listdir()
            if len(e) >= 2 and e[0] == "BodyText" and e[1].startswith("Section")
        )

        return DocumentMetadata(
            title=_str_or_none(meta_dict.get("title")),
            subject=_str_or_none(meta_dict.get("subject")),
            author=_str_or_none(meta_dict.get("author")),
            keywords=_str_or_none(meta_dict.get("keywords")),
            comments=_str_or_none(meta_dict.get("comments")),
            last_saved_by=_str_or_none(meta_dict.get("last_saved_by")),
            create_time=meta_dict.get("create_time") if isinstance(meta_dict.get("create_time"), datetime) else None,
            last_saved_time=meta_dict.get("last_saved_time") if isinstance(meta_dict.get("last_saved_time"), datetime) else None,
            page_count=section_count or None,
        )

    def get_format_name(self) -> str:
        return "hwp"

    # ── internal ──────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(source: Any) -> Optional[olefile.OleFileIO]:
        """Accept olefile.OleFileIO or PreprocessedData wrapping one."""
        if isinstance(source, olefile.OleFileIO):
            return source
        if hasattr(source, "content") and isinstance(source.content, olefile.OleFileIO):
            return source.content
        if hasattr(source, "listdir"):
            return source
        return None


# ══════════════════════════════════════════════════════════════════════════
# HwpSummaryInformation parser (OLE Property Set format)
# ══════════════════════════════════════════════════════════════════════════


def _parse_hwp_summary(data: bytes) -> Dict[str, Any]:
    """Parse the HWP-specific summary information stream."""
    meta: Dict[str, Any] = {}
    if len(data) < 28:
        return meta

    try:
        pos = 28  # skip header
        if len(data) < pos + 20:
            return meta

        section_offset = struct.unpack_from("<I", data, pos + 16)[0]
        if section_offset >= len(data):
            return meta

        pos = section_offset
        if len(data) < pos + 8:
            return meta

        num_props = struct.unpack_from("<I", data, pos + 4)[0]
        pos += 8

        props = []
        for _ in range(min(num_props, 50)):
            if len(data) < pos + 8:
                break
            pid = struct.unpack_from("<I", data, pos)[0]
            poff = struct.unpack_from("<I", data, pos + 4)[0]
            props.append((pid, poff))
            pos += 8

        _PROP_MAP = {
            0x02: "title",
            0x03: "subject",
            0x04: "author",
            0x05: "keywords",
            0x06: "comments",
            0x08: "last_saved_by",
            0x0C: "create_time",
            0x0D: "last_saved_time",
        }

        for pid, poff in props:
            abs_off = section_offset + poff
            if abs_off + 4 >= len(data):
                continue
            ptype = struct.unpack_from("<I", data, abs_off)[0]
            voff = abs_off + 4

            value = None

            if ptype == 0x1E:  # ANSI string (typically cp949)
                if voff + 4 < len(data):
                    slen = struct.unpack_from("<I", data, voff)[0]
                    if 0 < slen and voff + 4 + slen <= len(data):
                        raw = data[voff + 4: voff + 4 + slen]
                        try:
                            value = raw.decode("cp949", errors="ignore").rstrip("\x00")
                        except Exception:
                            value = raw.decode("utf-8", errors="ignore").rstrip("\x00")

            elif ptype == 0x1F:  # Unicode string
                if voff + 4 < len(data):
                    slen = struct.unpack_from("<I", data, voff)[0]
                    blen = slen * 2
                    if 0 < slen and voff + 4 + blen <= len(data):
                        value = data[voff + 4: voff + 4 + blen].decode(
                            "utf-16le", errors="ignore"
                        ).rstrip("\x00")

            elif ptype == 0x40:  # FILETIME
                if voff + 8 <= len(data):
                    ft = struct.unpack_from("<Q", data, voff)[0]
                    if ft > 0:
                        try:
                            secs = ft / 10_000_000
                            unix = secs - 11_644_473_600
                            if 0 < unix < 2_000_000_000:
                                value = datetime.fromtimestamp(unix)
                        except Exception:
                            pass

            if value and pid in _PROP_MAP:
                meta[_PROP_MAP[pid]] = value

    except Exception as exc:
        logger.debug("Error parsing HWP summary: %s", exc)

    return meta


def _str_or_none(v: Any) -> Optional[str]:
    """Stringify non-None values, stripping whitespace."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


__all__ = ["HwpMetadataExtractor"]
