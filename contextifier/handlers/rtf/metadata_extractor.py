# contextifier_new/handlers/rtf/metadata_extractor.py
"""
RtfMetadataExtractor — Stage 3: RTF Metadata Extraction.

Extracts document metadata from the RTF ``\\info`` group:
- Text fields: title, subject, author, keywords, comments, last_saved_by
- Date fields: create_time (\\creatim), last_saved_time (\\revtim)

Input: RtfParsedData from RtfPreprocessor (decoded RTF string + encoding).
Output: DocumentMetadata with populated standard fields.

Ported from v1.0 rtf_metadata_extractor.py with:
- Accepts RtfParsedData (NamedTuple) instead of RTFSourceInfo (dataclass)
- Uses shared _decoder.decode_hex_escapes and _cleaner.clean_rtf_text
- Conforms to BaseMetadataExtractor.extract() / get_format_name()
- Improved \\info group regex to handle nested braces reliably
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata
from contextifier.handlers.rtf.preprocessor import RtfParsedData
from contextifier.handlers.rtf._decoder import decode_hex_escapes
from contextifier.handlers.rtf._cleaner import clean_rtf_text

_logger = logging.getLogger("contextifier.rtf.metadata")


# ── Field extraction patterns (within \info group) ──────────────────────
# RTF structure: {\title content} — the { precedes the control word
_METADATA_FIELDS: Dict[str, str] = {
    "title":         r"\{\\title\s+([^}]*)\}",
    "subject":       r"\{\\subject\s+([^}]*)\}",
    "author":        r"\{\\author\s+([^}]*)\}",
    "keywords":      r"\{\\keywords\s+([^}]*)\}",
    "comments":      r"\{\\doccomm\s+([^}]*)\}",
    "last_saved_by": r"\{\\operator\s+([^}]*)\}",
}

# Date patterns: \creatim and \revtim with year/month/day + optional hr/min
_DATE_PATTERNS: Dict[str, str] = {
    "create_time": (
        r"\\creatim\s*"
        r"\\yr(\d+)\s*\\mo(\d+)\s*\\dy(\d+)"
        r"(?:\s*\\hr(\d+))?(?:\s*\\min(\d+))?"
    ),
    "last_saved_time": (
        r"\\revtim\s*"
        r"\\yr(\d+)\s*\\mo(\d+)\s*\\dy(\d+)"
        r"(?:\s*\\hr(\d+))?(?:\s*\\min(\d+))?"
    ),
}


class RtfMetadataExtractor(BaseMetadataExtractor):
    """
    Extract metadata from a decoded RTF string.

    Supported metadata fields:
    - title, subject, author, keywords, comments, last_saved_by
    - create_time, last_saved_time

    All text values are decoded for hex escapes (``\\'XX``) and cleaned
    of residual RTF control codes via ``_cleaner.clean_rtf_text``.
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from RTF content.

        Args:
            source: RtfParsedData (from preprocessor) or dict.

        Returns:
            DocumentMetadata with populated fields.
        """
        text, encoding = self._unpack(source)
        if not text:
            return DocumentMetadata()

        # Locate the \info group (may contain nested brace groups)
        info_content = self._extract_info_group(text)
        if not info_content:
            _logger.debug("No \\info group found in RTF content")
            return DocumentMetadata()

        # Extract text fields
        fields: Dict[str, Optional[str]] = {}
        for field_name, pattern in _METADATA_FIELDS.items():
            fields[field_name] = self._extract_field(
                info_content, pattern, encoding,
            )

        # Extract date fields (search full content, not just \info)
        dates: Dict[str, Optional[datetime]] = {}
        for date_field, pattern in _DATE_PATTERNS.items():
            dates[date_field] = self._extract_date(text, pattern)

        _logger.debug(
            "RTF metadata: %d text fields, %d dates",
            sum(1 for v in fields.values() if v),
            sum(1 for v in dates.values() if v),
        )

        return DocumentMetadata(
            title=fields.get("title"),
            subject=fields.get("subject"),
            author=fields.get("author"),
            keywords=fields.get("keywords"),
            comments=fields.get("comments"),
            last_saved_by=fields.get("last_saved_by"),
            create_time=dates.get("create_time"),
            last_saved_time=dates.get("last_saved_time"),
        )

    def get_format_name(self) -> str:
        return "rtf"

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _unpack(source: Any) -> tuple[str, str]:
        """Extract text and encoding from various input types."""
        if isinstance(source, RtfParsedData):
            return source.text, source.encoding
        if isinstance(source, dict):
            return source.get("text", ""), source.get("encoding", "cp949")
        if isinstance(source, str):
            return source, "cp949"
        return "", "cp949"

    @staticmethod
    def _extract_info_group(content: str) -> Optional[str]:
        """
        Extract the content of the ``\\info`` group.

        RTF structure: ``{\\info {\\title ...}{\\author ...}}``
        The opening ``{`` is BEFORE ``\\info``, so we search for
        ``{\\info`` and use brace-depth matching from the ``{``
        to find the matching closing ``}``.
        """
        match = re.search(r"\{\\info\b", content)
        if not match:
            return None

        # Start scanning from the { that opens the group
        brace_start = match.start()
        # The inner content starts after \info (and optional whitespace)
        inner_start = match.end()

        # Find matching closing brace
        depth = 1
        i = brace_start + 1  # position after the opening {

        while i < len(content) and depth > 0:
            ch = content[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1

        if depth != 0:
            return None

        # Return the content between \info and the matching }
        return content[inner_start : i - 1]

    @staticmethod
    def _extract_field(
        info_content: str,
        pattern: str,
        encoding: str,
    ) -> Optional[str]:
        """Extract and clean a single metadata field."""
        match = re.search(pattern, info_content)
        if not match:
            return None

        raw_value = match.group(1)
        if not raw_value.strip():
            return None

        # Decode hex escapes (Korean, CJK, special chars)
        value = decode_hex_escapes(raw_value, encoding)
        # Clean residual RTF control codes
        value = clean_rtf_text(value, encoding)
        value = value.strip()

        return value if value else None

    @staticmethod
    def _extract_date(content: str, pattern: str) -> Optional[datetime]:
        """
        Extract a datetime from RTF date control words.

        RTF encodes dates as:
            ``\\creatim\\yr2024\\mo1\\dy15\\hr10\\min30``
        """
        match = re.search(pattern, content)
        if not match:
            return None
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4)) if match.group(4) else 0
            minute = int(match.group(5)) if match.group(5) else 0
            return datetime(year, month, day, hour, minute)
        except (ValueError, TypeError):
            return None


__all__ = ["RtfMetadataExtractor"]
