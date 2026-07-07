# contextifier/handlers/html/metadata_extractor.py
"""
HtmlMetadataExtractor — Stage 3: Extract metadata from <meta> tags.

Pulls author, description, keywords, creation date, etc. from
standard <meta> elements and <title>.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata
from contextifier.handlers.html.preprocessor import HtmlParsedData

logger = logging.getLogger(__name__)

# Map <meta name="..."> values → DocumentMetadata fields
_META_NAME_MAP = {
    "author": "author",
    "creator": "author",
    "dc.creator": "author",
    "description": "subject",
    "dc.description": "subject",
    "keywords": "keywords",
    "dc.subject": "keywords",
    "generator": "last_saved_by",
}

_DATE_NAMES = {
    "date": "create_time",
    "dc.date": "create_time",
    "dcterms.created": "create_time",
    "dcterms.modified": "last_saved_time",
    "last-modified": "last_saved_time",
}


class HtmlMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata from HTML <meta> tags and <title>."""

    def extract(self, source: Any) -> DocumentMetadata:
        soup, title, _ = self._unpack(source)
        if soup is None:
            return DocumentMetadata()

        fields: dict[str, Optional[str]] = {}
        dates: dict[str, Optional[datetime]] = {}

        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").lower().strip()
            content = meta.get("content", "")
            if not name or not content:
                continue

            if name in _META_NAME_MAP:
                fields[_META_NAME_MAP[name]] = content
            elif name in _DATE_NAMES:
                dt = self._try_parse_date(content)
                if dt:
                    dates[_DATE_NAMES[name]] = dt

        return DocumentMetadata(
            title=title or fields.get("title"),
            subject=fields.get("subject"),
            author=fields.get("author"),
            keywords=fields.get("keywords"),
            last_saved_by=fields.get("last_saved_by"),
            create_time=dates.get("create_time"),
            last_saved_time=dates.get("last_saved_time"),
        )

    def get_format_name(self) -> str:
        return "html"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(source: Any):
        if isinstance(source, HtmlParsedData):
            return source.soup, source.title, source.encoding
        return None, "", "utf-8"

    @staticmethod
    def _try_parse_date(value: str) -> Optional[datetime]:
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
        return None
