# contextifier/handlers/html/converter.py
"""
HtmlConverter — Stage 1: Validate and decode HTML bytes.

Validates that the input looks like HTML (magic bytes or common
markers), detects encoding via BOM / charset declaration, and
returns the decoded string for downstream processing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple, Optional

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError

logger = logging.getLogger(__name__)

# Common byte-order marks
_BOMS = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]

# Regex to find charset in <meta> tags (bytes-level)
_CHARSET_RE = re.compile(
    rb'<meta[^>]+charset=["\']?([a-zA-Z0-9_-]+)',
    re.IGNORECASE,
)

# HTML markers expected at the start of a file
_HTML_MARKERS = (
    b"<!doctype",
    b"<html",
    b"<head",
    b"<body",
    b"<!--",
    b"<?xml",
)


class HtmlConvertedData(NamedTuple):
    """Result of Stage 1 conversion for HTML."""
    html_text: str
    encoding: str
    file_extension: str


class HtmlConverter(BaseConverter):
    """Validate HTML input and decode to string."""

    def convert(
        self, file_context: FileContext, **kwargs: Any
    ) -> HtmlConvertedData:
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data for HTML handler",
                stage="convert",
                handler="HtmlHandler",
            )

        encoding = self._detect_encoding(file_data)
        try:
            html_text = file_data.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            html_text = file_data.decode("utf-8", errors="replace")
            encoding = "utf-8"

        ext = file_context.get("file_extension", "html")
        return HtmlConvertedData(
            html_text=html_text,
            encoding=encoding,
            file_extension=ext,
        )

    def validate(self, file_context: FileContext) -> bool:
        data = file_context.get("file_data", b"")
        if not data:
            return False
        header = data[:512].lstrip().lower()
        return any(header.startswith(m) for m in _HTML_MARKERS) or b"<" in header[:64]

    def get_format_name(self) -> str:
        return "html"

    def close(self, converted: Any) -> None:
        pass  # nothing to clean up

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_encoding(data: bytes) -> str:
        """Detect encoding from BOM or <meta charset>."""
        for bom, enc in _BOMS:
            if data.startswith(bom):
                return enc

        match = _CHARSET_RE.search(data[:2048])
        if match:
            declared = match.group(1).decode("ascii", errors="ignore").strip()
            if declared:
                return declared

        return "utf-8"
