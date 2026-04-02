# contextifier/handlers/html/preprocessor.py
"""
HtmlPreprocessor — Stage 2: Parse HTML and prepare resources.

Uses BeautifulSoup to parse the HTML string into a navigable tree.
Strips <script>, <style>, and non-visible elements.  Extracts
embedded images (base64 data-URIs) as resources for the content
extractor.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional

from bs4 import BeautifulSoup, Tag

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.handlers.html.converter import HtmlConvertedData

logger = logging.getLogger(__name__)

# Maximum decoded size for base64 embedded images (50 MB).
# Images exceeding this limit are silently skipped to prevent
# memory exhaustion from maliciously crafted HTML.
_MAX_IMAGE_DECODE_BYTES = 50 * 1024 * 1024

_DATA_URI_RE = re.compile(
    r"data:image/([a-zA-Z0-9+.-]+);base64,([A-Za-z0-9+/=\s]+)"
)


class HtmlParsedData(NamedTuple):
    """Preprocessed HTML payload."""
    soup: BeautifulSoup
    title: str
    encoding: str


class HtmlPreprocessor(BasePreprocessor):
    """Parse HTML, strip scripts/styles, extract embedded images."""

    def preprocess(
        self, converted_data: Any, **kwargs: Any
    ) -> PreprocessedData:
        html_text, encoding, file_ext = self._unpack(converted_data)

        if not html_text:
            empty_soup = BeautifulSoup("", "html.parser")
            return PreprocessedData(
                content=HtmlParsedData(soup=empty_soup, title="", encoding=encoding),
                raw_content="",
                encoding=encoding,
            )

        soup = BeautifulSoup(html_text, "html.parser")

        # Extract title before stripping
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove non-visible elements
        for tag_name in ("script", "style", "noscript"):
            for element in soup.find_all(tag_name):
                element.decompose()

        # Remove HTML comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Extract base64 embedded images
        images: List[Dict[str, Any]] = []
        for img in soup.find_all("img"):
            src = img.get("src", "")
            m = _DATA_URI_RE.match(src)
            if m:
                fmt = m.group(1)
                b64_str = m.group(2)
                # Estimate decoded size (~3/4 of base64 string length)
                estimated_size = len(b64_str) * 3 // 4
                if estimated_size > _MAX_IMAGE_DECODE_BYTES:
                    logger.warning(
                        "Skipping oversized base64 image (~%d MB, limit %d MB)",
                        estimated_size // (1024 * 1024),
                        _MAX_IMAGE_DECODE_BYTES // (1024 * 1024),
                    )
                    continue
                try:
                    data = base64.b64decode(b64_str)
                    images.append({"format": fmt, "data": data})
                except Exception:
                    pass

        parsed = HtmlParsedData(
            soup=soup,
            title=title,
            encoding=encoding,
        )

        return PreprocessedData(
            content=parsed,
            raw_content=html_text,
            encoding=encoding,
            resources={"images": images} if images else {},
            properties={
                "file_extension": file_ext,
                "encoding": encoding,
                "title": title,
                "image_count": len(images),
            },
        )

    def get_format_name(self) -> str:
        return "html"

    def validate(self, data: Any) -> bool:
        if isinstance(data, HtmlConvertedData):
            return True
        return data is not None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(converted_data: Any):
        if isinstance(converted_data, HtmlConvertedData):
            return converted_data.html_text, converted_data.encoding, converted_data.file_extension
        if isinstance(converted_data, str):
            return converted_data, "utf-8", "html"
        return "", "utf-8", "html"
