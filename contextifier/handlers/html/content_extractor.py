# contextifier/handlers/html/content_extractor.py
"""
HtmlContentExtractor — Stage 4: Convert parsed HTML to structured text.

Traverses the BeautifulSoup tree and produces AI-friendly text while
preserving document structure:

- Headings → prefixed with ``#`` markers (Markdown-style)
- Paragraphs → double-newline separated
- Lists (``<ul>``/``<ol>``) → bulleted / numbered items
- Tables → HTML pass-through (or TableService formatting)
- Links → ``[text](href)`` notation
- Images → saved via ImageService (base64 data-URIs + external ``src``)
- ``<pre>``/``<code>`` → fenced code blocks
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from bs4 import BeautifulSoup, NavigableString, Tag

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    PreprocessedData,
    TableData,
    TableCell,
    ChartData,
)
from contextifier.handlers.html.preprocessor import HtmlParsedData

logger = logging.getLogger(__name__)

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "aside", "main", "header",
    "footer", "nav", "blockquote", "figure", "figcaption", "details",
    "summary", "address", "hgroup",
}
_LIST_TAGS = {"ul", "ol"}
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "math"}


class HtmlContentExtractor(BaseContentExtractor):
    """Extract structured text, tables, and images from parsed HTML."""

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        soup, _, _ = self._unpack(preprocessed)
        if soup is None:
            return ""

        # Start from <body> if present, otherwise whole soup
        root = soup.find("body") or soup
        parts: List[str] = []
        self._walk(root, parts)

        # Save any base64 embedded images from preprocessor
        image_tags = self._save_embedded_images(preprocessed)
        if image_tags:
            parts.extend(image_tags)

        result = "\n\n".join(p for p in parts if p.strip())
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        soup, _, _ = self._unpack(preprocessed)
        if soup is None:
            return []

        tables: List[TableData] = []
        for table_tag in (soup.find("body") or soup).find_all("table"):
            td = self._parse_table(table_tag)
            if td and td.num_rows > 0:
                tables.append(td)
        return tables

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        return self._save_embedded_images(preprocessed)

    def get_format_name(self) -> str:
        return "html"

    # ── Tree walking ──────────────────────────────────────────────────────

    def _walk(self, element: Any, parts: List[str]) -> None:
        """Recursively convert an element subtree to text parts."""
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                parts.append(text)
            return

        if not isinstance(element, Tag):
            return

        tag_name = element.name.lower() if element.name else ""

        if tag_name in _SKIP_TAGS:
            return

        if tag_name in _HEADING_TAGS:
            level = int(tag_name[1])
            text = element.get_text(" ", strip=True)
            if text:
                parts.append(f"{'#' * level} {text}")
            return

        if tag_name == "table":
            html = self._render_table(element)
            if html:
                parts.append(html)
            return

        if tag_name in _LIST_TAGS:
            items = self._render_list(element, ordered=(tag_name == "ol"))
            if items:
                parts.append(items)
            return

        if tag_name == "pre":
            code = element.get_text()
            if code.strip():
                parts.append(f"```\n{code.rstrip()}\n```")
            return

        if tag_name == "code" and element.parent and element.parent.name != "pre":
            text = element.get_text()
            if text.strip():
                parts.append(f"`{text.strip()}`")
            return

        if tag_name == "a":
            text = element.get_text(strip=True)
            href = element.get("href", "")
            if text and href and not href.startswith(("#", "javascript:")):
                parts.append(f"[{text}]({href})")
            elif text:
                parts.append(text)
            return

        if tag_name == "img":
            alt = element.get("alt", "").strip()
            if alt:
                parts.append(f"[Image: {alt}]")
            return

        if tag_name == "br":
            parts.append("\n")
            return

        if tag_name == "hr":
            parts.append("---")
            return

        if tag_name == "blockquote":
            inner_parts: List[str] = []
            self._walk_children(element, inner_parts)
            text = "\n".join(inner_parts)
            if text.strip():
                quoted = "\n".join(f"> {line}" for line in text.split("\n"))
                parts.append(quoted)
            return

        # Generic block element → walk children with spacing
        if tag_name in _BLOCK_TAGS:
            self._walk_children(element, parts)
            return

        # Inline or unknown → walk children
        self._walk_children(element, parts)

    def _walk_children(self, element: Tag, parts: List[str]) -> None:
        """Walk all child nodes."""
        for child in element.children:
            self._walk(child, parts)

    # ── Lists ─────────────────────────────────────────────────────────────

    def _render_list(self, list_tag: Tag, ordered: bool = False) -> str:
        items: List[str] = []
        for idx, li in enumerate(list_tag.find_all("li", recursive=False), 1):
            text = li.get_text(" ", strip=True)
            if text:
                prefix = f"{idx}." if ordered else "-"
                items.append(f"{prefix} {text}")
        return "\n".join(items)

    # ── Tables ────────────────────────────────────────────────────────────

    def _render_table(self, table_tag: Tag) -> str:
        """Render a <table> as HTML string for TableService or pass-through."""
        td = self._parse_table(table_tag)
        if td is None or td.num_rows == 0:
            return ""

        if self._table_service is not None:
            try:
                return self._table_service.format_table(td)
            except Exception as exc:
                logger.debug("TableService failed for HTML table: %s", exc)

        # Fallback: return cleaned outer HTML
        return str(table_tag)

    def _parse_table(self, table_tag: Tag) -> Optional[TableData]:
        """Parse an HTML <table> into a ``TableData`` object."""
        rows_data: List[List[TableCell]] = []

        for tr in table_tag.find_all("tr"):
            row: List[TableCell] = []
            for cell in tr.find_all(["td", "th"]):
                content = cell.get_text(" ", strip=True)
                is_header = cell.name == "th"
                rowspan = int(cell.get("rowspan", 1) or 1)
                colspan = int(cell.get("colspan", 1) or 1)
                row.append(
                    TableCell(
                        content=content,
                        is_header=is_header,
                        row_span=rowspan,
                        col_span=colspan,
                    )
                )
            if row:
                rows_data.append(row)

        if not rows_data:
            return None

        num_cols = max(
            sum(c.col_span for c in r) for r in rows_data
        ) if rows_data else 0

        has_header = bool(rows_data) and all(c.is_header for c in rows_data[0])

        caption_tag = table_tag.find("caption")
        caption = caption_tag.get_text(strip=True) if caption_tag else None

        return TableData(
            rows=rows_data,
            num_rows=len(rows_data),
            num_cols=num_cols,
            has_header=has_header,
            caption=caption,
        )

    # ── Images ────────────────────────────────────────────────────────────

    def _save_embedded_images(self, preprocessed: PreprocessedData) -> List[str]:
        """Save base64 embedded images and return tags."""
        if self._image_service is None:
            return []

        images = preprocessed.resources.get("images", [])
        tags: List[str] = []
        for idx, img_info in enumerate(images):
            data: bytes = img_info.get("data", b"")
            fmt: str = img_info.get("format", "png")
            if not data:
                continue
            tag = self._image_service.save_and_tag(
                image_bytes=data,
                custom_name=f"html_image_{idx}.{fmt}",
            )
            if tag:
                tags.append(tag)
        return tags

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(preprocessed: PreprocessedData):
        content = preprocessed.content
        if isinstance(content, HtmlParsedData):
            return content.soup, content.title, content.encoding
        return None, "", "utf-8"
