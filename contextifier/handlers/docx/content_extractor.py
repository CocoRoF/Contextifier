"""
DocxContentExtractor — Stage 4: Extract text, tables, images, charts.

This is the main workhorse that traverses the DOCX document body
(``doc.element.body``) and processes each element in document order:

- ``<w:p>`` (paragraph)  → text, with inline images/charts/diagrams
- ``<w:tbl>`` (table)    → ``TableData`` via ``_table_extractor``
- ``<w:sectPr>`` (section) → ignored (formatting-only)

Images are saved via ``ImageService`` when a drawing/pict element
references a relationship ID.  Charts are matched against the
pre-extracted chart list from the preprocessor.

Page breaks within paragraphs trigger page numbering that is
formatted via the ``TagService`` if available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    ChartData,
    PreprocessedData,
    TableData,
)

from contextifier.handlers.docx._constants import NAMESPACES, ElementType
from contextifier.handlers.docx._paragraph import (
    process_paragraph,
    extract_diagram_text,
    DrawingInfo,
    DrawingKind,
    PictInfo,
)
from contextifier.handlers.docx._table_extractor import extract_table

logger = logging.getLogger(__name__)

_W = NAMESPACES["w"]
_R = NAMESPACES["r"]


class DocxContentExtractor(BaseContentExtractor):
    """
    Content extractor for DOCX files.

    Traverses the document body element and extracts all content
    types (text, tables, images, charts) in document order.
    """

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from all paragraphs, interleaving tables and images.

        The resulting text includes:
        - Paragraph text
        - Page break tags (via TagService)
        - Table HTML (via TableService)
        - Image tags (via ImageService)
        - Chart blocks (pre-extracted in preprocessor)
        - Diagram text
        """
        doc = self._get_document(preprocessed)
        if doc is None:
            return ""

        # Pre-extracted charts from preprocessor
        charts: List[str] = preprocessed.resources.get("charts", [])
        chart_index = 0

        # Page tracking
        page_number = 1
        parts: List[str] = []
        processed_images: Set[str] = set()

        # Add initial page tag
        page_tag = self._make_page_tag(page_number)
        if page_tag:
            parts.append(page_tag)

        # Iterate body elements in document order
        body = doc.element.body
        if body is None:
            return ""

        for element in body:
            local = _local_name(element)

            if local == "p":
                # Paragraph
                text, drawings, picts, has_break = process_paragraph(element)

                # Handle page breaks BEFORE content
                if has_break:
                    page_number += 1
                    tag = self._make_page_tag(page_number)
                    if tag:
                        parts.append(tag)

                # Process drawings (images, charts, diagrams)
                for drawing in drawings:
                    content = self._process_drawing(
                        drawing, doc, charts, chart_index, processed_images
                    )
                    if drawing.kind == DrawingKind.CHART:
                        chart_index += 1
                    if content:
                        parts.append(content)

                # Process VML pict elements
                for pict in picts:
                    content = self._process_pict(pict, doc, processed_images)
                    if content:
                        parts.append(content)

                # Add text
                if text.strip():
                    parts.append(text.strip())

            elif local == "tbl":
                # Table
                table_data = extract_table(element)
                if table_data is not None:
                    formatted = self._format_table(table_data)
                    if formatted:
                        parts.append(formatted)

            elif local == "sectPr":
                # Section properties — skip (formatting only)
                pass

        result = "\n\n".join(parts)
        # Clean up excessive whitespace
        import re
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """
        Extract all tables as ``TableData`` objects.
        """
        doc = self._get_document(preprocessed)
        if doc is None:
            return []

        tables: List[TableData] = []
        body = doc.element.body
        if body is None:
            return tables

        for element in body:
            if _local_name(element) == "tbl":
                td = extract_table(element)
                if td is not None:
                    tables.append(td)

        return tables

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        """
        Extract all image tags from the document.
        """
        doc = self._get_document(preprocessed)
        if doc is None:
            return []

        tags: List[str] = []
        processed: Set[str] = set()
        body = doc.element.body
        if body is None:
            return tags

        for element in body:
            if _local_name(element) == "p":
                _, drawings, picts, _ = process_paragraph(element)
                for drawing in drawings:
                    if drawing.kind == DrawingKind.IMAGE:
                        tag = self._extract_image_by_rel(
                            drawing.rel_id, doc, processed
                        )
                        if tag:
                            tags.append(tag)
                for pict in picts:
                    tag = self._extract_image_by_rel(
                        pict.rel_id, doc, processed
                    )
                    if tag:
                        tags.append(tag)

        return tags

    def get_format_name(self) -> str:
        return "docx"

    # ── Image processing ──────────────────────────────────────────────────

    def _process_drawing(
        self,
        drawing: DrawingInfo,
        doc: Any,
        charts: List[str],
        chart_index: int,
        processed_images: Set[str],
    ) -> str:
        """Process a drawing element → return content string."""

        if drawing.kind == DrawingKind.IMAGE:
            tag = self._extract_image_by_rel(
                drawing.rel_id, doc, processed_images
            )
            return tag or ""

        if drawing.kind == DrawingKind.CHART:
            if chart_index < len(charts):
                return charts[chart_index]
            return "[Chart]"

        if drawing.kind == DrawingKind.DIAGRAM:
            if drawing.graphic_data is not None:
                return extract_diagram_text(drawing.graphic_data)
            return "[Diagram]"

        return ""

    def _process_pict(
        self,
        pict: PictInfo,
        doc: Any,
        processed_images: Set[str],
    ) -> str:
        """Process a VML pict element → return image tag string."""
        tag = self._extract_image_by_rel(pict.rel_id, doc, processed_images)
        return tag or ""

    def _extract_image_by_rel(
        self,
        rel_id: Optional[str],
        doc: Any,
        processed_images: Set[str],
    ) -> Optional[str]:
        """
        Extract image data from a relationship ID and save via ImageService.

        Deduplicates by rel_id to avoid saving the same image multiple times.
        """
        if not rel_id or self._image_service is None:
            return None

        if rel_id in processed_images:
            return None

        try:
            rel = doc.part.rels.get(rel_id)
            if rel is None:
                return None

            if not hasattr(rel, "target_part") or not hasattr(rel.target_part, "blob"):
                return None

            image_data: bytes = rel.target_part.blob
            if not image_data:
                return None

            # Derive a name from the part name
            custom_name = f"docx_{rel_id}"
            if hasattr(rel.target_part, "partname"):
                partname = str(rel.target_part.partname)
                if "/" in partname:
                    custom_name = partname.split("/")[-1]

            tag = self._image_service.save_and_tag(
                image_bytes=image_data,
                custom_name=custom_name,
            )

            if tag:
                processed_images.add(rel_id)
                return tag

        except Exception as exc:
            logger.debug("Failed to extract image for rel %s: %s", rel_id, exc)

        return None

    # ── Table formatting ──────────────────────────────────────────────────

    def _format_table(self, table_data: TableData) -> str:
        """
        Format a TableData into a string (HTML or text).

        Uses TableService if available, otherwise falls back to
        simple text representation.
        """
        # Special case: 1×1 table (container table) → just return content
        if table_data.num_rows == 1 and table_data.num_cols == 1:
            if table_data.rows and table_data.rows[0]:
                return table_data.rows[0][0].content
            return ""

        # Special case: single column table → line-separated text
        if table_data.num_cols == 1:
            items: List[str] = []
            for row in table_data.rows:
                if row and row[0].content:
                    items.append(row[0].content)
            if items:
                return "\n\n".join(items)
            return ""

        # Use TableService if available
        if self._table_service is not None:
            try:
                return self._table_service.format_table(table_data)
            except Exception as exc:
                logger.debug("TableService formatting failed: %s", exc)

        # Fallback: simple HTML generation
        return self._table_to_html(table_data)

    @staticmethod
    def _table_to_html(table_data: TableData) -> str:
        """Simple HTML table generation as fallback."""
        lines: List[str] = ["<table border='1'>"]

        for row in table_data.rows:
            lines.append("  <tr>")
            for cell in row:
                tag = "th" if cell.is_header else "td"
                attrs = ""
                if cell.row_span > 1:
                    attrs += f" rowspan='{cell.row_span}'"
                if cell.col_span > 1:
                    attrs += f" colspan='{cell.col_span}'"
                content = cell.content.replace("\n", "<br>") if cell.content else ""
                lines.append(f"    <{tag}{attrs}>{content}</{tag}>")
            lines.append("  </tr>")

        lines.append("</table>")
        return "\n".join(lines)

    # ── Page tags ─────────────────────────────────────────────────────────

    def _make_page_tag(self, page_number: int) -> Optional[str]:
        """Generate a page tag using TagService, or None if unavailable."""
        if self._tag_service is not None:
            try:
                return self._tag_service.make_page_tag(page_number)
            except Exception:
                pass
        return None

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_document(preprocessed: PreprocessedData) -> Any:
        """Resolve the python-docx Document from PreprocessedData."""
        content = preprocessed.content
        if hasattr(content, "element") and hasattr(content.element, "body"):
            return content
        raw = preprocessed.raw_content
        if hasattr(raw, "element") and hasattr(raw.element, "body"):
            return raw
        return None


def _local_name(element: Any) -> str:
    """Get the local name of an lxml element (without namespace)."""
    tag = element.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return str(tag)


__all__ = ["DocxContentExtractor"]
