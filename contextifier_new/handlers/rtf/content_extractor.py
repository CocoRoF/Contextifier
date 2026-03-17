# contextifier_new/handlers/rtf/content_extractor.py
"""
RtfContentExtractor — Stage 4: Text + Table + Image extraction from RTF.

Takes PreprocessedData (containing RtfParsedData) and extracts:

1. **Tables** — via ``_table_parser.extract_tables_with_positions()``
   Produces standard ``TableData`` / ``TableCell`` instances.
   Formatted by ``TableService`` and inserted at original document positions.

2. **Text** — via inline content extraction
   Cleans RTF control codes, decodes hex escapes, removes
   destination groups (fonttbl, colortbl, stylesheet), removes
   shape/property groups, and excludes header/footer/footnote regions.
   Tables are embedded at their original positions in the text.

3. **Images** — from PreprocessedData.resources["images"]
   Each ``RtfImageData`` is saved via ``ImageService.save_and_tag()``
   and the tag is appended to the text output.

Fallback: if primary extraction produces no text, try ``striprtf``
library for basic text extraction (lossy, no tables/images).

Ported from v1.0 rtf_content_extractor.py and rtf_handler.py with:
- Produces TableData/TableCell instead of HTML strings
- Uses TableService for table formatting
- Uses ImageService for image saving
- Conforms to BaseContentExtractor interface
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.types import (
    ExtractionResult,
    PreprocessedData,
    TableData,
)
from contextifier_new.handlers.rtf.preprocessor import RtfParsedData, RtfImageData
from contextifier_new.handlers.rtf._decoder import decode_hex_escapes
from contextifier_new.handlers.rtf._cleaner import (
    clean_rtf_text,
    find_excluded_regions,
    remove_destination_groups,
    remove_shape_groups,
    remove_shape_property_groups,
    remove_shprslt_blocks,
)
from contextifier_new.handlers.rtf._table_parser import (
    extract_tables_with_positions,
    single_column_to_text,
)

if TYPE_CHECKING:
    from contextifier_new.services.image_service import ImageService
    from contextifier_new.services.tag_service import TagService
    from contextifier_new.services.chart_service import ChartService
    from contextifier_new.services.table_service import TableService

_logger = logging.getLogger("contextifier.rtf.content")


class RtfContentExtractor(BaseContentExtractor):
    """
    RTF-specific content extractor.

    Orchestrates:
    - RTF → clean text (via _cleaner / _decoder)
    - Table extraction (via _table_parser)
    - Image saving (via ImageService)
    - Inline content assembly (text + tables at original positions)
    """

    # ── Required: extract_text ────────────────────────────────────────────

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from RTF with tables inline at original positions.

        Steps:
        1. Extract tables with positions.
        2. Build inline content (text interleaved with formatted tables).
        3. Save images and append image tags.
        4. If no result, fall back to striprtf.
        """
        parsed, encoding = self._unpack(preprocessed)
        if not parsed:
            return ""

        content = parsed.text

        # Extract tables with positions
        _, table_regions = extract_tables_with_positions(content, encoding)

        # Build inline content
        text = self._build_inline_content(content, table_regions, encoding)

        # Save images and collect tags
        image_tags = self._save_images(preprocessed)
        if image_tags:
            text = text.rstrip() + "\n\n" + "\n".join(image_tags)

        # Fallback: striprtf
        if not text.strip():
            text = self._fallback_striprtf(parsed)

        return text.strip()

    def get_format_name(self) -> str:
        return "rtf"

    # ── Override: extract_tables ──────────────────────────────────────────

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """
        Extract tables as structured TableData.

        These are also used in extract_text() for inline positioning,
        but returned separately for consumers that need raw table data.
        """
        parsed, encoding = self._unpack(preprocessed)
        if not parsed:
            return []

        _, table_regions = extract_tables_with_positions(
            parsed.text, encoding,
        )
        return [table for _, _, table in table_regions]

    # ── Override: extract_images ──────────────────────────────────────────

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        """
        Save extracted images via ImageService and return tag strings.

        Images are pre-extracted by the Preprocessor and stored
        in ``preprocessed.resources["images"]``.
        """
        return self._save_images(preprocessed)

    # ── Internal: Inline content assembly ─────────────────────────────────

    def _build_inline_content(
        self,
        content: str,
        table_regions: List[Tuple[int, int, TableData]],
        encoding: str,
    ) -> str:
        """
        Build text with tables inserted at their original positions.

        The flow:
        1. Find header end (first ``\\pard``)
        2. Walk through content, inserting cleaned text segments and
           formatted table blocks at the correct positions.
        """
        header_end = self._find_header_end(content)
        excluded = find_excluded_regions(content)

        if not table_regions:
            clean = self._clean_segment(
                content[header_end:], header_end, encoding, excluded,
            )
            return clean.strip()

        # Adjust table regions to start after header
        adjusted = [
            (max(s, header_end), e, t)
            for s, e, t in table_regions
            if e > header_end
        ]

        parts: List[str] = []
        last_end = header_end

        for start_pos, end_pos, table in adjusted:
            # Text before this table
            if start_pos > last_end:
                segment = content[last_end:start_pos]
                clean = self._clean_segment(
                    segment, last_end, encoding, excluded,
                )
                if clean.strip():
                    parts.append(clean)

            # Table
            formatted = self._format_table(table)
            if formatted:
                parts.append(formatted)

            last_end = end_pos

        # Text after last table
        if last_end < len(content):
            segment = content[last_end:]
            clean = self._clean_segment(
                segment, last_end, encoding, excluded,
            )
            if clean.strip():
                parts.append(clean)

        return "\n\n".join(parts)

    def _clean_segment(
        self,
        segment: str,
        start_pos: int,
        encoding: str,
        excluded_regions: List[Tuple[int, int]],
    ) -> str:
        """
        Clean an RTF text segment: remove destinations, shapes,
        decode hex escapes, strip control codes.

        Respects excluded regions (header/footer/footnote).
        """
        if not excluded_regions:
            return self._clean_raw(segment, encoding)

        result_parts: List[str] = []
        seg_pos = 0

        for excl_start, excl_end in excluded_regions:
            rel_start = excl_start - start_pos
            rel_end = excl_end - start_pos

            if rel_end <= 0 or rel_start >= len(segment):
                continue

            rel_start = max(0, rel_start)
            rel_end = min(len(segment), rel_end)

            if rel_start > seg_pos:
                part = segment[seg_pos:rel_start]
                clean = self._clean_raw(part, encoding)
                if clean.strip():
                    result_parts.append(clean)

            seg_pos = rel_end

        if seg_pos < len(segment):
            part = segment[seg_pos:]
            clean = self._clean_raw(part, encoding)
            if clean.strip():
                result_parts.append(clean)

        return " ".join(result_parts)

    @staticmethod
    def _clean_raw(segment: str, encoding: str) -> str:
        """Clean a single RTF text segment."""
        segment = remove_destination_groups(segment)
        segment = remove_shape_groups(segment)
        segment = remove_shape_property_groups(segment)
        segment = remove_shprslt_blocks(segment)
        decoded = decode_hex_escapes(segment, encoding)
        return clean_rtf_text(decoded, encoding)

    @staticmethod
    def _find_header_end(content: str) -> int:
        """Find the end of the RTF header (first \\pard)."""
        match = re.search(r"\\pard\b", content)
        return match.start() if match else 0

    # ── Internal: Table formatting ────────────────────────────────────────

    def _format_table(self, table: TableData) -> str:
        """
        Format a TableData using TableService if available,
        otherwise render as plain HTML.
        """
        if self._table_service is not None:
            try:
                return self._table_service.format_table(table)
            except Exception as e:
                _logger.warning("TableService failed, using fallback: %s", e)

        # Fallback: simple HTML rendering
        return self._table_to_html_fallback(table)

    @staticmethod
    def _table_to_html_fallback(table: TableData) -> str:
        """Minimal HTML table rendering without TableService."""
        if not table.rows:
            return ""

        lines = ['<table border="1">']
        for row_cells in table.rows:
            row_parts = []
            for cell in row_cells:
                attrs = ""
                if cell.col_span > 1:
                    attrs += f' colspan="{cell.col_span}"'
                if cell.row_span > 1:
                    attrs += f' rowspan="{cell.row_span}"'
                row_parts.append(f"<td{attrs}>{cell.content}</td>")
            lines.append(f"<tr>{''.join(row_parts)}</tr>")
        lines.append("</table>")
        return "\n".join(lines)

    # ── Internal: Image saving ────────────────────────────────────────────

    def _save_images(self, preprocessed: PreprocessedData) -> List[str]:
        """
        Save images from preprocessed.resources via ImageService.

        Returns list of image tag strings.
        """
        if self._image_service is None:
            return []

        images: List[RtfImageData] = (
            preprocessed.resources.get("images", [])
            if preprocessed.resources
            else []
        )
        if not images:
            return []

        tags: List[str] = []
        for img in images:
            try:
                tag = self._image_service.save_and_tag(
                    img.image_bytes,
                    custom_name=f"rtf_img_{img.content_hash[:8]}.{img.image_format}",
                )
                if tag:
                    tags.append(tag)
            except Exception as e:
                _logger.warning("Failed to save RTF image: %s", e)

        return tags

    # ── Internal: Fallback ────────────────────────────────────────────────

    @staticmethod
    def _fallback_striprtf(parsed: RtfParsedData) -> str:
        """
        Fallback text extraction using striprtf library.

        Used when primary extraction produces no text.
        Lossy: no table structure, no images.
        """
        try:
            from striprtf.striprtf import rtf_to_text  # type: ignore
            text = rtf_to_text(parsed.text)
            if text:
                _logger.info("Using striprtf fallback (%d chars)", len(text))
            return text or ""
        except ImportError:
            _logger.debug("striprtf not installed, no fallback available")
            return ""
        except Exception as e:
            _logger.warning("striprtf fallback failed: %s", e)
            return ""

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(
        preprocessed: PreprocessedData,
    ) -> Tuple[Optional[RtfParsedData], str]:
        """Extract RtfParsedData and encoding from PreprocessedData."""
        content = preprocessed.content
        if isinstance(content, RtfParsedData):
            return content, content.encoding
        if isinstance(content, str):
            return RtfParsedData(text=content, encoding="cp949", image_count=0), "cp949"
        return None, "cp949"


__all__ = ["RtfContentExtractor"]
