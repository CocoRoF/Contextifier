# contextifier/handlers/xls/content_extractor.py
"""
XlsContentExtractor — extract text / tables / images from XLS (BIFF) files.

Text and tables are extracted via xlrd.  Images are extracted by
scanning the raw OLE2 compound file for known image signatures
(reusing the DOC handler pattern).

Charts are detected as chart-type sheets via xlrd's sheet type
information (sheet type ``XL_CHART_SHEET = 2``).
"""

from __future__ import annotations

import io
import hashlib
import logging
from typing import Any, List, Optional, Set

import olefile

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    ChartData,
    ExtractionResult,
    PreprocessedData,
    TableData,
)

from contextifier.handlers.xls._layout import (
    LayoutRange,
    layout_detect_range,
    object_detect,
)
from contextifier.handlers.xls._table import (
    convert_region_to_table,
    convert_sheet_to_text,
)

logger = logging.getLogger(__name__)

# Image signatures for OLE stream scanning (same as DOC handler)
_IMAGE_SIGNATURES: dict[str, tuple[bytes, int]] = {
    "png":     (b"\x89PNG\r\n\x1a\n", 8),
    "jpeg":    (b"\xff\xd8",           2),
    "gif87":   (b"GIF87a",             6),
    "gif89":   (b"GIF89a",             6),
    "bmp":     (b"BM",                 2),
    "tiff_le": (b"II\x2a\x00",        4),
    "tiff_be": (b"MM\x00\x2a",        4),
    "emf":     (b"\x01\x00\x00\x00",  4),
}

# OLE stream keywords that may contain images
_IMAGE_STREAM_KEYWORDS: frozenset[str] = frozenset({
    "pictures", "data", "object", "oleobject", "objectpool",
    "mbd", "workbook",  # Some images are embedded in Drawing records
})

# BIFF sheet type for charts
_XL_CHART_SHEET = 2


class XlsContentExtractor(BaseContentExtractor):
    """Extract text, tables, images, and charts from XLS workbooks."""

    def __init__(
        self,
        *,
        image_service: Any = None,
        tag_service: Any = None,
        table_service: Any = None,
        **kwargs: Any,
    ) -> None:
        self._image_service = image_service
        self._tag_service = tag_service
        self._table_service = table_service

    def get_format_name(self) -> str:
        return "xls"

    # ── text ─────────────────────────────────────────────────────────────

    def extract_text(self, preprocessed: PreprocessedData, **kw: Any) -> str:
        book = self._get_book(preprocessed)
        if book is None:
            return ""

        parts: List[str] = []

        for idx in range(book.nsheets):
            ws = book.sheet_by_index(idx)
            sheet_tag = self._make_sheet_tag(ws.name)
            parts.append(f"\n{sheet_tag}\n")

            regions = object_detect(ws, book)
            if not regions:
                continue

            for i, region in enumerate(regions, 1):
                text = convert_sheet_to_text(ws, book, region)
                if text:
                    if len(regions) > 1:
                        parts.append(f"\n[Table {i}]\n{text}\n")
                    else:
                        parts.append(f"\n{text}\n")

        return "".join(parts)

    # ── tables ───────────────────────────────────────────────────────────

    def extract_tables(self, preprocessed: PreprocessedData, **kw: Any) -> List[TableData]:
        book = self._get_book(preprocessed)
        if book is None:
            return []

        tables: List[TableData] = []
        for idx in range(book.nsheets):
            ws = book.sheet_by_index(idx)
            regions = object_detect(ws, book)
            for region in regions:
                td = convert_region_to_table(ws, book, region)
                if td is not None:
                    tables.append(td)
        return tables

    # ── images (OLE stream scanning) ────────────────────────────────────

    def extract_images(self, preprocessed: PreprocessedData, **kw: Any) -> List[str]:
        """
        Extract images by scanning OLE2 streams for known signatures.

        Opens the raw file bytes as an OLE compound file and scans
        streams with image-related names for PNG, JPEG, GIF, BMP,
        TIFF, and EMF signatures.

        Returns:
            List of image tag strings.
        """
        if self._image_service is None:
            return []

        file_data = preprocessed.resources.get("file_data", b"")
        if not file_data:
            return []

        try:
            ole = olefile.OleFileIO(io.BytesIO(file_data))
        except Exception:
            return []

        tags: List[str] = []
        processed: Set[str] = set()

        try:
            for entry in ole.listdir():
                entry_path = "/".join(entry)
                if not any(
                    kw in part.lower()
                    for part in entry
                    for kw in _IMAGE_STREAM_KEYWORDS
                ):
                    continue

                try:
                    data = ole.openstream(entry).read()
                except Exception:
                    continue

                fmt = _detect_image_format(data)
                if fmt is None:
                    continue

                content_hash = hashlib.md5(data).hexdigest()
                if content_hash in processed:
                    continue
                processed.add(content_hash)

                try:
                    tag = self._image_service.save_and_tag(
                        image_bytes=data,
                        custom_name=f"xls_ole_{content_hash[:12]}",
                    )
                    if tag:
                        tags.append(tag)
                except Exception as exc:
                    logger.debug("Failed to save XLS OLE image: %s", exc)
        finally:
            try:
                ole.close()
            except Exception:
                pass

        return tags

    # ── charts (BIFF sheet type detection) ───────────────────────────────

    def extract_charts(self, preprocessed: PreprocessedData, **kw: Any) -> List[ChartData]:
        """
        Detect chart sheets in the XLS workbook.

        xlrd identifies chart sheets by their sheet type.
        Full BIFF chart record parsing is extremely complex;
        we report chart presence with available metadata.

        Returns:
            List of ChartData with sheet name and type.
        """
        book = self._get_book(preprocessed)
        if book is None:
            return []

        charts: List[ChartData] = []
        for idx in range(book.nsheets):
            try:
                sheet = book.sheet_by_index(idx)
                # xlrd Sheet objects have a `sheet_type` attribute in
                # the internal book.sheet_types list
                sheet_type = book.sheet_type(idx) if hasattr(book, "sheet_type") else None
            except Exception:
                continue

            if sheet_type == _XL_CHART_SHEET:
                charts.append(ChartData(
                    chart_type="biff_chart_sheet",
                    title=sheet.name,
                    raw_content=f"BIFF chart sheet: {sheet.name}",
                ))

        return charts

    # ── helpers ──────────────────────────────────────────────────────────

    def _get_book(self, preprocessed: PreprocessedData) -> Any:
        if preprocessed is None:
            return None
        content = preprocessed.content if isinstance(preprocessed, PreprocessedData) else preprocessed
        if content is None:
            return None
        if hasattr(content, "nsheets"):
            return content
        return None

    def _make_sheet_tag(self, name: str) -> str:
        if self._tag_service is not None:
            try:
                return self._tag_service.make_sheet_tag(name)
            except Exception:
                pass
        return f"[Sheet: {name}]"


# ── Module-level helpers ─────────────────────────────────────────────────────

def _detect_image_format(data: bytes) -> Optional[str]:
    """Detect image format from binary data using header signatures."""
    if not data or len(data) < 2:
        return None
    for fmt_name, (signature, min_len) in _IMAGE_SIGNATURES.items():
        if len(data) >= min_len and data[: len(signature)] == signature:
            if fmt_name.startswith("gif"):
                return "gif"
            if fmt_name.startswith("tiff"):
                return "tiff"
            return fmt_name
    return None


__all__ = ["XlsContentExtractor"]
