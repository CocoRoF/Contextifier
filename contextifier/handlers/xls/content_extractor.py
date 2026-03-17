# contextifier_new/handlers/xls/content_extractor.py
"""
XlsContentExtractor — extract text / tables from XLS (xlrd) files.

Charts and images are NOT extracted from legacy .xls files (BIFF charts
are not OOXML charts, and embedded OLE images lack standardised access).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

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


class XlsContentExtractor(BaseContentExtractor):
    """Extract text and tables from XLS workbooks."""

    def __init__(
        self,
        *,
        tag_service: Any = None,
        table_service: Any = None,
        **kwargs: Any,
    ) -> None:
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

    # ── images / charts (not supported for BIFF) ────────────────────────

    def extract_images(self, preprocessed: PreprocessedData, **kw: Any) -> List[str]:
        return []

    def extract_charts(self, preprocessed: PreprocessedData, **kw: Any) -> List[ChartData]:
        return []

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


__all__ = ["XlsContentExtractor"]
