# contextifier/handlers/xls/metadata_extractor.py
"""
XlsMetadataExtractor — extract metadata from XLS (BIFF) files.

Tries two sources:
1. OLE SummaryInformation via olefile (title, author, keywords, dates …)
2. xlrd user_name (author fallback)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata, PreprocessedData

import olefile

from contextifier.handlers.xls.converter import XlsConvertedData

logger = logging.getLogger(__name__)


class XlsMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata from an XLS file via OLE properties + xlrd."""

    def get_format_name(self) -> str:
        return "xls"

    # ── public API ──────────────────────────────────────────────────────

    def extract(self, source: Any) -> DocumentMetadata:
        book, file_data = self._unwrap(source)
        if book is None:
            return DocumentMetadata()

        meta = DocumentMetadata()

        # 1. Try OLE SummaryInformation (richer)
        if file_data:
            meta = self._extract_from_ole(file_data)

        # 2. Fallback: xlrd user_name as author
        if not meta.author and book is not None:
            try:
                un = getattr(book, "user_name", None)
                if un:
                    meta.author = str(un).strip() or None
            except Exception:
                pass

        # 3. page_count = number of sheets
        try:
            meta.page_count = book.nsheets
        except Exception:
            pass

        return meta

    # ── unwrap helpers ──────────────────────────────────────────────────

    def _unwrap(self, source: Any):
        """Return (book, file_data) from various source shapes."""
        if source is None:
            return None, b""
        if isinstance(source, XlsConvertedData):
            return source.book, source.file_data
        if isinstance(source, PreprocessedData):
            fd = source.resources.get("file_data", b"") if source.resources else b""
            return source.content, fd
        # Bare xlrd.Book
        if hasattr(source, "nsheets"):
            return source, b""
        return None, b""

    # ── OLE extraction ──────────────────────────────────────────────────

    def _extract_from_ole(self, file_data: bytes) -> DocumentMetadata:
        try:
            ole = olefile.OleFileIO(file_data)
        except Exception:
            return DocumentMetadata()

        try:
            meta_obj = ole.get_metadata()

            title = self._safe_str(getattr(meta_obj, "title", None))
            subject = self._safe_str(getattr(meta_obj, "subject", None))
            author = self._safe_str(getattr(meta_obj, "author", None))
            keywords = self._safe_str(getattr(meta_obj, "keywords", None))
            comments = self._safe_str(getattr(meta_obj, "comments", None))
            last_saved_by = self._safe_str(getattr(meta_obj, "last_saved_by", None))
            create_time = self._safe_datetime(getattr(meta_obj, "create_time", None))
            last_saved_time = self._safe_datetime(getattr(meta_obj, "last_saved_time", None))
            category = self._safe_str(getattr(meta_obj, "category", None))
            revision = self._safe_str(getattr(meta_obj, "revision_number", None))

            return DocumentMetadata(
                title=title,
                subject=subject,
                author=author,
                keywords=keywords,
                comments=comments,
                last_saved_by=last_saved_by,
                create_time=create_time,
                last_saved_time=last_saved_time,
                category=category,
                revision=revision,
            )
        except Exception as exc:
            logger.debug("OLE metadata extraction failed: %s", exc)
            return DocumentMetadata()
        finally:
            try:
                ole.close()
            except Exception:
                pass

    # ── safe‐cast helpers ───────────────────────────────────────────────

    @staticmethod
    def _safe_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception:
                return None
        s = str(value).strip()
        return s if s else None

    @staticmethod
    def _safe_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        return None


__all__ = ["XlsMetadataExtractor"]
