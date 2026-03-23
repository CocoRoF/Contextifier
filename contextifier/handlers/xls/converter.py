# contextifier/handlers/xls/converter.py
"""
XlsConverter — validate & open .xls files via xlrd.

Returns XlsConvertedData(book, file_data) so downstream stages can
access both the xlrd Book and raw bytes (for OLE metadata).
"""

from __future__ import annotations

import io
import logging
from typing import Any, NamedTuple

from contextifier.errors import ConversionError
from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext

import xlrd

from contextifier.handlers.xls._constants import OLE2_MAGIC

logger = logging.getLogger(__name__)


class XlsConvertedData(NamedTuple):
    """Wrapper returned by XlsConverter.convert()."""
    book: Any           # xlrd.Book
    file_data: bytes    # original bytes (for OLE metadata)


class XlsConverter(BaseConverter):
    """Open a .xls (BIFF) file with xlrd."""

    def get_format_name(self) -> str:
        return "xls"

    # ── validation ──────────────────────────────────────────────────────

    def validate(self, file_context: FileContext) -> bool:
        data: bytes = file_context.get("file_data", b"")
        if not data or len(data) < 8:
            return False
        return data[:8] == OLE2_MAGIC

    # ── conversion ──────────────────────────────────────────────────────

    def convert(self, file_context: FileContext) -> XlsConvertedData:
        data: bytes = file_context.get("file_data", b"")
        if not data:
            raise ConversionError("Empty file data", stage="convert", handler="xls")

        if len(data) < 8 or data[:8] != OLE2_MAGIC:
            raise ConversionError(
                "Not a valid OLE2/XLS file (bad magic bytes)",
                stage="convert",
                handler="xls",
            )

        try:
            book = xlrd.open_workbook(file_contents=data, formatting_info=True)
        except Exception as exc:
            raise ConversionError(
                f"Failed to open XLS file: {exc}",
                stage="convert",
                handler="xls",
            ) from exc

        return XlsConvertedData(book=book, file_data=data)

    # ── cleanup ─────────────────────────────────────────────────────────

    def close(self, converted: Any) -> None:
        if converted is None:
            return
        book = None
        if isinstance(converted, XlsConvertedData):
            book = converted.book
        elif hasattr(converted, "release_resources"):
            book = converted
        if book is not None:
            try:
                book.release_resources()
            except Exception:
                pass


__all__ = ["XlsConverter", "XlsConvertedData"]
