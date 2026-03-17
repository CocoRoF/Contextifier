# contextifier_new/handlers/xls/preprocessor.py
"""
XlsPreprocessor — unwrap XlsConvertedData, record sheet info.

XLS preprocessing is lightweight: the Book object is already usable,
and we just populate properties / resources for downstream stages.
"""

from __future__ import annotations

import logging
from typing import Any

from contextifier.errors import PreprocessingError
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData

from contextifier.handlers.xls.converter import XlsConvertedData

logger = logging.getLogger(__name__)


class XlsPreprocessor(BasePreprocessor):
    """Lightweight preprocessor for XLS files."""

    def get_format_name(self) -> str:
        return "xls"

    def preprocess(self, converted: Any, **kwargs: Any) -> PreprocessedData:
        if converted is None:
            raise PreprocessingError("None input", stage="preprocess", handler="xls")

        # Unwrap
        book: Any = None
        file_data: bytes = b""

        if isinstance(converted, XlsConvertedData):
            book = converted.book
            file_data = converted.file_data
        else:
            # Bare xlrd.Book
            book = converted

        sheet_names = [book.sheet_by_index(i).name for i in range(book.nsheets)]

        return PreprocessedData(
            content=book,
            raw_content=book,
            encoding="biff",
            resources={
                "file_data": file_data,
            },
            properties={
                "sheet_count": book.nsheets,
                "sheet_names": sheet_names,
            },
        )


__all__ = ["XlsPreprocessor"]
