# contextifier_new/handlers/pdf/preprocessor.py
"""
PdfPreprocessor — Stage 2: pass-through with basic metadata

Shared by both pdf_default and pdf_plus modes.
The fitz.Document is already in a workable state after conversion;
the preprocessor simply wraps it in ``PreprocessedData`` and stores
useful metadata (page_count, encrypted flag) in ``properties``.
"""

from __future__ import annotations

import logging
from typing import Any

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.handlers.pdf.converter import PdfConvertedData

logger = logging.getLogger(__name__)


class PdfPreprocessor(BasePreprocessor):
    """
    PDF preprocessor — mostly pass-through.

    Stores ``page_count`` and ``is_encrypted`` in *properties* for
    downstream stages.
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        if isinstance(converted_data, PdfConvertedData):
            doc = converted_data.doc
            file_data = converted_data.file_data
        elif hasattr(converted_data, "page_count"):
            # Accept a bare fitz.Document for convenience
            doc = converted_data
            file_data = b""
        else:
            doc = converted_data
            file_data = b""

        page_count = getattr(doc, "page_count", 0)
        is_encrypted = getattr(doc, "is_encrypted", False)

        return PreprocessedData(
            content=doc,              # fitz.Document
            raw_content=file_data,    # original bytes
            encoding="binary",
            resources={"document": doc},
            properties={
                "page_count": page_count,
                "is_encrypted": is_encrypted,
            },
        )

    def get_format_name(self) -> str:
        return "pdf"


__all__ = ["PdfPreprocessor"]
