# contextifier/handlers/pdf/preprocessor.py
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
        needs_ocr = self._detect_scan(doc, page_count)

        return PreprocessedData(
            content=doc,              # fitz.Document
            raw_content=file_data,    # original bytes
            encoding="binary",
            resources={"document": doc},
            properties={
                "page_count": page_count,
                "is_encrypted": is_encrypted,
                "needs_ocr": needs_ocr,
            },
        )

    def get_format_name(self) -> str:
        return "pdf"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_scan(doc: Any, page_count: int) -> bool:
        """Sample the first few pages to detect a scanned (image-only) PDF.

        If the average character count per page is below a threshold,
        the document is likely a scan with no text layer.
        """
        if page_count == 0:
            return False

        sample_size = min(page_count, 5)
        _MIN_CHARS_PER_PAGE = 20
        total_chars = 0

        try:
            for idx in range(sample_size):
                page = doc[idx]
                text = page.get_text("text") or ""
                total_chars += len(text.strip())
        except Exception as exc:
            logger.debug("Scan detection sampling failed: %s", exc)
            return False

        avg = total_chars / sample_size
        is_scan = avg < _MIN_CHARS_PER_PAGE
        if is_scan:
            logger.info(
                "PDF appears to be a scan (avg %.1f chars/page across %d sample pages)",
                avg, sample_size,
            )
        return is_scan


__all__ = ["PdfPreprocessor"]
