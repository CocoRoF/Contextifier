# contextifier_new/handlers/pdf/metadata_extractor.py
"""
PdfMetadataExtractor — Stage 3: read PDF metadata

Shared by both pdf_default and pdf_plus modes.
Reads ``fitz.Document.metadata`` (title, author, subject, keywords,
creator, producer, creation/modification dates) and maps them to
``DocumentMetadata``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata, PreprocessedData
from contextifier.handlers.pdf._constants import parse_pdf_date

logger = logging.getLogger(__name__)


class PdfMetadataExtractor(BaseMetadataExtractor):
    """Extract metadata from a PDF document via PyMuPDF."""

    def extract(self, content: Any, **kwargs: Any) -> DocumentMetadata:
        meta = DocumentMetadata()
        doc = self._unwrap(content)
        if doc is None:
            return meta

        pdf_meta: dict = getattr(doc, "metadata", None) or {}

        meta.title = self._str(pdf_meta.get("title"))
        meta.subject = self._str(pdf_meta.get("subject"))
        meta.author = self._str(pdf_meta.get("author"))
        meta.keywords = self._str(pdf_meta.get("keywords"))
        meta.page_count = getattr(doc, "page_count", None)

        create_date = parse_pdf_date(pdf_meta.get("creationDate"))
        mod_date = parse_pdf_date(pdf_meta.get("modDate"))
        if create_date:
            meta.create_time = create_date.strftime("%Y-%m-%d %H:%M:%S")
        if mod_date:
            meta.last_saved_time = mod_date.strftime("%Y-%m-%d %H:%M:%S")

        # Extra fields → custom dict
        custom = {}
        for key in ("creator", "producer", "format"):
            val = self._str(pdf_meta.get(key))
            if val:
                custom[key] = val
        if pdf_meta.get("encryption"):
            custom["encryption"] = pdf_meta["encryption"]
        if custom:
            meta.custom = custom

        return meta

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(content: Any) -> Any:
        """Accept fitz.Document or PreprocessedData."""
        if isinstance(content, PreprocessedData):
            return content.content
        if hasattr(content, "metadata") and hasattr(content, "page_count"):
            return content
        return None

    @staticmethod
    def _str(value: Any) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    def get_format_name(self) -> str:
        return "pdf"


__all__ = ["PdfMetadataExtractor"]
