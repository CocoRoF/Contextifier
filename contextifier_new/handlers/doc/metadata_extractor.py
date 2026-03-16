# contextifier_new/handlers/doc/metadata_extractor.py
"""
DocMetadataExtractor — Stage 3: Extract document metadata from OLE properties.

OLE2 files store standard ``SummaryInformation`` and
``DocumentSummaryInformation`` property sets that ``olefile`` exposes via
``ole.get_metadata()``.  This extractor maps those properties to the
unified ``DocumentMetadata`` dataclass.

Input accepts one of:
- ``DocConvertedData`` (from Converter, has ``.ole``)
- ``PreprocessedData`` (from Preprocessor, has ``.raw_content``)
- Raw ``olefile.OleFileIO`` object

The OLE object must still be open when ``extract()`` is called.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.types import DocumentMetadata, PreprocessedData

from contextifier_new.handlers.doc._constants import OLE_STRING_ENCODINGS
from contextifier_new.handlers.doc.converter import DocConvertedData

logger = logging.getLogger(__name__)


class DocMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for OLE2 DOC files.

    Uses ``olefile.OleFileIO.get_metadata()`` to read OLE property sets.
    """

    # ── BaseMetadataExtractor abstract methods ────────────────────────────

    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from the OLE compound file.

        Args:
            source: One of ``DocConvertedData``, ``PreprocessedData``,
                    or a raw ``olefile.OleFileIO`` object.

        Returns:
            Populated ``DocumentMetadata``.
        """
        ole = self._unwrap_ole(source)
        if ole is None:
            logger.debug("No OLE object found for metadata extraction")
            return DocumentMetadata()

        try:
            return self._extract_ole_metadata(ole)
        except Exception as exc:
            logger.warning("OLE metadata extraction failed: %s", exc)
            return DocumentMetadata()

    def get_format_name(self) -> str:
        return "doc"

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _unwrap_ole(source: Any) -> Any:
        """
        Resolve the ``olefile.OleFileIO`` object from various input types.

        Returns:
            The OLE object, or *None* if it cannot be found.
        """
        # DocConvertedData (from converter)
        if isinstance(source, DocConvertedData):
            return source.ole

        # PreprocessedData (from preprocessor) — raw_content holds DocConvertedData
        if isinstance(source, PreprocessedData):
            raw = source.raw_content
            if isinstance(raw, DocConvertedData):
                return raw.ole
            if hasattr(raw, "ole"):
                return raw.ole
            # Maybe raw is the OLE object itself
            if hasattr(raw, "get_metadata"):
                return raw
            return None

        # Direct olefile.OleFileIO
        if hasattr(source, "get_metadata"):
            return source

        return None

    def _extract_ole_metadata(self, ole: Any) -> DocumentMetadata:
        """Extract metadata fields from OLE property sets."""
        meta = ole.get_metadata()
        if meta is None:
            return DocumentMetadata()

        title = self._decode_ole_value(meta.title)
        subject = self._decode_ole_value(meta.subject)
        author = self._decode_ole_value(meta.author)
        keywords = self._decode_ole_value(meta.keywords)
        comments = self._decode_ole_value(meta.comments)
        last_saved_by = self._decode_ole_value(meta.last_saved_by)

        # Dates — olefile returns datetime or None
        create_time = self._to_datetime(meta.create_time)
        last_saved_time = self._to_datetime(meta.last_saved_time)

        # Page / word counts (from DocumentSummaryInformation)
        page_count: Optional[int] = None
        word_count: Optional[int] = None
        if hasattr(meta, "num_pages") and meta.num_pages:
            page_count = int(meta.num_pages)
        if hasattr(meta, "num_words") and meta.num_words:
            word_count = int(meta.num_words)

        # Revision number
        revision: Optional[str] = None
        if hasattr(meta, "revision_number") and meta.revision_number:
            revision = self._decode_ole_value(meta.revision_number)

        return DocumentMetadata(
            title=title or None,
            subject=subject or None,
            author=author or None,
            keywords=keywords or None,
            comments=comments or None,
            last_saved_by=last_saved_by or None,
            create_time=create_time,
            last_saved_time=last_saved_time,
            page_count=page_count,
            word_count=word_count,
            revision=revision or None,
        )

    # ── Value decoding ────────────────────────────────────────────────────

    @staticmethod
    def _decode_ole_value(value: Any) -> str:
        """
        Decode an OLE property value to a Python ``str``.

        olefile may return ``str``, ``bytes``, or ``None``.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bytes):
            for enc in OLE_STRING_ENCODINGS:
                try:
                    return value.decode(enc).strip()
                except (UnicodeDecodeError, UnicodeError):
                    continue
            # Last resort: replace errors
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()

    @staticmethod
    def _to_datetime(value: Any) -> Optional[datetime]:
        """Convert an OLE date/time value to ``datetime`` or *None*."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        # olefile sometimes returns int timestamps
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value)
            except (OSError, OverflowError, ValueError):
                return None
        return None


__all__ = ["DocMetadataExtractor"]
