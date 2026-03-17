"""
PptMetadataExtractor — Stage 3: OLE2 metadata → DocumentMetadata.

Reads standard OLE2 summary information properties
(title, author, subject, dates, etc.) from the compound file.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata, PreprocessedData

from contextifier.handlers.ppt.converter import PptConvertedData

logger = logging.getLogger(__name__)


class PptMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for genuine OLE2 PPT files.

    Reads the ``\\x05SummaryInformation`` stream via olefile's
    ``get_metadata()`` method.
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from an OLE2 PPT file.

        ``source`` may be:
        - ``PptConvertedData`` namedtuple
        - ``olefile.OleFileIO`` directly
        - ``PreprocessedData`` wrapping an OLE object
        """
        ole = self._unwrap_ole(source)
        if ole is None:
            self._logger.warning("Cannot extract PPT metadata: no OLE object found")
            return DocumentMetadata()

        try:
            meta = ole.get_metadata()
            return DocumentMetadata(
                title=self._decode(meta.title),
                subject=self._decode(meta.subject),
                author=self._decode(meta.author),
                keywords=self._decode(meta.keywords),
                comments=self._decode(meta.comments),
                last_saved_by=self._decode(meta.last_saved_by),
                create_time=self._to_datetime(meta.create_time),
                last_saved_time=self._to_datetime(meta.last_saved_time),
                revision=str(meta.revision_number) if meta.revision_number else None,
                category=self._decode(meta.category) if hasattr(meta, "category") else None,
            )
        except Exception as exc:
            self._logger.warning("Failed to extract PPT metadata: %s", exc)
            return DocumentMetadata()

    def get_format_name(self) -> str:
        return "ppt"

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_ole(source: Any) -> Any:
        """Resolve olefile.OleFileIO from various input types."""
        import olefile

        def _is_ole(obj: Any) -> bool:
            try:
                return isinstance(obj, olefile.OleFileIO)
            except TypeError:
                return False

        # Direct OLE
        if _is_ole(source):
            return source
        # PptConvertedData
        if isinstance(source, PptConvertedData):
            return source.ole
        # PreprocessedData
        if isinstance(source, PreprocessedData):
            for attr in ("content", "raw_content"):
                obj = getattr(source, attr, None)
                if _is_ole(obj):
                    return obj
            # Fallback: if content has get_metadata, it's likely an OLE-like
            content = getattr(source, "content", None)
            if content is not None and hasattr(content, "get_metadata"):
                return content
        # Direct OLE-like object with get_metadata (e.g. from preprocessed.content)
        if hasattr(source, "get_metadata") and callable(getattr(source, "get_metadata", None)):
            return source
        # Has .ole attribute
        if hasattr(source, "ole"):
            return source.ole
        return None

    @staticmethod
    def _decode(value: Any) -> Optional[str]:
        """Decode an OLE metadata field to string."""
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8")
            except UnicodeDecodeError:
                decoded = value.decode("cp1252", errors="replace")
            return decoded.strip() if decoded.strip() else None
        s = str(value).strip()
        return s if s else None

    @staticmethod
    def _to_datetime(value: Any) -> Optional[datetime]:
        """Convert an OLE datetime value to Python datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return None


__all__ = ["PptMetadataExtractor"]
