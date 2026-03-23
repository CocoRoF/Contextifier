"""
DocxMetadataExtractor — Stage 3: Extract metadata from DOCX core properties.

python-docx exposes ``Document.core_properties`` which maps to the
``docProps/core.xml`` file inside the DOCX ZIP.  This extractor reads
those properties and maps them to the unified ``DocumentMetadata``
dataclass.

Supported properties:
- title, subject, author, keywords, comments
- last_modified_by (last_saved_by)
- created, modified (create_time, last_saved_time)
- revision
- category
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata, PreprocessedData

logger = logging.getLogger(__name__)


class DocxMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for DOCX files.

    Uses ``python_docx.Document.core_properties`` to read standard
    OOXML metadata fields.
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from the python-docx Document.

        Args:
            source: One of:
                    - ``python_docx.Document`` (from Converter)
                    - ``PreprocessedData`` (from Preprocessor)

        Returns:
            Populated ``DocumentMetadata``.
        """
        doc = self._unwrap_document(source)
        if doc is None:
            logger.debug("No Document object found for metadata extraction")
            return DocumentMetadata()

        try:
            return self._extract_core_properties(doc)
        except Exception as exc:
            logger.warning("DOCX metadata extraction failed: %s", exc)
            return DocumentMetadata()

    def get_format_name(self) -> str:
        return "docx"

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _unwrap_document(source: Any) -> Any:
        """
        Resolve the python-docx Document from various input types.
        """
        # Direct Document object
        if hasattr(source, "core_properties"):
            return source

        # PreprocessedData — content or raw_content holds the Document
        if isinstance(source, PreprocessedData):
            if hasattr(source.content, "core_properties"):
                return source.content
            if hasattr(source.raw_content, "core_properties"):
                return source.raw_content
            return None

        return None

    def _extract_core_properties(self, doc: Any) -> DocumentMetadata:
        """Extract metadata from Document.core_properties."""
        props = doc.core_properties

        title = self._safe_str(props.title)
        subject = self._safe_str(props.subject)
        author = self._safe_str(props.author)
        keywords = self._safe_str(props.keywords)
        comments = self._safe_str(props.comments)
        last_saved_by = self._safe_str(props.last_modified_by)
        category = self._safe_str(props.category)
        revision = self._safe_str(props.revision)

        create_time = self._to_datetime(props.created)
        last_saved_time = self._to_datetime(props.modified)

        return DocumentMetadata(
            title=title or None,
            subject=subject or None,
            author=author or None,
            keywords=keywords or None,
            comments=comments or None,
            last_saved_by=last_saved_by or None,
            create_time=create_time,
            last_saved_time=last_saved_time,
            category=category or None,
            revision=revision or None,
        )

    @staticmethod
    def _safe_str(value: Any) -> str:
        """Safely convert a property value to string."""
        if value is None:
            return ""
        try:
            return str(value).strip()
        except Exception:
            return ""

    @staticmethod
    def _to_datetime(value: Any) -> Optional[datetime]:
        """Convert a core_properties date to datetime or None."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return None


__all__ = ["DocxMetadataExtractor"]
