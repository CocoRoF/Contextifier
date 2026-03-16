"""
PptxMetadataExtractor — Stage 3: core_properties → DocumentMetadata.

Reads standard OOXML core properties (title, author, dates, etc.)
from the ``pptx.Presentation`` object.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.types import DocumentMetadata, PreprocessedData

logger = logging.getLogger(__name__)


class PptxMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for PPTX files.

    Reads ``core_properties`` from the python-pptx Presentation.
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from a Presentation.

        ``source`` may be:
        - ``pptx.Presentation`` directly
        - ``PreprocessedData`` wrapping a Presentation
        """
        prs = self._unwrap_presentation(source)
        if prs is None:
            self._logger.warning("Cannot extract PPTX metadata: no Presentation found")
            return DocumentMetadata()

        try:
            props = prs.core_properties
            return DocumentMetadata(
                title=self._get(props.title),
                subject=self._get(props.subject),
                author=self._get(props.author),
                keywords=self._get(props.keywords),
                comments=self._get(props.comments),
                last_saved_by=self._get(props.last_modified_by),
                create_time=props.created,
                last_saved_time=props.modified,
                page_count=len(prs.slides) if hasattr(prs, "slides") else None,
                category=self._get(props.category) if hasattr(props, "category") else None,
                revision=str(props.revision) if hasattr(props, "revision") and props.revision else None,
            )
        except Exception as exc:
            self._logger.warning("Failed to extract PPTX metadata: %s", exc)
            return DocumentMetadata()

    def get_format_name(self) -> str:
        return "pptx"

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_presentation(source: Any) -> Any:
        """Resolve pptx.Presentation from various input types."""
        # Direct Presentation
        if hasattr(source, "slides") and hasattr(source, "slide_width"):
            return source
        # PreprocessedData
        if isinstance(source, PreprocessedData):
            for attr in ("content", "raw_content"):
                obj = getattr(source, attr, None)
                if obj is not None and hasattr(obj, "slides"):
                    return obj
        return None

    @staticmethod
    def _get(value: Optional[str]) -> Optional[str]:
        """Return value if non-empty, else None."""
        return value if value else None


__all__ = ["PptxMetadataExtractor"]
