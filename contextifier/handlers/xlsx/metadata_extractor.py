"""
XlsxMetadataExtractor — Stage 3: Workbook properties → DocumentMetadata.

Reads OOXML core properties from the openpyxl Workbook's ``properties``
attribute (aka ``DocumentProperties``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import openpyxl

from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata, PreprocessedData
from contextifier.handlers.xlsx.converter import XlsxConvertedData

logger = logging.getLogger(__name__)


class XlsxMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for XLSX files.

    Reads OOXML ``core.xml`` properties exposed by openpyxl's
    ``Workbook.properties`` (``DocumentProperties`` object).
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from an openpyxl Workbook or PreprocessedData.

        ``source`` may be:
        - ``openpyxl.Workbook`` directly
        - ``PreprocessedData`` wrapping a Workbook in ``.content``
        """
        wb = self._unwrap_workbook(source)
        if wb is None:
            self._logger.warning("Cannot extract XLSX metadata: no Workbook found")
            return DocumentMetadata()

        try:
            props = wb.properties
            if props is None:
                return DocumentMetadata()

            return DocumentMetadata(
                title=self._safe_str(props.title),
                subject=self._safe_str(props.subject),
                author=self._safe_str(props.creator),
                keywords=self._safe_str(props.keywords),
                comments=self._safe_str(props.description),
                last_saved_by=self._safe_str(props.lastModifiedBy),
                create_time=self._safe_datetime(props.created),
                last_saved_time=self._safe_datetime(props.modified),
                category=self._safe_str(props.category),
                revision=self._safe_str(props.revision) if hasattr(props, "revision") else None,
                page_count=len(wb.sheetnames),
            )
        except Exception as exc:
            self._logger.warning("Failed to extract XLSX metadata: %s", exc)
            return DocumentMetadata()

    def get_format_name(self) -> str:
        return "xlsx"

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_workbook(source: Any) -> Any:
        """Resolve openpyxl Workbook from various input types."""
        if isinstance(source, openpyxl.Workbook):
            return source
        if isinstance(source, XlsxConvertedData):
            return source.workbook
        if isinstance(source, PreprocessedData):
            content = getattr(source, "content", None)
            if isinstance(content, openpyxl.Workbook):
                return content
        # Duck typing: has .properties and .sheetnames
        if hasattr(source, "properties") and hasattr(source, "sheetnames"):
            return source
        return None

    @staticmethod
    def _safe_str(value: Any) -> Optional[str]:
        """Convert a property value to string, returning None for empty."""
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    @staticmethod
    def _safe_datetime(value: Any) -> Optional[datetime]:
        """Convert a property value to datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return None


__all__ = ["XlsxMetadataExtractor"]
