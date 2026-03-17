# contextifier_new/pipeline/metadata_extractor.py
"""
BaseMetadataExtractor — Stage 3: Document Metadata Extraction

Responsible for:
- Taking the converted/preprocessed format object
- Extracting standard metadata fields (title, author, dates, etc.)
- Returning a standardized DocumentMetadata instance

Contract:
- extract() is the ONLY abstract method
- format() converts metadata to tagged string (uses MetadataService)
- extract_and_format() convenience: extract + format in one call

Design decisions vs old version:
1. MetadataFormatter is now a separate service (MetadataService),
   not embedded inside the extractor
2. Every handler MUST have a MetadataExtractor — use NullMetadataExtractor
   for formats with no metadata (text, images)
3. No more returning None from factory — NullMetadataExtractor is explicit
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from contextifier_new.types import DocumentMetadata
from contextifier_new.errors import ExtractionError


class BaseMetadataExtractor(ABC):
    """
    Abstract base for all metadata extractors.

    Each format implements extract() to produce DocumentMetadata
    from its native format object.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"contextifier.metadata.{self.__class__.__name__}"
        )

    @abstractmethod
    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from a format-specific source object.

        Args:
            source: Format-specific object
                    - PDF: fitz.Document
                    - DOCX: python-docx Document
                    - XLSX: openpyxl Workbook
                    - OLE: olefile.OleFileIO
                    - etc.

        Returns:
            DocumentMetadata with populated fields.

        Raises:
            ExtractionError: If metadata extraction fails.
        """
        ...

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the canonical format name."""
        ...


class NullMetadataExtractor(BaseMetadataExtractor):
    """
    Null metadata extractor — always returns empty metadata.

    Used for formats that have no inherent metadata
    (plain text, CSV, images, etc.).
    """

    def extract(self, source: Any) -> DocumentMetadata:
        """Return empty metadata."""
        return DocumentMetadata()

    def get_format_name(self) -> str:
        return "null"


__all__ = ["BaseMetadataExtractor", "NullMetadataExtractor"]
