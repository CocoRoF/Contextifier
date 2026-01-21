# contextifier/core/processor/docx_helper/docx_metadata.py
"""
DOCX Metadata Extraction Module

Provides DOCXMetadataExtractor class for extracting metadata from DOCX documents
using python-docx core_properties. Implements BaseMetadataExtractor interface.
"""
import logging
from typing import Any, Optional

from docx import Document

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)

logger = logging.getLogger("document-processor")


class DOCXMetadataExtractor(BaseMetadataExtractor):
    """
    DOCX Metadata Extractor.
    
    Extracts metadata from python-docx Document objects.
    
    Supported fields:
    - title, subject, author, keywords, comments
    - last_saved_by, create_time, last_saved_time
    
    Usage:
        extractor = DOCXMetadataExtractor()
        metadata = extractor.extract(docx_document)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: Document) -> DocumentMetadata:
        """
        Extract metadata from DOCX document.
        
        Args:
            source: python-docx Document object
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        try:
            props = source.core_properties
            
            return DocumentMetadata(
                title=self._get_stripped(props.title),
                subject=self._get_stripped(props.subject),
                author=self._get_stripped(props.author),
                keywords=self._get_stripped(props.keywords),
                comments=self._get_stripped(props.comments),
                last_saved_by=self._get_stripped(props.last_modified_by),
                create_time=props.created,
                last_saved_time=props.modified,
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract DOCX metadata: {e}")
            return DocumentMetadata()
    
    def _get_stripped(self, value: Optional[str]) -> Optional[str]:
        """Return stripped string value, or None if empty."""
        return value.strip() if value else None


__all__ = [
    'DOCXMetadataExtractor',
]
