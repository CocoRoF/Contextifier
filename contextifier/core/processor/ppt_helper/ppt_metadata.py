# contextifier/core/processor/ppt_helper/ppt_metadata.py
"""
PPT Metadata Extraction Module

Provides PPTMetadataExtractor class for extracting metadata from PowerPoint documents.
Implements BaseMetadataExtractor interface.
"""
import logging
from typing import Any, Optional

from pptx import Presentation

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)

logger = logging.getLogger("document-processor")


class PPTMetadataExtractor(BaseMetadataExtractor):
    """
    PPT/PPTX Metadata Extractor.
    
    Extracts metadata from python-pptx Presentation objects.
    
    Supported fields:
    - title, subject, author, keywords, comments
    - last_saved_by, create_time, last_saved_time
    
    Usage:
        extractor = PPTMetadataExtractor()
        metadata = extractor.extract(presentation)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: Presentation) -> DocumentMetadata:
        """
        Extract metadata from PPT document.
        
        Args:
            source: python-pptx Presentation object
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        try:
            props = source.core_properties
            
            return DocumentMetadata(
                title=self._get_value(props.title),
                subject=self._get_value(props.subject),
                author=self._get_value(props.author),
                keywords=self._get_value(props.keywords),
                comments=self._get_value(props.comments),
                last_saved_by=self._get_value(props.last_modified_by),
                create_time=props.created,
                last_saved_time=props.modified,
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract PPT metadata: {e}")
            return DocumentMetadata()
    
    def _get_value(self, value: Optional[str]) -> Optional[str]:
        """Return value if present, None otherwise."""
        return value if value else None


__all__ = [
    'PPTMetadataExtractor',
]
