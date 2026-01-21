# libs/core/processor/pdf_helpers/pdf_metadata.py
"""
PDF Metadata Extraction Module

Provides PDFMetadataExtractor class for extracting and formatting PDF document metadata.
Implements BaseMetadataExtractor interface from contextifier.core.functions.
"""
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)

logger = logging.getLogger("document-processor")


class PDFMetadataExtractor(BaseMetadataExtractor):
    """
    PDF Metadata Extractor.
    
    Extracts metadata from PyMuPDF (fitz) document objects.
    
    Supported fields:
    - title, subject, author, keywords
    - create_time, last_saved_time
    
    Usage:
        extractor = PDFMetadataExtractor()
        metadata = extractor.extract(pdf_doc)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from PDF document.
        
        Args:
            source: PyMuPDF document object (fitz.Document)
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        try:
            pdf_meta = source.metadata
            if not pdf_meta:
                return DocumentMetadata()
            
            return DocumentMetadata(
                title=self._get_stripped(pdf_meta, 'title'),
                subject=self._get_stripped(pdf_meta, 'subject'),
                author=self._get_stripped(pdf_meta, 'author'),
                keywords=self._get_stripped(pdf_meta, 'keywords'),
                create_time=parse_pdf_date(pdf_meta.get('creationDate')),
                last_saved_time=parse_pdf_date(pdf_meta.get('modDate')),
            )
        except Exception as e:
            self.logger.debug(f"[PDF] Error extracting metadata: {e}")
            return DocumentMetadata()
    
    def _get_stripped(self, meta: Dict[str, Any], key: str) -> Optional[str]:
        """Get stripped string value from metadata dict."""
        value = meta.get(key)
        return value.strip() if value else None


def parse_pdf_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Convert a PDF date string to datetime.

    Args:
        date_str: PDF date string (e.g., "D:20231215120000")

    Returns:
        datetime object or None
    """
    if not date_str:
        return None

    try:
        if date_str.startswith("D:"):
            date_str = date_str[2:]

        if len(date_str) >= 14:
            return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        elif len(date_str) >= 8:
            return datetime.strptime(date_str[:8], "%Y%m%d")

    except Exception as e:
        logger.debug(f"[PDF] Error parsing date '{date_str}': {e}")

    return None


__all__ = [
    "PDFMetadataExtractor",
    "parse_pdf_date",
]
