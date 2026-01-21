# contextifier/core/processor/excel_helper/excel_metadata.py
"""
Excel Metadata Extraction Module

Provides ExcelMetadataExtractor classes for extracting metadata from Excel documents.
Supports both XLSX (openpyxl) and XLS (xlrd) formats.
Implements BaseMetadataExtractor interface.
"""
import logging
from typing import Any, Optional

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)

logger = logging.getLogger("document-processor")


class XLSXMetadataExtractor(BaseMetadataExtractor):
    """
    XLSX Metadata Extractor.
    
    Extracts metadata from openpyxl Workbook objects.
    
    Supported fields:
    - title, subject, author (creator), keywords
    - comments (description), last_saved_by
    - create_time, last_saved_time
    
    Usage:
        extractor = XLSXMetadataExtractor()
        metadata = extractor.extract(workbook)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from XLSX document.
        
        Args:
            source: openpyxl Workbook object
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        try:
            props = source.properties
            
            return DocumentMetadata(
                title=self._get_stripped(props.title),
                subject=self._get_stripped(props.subject),
                author=self._get_stripped(props.creator),
                keywords=self._get_stripped(props.keywords),
                comments=self._get_stripped(props.description),
                last_saved_by=self._get_stripped(props.lastModifiedBy),
                create_time=props.created,
                last_saved_time=props.modified,
            )
        except Exception as e:
            self.logger.warning(f"Failed to extract XLSX metadata: {e}")
            return DocumentMetadata()
    
    def _get_stripped(self, value: Optional[str]) -> Optional[str]:
        """Return stripped string value, or None if empty."""
        return value.strip() if value else None


class XLSMetadataExtractor(BaseMetadataExtractor):
    """
    XLS Metadata Extractor.
    
    Extracts metadata from xlrd Workbook objects.
    Note: xlrd has limited metadata support.
    
    Supported fields:
    - author (user_name)
    
    Usage:
        extractor = XLSMetadataExtractor()
        metadata = extractor.extract(workbook)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from XLS document.
        
        Args:
            source: xlrd Workbook object
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        try:
            author = None
            if hasattr(source, 'user_name') and source.user_name:
                author = source.user_name
            
            return DocumentMetadata(author=author)
        except Exception as e:
            self.logger.warning(f"Failed to extract XLS metadata: {e}")
            return DocumentMetadata()


class ExcelMetadataExtractor(BaseMetadataExtractor):
    """
    Unified Excel Metadata Extractor.
    
    Selects appropriate extractor based on file format.
    
    Usage:
        extractor = ExcelMetadataExtractor()
        # For XLSX
        metadata = extractor.extract(xlsx_workbook, file_type='xlsx')
        # For XLS
        metadata = extractor.extract(xls_workbook, file_type='xls')
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xlsx_extractor = XLSXMetadataExtractor(**kwargs)
        self._xls_extractor = XLSMetadataExtractor(**kwargs)
    
    def extract(self, source: Any, file_type: str = 'xlsx') -> DocumentMetadata:
        """
        Extract metadata from Excel document.
        
        Args:
            source: openpyxl Workbook or xlrd Workbook object
            file_type: File format ('xlsx' or 'xls')
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        if file_type.lower() == 'xls':
            return self._xls_extractor.extract(source)
        return self._xlsx_extractor.extract(source)


__all__ = [
    'ExcelMetadataExtractor',
    'XLSXMetadataExtractor',
    'XLSMetadataExtractor',
]
