# contextifier/core/functions/metadata_extractor.py
"""
Metadata Extractor Interface

Provides abstract base class and common utilities for document metadata extraction.
Each handler's helper module should implement a concrete extractor inheriting from
BaseMetadataExtractor.

This module defines:
- DocumentMetadata: Standardized metadata container dataclass
- MetadataField: Enum for standard metadata field names
- BaseMetadataExtractor: Abstract base class for metadata extractors
- MetadataFormatter: Shared formatter for consistent metadata output

Usage Example:
    from contextifier.core.functions.metadata_extractor import (
        BaseMetadataExtractor,
        DocumentMetadata,
        MetadataFormatter,
    )

    class PDFMetadataExtractor(BaseMetadataExtractor):
        def extract(self, source: Any) -> DocumentMetadata:
            # PDF-specific extraction logic
            ...
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("contextify.metadata")


class MetadataField(str, Enum):
    """
    Standard metadata field names.
    
    These field names are used consistently across all document formats
    to ensure uniform metadata handling.
    """
    TITLE = "title"
    SUBJECT = "subject"
    AUTHOR = "author"
    KEYWORDS = "keywords"
    COMMENTS = "comments"
    LAST_SAVED_BY = "last_saved_by"
    CREATE_TIME = "create_time"
    LAST_SAVED_TIME = "last_saved_time"
    
    # Additional fields for specific formats
    VERSION = "version"
    CATEGORY = "category"
    COMPANY = "company"
    MANAGER = "manager"
    
    # File-level metadata (for CSV, etc.)
    FILE_NAME = "file_name"
    FILE_SIZE = "file_size"
    ENCODING = "encoding"
    ROW_COUNT = "row_count"
    COL_COUNT = "col_count"


@dataclass
class DocumentMetadata:
    """
    Standardized metadata container for all document types.
    
    This dataclass provides a unified structure for storing document metadata
    across all supported file formats. It includes common fields and allows
    for format-specific custom fields.
    
    Attributes:
        title: Document title
        subject: Document subject
        author: Document author/creator
        keywords: Document keywords
        comments: Document comments/description
        last_saved_by: Last person who saved the document
        create_time: Document creation timestamp
        last_saved_time: Last modification timestamp
        custom: Dictionary for format-specific additional fields
        
    Example:
        >>> metadata = DocumentMetadata(
        ...     title="Annual Report",
        ...     author="John Doe",
        ...     create_time=datetime.now()
        ... )
        >>> metadata.to_dict()
        {'title': 'Annual Report', 'author': 'John Doe', ...}
    """
    title: Optional[str] = None
    subject: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[str] = None
    comments: Optional[str] = None
    last_saved_by: Optional[str] = None
    create_time: Optional[datetime] = None
    last_saved_time: Optional[datetime] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.
        
        Returns:
            Dictionary containing all non-None metadata fields.
        """
        result = {}
        
        if self.title:
            result[MetadataField.TITLE.value] = self.title
        if self.subject:
            result[MetadataField.SUBJECT.value] = self.subject
        if self.author:
            result[MetadataField.AUTHOR.value] = self.author
        if self.keywords:
            result[MetadataField.KEYWORDS.value] = self.keywords
        if self.comments:
            result[MetadataField.COMMENTS.value] = self.comments
        if self.last_saved_by:
            result[MetadataField.LAST_SAVED_BY.value] = self.last_saved_by
        if self.create_time:
            result[MetadataField.CREATE_TIME.value] = self.create_time
        if self.last_saved_time:
            result[MetadataField.LAST_SAVED_TIME.value] = self.last_saved_time
        
        # Add custom fields
        result.update(self.custom)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """
        Create DocumentMetadata from dictionary.
        
        Standard fields are extracted into their respective attributes,
        while non-standard fields go into the custom dictionary.
        
        Args:
            data: Dictionary containing metadata fields.
            
        Returns:
            DocumentMetadata instance.
        """
        standard_fields = {
            MetadataField.TITLE.value,
            MetadataField.SUBJECT.value,
            MetadataField.AUTHOR.value,
            MetadataField.KEYWORDS.value,
            MetadataField.COMMENTS.value,
            MetadataField.LAST_SAVED_BY.value,
            MetadataField.CREATE_TIME.value,
            MetadataField.LAST_SAVED_TIME.value,
        }
        
        custom = {k: v for k, v in data.items() if k not in standard_fields}
        
        return cls(
            title=data.get(MetadataField.TITLE.value),
            subject=data.get(MetadataField.SUBJECT.value),
            author=data.get(MetadataField.AUTHOR.value),
            keywords=data.get(MetadataField.KEYWORDS.value),
            comments=data.get(MetadataField.COMMENTS.value),
            last_saved_by=data.get(MetadataField.LAST_SAVED_BY.value),
            create_time=data.get(MetadataField.CREATE_TIME.value),
            last_saved_time=data.get(MetadataField.LAST_SAVED_TIME.value),
            custom=custom,
        )
    
    def is_empty(self) -> bool:
        """
        Check if metadata is empty (no fields set).
        
        Returns:
            True if no metadata fields are set.
        """
        return not self.to_dict()
    
    def __bool__(self) -> bool:
        """Return True if metadata has any fields set."""
        return not self.is_empty()


class MetadataFormatter:
    """
    Shared formatter for consistent metadata output.
    
    This class provides a unified way to format DocumentMetadata objects
    as strings for inclusion in extracted text output.
    
    Attributes:
        tag_open: Opening tag for metadata section (default: "<Document-Metadata>")
        tag_close: Closing tag for metadata section (default: "</Document-Metadata>")
        field_labels: Dictionary mapping field names to display labels
        date_format: Date/time format string
        language: Output language ('ko' for Korean, 'en' for English)
        
    Example:
        >>> formatter = MetadataFormatter(language='en')
        >>> text = formatter.format(metadata)
        >>> print(text)
        <Document-Metadata>
          Title: Annual Report
          Author: John Doe
        </Document-Metadata>
    """
    
    # Field labels in Korean
    LABELS_KO = {
        MetadataField.TITLE.value: "제목",
        MetadataField.SUBJECT.value: "주제",
        MetadataField.AUTHOR.value: "작성자",
        MetadataField.KEYWORDS.value: "키워드",
        MetadataField.COMMENTS.value: "설명",
        MetadataField.LAST_SAVED_BY.value: "마지막 저장자",
        MetadataField.CREATE_TIME.value: "작성일",
        MetadataField.LAST_SAVED_TIME.value: "수정일",
        # Additional fields
        MetadataField.VERSION.value: "버전",
        MetadataField.CATEGORY.value: "범주",
        MetadataField.COMPANY.value: "회사",
        MetadataField.MANAGER.value: "관리자",
        MetadataField.FILE_NAME.value: "파일명",
        MetadataField.FILE_SIZE.value: "파일 크기",
        MetadataField.ENCODING.value: "인코딩",
        MetadataField.ROW_COUNT.value: "행 수",
        MetadataField.COL_COUNT.value: "열 수",
    }
    
    # Field labels in English
    LABELS_EN = {
        MetadataField.TITLE.value: "Title",
        MetadataField.SUBJECT.value: "Subject",
        MetadataField.AUTHOR.value: "Author",
        MetadataField.KEYWORDS.value: "Keywords",
        MetadataField.COMMENTS.value: "Comments",
        MetadataField.LAST_SAVED_BY.value: "Last Saved By",
        MetadataField.CREATE_TIME.value: "Created",
        MetadataField.LAST_SAVED_TIME.value: "Last Modified",
        # Additional fields
        MetadataField.VERSION.value: "Version",
        MetadataField.CATEGORY.value: "Category",
        MetadataField.COMPANY.value: "Company",
        MetadataField.MANAGER.value: "Manager",
        MetadataField.FILE_NAME.value: "File Name",
        MetadataField.FILE_SIZE.value: "File Size",
        MetadataField.ENCODING.value: "Encoding",
        MetadataField.ROW_COUNT.value: "Row Count",
        MetadataField.COL_COUNT.value: "Column Count",
    }
    
    # Standard field order for output
    FIELD_ORDER = [
        MetadataField.TITLE.value,
        MetadataField.SUBJECT.value,
        MetadataField.AUTHOR.value,
        MetadataField.KEYWORDS.value,
        MetadataField.COMMENTS.value,
        MetadataField.LAST_SAVED_BY.value,
        MetadataField.CREATE_TIME.value,
        MetadataField.LAST_SAVED_TIME.value,
    ]
    
    def __init__(
        self,
        tag_open: str = "<Document-Metadata>",
        tag_close: str = "</Document-Metadata>",
        date_format: str = "%Y-%m-%d %H:%M:%S",
        language: str = "ko",
        indent: str = "  ",
    ):
        """
        Initialize MetadataFormatter.
        
        Args:
            tag_open: Opening tag for metadata section
            tag_close: Closing tag for metadata section
            date_format: strftime format for datetime values
            language: Output language ('ko' or 'en')
            indent: Indentation string for each field
        """
        self.tag_open = tag_open
        self.tag_close = tag_close
        self.date_format = date_format
        self.language = language
        self.indent = indent
        
        # Select labels based on language
        self.field_labels = self.LABELS_KO if language == "ko" else self.LABELS_EN
    
    def format(self, metadata: DocumentMetadata) -> str:
        """
        Format DocumentMetadata as a string.
        
        Args:
            metadata: DocumentMetadata instance to format.
            
        Returns:
            Formatted metadata string, or empty string if metadata is empty.
        """
        if not metadata:
            return ""
        
        data = metadata.to_dict()
        if not data:
            return ""
        
        lines = [self.tag_open]
        
        # Output standard fields in order
        for field_name in self.FIELD_ORDER:
            if field_name in data:
                value = data.pop(field_name)
                formatted_line = self._format_field(field_name, value)
                if formatted_line:
                    lines.append(formatted_line)
        
        # Output remaining custom fields
        for field_name, value in data.items():
            formatted_line = self._format_field(field_name, value)
            if formatted_line:
                lines.append(formatted_line)
        
        lines.append(self.tag_close)
        
        return "\n".join(lines)
    
    def format_dict(self, metadata_dict: Dict[str, Any]) -> str:
        """
        Format metadata dictionary as a string.
        
        Convenience method for formatting raw dictionaries without
        first converting to DocumentMetadata.
        
        Args:
            metadata_dict: Dictionary containing metadata fields.
            
        Returns:
            Formatted metadata string.
        """
        if not metadata_dict:
            return ""
        
        return self.format(DocumentMetadata.from_dict(metadata_dict))
    
    def _format_field(self, field_name: str, value: Any) -> Optional[str]:
        """
        Format a single metadata field.
        
        Args:
            field_name: Field name
            value: Field value
            
        Returns:
            Formatted field string, or None if value is empty.
        """
        if value is None:
            return None
        
        # Format datetime values
        if isinstance(value, datetime):
            value = value.strftime(self.date_format)
        
        # Get label (use field name as fallback)
        label = self.field_labels.get(field_name, field_name.replace("_", " ").title())
        
        return f"{self.indent}{label}: {value}"
    
    def get_label(self, field_name: str) -> str:
        """
        Get display label for a field name.
        
        Args:
            field_name: Field name
            
        Returns:
            Display label for the field.
        """
        return self.field_labels.get(field_name, field_name.replace("_", " ").title())


class BaseMetadataExtractor(ABC):
    """
    Abstract base class for metadata extractors.
    
    Each document format should implement a concrete extractor
    that inherits from this class and provides format-specific
    extraction logic.
    
    Subclasses must implement:
        - extract(): Extract metadata from format-specific source object
        
    Subclasses may optionally override:
        - format(): Customize metadata formatting
        - get_formatter(): Provide custom formatter instance
        
    Attributes:
        formatter: MetadataFormatter instance for output formatting
        logger: Logger instance for this extractor
        
    Example:
        class PDFMetadataExtractor(BaseMetadataExtractor):
            def extract(self, doc) -> DocumentMetadata:
                # Extract from PyMuPDF document object
                pdf_meta = doc.metadata
                return DocumentMetadata(
                    title=pdf_meta.get('title'),
                    author=pdf_meta.get('author'),
                    ...
                )
    """
    
    def __init__(
        self,
        formatter: Optional[MetadataFormatter] = None,
        language: str = "ko",
    ):
        """
        Initialize BaseMetadataExtractor.
        
        Args:
            formatter: Custom MetadataFormatter instance (optional)
            language: Default language for formatter if not provided
        """
        self._formatter = formatter or MetadataFormatter(language=language)
        self._logger = logging.getLogger(
            f"contextify.metadata.{self.__class__.__name__}"
        )
    
    @property
    def formatter(self) -> MetadataFormatter:
        """Get the metadata formatter instance."""
        return self._formatter
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger
    
    @abstractmethod
    def extract(self, source: Any) -> DocumentMetadata:
        """
        Extract metadata from source object.
        
        This method must be implemented by subclasses to provide
        format-specific metadata extraction logic.
        
        Args:
            source: Format-specific source object (e.g., PyMuPDF doc,
                    python-docx Document, openpyxl Workbook, etc.)
                    
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        pass
    
    def format(self, metadata: DocumentMetadata) -> str:
        """
        Format metadata as a string.
        
        Uses the formatter to convert DocumentMetadata to a string.
        Can be overridden by subclasses for custom formatting.
        
        Args:
            metadata: DocumentMetadata instance to format.
            
        Returns:
            Formatted metadata string.
        """
        return self._formatter.format(metadata)
    
    def extract_and_format(self, source: Any) -> str:
        """
        Extract metadata and format as string in one step.
        
        Convenience method that combines extract() and format().
        
        Args:
            source: Format-specific source object.
            
        Returns:
            Formatted metadata string.
        """
        try:
            metadata = self.extract(source)
            return self.format(metadata)
        except Exception as e:
            self._logger.warning(f"Failed to extract metadata: {e}")
            return ""
    
    def extract_to_dict(self, source: Any) -> Dict[str, Any]:
        """
        Extract metadata and return as dictionary.
        
        Convenience method that extracts metadata and converts to dict.
        
        Args:
            source: Format-specific source object.
            
        Returns:
            Dictionary containing metadata fields.
        """
        try:
            metadata = self.extract(source)
            return metadata.to_dict()
        except Exception as e:
            self._logger.warning(f"Failed to extract metadata: {e}")
            return {}


# Default formatter instance (Korean)
_default_formatter = MetadataFormatter(language="ko")


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format metadata dictionary as a string.
    
    Convenience function using default formatter for backward compatibility.
    
    Args:
        metadata: Dictionary containing metadata fields.
        
    Returns:
        Formatted metadata string.
    """
    return _default_formatter.format_dict(metadata)


__all__ = [
    "MetadataField",
    "DocumentMetadata",
    "MetadataFormatter",
    "BaseMetadataExtractor",
    "format_metadata",
]
