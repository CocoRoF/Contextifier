# contextifier_new/services/metadata_service.py
"""
MetadataService — Metadata Formatting

Replaces the old MetadataFormatter that was tightly coupled inside
BaseMetadataExtractor. Now it's a standalone service that:

1. Takes a DocumentMetadata instance
2. Produces a formatted text block wrapped in metadata tags
3. Supports Korean and English field labels
4. Handles date formatting

Separation of concerns:
- MetadataExtractor (pipeline): extracts metadata FROM documents
- MetadataService (service): formats metadata FOR output
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from contextifier_new.config import ProcessingConfig, MetadataConfig
from contextifier_new.types import DocumentMetadata, MetadataField


# Field label mappings
_LABELS_KO: Dict[str, str] = {
    MetadataField.TITLE: "제목",
    MetadataField.SUBJECT: "주제",
    MetadataField.AUTHOR: "작성자",
    MetadataField.KEYWORDS: "키워드",
    MetadataField.COMMENTS: "설명",
    MetadataField.LAST_SAVED_BY: "최종 수정자",
    MetadataField.CREATE_TIME: "작성일",
    MetadataField.LAST_SAVED_TIME: "최종 수정일",
    MetadataField.PAGE_COUNT: "페이지 수",
    MetadataField.WORD_COUNT: "단어 수",
    MetadataField.CATEGORY: "범주",
    MetadataField.REVISION: "버전",
}

_LABELS_EN: Dict[str, str] = {
    MetadataField.TITLE: "Title",
    MetadataField.SUBJECT: "Subject",
    MetadataField.AUTHOR: "Author",
    MetadataField.KEYWORDS: "Keywords",
    MetadataField.COMMENTS: "Comments",
    MetadataField.LAST_SAVED_BY: "Last Saved By",
    MetadataField.CREATE_TIME: "Created",
    MetadataField.LAST_SAVED_TIME: "Last Modified",
    MetadataField.PAGE_COUNT: "Page Count",
    MetadataField.WORD_COUNT: "Word Count",
    MetadataField.CATEGORY: "Category",
    MetadataField.REVISION: "Revision",
}


class MetadataService:
    """
    Formats DocumentMetadata into tagged text blocks.

    Example output:
        <Document-Metadata>
          제목: My Document
          작성자: John Doe
          작성일: 2024-01-15 10:30:00
        </Document-Metadata>
    """

    def __init__(self, config: ProcessingConfig) -> None:
        self._config = config
        self._meta_config: MetadataConfig = config.metadata
        self._tag_config = config.tags
        self._logger = logging.getLogger("contextifier.services.metadata")

        # Select label set
        self._labels = (
            _LABELS_KO if self._meta_config.language == "ko" else _LABELS_EN
        )

    def format_metadata(self, metadata: Optional[DocumentMetadata]) -> str:
        """
        Format document metadata into a tagged text block.

        Args:
            metadata: DocumentMetadata instance, or None.

        Returns:
            Formatted metadata string, or empty string if no metadata.
        """
        if metadata is None or metadata.is_empty():
            return ""

        lines: List[str] = [self._tag_config.metadata_prefix]
        indent = self._meta_config.indent

        # Standard fields in defined order
        field_order = [
            (MetadataField.TITLE, metadata.title),
            (MetadataField.SUBJECT, metadata.subject),
            (MetadataField.AUTHOR, metadata.author),
            (MetadataField.KEYWORDS, metadata.keywords),
            (MetadataField.COMMENTS, metadata.comments),
            (MetadataField.LAST_SAVED_BY, metadata.last_saved_by),
            (MetadataField.CREATE_TIME, metadata.create_time),
            (MetadataField.LAST_SAVED_TIME, metadata.last_saved_time),
            (MetadataField.PAGE_COUNT, metadata.page_count),
            (MetadataField.WORD_COUNT, metadata.word_count),
            (MetadataField.CATEGORY, metadata.category),
            (MetadataField.REVISION, metadata.revision),
        ]

        for field_enum, value in field_order:
            if value is None:
                continue
            label = self._labels.get(field_enum, field_enum.value)
            formatted_value = self._format_value(value)
            if formatted_value:
                lines.append(f"{indent}{label}: {formatted_value}")

        # Custom fields
        if metadata.custom:
            for key, value in metadata.custom.items():
                formatted = self._format_value(value)
                if formatted:
                    lines.append(f"{indent}{key}: {formatted}")

        lines.append(self._tag_config.metadata_suffix)
        return "\n".join(lines)

    def format_metadata_dict(self, data: Dict[str, Any]) -> str:
        """Format metadata from a dictionary."""
        metadata = DocumentMetadata.from_dict(data)
        return self.format_metadata(metadata)

    def _format_value(self, value: Any) -> Optional[str]:
        """Format a single metadata value."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.strftime(self._meta_config.date_format)
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value).strip()
        return text if text else None


__all__ = ["MetadataService"]
