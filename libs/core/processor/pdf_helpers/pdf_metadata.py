# libs/core/processor/pdf_helpers/pdf_metadata.py
"""
PDF Metadata Extraction Module

Provides functions for extracting and formatting PDF document metadata.
"""
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("document-processor")


def extract_pdf_metadata(doc) -> Dict[str, Any]:
    """
    Extract metadata from a PDF document.

    Args:
        doc: PyMuPDF document object

    Returns:
        Metadata dictionary
    """
    metadata = {}

    try:
        pdf_meta = doc.metadata
        if not pdf_meta:
            return metadata

        if pdf_meta.get('title'):
            metadata['title'] = pdf_meta['title'].strip()

        if pdf_meta.get('subject'):
            metadata['subject'] = pdf_meta['subject'].strip()

        if pdf_meta.get('author'):
            metadata['author'] = pdf_meta['author'].strip()

        if pdf_meta.get('keywords'):
            metadata['keywords'] = pdf_meta['keywords'].strip()

        if pdf_meta.get('creationDate'):
            create_time = parse_pdf_date(pdf_meta['creationDate'])
            if create_time:
                metadata['create_time'] = create_time

        if pdf_meta.get('modDate'):
            mod_time = parse_pdf_date(pdf_meta['modDate'])
            if mod_time:
                metadata['last_saved_time'] = mod_time

    except Exception as e:
        logger.debug(f"[PDF] Error extracting metadata: {e}")

    return metadata


def parse_pdf_date(date_str: str) -> Optional[datetime]:
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


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format metadata as a string.

    Args:
        metadata: Metadata dictionary

    Returns:
        Formatted metadata string
    """
    if not metadata:
        return ""

    lines = ["<Document-Metadata>"]

    field_names = {
        'title': 'Title',
        'subject': 'Subject',
        'author': 'Author',
        'keywords': 'Keywords',
        'create_time': 'Created',
        'last_saved_time': 'Last Modified'
    }

    for key, label in field_names.items():
        value = metadata.get(key)
        if value:
            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"  {label}: {value}")

    lines.append("</Document-Metadata>\n")

    return "\n".join(lines)
