# libs/core/processor/pdf_helpers/pdf_helper.py
"""
PDF Processing Common Helper Module

Defines utility functions commonly used by PDF handlers.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("document-processor")


# ============================================================================
# PDF Metadata Extraction
# ============================================================================

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


# ============================================================================
# HTML Escape
# ============================================================================

def escape_html(text: str) -> str:
    """
    Escape HTML special characters.

    Args:
        text: Original text

    Returns:
        Escaped text
    """
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ============================================================================
# Bounding Box Utilities
# ============================================================================

def calculate_overlap_ratio(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes.

    Args:
        bbox1: First bbox (x0, y0, x1, y1)
        bbox2: Second bbox (x0, y0, x1, y1)

    Returns:
        Overlap ratio relative to bbox1 (0.0 ~ 1.0)
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    overlap_area = (x1 - x0) * (y1 - y0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    if bbox1_area <= 0:
        return 0.0

    return overlap_area / bbox1_area


def is_inside_any_bbox(
    bbox: Tuple[float, float, float, float],
    bbox_list: List[Tuple[float, float, float, float]],
    threshold: float = 0.5
) -> bool:
    """
    Check if a bbox is contained within any bbox in the list.

    Args:
        bbox: Bounding box to check
        bbox_list: List of bounding boxes
        threshold: Overlap ratio threshold

    Returns:
        True if contained, False otherwise
    """
    for target_bbox in bbox_list:
        overlap = calculate_overlap_ratio(bbox, target_bbox)
        if overlap > threshold:
            return True
    return False


# ============================================================================
# Image Position Detection
# ============================================================================

def find_image_position(page, xref: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Find the position of an image within a page.

    Args:
        page: PyMuPDF page object
        xref: Image xref

    Returns:
        Bounding box or None
    """
    try:
        image_list = page.get_image_info(xrefs=True)

        for img_info in image_list:
            if img_info.get("xref") == xref:
                bbox = img_info.get("bbox")
                if bbox:
                    return tuple(bbox)

        return None

    except Exception as e:
        logger.debug(f"[PDF] Error finding image position: {e}")
        return None


# ============================================================================
# Text Line Extraction
# ============================================================================

def get_text_lines_with_positions(page) -> List[Dict]:
    """
    Extract text lines and position information from a page.

    Args:
        page: PyMuPDF page object

    Returns:
        List of text line information
    """
    lines = []
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            line_bbox = line.get("bbox", (0, 0, 0, 0))
            text_parts = []

            for span in line.get("spans", []):
                text_parts.append(span.get("text", ""))

            full_text = "".join(text_parts).strip()
            if full_text:
                lines.append({
                    'text': full_text,
                    'y0': line_bbox[1],
                    'y1': line_bbox[3],
                    'x0': line_bbox[0],
                    'x1': line_bbox[2]
                })

    return lines
