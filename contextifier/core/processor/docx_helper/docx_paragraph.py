# service/document_processor/processor/docx_helper/docx_paragraph.py
"""
DOCX Paragraph Processing Utility

Processes Paragraph elements in DOCX documents.
- process_paragraph_element: Process Paragraph element
- has_page_break_element: Check for page break
"""
import logging
from typing import Optional, Set, Tuple, Callable

from docx import Document

from contextifier.core.processor.docx_helper.docx_constants import ElementType, NAMESPACES
from contextifier.core.processor.docx_helper.docx_drawing import process_drawing_element
from contextifier.core.processor.docx_helper.docx_image import process_pict_element
from contextifier.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")


def process_paragraph_element(
    para_elem,
    doc: Document,
    processed_images: Set[str],
    file_path: str = None,
    image_processor: Optional[ImageProcessor] = None,
    chart_callback: Optional[Callable[[], str]] = None
) -> Tuple[str, bool, int, int]:
    """
    Process Paragraph element.

    Extracts text, images, charts and detects page breaks.

    Args:
        para_elem: paragraph XML element
        doc: python-docx Document object
        processed_images: Set of processed image paths (deduplication)
        file_path: Original file path
        image_processor: ImageProcessor instance
        chart_callback: Callback function to get next chart content

    Returns:
        (content, has_page_break, image_count, chart_count) tuple
    """
    content_parts = []
    has_page_break = False
    image_count = 0
    chart_count = 0

    try:
        # Check for page break
        has_page_break = has_page_break_element(para_elem)

        # Traverse Run elements
        for run_elem in para_elem.findall('.//w:r', NAMESPACES):
            # Extract text
            for t_elem in run_elem.findall('w:t', NAMESPACES):
                if t_elem.text:
                    content_parts.append(t_elem.text)

            # Process Drawing (image/chart/diagram)
            for drawing_elem in run_elem.findall('w:drawing', NAMESPACES):
                drawing_content, drawing_type = process_drawing_element(
                    drawing_elem, doc, processed_images, file_path, 
                    image_processor, 
                    chart_callback=chart_callback
                )
                if drawing_content:
                    content_parts.append(drawing_content)
                    if drawing_type == ElementType.IMAGE:
                        image_count += 1
                    elif drawing_type == ElementType.CHART:
                        chart_count += 1

            # Process pict element (legacy VML image)
            for pict_elem in run_elem.findall('w:pict', NAMESPACES):
                pict_content = process_pict_element(pict_elem, doc, processed_images, image_processor)
                if pict_content:
                    content_parts.append(pict_content)
                    image_count += 1

    except Exception as e:
        logger.warning(f"Error processing paragraph: {e}")
        # Fallback: simple text extraction
        try:
            texts = para_elem.findall('.//w:t', NAMESPACES)
            content_parts = [t.text or '' for t in texts]
        except:
            pass

    return ''.join(content_parts), has_page_break, image_count, chart_count


def has_page_break_element(element) -> bool:
    """
    Check if element contains a page break.

    Args:
        element: XML element

    Returns:
        Whether page break exists
    """
    try:
        # Explicit page break
        if element.findall('.//w:br[@w:type="page"]', NAMESPACES):
            return True
        # Rendered page break
        if element.findall('.//w:lastRenderedPageBreak', NAMESPACES):
            return True
        return False
    except Exception:
        return False


__all__ = [
    'process_paragraph_element',
    'has_page_break_element',
]
