# service/document_processor/processor/docx_helper/docx_drawing.py
"""
DOCX Drawing Element Processing Utility

Processes Drawing elements (images, charts, diagrams) in DOCX documents.
- process_drawing_element: Process Drawing element (branch to image/chart/diagram)
- extract_diagram_from_drawing: Extract diagram from Drawing

Note: Chart extraction is handled separately by DOCXChartExtractor.
      This module only detects chart presence for counting/positioning.
"""
import logging
from typing import Optional, Set, Tuple, Callable

from docx import Document

from contextifier.core.processor.docx_helper.docx_constants import ElementType, NAMESPACES
from contextifier.core.processor.docx_helper.docx_image import extract_image_from_drawing
from contextifier.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")


def process_drawing_element(
    drawing_elem,
    doc: Document,
    processed_images: Set[str],
    file_path: str = None,
    image_processor: Optional[ImageProcessor] = None,
    chart_callback: Optional[Callable[[], str]] = None
) -> Tuple[str, Optional[ElementType]]:
    """
    Process Drawing element (image, chart, diagram).

    Args:
        drawing_elem: drawing XML element
        doc: python-docx Document object
        processed_images: Set of processed image paths (deduplication)
        file_path: Original file path
        image_processor: ImageProcessor instance
        chart_callback: Callback function to get next chart content.
                       Called when chart is detected, should return formatted chart string.

    Returns:
        (content, element_type) tuple
    """
    try:
        # Check inline or anchor
        inline = drawing_elem.find('.//wp:inline', NAMESPACES)
        anchor = drawing_elem.find('.//wp:anchor', NAMESPACES)

        container = inline if inline is not None else anchor
        if container is None:
            return "", None

        # Check graphic data
        graphic = container.find('.//a:graphic', NAMESPACES)
        if graphic is None:
            return "", None

        graphic_data = graphic.find('a:graphicData', NAMESPACES)
        if graphic_data is None:
            return "", None

        uri = graphic_data.get('uri', '')

        # Image case
        if 'picture' in uri.lower():
            return extract_image_from_drawing(graphic_data, doc, processed_images, image_processor)

        # Chart case - use callback to get pre-extracted chart content
        if 'chart' in uri.lower():
            if chart_callback:
                chart_content = chart_callback()
                return chart_content, ElementType.CHART
            return "", ElementType.CHART

        # Diagram case
        if 'diagram' in uri.lower():
            return extract_diagram_from_drawing(graphic_data, doc)

        # Other drawing
        return "", None

    except Exception as e:
        logger.warning(f"Error processing drawing element: {e}")
        return "", None


def extract_diagram_from_drawing(graphic_data, doc: Document) -> Tuple[str, Optional[ElementType]]:
    """
    Extract diagram information from Drawing.

    Args:
        graphic_data: graphicData XML element
        doc: python-docx Document object

    Returns:
        (content, element_type) tuple
    """
    try:
        # Try to extract text from diagram
        texts = []
        for t_elem in graphic_data.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
            if t_elem.text:
                texts.append(t_elem.text.strip())

        if texts:
            return f"[Diagram: {' / '.join(texts)}]", ElementType.DIAGRAM

        return "[Diagram]", ElementType.DIAGRAM

    except Exception as e:
        logger.warning(f"Error extracting diagram from drawing: {e}")
        return "[Diagram]", ElementType.DIAGRAM


__all__ = [
    'process_drawing_element',
    'extract_diagram_from_drawing',
]
