# libs/core/processor/pdf_helpers/pdf_image.py
"""
PDF Image Extraction Module

Provides functions for extracting images from PDF pages.
"""
import logging
from typing import List, Set, Tuple

from libs.core.processor.pdf_helpers.types import (
    ElementType,
    PageElement,
)
from libs.core.processor.pdf_helpers.pdf_utils import (
    find_image_position,
    is_inside_any_bbox,
)
from libs.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")

# Module level image processor
_image_processor = ImageProcessor(
    directory_path="temp/images",
    tag_prefix="[Image:",
    tag_suffix="]"
)


def extract_images_from_page(
    page,
    page_num: int,
    doc,
    processed_images: Set[int],
    table_bboxes: List[Tuple[float, float, float, float]],
    min_image_size: int = 50,
    min_image_area: int = 2500
) -> List[PageElement]:
    """
    Extract images from page and save locally.

    Args:
        page: PyMuPDF page object
        page_num: Page number (0-indexed)
        doc: PyMuPDF document object
        processed_images: Set of already processed image xrefs
        table_bboxes: List of table bounding boxes to exclude
        min_image_size: Minimum image dimension (width/height)
        min_image_area: Minimum image area

    Returns:
        List of PageElement for extracted images
    """
    elements = []

    try:
        image_list = page.get_images()

        for img_info in image_list:
            xref = img_info[0]

            if xref in processed_images:
                continue

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image.get("image")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < min_image_size or height < min_image_size:
                    continue
                if width * height < min_image_area:
                    continue

                img_bbox = find_image_position(page, xref)
                if img_bbox is None:
                    continue

                if is_inside_any_bbox(img_bbox, table_bboxes, threshold=0.7):
                    continue

                image_tag = _image_processor.save_image(image_bytes)

                if image_tag:
                    processed_images.add(xref)

                    elements.append(PageElement(
                        element_type=ElementType.IMAGE,
                        content=f'\n{image_tag}\n',
                        bbox=img_bbox,
                        page_num=page_num
                    ))

            except Exception as e:
                logger.debug(f"[PDF] Error extracting image xref={xref}: {e}")
                continue

    except Exception as e:
        logger.warning(f"[PDF] Error extracting images: {e}")

    return elements


def get_image_processor() -> ImageProcessor:
    """
    Get the module-level image processor instance.

    Returns:
        ImageProcessor instance
    """
    return _image_processor
