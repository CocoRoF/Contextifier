# libs/core/processor/pdf_handler.py
"""
PDF Handler - Adaptive Complexity-based PDF Processor

=============================================================================
Core Features:
=============================================================================
1. Complexity Analysis - Calculate complexity scores per page/region
2. Adaptive Processing Strategy - Select optimal strategy based on complexity
3. Block Imaging - Render complex regions as images
4. Local Storage - Save imaged blocks locally and generate [image:{path}] tags
5. Multi-column Layout - Handle newspaper/magazine style multi-column layouts
6. Text Quality Analysis - Automatic vector text quality evaluation

=============================================================================
Core Algorithms:
=============================================================================
1. Line Analysis:
   - Extract all lines from drawings/rects
   - Classify by line thickness (thin < 0.5pt, normal 0.5-2pt, thick > 2pt)
   - Merge adjacent double lines (gap < 5pt)
   - Recover incomplete borders (complete 4 sides when 3+ exist)

2. Table Detection:
   - Strategy 1: PyMuPDF find_tables() - Calculate confidence score
   - Strategy 2: pdfplumber - Calculate confidence score
   - Strategy 3: Line analysis based grid construction - Calculate confidence score
   - Select highest confidence strategy or merge results

3. Cell Analysis:
   - Extract physical cell bbox
   - Grid line mapping (tolerance based)
   - Precise rowspan/colspan calculation
   - Merge validation based on text position

4. Annotation Integration:
   - Detect annotation rows immediately after tables (e.g., "Note: ...")
   - Collect footnote/endnote text
   - Integrate appropriately into table data
"""
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Set

# Import from new modular helpers
from libs.core.processor.pdf_helpers.pdf_metadata import (
    extract_pdf_metadata,
    format_metadata,
)
from libs.core.processor.pdf_helpers.pdf_utils import (
    bbox_overlaps,
)
from libs.core.processor.pdf_helpers.pdf_image import (
    extract_images_from_page,
)
from libs.core.processor.pdf_helpers.pdf_text_extractor import (
    extract_text_blocks,
)
from libs.core.processor.pdf_helpers.pdf_page_analyzer import (
    detect_page_border,
    is_table_likely_border,
)
from libs.core.processor.pdf_helpers.pdf_element_merger import (
    merge_page_elements,
)
from libs.core.processor.pdf_helpers.pdf_table_processor import (
    extract_all_tables,
)

# Modularized component imports
from libs.core.processor.pdf_helpers.types import (
    TableDetectionStrategy as TableDetectionStrategyType,
    ElementType,
    PageElement,
    PageBorderInfo,
)
from libs.core.processor.pdf_helpers.pdf_vector_text_ocr import (
    VectorTextOCREngine,
)

# Complexity analysis module
from libs.core.processor.pdf_helpers.pdf_complexity_analyzer import (
    ComplexityAnalyzer,
    ProcessingStrategy,
    PageComplexity,
)
from libs.core.processor.pdf_helpers.pdf_block_image_engine import (
    BlockImageEngine,
    MultiBlockResult,
)

from libs.core.processor.pdf_helpers.pdf_table_quality_analyzer import (
    TableQualityAnalyzer,
    TableQuality,
)

logger = logging.getLogger("document-processor")

# PyMuPDF import
import fitz


# Enum aliases for backward compatibility
TableDetectionStrategy = TableDetectionStrategyType


# ============================================================================
# Main Function
# ============================================================================

def extract_text_from_pdf(
    file_path: str,
    current_config: Dict[str, Any] = None,
    extract_default_metadata: bool = True
) -> str:
    """
    PDF text extraction (adaptive complexity-based processing).

    Analyzes page complexity first and selects optimal processing strategy:
    - SIMPLE: Standard text extraction
    - MODERATE: Hybrid processing (text + partial OCR)
    - COMPLEX: Block imaging + OCR
    - EXTREME: Full page OCR

    Args:
        file_path: PDF file path
        current_config: Configuration dictionary
        extract_default_metadata: Whether to extract metadata (default: True)

    Returns:
        Extracted text (including inline image tags, table HTML)
    """
    if current_config is None:
        current_config = {}

    logger.info(f"[PDF] Processing: {file_path}")
    return _extract_pdf(file_path, current_config, extract_default_metadata)


# ============================================================================
# Core Processing Logic
# ============================================================================

def _extract_pdf(
    file_path: str,
    current_config: Dict[str, Any],
    extract_default_metadata: bool = True
) -> str:
    """
    Enhanced PDF processing - adaptive complexity-based.

    Processing order:
    1. Open document and extract metadata
    2. For each page:
       a. Complexity analysis
       b. Determine processing strategy
       c. Process according to strategy:
          - TEXT_EXTRACTION: Standard text extraction
          - HYBRID: Text + partial OCR
          - BLOCK_IMAGE_OCR: Complex region imaging + OCR
          - FULL_PAGE_OCR: Full page OCR
       d. Integrate results
    3. Generate and integrate final HTML
    """
    try:
        doc = fitz.open(file_path)
        all_pages_text = []
        processed_images: Set[int] = set()

        # Extract metadata (only if extract_default_metadata is True)
        if extract_default_metadata:
            metadata = extract_pdf_metadata(doc)
            metadata_text = format_metadata(metadata)
            if metadata_text:
                all_pages_text.append(metadata_text)

        # Extract all document tables
        all_tables = _extract_all_tables(doc, file_path)

        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]

            logger.debug(f"[PDF] Processing page {page_num + 1}")

            # Phase 0: Complexity analysis
            complexity_analyzer = ComplexityAnalyzer(page, page_num)
            page_complexity = complexity_analyzer.analyze()

            logger.info(f"[PDF] Page {page_num + 1}: "
                       f"complexity={page_complexity.overall_complexity.name}, "
                       f"score={page_complexity.overall_score:.2f}, "
                       f"strategy={page_complexity.recommended_strategy.name}")

            # Branch by processing strategy
            strategy = page_complexity.recommended_strategy

            if strategy == ProcessingStrategy.FULL_PAGE_OCR:
                # Full page OCR
                page_text = _process_page_full_ocr(
                    page, page_num, doc, processed_images, all_tables
                )
            elif strategy == ProcessingStrategy.BLOCK_IMAGE_OCR:
                # Complex region block imaging + OCR
                page_text = _process_page_block_ocr(
                    page, page_num, doc, processed_images, all_tables,
                    page_complexity.complex_regions
                )
            elif strategy == ProcessingStrategy.HYBRID:
                # Hybrid (text + partial OCR)
                page_text = _process_page_hybrid(
                    page, page_num, doc, processed_images, all_tables,
                    page_complexity
                )
            else:
                # TEXT_EXTRACTION: Standard text extraction
                page_text = _process_page_text_extraction(
                    page, page_num, doc, processed_images, all_tables
                )

            if page_text.strip():
                all_pages_text.append(f"<Page {page_num + 1}>\n{page_text}\n</Page {page_num + 1}>")

        doc.close()

        final_text = "\n\n".join(all_pages_text)
        logger.info(f"[PDF] Extracted {len(final_text)} chars from {file_path}")

        return final_text

    except Exception as e:
        logger.error(f"[PDF] Error processing {file_path}: {e}")
        logger.debug(traceback.format_exc())
        raise


def _process_page_text_extraction(
    page, page_num: int, doc, processed_images: Set[int],
    all_tables: Dict[int, List[PageElement]]
) -> str:
    """
    TEXT_EXTRACTION strategy - standard text extraction.
    Suitable for simple pages.
    """
    page_elements: List[PageElement] = []

    # 1. Page border analysis
    border_info = _detect_page_border(page)

    # 1.5. Vector text (Outlined/Path Text) detection and OCR
    vector_text_engine = VectorTextOCREngine(page, page_num)
    vector_text_regions = vector_text_engine.detect_and_extract()

    for region in vector_text_regions:
        if region.ocr_text and region.confidence > 0.3:
            page_elements.append(PageElement(
                element_type=ElementType.TEXT,
                content=region.ocr_text,
                bbox=region.bbox,
                page_num=page_num
            ))

    # 2. Get tables for this page
    page_tables = all_tables.get(page_num, [])
    for table_element in page_tables:
        page_elements.append(table_element)

    # 3. Calculate table regions (for text filtering)
    table_bboxes = [elem.bbox for elem in page_tables]

    # 4. Extract text (excluding table regions)
    text_elements = _extract_text_blocks(page, page_num, table_bboxes, border_info)
    page_elements.extend(text_elements)

    # 5. Extract images
    image_elements = _extract_images_from_page(
        page, page_num, doc, processed_images, table_bboxes
    )
    page_elements.extend(image_elements)

    # 6. Sort and merge elements
    return _merge_page_elements(page_elements)


def _process_page_hybrid(
    page, page_num: int, doc, processed_images: Set[int],
    all_tables: Dict[int, List[PageElement]],
    page_complexity: PageComplexity
) -> str:
    """
    HYBRID strategy - text extraction + complex region imaging.
    Suitable for medium complexity pages.
    Complex regions are converted to [image:{path}] format.
    """
    page_elements: List[PageElement] = []

    # 1. Basic text extraction
    border_info = _detect_page_border(page)

    # Vector text OCR
    vector_text_engine = VectorTextOCREngine(page, page_num)
    vector_text_regions = vector_text_engine.detect_and_extract()

    for region in vector_text_regions:
        if region.ocr_text and region.confidence > 0.3:
            page_elements.append(PageElement(
                element_type=ElementType.TEXT,
                content=region.ocr_text,
                bbox=region.bbox,
                page_num=page_num
            ))

    # 2. Get tables
    page_tables = all_tables.get(page_num, [])
    for table_element in page_tables:
        page_elements.append(table_element)

    table_bboxes = [elem.bbox for elem in page_tables]

    # 3. Separate complex and simple regions
    complex_bboxes = page_complexity.complex_regions

    # 4. Simple regions: text extraction
    text_elements = _extract_text_blocks(page, page_num, table_bboxes, border_info)

    # Use only text that doesn't overlap with complex regions
    for elem in text_elements:
        is_in_complex = False
        for complex_bbox in complex_bboxes:
            if _bbox_overlaps(elem.bbox, complex_bbox):
                is_in_complex = True
                break
        if not is_in_complex:
            page_elements.append(elem)

    # 5. Complex regions: block imaging → local save → [image:path] tag
    if complex_bboxes:
        block_engine = BlockImageEngine(page, page_num)

        for complex_bbox in complex_bboxes:
            result = block_engine.process_region(complex_bbox, region_type="complex_region")

            if result.success and result.image_tag:
                page_elements.append(PageElement(
                    element_type=ElementType.IMAGE,
                    content=result.image_tag,
                    bbox=complex_bbox,
                    page_num=page_num
                ))

    # 6. Extract images
    image_elements = _extract_images_from_page(
        page, page_num, doc, processed_images, table_bboxes
    )
    page_elements.extend(image_elements)

    # 7. Sort and merge elements
    return _merge_page_elements(page_elements)


def _process_page_block_ocr(
    page, page_num: int, doc, processed_images: Set[int],
    all_tables: Dict[int, List[PageElement]],
    complex_regions: List[Tuple[float, float, float, float]]
) -> str:
    """
    BLOCK_IMAGE_OCR strategy - render complex regions as images and save locally.
    Suitable for complex pages.
    Complex regions are converted to [image:{path}] format.
    """
    page_elements: List[PageElement] = []

    # 1. Get tables
    page_tables = all_tables.get(page_num, [])
    for table_element in page_tables:
        page_elements.append(table_element)

    table_bboxes = [elem.bbox for elem in page_tables]

    # 2. Complex regions: block imaging → local save → [image:path] tag
    if complex_regions:
        block_engine = BlockImageEngine(page, page_num)

        for complex_bbox in complex_regions:
            # Skip if overlaps with table region
            if any(_bbox_overlaps(complex_bbox, tb) for tb in table_bboxes):
                continue

            result = block_engine.process_region(complex_bbox, region_type="complex_region")

            if result.success and result.image_tag:
                page_elements.append(PageElement(
                    element_type=ElementType.IMAGE,
                    content=result.image_tag,
                    bbox=complex_bbox,
                    page_num=page_num
                ))

    # 3. Simple regions: text extraction
    border_info = _detect_page_border(page)
    text_elements = _extract_text_blocks(page, page_num, table_bboxes, border_info)

    for elem in text_elements:
        is_in_complex = any(
            _bbox_overlaps(elem.bbox, cr) for cr in complex_regions
        )
        if not is_in_complex:
            page_elements.append(elem)

    # 4. Extract images
    image_elements = _extract_images_from_page(
        page, page_num, doc, processed_images, table_bboxes
    )
    page_elements.extend(image_elements)

    return _merge_page_elements(page_elements)


def _process_page_full_ocr(
    page, page_num: int, doc, processed_images: Set[int],
    all_tables: Dict[int, List[PageElement]]
) -> str:
    """
    FULL_PAGE_OCR strategy - advanced smart block processing.

    Suitable for extremely complex pages (newspapers, magazines with multi-column layouts).

    Improvements:
    - Analyze table quality first, extract processable tables as text/structure
    - Select optimal processing strategy per block
    - Image conversion only for truly necessary regions

    Processing flow:
    1. First analyze table quality to check processability
    2. Extract processable tables structurally
    3. Only image remaining complex regions
    """
    page_elements: List[PageElement] = []

    # Phase 1: Table quality analysis
    table_quality_analyzer = TableQualityAnalyzer(page)
    table_quality_result = table_quality_analyzer.analyze_page_tables()

    processable_tables: List[PageElement] = []
    unprocessable_table_bboxes: List[Tuple] = []

    if table_quality_result and table_quality_result.get('table_candidates'):
        for table_info in table_quality_result['table_candidates']:
            quality = table_info.get('quality', TableQuality.UNPROCESSABLE)
            bbox = table_info.get('bbox')

            # EXCELLENT, GOOD, MODERATE = processable
            if quality in (TableQuality.EXCELLENT, TableQuality.GOOD, TableQuality.MODERATE):
                # Processable table → structured extraction
                logger.info(f"[PDF] Page {page_num + 1}: Processable table found "
                           f"(quality={quality.name}) at {bbox}")
            else:
                # Unprocessable table (POOR, UNPROCESSABLE) → image target
                if bbox:
                    unprocessable_table_bboxes.append(bbox)

    # Phase 2: If processable tables exist, try structured extraction
    page_tables = all_tables.get(page_num, [])
    has_processable_tables = len(page_tables) > 0 or (
        table_quality_result and
        any(t.get('quality') in (TableQuality.EXCELLENT, TableQuality.GOOD, TableQuality.MODERATE)
            for t in table_quality_result.get('table_candidates', []))
    )

    if has_processable_tables:
        logger.info(f"[PDF] Page {page_num + 1}: Found processable tables, "
                   f"using hybrid extraction instead of full OCR")

        # Add tables as page elements
        table_bboxes = [elem.bbox for elem in page_tables]
        for table_element in page_tables:
            page_elements.append(table_element)

        # Extract text outside table regions
        border_info = _detect_page_border(page)
        text_elements = _extract_text_blocks(page, page_num, table_bboxes, border_info)
        page_elements.extend(text_elements)

        # Extract images outside table regions
        image_elements = _extract_images_from_page(
            page, page_num, doc, processed_images, table_bboxes
        )
        page_elements.extend(image_elements)

        logger.info(f"[PDF] Page {page_num + 1}: Hybrid extraction completed - "
                   f"tables={len(page_tables)}, text_blocks={len(text_elements)}, "
                   f"images={len(image_elements)}")

        return _merge_page_elements(page_elements)

    # Phase 3: If table processing not possible, use smart block processing
    block_engine = BlockImageEngine(page, page_num)
    multi_result: MultiBlockResult = block_engine.process_page_smart()

    if multi_result.success and multi_result.block_results:
        # Convert per-block image tags to page elements
        for block_result in multi_result.block_results:
            if block_result.success and block_result.image_tag:
                page_elements.append(PageElement(
                    element_type=ElementType.IMAGE,
                    content=block_result.image_tag,
                    bbox=block_result.bbox,
                    page_num=page_num
                ))

        logger.info(f"[PDF] Page {page_num + 1}: Smart block processing - "
                   f"strategy={multi_result.strategy_used.name}, "
                   f"blocks={multi_result.successful_blocks}/{multi_result.total_blocks}")
    else:
        # Fallback: full page imaging
        logger.warning(f"[PDF] Page {page_num + 1}: Smart processing failed, "
                      f"falling back to full page image")

        result = block_engine.process_full_page(region_type="full_page")

        if result.success and result.image_tag:
            page_elements.append(PageElement(
                element_type=ElementType.IMAGE,
                content=result.image_tag,
                bbox=(0, 0, page.rect.width, page.rect.height),
                page_num=page_num
            ))
            logger.info(f"[PDF] Page {page_num + 1}: Full page image saved: {result.image_path}")
        else:
            # Last resort fallback: text extraction
            logger.warning(f"[PDF] Page {page_num + 1}: Full page image failed, "
                          f"falling back to text extraction")
            border_info = _detect_page_border(page)
            page_tables = all_tables.get(page_num, [])
            table_bboxes = [elem.bbox for elem in page_tables]

            for table_element in page_tables:
                page_elements.append(table_element)

            text_elements = _extract_text_blocks(page, page_num, table_bboxes, border_info)
            page_elements.extend(text_elements)

            image_elements = _extract_images_from_page(
                page, page_num, doc, processed_images, table_bboxes
            )
            page_elements.extend(image_elements)

    return _merge_page_elements(page_elements)


# ============================================================================
# Wrapper Functions (delegate to new modules)
# ============================================================================

def _bbox_overlaps(bbox1: Tuple, bbox2: Tuple) -> bool:
    """Check if two bboxes overlap. Delegates to pdf_utils.bbox_overlaps."""
    return bbox_overlaps(bbox1, bbox2)


def _extract_all_tables(doc, file_path: str) -> Dict[int, List[PageElement]]:
    """Extracts tables from entire document. Delegates to pdf_table_processor."""
    return extract_all_tables(doc, file_path, _detect_page_border, _is_table_likely_border)


def _detect_page_border(page) -> PageBorderInfo:
    """Detects page borders. Delegates to pdf_page_analyzer."""
    return detect_page_border(page)


def _is_table_likely_border(table_bbox: Tuple, border_info: PageBorderInfo, page) -> bool:
    """Check if a table is likely a page border. Delegates to pdf_page_analyzer."""
    return is_table_likely_border(table_bbox, border_info, page)


def _extract_text_blocks(
    page,
    page_num: int,
    table_bboxes: List[Tuple[float, float, float, float]],
    border_info: PageBorderInfo,
    use_quality_check: bool = True
) -> List[PageElement]:
    """Extract text blocks. Delegates to pdf_text_extractor."""
    return extract_text_blocks(page, page_num, table_bboxes, border_info, use_quality_check)


def _extract_images_from_page(
    page,
    page_num: int,
    doc,
    processed_images: Set[int],
    table_bboxes: List[Tuple[float, float, float, float]],
    min_image_size: int = 50,
    min_image_area: int = 2500
) -> List[PageElement]:
    """Extract images from page. Delegates to pdf_image."""
    return extract_images_from_page(
        page, page_num, doc, processed_images, table_bboxes,
        min_image_size, min_image_area
    )


def _merge_page_elements(elements: List[PageElement]) -> str:
    """Merge page elements. Delegates to pdf_element_merger."""
    return merge_page_elements(elements)
