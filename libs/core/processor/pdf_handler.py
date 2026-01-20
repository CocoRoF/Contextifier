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
Class-based Handler:
=============================================================================
PDFHandler class inherits from BaseHandler and manages config/image_processor
at instance level. All internal methods can access these via self.

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
from typing import Any, Dict, List, Optional, Tuple, Set, TYPE_CHECKING

# Base handler
from libs.core.processor.base_handler import BaseHandler
from libs.core.functions.img_processor import ImageProcessor

if TYPE_CHECKING:
    from libs.core.document_processor import CurrentFile

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
# PDFHandler Class
# ============================================================================

class PDFHandler(BaseHandler):
    """
    PDF Document Handler
    
    Inherits from BaseHandler to manage config and image_processor at instance level.
    All internal methods access these via self.config, self.image_processor.
    
    Usage:
        handler = PDFHandler(config=config, image_processor=image_processor)
        text = handler.extract_text(current_file)
    """
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from PDF file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        self.logger.info(f"[PDF] Processing: {file_path}")
        return self._extract_pdf(current_file, extract_metadata)
    
    def _extract_pdf(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True
    ) -> str:
        """
        Enhanced PDF processing - adaptive complexity-based.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        
        try:
            # Open PDF from stream to avoid path encoding issues
            file_stream = self.get_file_stream(current_file)
            doc = fitz.open(stream=file_stream, filetype="pdf")
            all_pages_text = []
            processed_images: Set[int] = set()

            # Extract metadata
            if extract_metadata:
                metadata = extract_pdf_metadata(doc)
                metadata_text = format_metadata(metadata)
                if metadata_text:
                    all_pages_text.append(metadata_text)

            # Extract all document tables
            # NOTE: file_path is passed for pdfplumber compatibility
            all_tables = self._extract_all_tables(doc, file_path)

            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                self.logger.debug(f"[PDF] Processing page {page_num + 1}")

                # Complexity analysis
                complexity_analyzer = ComplexityAnalyzer(page, page_num)
                page_complexity = complexity_analyzer.analyze()

                self.logger.info(f"[PDF] Page {page_num + 1}: "
                           f"complexity={page_complexity.overall_complexity.name}, "
                           f"score={page_complexity.overall_score:.2f}, "
                           f"strategy={page_complexity.recommended_strategy.name}")

                # Branch by processing strategy
                strategy = page_complexity.recommended_strategy

                if strategy == ProcessingStrategy.FULL_PAGE_OCR:
                    page_text = self._process_page_full_ocr(
                        page, page_num, doc, processed_images, all_tables
                    )
                elif strategy == ProcessingStrategy.BLOCK_IMAGE_OCR:
                    page_text = self._process_page_block_ocr(
                        page, page_num, doc, processed_images, all_tables,
                        page_complexity.complex_regions
                    )
                elif strategy == ProcessingStrategy.HYBRID:
                    page_text = self._process_page_hybrid(
                        page, page_num, doc, processed_images, all_tables,
                        page_complexity
                    )
                else:
                    page_text = self._process_page_text_extraction(
                        page, page_num, doc, processed_images, all_tables
                    )

                if page_text.strip():
                    page_tag = self.create_page_tag(page_num + 1)
                    all_pages_text.append(f"{page_tag}\n{page_text}")

            doc.close()

            final_text = "\n\n".join(all_pages_text)
            self.logger.info(f"[PDF] Extracted {len(final_text)} chars from {file_path}")

            return final_text

        except Exception as e:
            self.logger.error(f"[PDF] Error processing {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def _process_page_text_extraction(
        self, page, page_num: int, doc, processed_images: Set[int],
        all_tables: Dict[int, List[PageElement]]
    ) -> str:
        """TEXT_EXTRACTION strategy - standard text extraction."""
        page_elements: List[PageElement] = []

        border_info = detect_page_border(page)

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

        page_tables = all_tables.get(page_num, [])
        for table_element in page_tables:
            page_elements.append(table_element)

        table_bboxes = [elem.bbox for elem in page_tables]

        text_elements = extract_text_blocks(page, page_num, table_bboxes, border_info)
        page_elements.extend(text_elements)

        image_elements = self._extract_images_from_page(
            page, page_num, doc, processed_images, table_bboxes
        )
        page_elements.extend(image_elements)

        return merge_page_elements(page_elements)

    def _process_page_hybrid(
        self, page, page_num: int, doc, processed_images: Set[int],
        all_tables: Dict[int, List[PageElement]],
        page_complexity: PageComplexity
    ) -> str:
        """HYBRID strategy - text extraction + complex region imaging."""
        page_elements: List[PageElement] = []

        border_info = detect_page_border(page)

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

        page_tables = all_tables.get(page_num, [])
        for table_element in page_tables:
            page_elements.append(table_element)

        table_bboxes = [elem.bbox for elem in page_tables]
        complex_bboxes = page_complexity.complex_regions

        text_elements = extract_text_blocks(page, page_num, table_bboxes, border_info)

        for elem in text_elements:
            is_in_complex = False
            for complex_bbox in complex_bboxes:
                if bbox_overlaps(elem.bbox, complex_bbox):
                    is_in_complex = True
                    break
            if not is_in_complex:
                page_elements.append(elem)

        if complex_bboxes:
            block_engine = BlockImageEngine(page, page_num, image_processor=self.image_processor)

            for complex_bbox in complex_bboxes:
                result = block_engine.process_region(complex_bbox, region_type="complex_region")

                if result.success and result.image_tag:
                    page_elements.append(PageElement(
                        element_type=ElementType.IMAGE,
                        content=result.image_tag,
                        bbox=complex_bbox,
                        page_num=page_num
                    ))

        image_elements = self._extract_images_from_page(
            page, page_num, doc, processed_images, table_bboxes
        )
        page_elements.extend(image_elements)

        return merge_page_elements(page_elements)

    def _process_page_block_ocr(
        self, page, page_num: int, doc, processed_images: Set[int],
        all_tables: Dict[int, List[PageElement]],
        complex_regions: List[Tuple[float, float, float, float]]
    ) -> str:
        """BLOCK_IMAGE_OCR strategy - render complex regions as images."""
        page_elements: List[PageElement] = []

        page_tables = all_tables.get(page_num, [])
        for table_element in page_tables:
            page_elements.append(table_element)

        table_bboxes = [elem.bbox for elem in page_tables]

        if complex_regions:
            block_engine = BlockImageEngine(page, page_num, image_processor=self.image_processor)

            for complex_bbox in complex_regions:
                if any(bbox_overlaps(complex_bbox, tb) for tb in table_bboxes):
                    continue

                result = block_engine.process_region(complex_bbox, region_type="complex_region")

                if result.success and result.image_tag:
                    page_elements.append(PageElement(
                        element_type=ElementType.IMAGE,
                        content=result.image_tag,
                        bbox=complex_bbox,
                        page_num=page_num
                    ))

        border_info = detect_page_border(page)
        text_elements = extract_text_blocks(page, page_num, table_bboxes, border_info)

        for elem in text_elements:
            is_in_complex = any(
                bbox_overlaps(elem.bbox, cr) for cr in complex_regions
            )
            if not is_in_complex:
                page_elements.append(elem)

        image_elements = self._extract_images_from_page(
            page, page_num, doc, processed_images, table_bboxes
        )
        page_elements.extend(image_elements)

        return merge_page_elements(page_elements)

    def _process_page_full_ocr(
        self, page, page_num: int, doc, processed_images: Set[int],
        all_tables: Dict[int, List[PageElement]]
    ) -> str:
        """FULL_PAGE_OCR strategy - advanced smart block processing."""
        page_elements: List[PageElement] = []

        # Table quality analysis
        table_quality_analyzer = TableQualityAnalyzer(page)
        table_quality_result = table_quality_analyzer.analyze_page_tables()

        unprocessable_table_bboxes: List[Tuple] = []

        if table_quality_result and table_quality_result.get('table_candidates'):
            for table_info in table_quality_result['table_candidates']:
                quality = table_info.get('quality', TableQuality.UNPROCESSABLE)
                bbox = table_info.get('bbox')

                if quality not in (TableQuality.EXCELLENT, TableQuality.GOOD, TableQuality.MODERATE):
                    if bbox:
                        unprocessable_table_bboxes.append(bbox)

        page_tables = all_tables.get(page_num, [])
        has_processable_tables = len(page_tables) > 0 or (
            table_quality_result and
            any(t.get('quality') in (TableQuality.EXCELLENT, TableQuality.GOOD, TableQuality.MODERATE)
                for t in table_quality_result.get('table_candidates', []))
        )

        if has_processable_tables:
            self.logger.info(f"[PDF] Page {page_num + 1}: Found processable tables, "
                           f"using hybrid extraction")

            table_bboxes = [elem.bbox for elem in page_tables]
            for table_element in page_tables:
                page_elements.append(table_element)

            border_info = detect_page_border(page)
            text_elements = extract_text_blocks(page, page_num, table_bboxes, border_info)
            page_elements.extend(text_elements)

            image_elements = self._extract_images_from_page(
                page, page_num, doc, processed_images, table_bboxes
            )
            page_elements.extend(image_elements)

            return merge_page_elements(page_elements)

        # Smart block processing
        block_engine = BlockImageEngine(page, page_num, image_processor=self.image_processor)
        multi_result: MultiBlockResult = block_engine.process_page_smart()

        if multi_result.success and multi_result.block_results:
            for block_result in multi_result.block_results:
                if block_result.success and block_result.image_tag:
                    page_elements.append(PageElement(
                        element_type=ElementType.IMAGE,
                        content=block_result.image_tag,
                        bbox=block_result.bbox,
                        page_num=page_num
                    ))

            self.logger.info(f"[PDF] Page {page_num + 1}: Smart block processing - "
                           f"strategy={multi_result.strategy_used.name}, "
                           f"blocks={multi_result.successful_blocks}/{multi_result.total_blocks}")
        else:
            self.logger.warning(f"[PDF] Page {page_num + 1}: Smart processing failed, "
                              f"falling back to full page image")

            result = block_engine.process_full_page(region_type="full_page")

            if result.success and result.image_tag:
                page_elements.append(PageElement(
                    element_type=ElementType.IMAGE,
                    content=result.image_tag,
                    bbox=(0, 0, page.rect.width, page.rect.height),
                    page_num=page_num
                ))
            else:
                self.logger.warning(f"[PDF] Page {page_num + 1}: Full page image failed")
                border_info = detect_page_border(page)
                page_tables = all_tables.get(page_num, [])
                table_bboxes = [elem.bbox for elem in page_tables]

                for table_element in page_tables:
                    page_elements.append(table_element)

                text_elements = extract_text_blocks(page, page_num, table_bboxes, border_info)
                page_elements.extend(text_elements)

                image_elements = self._extract_images_from_page(
                    page, page_num, doc, processed_images, table_bboxes
                )
                page_elements.extend(image_elements)

        return merge_page_elements(page_elements)

    def _extract_all_tables(self, doc, file_path: str) -> Dict[int, List[PageElement]]:
        """Extract tables from entire document."""
        return extract_all_tables(doc, file_path, detect_page_border, is_table_likely_border)

    def _extract_images_from_page(
        self, page, page_num: int, doc, processed_images: Set[int],
        table_bboxes: List[Tuple[float, float, float, float]],
        min_image_size: int = 50,
        min_image_area: int = 2500
    ) -> List[PageElement]:
        """Extract images from page using instance's image_processor."""
        return extract_images_from_page(
            page, page_num, doc, processed_images, table_bboxes,
            image_processor=self.image_processor,
            min_image_size=min_image_size, min_image_area=min_image_area
        )


# ============================================================================
# Legacy Function Interface (for backward compatibility)
# ============================================================================

def extract_text_from_pdf(
    file_path: str,
    current_config: Dict[str, Any] = None,
    extract_default_metadata: bool = True
) -> str:
    """
    PDF text extraction (legacy function interface).
    
    This function creates a PDFHandler instance and delegates to it.
    For new code, consider using PDFHandler class directly.

    Args:
        file_path: PDF file path
        current_config: Configuration dictionary
        extract_default_metadata: Whether to extract metadata (default: True)

    Returns:
        Extracted text (including inline image tags, table HTML)
    """
    if current_config is None:
        current_config = {}
    
    # Extract image_processor from config if available
    image_processor = current_config.get("image_processor")
    
    # Create handler instance with config and image_processor
    handler = PDFHandler(config=current_config, image_processor=image_processor)
    
    return handler.extract_text(file_path, extract_metadata=extract_default_metadata)


# ============================================================================
# Deprecated Legacy Functions (kept for reference, not used)
# ============================================================================

def _extract_pdf(
    file_path: str,
    current_config: Dict[str, Any],
    extract_default_metadata: bool = True
) -> str:
    """Deprecated: Use PDFHandler.extract_text() instead."""
    return extract_text_from_pdf(file_path, current_config, extract_default_metadata)

