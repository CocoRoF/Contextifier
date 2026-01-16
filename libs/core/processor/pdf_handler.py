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
Architecture:
=============================================================================
┌─────────────────────────────────────────────────────────────────────────┐
│                         PDF Document Input                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 0: Complexity Analysis                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Drawing       │  │ Text Quality  │  │ Layout        │                │
│  │ Density       │  │ Analysis      │  │ Complexity    │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
│                    ↓ Processing Strategy Selection ↓                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ SIMPLE → TEXT_EXTRACTION | COMPLEX → BLOCK_IMAGE              │    │
│  │ MODERATE → HYBRID        | EXTREME → FULL_PAGE_IMAGE          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 1: Line Analysis Engine                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Thin Lines    │  │ Double Lines  │  │ Incomplete    │                │
│  │ Detection     │  │ Merger        │  │ Border Fix    │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 2: Table Detection Engine                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ PyMuPDF       │  │ pdfplumber    │  │ Line-Based    │                │
│  │ Strategy      │  │ Strategy      │  │ Strategy      │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
│                    ↓ Confidence Scoring & Selection ↓                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 3: Cell Analysis Engine                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Physical Cell │  │ Text Position │  │ Merge Cell    │                │
│  │ Detection     │  │ Analysis      │  │ Calculation   │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 4: Annotation Integration                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Footnote      │  │ Endnote       │  │ Table Note    │                │
│  │ Detection     │  │ Detection     │  │ Integration   │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 4.5: Block Image Upload                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Complex Region│  │ High-DPI      │  │ Local         │                │
│  │ Detection     │  │ Rendering     │  │ Upload        │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 5: HTML Generation                              │
│  ┌───────────────────────────────────────────────────────────┐          │
│  │ Semantic HTML with rowspan/colspan/accessibility          │          │
│  └───────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘

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
import copy
import traceback
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from libs.core.processor.pdf_helpers.pdf_helper import (
    extract_pdf_metadata,
    format_metadata,
    escape_html,
    is_inside_any_bbox,
    find_image_position,
    get_text_lines_with_positions,
)

# Image processing module
from libs.core.functions.img_processor import ImageProcessor

# Module level image processor
_image_processor = ImageProcessor(
    directory_path="temp/images",
    tag_prefix="[Image:",
    tag_suffix="]"
)

# Modularized component imports
from libs.core.processor.pdf_helpers.types import (
    TableDetectionStrategy as TableDetectionStrategyType,
    ElementType,
    PDFConfig,
    LineInfo,
    GridInfo,
    CellInfo,
    PageElement,
    PageBorderInfo,
)
from libs.core.processor.pdf_helpers.vector_text_ocr import (
    VectorTextOCREngine,
)
from libs.core.processor.pdf_helpers.table_detection import (
    TableDetectionEngine,
)
from libs.core.processor.pdf_helpers.cell_analysis import (
    CellAnalysisEngine,
)
from libs.core.processor.pdf_helpers.text_quality_analyzer import (
    TextQualityAnalyzer,
    QualityAwareTextExtractor,
    TextQualityConfig,
)

# Complexity analysis module
from libs.core.processor.pdf_helpers.complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityLevel,
    ProcessingStrategy,
    PageComplexity,
    ComplexityConfig,
)
from libs.core.processor.pdf_helpers.block_image_engine import (
    BlockStrategy,
    BlockImageEngine,
    BlockImageConfig,
    BlockImageResult,
    MultiBlockResult,
)
from libs.core.processor.pdf_helpers.layout_block_detector import (
    LayoutBlockDetector,
    LayoutBlockType,
    LayoutBlock,
    LayoutAnalysisResult,
)
from libs.core.processor.pdf_helpers.table_quality_analyzer import (
    TableQualityAnalyzer,
    TableQuality,
)

logger = logging.getLogger("document-processor")

# PyMuPDF import
import fitz


# ============================================================================
# Config Extension (PDFConfig based)
# ============================================================================

class PDFConfig(PDFConfig):
    """PDF processing config constants - defaults + additional settings"""
    # Line analysis
    THIN_LINE_THRESHOLD = 0.5      # pt
    THICK_LINE_THRESHOLD = 2.0     # pt
    DOUBLE_LINE_GAP = 5.0          # pt - Max gap to consider as double line
    LINE_MERGE_TOLERANCE = 3.0     # pt - Tolerance for same position judgment

    # Table detection additional settings
    MIN_CELL_SIZE = 10.0           # pt - Minimum cell size
    PAGE_BORDER_MARGIN = 0.1       # Border margin ratio relative to page size
    PAGE_SPANNING_RATIO = 0.85     # Ratio to consider as spanning across page

    # Incomplete border recovery
    BORDER_EXTENSION_MARGIN = 20.0  # pt - Margin when extending borders
    INCOMPLETE_BORDER_MIN_SIDES = 3  # Minimum sides for incomplete border judgment

    # Annotation detection
    ANNOTATION_Y_MARGIN = 30.0     # pt - Annotation search range below table
    ANNOTATION_PATTERNS = ['Note)', 'Note )', '※', '*', '†', '‡', '¹', '²', '³']

    # Vector text OCR settings (Outlined Text / Path Text)
    VECTOR_TEXT_MIN_ITEMS = 20     # Minimum drawing items for vector text judgment
    VECTOR_TEXT_MAX_HEIGHT = 30.0  # pt - Maximum height for vector text judgment
    VECTOR_TEXT_OCR_DPI = 300      # OCR image rendering DPI
    VECTOR_TEXT_OCR_SCALE = 4      # OCR image scale factor
    VECTOR_TEXT_OCR_LANG = 'kor+eng'  # Tesseract language setting

    # Grid regularity validation
    GRID_VARIANCE_THRESHOLD = 0.5          # Cell size variance threshold (lower = more regular)
    GRID_MIN_ORTHOGONAL_RATIO = 0.7        # Minimum orthogonal line (horizontal/vertical) ratio

    # Image/illustration area protection
    IMAGE_AREA_MARGIN = 5.0               # Image surrounding margin (pt)


class AdaptiveConfig:
    """Adaptive complexity-based processing config constants."""

    # ========== Complexity Analysis Settings ==========
    # Complexity thresholds
    COMPLEXITY_MODERATE_THRESHOLD = 0.3   # Above this = HYBRID processing
    COMPLEXITY_COMPLEX_THRESHOLD = 0.6    # Above this = BLOCK_IMAGE_OCR
    COMPLEXITY_EXTREME_THRESHOLD = 0.8    # Above this = FULL_PAGE_OCR

    # Drawing density (per 1000pt²)
    DRAWING_DENSITY_MODERATE = 0.5
    DRAWING_DENSITY_COMPLEX = 2.0
    DRAWING_DENSITY_EXTREME = 5.0

    # Image density
    IMAGE_DENSITY_MODERATE = 0.1
    IMAGE_DENSITY_COMPLEX = 0.3

    # Text quality thresholds
    TEXT_QUALITY_POOR = 0.7    # Below this = quality issue
    TEXT_QUALITY_BAD = 0.5     # Below this = OCR recommended

    # Layout complexity
    COLUMN_COUNT_MODERATE = 3   # Above this = multi-column layout
    COLUMN_COUNT_COMPLEX = 5    # Above this = complex multi-column

    # ========== Block Image Settings ==========
    BLOCK_IMAGE_DPI = 300              # OCR rendering DPI
    BLOCK_IMAGE_MAX_SIZE = 4096        # Max image size (px)

    # OCR settings
    OCR_LANGUAGE = 'kor+eng'           # Tesseract language
    OCR_CONFIG = '--oem 3 --psm 3'     # Tesseract config
    OCR_MIN_CONFIDENCE = 60.0          # Minimum confidence

    # Image preprocessing
    IMAGE_CONTRAST_ENHANCE = 1.5       # Contrast enhancement
    IMAGE_SHARPEN = True               # Apply sharpening

    # ========== Region Analysis Settings ==========
    REGION_GRID_SIZE = 200             # Analysis grid size (pt)
    MIN_COMPLEX_REGION_SIZE = 100      # Minimum complex region size (pt)
    COMPLEX_REGION_OVERLAP_RATIO = 0.5 # Complex region overlap ratio

    # ========== Processing Strategy Settings ==========
    # Auto strategy selection
    AUTO_STRATEGY_ENABLED = True

    # Force OCR conditions
    FORCE_OCR_TEXT_QUALITY = 0.4       # Force OCR if text quality below this
    FORCE_OCR_BROKEN_RATIO = 0.2       # Force OCR if broken char ratio above this


# Enum aliases for backward compatibility
# Enum aliases for backward compatibility
TableDetectionStrategy = TableDetectionStrategyType


# ============================================================================
# Internal Type Definitions
# ============================================================================

@dataclass
class TableCandidate:
    """Table candidate - internal use."""
    strategy: TableDetectionStrategy
    confidence: float
    bbox: Tuple[float, float, float, float]
    grid: Optional[GridInfo]
    cells: List[CellInfo]
    data: List[List[Optional[str]]]
    raw_table: Any = None  # Original table object

    @property
    def row_count(self) -> int:
        return len(self.data)

    @property
    def col_count(self) -> int:
        return max(len(row) for row in self.data) if self.data else 0


@dataclass
class AnnotationInfo:
    """Annotation/footnote/endnote info."""
    text: str
    bbox: Tuple[float, float, float, float]
    type: str  # 'footnote', 'endnote', 'table_note'
    related_table_idx: Optional[int] = None


@dataclass
class PageElementExtended(PageElement):
    """Page element - extended."""

    @property
    def sort_key(self) -> Tuple[float, float]:
        """Sort key: (y0, x0)"""
        return (self.bbox[1], self.bbox[0])


@dataclass
class TableInfo:
    """Final table info."""
    page_num: int
    table_idx: int
    bbox: Tuple[float, float, float, float]
    data: List[List[Optional[str]]]
    col_count: int
    row_count: int
    page_height: float
    cells_info: Optional[List[Dict]] = None
    annotations: Optional[List[AnnotationInfo]] = None
    detection_strategy: Optional[TableDetectionStrategy] = None
    confidence: float = 1.0


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
    return _extract_pdf_enhanced(file_path, current_config, extract_default_metadata)


# ============================================================================
# Core Processing Logic
# ============================================================================

def _extract_pdf_enhanced(
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

        # 테이블을 페이지 요소로 추가
        table_bboxes = [elem.bbox for elem in page_tables]
        for table_element in page_tables:
            page_elements.append(table_element)

        # 테이블 영역 외의 텍스트 추출
        border_info = _detect_page_border(page)
        text_elements = _extract_text_blocks(page, page_num, table_bboxes, border_info)
        page_elements.extend(text_elements)

        # 테이블 영역 외의 이미지 추출
        image_elements = _extract_images_from_page(
            page, page_num, doc, processed_images, table_bboxes
        )
        page_elements.extend(image_elements)

        logger.info(f"[PDF] Page {page_num + 1}: Hybrid extraction completed - "
                   f"tables={len(page_tables)}, text_blocks={len(text_elements)}, "
                   f"images={len(image_elements)}")

        return _merge_page_elements(page_elements)

    # Phase 3: 테이블 처리가 불가능하면 스마트 블록 처리
    block_engine = BlockImageEngine(page, page_num)
    multi_result: MultiBlockResult = block_engine.process_page_smart()

    if multi_result.success and multi_result.block_results:
        # 블록별 이미지 태그를 페이지 요소로 변환
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
        # 폴백: 전체 페이지 이미지화
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
            # 최후의 폴백: 텍스트 추출
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


def _bbox_overlaps(bbox1: Tuple, bbox2: Tuple) -> bool:
    """두 bbox가 겹치는지 확인"""
    return not (
        bbox1[2] <= bbox2[0] or
        bbox1[0] >= bbox2[2] or
        bbox1[3] <= bbox2[1] or
        bbox1[1] >= bbox2[3]
    )


# ============================================================================
# 테이블 추출 함수
# ============================================================================

def _extract_all_tables(doc, file_path: str) -> Dict[int, List[PageElement]]:
    """
    문서 전체에서 테이블을 추출합니다.

    전략:
    1. 다중 전략 테이블 감지
    2. 신뢰도 기반 최선의 결과 선택
    3. 셀 분석 및 병합셀 처리
    4. 주석 통합
    5. 페이지 간 연속성 처리
    """
    tables_by_page: Dict[int, List[PageElement]] = {}
    all_table_infos: List[TableInfo] = []

    # 1단계: 각 페이지에서 테이블 감지
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height

        # 페이지 테두리 감지
        border_info = _detect_page_border(page)

        try:
            # 테이블 감지 엔진 사용
            detection_engine = TableDetectionEngine(page, page_num, file_path)
            candidates = detection_engine.detect_tables()

            for idx, candidate in enumerate(candidates):
                # 페이지 테두리와 겹치는지 확인
                if border_info.has_border and _is_table_likely_border(candidate.bbox, border_info, page):
                    logger.debug(f"[PDF] Skipping page border table: {candidate.bbox}")
                    continue

                # 셀 정보를 딕셔너리로 변환
                cells_info = None
                if candidate.cells:
                    cells_info = [
                        {
                            'row': cell.row,
                            'col': cell.col,
                            'rowspan': cell.rowspan,
                            'colspan': cell.colspan,
                            'bbox': cell.bbox
                        }
                        for cell in candidate.cells
                    ]

                table_info = TableInfo(
                    page_num=page_num,
                    table_idx=idx,
                    bbox=candidate.bbox,
                    data=candidate.data,
                    col_count=candidate.col_count,
                    row_count=candidate.row_count,
                    page_height=page_height,
                    cells_info=cells_info,
                    detection_strategy=candidate.strategy,
                    confidence=candidate.confidence
                )

                all_table_infos.append(table_info)

        except Exception as e:
            logger.debug(f"[PDF] Error detecting tables on page {page_num}: {e}")
            continue

    # 2단계: 인접 테이블 병합
    merged_tables = _merge_adjacent_tables(all_table_infos)

    # 3단계: 주석 행 찾아서 삽입
    merged_tables = _find_and_insert_annotations(doc, merged_tables)

    # 4단계: 테이블 연속성 처리
    processed_tables = _process_table_continuity(merged_tables)

    # 5단계: HTML 변환 및 PageElement 생성
    # 1열 테이블은 TEXT로, 2열 이상은 TABLE로 처리
    single_col_count = 0
    real_table_count = 0

    for table_info in processed_tables:
        try:
            page_num = table_info.page_num

            if page_num not in tables_by_page:
                tables_by_page[page_num] = []

            # 1열 테이블인지 확인
            if _is_single_column_table(table_info):
                # 1열 테이블: 텍스트 리스트로 변환하여 TEXT 타입으로 처리
                text_content = _convert_single_column_to_text(table_info)

                if text_content:
                    tables_by_page[page_num].append(PageElement(
                        element_type=ElementType.TEXT,
                        content=text_content,
                        bbox=table_info.bbox,
                        page_num=page_num
                    ))
                    single_col_count += 1
            else:
                # 2열 이상: HTML 테이블로 변환
                html_table = _convert_table_to_html(table_info)

                if html_table:
                    tables_by_page[page_num].append(PageElement(
                        element_type=ElementType.TABLE,
                        content=html_table,
                        bbox=table_info.bbox,
                        page_num=page_num
                    ))
                    real_table_count += 1

        except Exception as e:
            logger.debug(f"[PDF] Error converting table to HTML: {e}")
            continue

    if single_col_count > 0:
        logger.info(f"[PDF] Converted {single_col_count} single-column tables to text")
    logger.info(f"[PDF] Extracted {real_table_count} tables from {len(tables_by_page)} pages")
    return tables_by_page


# ============================================================================
# Phase 4: Annotation Integration
# ============================================================================

def _find_and_insert_annotations(doc, tables: List[TableInfo]) -> List[TableInfo]:
    """
    테이블 내부 및 직후에 주석/각주/미주를 찾아서 통합합니다.

    감지 패턴:
    1. 테이블 직후 "주)" 등으로 시작하는 행
    2. 테이블 내부의 서브헤더 행 (예: (A), (B))
    3. 각주/미주 표시 (※, *, †, ‡ 등)
    """
    if not tables:
        return tables

    result = []
    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)

    for table in tables:
        tables_by_page[table.page_num].append(table)

    for page_num, page_tables in tables_by_page.items():
        page = doc[page_num]
        page_height = page.rect.height

        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])
        text_lines = get_text_lines_with_positions(page)

        for i, table in enumerate(sorted_tables):
            table_top = table.bbox[1]
            table_bottom = table.bbox[3]
            table_left = table.bbox[0]
            table_right = table.bbox[2]

            next_table_top = sorted_tables[i + 1].bbox[1] if i + 1 < len(sorted_tables) else page_height

            # 1. 테이블 직후 주석 행 찾기
            annotation_lines = []
            for line in text_lines:
                # 테이블 바로 아래, 다음 테이블 전
                if table_bottom - 3 <= line['y0'] <= table_bottom + PDFConfig.ANNOTATION_Y_MARGIN:
                    if line['x0'] >= table_left - 10 and line['x1'] <= table_right + 10:
                        if line['y0'] < next_table_top - 20:
                            # 주석 패턴 확인
                            for pattern in PDFConfig.ANNOTATION_PATTERNS:
                                if line['text'].startswith(pattern):
                                    annotation_lines.append(line)
                                    break

            if annotation_lines:
                table = _add_annotation_to_table(table, annotation_lines, 'footer')
                logger.debug(f"[PDF] Added annotation to table on page {page_num + 1}")

            # 2. 서브헤더 행 찾기 (예: (A), (B)) - 이미 서브헤더가 없을 때만
            has_subheader = False
            if table.row_count >= 2 and table.data and len(table.data) >= 2:
                # 두 번째 행이 서브헤더 패턴인지 확인
                second_row = table.data[1] if len(table.data) > 1 else []
                for cell in second_row:
                    if cell and ('(A)' in str(cell) or '(B)' in str(cell)):
                        has_subheader = True
                        break

            if not has_subheader and table.row_count >= 2 and table.data:
                row_height_estimate = (table_bottom - table_top) / table.row_count
                header_bottom_estimate = table_top + row_height_estimate
                second_row_top_estimate = table_top + row_height_estimate * 2

                subheader_lines = []
                for line in text_lines:
                    if header_bottom_estimate - 5 <= line['y0'] <= second_row_top_estimate - 5:
                        if line['x0'] >= table_left - 10 and line['x1'] <= table_right + 10:
                            # (A), (B) 패턴 확인
                            if '(A)' in line['text'] or '(B)' in line['text']:
                                subheader_lines.append(line)

                if subheader_lines:
                    table = _add_annotation_to_table(table, subheader_lines, 'subheader')
                    logger.debug(f"[PDF] Added subheader to table on page {page_num + 1}")

            result.append(table)

    result.sort(key=lambda t: (t.page_num, t.bbox[1]))
    return result


def _add_annotation_to_table(table: TableInfo, text_lines: List[Dict], position: str) -> TableInfo:
    """주석 행을 테이블에 추가"""
    if not text_lines:
        return table

    text_lines_sorted = sorted(text_lines, key=lambda l: l['x0'])

    table_width = table.bbox[2] - table.bbox[0]
    col_width = table_width / table.col_count if table.col_count > 0 else table_width

    new_row = [''] * table.col_count

    for line in text_lines_sorted:
        relative_x = line['x0'] - table.bbox[0]
        col_idx = min(int(relative_x / col_width), table.col_count - 1)
        col_idx = max(0, col_idx)

        if new_row[col_idx]:
            new_row[col_idx] += " " + line['text']
        else:
            new_row[col_idx] = line['text']

    non_empty_cols = sum(1 for c in new_row if c)
    if non_empty_cols == 1 and new_row[0]:
        combined_text = " ".join(line['text'] for line in text_lines_sorted)
        new_row = [combined_text] + [''] * (table.col_count - 1)

    new_data = list(table.data)

    # 셀 정보 업데이트
    new_cells_info = None
    if table.cells_info:
        new_cells_info = list(table.cells_info)
    else:
        new_cells_info = []

    if position == 'subheader':
        if len(new_data) > 0:
            new_data.insert(1, new_row)
            # 기존 셀 정보의 row 인덱스 조정 (row >= 1인 경우 +1)
            adjusted_cells = []
            for cell in new_cells_info:
                if cell['row'] >= 1:
                    adjusted_cell = dict(cell)
                    adjusted_cell['row'] = cell['row'] + 1
                    adjusted_cells.append(adjusted_cell)
                else:
                    adjusted_cells.append(cell)
            new_cells_info = adjusted_cells
            # 새 서브헤더 행에 대한 셀 정보 추가 (각 셀은 colspan=1)
            for col_idx in range(table.col_count):
                new_cells_info.append({
                    'row': 1,
                    'col': col_idx,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })
        else:
            new_data.append(new_row)
    else:
        new_data.append(new_row)
        # footer 행에 대한 셀 정보는 _generate_html_from_cells에서 처리됨

    all_y = [line['y0'] for line in text_lines] + [line['y1'] for line in text_lines]
    min_y = min(all_y)
    max_y = max(all_y)

    new_bbox = (
        table.bbox[0],
        min(table.bbox[1], min_y),
        table.bbox[2],
        max(table.bbox[3], max_y)
    )

    return TableInfo(
        page_num=table.page_num,
        table_idx=table.table_idx,
        bbox=new_bbox,
        data=new_data,
        col_count=table.col_count,
        row_count=len(new_data),
        page_height=table.page_height,
        cells_info=new_cells_info if new_cells_info else None,
        annotations=table.annotations,
        detection_strategy=table.detection_strategy,
        confidence=table.confidence
    )


# ============================================================================
# Phase 5: HTML Generation
# ============================================================================

def _is_single_column_table(table_info: TableInfo) -> bool:
    """
    테이블이 n rows × 1 column 형태인지 판단합니다.

    n rows × 1 column 테이블은 실제 테이블이 아닌 경우가 많으므로
    텍스트 리스트로 변환하는 것이 더 적합합니다.

    Args:
        table_info: 테이블 정보

    Returns:
        True if 1열 테이블, False otherwise
    """
    data = table_info.data

    if not data:
        return False

    # 모든 행의 최대 열 수 계산
    max_cols = max(len(row) for row in data) if data else 0

    # 1열이면 단일 열 테이블
    return max_cols == 1


def _convert_single_column_to_text(table_info: TableInfo) -> str:
    """
    1열 테이블을 텍스트 리스트로 변환합니다.

    n rows × 1 column 형태의 데이터는 테이블보다
    구조화된 텍스트로 표현하는 것이 더 의미론적으로 적합합니다.

    Args:
        table_info: 테이블 정보

    Returns:
        텍스트 리스트 형식의 문자열
    """
    data = table_info.data

    if not data:
        return ""

    lines = []
    for row in data:
        if row and len(row) > 0:
            cell_text = str(row[0]).strip() if row[0] else ""
            if cell_text:
                lines.append(cell_text)

    return '\n'.join(lines)


def _convert_table_to_html(table_info: TableInfo) -> str:
    """
    테이블을 HTML로 변환.

    개선사항:
    1. PyMuPDF 셀 정보 우선 사용
    2. CellAnalysisEngine 적용
    3. 정확한 rowspan/colspan 처리
    4. 주석 행 전체 colspan 처리
    5. 접근성 고려한 시맨틱 HTML
    """
    data = table_info.data

    if not data:
        return ""

    num_rows = len(data)
    num_cols = max(len(row) for row in data) if data else 0

    if num_cols == 0:
        return ""

    # CellAnalysisEngine을 사용하여 셀 분석 수행
    cell_engine = CellAnalysisEngine(table_info, None)
    analyzed_cells = cell_engine.analyze()

    # 분석된 셀 정보로 HTML 생성
    return _generate_html_from_cells_v2(data, analyzed_cells, num_rows, num_cols)


def _generate_html_from_cells(
    data: List[List[Optional[str]]],
    cells_info: List[Dict],
    num_rows: int,
    num_cols: int
) -> str:
    """분석된 셀 정보를 사용하여 HTML 생성 (호환성 유지)"""
    return _generate_html_from_cells_v2(data, cells_info, num_rows, num_cols)


def _generate_html_from_cells_v2(
    data: List[List[Optional[str]]],
    cells_info: List[Dict],
    num_rows: int,
    num_cols: int
) -> str:
    """
    개선된 HTML 생성

    개선사항:
    - 셀 정보가 불완전해도 모든 셀 처리
    - 빈 셀을 올바르게 렌더링
    - 데이터 범위 검증 강화
    """
    # span_map 생성: (row, col) -> {rowspan, colspan}
    span_map: Dict[Tuple[int, int], Dict] = {}

    for cell in cells_info:
        row = cell.get('row', 0)
        col = cell.get('col', 0)
        rowspan = max(1, cell.get('rowspan', 1))
        colspan = max(1, cell.get('colspan', 1))

        # 데이터 범위 내로 조정
        if row >= num_rows or col >= num_cols:
            continue

        rowspan = min(rowspan, num_rows - row)
        colspan = min(colspan, num_cols - col)

        key = (row, col)
        span_map[key] = {
            'rowspan': rowspan,
            'colspan': colspan
        }

    # skip_set 생성: 병합셀에 의해 커버되는 위치
    skip_set: Set[Tuple[int, int]] = set()

    for (row, col), spans in span_map.items():
        rowspan = spans['rowspan']
        colspan = spans['colspan']

        for r in range(row, min(row + rowspan, num_rows)):
            for c in range(col, min(col + colspan, num_cols)):
                if (r, c) != (row, col):
                    skip_set.add((r, c))

    # 주석 행 감지 및 전체 colspan 처리
    for row_idx, row in enumerate(data):
        if not row:
            continue
        first_val = str(row[0]).strip() if row[0] else ""

        is_annotation = False
        for pattern in PDFConfig.ANNOTATION_PATTERNS:
            if first_val.startswith(pattern):
                is_annotation = True
                break

        if is_annotation:
            # 주석 행은 전체 colspan
            span_map[(row_idx, 0)] = {'rowspan': 1, 'colspan': num_cols}
            for col_idx in range(1, num_cols):
                skip_set.add((row_idx, col_idx))

    # HTML 생성
    html_parts = ["<table>"]

    for row_idx in range(num_rows):
        html_parts.append("  <tr>")

        row_data = data[row_idx] if row_idx < len(data) else []

        for col_idx in range(num_cols):
            # 스킵해야 하는 셀인지 확인
            if (row_idx, col_idx) in skip_set:
                continue

            # 셀 내용 추출
            content = ""
            if col_idx < len(row_data):
                content = row_data[col_idx]
            content = escape_html(str(content).strip() if content else "")

            # span 정보 가져오기 (없으면 기본값 1)
            spans = span_map.get((row_idx, col_idx), {'rowspan': 1, 'colspan': 1})
            attrs = []

            if spans['rowspan'] > 1:
                attrs.append(f'rowspan="{spans["rowspan"]}"')
            if spans['colspan'] > 1:
                attrs.append(f'colspan="{spans["colspan"]}"')

            attr_str = " " + " ".join(attrs) if attrs else ""

            # 첫 행은 헤더로 처리
            tag = "th" if row_idx == 0 else "td"
            html_parts.append(f"    <{tag}{attr_str}>{content}</{tag}>")

        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


# ============================================================================
# 테이블 병합 및 연속성 처리
# ============================================================================

def _merge_adjacent_tables(tables: List[TableInfo]) -> List[TableInfo]:
    """인접 테이블 병합"""
    if not tables:
        return tables

    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)
    for table in tables:
        tables_by_page[table.page_num].append(table)

    merged_result = []

    for page_num, page_tables in tables_by_page.items():
        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])

        i = 0
        while i < len(sorted_tables):
            current = sorted_tables[i]

            merged = current
            while i + 1 < len(sorted_tables):
                next_table = sorted_tables[i + 1]

                if _should_merge_tables(merged, next_table):
                    merged = _do_merge_tables(merged, next_table)
                    i += 1
                    logger.debug(f"[PDF] Merged adjacent tables on page {page_num + 1}")
                else:
                    break

            merged_result.append(merged)
            i += 1

    merged_result.sort(key=lambda t: (t.page_num, t.bbox[1]))
    return merged_result


def _should_merge_tables(t1: TableInfo, t2: TableInfo) -> bool:
    """두 테이블 병합 여부 판단"""
    if t1.page_num != t2.page_num:
        return False

    y_gap = t2.bbox[1] - t1.bbox[3]
    if y_gap < 0 or y_gap > 30:
        return False

    x_overlap_start = max(t1.bbox[0], t2.bbox[0])
    x_overlap_end = min(t1.bbox[2], t2.bbox[2])
    x_overlap = max(0, x_overlap_end - x_overlap_start)

    t1_width = t1.bbox[2] - t1.bbox[0]
    t2_width = t2.bbox[2] - t2.bbox[0]

    overlap_ratio = x_overlap / max(t1_width, t2_width, 1)
    if overlap_ratio < 0.8:
        return False

    if t1.col_count == t2.col_count:
        return True
    if t1.row_count == 1 and t1.col_count < t2.col_count:
        return True

    return False


def _do_merge_tables(t1: TableInfo, t2: TableInfo) -> TableInfo:
    """
    두 테이블 병합 수행

    개선:
    - cells_info가 없어도 기본 셀 정보 유지
    - 병합 후 셀 인덱스 정확하게 조정
    """
    merged_bbox = (
        min(t1.bbox[0], t2.bbox[0]),
        t1.bbox[1],
        max(t1.bbox[2], t2.bbox[2]),
        t2.bbox[3]
    )

    merged_col_count = max(t1.col_count, t2.col_count)

    merged_data = []
    merged_cells = []

    # t1 데이터 처리
    t1_row_count = len(t1.data)

    if t1.col_count < merged_col_count and t1.row_count == 1 and t1.data:
        # 헤더 행이 적은 열을 가진 경우 colspan 처리
        extra_cols = merged_col_count - t1.col_count
        header_row = list(t1.data[0])

        new_header = []
        col_position = 0

        for orig_col_idx, value in enumerate(header_row):
            new_header.append(value)

            if orig_col_idx == 1 and extra_cols > 0:
                colspan = 1 + extra_cols
                merged_cells.append({
                    'row': 0,
                    'col': col_position,
                    'rowspan': 1,
                    'colspan': colspan,
                    'bbox': None
                })
                for _ in range(extra_cols):
                    new_header.append('')
                col_position += colspan
            else:
                merged_cells.append({
                    'row': 0,
                    'col': col_position,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })
                col_position += 1

        merged_data.append(new_header)
    else:
        # 일반 행 처리
        for row_idx, row in enumerate(t1.data):
            if len(row) < merged_col_count:
                adjusted_row = list(row) + [''] * (merged_col_count - len(row))
            else:
                adjusted_row = list(row)
            merged_data.append(adjusted_row)

        # t1의 셀 정보 복사
        if t1.cells_info:
            merged_cells.extend(t1.cells_info)

    # t2 데이터 처리
    row_offset = t1_row_count

    for row in t2.data:
        if len(row) < merged_col_count:
            adjusted_row = list(row) + [''] * (merged_col_count - len(row))
        else:
            adjusted_row = list(row)
        merged_data.append(adjusted_row)

    # t2의 셀 정보 복사 (row offset 적용)
    if t2.cells_info:
        for cell in t2.cells_info:
            adjusted_cell = dict(cell)
            adjusted_cell['row'] = cell.get('row', 0) + row_offset
            merged_cells.append(adjusted_cell)

    # 셀 정보가 비어있으면 None으로 (CellAnalysisEngine에서 처리)
    final_cells_info = merged_cells if merged_cells else None

    return TableInfo(
        page_num=t1.page_num,
        table_idx=t1.table_idx,
        bbox=merged_bbox,
        data=merged_data,
        col_count=merged_col_count,
        row_count=len(merged_data),
        page_height=t1.page_height,
        cells_info=final_cells_info,
        detection_strategy=t1.detection_strategy,
        confidence=max(t1.confidence, t2.confidence)
    )


def _process_table_continuity(all_tables: List[TableInfo]) -> List[TableInfo]:
    """페이지 간 테이블 연속성 처리"""
    if not all_tables:
        return all_tables

    result = []
    last_category = None

    for i, table_info in enumerate(all_tables):
        table_info = TableInfo(
            page_num=table_info.page_num,
            table_idx=table_info.table_idx,
            bbox=table_info.bbox,
            data=copy.deepcopy(table_info.data),
            col_count=table_info.col_count,
            row_count=table_info.row_count,
            page_height=table_info.page_height,
            cells_info=table_info.cells_info,
            annotations=table_info.annotations,
            detection_strategy=table_info.detection_strategy,
            confidence=table_info.confidence
        )

        curr_data = table_info.data

        if i == 0:
            last_category = _extract_last_category(curr_data)
            result.append(table_info)
            continue

        prev_table = all_tables[i - 1]

        is_continuation = (
            table_info.page_num > prev_table.page_num and
            prev_table.bbox[3] > prev_table.page_height * 0.7 and
            table_info.bbox[1] < table_info.page_height * 0.3 and
            table_info.col_count == prev_table.col_count
        )

        if is_continuation and last_category:
            for row in curr_data:
                if len(row) >= 2:
                    first_col = row[0]
                    second_col = row[1] if len(row) > 1 else ""

                    if (not first_col or not str(first_col).strip()) and second_col and str(second_col).strip():
                        row[0] = last_category
                    elif first_col and str(first_col).strip():
                        last_category = first_col
        else:
            new_last = _extract_last_category(curr_data)
            if new_last:
                last_category = new_last

        result.append(table_info)

    return result


def _extract_last_category(table_data: List[List[Optional[str]]]) -> Optional[str]:
    """테이블에서 마지막 카테고리 추출"""
    if not table_data:
        return None

    last_category = None

    for row in table_data:
        if len(row) >= 1 and row[0] and str(row[0]).strip():
            last_category = str(row[0]).strip()

    return last_category


# ============================================================================
# 페이지 테두리 감지
# ============================================================================

def _detect_page_border(page) -> PageBorderInfo:
    """
    페이지 테두리(장식용)를 감지합니다.

    개선점:
    1. 얇은 선도 감지
    2. 이중선 처리
    3. 더 정확한 테두리 판별
    """
    result = PageBorderInfo()

    drawings = page.get_drawings()
    if not drawings:
        return result

    page_width = page.rect.width
    page_height = page.rect.height

    edge_margin = min(page_width, page_height) * PDFConfig.PAGE_BORDER_MARGIN
    page_spanning_ratio = PDFConfig.PAGE_SPANNING_RATIO

    border_lines = {
        'top': False,
        'bottom': False,
        'left': False,
        'right': False
    }

    for drawing in drawings:
        rect = drawing.get('rect')
        if not rect:
            continue

        w = rect.width
        h = rect.height

        # 얇은 선도 감지 (두께 제한 완화)
        # 가로선 (높이가 작고 너비가 큼)
        if h <= 10 and w > page_width * page_spanning_ratio:
            if rect.y0 < edge_margin:
                border_lines['top'] = True
            elif rect.y1 > page_height - edge_margin:
                border_lines['bottom'] = True

        # 세로선 (너비가 작고 높이가 큼)
        if w <= 10 and h > page_height * page_spanning_ratio:
            if rect.x0 < edge_margin:
                border_lines['left'] = True
            elif rect.x1 > page_width - edge_margin:
                border_lines['right'] = True

    # 4면 모두 있으면 페이지 테두리
    if all(border_lines.values()):
        result.has_border = True
        result.border_bbox = (edge_margin, edge_margin, page_width - edge_margin, page_height - edge_margin)
        result.border_lines = border_lines

    return result


def _is_table_likely_border(
    table_bbox: Tuple[float, float, float, float],
    border_info: PageBorderInfo,
    page
) -> bool:
    """테이블이 페이지 테두리인지 확인"""
    if not border_info.has_border or not border_info.border_bbox:
        return False

    page_width = page.rect.width
    page_height = page.rect.height

    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]

    if table_width > page_width * 0.85 and table_height > page_height * 0.85:
        return True

    return False


# ============================================================================
# 텍스트 추출
# ============================================================================

def _extract_text_blocks(
    page,
    page_num: int,
    table_bboxes: List[Tuple[float, float, float, float]],
    border_info: PageBorderInfo,
    use_quality_check: bool = True
) -> List[PageElement]:
    """
    테이블 영역을 제외한 텍스트 블록 추출

    개선 사항:
    1. 텍스트 품질 분석 (깨진 텍스트 감지)
    2. 품질이 낮은 경우 OCR 폴백
    """
    elements = []

    # 텍스트 품질 분석
    if use_quality_check:
        analyzer = TextQualityAnalyzer(page, page_num)
        page_analysis = analyzer.analyze_page()

        # 품질이 너무 낮으면 전체 페이지 OCR 폴백
        if page_analysis.quality_result.needs_ocr:
            logger.info(
                f"[PDF] Page {page_num + 1}: Low text quality "
                f"({page_analysis.quality_result.quality_score:.2f}), "
                f"PUA={page_analysis.quality_result.pua_count}, "
                f"using OCR fallback"
            )

            extractor = QualityAwareTextExtractor(page, page_num)
            ocr_text, _ = extractor.extract()

            if ocr_text.strip():
                # OCR 텍스트를 블록별로 분리하여 반환
                # 테이블 영역 제외
                ocr_blocks = _split_ocr_text_to_blocks(ocr_text, page, table_bboxes)
                return ocr_blocks

    # 기존 로직: 일반 텍스트 추출
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_bbox = block.get("bbox", (0, 0, 0, 0))

        if is_inside_any_bbox(block_bbox, table_bboxes):
            continue

        text_parts = []
        block_quality_ok = True

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                text_parts.append(line_text.strip())

        if text_parts:
            full_text = "\n".join(text_parts)

            # 개별 블록 품질 체크 (use_quality_check가 True인 경우)
            if use_quality_check:
                analyzer = TextQualityAnalyzer(page, page_num)
                block_quality = analyzer.analyze_text(full_text)

                if block_quality.needs_ocr:
                    # 해당 블록만 OCR
                    from libs.core.processor.pdf_helpers.text_quality_analyzer import PageOCRFallbackEngine
                    ocr_engine = PageOCRFallbackEngine(page, page_num)
                    ocr_text = ocr_engine.ocr_region(block_bbox)
                    if ocr_text.strip():
                        full_text = ocr_text
                        logger.debug(f"[PDF] Block OCR: '{ocr_text[:50]}...'")

            elements.append(PageElement(
                element_type=ElementType.TEXT,
                content=full_text,
                bbox=block_bbox,
                page_num=page_num
            ))

    return elements


def _split_ocr_text_to_blocks(
    ocr_text: str,
    page,
    table_bboxes: List[Tuple[float, float, float, float]]
) -> List[PageElement]:
    """
    OCR 텍스트를 페이지 요소로 변환

    OCR은 위치 정보가 없으므로, 전체 텍스트를 하나의 블록으로 처리합니다.
    테이블 영역은 제외됩니다.
    """
    if not ocr_text.strip():
        return []

    # 테이블 영역을 제외한 페이지 영역 계산
    page_width = page.rect.width
    page_height = page.rect.height

    # OCR 텍스트를 단일 블록으로 반환 (위치는 페이지 전체)
    # 실제 위치 정보가 필요하면 pytesseract의 image_to_data 사용 가능
    return [PageElement(
        element_type=ElementType.TEXT,
        content=ocr_text,
        bbox=(0, 0, page_width, page_height),
        page_num=page.number
    )]


# ============================================================================
# 이미지 추출
# ============================================================================

def _extract_images_from_page(
    page,
    page_num: int,
    doc,
    processed_images: Set[int],
    table_bboxes: List[Tuple[float, float, float, float]],
    min_image_size: int = 50,
    min_image_area: int = 2500
) -> List[PageElement]:
    """페이지에서 이미지 추출 및 로컬 저장"""
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


# ============================================================================
# 요소 병합
# ============================================================================

def _merge_page_elements(elements: List[PageElement]) -> str:
    """페이지 요소들을 위치 기반으로 정렬하여 병합"""
    if not elements:
        return ""

    sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))

    text_parts = []

    for element in sorted_elements:
        content = element.content.strip()
        if not content:
            continue

        if element.element_type == ElementType.TABLE:
            text_parts.append(f"\n{content}\n")
        else:
            text_parts.append(content)

    return "\n".join(text_parts)
