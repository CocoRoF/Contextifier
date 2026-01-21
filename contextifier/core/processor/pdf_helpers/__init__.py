"""
PDF Helpers Package

Contains helper modules for PDF processing.
"""

# Metadata - class-based extractor
from contextifier.core.processor.pdf_helpers.pdf_metadata import (
    PDFMetadataExtractor,
    parse_pdf_date,
)

from contextifier.core.processor.pdf_helpers.pdf_utils import (
    escape_html,
    calculate_overlap_ratio,
    is_inside_any_bbox,
    find_image_position,
    get_text_lines_with_positions,
    bbox_overlaps,
)

from contextifier.core.processor.pdf_helpers.pdf_image import (
    extract_images_from_page,
)

from contextifier.core.processor.pdf_helpers.pdf_text_extractor import (
    extract_text_blocks,
    split_ocr_text_to_blocks,
)

from contextifier.core.processor.pdf_helpers.pdf_page_analyzer import (
    detect_page_border,
    is_table_likely_border,
)

from contextifier.core.processor.pdf_helpers.pdf_element_merger import (
    merge_page_elements,
)

from contextifier.core.processor.pdf_helpers.pdf_table_processor import (
    TableInfo,
    AnnotationInfo as TableAnnotationInfo,
    extract_all_tables,
    find_and_insert_annotations,
    add_annotation_to_table,
    merge_adjacent_tables,
    should_merge_tables,
    do_merge_tables,
    process_table_continuity,
    extract_last_category,
    is_single_column_table,
    convert_single_column_to_text,
    convert_table_to_html,
    generate_html_from_cells,
)

from contextifier.core.processor.pdf_helpers.types import (
    LineThickness,
    TableDetectionStrategy,
    ElementType,
    PDFConfig,
    LineInfo,
    GridInfo,
    CellInfo,
    AnnotationInfo,
    VectorTextRegion,
    GraphicRegionInfo,
    TableCandidate,
    PageElement,
    PageBorderInfo,
)

from contextifier.core.processor.pdf_helpers.pdf_vector_text_ocr import (
    VectorTextConfig,
    VectorTextOCREngine,
)

from contextifier.core.processor.pdf_helpers.pdf_graphic_detector import (
    GraphicRegionDetector,
)

from contextifier.core.processor.pdf_helpers.pdf_table_validator import (
    TableQualityValidator,
)

from contextifier.core.processor.pdf_helpers.pdf_line_analysis import (
    LineAnalysisEngine,
)

from contextifier.core.processor.pdf_helpers.pdf_table_detection import (
    TableDetectionEngine,
)

from contextifier.core.processor.pdf_helpers.pdf_cell_analysis import (
    CellAnalysisEngine,
)

from contextifier.core.processor.pdf_helpers.pdf_complexity_analyzer import (
    ComplexityLevel,
    ProcessingStrategy,
    RegionComplexity,
    PageComplexity,
    ComplexityConfig,
    ComplexityAnalyzer,
)

from contextifier.core.processor.pdf_helpers.pdf_block_image_engine import (
    BlockStrategy,
    BlockImageConfig,
    BlockImageResult,
    MultiBlockResult,
    BlockImageEngine,
)

from contextifier.core.processor.pdf_helpers.pdf_layout_block_detector import (
    LayoutBlockType,
    ContentElement,
    LayoutBlock,
    ColumnInfo,
    LayoutAnalysisResult,
    LayoutDetectorConfig,
    LayoutBlockDetector,
)

from contextifier.core.processor.pdf_helpers.pdf_table_quality_analyzer import (
    TableQuality,
    TableQualityResult,
    TableQualityAnalyzer,
)

__all__ = [
    # pdf_helper
    'extract_pdf_metadata',
    'format_metadata',
    'escape_html',
    'calculate_overlap_ratio',
    'is_inside_any_bbox',
    'find_image_position',
    'get_text_lines_with_positions',
    # types
    'LineThickness',
    'TableDetectionStrategy',
    'ElementType',
    'PDFConfig',
    'LineInfo',
    'GridInfo',
    'CellInfo',
    'AnnotationInfo',
    'VectorTextRegion',
    'GraphicRegionInfo',
    'TableCandidate',
    'PageElement',
    'PageBorderInfo',
    # vector_text_ocr
    'VectorTextConfig',
    'VectorTextOCREngine',
    # graphic_detector
    'GraphicRegionDetector',
    # table_validator
    'TableQualityValidator',
    # line_analysis
    'LineAnalysisEngine',
    # table_detection
    'TableDetectionEngine',
    # cell_analysis
    'CellAnalysisEngine',
    # complexity_analyzer
    'ComplexityLevel',
    'ProcessingStrategy',
    'RegionComplexity',
    'PageComplexity',
    'ComplexityConfig',
    'ComplexityAnalyzer',
    # block_image_engine
    'BlockStrategy',
    'BlockImageConfig',
    'BlockImageResult',
    'MultiBlockResult',
    'BlockImageEngine',
    # layout_block_detector
    'LayoutBlockType',
    'ContentElement',
    'LayoutBlock',
    'ColumnInfo',
    'LayoutAnalysisResult',
    'LayoutDetectorConfig',
    'LayoutBlockDetector',
    # table_quality_analyzer
    'TableQuality',
    'TableQualityResult',
    'TableQualityAnalyzer',
    # pdf_metadata
    'extract_pdf_metadata',
    'format_metadata',
    'parse_pdf_date',
    # pdf_utils
    'escape_html',
    'calculate_overlap_ratio',
    'is_inside_any_bbox',
    'find_image_position',
    'get_text_lines_with_positions',
    'bbox_overlaps',
    # pdf_image
    'extract_images_from_page',
    # pdf_text_extractor
    'extract_text_blocks',
    'split_ocr_text_to_blocks',
    # pdf_page_analyzer
    'detect_page_border',
    'is_table_likely_border',
    # pdf_element_merger
    'merge_page_elements',
    # pdf_table_processor
    'TableInfo',
    'TableAnnotationInfo',
    'extract_all_tables',
    'find_and_insert_annotations',
    'add_annotation_to_table',
    'merge_adjacent_tables',
    'should_merge_tables',
    'do_merge_tables',
    'process_table_continuity',
    'extract_last_category',
    'is_single_column_table',
    'convert_single_column_to_text',
    'convert_table_to_html',
    'generate_html_from_cells',
]
