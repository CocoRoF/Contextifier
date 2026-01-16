"""
PDF Helpers Package

Contains helper modules for PDF processing.
"""

from libs.core.processor.pdf_helpers.pdf_helper import (
    extract_pdf_metadata,
    format_metadata,
    escape_html,
    calculate_overlap_ratio,
    is_inside_any_bbox,
    find_image_position,
    get_text_lines_with_positions,
)

from libs.core.processor.pdf_helpers.types import (
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

from libs.core.processor.pdf_helpers.vector_text_ocr import (
    VectorTextConfig,
    VectorTextOCREngine,
)

from libs.core.processor.pdf_helpers.graphic_detector import (
    GraphicRegionDetector,
)

from libs.core.processor.pdf_helpers.table_validator import (
    TableQualityValidator,
)

from libs.core.processor.pdf_helpers.line_analysis import (
    LineAnalysisEngine,
)

from libs.core.processor.pdf_helpers.table_detection import (
    TableDetectionEngine,
)

from libs.core.processor.pdf_helpers.cell_analysis import (
    CellAnalysisEngine,
)

from libs.core.processor.pdf_helpers.complexity_analyzer import (
    ComplexityLevel,
    ProcessingStrategy,
    RegionComplexity,
    PageComplexity,
    ComplexityConfig,
    ComplexityAnalyzer,
)

from libs.core.processor.pdf_helpers.block_image_engine import (
    BlockStrategy,
    BlockImageConfig,
    BlockImageResult,
    MultiBlockResult,
    BlockImageEngine,
)

from libs.core.processor.pdf_helpers.layout_block_detector import (
    LayoutBlockType,
    ContentElement,
    LayoutBlock,
    ColumnInfo,
    LayoutAnalysisResult,
    LayoutDetectorConfig,
    LayoutBlockDetector,
)

from libs.core.processor.pdf_helpers.table_quality_analyzer import (
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
]
