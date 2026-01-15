"""
PDF Helpers Package

PDF 처리에 필요한 헬퍼 모듈들을 포함합니다.
"""

from .pdf_helper import (
    upload_image_to_minio,
    extract_pdf_metadata,
    format_metadata,
    escape_html,
    calculate_overlap_ratio,
    is_inside_any_bbox,
    bboxes_overlap,
    find_image_position,
    get_text_lines_with_positions,
)

from .v3_types import (
    LineThickness,
    TableDetectionStrategy,
    ElementType,
    V3Config,
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

from .vector_text_ocr import (
    VectorTextConfig,
    VectorTextOCREngine,
)

from .graphic_detector import (
    GraphicRegionDetector,
)

from .table_validator import (
    TableQualityValidator,
)

from .line_analysis import (
    LineAnalysisEngine,
)

from .table_detection import (
    TableDetectionEngine,
)

from .cell_analysis import (
    CellAnalysisEngine,
)

# V4 신규 모듈
from .complexity_analyzer import (
    ComplexityLevel,
    ProcessingStrategy,
    RegionComplexity,
    PageComplexity,
    ComplexityConfig,
    ComplexityAnalyzer,
)

from .block_image_engine import (
    BlockStrategy,
    BlockImageConfig,
    BlockImageResult,
    MultiBlockResult,
    BlockImageEngine,
)

from .layout_block_detector import (
    LayoutBlockType,
    ContentElement,
    LayoutBlock,
    ColumnInfo,
    LayoutAnalysisResult,
    LayoutDetectorConfig,
    LayoutBlockDetector,
)

# V4 테이블 품질 분석 모듈
from .table_quality_analyzer import (
    TableQuality,
    TableQualityResult,
    TableQualityAnalyzer,
)

__all__ = [
    # pdf_helper
    'upload_image_to_minio',
    'extract_pdf_metadata',
    'format_metadata',
    'escape_html',
    'calculate_overlap_ratio',
    'is_inside_any_bbox',
    'bboxes_overlap',
    'find_image_position',
    'get_text_lines_with_positions',
    # v3_types
    'LineThickness',
    'TableDetectionStrategy',
    'ElementType',
    'V3Config',
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
    # V4: complexity_analyzer
    'ComplexityLevel',
    'ProcessingStrategy',
    'RegionComplexity',
    'PageComplexity',
    'ComplexityConfig',
    'ComplexityAnalyzer',
    # V4: block_image_engine
    'BlockStrategy',
    'BlockImageConfig',
    'BlockImageResult',
    'MultiBlockResult',
    'BlockImageEngine',
    # V4: layout_block_detector
    'LayoutBlockType',
    'ContentElement',
    'LayoutBlock',
    'ColumnInfo',
    'LayoutAnalysisResult',
    'LayoutDetectorConfig',
    'LayoutBlockDetector',
    # V4: table_quality_analyzer
    'TableQuality',
    'TableQualityResult',
    'TableQualityAnalyzer',
]
