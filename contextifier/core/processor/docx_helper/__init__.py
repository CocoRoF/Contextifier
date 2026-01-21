# service/document_processor/processor/docx_helper/__init__.py
"""
DOCX Helper 모듈

DOCX 문서 처리에 필요한 유틸리티를 기능별로 분리한 모듈입니다.

모듈 구성:
- docx_constants: 상수, Enum, 데이터클래스 (ElementType, NAMESPACES 등)
- docx_metadata: 메타데이터 추출 및 포맷팅
- docx_chart_extractor: 차트 추출 (ChartExtractor)
- docx_image: 이미지 추출 및 업로드
- docx_table: 테이블 HTML 변환 (rowspan/colspan 지원)
- docx_drawing: Drawing 요소 처리 (이미지/다이어그램)
- docx_paragraph: Paragraph 처리 및 페이지 브레이크
"""

# Constants
from contextifier.core.processor.docx_helper.docx_constants import (
    ElementType,
    DocxElement,
    NAMESPACES,
    CHART_TYPE_MAP,
)

# Metadata
from contextifier.core.processor.docx_helper.docx_metadata import (
    DOCXMetadataExtractor,
)

# Chart Extractor
from contextifier.core.processor.docx_helper.docx_chart_extractor import (
    DOCXChartExtractor,
)

# Image
from contextifier.core.processor.docx_helper.docx_image import (
    extract_image_from_drawing,
    process_pict_element,
)

# Table
from contextifier.core.processor.docx_helper.docx_table import (
    TableCellInfo,
    process_table_element,
    calculate_all_rowspans,
    estimate_column_count,
    extract_cell_text,
    extract_table_as_text,
)

# Drawing
from contextifier.core.processor.docx_helper.docx_drawing import (
    process_drawing_element,
)

# Paragraph
from contextifier.core.processor.docx_helper.docx_paragraph import (
    process_paragraph_element,
    has_page_break_element,
)


__all__ = [
    # Constants
    'ElementType',
    'DocxElement',
    'NAMESPACES',
    'CHART_TYPE_MAP',
    # Metadata
    'extract_docx_metadata',
    'format_metadata',
    # Chart Extractor
    'DOCXChartExtractor',
    # Image
    'extract_image_from_drawing',
    'process_pict_element',
    # Table
    'TableCellInfo',
    'process_table_element',
    'calculate_all_rowspans',
    'estimate_column_count',
    'extract_cell_text',
    'extract_table_as_text',
    # Drawing
    'process_drawing_element',
    # Paragraph
    'process_paragraph_element',
    'has_page_break_element',
]
