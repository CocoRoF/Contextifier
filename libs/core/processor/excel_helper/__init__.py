"""
Excel Helper 모듈

XLSX/XLS 파일의 세부 요소(텍스트박스, 차트, 이미지, 테이블 등) 추출을 담당합니다.

모듈 구성:
- excel_chart_constants: 차트 타입 맵핑 상수
- excel_chart_parser: OOXML 차트 XML 파싱
- excel_chart_formatter: 차트 데이터 테이블 포맷팅
- excel_chart_renderer: matplotlib 이미지 렌더링
- excel_chart_processor: 차트 처리 메인 (테이블/이미지 폴백)
- excel_table_xlsx: XLSX 테이블 변환
- excel_table_xls: XLS 테이블 변환
- textbox_extractor: 텍스트박스 추출
- metadata_extractor: 메타데이터 추출
- image_extractor: 이미지 추출
"""

# === Textbox ===
from .excel_textbox import extract_textboxes_from_xlsx

# === Metadata ===
from .excel_metadata import (
    extract_xlsx_metadata,
    extract_xls_metadata,
    format_metadata,
)

# === Chart Constants ===
from .excel_chart_constants import (
    CHART_TYPE_MAP,
    CHART_NAMESPACES,
)

# === Chart Parser ===
from .excel_chart_parser import (
    extract_charts_from_xlsx,
    parse_ooxml_chart_xml,
    extract_chart_info_basic,
)

# === Chart Formatter ===
from .excel_chart_formatter import (
    format_chart_data_as_table,
    format_chart_fallback,
)

# === Chart Renderer ===
from .excel_chart_renderer import (
    render_chart_to_image,
)

# === Chart Processor ===
from .excel_chart_processor import (
    process_chart,
)

# === Image ===
from .excel_image import (
    extract_images_from_xlsx,
    get_sheet_images,
    SUPPORTED_IMAGE_EXTENSIONS,
)

# === Table XLSX ===
from .excel_table_xlsx import (
    has_merged_cells_xlsx,
    convert_xlsx_sheet_to_table,
    convert_xlsx_sheet_to_markdown,
    convert_xlsx_sheet_to_html,
    convert_xlsx_objects_to_tables,
)

# === Table XLS ===
from .excel_table_xls import (
    has_merged_cells_xls,
    convert_xls_sheet_to_table,
    convert_xls_sheet_to_markdown,
    convert_xls_sheet_to_html,
    convert_xls_objects_to_tables,
)

# === Layout Detector ===
from .excel_layout_detector import (
    layout_detect_range_xlsx,
    layout_detect_range_xls,
    object_detect_xlsx,
    object_detect_xls,
    LayoutRange,
)


__all__ = [
    # Textbox
    'extract_textboxes_from_xlsx',
    # Metadata
    'extract_xlsx_metadata',
    'extract_xls_metadata',
    'format_metadata',
    # Chart Constants
    'CHART_TYPE_MAP',
    'CHART_NAMESPACES',
    # Chart Parser
    'extract_charts_from_xlsx',
    'parse_ooxml_chart_xml',
    'extract_chart_info_basic',
    # Chart Formatter
    'format_chart_data_as_table',
    'format_chart_fallback',
    # Chart Renderer
    'render_chart_to_image',
    # Chart Processor
    'process_chart',
    # Image
    'extract_images_from_xlsx',
    'get_sheet_images',
    'SUPPORTED_IMAGE_EXTENSIONS',
    # Table XLSX
    'has_merged_cells_xlsx',
    'convert_xlsx_sheet_to_table',
    'convert_xlsx_sheet_to_markdown',
    'convert_xlsx_sheet_to_html',
    'convert_xlsx_objects_to_tables',
    # Table XLS
    'has_merged_cells_xls',
    'convert_xls_sheet_to_table',
    'convert_xls_sheet_to_markdown',
    'convert_xls_sheet_to_html',
    'convert_xls_objects_to_tables',
    # Layout Detector
    'layout_detect_range_xlsx',
    'layout_detect_range_xls',
    'object_detect_xlsx',
    'object_detect_xls',
    'LayoutRange',
]
