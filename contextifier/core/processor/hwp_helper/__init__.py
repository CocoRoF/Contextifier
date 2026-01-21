# service/document_processor/processor/hwp_helper/__init__.py
"""
HWP/HWPX 공통 헬퍼 모듈

HWP 5.0 OLE 파일 처리에 필요한 유틸리티 모듈을 제공합니다.

파일 구조:
- hwp_constants.py: 상수 정의 (태그 ID, 차트 타입 등)
- hwp_record.py: HWP 레코드 파싱 클래스
- hwp_decoder.py: 압축/인코딩 유틸리티
- hwp_metadata.py: 메타데이터 추출
- hwp_image.py: 이미지 처리
- hwp_chart.py: 차트 처리
- hwp_docinfo.py: DocInfo 파싱
- hwp_table.py: 테이블 파싱
- hwp_recovery.py: 손상 파일 복구
"""

# Constants
from contextifier.core.processor.hwp_helper.hwp_constants import (
    HWPTAG_BEGIN,
    HWPTAG_BIN_DATA,
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_CTRL_HEADER,
    HWPTAG_LIST_HEADER,
    HWPTAG_SHAPE_COMPONENT,
    HWPTAG_SHAPE_COMPONENT_PICTURE,
    HWPTAG_TABLE,
    HWPTAG_SHAPE_COMPONENT_OLE,
    HWPTAG_CHART_DATA,
    CHART_TYPES,
    CTRL_CHAR_DRAWING_TABLE_OBJECT,
)

# Record Parser
from contextifier.core.processor.hwp_helper.hwp_record import HwpRecord

# Decoder
from contextifier.core.processor.hwp_helper.hwp_decoder import (
    is_compressed,
    decompress_stream,
    decompress_section,
)

# Metadata
from contextifier.core.processor.hwp_helper.hwp_metadata import (
    extract_metadata,
    parse_hwp_summary_information,
    format_metadata,
    MetadataHelper,
)

# Image
from contextifier.core.processor.hwp_helper.hwp_image import (
    try_decompress_image,
    save_image_to_local,
    find_bindata_stream,
    extract_bindata_index,
    extract_and_upload_image,
    process_images_from_bindata,
    ImageHelper,
)

# Chart Extractor
from contextifier.core.processor.hwp_helper.hwp_chart_extractor import HWPChartExtractor

# DocInfo
from contextifier.core.processor.hwp_helper.hwp_docinfo import (
    parse_doc_info,
    scan_bindata_folder,
)

# Table
from contextifier.core.processor.hwp_helper.hwp_table import (
    parse_table,
    build_table_grid,
    render_table_html,
)

# Recovery
from contextifier.core.processor.hwp_helper.hwp_recovery import (
    extract_text_from_stream_raw,
    find_zlib_streams,
    recover_images_from_raw,
    check_file_signature,
)


__all__ = [
    # Constants
    'HWPTAG_BEGIN',
    'HWPTAG_BIN_DATA',
    'HWPTAG_PARA_HEADER',
    'HWPTAG_PARA_TEXT',
    'HWPTAG_CTRL_HEADER',
    'HWPTAG_LIST_HEADER',
    'HWPTAG_SHAPE_COMPONENT',
    'HWPTAG_SHAPE_COMPONENT_PICTURE',
    'HWPTAG_TABLE',
    'HWPTAG_SHAPE_COMPONENT_OLE',
    'HWPTAG_CHART_DATA',
    'CHART_TYPES',
    'CTRL_CHAR_DRAWING_TABLE_OBJECT',
    # Record
    'HwpRecord',
    # Decoder
    'is_compressed',
    'decompress_stream',
    'decompress_section',
    # Metadata
    'extract_metadata',
    'parse_hwp_summary_information',
    'format_metadata',
    'MetadataHelper',
    # Image
    'try_decompress_image',
    'save_image_to_local',
    'find_bindata_stream',
    'extract_bindata_index',
    'extract_and_upload_image',
    'process_images_from_bindata',
    'ImageHelper',
    # Chart Extractor
    'HWPChartExtractor',
    # DocInfo
    'parse_doc_info',
    'scan_bindata_folder',
    # Table
    'parse_table',
    'build_table_grid',
    'render_table_html',
    # Recovery
    'extract_text_from_stream_raw',
    'find_zlib_streams',
    'recover_images_from_raw',
    'check_file_signature',
]
