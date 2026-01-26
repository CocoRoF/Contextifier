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
- hwp_table_extractor.py: 테이블 추출 (BaseTableExtractor)
- hwp_table_processor.py: 테이블 포맷팅
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
    HWPMetadataExtractor,
    parse_hwp_summary_information,
)

# Image Processor (replaces hwp_image.py utility functions)
from contextifier.core.processor.hwp_helper.hwp_image_processor import HWPImageProcessor

# Chart Extractor
from contextifier.core.processor.hwp_helper.hwp_chart_extractor import HWPChartExtractor

# DocInfo
from contextifier.core.processor.hwp_helper.hwp_docinfo import (
    parse_doc_info,
    scan_bindata_folder,
)

# Table Extractor and Processor (for HWP 2.0/3.0 legacy format)
from contextifier.core.processor.hwp_helper.hwp_table_extractor import (
    HWPLegacyTableExtractor,
    HWPLegacyTableExtractorConfig,
    HWPLegacySection,
    HWPLegacyTableRegionInfo,
    # Aliases
    HWPTableExtractor,
    HWPTableExtractorConfig,
)
from contextifier.core.processor.hwp_helper.hwp_table_processor import (
    HWPLegacyTableProcessor,
    HWPLegacyTableProcessorConfig,
    # Aliases  
    HWPTableProcessor,
    HWPTableProcessorConfig,
)

# Recovery
from contextifier.core.processor.hwp_helper.hwp_recovery import (
    extract_text_from_stream_raw,
    find_zlib_streams,
    recover_images_from_raw,
    check_file_signature,
)

# Legacy Format Parser (HWP 2.0/3.0)
from contextifier.core.processor.hwp_helper.hwp_legacy_parser import (
    HWP2Parser,
    HWP2Config,
    HWP2ParseResult,
    parse_hwp_legacy,
    is_hwp_legacy_format,
    get_hwp_legacy_version,
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
    'HWPMetadataExtractor',
    'parse_hwp_summary_information',
    # Image Processor
    'HWPImageProcessor',
    # Chart Extractor
    'HWPChartExtractor',
    # DocInfo
    'parse_doc_info',
    'scan_bindata_folder',
    # Table Extractor and Processor
    'HWPLegacyTableExtractor',
    'HWPLegacyTableExtractorConfig',
    'HWPLegacySection',
    'HWPLegacyTableRegionInfo',
    'HWPLegacyTableProcessor',
    'HWPLegacyTableProcessorConfig',
    # Aliases
    'HWPTableExtractor',
    'HWPTableExtractorConfig',
    'HWPTableProcessor',
    'HWPTableProcessorConfig',
    # Recovery
    'extract_text_from_stream_raw',
    'find_zlib_streams',
    'recover_images_from_raw',
    'check_file_signature',
    # Legacy Format Parser
    'HWP2Parser',
    'HWP2Config',
    'HWP2ParseResult',
    'parse_hwp_legacy',
    'is_hwp_legacy_format',
    'get_hwp_legacy_version',
]
