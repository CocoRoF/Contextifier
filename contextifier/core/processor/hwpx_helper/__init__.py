# hwpx_helper/__init__.py
"""
HWPX Helper 모듈

hwpx_processor.py에서 사용하는 기능적 구성요소들을 모듈화하여 제공합니다.

모듈 구성:
- hwpx_constants: 상수 및 네임스페이스 정의
- hwpx_metadata: 메타데이터 추출 및 BinItem 매핑
- hwpx_table_extractor: 테이블 추출 (BaseTableExtractor 인터페이스)
- hwpx_table_processor: 테이블 변환 (HTML/Markdown/Text)
- hwpx_section: 섹션 XML 파싱
- hwpx_image: 이미지 처리 및 업로드
- hwpx_chart_extractor: 차트 추출 (ChartExtractor)
"""

# Constants
from contextifier.core.processor.hwpx_helper.hwpx_constants import (
    HWPX_NAMESPACES,
    OPF_NAMESPACES,
    SUPPORTED_IMAGE_EXTENSIONS,
    SKIP_IMAGE_EXTENSIONS,
    HEADER_FILE_PATHS,
    HPF_PATH,
)

# Metadata
from contextifier.core.processor.hwpx_helper.hwpx_metadata import (
    HWPXMetadataExtractor,
    parse_bin_item_map,
)

# Table Extractor (new structure)
from contextifier.core.processor.hwpx_helper.hwpx_table_extractor import (
    HWPXTableExtractor,
    HWPXTableExtractorConfig,
    HWPXTableRegionInfo,
    HWPXCellInfo,
    # Backward compatible functions
    parse_hwpx_table,
    extract_cell_content,
)

# Table Processor (new structure)
from contextifier.core.processor.hwpx_helper.hwpx_table_processor import (
    HWPXTableProcessor,
    HWPXTableProcessorConfig,
)

# Section
from contextifier.core.processor.hwpx_helper.hwpx_section import (
    parse_hwpx_section,
)

# Image Processor (replaces hwpx_image.py utility functions)
from contextifier.core.processor.hwpx_helper.hwpx_image_processor import (
    HWPXImageProcessor,
)

# Chart Extractor
from contextifier.core.processor.hwpx_helper.hwpx_chart_extractor import (
    HWPXChartExtractor,
)

__all__ = [
    # Constants
    "HWPX_NAMESPACES",
    "OPF_NAMESPACES",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SKIP_IMAGE_EXTENSIONS",
    "HEADER_FILE_PATHS",
    "HPF_PATH",
    # Metadata
    "HWPXMetadataExtractor",
    "parse_bin_item_map",
    # Table Extractor (new)
    "HWPXTableExtractor",
    "HWPXTableExtractorConfig",
    "HWPXTableRegionInfo",
    "HWPXCellInfo",
    # Table Processor (new)
    "HWPXTableProcessor",
    "HWPXTableProcessorConfig",
    # Backward compatible functions
    "parse_hwpx_table",
    "extract_cell_content",
    # Section
    "parse_hwpx_section",
    # Image Processor
    "HWPXImageProcessor",
    # Chart Extractor
    "HWPXChartExtractor",
]
