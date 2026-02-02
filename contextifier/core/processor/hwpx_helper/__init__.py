# hwpx_helper/__init__.py
"""
HWPX Helper 모듈

hwpx_processor.py에서 사용하는 기능적 구성요소들을 모듈화하여 제공합니다.

모듈 구성:
- hwpx_constants: 상수 및 네임스페이스 정의
- hwpx_metadata: 메타데이터 추출 및 BinItem 매핑
- hwpx_table_extractor: 테이블 추출 (HWPXTableExtractor) - BaseTableExtractor 구현
- hwpx_table_processor: 테이블 포맷팅 (HWPXTableProcessor) - TableProcessor 확장
- hwpx_section: 섹션 XML 파싱
- hwpx_image_processor: 이미지 처리 및 업로드
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

# Table Extractor (NEW - BaseTableExtractor implementation)
from contextifier.core.processor.hwpx_helper.hwpx_table_extractor import (
    HWPXTableExtractor,
    create_hwpx_table_extractor,
)

# Table Processor (NEW - TableProcessor extension)
from contextifier.core.processor.hwpx_helper.hwpx_table_processor import (
    HWPXTableProcessor,
    HWPXTableProcessorConfig,
    create_hwpx_table_processor,
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
    # Table Extractor (NEW)
    "HWPXTableExtractor",
    "create_hwpx_table_extractor",
    # Table Processor (NEW)
    "HWPXTableProcessor",
    "HWPXTableProcessorConfig",
    "create_hwpx_table_processor",
    # Section
    "parse_hwpx_section",
    # Image Processor
    "HWPXImageProcessor",
    # Chart Extractor
    "HWPXChartExtractor",
]
