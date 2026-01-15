# hwpx_helper/__init__.py
"""
HWPX Helper 모듈

hwpx_processor.py에서 사용하는 기능적 구성요소들을 모듈화하여 제공합니다.

모듈 구성:
- hwpx_constants: 상수 및 네임스페이스 정의
- hwpx_metadata: 메타데이터 추출 및 BinItem 매핑
- hwpx_table: 테이블 파싱 및 HTML 변환
- hwpx_section: 섹션 XML 파싱
- hwpx_image: 이미지 처리 및 업로드
- hwpx_chart: 차트 추출
"""

# Constants
from .hwpx_constants import (
    HWPX_NAMESPACES,
    OPF_NAMESPACES,
    SUPPORTED_IMAGE_EXTENSIONS,
    SKIP_IMAGE_EXTENSIONS,
    HEADER_FILE_PATHS,
    HPF_PATH,
)

# Metadata
from .hwpx_metadata import (
    extract_hwpx_metadata,
    parse_bin_item_map,
)

# Table
from .hwpx_table import (
    parse_hwpx_table,
    extract_cell_content,
)

# Section
from .hwpx_section import (
    parse_hwpx_section,
)

# Image
from .hwpx_image import (
    process_hwpx_images,
    get_remaining_images,
)

# Chart
from .hwpx_chart import (
    extract_charts_from_hwpx,
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
    "extract_hwpx_metadata",
    "parse_bin_item_map",
    # Table
    "parse_hwpx_table",
    "extract_cell_content",
    # Section
    "parse_hwpx_section",
    # Image
    "process_hwpx_images",
    "get_remaining_images",
    # Chart
    "extract_charts_from_hwpx",
]
