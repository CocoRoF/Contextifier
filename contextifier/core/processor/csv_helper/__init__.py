# csv_helper/__init__.py
"""
CSV Helper 모듈

csv_handler.py에서 사용하는 기능적 구성요소들을 모듈화하여 제공합니다.

모듈 구성:
- csv_constants: 상수 및 데이터 클래스 정의
- csv_metadata: 메타데이터 추출 및 포맷팅
- csv_encoding: 인코딩 감지 및 파일 읽기
- csv_parser: CSV 파싱, 구분자/헤더 감지
- csv_table_extractor: CSV 테이블 추출 (BaseTableExtractor 인터페이스)
- csv_table_processor: CSV 테이블 변환 (HTML/Markdown/Text)
"""

# Constants
from contextifier.core.processor.csv_helper.csv_constants import (
    ENCODING_CANDIDATES,
    DELIMITER_CANDIDATES,
    DELIMITER_NAMES,
    MAX_ROWS,
    MAX_COLS,
    CSVMetadata,
)

# Metadata
from contextifier.core.processor.csv_helper.csv_metadata import (
    CSVMetadataExtractor,
    CSVSourceInfo,
)

# Image Processor
from contextifier.core.processor.csv_helper.csv_image_processor import (
    CSVImageProcessor,
)

# Encoding
from contextifier.core.processor.csv_helper.csv_encoding import (
    detect_bom,
    read_file_with_encoding,
)

# Parser
from contextifier.core.processor.csv_helper.csv_parser import (
    detect_delimiter,
    parse_csv_content,
    parse_csv_simple,
    detect_header,
    is_numeric,
)

# Table Extractor (new structure)
from contextifier.core.processor.csv_helper.csv_table_extractor import (
    CSVTableExtractor,
    CSVTableExtractorConfig,
    CSVTableRegionInfo,
    CSVCellMergeInfo,
    # Backward compatible functions
    has_merged_cells,
    analyze_merge_info,
)

# Table Processor (new structure)
from contextifier.core.processor.csv_helper.csv_table_processor import (
    CSVTableProcessor,
    CSVTableProcessorConfig,
    # Backward compatible functions
    convert_rows_to_table,
    convert_rows_to_markdown,
    convert_rows_to_html,
)

__all__ = [
    # Constants
    "ENCODING_CANDIDATES",
    "DELIMITER_CANDIDATES",
    "DELIMITER_NAMES",
    "MAX_ROWS",
    "MAX_COLS",
    "CSVMetadata",
    # Metadata
    "CSVMetadataExtractor",
    "CSVSourceInfo",
    # Image Processor
    "CSVImageProcessor",
    # Encoding
    "detect_bom",
    "read_file_with_encoding",
    # Parser
    "detect_delimiter",
    "parse_csv_content",
    "parse_csv_simple",
    "detect_header",
    "is_numeric",
    # Table Extractor (new)
    "CSVTableExtractor",
    "CSVTableExtractorConfig",
    "CSVTableRegionInfo",
    "CSVCellMergeInfo",
    # Table Processor (new)
    "CSVTableProcessor",
    "CSVTableProcessorConfig",
    # Backward compatible functions
    "has_merged_cells",
    "analyze_merge_info",
    "convert_rows_to_table",
    "convert_rows_to_markdown",
    "convert_rows_to_html",
]
