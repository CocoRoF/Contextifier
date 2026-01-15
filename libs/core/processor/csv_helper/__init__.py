# csv_helper/__init__.py
"""
CSV Helper 모듈

csv_handler.py에서 사용하는 기능적 구성요소들을 모듈화하여 제공합니다.

모듈 구성:
- csv_constants: 상수 및 데이터 클래스 정의
- csv_metadata: 메타데이터 추출 및 포맷팅
- csv_encoding: 인코딩 감지 및 파일 읽기
- csv_parser: CSV 파싱, 구분자/헤더 감지
- csv_table: 테이블 변환 (Markdown/HTML)
"""

# Constants
from .csv_constants import (
    ENCODING_CANDIDATES,
    DELIMITER_CANDIDATES,
    DELIMITER_NAMES,
    MAX_ROWS,
    MAX_COLS,
    CSVMetadata,
)

# Metadata
from .csv_metadata import (
    format_file_size,
    get_delimiter_name,
    extract_csv_metadata,
    format_metadata,
)

# Encoding
from .csv_encoding import (
    detect_bom,
    read_file_with_encoding,
)

# Parser
from .csv_parser import (
    detect_delimiter,
    parse_csv_content,
    parse_csv_simple,
    detect_header,
    is_numeric,
)

# Table
from .csv_table import (
    has_merged_cells,
    analyze_merge_info,
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
    "format_file_size",
    "get_delimiter_name",
    "extract_csv_metadata",
    "format_metadata",
    # Encoding
    "detect_bom",
    "read_file_with_encoding",
    # Parser
    "detect_delimiter",
    "parse_csv_content",
    "parse_csv_simple",
    "detect_header",
    "is_numeric",
    # Table
    "has_merged_cells",
    "analyze_merge_info",
    "convert_rows_to_table",
    "convert_rows_to_markdown",
    "convert_rows_to_html",
]
