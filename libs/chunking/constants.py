# chunking_helper/constants.py
"""
Chunking Module Constants - 청킹 관련 상수, 패턴, 데이터클래스 정의

이 모듈은 청킹 시스템 전반에서 사용되는 모든 상수와 데이터 구조를 정의합니다.
"""
import logging
from dataclasses import dataclass
from typing import List
from langchain_text_splitters import Language

logger = logging.getLogger("document-processor")


# ============================================================================
# 코드 언어 맵핑
# ============================================================================

LANGCHAIN_CODE_LANGUAGE_MAP = {
    'py': Language.PYTHON, 'js': Language.JS, 'ts': Language.TS,
    'java': Language.JAVA, 'cpp': Language.CPP, 'c': Language.CPP,
    'cs': Language.CSHARP, 'go': Language.GO, 'rs': Language.RUST,
    'php': Language.PHP, 'rb': Language.RUBY, 'swift': Language.SWIFT,
    'kt': Language.KOTLIN, 'scala': Language.SCALA,
    'html': Language.HTML, 'jsx': Language.JS, 'tsx': Language.TS,
}


# ============================================================================
# 보호 영역 패턴 (청킹 시 분할되지 않아야 하는 블록)
# ============================================================================

# HTML 테이블 - 모든 <table> 태그 보호 (속성에 관계없이)
HTML_TABLE_PATTERN = r'<table[^>]*>.*?</table>'

# 차트 블록 - 항상 보호됨 (어떤 조건에서도 chunking 불가)
CHART_BLOCK_PATTERN = r'\[chart\].*?\[/chart\]'

# 텍스트박스 블록 - 항상 보호됨 (어떤 조건에서도 chunking 불가)
TEXTBOX_BLOCK_PATTERN = r'\[textbox\].*?\[/textbox\]'

# 이미지 태그 - 항상 보호됨 (어떤 조건에서도 chunking 불가)
# 형식: [image:path], [Image: {path}], [image : path] 등 (대소문자 무관, 띄어쓰기 허용, {} 감싸기 허용)
IMAGE_TAG_PATTERN = r'\[(?i:image)\s*:\s*\{?[^\]\}]+\}?\]'

# Markdown 테이블 (| 로 시작하는 연속된 행들, 헤더 구분선 |---|---| 포함)
MARKDOWN_TABLE_PATTERN = r'(?:^|\n)(\|[^\n]+\|\n\|[-:|\s]+\|\n(?:\|[^\n]+\|(?:\n|$))+)'

# Markdown 테이블 개별 행 패턴 (row 단위 보호용)
MARKDOWN_TABLE_ROW_PATTERN = r'\|[^\n]+\|'


# ============================================================================
# 테이블 청킹 관련 상수
# ============================================================================

# 테이블 래핑 오버헤드 (테이블 태그, 줄바꿈 등)
TABLE_WRAPPER_OVERHEAD = 30  # <table border='1'>\n</table>

# 행당 최소 오버헤드 (<tr>\n</tr>)
ROW_OVERHEAD = 12

# 셀당 오버헤드 (<td></td> 또는 <th></th>)
CELL_OVERHEAD = 10

# 청크 인덱스 메타데이터 오버헤드
CHUNK_INDEX_OVERHEAD = 30  # [테이블 청크 1/10]\n

# 테이블이 이 크기보다 크면 분할 대상
TABLE_SIZE_THRESHOLD_MULTIPLIER = 1.2  # chunk_size의 1.2배

# 테이블 기반 파일 타입 (CSV, Excel)
TABLE_BASED_FILE_TYPES = {'csv', 'tsv', 'xlsx', 'xls'}


# ============================================================================
# 데이터클래스
# ============================================================================

@dataclass
class TableRow:
    """테이블 행 데이터"""
    html: str
    is_header: bool
    cell_count: int
    char_length: int


@dataclass
class ParsedTable:
    """파싱된 테이블 정보"""
    header_rows: List[TableRow]  # 헤더 행들
    data_rows: List[TableRow]    # 데이터 행들
    total_cols: int              # 총 열 수
    original_html: str           # 원본 HTML
    header_html: str             # 헤더 HTML (재사용용)
    header_size: int             # 헤더 크기 (문자 수)
