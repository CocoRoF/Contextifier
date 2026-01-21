# csv_helper/csv_constants.py
"""
CSV Handler 상수 및 타입 정의

CSV/TSV 파일 처리에 필요한 상수, 설정값, 데이터 클래스를 정의합니다.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# === 인코딩 관련 상수 ===

# 시도할 인코딩 목록 (우선순위 순)
ENCODING_CANDIDATES = [
    "utf-8",
    "utf-8-sig",  # BOM 포함 UTF-8
    "cp949",      # 한국어 Windows
    "euc-kr",     # 한국어 레거시
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "latin-1",    # 폴백 (모든 바이트 허용)
    "iso-8859-1",
]


# === 구분자 관련 상수 ===

# CSV delimiter candidates
DELIMITER_CANDIDATES = [',', '\t', ';', '|']

# Delimiter name mapping (Korean for output display)
DELIMITER_NAMES = {
    ',': '쉼표 (,)',
    '\t': '탭 (\\t)',
    ';': '세미콜론 (;)',
    '|': '파이프 (|)',
}


# === Processing limit constants ===

# Maximum rows to process (memory protection)
MAX_ROWS = 100000

# Maximum columns
MAX_COLS = 1000


# === 데이터 클래스 ===

@dataclass
class CSVMetadata:
    """CSV 파일 메타데이터"""
    encoding: str
    delimiter: str
    has_header: bool
    row_count: int
    col_count: int
    file_size: int
    file_name: Optional[str] = None
    modified_time: Optional[datetime] = None
