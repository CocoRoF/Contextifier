# csv_helper/csv_parser.py
"""
CSV 파싱 및 분석

CSV 파일의 구분자 감지, 파싱, 헤더 감지 기능을 제공합니다.
"""
import csv
import io
import logging
import re
from typing import List

from .csv_constants import DELIMITER_CANDIDATES, MAX_ROWS, MAX_COLS

logger = logging.getLogger("document-processor")


def detect_delimiter(content: str) -> str:
    """
    CSV 구분자를 자동 감지합니다.

    감지 방법:
    1. csv.Sniffer 사용 시도
    2. 각 구분자의 일관성 분석

    Args:
        content: CSV 파일 내용

    Returns:
        감지된 구분자 문자
    """
    try:
        # 처음 몇 줄만 분석
        sample_lines = content.split('\n')[:20]
        sample = '\n'.join(sample_lines)

        # csv.Sniffer 사용
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t;|')
            return dialect.delimiter
        except csv.Error:
            pass

        # 수동 감지: 각 구분자의 일관성 확인
        best_delimiter = ','
        best_score = 0

        for delim in DELIMITER_CANDIDATES:
            counts = [line.count(delim) for line in sample_lines if line.strip()]

            if not counts:
                continue

            # 모든 행에서 구분자 수가 일정한지 확인
            if len(set(counts)) == 1 and counts[0] > 0:
                score = counts[0] * 10  # 일관성 보너스
            else:
                score = sum(counts) / len(counts) if counts else 0

            if score > best_score:
                best_score = score
                best_delimiter = delim

        return best_delimiter

    except Exception:
        return ','


def parse_csv_content(content: str, delimiter: str) -> List[List[str]]:
    """
    CSV 내용을 파싱합니다.

    Args:
        content: CSV 파일 내용
        delimiter: 구분자

    Returns:
        파싱된 행 데이터 (2차원 리스트)
    """
    rows = []

    try:
        # 줄바꿈 정규화
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        reader = csv.reader(
            io.StringIO(content),
            delimiter=delimiter,
            quotechar='"',
            doublequote=True,
            skipinitialspace=True
        )

        for i, row in enumerate(reader):
            if i >= MAX_ROWS:
                logger.warning(f"CSV row limit reached: {MAX_ROWS}")
                break

            # 열 수 제한
            if len(row) > MAX_COLS:
                row = row[:MAX_COLS]

            # 빈 행 건너뛰기
            if any(cell.strip() for cell in row):
                rows.append(row)

        return rows

    except csv.Error as e:
        logger.warning(f"CSV parsing error: {e}")
        # 폴백: 단순 분할
        return parse_csv_simple(content, delimiter)


def parse_csv_simple(content: str, delimiter: str) -> List[List[str]]:
    """
    단순 분할로 CSV를 파싱합니다 (폴백용).

    csv 모듈 파싱 실패 시 사용됩니다.

    Args:
        content: CSV 파일 내용
        delimiter: 구분자

    Returns:
        파싱된 행 데이터
    """
    rows = []

    for i, line in enumerate(content.split('\n')):
        if i >= MAX_ROWS:
            break

        line = line.strip()
        if not line:
            continue

        cells = line.split(delimiter)
        if len(cells) > MAX_COLS:
            cells = cells[:MAX_COLS]

        rows.append(cells)

    return rows


def detect_header(rows: List[List[str]]) -> bool:
    """
    첫 번째 행이 헤더인지 감지합니다.

    판단 기준:
    1. 첫 번째 행의 모든 셀이 문자열인지
    2. 두 번째 행에 숫자가 있는지
    3. 첫 번째 행의 셀들이 고유한지

    Args:
        rows: 파싱된 행 데이터

    Returns:
        첫 번째 행이 헤더이면 True
    """
    if len(rows) < 2:
        return False

    first_row = rows[0]
    second_row = rows[1]

    # 1. 첫 번째 행의 모든 셀이 문자열인지 확인
    first_all_text = all(
        not is_numeric(cell) for cell in first_row if cell.strip()
    )

    # 2. 두 번째 행에 숫자가 있는지 확인
    second_has_numbers = any(
        is_numeric(cell) for cell in second_row if cell.strip()
    )

    # 3. 첫 번째 행의 셀들이 고유한지 확인 (헤더는 보통 고유함)
    first_unique = len(set(first_row)) == len(first_row)

    # 헤더일 가능성 판단
    if first_all_text and (second_has_numbers or first_unique):
        return True

    return False


def is_numeric(value: str) -> bool:
    """
    값이 숫자인지 확인합니다.

    지원 형식:
    - 정수: 123, -456
    - 실수: 12.34, -56.78
    - 천단위 구분: 1,234,567
    - 퍼센트: 50%
    - 통화: $100, ₩10,000

    Args:
        value: 확인할 값

    Returns:
        숫자이면 True
    """
    if not value or not value.strip():
        return False

    value = value.strip()

    # 숫자 패턴
    patterns = [
        r'^-?\d+$',                       # 정수
        r'^-?\d+\.\d+$',                  # 실수
        r'^-?\d{1,3}(,\d{3})*(\.\d+)?$',  # 천단위 구분
        r'^-?\d+%$',                      # 퍼센트
        r'^\$-?\d+(\.\d+)?$',             # 달러
        r'^₩-?\d+(,\d{3})*$',             # 원화
    ]

    for pattern in patterns:
        if re.match(pattern, value):
            return True

    return False
