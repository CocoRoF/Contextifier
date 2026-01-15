# chunking_helper/table_parser.py
"""
Table Parser - HTML 테이블 파싱 관련 함수들

주요 기능:
- HTML 테이블 파싱 및 구조 분석
- 셀 span 정보 추출 (rowspan, colspan)
- 테이블 복잡도 분석
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from .constants import ParsedTable, TableRow

logger = logging.getLogger("document-processor")


def parse_html_table(table_html: str) -> Optional[ParsedTable]:
    """
    HTML 테이블을 파싱하여 구조화된 정보를 추출합니다.

    Args:
        table_html: HTML 테이블 문자열

    Returns:
        ParsedTable 객체 또는 None (파싱 실패 시)
    """
    try:
        # 행 추출
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        row_matches = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

        if not row_matches:
            logger.debug("No rows found in table")
            return None

        header_rows: List[TableRow] = []
        data_rows: List[TableRow] = []
        max_cols = 0

        for row_content in row_matches:
            # 셀 추출
            th_cells = re.findall(r'<th[^>]*>(.*?)</th>', row_content, re.DOTALL | re.IGNORECASE)
            td_cells = re.findall(r'<td[^>]*>(.*?)</td>', row_content, re.DOTALL | re.IGNORECASE)

            is_header = len(th_cells) > 0 and len(td_cells) == 0
            cell_count = len(th_cells) if is_header else len(td_cells)
            max_cols = max(max_cols, cell_count)

            # 원본 행 HTML 재구성
            row_html = f"<tr>{row_content}</tr>"
            row_length = len(row_html)

            table_row = TableRow(
                html=row_html,
                is_header=is_header,
                cell_count=cell_count,
                char_length=row_length
            )

            if is_header and not data_rows:
                # 데이터 행이 나오기 전 헤더 행
                header_rows.append(table_row)
            else:
                data_rows.append(table_row)

        # 헤더 HTML 구성
        if header_rows:
            header_html = "\n".join(row.html for row in header_rows)
            header_size = sum(row.char_length for row in header_rows) + len(header_rows)  # 줄바꿈 포함
        else:
            header_html = ""
            header_size = 0

        return ParsedTable(
            header_rows=header_rows,
            data_rows=data_rows,
            total_cols=max_cols,
            original_html=table_html,
            header_html=header_html,
            header_size=header_size
        )

    except Exception as e:
        logger.warning(f"Failed to parse HTML table: {e}")
        return None


def extract_cell_spans(row_html: str) -> List[Tuple[int, int]]:
    """
    행에서 셀의 rowspan/colspan 정보를 추출합니다.

    Args:
        row_html: 행 HTML

    Returns:
        [(rowspan, colspan), ...] 리스트
    """
    spans = []

    # th와 td 셀 찾기
    cell_pattern = r'<(th|td)([^>]*)>'

    for match in re.finditer(cell_pattern, row_html, re.IGNORECASE):
        attrs = match.group(2)

        # rowspan 추출
        rowspan_match = re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        rowspan = int(rowspan_match.group(1)) if rowspan_match else 1

        # colspan 추출
        colspan_match = re.search(r'colspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        colspan = int(colspan_match.group(1)) if colspan_match else 1

        spans.append((rowspan, colspan))

    return spans


def extract_cell_spans_with_positions(row_html: str) -> Dict[int, int]:
    """
    행에서 열 위치별 rowspan 정보를 추출합니다 (colspan 고려).

    Args:
        row_html: 행 HTML

    Returns:
        {열_위치: rowspan} 딕셔너리 (rowspan > 1인 셀만)
    """
    spans: Dict[int, int] = {}
    cell_pattern = r'<(th|td)([^>]*)>'

    current_col = 0
    for match in re.finditer(cell_pattern, row_html, re.IGNORECASE):
        attrs = match.group(2)

        # rowspan 추출
        rowspan_match = re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        rowspan = int(rowspan_match.group(1)) if rowspan_match else 1

        # colspan 추출
        colspan_match = re.search(r'colspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        colspan = int(colspan_match.group(1)) if colspan_match else 1

        if rowspan > 1:
            spans[current_col] = rowspan

        current_col += colspan

    return spans


def has_complex_spans(table_html: str) -> bool:
    """
    테이블에 복잡한 rowspan이 있는지 확인합니다.
    (colspan은 행 분할에 영향 없음, rowspan만 문제됨)

    Args:
        table_html: 테이블 HTML

    Returns:
        rowspan > 1인 셀이 있으면 True
    """
    rowspan_pattern = r'rowspan=["\']?(\d+)["\']?'
    matches = re.findall(rowspan_pattern, table_html, re.IGNORECASE)

    for val in matches:
        if int(val) > 1:
            return True

    return False
