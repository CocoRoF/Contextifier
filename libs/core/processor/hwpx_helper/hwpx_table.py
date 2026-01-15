# hwpx_helper/hwpx_table.py
"""
HWPX 테이블 처리

HWPX 문서의 테이블을 HTML로 변환합니다.
병합 셀(rowspan/colspan)과 중첩 테이블을 지원합니다.
"""
import logging
import xml.etree.ElementTree as ET
from typing import Dict

logger = logging.getLogger("document-processor")


def parse_hwpx_table(table_node: ET.Element, ns: Dict[str, str]) -> str:
    """
    HWPX 테이블을 HTML로 변환합니다.

    병합 셀(rowspan/colspan)을 지원합니다.
    중첩 테이블도 재귀적으로 처리합니다.

    HWPX 테이블 XML 구조:
    <hp:tbl rowCnt="8" colCnt="6">
      <hp:tr>
        <hp:tc>
          <hp:subList>
            <hp:p><hp:run><hp:t>텍스트</hp:t></hp:run></hp:p>
          </hp:subList>
          <hp:cellAddr colAddr="0" rowAddr="0"/>
          <hp:cellSpan colSpan="1" rowSpan="5"/>  <!-- 병합 정보! -->
          <hp:cellSz width="6992" height="1410"/>
        </hp:tc>
      </hp:tr>
    </hp:tbl>

    Args:
        table_node: hp:tbl 요소
        ns: 네임스페이스 딕셔너리

    Returns:
        HTML 테이블 문자열
    """
    try:
        # 테이블 메타 정보 (참고용)
        total_rows = int(table_node.get('rowCnt', 0))
        total_cols = int(table_node.get('colCnt', 0))

        # 테이블 셀을 그리드 기반으로 수집
        # grid[row_addr][col_addr] = cell_info
        grid = {}
        max_row = 0
        max_col = 0

        for tr in table_node.findall('hp:tr', ns):
            for tc in tr.findall('hp:tc', ns):
                # 병합 정보 추출 - <hp:cellSpan> 요소에서!
                col_span = 1
                row_span = 1

                cell_span = tc.find('hp:cellSpan', ns)
                if cell_span is not None:
                    try:
                        col_span = int(cell_span.get('colSpan', 1))
                    except (ValueError, TypeError):
                        col_span = 1
                    try:
                        row_span = int(cell_span.get('rowSpan', 1))
                    except (ValueError, TypeError):
                        row_span = 1

                # 셀 위치 추출 - <hp:cellAddr> 요소에서
                col_addr = 0
                row_addr = 0

                cell_addr = tc.find('hp:cellAddr', ns)
                if cell_addr is not None:
                    try:
                        col_addr = int(cell_addr.get('colAddr', 0))
                    except (ValueError, TypeError):
                        col_addr = 0
                    try:
                        row_addr = int(cell_addr.get('rowAddr', 0))
                    except (ValueError, TypeError):
                        row_addr = 0

                # 셀 내용 추출 (중첩 테이블 포함)
                cell_text = extract_cell_content(tc, ns)

                # 그리드에 저장
                grid[(row_addr, col_addr)] = {
                    'text': cell_text,
                    'colspan': col_span,
                    'rowspan': row_span
                }

                max_row = max(max_row, row_addr)
                max_col = max(max_col, col_addr)

        # total_rows/total_cols가 있으면 사용, 없으면 계산된 값 사용
        if total_rows == 0:
            total_rows = max_row + 1
        if total_cols == 0:
            total_cols = max_col + 1

        if not grid:
            return ""

        # 1×1 테이블 -> 셀 내용만 반환 (컨테이너 테이블)
        if total_rows == 1 and total_cols == 1:
            if (0, 0) in grid:
                return grid[(0, 0)]['text']
            return ""

        # 단일 컬럼 테이블 (1열, 다중 행) -> 셀 내용을 줄바꿈으로 구분
        if total_cols == 1:
            text_items = []
            for r in range(total_rows):
                if (r, 0) in grid:
                    cell_text = grid[(r, 0)]['text']
                    if cell_text:
                        text_items.append(cell_text)
            if text_items:
                return "\n\n".join(text_items)
            return ""

        # HTML 테이블 생성 (2+ 컬럼)
        html_parts = ["<table border='1'>"]
        skip_map = set()  # 병합으로 인해 건너뛸 셀

        for r in range(total_rows):
            html_parts.append("<tr>")
            for c in range(total_cols):
                # 병합된 셀은 건너뜀
                if (r, c) in skip_map:
                    continue

                if (r, c) in grid:
                    cell = grid[(r, c)]
                    text = cell['text']
                    rowspan = cell['rowspan']
                    colspan = cell['colspan']

                    attr = ""
                    if rowspan > 1:
                        attr += f" rowspan='{rowspan}'"
                    if colspan > 1:
                        attr += f" colspan='{colspan}'"

                    html_parts.append(f"<td{attr}>{text}</td>")

                    # 병합된 영역을 skip_map에 추가
                    for rs in range(rowspan):
                        for cs in range(colspan):
                            if rs == 0 and cs == 0:
                                continue
                            skip_map.add((r + rs, c + cs))
                else:
                    # 그리드에 없는 셀 (병합으로 이미 처리된 경우 아님)
                    html_parts.append("<td></td>")

            html_parts.append("</tr>")

        html_parts.append("</table>")
        return "\n".join(html_parts)

    except Exception as e:
        logger.warning(f"Failed to parse HWPX table: {e}")
        return "[Table Extraction Failed]"


def extract_cell_content(tc: ET.Element, ns: Dict[str, str]) -> str:
    """
    셀 내용을 추출합니다. 중첩 테이블도 재귀적으로 처리합니다.

    Args:
        tc: hp:tc 요소
        ns: 네임스페이스 딕셔너리

    Returns:
        셀 내용 문자열 (중첩 테이블은 HTML로 변환)
    """
    content_parts = []

    sublist = tc.find('hp:subList', ns)
    if sublist is not None:
        for p in sublist.findall('hp:p', ns):
            para_parts = []
            for run in p.findall('hp:run', ns):
                # 텍스트 추출
                t = run.find('hp:t', ns)
                if t is not None and t.text:
                    para_parts.append(t.text)

                # 중첩 테이블 처리 (재귀 호출)
                nested_table = run.find('hp:tbl', ns)
                if nested_table is not None:
                    nested_html = parse_hwpx_table(nested_table, ns)
                    if nested_html:
                        para_parts.append(nested_html)

            if para_parts:
                content_parts.append("".join(para_parts))

    return " ".join(content_parts).strip()
