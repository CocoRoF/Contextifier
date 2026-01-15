# csv_helper/csv_table.py
"""
CSV 테이블 변환

CSV 데이터를 Markdown 또는 HTML 테이블로 변환합니다.
병합셀 분석 및 처리를 포함합니다.
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger("document-processor")


def has_merged_cells(rows: List[List[str]]) -> bool:
    """
    CSV 데이터에 병합셀(빈 셀)이 존재하는지 확인합니다.

    병합셀의 판단 기준:
    - 연속된 빈 셀이 존재하는 경우
    - 첫 번째 열에 빈 셀이 있고 이전 행에 값이 있는 경우 (세로 병합 패턴)

    Args:
        rows: 파싱된 행 데이터

    Returns:
        병합셀이 존재하면 True, 아니면 False
    """
    if not rows or len(rows) < 2:
        return False

    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            cell_value = cell.strip() if cell else ""

            # 빈 셀 발견
            if not cell_value:
                # 첫 번째 행이 아니고 첫 번째 열의 빈 셀 -> 세로 병합 가능성
                if row_idx > 0 and col_idx == 0:
                    return True

                # 이전 셀이 비어있지 않고 현재 셀이 비어있음 -> 가로 병합 가능성
                if col_idx > 0:
                    prev_cell = row[col_idx - 1].strip() if col_idx - 1 < len(row) else ""
                    if prev_cell:
                        return True

    return False


def analyze_merge_info(rows: List[List[str]]) -> List[List[Dict[str, Any]]]:
    """
    CSV 데이터의 병합셀 정보를 분석합니다.

    빈 셀을 기반으로 colspan(가로 병합)과 rowspan(세로 병합)을 계산합니다.

    Args:
        rows: 파싱된 행 데이터

    Returns:
        각 셀의 병합 정보를 담은 2차원 리스트
        각 셀 정보: {
            'value': str,      # 셀 값
            'colspan': int,    # 가로 병합 수 (1 이상)
            'rowspan': int,    # 세로 병합 수 (1 이상)
            'skip': bool       # 이 셀이 다른 셀에 병합되어 렌더링 생략해야 하는지
        }
    """
    if not rows:
        return []

    row_count = len(rows)
    col_count = max(len(row) for row in rows) if rows else 0

    # 초기화: 모든 셀 정보 생성
    merge_info: List[List[Dict[str, Any]]] = []
    for row_idx, row in enumerate(rows):
        row_info = []
        for col_idx in range(col_count):
            cell_value = row[col_idx].strip() if col_idx < len(row) else ""
            row_info.append({
                'value': cell_value,
                'colspan': 1,
                'rowspan': 1,
                'skip': False
            })
        merge_info.append(row_info)

    # 1단계: 가로 병합 (colspan) 계산 - 오른쪽으로 연속된 빈 셀
    for row_idx in range(row_count):
        col_idx = 0
        while col_idx < col_count:
            cell_info = merge_info[row_idx][col_idx]

            # 이미 스킵된 셀이거나 빈 셀이면 패스
            if cell_info['skip'] or not cell_info['value']:
                col_idx += 1
                continue

            # 오른쪽으로 연속된 빈 셀 카운트
            colspan = 1
            next_col = col_idx + 1
            while next_col < col_count:
                next_cell = merge_info[row_idx][next_col]
                if not next_cell['value'] and not next_cell['skip']:
                    colspan += 1
                    next_cell['skip'] = True  # 병합되어 렌더링 생략
                    next_col += 1
                else:
                    break

            cell_info['colspan'] = colspan
            col_idx = next_col

    # 2단계: 세로 병합 (rowspan) 계산 - 아래로 연속된 빈 셀
    for col_idx in range(col_count):
        row_idx = 0
        while row_idx < row_count:
            cell_info = merge_info[row_idx][col_idx]

            # 이미 스킵된 셀이거나 빈 셀이면 패스
            if cell_info['skip'] or not cell_info['value']:
                row_idx += 1
                continue

            # 아래로 연속된 빈 셀 카운트
            rowspan = 1
            next_row = row_idx + 1
            while next_row < row_count:
                next_cell = merge_info[next_row][col_idx]
                # 빈 셀이고 아직 스킵되지 않은 경우에만 세로 병합
                if not next_cell['value'] and not next_cell['skip']:
                    rowspan += 1
                    next_cell['skip'] = True  # 병합되어 렌더링 생략
                    next_row += 1
                else:
                    break

            cell_info['rowspan'] = rowspan
            row_idx = next_row

    return merge_info


def convert_rows_to_table(rows: List[List[str]], has_header: bool) -> str:
    """
    CSV 행을 테이블로 변환합니다.
    병합셀이 없으면 Markdown, 있으면 HTML로 변환합니다.

    Args:
        rows: 파싱된 행 데이터
        has_header: 헤더 존재 여부

    Returns:
        변환된 테이블 문자열
    """
    if not rows:
        return ""

    # 병합셀 유무 확인
    has_merged = has_merged_cells(rows)

    if has_merged:
        logger.debug("Merged cells detected, using HTML format")
        return convert_rows_to_html(rows, has_header)
    else:
        logger.debug("No merged cells, using Markdown format")
        return convert_rows_to_markdown(rows, has_header)


def convert_rows_to_markdown(rows: List[List[str]], _has_header: bool) -> str:
    """
    CSV 행을 Markdown 테이블로 변환합니다.

    Note:
        Markdown 테이블은 첫 행이 항상 헤더로 취급되므로
        _has_header 인자는 HTML 변환과의 인터페이스 일관성을 위해 유지됩니다.

    Args:
        rows: 파싱된 행 데이터
        _has_header: 헤더 존재 여부 (Markdown에서는 미사용)

    Returns:
        Markdown 테이블 문자열
    """
    if not rows:
        return ""

    md_parts = []

    for row_idx, row in enumerate(rows):
        # 셀 값 정리 (파이프 문자 이스케이프)
        cells = []
        for cell in row:
            cell_value = cell.strip() if cell else ""
            # Markdown 테이블에서 파이프는 이스케이프 필요
            cell_value = cell_value.replace("|", "\\|")
            # 줄바꿈을 공백으로 변환 (Markdown 테이블은 줄바꿈 미지원)
            cell_value = cell_value.replace("\n", " ")
            cells.append(cell_value)

        # 행 생성
        row_str = "| " + " | ".join(cells) + " |"
        md_parts.append(row_str)

        # 헤더 구분선 추가 (첫 번째 행 다음)
        if row_idx == 0:
            separator = "| " + " | ".join(["---"] * len(cells)) + " |"
            md_parts.append(separator)

    return "\n".join(md_parts)


def convert_rows_to_html(rows: List[List[str]], has_header: bool) -> str:
    """
    CSV 행을 HTML 테이블로 변환합니다.
    병합셀(빈 셀)을 분석하여 colspan과 rowspan을 적용합니다.

    Args:
        rows: 파싱된 행 데이터
        has_header: 헤더 존재 여부

    Returns:
        HTML 테이블 문자열
    """
    if not rows:
        return ""

    # 병합 정보 분석
    merge_info = analyze_merge_info(rows)

    html_parts = ["<table border='1'>"]

    for row_idx, row_info in enumerate(merge_info):
        html_parts.append("<tr>")

        for cell_info in row_info:
            # 다른 셀에 병합되어 스킵해야 하는 경우
            if cell_info['skip']:
                continue

            cell_value = cell_info['value']

            # HTML 이스케이프
            cell_value = cell_value.replace("&", "&amp;")
            cell_value = cell_value.replace("<", "&lt;")
            cell_value = cell_value.replace(">", "&gt;")
            cell_value = cell_value.replace("\n", "<br>")

            # 헤더 또는 데이터 셀
            tag = "th" if (has_header and row_idx == 0) else "td"

            # 병합 속성 생성
            attrs = []
            if cell_info['colspan'] > 1:
                attrs.append(f"colspan='{cell_info['colspan']}'")
            if cell_info['rowspan'] > 1:
                attrs.append(f"rowspan='{cell_info['rowspan']}'")

            attr_str = " " + " ".join(attrs) if attrs else ""
            html_parts.append(f"<{tag}{attr_str}>{cell_value}</{tag}>")

        html_parts.append("</tr>")

    html_parts.append("</table>")

    return "\n".join(html_parts)
