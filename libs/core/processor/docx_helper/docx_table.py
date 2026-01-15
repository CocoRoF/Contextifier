# service/document_processor/processor/docx_helper/docx_table.py
"""
DOCX 테이블 처리 유틸리티

DOCX 문서의 테이블을 HTML로 변환합니다.
- TableCellInfo: 테이블 셀 정보 데이터 클래스
- process_table_element: Table 요소를 HTML로 변환
- calculate_all_rowspans: vMerge restart 셀의 rowspan 계산
- extract_cell_text: 셀 내용 추출
- extract_table_as_text: 테이블을 평문으로 추출 (폴백용)

OOXML 테이블 구조:
- w:tblGrid: 테이블의 그리드 열 정의
- w:tr: 테이블 행
- w:tc: 테이블 셀
- w:tcPr/w:gridSpan: colspan (가로 병합)
- w:tcPr/w:vMerge val="restart": rowspan 시작
- w:tcPr/w:vMerge (val 없음): rowspan 계속 (병합된 셀)
"""
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from docx import Document
from docx.oxml.ns import qn

from .docx_constants import NAMESPACES

logger = logging.getLogger("document-processor")


@dataclass
class TableCellInfo:
    """테이블 셀 정보를 저장하는 데이터 클래스"""
    grid_row: int       # 그리드 상의 행 위치 (0-based)
    grid_col: int       # 그리드 상의 열 위치 (0-based)
    rowspan: int        # 실제 rowspan 값
    colspan: int        # 실제 colspan 값 (gridSpan)
    content: str        # 셀 내용
    is_merged_away: bool  # True면 다른 셀에 병합되어 렌더링하지 않음


def process_table_element(table_elem, doc: Document) -> str:
    """
    Table 요소를 HTML로 변환합니다.
    셀 병합(rowspan/colspan)을 정확하게 지원합니다.

    핵심 알고리즘 (v3):
    1. calculate_all_rowspans를 통해 모든 셀의 rowspan과 grid_col 위치 계산
    2. continue 셀은 렌더링하지 않음
    3. restart/none 셀만 렌더링하며, 정확한 rowspan 값 사용
    
    특수 처리:
    - 1×1 테이블: 셀 내용만 텍스트로 반환 (컨테이너 테이블)
    - 단일 컬럼 테이블 (1열): 셀 내용을 줄바꿈으로 구분하여 반환

    Args:
        table_elem: table XML 요소
        doc: python-docx Document 객체

    Returns:
        HTML 테이블 문자열, 또는 단순 텍스트
    """
    try:
        rows = table_elem.findall('w:tr', NAMESPACES)
        if not rows:
            return ""

        num_rows = len(rows)

        # === 1단계: vMerge restart 셀들의 rowspan 및 셀 위치 정보 계산 ===
        rowspan_map, cell_grid_col = calculate_all_rowspans(table_elem, rows, num_rows)

        # 테이블의 총 컬럼 수 계산 (tblGrid 또는 첫 번째 행의 셀 수)
        tblGrid = table_elem.find('w:tblGrid', NAMESPACES)
        if tblGrid is not None:
            num_cols = len(tblGrid.findall('w:gridCol', NAMESPACES))
        else:
            # 첫 번째 행의 셀들로 컬럼 수 계산
            num_cols = 0
            if rows:
                for cell in rows[0].findall('w:tc', NAMESPACES):
                    tcPr = cell.find('w:tcPr', NAMESPACES)
                    colspan = 1
                    if tcPr is not None:
                        gs = tcPr.find('w:gridSpan', NAMESPACES)
                        if gs is not None:
                            try:
                                colspan = int(gs.get(qn('w:val'), 1))
                            except (ValueError, TypeError):
                                colspan = 1
                    num_cols += colspan

        # === 2단계: 렌더링할 셀 정보 수집 ===
        render_cells: List[Tuple[int, int, int, int, str]] = []  # (row, col, rowspan, colspan, content)

        for row_idx, row in enumerate(rows):
            cells = row.findall('w:tc', NAMESPACES)

            for cell_idx, cell in enumerate(cells):
                # 셀 속성 가져오기
                tcPr = cell.find('w:tcPr', NAMESPACES)
                colspan = 1
                is_vmerge_continue = False

                if tcPr is not None:
                    # gridSpan (colspan) 파악
                    gs = tcPr.find('w:gridSpan', NAMESPACES)
                    if gs is not None:
                        try:
                            colspan = int(gs.get(qn('w:val'), 1))
                        except (ValueError, TypeError):
                            colspan = 1

                    # vMerge 상태 확인
                    vMerge = tcPr.find('w:vMerge', NAMESPACES)
                    if vMerge is not None:
                        val = vMerge.get(qn('w:val'))
                        if val != 'restart':
                            # vMerge continue (val이 없거나 'continue')
                            is_vmerge_continue = True

                if is_vmerge_continue:
                    # 이 셀은 이전 행의 vMerge에 의해 병합됨 - 렌더링하지 않음
                    continue

                # 일반 셀 또는 vMerge restart 셀
                # cell_grid_col에서 이 셀의 grid_col 위치 가져오기
                if cell_idx < len(cell_grid_col[row_idx]):
                    start_col, end_col = cell_grid_col[row_idx][cell_idx]
                else:
                    # 안전 폴백
                    start_col = cell_idx

                # rowspan 가져오기 (미리 계산된 값)
                rowspan = rowspan_map.get((row_idx, start_col), 1)

                # 셀 내용 추출
                content = extract_cell_text(cell)

                # 렌더링할 셀 정보 저장
                render_cells.append((row_idx, start_col, rowspan, colspan, content))

        # === 2.5단계: 1×1 또는 단일 컬럼 테이블 특수 처리 ===
        
        # 1×1 테이블 -> 셀 내용만 반환 (컨테이너 테이블)
        if num_rows == 1 and num_cols == 1:
            if render_cells:
                return render_cells[0][4]  # content
            return ""
        
        # 단일 컬럼 테이블 (1열, 다중 행) -> 셀 내용을 줄바꿈으로 구분
        if num_cols == 1:
            text_items = []
            for (_, _, _, _, content) in render_cells:
                if content:
                    text_items.append(content)
            if text_items:
                return "\n\n".join(text_items)
            return ""

        # === 3단계: HTML 생성 ===
        html_parts = ["<table border='1'>"]

        # 셀 정보를 행별로 그룹화
        cells_by_row: Dict[int, List[Tuple[int, int, int, int, str]]] = {}
        for cell_info in render_cells:
            row_idx = cell_info[0]
            if row_idx not in cells_by_row:
                cells_by_row[row_idx] = []
            cells_by_row[row_idx].append(cell_info)

        # 각 행 내에서 열 순서로 정렬
        for row_idx in cells_by_row:
            cells_by_row[row_idx].sort(key=lambda x: x[1])  # grid_col 기준 정렬

        for row_idx in range(num_rows):
            html_parts.append("<tr>")

            row_cells = cells_by_row.get(row_idx, [])
            for (_, grid_col, rowspan, colspan, content) in row_cells:
                # 셀 내용 HTML 이스케이프
                cell_text = content
                cell_text = cell_text.replace("&", "&amp;")
                cell_text = cell_text.replace("<", "&lt;")
                cell_text = cell_text.replace(">", "&gt;")
                cell_text = cell_text.replace("\n", "<br>")

                # 첫 번째 행은 헤더로 처리
                tag = "th" if row_idx == 0 else "td"

                # 속성 생성
                attrs = []
                if rowspan > 1:
                    attrs.append(f"rowspan='{rowspan}'")
                if colspan > 1:
                    attrs.append(f"colspan='{colspan}'")

                attr_str = " " + " ".join(attrs) if attrs else ""
                html_parts.append(f"<{tag}{attr_str}>{cell_text}</{tag}>")

            html_parts.append("</tr>")

        html_parts.append("</table>")
        return "\n".join(html_parts)

    except Exception as e:
        logger.warning(f"Error processing table element: {e}")
        logger.debug(traceback.format_exc())
        # 폴백: 단순 텍스트 형식
        return extract_table_as_text(table_elem)


def calculate_all_rowspans(table_elem, rows, num_rows: int) -> Tuple[Dict[Tuple[int, int], int], List[List[Tuple[int, int]]]]:
    """
    테이블의 모든 vMerge restart 셀에 대해 rowspan을 계산합니다.

    개선된 알고리즘 (v3):
    1. 모든 셀 정보를 수집
    2. merge_info 매트릭스를 사용하여 각 grid_col이 어떤 셀에 속하는지 추적
    3. continue 셀은 위쪽 행의 같은 grid_col 위치에 있는 셀과 연결
    4. restart 셀에 대해 아래 행들에서 같은 owner를 가진 셀을 찾아 rowspan 계산

    핵심 개선:
    - continue 셀은 단순히 colspan만큼 grid_col을 소비하지 않고,
      위쪽 행의 같은 grid_col 위치에 있는 restart/continue와 연결됨
    - 이를 통해 복잡한 중첩 병합 테이블도 정확하게 처리

    Args:
        table_elem: table XML 요소
        rows: 테이블 행 리스트
        num_rows: 행 개수

    Returns:
        Tuple[Dict[(row_idx, grid_col), rowspan], List[List[(start_col, end_col)]]]
        - rowspan_map: 각 셀의 rowspan 값
        - cell_grid_col: 각 셀의 grid_col 위치 정보
    """
    rowspan_map: Dict[Tuple[int, int], int] = {}

    # 모든 셀 정보 수집: all_cells_info[row][cell] = (colspan, vmerge_status)
    all_cells_info: List[List[Tuple[int, str]]] = []

    for row in rows:
        cells = row.findall('w:tc', NAMESPACES)
        row_info = []
        for cell in cells:
            tcPr = cell.find('w:tcPr', NAMESPACES)
            colspan = 1
            vmerge_status = 'none'

            if tcPr is not None:
                gs = tcPr.find('w:gridSpan', NAMESPACES)
                if gs is not None:
                    try:
                        colspan = int(gs.get(qn('w:val'), 1))
                    except (ValueError, TypeError):
                        colspan = 1

                vMerge = tcPr.find('w:vMerge', NAMESPACES)
                if vMerge is not None:
                    val = vMerge.get(qn('w:val'))
                    vmerge_status = 'restart' if val == 'restart' else 'continue'

            row_info.append((colspan, vmerge_status))
        all_cells_info.append(row_info)

    # 1단계: 모든 셀의 정확한 grid_col 계산
    max_cols = 30
    cell_grid_col: List[List[Tuple[int, int]]] = []

    # merge_info[row][col] = (owner_row, owner_col, colspan)
    # 해당 grid_col이 어떤 셀에 속하는지 추적
    merge_info: List[List[Optional[Tuple[int, int, int]]]] = [
        [None] * max_cols for _ in range(num_rows)
    ]

    for row_idx, row_info in enumerate(all_cells_info):
        grid_col = 0
        row_grid_cols: List[Tuple[int, int]] = []

        for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
            # 이미 점유된 열 건너뛰기 (이전 행의 vMerge에 의해)
            while grid_col < max_cols and merge_info[row_idx][grid_col] is not None:
                grid_col += 1

            # 열 확장 필요시
            while grid_col + colspan > max_cols:
                for r in range(num_rows):
                    merge_info[r].extend([None] * 10)
                max_cols += 10

            start_col = grid_col
            end_col = grid_col + colspan - 1
            row_grid_cols.append((start_col, end_col))

            if vmerge_status == 'restart':
                # restart 셀: 현재 행만 표시 (rowspan은 나중에 계산)
                for c in range(start_col, start_col + colspan):
                    merge_info[row_idx][c] = (row_idx, start_col, colspan)

            elif vmerge_status == 'continue':
                # continue 셀: 위쪽 행의 같은 위치 셀과 연결
                for prev_row in range(row_idx - 1, -1, -1):
                    if merge_info[prev_row][start_col] is not None:
                        owner = merge_info[prev_row][start_col]
                        for c in range(start_col, start_col + colspan):
                            merge_info[row_idx][c] = owner
                        break
                else:
                    # 위에서 못 찾음 - 이상한 경우지만 현재 셀로 설정
                    for c in range(start_col, start_col + colspan):
                        merge_info[row_idx][c] = (row_idx, start_col, colspan)
            else:
                # 일반 셀
                for c in range(start_col, start_col + colspan):
                    merge_info[row_idx][c] = (row_idx, start_col, colspan)

            grid_col += colspan

        cell_grid_col.append(row_grid_cols)

    # 2단계: restart 셀들에 대해 rowspan 계산
    for row_idx, row_info in enumerate(all_cells_info):
        for cell_idx, (colspan, vmerge_status) in enumerate(row_info):
            if cell_idx >= len(cell_grid_col[row_idx]):
                continue
            start_col, end_col = cell_grid_col[row_idx][cell_idx]

            if vmerge_status == 'restart':
                # 아래 행들에서 같은 owner를 가진 셀 찾기
                rowspan = 1
                for next_row in range(row_idx + 1, num_rows):
                    if start_col < max_cols and merge_info[next_row][start_col] == (row_idx, start_col, colspan):
                        rowspan += 1
                    else:
                        break
                rowspan_map[(row_idx, start_col)] = rowspan

            elif vmerge_status == 'none':
                rowspan_map[(row_idx, start_col)] = 1

    return rowspan_map, cell_grid_col


def estimate_column_count(first_row) -> int:
    """
    첫 번째 행에서 열 수를 추정합니다.

    Args:
        first_row: 첫 번째 행 XML 요소

    Returns:
        추정된 열 개수
    """
    cells = first_row.findall('w:tc', NAMESPACES)
    total_cols = 0
    for cell in cells:
        colspan = 1
        tcPr = cell.find('w:tcPr', NAMESPACES)
        if tcPr is not None:
            gs = tcPr.find('w:gridSpan', NAMESPACES)
            if gs is not None:
                try:
                    colspan = int(gs.get(qn('w:val'), 1))
                except (ValueError, TypeError):
                    colspan = 1
        total_cols += colspan
    return total_cols


def extract_cell_text(cell_elem) -> str:
    """
    셀 내용을 추출합니다.

    Args:
        cell_elem: 셀 XML 요소

    Returns:
        셀 텍스트 내용
    """
    texts = []

    for p in cell_elem.findall('.//w:p', NAMESPACES):
        p_texts = []
        for t in p.findall('.//w:t', NAMESPACES):
            if t.text:
                p_texts.append(t.text)
        if p_texts:
            texts.append(''.join(p_texts))

    return '\n'.join(texts)


def extract_table_as_text(table_elem) -> str:
    """
    테이블을 평문 형식으로 추출합니다 (폴백용).

    Args:
        table_elem: table XML 요소

    Returns:
        평문 테이블 문자열
    """
    try:
        rows_text = []

        for row in table_elem.findall('w:tr', NAMESPACES):
            row_cells = []
            for cell in row.findall('w:tc', NAMESPACES):
                cell_text = extract_cell_text(cell)
                row_cells.append(cell_text.replace('\n', ' '))
            if any(c.strip() for c in row_cells):
                rows_text.append(" | ".join(row_cells))

        return "\n".join(rows_text) if rows_text else ""

    except Exception:
        return ""


__all__ = [
    'TableCellInfo',
    'process_table_element',
    'calculate_all_rowspans',
    'estimate_column_count',
    'extract_cell_text',
    'extract_table_as_text',
]
