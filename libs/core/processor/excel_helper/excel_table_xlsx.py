"""
Excel XLSX 테이블 변환 모듈

XLSX 시트를 Markdown 또는 HTML 테이블로 변환합니다.
병합셀이 있으면 HTML, 없으면 Markdown을 사용합니다.
layout_detect_range를 통해 실제 데이터 영역만 추출합니다.
object_detect를 통해 개별 객체(테이블)별로 청킹할 수 있습니다.
"""

import logging
from typing import Optional, List
from .excel_layout_detector import layout_detect_range_xlsx, object_detect_xlsx, LayoutRange

logger = logging.getLogger("document-processor")


def has_merged_cells_xlsx(ws, layout: Optional[LayoutRange] = None) -> bool:
    """
    XLSX 워크시트에 병합셀이 존재하는지 확인합니다.
    layout이 주어지면 해당 영역 내의 병합셀만 확인합니다.

    Args:
        ws: openpyxl Worksheet 객체
        layout: 검사할 레이아웃 범위 (None이면 전체 시트)

    Returns:
        병합셀이 존재하면 True
    """
    try:
        if len(ws.merged_cells.ranges) == 0:
            return False
        
        # layout이 없으면 전체 시트에 병합셀 존재 여부만 확인
        if layout is None:
            return True
        
        # layout 영역 내에 병합셀이 있는지 확인
        for merged_range in ws.merged_cells.ranges:
            # 병합 영역이 layout 영역과 겹치는지 확인
            if (merged_range.min_row <= layout.max_row and
                merged_range.max_row >= layout.min_row and
                merged_range.min_col <= layout.max_col and
                merged_range.max_col >= layout.min_col):
                return True
        
        return False
    except Exception:
        return False


def convert_xlsx_sheet_to_table(ws, layout: Optional[LayoutRange] = None) -> str:
    """
    XLSX 워크시트를 테이블로 변환합니다.
    병합셀이 없으면 Markdown, 있으면 HTML로 변환합니다.
    layout이 None이면 자동으로 감지합니다.

    Args:
        ws: openpyxl Worksheet 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        변환된 테이블 문자열
    """
    # layout이 없으면 자동 감지
    if layout is None:
        layout = layout_detect_range_xlsx(ws)
        if layout is None:
            logger.debug("No data found in worksheet")
            return ""
    
    if has_merged_cells_xlsx(ws, layout):
        logger.debug("Merged cells detected in XLSX, using HTML format")
        return convert_xlsx_sheet_to_html(ws, layout)
    else:
        logger.debug("No merged cells in XLSX, using Markdown format")
        return convert_xlsx_sheet_to_markdown(ws, layout)


def convert_xlsx_sheet_to_markdown(ws, layout: Optional[LayoutRange] = None) -> str:
    """
    XLSX 워크시트를 Markdown 테이블로 변환합니다.
    layout_detect_range를 통해 실제 데이터 영역만 추출합니다.

    Args:
        ws: openpyxl Worksheet 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        Markdown 테이블 문자열
    """
    try:
        # layout이 없으면 자동 감지
        if layout is None:
            layout = layout_detect_range_xlsx(ws)
            if layout is None:
                return ""

        # 병합 셀의 시작점이 layout 밖에 있는 경우, layout 내 첫 번째 셀에 값을 표시
        merged_value_override = {}  # (row, col) -> value
        for merged_range in ws.merged_cells.ranges:
            mr_min_row, mr_min_col = merged_range.min_row, merged_range.min_col
            mr_max_row, mr_max_col = merged_range.max_row, merged_range.max_col
            
            # layout 영역과 겹치는지 확인
            if (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
                
                # 병합 셀의 시작점이 layout 밖에 있는 경우
                start_in_layout = (layout.min_row <= mr_min_row <= layout.max_row and
                                   layout.min_col <= mr_min_col <= layout.max_col)
                
                if not start_in_layout:
                    merged_value = ws.cell(row=mr_min_row, column=mr_min_col).value
                    if merged_value is not None:
                        first_row_in_layout = max(mr_min_row, layout.min_row)
                        first_col_in_layout = max(mr_min_col, layout.min_col)
                        merged_value_override[(first_row_in_layout, first_col_in_layout)] = merged_value

        md_parts = []
        row_count = 0

        for row_idx in range(layout.min_row, layout.max_row + 1):
            cells = []
            row_has_content = False

            for col_idx in range(layout.min_col, layout.max_col + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell_value = ""

                # 병합 셀 override 확인
                if (row_idx, col_idx) in merged_value_override:
                    cell_value = str(merged_value_override[(row_idx, col_idx)]).strip()
                    if cell_value:
                        row_has_content = True
                elif cell.value is not None:
                    cell_value = str(cell.value).strip()
                    if cell_value:
                        row_has_content = True

                # Markdown 테이블에서 파이프는 이스케이프 필요
                cell_value = cell_value.replace("|", "\\|")
                cell_value = cell_value.replace("\n", " ")
                cells.append(cell_value)

            if not row_has_content:
                continue

            row_str = "| " + " | ".join(cells) + " |"
            md_parts.append(row_str)
            row_count += 1

            # 첫 번째 데이터 행 다음에 구분선 추가
            if row_count == 1:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                md_parts.append(separator)

        return "\n".join(md_parts) if md_parts else ""

    except Exception as e:
        logger.warning(f"Error converting sheet to Markdown: {e}")
        return ""


def convert_xlsx_sheet_to_html(ws, layout: Optional[LayoutRange] = None) -> str:
    """
    XLSX 워크시트를 HTML 테이블로 변환합니다.
    셀 병합(rowspan/colspan)을 지원합니다.
    layout_detect_range를 통해 실제 데이터 영역만 추출합니다.
    
    병합셀이 있는 경우 빈 행도 테이블 구조의 일부이므로 포함합니다.

    Args:
        ws: openpyxl Worksheet 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        HTML 테이블 문자열
    """
    try:
        # layout이 없으면 자동 감지
        if layout is None:
            layout = layout_detect_range_xlsx(ws)
            if layout is None:
                return ""

        # 병합된 셀 정보 수집 (layout 영역 내만)
        merged_cells_info = {}  # (row, col) -> (rowspan, colspan)
        skip_cells = set()  # 건너뛸 셀 (병합된 영역의 일부)
        # 병합 셀의 시작점이 layout 밖에 있는 경우, layout 내 첫 번째 셀에 값을 표시
        merged_value_override = {}  # (row, col) -> value

        for merged_range in ws.merged_cells.ranges:
            mr_min_row, mr_min_col = merged_range.min_row, merged_range.min_col
            mr_max_row, mr_max_col = merged_range.max_row, merged_range.max_col
            
            # layout 영역과 겹치는 병합 셀만 처리
            if (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
                
                # 병합 셀의 시작점이 layout 안에 있는지 확인
                start_in_layout = (layout.min_row <= mr_min_row <= layout.max_row and
                                   layout.min_col <= mr_min_col <= layout.max_col)
                
                if start_in_layout:
                    # 일반적인 경우: 병합 정보 저장
                    rowspan = mr_max_row - mr_min_row + 1
                    colspan = mr_max_col - mr_min_col + 1
                    merged_cells_info[(mr_min_row, mr_min_col)] = (rowspan, colspan)
                    
                    # 병합된 영역의 나머지 셀들은 건너뛰기
                    for r in range(mr_min_row, mr_max_row + 1):
                        for c in range(mr_min_col, mr_max_col + 1):
                            if r != mr_min_row or c != mr_min_col:
                                skip_cells.add((r, c))
                else:
                    # 병합 셀의 시작점이 layout 밖에 있는 경우
                    # layout 내 첫 번째 셀에 병합 셀의 값을 표시
                    merged_value = ws.cell(row=mr_min_row, column=mr_min_col).value
                    if merged_value is not None:
                        # layout 내에서 병합 영역의 첫 번째 셀 찾기
                        first_row_in_layout = max(mr_min_row, layout.min_row)
                        first_col_in_layout = max(mr_min_col, layout.min_col)
                        merged_value_override[(first_row_in_layout, first_col_in_layout)] = merged_value
                    
                    # layout 내의 병합 영역 나머지 셀들은 건너뛰기
                    for r in range(max(mr_min_row, layout.min_row), min(mr_max_row, layout.max_row) + 1):
                        for c in range(max(mr_min_col, layout.min_col), min(mr_max_col, layout.max_col) + 1):
                            # 값을 표시할 첫 번째 셀은 skip하지 않음
                            if (r, c) in merged_value_override:
                                continue
                            skip_cells.add((r, c))

        # HTML 생성
        html_parts = ["<table border='1'>"]
        has_data = False

        for row_idx in range(layout.min_row, layout.max_row + 1):
            row_parts = ["<tr>"]

            for col_idx in range(layout.min_col, layout.max_col + 1):
                # 건너뛸 셀 확인 (병합된 영역의 일부)
                if (row_idx, col_idx) in skip_cells:
                    continue

                cell = ws.cell(row=row_idx, column=col_idx)

                # 셀 값 추출 (병합 셀 override 확인)
                cell_value = ""
                if (row_idx, col_idx) in merged_value_override:
                    cell_value = str(merged_value_override[(row_idx, col_idx)]).strip()
                    if cell_value:
                        has_data = True
                elif cell.value is not None:
                    cell_value = str(cell.value).strip()
                    if cell_value:
                        has_data = True

                # HTML 이스케이프
                cell_value = _escape_html(cell_value)

                # 첫 번째 행은 헤더로 처리
                tag = "th" if row_idx == layout.min_row else "td"

                # 병합 속성
                attrs = []
                if (row_idx, col_idx) in merged_cells_info:
                    rowspan, colspan = merged_cells_info[(row_idx, col_idx)]
                    if rowspan > 1:
                        attrs.append(f"rowspan='{rowspan}'")
                    if colspan > 1:
                        attrs.append(f"colspan='{colspan}'")

                attr_str = " " + " ".join(attrs) if attrs else ""
                row_parts.append(f"<{tag}{attr_str}>{cell_value}</{tag}>")

            row_parts.append("</tr>")
            
            # 모든 행을 추가 (빈 행도 테이블 구조의 일부)
            html_parts.append("".join(row_parts))

        html_parts.append("</table>")

        if has_data:
            return "\n".join(html_parts)
        return ""

    except Exception as e:
        logger.warning(f"Error converting sheet to HTML: {e}")
        return ""


def _escape_html(text: str) -> str:
    """
    HTML 특수 문자를 이스케이프합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        이스케이프된 텍스트
    """
    if not text:
        return ""
    
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    
    return text


def convert_xlsx_objects_to_tables(ws, layout: Optional[LayoutRange] = None) -> List[str]:
    """
    XLSX 워크시트에서 개별 객체(테이블)를 감지하고 각각을 테이블 문자열로 변환합니다.
    
    알고리즘:
    1. 테두리가 있는 영역을 먼저 개별 개체로 인식
    2. 테두리가 없는 값 영역을 감지
    3. 완전히 인접한 개체들을 병합
    4. 각 객체를 테이블로 변환
    
    Args:
        ws: openpyxl Worksheet 객체
        layout: 탐색할 레이아웃 범위 (None이면 자동 감지)
    
    Returns:
        개별 객체 테이블 문자열 목록 (위→아래, 왼쪽→오른쪽 순서)
    """
    objects = object_detect_xlsx(ws, layout)
    
    if not objects:
        return []
    
    tables = []
    for obj_layout in objects:
        table_str = convert_xlsx_sheet_to_table(ws, obj_layout)
        # 빈 테이블 필터링 (공백, 줄바꿈, 테이블 기호만 있는 경우 제외)
        if table_str and table_str.strip():
            # Markdown 테이블에서 실제 데이터가 있는지 확인
            # 헤더 구분선(---)만 있고 데이터가 없는 경우 제외
            lines = [line.strip() for line in table_str.strip().split('\n') if line.strip()]
            has_data = False
            for line in lines:
                # 구분선이 아닌 행에서 | 사이에 실제 값이 있는지 확인
                if '---' not in line:
                    # | col1 | col2 | 형태에서 값 추출
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if parts:
                        has_data = True
                        break
            
            if has_data:
                tables.append(table_str)
    
    logger.debug(f"Converted {len(tables)} objects to tables (XLSX)")
    return tables
