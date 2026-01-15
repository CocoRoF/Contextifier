"""
Excel XLS 테이블 변환 모듈

XLS 시트를 Markdown 또는 HTML 테이블로 변환합니다.
병합셀이 있으면 HTML, 없으면 Markdown을 사용합니다.
layout_detect_range를 통해 실제 데이터 영역만 추출합니다.
object_detect를 통해 개별 객체(테이블)별로 청킹할 수 있습니다.
"""

import logging
from typing import Optional, List
from .excel_layout_detector import layout_detect_range_xls, object_detect_xls, LayoutRange

logger = logging.getLogger("document-processor")

# xlrd (xls)
try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False


def has_merged_cells_xls(sheet, layout: Optional[LayoutRange] = None) -> bool:
    """
    XLS 시트에 병합셀이 존재하는지 확인합니다.
    layout이 주어지면 해당 영역 내의 병합셀만 확인합니다.

    Args:
        sheet: xlrd Sheet 객체
        layout: 검사할 레이아웃 범위 (None이면 전체 시트)

    Returns:
        병합셀이 존재하면 True
    """
    try:
        if len(sheet.merged_cells) == 0:
            return False
        
        # layout이 없으면 전체 시트에 병합셀 존재 여부만 확인
        if layout is None:
            return True
        
        # layout 영역 내에 병합셀이 있는지 확인
        # xlrd merged_cells는 (rlo, rhi, clo, chi) 튜플, 0-based
        for (rlo, rhi, clo, chi) in sheet.merged_cells:
            # 1-based로 변환하여 비교
            mr_min_row = rlo + 1
            mr_max_row = rhi  # rhi는 exclusive이므로 +1 불필요
            mr_min_col = clo + 1
            mr_max_col = chi  # chi는 exclusive이므로 +1 불필요
            
            if (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
                return True
        
        return False
    except Exception:
        return False


def convert_xls_sheet_to_table(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """
    XLS 시트를 테이블로 변환합니다.
    병합셀이 없으면 Markdown, 있으면 HTML로 변환합니다.
    layout이 None이면 자동으로 감지합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        변환된 테이블 문자열
    """
    # layout이 없으면 자동 감지
    if layout is None:
        layout = layout_detect_range_xls(sheet)
        if layout is None:
            logger.debug("No data found in XLS sheet")
            return ""
    
    if has_merged_cells_xls(sheet, layout):
        logger.debug("Merged cells detected in XLS, using HTML format")
        return convert_xls_sheet_to_html(sheet, wb, layout)
    else:
        logger.debug("No merged cells in XLS, using Markdown format")
        return convert_xls_sheet_to_markdown(sheet, wb, layout)


def convert_xls_sheet_to_markdown(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """
    XLS 시트를 Markdown 테이블로 변환합니다.
    layout_detect_range를 통해 실제 데이터 영역만 추출합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        Markdown 테이블 문자열
    """
    if not XLRD_AVAILABLE:
        logger.warning("xlrd not available for XLS processing")
        return ""

    try:
        # layout이 없으면 자동 감지
        if layout is None:
            layout = layout_detect_range_xls(sheet)
            if layout is None:
                return ""

        md_parts = []
        row_count = 0

        # 1-based layout을 0-based로 변환하여 사용
        for row_idx in range(layout.min_row - 1, layout.max_row):  # 0-based
            cells = []
            row_has_content = False

            for col_idx in range(layout.min_col - 1, layout.max_col):  # 0-based
                cell_value = ""
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value:
                        cell_type = sheet.cell_type(row_idx, col_idx)
                        cell_value = _format_xls_cell_value(value, cell_type, wb)

                        if cell_value:
                            row_has_content = True
                except Exception:
                    pass

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
        logger.warning(f"Error converting XLS sheet to Markdown: {e}")
        return ""


def convert_xls_sheet_to_html(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """
    XLS 시트를 HTML 테이블로 변환합니다.
    병합셀(rowspan/colspan)을 지원합니다.
    layout_detect_range를 통해 실제 데이터 영역만 추출합니다.
    
    병합셀이 있는 경우 빈 행도 테이블 구조의 일부이므로 포함합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        HTML 테이블 문자열
    """
    if not XLRD_AVAILABLE:
        logger.warning("xlrd not available for XLS processing")
        return ""

    try:
        # layout이 없으면 자동 감지
        if layout is None:
            layout = layout_detect_range_xls(sheet)
            if layout is None:
                return ""

        # 병합된 셀 정보 수집 (layout 영역 내만)
        # xlrd에서 merged_cells는 (rlo, rhi, clo, chi) 튜플 리스트
        # rlo, clo는 0-based, rhi, chi는 exclusive
        merged_cells_info = {}  # (row, col) -> (rowspan, colspan), 0-based
        skip_cells = set()  # 건너뛸 셀 (병합된 영역의 일부), 0-based

        for (rlo, rhi, clo, chi) in sheet.merged_cells:
            # layout 영역과 겹치는 병합 셀만 처리 (1-based로 변환하여 비교)
            mr_min_row = rlo + 1
            mr_max_row = rhi  # exclusive
            mr_min_col = clo + 1
            mr_max_col = chi  # exclusive
            
            if (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
                
                rowspan = rhi - rlo
                colspan = chi - clo

                merged_cells_info[(rlo, clo)] = (rowspan, colspan)

                # 병합된 영역의 나머지 셀들은 건너뛰기
                for r in range(rlo, rhi):
                    for c in range(clo, chi):
                        if r != rlo or c != clo:
                            skip_cells.add((r, c))

        # HTML 생성
        html_parts = ["<table border='1'>"]
        has_data = False

        # 1-based layout을 0-based로 변환하여 사용
        for row_idx in range(layout.min_row - 1, layout.max_row):  # 0-based
            row_parts = ["<tr>"]

            for col_idx in range(layout.min_col - 1, layout.max_col):  # 0-based
                # 건너뛸 셀 확인 (병합된 영역의 일부)
                if (row_idx, col_idx) in skip_cells:
                    continue

                cell_value = ""
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value:
                        cell_type = sheet.cell_type(row_idx, col_idx)
                        cell_value = _format_xls_cell_value(value, cell_type, wb)

                        if cell_value:
                            has_data = True
                except Exception:
                    pass

                # HTML 이스케이프
                cell_value = _escape_html(cell_value)

                # 첫 번째 행은 헤더로 처리
                tag = "th" if row_idx == layout.min_row - 1 else "td"

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
        logger.warning(f"Error converting XLS sheet to HTML: {e}")
        return ""


def _format_xls_cell_value(value, cell_type, wb) -> str:
    """
    XLS 셀 값을 문자열로 포맷합니다.
    
    Args:
        value: 셀 값
        cell_type: xlrd 셀 타입
        wb: xlrd Workbook 객체
        
    Returns:
        포맷된 문자열
    """
    if not XLRD_AVAILABLE:
        return str(value).strip() if value else ""
    
    try:
        if cell_type == xlrd.XL_CELL_NUMBER:
            if value == int(value):
                return str(int(value))
            else:
                return str(value)
        elif cell_type == xlrd.XL_CELL_DATE:
            try:
                date_tuple = xlrd.xldate_as_tuple(value, wb.datemode)
                return f"{date_tuple[0]:04d}-{date_tuple[1]:02d}-{date_tuple[2]:02d}"
            except Exception:
                return str(value)
        else:
            return str(value).strip()
    except Exception:
        return str(value).strip() if value else ""


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


def convert_xls_objects_to_tables(sheet, wb, layout: Optional[LayoutRange] = None) -> List[str]:
    """
    XLS 시트에서 개별 객체(테이블)를 감지하고 각각을 테이블 문자열로 변환합니다.
    
    알고리즘:
    1. 테두리가 있는 영역을 먼저 개별 개체로 인식
    2. 테두리가 없는 값 영역을 감지
    3. 완전히 인접한 개체들을 병합
    4. 각 객체를 테이블로 변환
    
    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 탐색할 레이아웃 범위 (None이면 자동 감지)
    
    Returns:
        개별 객체 테이블 문자열 목록 (위→아래, 왼쪽→오른쪽 순서)
    """
    objects = object_detect_xls(sheet, wb, layout)
    
    if not objects:
        return []
    
    tables = []
    for obj_layout in objects:
        table_str = convert_xls_sheet_to_table(sheet, wb, obj_layout)
        # 빈 테이블 필터링 (공백, 줄바꿈, 테이블 기호만 있는 경우 제외)
        if table_str and table_str.strip():
            # Markdown 테이블에서 실제 데이터가 있는지 확인
            lines = [line.strip() for line in table_str.strip().split('\n') if line.strip()]
            has_data = False
            for line in lines:
                if '---' not in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if parts:
                        has_data = True
                        break
            
            if has_data:
                tables.append(table_str)
    
    logger.debug(f"Converted {len(tables)} objects to tables (XLS)")
    return tables
