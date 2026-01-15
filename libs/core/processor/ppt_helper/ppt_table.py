"""
PPT 테이블 처리 모듈

포함 함수:
- is_simple_table(): 단순 표인지 확인
- extract_simple_table_as_text(): 단순 표를 텍스트로 추출
- convert_table_to_html(): 테이블을 HTML로 변환 (병합 지원)
- extract_table_as_text(): 테이블을 평문으로 추출

병합 셀(rowspan, colspan) 정확히 처리
"""
import logging
from typing import Dict

logger = logging.getLogger("document-processor")


def is_simple_table(table) -> bool:
    """
    단순 표인지 확인합니다.
    
    단순 표 조건:
    - 행 또는 열 중 하나라도 1개인 경우 (1xN, Nx1)
    
    이런 표는 텍스트박스처럼 사용되는 경우가 많아 HTML 테이블 대신 일반 텍스트로 처리합니다.
    
    Args:
        table: python-pptx의 Table 객체
        
    Returns:
        True면 단순 표 (텍스트로 처리), False면 일반 표 (HTML로 처리)
    """
    try:
        num_rows = len(table.rows)
        num_cols = len(table.columns)
        
        # 행 또는 열이 1개인 경우 (1xN, Nx1)
        if num_rows == 1 or num_cols == 1:
            return True
        
        return False
    except Exception:
        return False


def extract_simple_table_as_text(table) -> str:
    """
    단순 표(1xN, Nx1, 2x2 이하)를 일반 텍스트로 추출합니다.
    
    Args:
        table: python-pptx의 Table 객체
        
    Returns:
        줄바꿈으로 구분된 텍스트
    """
    try:
        texts = []
        for row in table.rows:
            row_texts = []
            for cell in row.cells:
                cell_text = cell.text.strip() if cell.text else ""
                if cell_text:
                    row_texts.append(cell_text)
            if row_texts:
                # 한 행의 셀들은 공백으로 구분
                texts.append(" ".join(row_texts))
        
        # 행들은 줄바꿈으로 구분
        return "\n".join(texts) if texts else ""
    except Exception:
        return ""


def convert_table_to_html(table) -> str:
    """
    테이블을 HTML 형식으로 변환합니다.
    병합된 셀(rowspan, colspan)을 정확히 처리합니다.
    
    Args:
        table: python-pptx의 Table 객체
        
    Returns:
        HTML 테이블 문자열
    """
    try:
        num_rows = len(table.rows)
        num_cols = len(table.columns)
        
        if num_rows == 0 or num_cols == 0:
            return ""
        
        # 병합 정보를 저장할 2D 배열
        # None: 아직 처리 안됨, 'skip': 병합으로 인해 스킵할 셀
        cell_info = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        # 1단계: 병합 정보 수집
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                if cell_info[row_idx][col_idx] == 'skip':
                    continue
                
                cell = table.cell(row_idx, col_idx)
                
                # 병합 정보 추출
                merge_info = _get_cell_merge_info(cell, table, row_idx, col_idx, num_rows, num_cols)
                
                rowspan = merge_info['rowspan']
                colspan = merge_info['colspan']
                
                # 병합된 영역의 다른 셀들을 'skip'으로 표시
                for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                    for c in range(col_idx, min(col_idx + colspan, num_cols)):
                        if r == row_idx and c == col_idx:
                            # 시작 셀에는 병합 정보 저장
                            cell_info[r][c] = {
                                'rowspan': rowspan,
                                'colspan': colspan,
                                'text': cell.text.strip() if cell.text else ""
                            }
                        else:
                            cell_info[r][c] = 'skip'
        
        # 2단계: HTML 생성
        html_parts = ["<table border='1'>"]
        
        for row_idx in range(num_rows):
            html_parts.append("<tr>")
            
            for col_idx in range(num_cols):
                info = cell_info[row_idx][col_idx]
                
                # 병합으로 스킵할 셀
                if info == 'skip':
                    continue
                
                # 셀 정보가 없으면 기본값
                if info is None:
                    cell = table.cell(row_idx, col_idx)
                    info = {
                        'rowspan': 1,
                        'colspan': 1,
                        'text': cell.text.strip() if cell.text else ""
                    }
                
                # 태그 결정 (첫 행은 th, 나머지는 td)
                tag = "th" if row_idx == 0 else "td"
                
                # 속성 생성
                attrs = []
                if info['rowspan'] > 1:
                    attrs.append(f"rowspan='{info['rowspan']}'")
                if info['colspan'] > 1:
                    attrs.append(f"colspan='{info['colspan']}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                # 텍스트 이스케이프
                text = _escape_html(info['text'])
                
                html_parts.append(f"<{tag}{attr_str}>{text}</{tag}>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    except Exception as e:
        logger.warning(f"Error converting table to HTML: {e}")
        return extract_table_as_text(table)


def extract_table_as_text(table) -> str:
    """
    테이블을 평문 형식으로 추출합니다.
    
    Args:
        table: python-pptx의 Table 객체
        
    Returns:
        파이프(|)로 구분된 텍스트
    """
    try:
        rows_text = []
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                cell_text = cell.text.strip() if cell.text else ""
                row_cells.append(cell_text)
            if any(c for c in row_cells):
                rows_text.append(" | ".join(row_cells))

        return "\n".join(rows_text) if rows_text else ""

    except Exception:
        return ""


def _get_cell_merge_info(cell, table, row_idx: int, col_idx: int, 
                         num_rows: int, num_cols: int) -> Dict[str, int]:
    """
    셀의 병합 정보를 추출합니다.
    
    python-pptx에서 병합 셀을 감지하는 방법:
    1. cell.is_merge_origin: 병합의 시작점인지
    2. cell.is_spanned: 다른 셀에 의해 병합된 셀인지
    3. cell.span_height: 세로 병합 크기
    4. cell.span_width: 가로 병합 크기
    
    Args:
        cell: 테이블 셀 객체
        table: 테이블 객체
        row_idx: 현재 행 인덱스
        col_idx: 현재 열 인덱스
        num_rows: 총 행 수
        num_cols: 총 열 수
        
    Returns:
        {'rowspan': int, 'colspan': int}
    """
    rowspan = 1
    colspan = 1
    
    try:
        # 방법 1: python-pptx의 내장 속성 사용 (권장)
        if hasattr(cell, 'is_merge_origin') and cell.is_merge_origin:
            # 병합의 시작 셀
            if hasattr(cell, 'span_height'):
                rowspan = cell.span_height
            if hasattr(cell, 'span_width'):
                colspan = cell.span_width
            return {'rowspan': rowspan, 'colspan': colspan}
        
        # 이미 병합된 셀 (다른 셀에 의해 덮어진 경우) - 스킵 대상
        if hasattr(cell, 'is_spanned') and cell.is_spanned:
            return {'rowspan': 0, 'colspan': 0}  # 스킵 표시
        
        # 방법 2: XML 직접 파싱 (폴백)
        tc = cell._tc
        
        # gridSpan 속성 (가로 병합)
        grid_span = tc.get('gridSpan')
        if grid_span:
            colspan = int(grid_span)
        
        # rowSpan 속성 (세로 병합)
        row_span_attr = tc.get('rowSpan')
        if row_span_attr:
            rowspan = int(row_span_attr)
        
        # 방법 3: 동일 셀 참조 비교 (추가 폴백)
        if colspan == 1:
            colspan = _detect_colspan_by_reference(table, row_idx, col_idx, num_cols)
        
        if rowspan == 1:
            rowspan = _detect_rowspan_by_reference(table, row_idx, col_idx, num_rows)
    
    except Exception as e:
        logger.debug(f"Error getting merge info: {e}")
    
    return {'rowspan': rowspan, 'colspan': colspan}


def _detect_colspan_by_reference(table, row_idx: int, col_idx: int, num_cols: int) -> int:
    """
    셀 참조 비교로 colspan을 감지합니다.
    
    Args:
        table: 테이블 객체
        row_idx: 현재 행 인덱스
        col_idx: 현재 열 인덱스
        num_cols: 총 열 수
        
    Returns:
        colspan 값
    """
    colspan = 1
    try:
        current_cell = table.cell(row_idx, col_idx)
        
        for c in range(col_idx + 1, num_cols):
            next_cell = table.cell(row_idx, c)
            
            # _tc 참조가 같으면 병합된 셀
            if next_cell._tc is current_cell._tc:
                colspan += 1
            else:
                break
    except Exception:
        pass
    
    return colspan


def _detect_rowspan_by_reference(table, row_idx: int, col_idx: int, num_rows: int) -> int:
    """
    셀 참조 비교로 rowspan을 감지합니다.
    
    Args:
        table: 테이블 객체
        row_idx: 현재 행 인덱스
        col_idx: 현재 열 인덱스
        num_rows: 총 행 수
        
    Returns:
        rowspan 값
    """
    rowspan = 1
    try:
        current_cell = table.cell(row_idx, col_idx)
        
        for r in range(row_idx + 1, num_rows):
            next_cell = table.cell(r, col_idx)
            
            if next_cell._tc is current_cell._tc:
                rowspan += 1
            else:
                break
    except Exception:
        pass
    
    return rowspan


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


def debug_table_structure(table):
    """
    테이블 구조를 디버깅합니다.
    병합 정보 확인을 위해 사용합니다.
    
    Args:
        table: python-pptx의 Table 객체
    """
    logger.debug("=== Table Structure Debug ===")
    logger.debug(f"Rows: {len(table.rows)}, Cols: {len(table.columns)}")
    
    for row_idx in range(len(table.rows)):
        for col_idx in range(len(table.columns)):
            try:
                cell = table.cell(row_idx, col_idx)
                tc = cell._tc
                
                # XML 속성 확인
                grid_span = tc.get('gridSpan', '1')
                row_span = tc.get('rowSpan', '1')
                
                # python-pptx 속성 확인
                is_merge_origin = getattr(cell, 'is_merge_origin', None)
                is_spanned = getattr(cell, 'is_spanned', None)
                span_width = getattr(cell, 'span_width', None)
                span_height = getattr(cell, 'span_height', None)
                
                text_preview = cell.text[:20] if cell.text else ""
                
                logger.debug(
                    f"[{row_idx},{col_idx}] "
                    f"text='{text_preview}' "
                    f"gridSpan={grid_span} rowSpan={row_span} "
                    f"is_merge_origin={is_merge_origin} "
                    f"is_spanned={is_spanned} "
                    f"span_width={span_width} span_height={span_height}"
                )
            except Exception as e:
                logger.debug(f"[{row_idx},{col_idx}] Error: {e}")
    
    logger.debug("=== End Debug ===")
