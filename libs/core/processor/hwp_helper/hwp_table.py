# service/document_processor/processor/hwp_helper/hwp_table.py
"""
HWP 테이블 파싱 유틸리티

HWP 5.0 OLE 파일에서 테이블을 파싱하여 HTML로 변환합니다.
- parse_table: 테이블 컨트롤을 파싱하여 HTML 또는 리스트로 변환
- build_table_grid: 셀 정보를 그리드로 구성
- render_table_html: 그리드를 HTML 테이블로 렌더링
"""
import struct
import logging
from typing import Dict, Callable, Optional, Any, Set

import olefile

from libs.core.processor.hwp_helper.hwp_constants import (
    HWPTAG_TABLE,
    HWPTAG_LIST_HEADER,
)
from libs.core.processor.hwp_helper.hwp_record import HwpRecord

logger = logging.getLogger("document-processor")


def parse_table(
    ctrl_header: HwpRecord,
    traverse_callback: Callable[[HwpRecord, Any, Dict, Any, Set], str],
    ole: olefile.OleFileIO = None,
    bin_data_map: Dict = None,
    app_db: Any = None,
    processed_images: Optional[Set[str]] = None
) -> str:
    """
    HWP 테이블 컨트롤을 파싱합니다.

    테이블 구조를 분석하여:
    - 1×1 테이블: 셀 내용만 텍스트로 반환 (컨테이너 테이블)
    - 단일 컬럼 테이블 (1열, 다중 행): 셀 내용을 줄바꿈으로 구분하여 반환
    - 다중 컬럼 테이블 (2+ 열): HTML 테이블로 변환

    Args:
        ctrl_header: 테이블을 포함하는 CTRL_HEADER 레코드
        traverse_callback: 셀 내용을 추출하기 위한 트리 순회 콜백 함수
        ole: OLE 파일 객체
        bin_data_map: BinData 매핑 정보
        app_db: 데이터베이스 연결
        processed_images: 처리된 이미지 경로 집합

    Returns:
        HTML 테이블, 텍스트 리스트, 또는 단순 텍스트
    """
    try:
        table_rec = next((c for c in ctrl_header.children if c.tag_id == HWPTAG_TABLE), None)
        if not table_rec:
            return ""

        if len(table_rec.payload) < 8:
            return ""

        row_cnt = struct.unpack('<H', table_rec.payload[4:6])[0]
        col_cnt = struct.unpack('<H', table_rec.payload[6:8])[0]

        # 셀 그리드 구성
        grid = build_table_grid(
            ctrl_header,
            traverse_callback,
            ole,
            bin_data_map,
            app_db,
            processed_images
        )

        # 1×1 테이블 -> 셀 내용만 반환 (컨테이너 테이블)
        if row_cnt == 1 and col_cnt == 1:
            if (0, 0) in grid:
                return grid[(0, 0)]['text']
            return ""

        # 단일 컬럼 테이블 (1열, 다중 행) -> 셀 내용을 줄바꿈으로 구분
        if col_cnt == 1:
            text_items = []
            for r in range(row_cnt):
                if (r, 0) in grid:
                    cell_text = grid[(r, 0)]['text']
                    if cell_text:
                        text_items.append(cell_text)
            if text_items:
                return "\n\n".join(text_items)
            return ""

        # HTML 테이블 생성 (2+ 컬럼)
        return render_table_html(grid, row_cnt, col_cnt)

    except Exception as e:
        logger.warning(f"Failed to parse HWP table: {e}")
        return "[Table Extraction Failed]"


def build_table_grid(
    ctrl_header: HwpRecord,
    traverse_callback: Callable[[HwpRecord, Any, Dict, Any, Set], str],
    ole: olefile.OleFileIO = None,
    bin_data_map: Dict = None,
    app_db: Any = None,
    processed_images: Optional[Set[str]] = None
) -> Dict:
    """
    테이블 셀 정보를 그리드로 구성합니다.

    Args:
        ctrl_header: 테이블을 포함하는 CTRL_HEADER 레코드
        traverse_callback: 셀 내용을 추출하기 위한 트리 순회 콜백 함수
        ole: OLE 파일 객체
        bin_data_map: BinData 매핑 정보
        app_db: 데이터베이스 연결
        processed_images: 처리된 이미지 경로 집합

    Returns:
        (row_idx, col_idx) -> {'text', 'rowspan', 'colspan'} 딕셔너리
    """
    grid = {}

    cells = [c for c in ctrl_header.children if c.tag_id == HWPTAG_LIST_HEADER]

    for cell in cells:
        if len(cell.payload) < 16:
            continue

        para_count = struct.unpack('<H', cell.payload[0:2])[0]
        col_idx = struct.unpack('<H', cell.payload[8:10])[0]
        row_idx = struct.unpack('<H', cell.payload[10:12])[0]
        col_span = struct.unpack('<H', cell.payload[12:14])[0]
        row_span = struct.unpack('<H', cell.payload[14:16])[0]

        cell_text_parts = []

        if cell.children:
            for child in cell.children:
                t = traverse_callback(child, ole, bin_data_map, app_db, processed_images)
                cell_text_parts.append(t)
        else:
            siblings = list(cell.get_next_siblings(para_count))
            for sibling in siblings:
                t = traverse_callback(sibling, ole, bin_data_map, app_db, processed_images)
                cell_text_parts.append(t)

        cell_content = "".join(cell_text_parts).strip()

        grid[(row_idx, col_idx)] = {
            'text': cell_content,
            'rowspan': row_span,
            'colspan': col_span
        }

    return grid


def render_table_html(grid: Dict, row_cnt: int, col_cnt: int) -> str:
    """
    그리드를 HTML 테이블로 렌더링합니다.

    Args:
        grid: (row_idx, col_idx) -> {'text', 'rowspan', 'colspan'} 딕셔너리
        row_cnt: 테이블 행 수
        col_cnt: 테이블 열 수

    Returns:
        HTML 테이블 문자열
    """
    html_parts = ["<table border='1'>"]
    skip_map = set()

    for r in range(row_cnt):
        html_parts.append("<tr>")
        for c in range(col_cnt):
            if (r, c) in skip_map:
                continue

            if (r, c) in grid:
                cell = grid[(r, c)]
                rowspan = cell['rowspan']
                colspan = cell['colspan']
                text = cell['text']

                attr = ""
                if rowspan > 1:
                    attr += f" rowspan='{rowspan}'"
                if colspan > 1:
                    attr += f" colspan='{colspan}'"

                html_parts.append(f"<td{attr}>{text}</td>")

                for rs in range(rowspan):
                    for cs in range(colspan):
                        if rs == 0 and cs == 0:
                            continue
                        skip_map.add((r + rs, c + cs))
            else:
                html_parts.append("<td></td>")
        html_parts.append("</tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


__all__ = [
    'parse_table',
    'build_table_grid',
    'render_table_html',
]
