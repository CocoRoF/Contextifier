# service/document_processor/processor/doc_helpers/rtf_table_extractor.py
"""
RTF 테이블 추출기

RTF 문서에서 테이블을 추출하고 파싱하는 기능을 제공합니다.
"""
import logging
import re
from typing import List, Optional, Tuple

from libs.core.processor.doc_helpers.rtf_models import (
    RTFCellInfo,
    RTFTable,
)
from libs.core.processor.doc_helpers.rtf_decoder import (
    decode_hex_escapes,
)
from libs.core.processor.doc_helpers.rtf_text_cleaner import (
    clean_rtf_text,
)
from libs.core.processor.doc_helpers.rtf_region_finder import (
    find_excluded_regions,
    is_in_excluded_region,
)

logger = logging.getLogger("document-processor")


def extract_tables_with_positions(
    content: str,
    encoding: str = "cp949"
) -> Tuple[List[RTFTable], List[Tuple[int, int, RTFTable]]]:
    r"""
    RTF에서 테이블을 추출합니다 (위치 정보 포함).

    RTF 테이블 구조:
    - \trowd: 테이블 행 시작 (row definition)
    - \cellx<N>: 셀 경계 위치 정의
    - \clmgf: 수평 병합 시작
    - \clmrg: 수평 병합 계속
    - \clvmgf: 수직 병합 시작
    - \clvmrg: 수직 병합 계속
    - \intbl: 셀 내 단락
    - \cell: 셀 끝
    - \row: 행 끝

    Args:
        content: RTF 문자열 콘텐츠
        encoding: 사용할 인코딩

    Returns:
        (테이블 리스트, 테이블 영역 리스트) 튜플
    """
    tables = []
    table_regions = []

    # 제외 영역 찾기 (header, footer, footnote 등)
    excluded_regions = find_excluded_regions(content)

    # 1단계: \row로 끝나는 모든 위치 찾기
    row_positions = []
    for match in re.finditer(r'\\row(?![a-z])', content):
        row_positions.append(match.end())

    if not row_positions:
        return tables, table_regions

    # 2단계: 각 \row 전에 있는 \trowd 찾기 (해당 행의 시작)
    all_rows = []
    for i, row_end in enumerate(row_positions):
        # 이전 \row 위치 또는 시작점
        if i == 0:
            search_start = 0
        else:
            search_start = row_positions[i - 1]

        # 이 영역에서 첫 번째 \trowd 찾기
        segment = content[search_start:row_end]
        trowd_match = re.search(r'\\trowd', segment)

        if trowd_match:
            row_start = search_start + trowd_match.start()

            # 제외 영역(header/footer/footnote) 안에 있는 행은 무시
            if is_in_excluded_region(row_start, excluded_regions):
                logger.debug(f"Skipping table row at {row_start} (in header/footer/footnote)")
                continue

            row_text = content[row_start:row_end]
            all_rows.append((row_start, row_end, row_text))

    if not all_rows:
        return tables, table_regions

    # 연속된 행들을 테이블로 그룹화
    table_groups = []  # [(start_pos, end_pos, [row_texts])]
    current_table = []
    current_start = -1
    current_end = -1
    prev_end = -1

    for row_start, row_end, row_text in all_rows:
        # 이전 행과 150자 이내면 같은 테이블
        if prev_end == -1 or row_start - prev_end < 150:
            if current_start == -1:
                current_start = row_start
            current_table.append(row_text)
            current_end = row_end
        else:
            if current_table:
                table_groups.append((current_start, current_end, current_table))
            current_table = [row_text]
            current_start = row_start
            current_end = row_end
        prev_end = row_end

    if current_table:
        table_groups.append((current_start, current_end, current_table))

    logger.info(f"Found {len(table_groups)} table groups")

    # 각 테이블 그룹 파싱
    for start_pos, end_pos, table_rows in table_groups:
        table = _parse_table_with_merge(table_rows, encoding)
        if table and table.rows:
            table.position = start_pos
            table.end_position = end_pos
            tables.append(table)
            table_regions.append((start_pos, end_pos, table))

    logger.info(f"Extracted {len(tables)} tables")
    return tables, table_regions


def _parse_table_with_merge(rows: List[str], encoding: str = "cp949") -> Optional[RTFTable]:
    """
    테이블 행들을 파싱하여 RTFTable 객체로 변환 (병합 셀 지원)

    Args:
        rows: 테이블 행 텍스트 리스트
        encoding: 사용할 인코딩

    Returns:
        RTFTable 객체
    """
    table = RTFTable()

    for row_text in rows:
        cells = _extract_cells_with_merge(row_text, encoding)
        if cells:
            table.rows.append(cells)
            if len(cells) > table.col_count:
                table.col_count = len(cells)

    return table if table.rows else None


def _extract_cells_with_merge(row_text: str, encoding: str = "cp949") -> List[RTFCellInfo]:
    """
    테이블 행에서 셀 내용과 병합 정보를 추출합니다.

    Args:
        row_text: 테이블 행 RTF 텍스트
        encoding: 사용할 인코딩

    Returns:
        RTFCellInfo 리스트
    """
    cells = []

    # 1단계: 셀 정의 파싱 (cellx 전까지의 속성들)
    cell_defs = []

    # \cell 다음에 x가 오지 않는 첫 번째 \cell 찾기
    first_cell_idx = -1
    pos = 0
    while True:
        idx = row_text.find('\\cell', pos)
        if idx == -1:
            first_cell_idx = len(row_text)
            break
        # \cell 다음이 x인지 확인 (\cellx는 건너뜀)
        if idx + 5 < len(row_text) and row_text[idx + 5] == 'x':
            pos = idx + 1
            continue
        first_cell_idx = idx
        break

    def_part = row_text[:first_cell_idx]

    current_def = {
        'h_merge_first': False,
        'h_merge_cont': False,
        'v_merge_first': False,
        'v_merge_cont': False,
        'right_boundary': 0
    }

    cell_def_pattern = r'\\cl(?:mgf|mrg|vmgf|vmrg)|\\cellx(-?\d+)'

    for match in re.finditer(cell_def_pattern, def_part):
        token = match.group()
        if token == '\\clmgf':
            current_def['h_merge_first'] = True
        elif token == '\\clmrg':
            current_def['h_merge_cont'] = True
        elif token == '\\clvmgf':
            current_def['v_merge_first'] = True
        elif token == '\\clvmrg':
            current_def['v_merge_cont'] = True
        elif token.startswith('\\cellx'):
            if match.group(1):
                current_def['right_boundary'] = int(match.group(1))
            cell_defs.append(current_def.copy())
            # 다음 셀을 위해 초기화
            current_def = {
                'h_merge_first': False,
                'h_merge_cont': False,
                'v_merge_first': False,
                'v_merge_cont': False,
                'right_boundary': 0
            }

    # 2단계: 셀 내용 추출
    cell_texts = _extract_cell_texts(row_text, encoding)

    # 3단계: 셀 정의와 내용 매칭
    for i, cell_text in enumerate(cell_texts):
        if i < len(cell_defs):
            cell_def = cell_defs[i]
        else:
            cell_def = {
                'h_merge_first': False,
                'h_merge_cont': False,
                'v_merge_first': False,
                'v_merge_cont': False,
                'right_boundary': 0
            }

        cells.append(RTFCellInfo(
            text=cell_text,
            h_merge_first=cell_def['h_merge_first'],
            h_merge_cont=cell_def['h_merge_cont'],
            v_merge_first=cell_def['v_merge_first'],
            v_merge_cont=cell_def['v_merge_cont'],
            right_boundary=cell_def['right_boundary']
        ))

    return cells


def _extract_cell_texts(row_text: str, encoding: str = "cp949") -> List[str]:
    r"""
    행에서 셀 텍스트만 추출합니다.

    Args:
        row_text: 테이블 행 RTF 텍스트
        encoding: 사용할 인코딩

    Returns:
        셀 텍스트 리스트
    """
    cell_texts = []

    # 1단계: 모든 \cell 위치 찾기 (cellx가 아닌 순수 \cell만)
    cell_positions = []
    pos = 0
    while True:
        idx = row_text.find('\\cell', pos)
        if idx == -1:
            break
        # \cell 다음이 x인지 확인
        next_pos = idx + 5
        if next_pos < len(row_text) and row_text[next_pos] == 'x':
            pos = idx + 1
            continue
        cell_positions.append(idx)
        pos = idx + 1

    if not cell_positions:
        return cell_texts

    # 2단계: 첫 번째 \cell 위치 이전에서 마지막 \cellx 찾기
    first_cell_pos = cell_positions[0]
    def_part = row_text[:first_cell_pos]

    last_cellx_end = 0
    for match in re.finditer(r'\\cellx-?\d+', def_part):
        last_cellx_end = match.end()

    if last_cellx_end == 0:
        last_cellx_end = 0

    # 3단계: 각 셀 내용 추출
    prev_end = last_cellx_end
    for cell_end in cell_positions:
        cell_content = row_text[prev_end:cell_end]

        # RTF 디코딩 및 클리닝
        decoded = decode_hex_escapes(cell_content, encoding)
        clean = clean_rtf_text(decoded, encoding)
        cell_texts.append(clean)

        # 다음 셀은 \cell 다음부터
        prev_end = cell_end + 5  # len('\\cell') = 5

    return cell_texts
