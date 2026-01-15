# your_package/document_processor/pdf_handler_v2.py
"""
PDF Handler V2 - 고도화된 PDF 문서 처리기

주요 개선 사항:
1. PyMuPDF의 find_tables() 기반 1차 추출 (가장 정확)
2. pdfplumber 보조 추출 (특수 레이아웃)
3. 분절된 drawings 기반 선 분석 (폴백)
4. 페이지 테두리(장식용) vs 실제 테이블 구분
5. 물리적 셀 bbox 기반 rowspan/colspan 계산
6. 페이지 간 테이블 연속성 처리
7. 열린 테두리(open border) 테이블 복구

V2 전략:
- 1단계: PyMuPDF find_tables() - 정확도 높고 복잡한 구조 지원
- 2단계: 물리적 셀 기반 rowspan/colspan 정밀 계산
- 3단계: 페이지 테두리 감지 및 필터링
- 4단계: 테이블 연속성 처리 (페이지 간)
"""
import logging
import os
import io
import copy
import tempfile
import hashlib
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from PIL import Image

from service.storage.minio_client import (
    get_minio_client,
    upload_file,
    ensure_bucket_exists,
    DEFAULT_BUCKET_NAME
)
from libs.core.processor.pdf_helpers.pdf_helper import (
    upload_image_to_minio,
    extract_pdf_metadata,
    format_metadata,
    escape_html,
    calculate_overlap_ratio,
    is_inside_any_bbox,
    find_image_position,
    get_text_lines_with_positions
)

logger = logging.getLogger("document-processor")

# PyMuPDF import
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False
    logger.error("PyMuPDF is required for PDF processing but not available")

# pdfplumber import
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")


# ============================================================================
# 타입 정의
# ============================================================================

class ElementType(Enum):
    """페이지 요소 타입"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class PageElement:
    """페이지 내 요소를 나타내는 데이터 클래스"""
    element_type: ElementType
    content: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int

    @property
    def sort_key(self) -> Tuple[float, float]:
        """정렬 키: (y0, x0)"""
        return (self.bbox[1], self.bbox[0])


@dataclass
class TableInfo:
    """테이블 정보"""
    page_num: int
    table_idx: int
    bbox: Tuple[float, float, float, float]
    data: List[List[Optional[str]]]
    col_count: int
    row_count: int
    page_height: float
    cells_info: Optional[List[Dict]] = None  # 물리적 셀 정보


@dataclass
class PageBorderInfo:
    """페이지 테두리 정보"""
    has_border: bool = False
    border_bbox: Optional[Tuple[float, float, float, float]] = None


# ============================================================================
# 메인 함수
# ============================================================================

async def extract_text_from_pdf_v2(
    file_path: str,
    current_config: Dict[str, Any] = None,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    PDF 텍스트 추출 V2 (고도화된 테이블 처리).

    Args:
        file_path: PDF 파일 경로
        current_config: 설정 딕셔너리
        app_db: 데이터베이스 연결 (이미지 메타데이터 저장용)
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (인라인 이미지 태그, 테이블 HTML 포함)
    """
    if current_config is None:
        current_config = {}

    logger.info(f"[PDF-V2] Processing: {file_path}")
    return await _extract_pdf_enhanced_v2(file_path, current_config, app_db, extract_default_metadata)


# ============================================================================
# 핵심 처리 로직
# ============================================================================

async def _extract_pdf_enhanced_v2(
    file_path: str,
    current_config: Dict[str, Any],
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    고도화된 PDF 처리 V2.
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF is required for PDF processing")

    try:
        doc = fitz.open(file_path)
        all_pages_text = []
        processed_images: Set[int] = set()

        # 메타데이터 추출 (extract_default_metadata가 True인 경우에만)
        if extract_default_metadata:
            metadata = extract_pdf_metadata(doc)
            metadata_text = format_metadata(metadata)
            if metadata_text:
                all_pages_text.append(metadata_text)

        # 전체 문서 테이블 추출 (페이지 간 연속성 처리 포함)
        all_tables = _extract_all_tables_v2(doc, file_path)

        # 페이지별 처리
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_elements: List[PageElement] = []

            logger.debug(f"[PDF-V2] Processing page {page_num + 1}")

            # 1. 페이지 테두리 분석
            border_info = _detect_page_border(page)

            # 2. 해당 페이지의 테이블 가져오기
            page_tables = all_tables.get(page_num, [])

            for table_element in page_tables:
                page_elements.append(table_element)

            # 3. 테이블 영역 계산 (텍스트 필터링용)
            table_bboxes = [elem.bbox for elem in page_tables]

            # 4. 텍스트 추출 (테이블 영역 제외)
            text_elements = _extract_text_blocks_v2(page, page_num, table_bboxes, border_info)
            page_elements.extend(text_elements)

            # 5. 이미지 추출
            image_elements = await _extract_images_from_page_v2(
                page, page_num, doc, app_db, processed_images, table_bboxes
            )
            page_elements.extend(image_elements)

            # 6. 요소 정렬 및 병합
            page_text = _merge_page_elements_v2(page_elements)
            if page_text.strip():
                all_pages_text.append(f"<Page {page_num + 1}>\n{page_text}\n</Page {page_num + 1}>")

        doc.close()

        final_text = "\n\n".join(all_pages_text)
        logger.info(f"[PDF-V2] Extracted {len(final_text)} chars from {file_path}")

        return final_text

    except Exception as e:
        logger.error(f"[PDF-V2] Error processing {file_path}: {e}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# 테이블 추출 V2
# ============================================================================

def _extract_all_tables_v2(doc, file_path: str) -> Dict[int, List[PageElement]]:
    """
    문서 전체에서 테이블을 추출합니다.

    전략:
    1. PyMuPDF find_tables() 사용 (가장 정확)
    2. 물리적 셀 정보 기반 rowspan/colspan 계산
    3. 페이지 간 테이블 연속성 처리
    """
    tables_by_page: Dict[int, List[PageElement]] = {}
    all_table_infos: List[TableInfo] = []

    # 1단계: 모든 페이지에서 테이블 수집
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height

        # 페이지 테두리 감지
        border_info = _detect_page_border(page)

        try:
            # PyMuPDF 테이블 탐지
            if not hasattr(page, 'find_tables'):
                continue

            tabs = page.find_tables()

            for table_idx, table in enumerate(tabs.tables):
                try:
                    table_data = table.extract()

                    if not table_data or not any(any(cell for cell in row if cell) for row in table_data):
                        continue

                    # 페이지 테두리와 겹치는지 확인 (장식용 테두리 필터링)
                    if border_info.has_border and _is_table_likely_border(table.bbox, border_info, page):
                        logger.debug(f"[PDF-V2] Skipping table that looks like page border: {table.bbox}")
                        continue

                    # 셀 정보 추출
                    cells_info = _extract_cells_info_pymupdf(table)

                    table_info = TableInfo(
                        page_num=page_num,
                        table_idx=table_idx,
                        bbox=table.bbox,
                        data=table_data,
                        col_count=table.col_count,
                        row_count=table.row_count,
                        page_height=page_height,
                        cells_info=cells_info
                    )

                    all_table_infos.append(table_info)

                except Exception as e:
                    logger.debug(f"[PDF-V2] Error extracting table {table_idx} from page {page_num}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"[PDF-V2] Error finding tables on page {page_num}: {e}")
            continue

    # 2단계: 인접한 테이블 병합 (헤더+데이터 분리 문제 해결)
    merged_tables = _merge_adjacent_tables(all_table_infos)

    # 2.5단계: 병합된 테이블에 누락된 행 찾아서 삽입
    merged_tables = _find_and_insert_missing_rows(doc, merged_tables)

    # 3단계: 테이블 연속성 처리
    processed_tables = _process_table_continuity_v2(merged_tables)

    # 4단계: HTML 변환 및 PageElement 생성
    for table_info in processed_tables:
        try:
            # HTML 테이블 생성
            html_table = _convert_table_to_html_v2(table_info)

            if html_table:
                page_num = table_info.page_num

                if page_num not in tables_by_page:
                    tables_by_page[page_num] = []

                tables_by_page[page_num].append(PageElement(
                    element_type=ElementType.TABLE,
                    content=html_table,
                    bbox=table_info.bbox,
                    page_num=page_num
                ))

        except Exception as e:
            logger.debug(f"[PDF-V2] Error converting table to HTML: {e}")
            continue

    logger.info(f"[PDF-V2] Extracted tables from {len(tables_by_page)} pages")
    return tables_by_page


def _extract_cells_info_pymupdf(table) -> List[Dict]:
    """
    PyMuPDF 테이블에서 셀 정보를 추출합니다.

    각 셀의 물리적 bbox를 사용하여 rowspan/colspan을 정확하게 계산합니다.
    """
    cells_info = []

    if not hasattr(table, 'cells') or not table.cells:
        return cells_info

    # 그리드 경계 추출
    h_lines = set()
    v_lines = set()

    for cell in table.cells:
        if cell is None:
            continue
        h_lines.add(round(cell[1], 1))  # y0
        h_lines.add(round(cell[3], 1))  # y1
        v_lines.add(round(cell[0], 1))  # x0
        v_lines.add(round(cell[2], 1))  # x1

    h_lines = sorted(h_lines)
    v_lines = sorted(v_lines)

    # 셀을 그리드 위치로 매핑
    for cell in table.cells:
        if cell is None:
            continue

        x0, y0, x1, y1 = cell

        # 그리드 인덱스 찾기
        row_start = _find_grid_index(round(y0, 1), h_lines)
        row_end = _find_grid_index(round(y1, 1), h_lines)
        col_start = _find_grid_index(round(x0, 1), v_lines)
        col_end = _find_grid_index(round(x1, 1), v_lines)

        if row_start is not None and col_start is not None:
            rowspan = max(1, (row_end or row_start + 1) - row_start)
            colspan = max(1, (col_end or col_start + 1) - col_start)

            cells_info.append({
                'row': row_start,
                'col': col_start,
                'rowspan': rowspan,
                'colspan': colspan,
                'bbox': (x0, y0, x1, y1)
            })

    return cells_info


def _find_grid_index(value: float, grid_lines: List[float], tolerance: float = 3.0) -> Optional[int]:
    """그리드 라인에서 값의 인덱스를 찾습니다."""
    for i, line in enumerate(grid_lines):
        if abs(value - line) <= tolerance:
            return i
    return None


def _find_and_insert_missing_rows(doc, tables: List[TableInfo]) -> List[TableInfo]:
    """
    테이블 내부 갭 또는 직후에 누락된 행을 찾아서 삽입합니다.

    PyMuPDF find_tables()가 놓치는 케이스:
    1. 헤더와 데이터 사이의 서브헤더 행 (예: (A), (B))
    2. 테이블 마지막의 주석 행 (예: 주) ...)

    이런 행들은 불완전한 테두리를 가지거나 특수한 구조로 되어 있어서
    테이블로 인식되지 않습니다.
    """
    if not tables:
        return tables

    result = []

    # 페이지별로 처리
    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)
    for table in tables:
        tables_by_page[table.page_num].append(table)

    for page_num, page_tables in tables_by_page.items():
        page = doc[page_num]
        page_height = page.rect.height

        # Y 좌표로 정렬
        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])

        # 페이지의 모든 텍스트 라인 정보 수집
        text_lines = get_text_lines_with_positions(page)

        for i, table in enumerate(sorted_tables):
            table_top = table.bbox[1]
            table_bottom = table.bbox[3]
            table_left = table.bbox[0]
            table_right = table.bbox[2]

            # 1. 테이블 직후의 주석 행 탐지 (예: "주) ...")
            next_table_top = sorted_tables[i + 1].bbox[1] if i + 1 < len(sorted_tables) else page_height

            # 테이블 끝 ~ 다음 테이블 시작 사이의 텍스트 찾기
            footer_lines = [
                line for line in text_lines
                if table_bottom - 3 <= line['y0'] <= table_bottom + 25
                and line['x0'] >= table_left - 10
                and line['x1'] <= table_right + 10
                and line['y0'] < next_table_top - 20
            ]

            if footer_lines:
                # "주)"로 시작하는 텍스트가 있으면 주석 행
                for line in footer_lines:
                    if line['text'].startswith('주)') or line['text'].startswith('주 )'):
                        table = _add_missing_row_to_table(table, footer_lines, 'footer')
                        logger.debug(f"[PDF-V2] Added missing footer row to table on page {page_num + 1}")
                        break

            # 2. 테이블 내부의 서브헤더 행 탐지 (예: (A), (B))
            # 테이블 데이터에서 첫 번째 행이 헤더인지 확인
            if table.row_count >= 2 and table.data:
                header_row = table.data[0]
                first_data_row = table.data[1]

                # 헤더 행의 Y 범위 추정 (테이블 상단 ~ 첫 데이터 행 시작)
                # 테이블 상단과 첫 데이터 사이에 텍스트가 있는지 확인
                row_height_estimate = (table_bottom - table_top) / table.row_count
                header_bottom_estimate = table_top + row_height_estimate

                # 헤더와 첫 데이터 사이의 텍스트 찾기
                # 더 엄격한 범위: 헤더 하단 ~ 두 번째 행 상단
                second_row_top_estimate = table_top + row_height_estimate * 2

                subheader_candidates = [
                    line for line in text_lines
                    if header_bottom_estimate - 5 <= line['y0'] <= second_row_top_estimate - 5
                    and line['x0'] >= table_left - 10
                    and line['x1'] <= table_right + 10
                ]

                # (A), (B) 패턴만 필터링 - 데이터 값은 제외
                subheader_filtered = [
                    line for line in subheader_candidates
                    if '(A)' in line['text'] or '(B)' in line['text']
                ]

                if subheader_filtered:
                    table = _add_missing_row_to_table(table, subheader_filtered, 'subheader')
                    logger.debug(f"[PDF-V2] Added missing subheader row (A)/(B) to table on page {page_num + 1}")

            result.append(table)

    # 원래 순서 유지
    result.sort(key=lambda t: (t.page_num, t.bbox[1]))

    return result


def _add_missing_row_to_table(table: TableInfo, text_lines: List[Dict], position: str) -> TableInfo:
    """
    누락된 행을 테이블에 추가합니다.

    Args:
        table: 원본 테이블
        text_lines: 추가할 텍스트 라인들
        position: 'subheader' (헤더 아래) 또는 'footer' (데이터 아래)
    """
    if not text_lines:
        return table

    # 텍스트 라인들을 하나의 행으로 조합
    # X 좌표를 기준으로 열 분리 시도
    text_lines_sorted = sorted(text_lines, key=lambda l: l['x0'])

    # 테이블 열 경계 추정
    table_width = table.bbox[2] - table.bbox[0]
    col_width = table_width / table.col_count if table.col_count > 0 else table_width

    # 각 라인의 X 좌표를 기준으로 열 할당
    new_row = [''] * table.col_count

    for line in text_lines_sorted:
        # 가장 가까운 열 찾기
        relative_x = line['x0'] - table.bbox[0]
        col_idx = min(int(relative_x / col_width), table.col_count - 1)
        col_idx = max(0, col_idx)

        if new_row[col_idx]:
            new_row[col_idx] += " " + line['text']
        else:
            new_row[col_idx] = line['text']

    # 만약 모든 텍스트가 첫 번째 열에만 있으면 전체 colspan으로 처리
    non_empty_cols = sum(1 for c in new_row if c)
    if non_empty_cols == 1 and new_row[0]:
        # 전체 텍스트를 합쳐서 첫 번째 열에만 넣고 colspan 처리
        combined_text = " ".join(line['text'] for line in text_lines_sorted)
        new_row = [combined_text] + [''] * (table.col_count - 1)

    # 새 데이터 생성
    new_data = list(table.data)

    if position == 'subheader':
        # 첫 번째 행 다음에 삽입
        if len(new_data) > 0:
            new_data.insert(1, new_row)
        else:
            new_data.append(new_row)
    else:  # footer
        new_data.append(new_row)

    # 새 bbox 계산
    all_y = [line['y0'] for line in text_lines] + [line['y1'] for line in text_lines]
    min_y = min(all_y)
    max_y = max(all_y)

    new_bbox = (
        table.bbox[0],
        min(table.bbox[1], min_y),
        table.bbox[2],
        max(table.bbox[3], max_y)
    )

    return TableInfo(
        page_num=table.page_num,
        table_idx=table.table_idx,
        bbox=new_bbox,
        data=new_data,
        col_count=table.col_count,
        row_count=len(new_data),
        page_height=table.page_height,
        cells_info=None  # 셀 정보는 재계산 필요
    )


def _merge_adjacent_tables(tables: List[TableInfo]) -> List[TableInfo]:
    """
    인접한 테이블을 병합합니다.

    헤더와 데이터 테이블이 분리된 경우 하나로 합칩니다.

    병합 조건:
    1. 같은 페이지에 있음
    2. Y 방향으로 인접 (간격 < 30px)
    3. X 범위가 비슷함 (같은 테이블의 일부)
    4. 열 수가 비슷하거나 헤더(1행)가 데이터 테이블 위에 있음
    """
    if not tables:
        return tables

    # 페이지별로 그룹화
    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)
    for table in tables:
        tables_by_page[table.page_num].append(table)

    merged_result = []

    for page_num, page_tables in tables_by_page.items():
        # Y 좌표로 정렬
        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])

        i = 0
        while i < len(sorted_tables):
            current = sorted_tables[i]

            # 다음 테이블과 병합 가능한지 확인
            merged = current
            while i + 1 < len(sorted_tables):
                next_table = sorted_tables[i + 1]

                if _should_merge_tables(merged, next_table):
                    merged = _do_merge_tables(merged, next_table)
                    i += 1
                    logger.debug(f"[PDF-V2] Merged adjacent tables on page {page_num + 1}")
                else:
                    break

            merged_result.append(merged)
            i += 1

    # 원래 순서 유지 (페이지 번호, Y 좌표)
    merged_result.sort(key=lambda t: (t.page_num, t.bbox[1]))

    return merged_result


def _should_merge_tables(t1: TableInfo, t2: TableInfo) -> bool:
    """
    두 테이블을 병합해야 하는지 판단합니다.
    """
    # 같은 페이지여야 함
    if t1.page_num != t2.page_num:
        return False

    # Y 방향 간격 확인 (t2가 t1 아래에 있어야 함)
    y_gap = t2.bbox[1] - t1.bbox[3]
    if y_gap < 0 or y_gap > 30:  # 겹치거나 너무 멀면 안됨
        return False

    # X 범위가 비슷해야 함 (80% 이상 겹침)
    x_overlap_start = max(t1.bbox[0], t2.bbox[0])
    x_overlap_end = min(t1.bbox[2], t2.bbox[2])
    x_overlap = max(0, x_overlap_end - x_overlap_start)

    t1_width = t1.bbox[2] - t1.bbox[0]
    t2_width = t2.bbox[2] - t2.bbox[0]

    overlap_ratio = x_overlap / max(t1_width, t2_width, 1)
    if overlap_ratio < 0.8:
        return False

    # 열 수 조건:
    # - 같은 열 수
    # - 또는 t1이 1행짜리 헤더이고 t2가 데이터
    # - 또는 t1의 열 수가 t2의 열 수보다 작음 (헤더에 colspan이 있는 경우)
    if t1.col_count == t2.col_count:
        return True
    if t1.row_count == 1 and t1.col_count < t2.col_count:
        # 헤더에 colspan이 있는 경우 (예: "시험결과"가 2열을 커버)
        return True

    return False


def _do_merge_tables(t1: TableInfo, t2: TableInfo) -> TableInfo:
    """
    두 테이블을 실제로 병합합니다.

    헤더 테이블(t1)의 열 수가 데이터 테이블(t2)보다 적으면
    적절한 colspan 정보를 생성하고 데이터를 재배열합니다.
    """
    # 병합된 bbox
    merged_bbox = (
        min(t1.bbox[0], t2.bbox[0]),
        t1.bbox[1],  # t1이 위에 있음
        max(t1.bbox[2], t2.bbox[2]),
        t2.bbox[3]   # t2가 아래에 있음
    )

    # 열 수는 더 큰 쪽 사용
    merged_col_count = max(t1.col_count, t2.col_count)
    header_col_count = t1.col_count

    # 데이터 병합
    merged_data = []
    merged_cells = []

    # 헤더 열 수 < 데이터 열 수인 경우, colspan 분배 및 데이터 재배열
    if header_col_count < merged_col_count and t1.row_count == 1 and t1.data:
        # 예: 헤더 ['구분', '시험결과', '기준'] (3열), 데이터 4열
        # 결과: ['구분', '시험결과', '', '기준']
        #       colspan: 1, 2, skip, 1

        extra_cols = merged_col_count - header_col_count
        header_row = list(t1.data[0])

        # 새 헤더 행 생성 (colspan에 맞춰 빈 셀 삽입)
        # 두 번째 셀에 colspan 적용, 나머지는 마지막으로
        new_header = []
        col_position = 0

        for orig_col_idx, value in enumerate(header_row):
            new_header.append(value)

            if orig_col_idx == 1 and extra_cols > 0:
                # 두 번째 셀 뒤에 빈 셀 추가 (colspan으로 커버됨)
                colspan = 1 + extra_cols
                merged_cells.append({
                    'row': 0,
                    'col': col_position,
                    'rowspan': 1,
                    'colspan': colspan,
                    'bbox': None
                })
                # 빈 셀 추가 (skip 대상)
                for _ in range(extra_cols):
                    new_header.append('')
                col_position += colspan
            else:
                merged_cells.append({
                    'row': 0,
                    'col': col_position,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })
                col_position += 1

        merged_data.append(new_header)
    else:
        # t1 데이터 (헤더) - 열 수 맞추기
        for row in t1.data:
            if len(row) < merged_col_count:
                adjusted_row = list(row) + [''] * (merged_col_count - len(row))
            else:
                adjusted_row = list(row)
            merged_data.append(adjusted_row)

        if t1.cells_info:
            merged_cells.extend(t1.cells_info)

    # t2 데이터 (본문)
    for row in t2.data:
        if len(row) < merged_col_count:
            adjusted_row = list(row) + [''] * (merged_col_count - len(row))
        else:
            adjusted_row = list(row)
        merged_data.append(adjusted_row)

    # t2의 셀 정보 (행 인덱스 조정)
    if t2.cells_info:
        row_offset = t1.row_count
        for cell in t2.cells_info:
            adjusted_cell = dict(cell)
            adjusted_cell['row'] = cell['row'] + row_offset
            merged_cells.append(adjusted_cell)

    return TableInfo(
        page_num=t1.page_num,
        table_idx=t1.table_idx,
        bbox=merged_bbox,
        data=merged_data,
        col_count=merged_col_count,
        row_count=t1.row_count + t2.row_count,
        page_height=t1.page_height,
        cells_info=merged_cells if merged_cells else None
    )


def _is_table_likely_border(
    table_bbox: Tuple[float, float, float, float],
    border_info: PageBorderInfo,
    page
) -> bool:
    """
    테이블이 실제로 페이지 테두리(장식용)인지 확인합니다.
    """
    if not border_info.has_border or not border_info.border_bbox:
        return False

    page_width = page.rect.width
    page_height = page.rect.height

    # 테이블이 페이지의 85% 이상을 차지하면 테두리일 가능성 높음
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]

    if table_width > page_width * 0.85 and table_height > page_height * 0.85:
        return True

    return False


# ============================================================================
# 페이지 테두리 감지
# ============================================================================

def _detect_page_border(page) -> PageBorderInfo:
    """
    페이지 테두리(장식용)를 감지합니다.

    페이지를 감싸는 선이 있는지 분석합니다.
    """
    result = PageBorderInfo()

    drawings = page.get_drawings()
    if not drawings:
        return result

    page_width = page.rect.width
    page_height = page.rect.height

    edge_margin = min(page_width, page_height) * 0.1
    page_spanning_ratio = 0.85

    border_lines = {
        'top': False,
        'bottom': False,
        'left': False,
        'right': False
    }

    for drawing in drawings:
        rect = drawing['rect']
        w = rect.width
        h = rect.height

        # 가로선 (얇고 넓은)
        if h <= 5 and w > page_width * page_spanning_ratio:
            if rect.y0 < edge_margin:
                border_lines['top'] = True
            elif rect.y1 > page_height - edge_margin:
                border_lines['bottom'] = True

        # 세로선 (좁고 높은)
        if w <= 5 and h > page_height * page_spanning_ratio:
            if rect.x0 < edge_margin:
                border_lines['left'] = True
            elif rect.x1 > page_width - edge_margin:
                border_lines['right'] = True

    # 4면 모두 있으면 페이지 테두리
    if all(border_lines.values()):
        result.has_border = True
        result.border_bbox = (edge_margin, edge_margin, page_width - edge_margin, page_height - edge_margin)

    return result


# ============================================================================
# 테이블 연속성 처리
# ============================================================================

def _process_table_continuity_v2(all_tables: List[TableInfo]) -> List[TableInfo]:
    """
    페이지 간 테이블 연속성을 처리합니다.

    연속되는 테이블의 빈 셀을 이전 페이지의 값으로 채웁니다.
    """
    if not all_tables:
        return all_tables

    result = []
    last_category = None

    for i, table_info in enumerate(all_tables):
        # 깊은 복사
        table_info = TableInfo(
            page_num=table_info.page_num,
            table_idx=table_info.table_idx,
            bbox=table_info.bbox,
            data=copy.deepcopy(table_info.data),
            col_count=table_info.col_count,
            row_count=table_info.row_count,
            page_height=table_info.page_height,
            cells_info=table_info.cells_info
        )

        curr_data = table_info.data

        if i == 0:
            # 첫 번째 테이블: 마지막 카테고리 추출
            last_category = _extract_last_category(curr_data)
            result.append(table_info)
            continue

        prev_table = all_tables[i - 1]

        # 페이지가 다르고 테이블 위치가 연속 조건을 만족하는지 확인
        is_continuation = (
            table_info.page_num > prev_table.page_num and
            prev_table.bbox[3] > prev_table.page_height * 0.7 and  # 이전 테이블이 하단
            table_info.bbox[1] < table_info.page_height * 0.3 and  # 현재 테이블이 상단
            table_info.col_count == prev_table.col_count  # 열 수 동일
        )

        if is_continuation and last_category:
            # 빈 첫 번째 열 채우기
            for row in curr_data:
                if len(row) >= 2:
                    first_col = row[0]
                    second_col = row[1] if len(row) > 1 else ""

                    if (not first_col or not str(first_col).strip()) and second_col and str(second_col).strip():
                        row[0] = last_category
                    elif first_col and str(first_col).strip():
                        last_category = first_col
        else:
            # 현재 테이블에서 마지막 카테고리 업데이트
            new_last = _extract_last_category(curr_data)
            if new_last:
                last_category = new_last

        result.append(table_info)

    return result


def _extract_last_category(table_data: List[List[Optional[str]]]) -> Optional[str]:
    """
    테이블에서 마지막 유효한 카테고리(첫 번째 열 값)를 추출합니다.
    """
    if not table_data:
        return None

    last_category = None

    for row in table_data:
        if len(row) >= 1 and row[0] and str(row[0]).strip():
            last_category = str(row[0]).strip()

    return last_category


# ============================================================================
# HTML 테이블 변환
# ============================================================================

def _convert_table_to_html_v2(table_info: TableInfo) -> str:
    """
    테이블을 HTML로 변환합니다.

    물리적 셀 정보가 있으면 정확한 rowspan/colspan을 적용합니다.
    헤더 행에서 빈 셀이 있으면 이전 셀의 colspan으로 처리합니다.
    """
    data = table_info.data
    cells_info = table_info.cells_info

    if not data:
        return ""

    num_rows = len(data)
    num_cols = max(len(row) for row in data) if data else 0

    if num_cols == 0:
        return ""

    # 헤더 colspan 처리 (첫 번째 행에서 빈 셀 → 이전 셀 colspan)
    data = _apply_header_colspan(data)

    # 셀 정보가 있으면 물리적 병합 처리
    if cells_info:
        return _convert_with_physical_cells(data, cells_info, num_rows, num_cols)

    # 셀 정보가 없으면 값 기반 병합 감지
    return _convert_with_value_based_spans(data, num_rows, num_cols)


def _apply_header_colspan(data: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """
    헤더 행에서 빈 셀을 colspan으로 처리합니다.

    예: ['구분', '시험결과', '기준', ''] → colspan 정보 마킹
    """
    if not data or len(data) < 2:
        return data

    # 첫 번째 행이 헤더인지 확인 (데이터 행보다 열 수가 적거나 빈 셀이 끝에 있음)
    header_row = data[0]
    data_row = data[1] if len(data) > 1 else []

    # 헤더 열 수가 데이터 열 수보다 적으면 colspan 필요
    header_non_empty = sum(1 for c in header_row if c and str(c).strip())
    data_non_empty = sum(1 for c in data_row if c and str(c).strip())

    if header_non_empty < data_non_empty and header_non_empty > 0:
        # 헤더의 빈 셀을 마킹 (나중에 colspan으로 처리)
        # 현재는 데이터를 변경하지 않고 별도 로직에서 처리
        pass

    return data


def _convert_with_physical_cells(
    data: List[List[Optional[str]]],
    cells_info: List[Dict],
    num_rows: int,
    num_cols: int
) -> str:
    """
    물리적 셀 정보를 사용하여 HTML 변환합니다.
    """
    # 셀 위치별 span 정보 맵
    span_map: Dict[Tuple[int, int], Dict] = {}

    for cell in cells_info:
        key = (cell['row'], cell['col'])
        span_map[key] = {
            'rowspan': cell.get('rowspan', 1),
            'colspan': cell.get('colspan', 1)
        }

    # 스킵 맵 (병합으로 인해 건너뛸 셀)
    skip_set: Set[Tuple[int, int]] = set()

    for (row, col), spans in span_map.items():
        rowspan = spans['rowspan']
        colspan = spans['colspan']

        for r in range(row, min(row + rowspan, num_rows)):
            for c in range(col, min(col + colspan, num_cols)):
                if (r, c) != (row, col):
                    skip_set.add((r, c))

    # HTML 생성
    html_parts = ["<table>"]

    for row_idx, row in enumerate(data):
        html_parts.append("  <tr>")

        for col_idx in range(num_cols):
            if (row_idx, col_idx) in skip_set:
                continue

            # 셀 내용
            content = row[col_idx] if col_idx < len(row) else ""
            content = escape_html(str(content).strip() if content else "")

            # span 정보
            spans = span_map.get((row_idx, col_idx), {'rowspan': 1, 'colspan': 1})
            attrs = []

            if spans['rowspan'] > 1:
                attrs.append(f'rowspan="{spans["rowspan"]}"')
            if spans['colspan'] > 1:
                attrs.append(f'colspan="{spans["colspan"]}"')

            attr_str = " " + " ".join(attrs) if attrs else ""
            html_parts.append(f"    <td{attr_str}>{content}</td>")

        html_parts.append("  </tr>")

    html_parts.append("</table>")

    return "\n".join(html_parts)


def _convert_with_value_based_spans(
    data: List[List[Optional[str]]],
    num_rows: int,
    num_cols: int
) -> str:
    """
    값 기반으로 병합 셀을 감지하여 HTML 변환합니다.

    특별 처리:
    1. 헤더 행의 빈 셀 → 이전 셀의 colspan
    2. (A)/(B) 서브헤더 패턴 → 첫 번째/마지막 열 rowspan
    3. "주)" 행 → 전체 colspan
    4. 첫 번째 열 빈 셀 연속 → rowspan
    """
    colspan_map: Dict[Tuple[int, int], int] = {}
    rowspan_map: Dict[Tuple[int, int], int] = {}
    skip_set: Set[Tuple[int, int]] = set()

    # 1. 서브헤더 패턴 감지 (두 번째 행에 (A), (B)가 있는 경우)
    has_subheader = False
    if num_rows >= 2:
        second_row = data[1]
        second_row_text = ' '.join(str(c) for c in second_row if c)
        if '(A)' in second_row_text or '(B)' in second_row_text:
            has_subheader = True

            # 첫 번째 행의 "구분"(첫 번째 열)과 "기 준"(마지막 열)에 rowspan=2
            # 그리고 두 번째 행의 해당 위치는 skip
            first_row = data[0]

            # 첫 번째 열: "구분" rowspan=2
            if first_row and first_row[0] and str(first_row[0]).strip():
                rowspan_map[(0, 0)] = 2
                skip_set.add((1, 0))  # 두 번째 행 첫 번째 열 skip

            # 마지막 열: "기 준" rowspan=2
            if first_row and len(first_row) >= num_cols:
                last_val = first_row[num_cols - 1] if num_cols > 0 else ""
                if last_val and str(last_val).strip():
                    rowspan_map[(0, num_cols - 1)] = 2
                    skip_set.add((1, num_cols - 1))  # 두 번째 행 마지막 열 skip

            # "시 험 결 과" colspan=2 (두 번째 열)
            if num_cols >= 3:
                colspan_map[(0, 1)] = 2
                skip_set.add((0, 2))  # 세 번째 열 skip (colspan에 포함)

    # 2. 주석 행 감지 ("주)"로 시작하는 행)
    for row_idx, row in enumerate(data):
        if not row:
            continue
        first_val = str(row[0]).strip() if row[0] else ""
        if first_val.startswith('주)') or first_val.startswith('주 )'):
            # 전체 colspan
            colspan_map[(row_idx, 0)] = num_cols
            for col_idx in range(1, num_cols):
                skip_set.add((row_idx, col_idx))

    # 3. 일반 헤더 colspan (끝에 있는 빈 셀)
    if not has_subheader and num_rows > 0 and len(data[0]) > 0:
        header_row = data[0]
        col = num_cols - 1

        while col >= 0:
            val = header_row[col] if col < len(header_row) else ""
            val = str(val).strip() if val else ""

            if not val and (0, col) not in skip_set:
                skip_set.add((0, col))
                col -= 1
            else:
                empty_count = 0
                for next_col in range(col + 1, num_cols):
                    if (0, next_col) in skip_set:
                        # colspan 계산에 skip 셀 포함
                        next_val = header_row[next_col] if next_col < len(header_row) else ""
                        if not next_val or not str(next_val).strip():
                            empty_count += 1
                    else:
                        break

                if empty_count > 0 and (0, col) not in colspan_map:
                    colspan_map[(0, col)] = empty_count + 1

                col -= 1

    # 4. 첫 번째 열 rowspan (빈 셀 연속)
    if num_cols >= 1 and not has_subheader:
        row = 0
        while row < num_rows:
            if (row, 0) in skip_set:
                row += 1
                continue

            first_val = data[row][0] if data[row] else ""
            first_val = str(first_val).strip() if first_val else ""

            span = 1
            for next_row in range(row + 1, num_rows):
                if (next_row, 0) in skip_set:
                    break
                next_val = data[next_row][0] if data[next_row] else ""
                next_val = str(next_val).strip() if next_val else ""

                if not next_val:
                    second_val = data[next_row][1] if len(data[next_row]) > 1 else ""
                    if second_val and str(second_val).strip():
                        span += 1
                        skip_set.add((next_row, 0))
                    else:
                        break
                else:
                    break

            if span > 1:
                rowspan_map[(row, 0)] = span

            row += span

    # HTML 생성
    html_parts = ["<table>"]

    for row_idx, row in enumerate(data):
        html_parts.append("  <tr>")

        for col_idx in range(num_cols):
            if (row_idx, col_idx) in skip_set:
                continue

            content = row[col_idx] if col_idx < len(row) else ""
            content = escape_html(str(content).strip() if content else "")

            attrs = []
            rowspan = rowspan_map.get((row_idx, col_idx), 1)
            colspan = colspan_map.get((row_idx, col_idx), 1)

            if rowspan > 1:
                attrs.append(f'rowspan="{rowspan}"')
            if colspan > 1:
                attrs.append(f'colspan="{colspan}"')

            attr_str = " " + " ".join(attrs) if attrs else ""
            html_parts.append(f"    <td{attr_str}>{content}</td>")

        html_parts.append("  </tr>")

    html_parts.append("</table>")

    return "\n".join(html_parts)


# ============================================================================
# 텍스트 추출
# ============================================================================

def _extract_text_blocks_v2(
    page,
    page_num: int,
    table_bboxes: List[Tuple[float, float, float, float]],
    border_info: PageBorderInfo
) -> List[PageElement]:
    """
    테이블 영역을 제외한 텍스트 블록을 추출합니다.
    """
    elements = []
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_bbox = block.get("bbox", (0, 0, 0, 0))

        # 테이블 영역과 겹치는지 확인
        if is_inside_any_bbox(block_bbox, table_bboxes):
            continue

        # 텍스트 추출
        text_parts = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                text_parts.append(line_text.strip())

        if text_parts:
            full_text = "\n".join(text_parts)
            elements.append(PageElement(
                element_type=ElementType.TEXT,
                content=full_text,
                bbox=block_bbox,
                page_num=page_num
            ))

    return elements


# ============================================================================
# 이미지 추출
# ============================================================================

async def _extract_images_from_page_v2(
    page,
    page_num: int,
    doc,
    app_db,
    processed_images: Set[int],
    table_bboxes: List[Tuple[float, float, float, float]],
    min_image_size: int = 50,
    min_image_area: int = 2500
) -> List[PageElement]:
    """
    페이지에서 이미지를 추출하고 MinIO에 업로드합니다.
    """
    elements = []

    try:
        image_list = page.get_images()

        for img_info in image_list:
            xref = img_info[0]

            if xref in processed_images:
                continue

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image.get("image")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < min_image_size or height < min_image_size:
                    continue
                if width * height < min_image_area:
                    continue

                img_bbox = find_image_position(page, xref)
                if img_bbox is None:
                    continue

                # 테이블 영역 내 이미지는 스킵
                if is_inside_any_bbox(img_bbox, table_bboxes, threshold=0.7):
                    continue

                image_url = upload_image_to_minio(image_bytes, app_db)

                if image_url:
                    processed_images.add(xref)

                    image_tag = f'\n[Image:{image_url}]\n'

                    elements.append(PageElement(
                        element_type=ElementType.IMAGE,
                        content=image_tag,
                        bbox=img_bbox,
                        page_num=page_num
                    ))

            except Exception as e:
                logger.debug(f"[PDF-V2] Error extracting image xref={xref}: {e}")
                continue

    except Exception as e:
        logger.warning(f"[PDF-V2] Error extracting images: {e}")

    return elements


# ============================================================================
# 요소 병합
# ============================================================================

def _merge_page_elements_v2(elements: List[PageElement]) -> str:
    """
    페이지 요소들을 위치 기반으로 정렬하여 병합합니다.
    """
    if not elements:
        return ""

    sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))

    text_parts = []

    for element in sorted_elements:
        content = element.content.strip()
        if not content:
            continue

        if element.element_type == ElementType.TABLE:
            text_parts.append(f"\n{content}\n")
        else:
            text_parts.append(content)

    return "\n".join(text_parts)
