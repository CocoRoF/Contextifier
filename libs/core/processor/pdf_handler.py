# your_package/document_processor/pdf_handler.py
"""
PDF Handler - PDF 문서 처리기 (Enhanced 전용)

주요 기능:
- 메타데이터 추출 (제목, 작성자, 주제, 키워드, 작성일, 수정일 등)
- 텍스트 추출 (PyMuPDF)
- 테이블 추출 (pdfplumber 우선, PyMuPDF 폴백)
- 인라인 이미지 추출 및 MinIO 업로드

테이블 추출은 pdfplumber를 우선 사용하고, 실패 시 PyMuPDF로 폴백합니다.
"""
import logging
import os
import io
import tempfile
import hashlib
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from PIL import Image
from libs.core.processor.pdf_helpers.pdf_helper import (
    upload_image_to_minio,
    extract_pdf_metadata,
    format_metadata,
    find_image_position
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
    logger.warning("pdfplumber not available, will use PyMuPDF for table extraction")


# === 페이지 요소 타입 정의 ===

class ElementType(Enum):
    """페이지 요소 타입"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class PageElement:
    """페이지 내 요소를 나타내는 데이터 클래스"""
    element_type: ElementType
    content: str  # 텍스트 또는 이미지 태그 또는 테이블 HTML
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int

    @property
    def sort_key(self) -> Tuple[float, float]:
        """정렬 키: (y0, x0) - 위에서 아래, 왼쪽에서 오른쪽"""
        return (self.bbox[1], self.bbox[0])


@dataclass
class TableContinuationInfo:
    """
    페이지 간 테이블 연속성 처리를 위한 정보 클래스.

    페이지 경계에서 끊어진 테이블의 rowspan 셀을 복원하기 위해 사용됩니다.
    """
    page_num: int
    table_idx: int
    bbox: Tuple[float, float, float, float]
    col_count: int
    x_coords: List[float]  # 컬럼 경계 X 좌표
    data: List[List[Optional[str]]]  # 테이블 데이터
    header_rows: int  # 헤더 행 수
    table_obj: Any  # 원본 PyMuPDF Table 객체
    page_height: float  # 페이지 높이


# === 메인 함수 ===

async def extract_text_from_pdf(
    file_path: str,
    current_config: Dict[str, Any] = None,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    PDF 텍스트 추출 (인라인 이미지 + 테이블 지원).

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

    logger.info(f"PDF processing with images: {file_path}")
    return await _extract_pdf_enhanced(file_path, current_config, app_db, extract_default_metadata=extract_default_metadata)


# === 고도화된 PDF 처리 ===

async def _extract_pdf_enhanced(
    file_path: str,
    current_config: Dict[str, Any],
    app_db=None,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_default_metadata: bool = True
) -> str:
    """
    고도화된 PDF 처리.

    - 인라인 이미지 추출 및 MinIO 업로드
    - 테이블 HTML 형식 보존 (셀 병합 지원)
    - 요소 위치 기반 텍스트 통합
    """
    logger.info(f"Enhanced PDF processing: {file_path}")

    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available, cannot process PDF")
        return f"[PDF 파일 처리 실패: PyMuPDF가 설치되지 않음]"

    try:
        result_parts = []
        processed_images = set()

        # PDF 열기
        doc = fitz.open(file_path)
        total_pages = len(doc)
        logger.info(f"PDF has {total_pages} pages")

        # 메타데이터 추출 및 추가 (extract_default_metadata가 True인 경우에만)
        if extract_default_metadata:
            metadata = extract_pdf_metadata(doc)
            if metadata:
                metadata_str = format_metadata(metadata)
                result_parts.append(metadata_str)

        # 테이블 정보 미리 추출 (PyMuPDF find_tables 사용)
        all_tables_by_page = {}
        if extract_tables:
            all_tables_by_page = _extract_all_tables(file_path)
            logger.info(f"Extracted tables from {len(all_tables_by_page)} pages")

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_height = page.rect.height

            # 페이지 헤더 추가
            result_parts.append(f"\n<페이지 번호> {page_num + 1} </페이지 번호>\n")

            # 페이지의 모든 요소 수집
            elements: List[PageElement] = []

            # 1. 텍스트 블록 추출
            text_elements = _extract_text_blocks(page, page_num, page_height)
            elements.extend(text_elements)

            # 2. 이미지 추출 (옵션)
            if extract_images:
                image_elements = await _extract_images_from_page(
                    page, page_num, page_height, doc, app_db, processed_images
                )
                elements.extend(image_elements)

            # 3. 테이블 추가 (이미 추출된 것)
            if page_num in all_tables_by_page:
                elements.extend(all_tables_by_page[page_num])

            # 4. 요소들을 위치 기준으로 정렬
            elements.sort(key=lambda e: e.sort_key)

            # 5. 중복 제거 및 통합
            merged_content = _merge_page_elements(elements)
            result_parts.append(merged_content)

        doc.close()

        result = "".join(result_parts)
        logger.info(f"Enhanced PDF processing completed: {len(result)} characters extracted")

        return result

    except Exception as e:
        logger.error(f"Error in enhanced PDF processing: {e}")
        logger.debug(traceback.format_exc())
        return f"[PDF 파일 처리 실패: {str(e)}]"


# === 텍스트 추출 ===

def _extract_text_blocks(
    page,
    page_num: int,
    page_height: float
) -> List[PageElement]:
    """
    페이지에서 텍스트 블록을 추출합니다.
    """
    elements = []

    try:
        # 텍스트 블록 정보 추출
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # 텍스트 블록
                bbox = block.get("bbox", (0, 0, 0, 0))

                # 블록 내 모든 라인의 텍스트 추출
                lines_text = []
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if text:
                            line_text += text
                    if line_text.strip():
                        lines_text.append(line_text)

                if lines_text:
                    text_content = "\n".join(lines_text)

                    elements.append(PageElement(
                        element_type=ElementType.TEXT,
                        content=text_content,
                        bbox=bbox,
                        page_num=page_num
                    ))

    except Exception as e:
        logger.warning(f"Error extracting text blocks from page {page_num}: {e}")
        # 폴백: 전체 텍스트 추출
        try:
            full_text = page.get_text("text")
            if full_text and full_text.strip():
                elements.append(PageElement(
                    element_type=ElementType.TEXT,
                    content=full_text,
                    bbox=(0, 0, page.rect.width, page.rect.height),
                    page_num=page_num
                ))
        except Exception:
            pass

    return elements


# === 이미지 추출 ===

async def _extract_images_from_page(
    page,
    page_num: int,
    page_height: float,
    doc,
    app_db,
    processed_images: set,
    min_image_size: int = 50,
    min_image_area: int = 2500
) -> List[PageElement]:
    """
    페이지에서 이미지를 추출하고 MinIO에 업로드합니다.
    """
    elements = []

    try:
        # 이미지 목록 가져오기
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]  # 이미지 참조 ID

                # 이미 처리된 이미지 건너뛰기
                if xref in processed_images:
                    continue

                # 이미지 데이터 추출
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_data = base_image.get("image")
                if not image_data:
                    continue

                # 이미지 크기 확인
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # 너무 작은 이미지는 건너뛰기 (아이콘, 장식 등)
                if width < min_image_size or height < min_image_size:
                    continue

                if width * height < min_image_area:
                    continue

                # 이미지 위치 찾기
                bbox = find_image_position(page, xref)
                if not bbox:
                    # 위치를 찾지 못하면 페이지 중앙에 배치
                    bbox = (0, page_height / 2, page.rect.width, page_height / 2 + 1)

                # MinIO에 업로드
                minio_path = upload_image_to_minio(image_data, app_db)

                if minio_path:
                    processed_images.add(xref)

                    elements.append(PageElement(
                        element_type=ElementType.IMAGE,
                        content=f"\n[image:{minio_path}]\n",
                        bbox=bbox,
                        page_num=page_num
                    ))
                    logger.debug(f"Extracted image from page {page_num + 1}: {width}x{height}")

            except Exception as img_e:
                logger.warning(f"Error extracting image {img_index} from page {page_num}: {img_e}")
                continue

    except Exception as e:
        logger.warning(f"Error extracting images from page {page_num}: {e}")

    return elements


# === 테이블 연속성 처리 ===

def _detect_header_rows(table_data: List[List[Optional[str]]]) -> int:
    """
    테이블에서 헤더 행 수를 감지합니다.

    헤더 특성:
    - '구 분', '최우선변제금액' 등의 키워드 포함
    - None 값이 많음 (병합 표시)
    - 데이터 행보다 앞에 위치

    Args:
        table_data: 테이블 데이터 (table.extract() 결과)

    Returns:
        헤더 행 수
    """
    header_keywords = {'구 분', '최우선변제금액', '최우선변제를', '받을 수', '임차인', '합계', '소계', '항목'}
    header_end = 0

    for i, row in enumerate(table_data):
        if i >= 10:  # 헤더가 10행을 넘기지는 않을 것
            break

        row_text = ' '.join([str(c) for c in row if c])
        has_header_keyword = any(kw in row_text for kw in header_keywords)
        none_count = sum(1 for c in row if c is None)

        if has_header_keyword or (none_count >= 1 and i < 6):
            header_end = i + 1
        elif i < 6 and (not row[0] or row[0] == ''):
            header_end = i + 1
        else:
            # 의미있는 데이터가 시작되면 중단
            if row[0] and row[0].strip() and len(row[0]) > 2:
                break

    return header_end


def _extract_table_info(table, page_num: int, table_idx: int, page_height: float) -> TableContinuationInfo:
    """
    PyMuPDF 테이블에서 연속성 처리에 필요한 정보를 추출합니다.

    Args:
        table: PyMuPDF Table 객체
        page_num: 페이지 번호
        table_idx: 테이블 인덱스
        page_height: 페이지 높이

    Returns:
        TableContinuationInfo 객체
    """
    table_data = table.extract()

    # X 좌표 추출 (컬럼 구조)
    x_coords = []
    if hasattr(table, 'cells') and table.cells:
        x_coords = sorted(set([c[0] for c in table.cells] + [c[2] for c in table.cells]))

    header_rows = _detect_header_rows(table_data)

    return TableContinuationInfo(
        page_num=page_num,
        table_idx=table_idx,
        bbox=table.bbox,
        col_count=table.col_count,
        x_coords=x_coords,
        data=table_data,
        header_rows=header_rows,
        table_obj=table,
        page_height=page_height
    )


def _is_table_continuation(prev_table: TableContinuationInfo,
                           curr_table: TableContinuationInfo,
                           tolerance: float = 5.0) -> bool:
    """
    두 테이블이 페이지 간 연속인지 판단합니다.

    연속 조건:
    1. 열 수가 동일
    2. X 좌표 (컬럼 구조)가 유사
    3. 이전 테이블이 페이지 하단에 위치
    4. 현재 테이블이 페이지 상단에 위치
    5. 현재 테이블의 첫 데이터 행에 빈 셀이 있음 (rowspan 연속)

    Args:
        prev_table: 이전 페이지의 테이블 정보
        curr_table: 현재 페이지의 테이블 정보
        tolerance: X 좌표 비교 허용 오차

    Returns:
        연속 여부
    """
    # 1. 열 수가 같아야 함
    if prev_table.col_count != curr_table.col_count:
        return False

    # 2. X 좌표가 유사해야 함
    if len(prev_table.x_coords) != len(curr_table.x_coords):
        return False

    for x1, x2 in zip(prev_table.x_coords, curr_table.x_coords):
        if abs(x1 - x2) > tolerance:
            return False

    # 3. 이전 테이블이 페이지 하단에 있어야 함 (y1이 페이지 높이의 70% 이상)
    if prev_table.bbox[3] < prev_table.page_height * 0.7:
        return False

    # 4. 현재 테이블이 페이지 상단에 있어야 함 (y0이 페이지 높이의 30% 이하)
    if curr_table.bbox[1] > curr_table.page_height * 0.3:
        return False

    # 5. 연속 테이블 특성: 데이터 시작 부분에 빈 첫 번째 열이 있음
    header_end = curr_table.header_rows
    for i in range(header_end, min(header_end + 3, len(curr_table.data))):
        row = curr_table.data[i]
        if len(row) > 1:
            # 첫 번째 열이 비어있고 두 번째 열에 데이터가 있으면 연속
            if (row[0] is None or row[0] == '') and row[1]:
                return True

    return False


def _fill_continuation_cells(curr_data: List[List[Optional[str]]],
                             prev_table: TableContinuationInfo,
                             header_rows: int) -> List[List[Optional[str]]]:
    """
    연속 테이블의 빈 셀을 이전 테이블의 rowspan 값으로 채웁니다.

    이전 테이블의 마지막 행에서 활성화된 rowspan 셀의 값을
    현재 테이블의 빈 셀에 복원합니다.

    Args:
        curr_data: 현재 테이블 데이터 (수정됨)
        prev_table: 이전 테이블 정보
        header_rows: 현재 테이블의 헤더 행 수

    Returns:
        수정된 테이블 데이터
    """
    import copy
    result_data = copy.deepcopy(curr_data)
    prev_data = prev_table.data

    # 이전 테이블의 마지막 데이터 행에서 활성화된 rowspan 값 찾기
    # 각 열에 대해 마지막 유효한 값 추적
    last_row_values = {}
    for col_idx in range(len(prev_data[0]) if prev_data else 0):
        for row_idx in range(len(prev_data) - 1, prev_table.header_rows - 1, -1):
            cell = prev_data[row_idx][col_idx]
            if cell and str(cell).strip() and cell != 'None':
                last_row_values[col_idx] = cell
                break

    if not last_row_values:
        return result_data

    logger.debug(f"Previous table last rowspan values: {list(last_row_values.keys())}")

    # 현재 테이블의 헤더 이후 빈 셀 채우기
    filled_count = 0

    for row_idx in range(header_rows, len(result_data)):
        row = result_data[row_idx]

        # 각 열에 대해 빈 셀 처리
        for col_idx in range(len(row)):
            if row[col_idx] is None or row[col_idx] == '':
                # 이전 테이블의 마지막 값으로 채우기
                if col_idx in last_row_values:
                    result_data[row_idx][col_idx] = last_row_values[col_idx]
                    filled_count += 1
            else:
                # 새로운 값이 있으면 해당 열의 rowspan 값 업데이트
                if row[col_idx] and str(row[col_idx]).strip():
                    last_row_values[col_idx] = row[col_idx]

    if filled_count > 0:
        logger.debug(f"Filled {filled_count} continuation cells")

    return result_data


@dataclass
class PageStructureAnalysis:
    """
    페이지 구조 분석 결과를 담는 데이터 클래스.

    페이지 테두리(장식용)와 내부 테이블 구조를 명확히 구분합니다.
    """
    has_page_border: bool  # 페이지 전체를 감싸는 장식용 테두리 존재 여부
    has_table_structure: bool  # 실제 테이블 구조 존재 여부
    has_lines_for_table: bool  # rects에 테이블용 선이 있어 lines 전략 사용 가능 여부
    table_regions: List[Tuple[float, float, float, float]]  # 테이블 영역 bbox 목록
    page_border_bbox: Optional[Tuple[float, float, float, float]]  # 페이지 테두리 bbox


def _analyze_page_structure(page) -> PageStructureAnalysis:
    """
    페이지의 rects와 lines를 분석하여 구조를 파악합니다.

    **핵심 목표**:
    1. 페이지 전체를 감싸는 장식용 테두리(border) 식별
    2. 내부에 존재하는 실제 테이블 구조 식별
    3. 테이블 영역의 정확한 bbox 계산
    4. 적절한 추출 전략 결정 (lines vs text)

    **일반화된 접근 방식**:
    - 페이지 테두리: 페이지 크기의 85% 이상을 차지하며 가장자리에 위치한 선
    - 테이블 구조: 교차하는 가로선과 세로선의 그리드
    - 교차 영역: 가로선과 세로선이 실제로 교차하는 영역만 테이블로 인정

    Args:
        page: pdfplumber Page 객체

    Returns:
        PageStructureAnalysis 객체
    """
    result = PageStructureAnalysis(
        has_page_border=False,
        has_table_structure=False,
        has_lines_for_table=False,
        table_regions=[],
        page_border_bbox=None
    )

    # lines와 rects 모두 분석
    lines = page.lines or []
    rects = page.rects or []

    if not lines and not rects:
        return result

    page_width = page.width
    page_height = page.height

    # === 1단계: 모든 선 수집 (lines + rects에서 선 추출) ===
    all_h_lines = []  # (y, x0, x1)
    all_v_lines = []  # (x, y0, y1)

    # lines에서 선 추출
    for line in lines:
        x0, y0, x1, y1 = line['x0'], line['top'], line['x1'], line['bottom']
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        if h < 3 and w > 10:  # 가로선
            avg_y = (y0 + y1) / 2
            all_h_lines.append((avg_y, min(x0, x1), max(x0, x1)))
        elif w < 3 and h > 10:  # 세로선
            avg_x = (x0 + x1) / 2
            all_v_lines.append((avg_x, min(y0, y1), max(y0, y1)))

    # rects에서 선 추출
    thin_threshold = 3.0
    page_border_h_lines = []  # 페이지 테두리용 가로선
    page_border_v_lines = []  # 페이지 테두리용 세로선

    # 페이지 테두리 판별 기준
    edge_margin = min(page_width, page_height) * 0.1
    page_spanning_ratio = 0.85

    for rect in rects:
        x0, y0, x1, y1 = rect['x0'], rect['top'], rect['x1'], rect['bottom']
        w = x1 - x0
        h = y1 - y0

        # 선인지 판단
        is_h_line = h <= thin_threshold and w > thin_threshold
        is_v_line = w <= thin_threshold and h > thin_threshold

        if not (is_h_line or is_v_line):
            continue

        # 페이지 테두리 판별
        at_left_edge = x0 < edge_margin
        at_right_edge = x1 > page_width - edge_margin
        at_top_edge = y0 < edge_margin
        at_bottom_edge = y1 > page_height - edge_margin
        spans_page_width = w > page_width * page_spanning_ratio
        spans_page_height = h > page_height * page_spanning_ratio

        is_page_border = False
        if is_h_line and (at_top_edge or at_bottom_edge) and spans_page_width:
            is_page_border = True
            page_border_h_lines.append((y0, x0, x1))
        elif is_v_line and (at_left_edge or at_right_edge) and spans_page_height:
            is_page_border = True
            page_border_v_lines.append((x0, y0, y1))

        # 내부 선으로 추가
        if not is_page_border:
            if is_h_line:
                avg_y = (y0 + y1) / 2
                all_h_lines.append((avg_y, min(x0, x1), max(x0, x1)))
            elif is_v_line:
                avg_x = (x0 + x1) / 2
                all_v_lines.append((avg_x, min(y0, y1), max(y0, y1)))

    # === 2단계: 페이지 테두리 판별 ===
    # 상/하/좌/우 4개의 테두리가 있으면 페이지 테두리로 판정
    if len(page_border_h_lines) >= 2 and len(page_border_v_lines) >= 2:
        result.has_page_border = True
        # 페이지 테두리 bbox 계산
        border_x0 = min(line[0] for line in page_border_v_lines)
        border_y0 = min(line[0] for line in page_border_h_lines)
        border_x1 = max(line[0] for line in page_border_v_lines)
        border_y1 = max(line[0] for line in page_border_h_lines)
        result.page_border_bbox = (border_x0, border_y0, border_x1, border_y1)

        logger.debug(
            f"[PageStructure] Page border detected: "
            f"H-lines={len(page_border_h_lines)}, V-lines={len(page_border_v_lines)}, "
            f"bbox={result.page_border_bbox}"
        )

    # === 3단계: 내부 테이블 구조 식별 ===
    if not all_h_lines or not all_v_lines:
        logger.debug(
            f"[PageStructure] No table structure: "
            f"H-lines={len(all_h_lines)}, V-lines={len(all_v_lines)}"
        )
        return result

    # Y 위치별 가로선 그룹화 (같은 Y 위치의 선들)
    h_lines_by_y = {}
    for y, x0, x1 in all_h_lines:
        y_key = round(y, 0)  # 1픽셀 단위로 그룹화
        if y_key not in h_lines_by_y:
            h_lines_by_y[y_key] = []
        h_lines_by_y[y_key].append((x0, x1))

    # X 위치별 세로선 그룹화 (같은 X 위치의 선들)
    v_lines_by_x = {}
    for x, y0, y1 in all_v_lines:
        x_key = round(x, 0)  # 1픽셀 단위로 그룹화
        if x_key not in v_lines_by_x:
            v_lines_by_x[x_key] = []
        v_lines_by_x[x_key].append((y0, y1))

    # === 4단계: 테이블 영역 식별 ===
    # 테이블 = 가로선과 세로선이 교차하는 그리드 영역
    # 최소 조건: 2개 이상의 Y 위치에 가로선, 2개 이상의 X 위치에 세로선

    h_y_positions = sorted(h_lines_by_y.keys())
    v_x_positions = sorted(v_lines_by_x.keys())

    if len(h_y_positions) < 2 or len(v_x_positions) < 2:
        logger.debug(
            f"[PageStructure] Insufficient grid structure: "
            f"H-Y positions={len(h_y_positions)}, V-X positions={len(v_x_positions)}"
        )
        return result

    # 테이블 영역 계산: 교차하는 선들의 범위
    # 각 가로선의 X 범위와 각 세로선의 Y 범위가 겹치는 영역

    # 가로선의 전체 X 범위
    h_x_min = min(min(x0 for x0, x1 in lines) for lines in h_lines_by_y.values())
    h_x_max = max(max(x1 for x0, x1 in lines) for lines in h_lines_by_y.values())

    # 세로선의 전체 Y 범위
    v_y_min = min(min(y0 for y0, y1 in lines) for lines in v_lines_by_x.values())
    v_y_max = max(max(y1 for y0, y1 in lines) for lines in v_lines_by_x.values())

    # 교차 영역 계산
    table_x0 = h_x_min
    table_y0 = v_y_min
    table_x1 = h_x_max
    table_y1 = v_y_max

    # 교차 영역이 유효한지 확인
    if table_x1 <= table_x0 or table_y1 <= table_y0:
        logger.debug("[PageStructure] Invalid table region (no overlap)")
        return result

    # 테이블 크기 검증 (너무 작으면 테이블이 아님)
    table_width = table_x1 - table_x0
    table_height = table_y1 - table_y0

    if table_width < 50 or table_height < 30:
        logger.debug(
            f"[PageStructure] Table region too small: "
            f"{table_width:.1f}x{table_height:.1f}"
        )
        return result

    # 테이블 구조 확인 (최소 3개의 Y 위치 또는 3개의 X 위치)
    has_h_dividers = len(h_y_positions) >= 3
    has_v_dividers = len(v_x_positions) >= 3

    if not (has_h_dividers or has_v_dividers):
        logger.debug(
            f"[PageStructure] Simple box, not a table: "
            f"H-Y positions={len(h_y_positions)}, V-X positions={len(v_x_positions)}"
        )
        return result

    result.has_table_structure = True
    result.has_lines_for_table = True  # rects에서 선을 찾았으므로 lines 전략 사용 가능
    result.table_regions.append((table_x0, table_y0, table_x1, table_y1))

    logger.debug(
        f"[PageStructure] Table structure found: "
        f"H-Y positions={len(h_y_positions)}, V-X positions={len(v_x_positions)}, "
        f"bbox=({table_x0:.1f}, {table_y0:.1f}, {table_x1:.1f}, {table_y1:.1f})"
    )

    return result


# === 테이블 추출 (pdfplumber 우선) ===

def _extract_tables_with_pdfplumber(file_path: str) -> Dict[int, List[PageElement]]:
    """
    pdfplumber를 사용하여 PDF에서 테이블을 추출합니다.

    pdfplumber는 PyMuPDF보다 테이블 감지가 더 정교하며,
    특히 셀 경계와 텍스트 위치 분석에 강점이 있습니다.

    **개선된 로직**:
    1. 페이지 구조 분석: 페이지 테두리와 내부 테이블 구조 구분
    2. 테이블 영역 크롭: 정확한 테이블 bbox만 추출
    3. 전략 선택: rects에 선이 있으면 lines 전략 사용

    Args:
        file_path: PDF 파일 경로

    Returns:
        페이지별 테이블 요소 딕셔너리
    """
    tables_by_page: Dict[int, List[PageElement]] = {}

    if not PDFPLUMBER_AVAILABLE:
        logger.debug("pdfplumber not available")
        return tables_by_page

    try:
        # 1단계: 모든 페이지에서 raw 테이블 데이터 수집
        all_page_tables: List[Dict] = []  # [{page_num, table_idx, data, bbox}, ...]

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # === 페이지 구조 분석 ===
                    page_analysis = _analyze_page_structure(page)

                    has_native_lines = len(page.lines) > 0
                    has_rects = len(page.rects) > 0

                    # 전략 결정 로직
                    strategies = []
                    target_page = page  # 추출 대상 페이지 (크롭될 수 있음)
                    table_region_bbox = None  # 테이블 영역 bbox

                    if has_native_lines:
                        # 네이티브 lines가 있으면 lines 전략 사용
                        strategies = [
                            {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
                            {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        ]
                    elif has_rects:
                        # rects가 있는 경우: 구조 분석 결과 사용
                        if not page_analysis.has_table_structure:
                            # 테이블 구조가 없으면 스킵
                            logger.debug(
                                f"[pdfplumber] Page {page_num + 1}: Skipping - "
                                f"no internal table structure detected"
                            )
                            continue

                        # 테이블 영역이 있고, 페이지 테두리가 있는 경우 크롭
                        if page_analysis.table_regions and page_analysis.has_page_border:
                            table_region_bbox = page_analysis.table_regions[0]

                            # 테이블 영역만 크롭
                            try:
                                # 약간의 마진 추가 (선이 잘리지 않도록)
                                margin = 1.0
                                crop_bbox = (
                                    max(0, table_region_bbox[0] - margin),
                                    max(0, table_region_bbox[1] - margin),
                                    min(page.width, table_region_bbox[2] + margin),
                                    min(page.height, table_region_bbox[3] + margin)
                                )
                                target_page = page.within_bbox(crop_bbox)

                                logger.debug(
                                    f"[pdfplumber] Page {page_num + 1}: Cropped to table region "
                                    f"({crop_bbox[0]:.1f}, {crop_bbox[1]:.1f}, {crop_bbox[2]:.1f}, {crop_bbox[3]:.1f})"
                                )
                            except Exception as crop_e:
                                logger.warning(f"[pdfplumber] Page {page_num + 1}: Crop failed: {crop_e}")
                                target_page = page

                        # rects에 테이블용 선이 있으면 lines 전략 사용
                        if page_analysis.has_lines_for_table:
                            strategies = [
                                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                                {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
                            ]
                            logger.debug(
                                f"[pdfplumber] Page {page_num + 1}: Using lines strategy "
                                f"(table structure from rects)"
                            )
                        else:
                            # 선이 없으면 text 전략 사용
                            strategies = [
                                {"vertical_strategy": "text", "horizontal_strategy": "text"},
                            ]
                    else:
                        # lines도 rects도 없으면 스킵
                        continue

                    best_tables = None
                    best_strategy = None

                    common_settings = {
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                        "edge_min_length": 10,
                        "intersection_tolerance": 5,
                    }

                    for strategy in strategies:
                        table_settings = {**common_settings, **strategy}

                        try:
                            # 크롭된 페이지(target_page)에서 테이블 추출
                            tables = target_page.extract_tables(table_settings)

                            if tables:
                                best_tables = tables
                                best_strategy = strategy
                                break
                        except Exception:
                            continue

                    if not best_tables:
                        continue

                    logger.debug(f"[pdfplumber] Page {page_num + 1}: using strategy {best_strategy}")

                    # bbox 정보 가져오기
                    found_tables = []
                    try:
                        final_settings = {**common_settings, **best_strategy}
                        found_tables = target_page.find_tables(final_settings)
                    except Exception:
                        pass

                    for table_idx, table_data in enumerate(best_tables):
                        if not table_data or not any(any(cell for cell in row if cell) for row in table_data):
                            continue

                        # 테이블 데이터 정제
                        cleaned_data = _clean_pdfplumber_table(table_data)

                        if not cleaned_data:
                            continue

                        # bbox 및 셀 정보 가져오기
                        # 참고: pdfplumber의 within_bbox()로 크롭해도 내부 객체의 좌표는
                        # 원본 페이지 좌표를 유지합니다. 따라서 offset 추가가 필요하지 않습니다.
                        bbox = (0, 0, page.width, page.height)
                        cells_info = None
                        row_boundaries = []

                        if found_tables and table_idx < len(found_tables):
                            # 크롭 여부와 관계없이 테이블 bbox는 원본 페이지 좌표입니다
                            bbox = found_tables[table_idx].bbox

                            # pdfplumber Table 객체에서 셀 정보 추출
                            cells_info = _extract_pdfplumber_cells(found_tables[table_idx])
                            # 행 경계 추출 (open border 복구용)
                            row_boundaries = _extract_row_boundaries_from_table(found_tables[table_idx])

                        # === Open Border 테이블 복구 ===
                        # 테두리가 없는 왼쪽/오른쪽 열을 복구
                        # 단, 테이블이 페이지의 큰 부분을 차지하면 복구 시도하지 않음
                        # 또는 이미 크롭된 영역에서 추출한 경우 복구 스킵
                        table_area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (page.width * page.height)

                        # 크롭된 페이지에서 추출한 경우 open border 복구 스킵
                        # (이미 정확한 테이블 영역에서 추출했으므로)
                        if table_region_bbox is None and table_area_ratio < 0.7:
                            recovered_data, recovered_bbox = _recover_open_border_table_columns(
                                page, bbox, cleaned_data, row_boundaries
                            )

                            # 복구된 데이터가 있으면 사용
                            if recovered_data != cleaned_data:
                                cleaned_data = recovered_data
                                bbox = recovered_bbox
                                # 복구 후 cells_info 갱신 (새 열이 추가되었으므로)
                                cells_info = None  # 복구된 테이블은 물리적 셀 정보 미사용
                        elif table_region_bbox is None:
                            logger.debug(
                                f"[pdfplumber] Page {page_num + 1}: Skipping open border recovery "
                                f"for large table (covers {table_area_ratio:.1%} of page)"
                            )

                        all_page_tables.append({
                            'page_num': page_num,
                            'table_idx': table_idx,
                            'data': cleaned_data,
                            'bbox': bbox,
                            'page_height': page.height,
                            'cells_info': cells_info,  # 물리적 셀 정보 추가
                        })

                except Exception as page_e:
                    logger.warning(f"[pdfplumber] Error extracting tables from page {page_num}: {page_e}")
                    continue

        if not all_page_tables:
            return tables_by_page

        # 2단계: 페이지 간 테이블 연속성 처리
        processed_tables = _process_table_continuity(all_page_tables)

        # 3단계: HTML 변환 및 PageElement 생성
        for table_info in processed_tables:
            page_num = table_info['page_num']
            table_data = table_info['data']
            bbox = table_info['bbox']
            cells_info = table_info.get('cells_info')  # 물리적 셀 정보

            # HTML로 변환 (물리적 셀 기반 rowspan 감지)
            html_table = _convert_table_to_html_with_cell_spans(table_data, cells_info)

            if html_table:
                if page_num not in tables_by_page:
                    tables_by_page[page_num] = []

                tables_by_page[page_num].append(PageElement(
                    element_type=ElementType.TABLE,
                    content=html_table,
                    bbox=bbox,
                    page_num=page_num
                ))
                logger.debug(f"[pdfplumber] Extracted table {table_info['table_idx'] + 1} from page {page_num + 1}")

        logger.info(f"[pdfplumber] Extracted tables from {len(tables_by_page)} pages")

    except Exception as e:
        logger.warning(f"[pdfplumber] Table extraction error: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return tables_by_page


def _process_table_continuity(all_tables: List[Dict]) -> List[Dict]:
    """
    페이지 간 테이블 연속성을 처리합니다.

    이 함수는 PDF에서 테이블이 여러 페이지에 걸쳐 분할될 때 발생하는
    구조적 문제를 해결합니다.

    주요 케이스:
    1. 열 수 감소 (2열 → 1열): 카테고리 열이 누락된 경우
    2. 하이브리드 페이지: 이전 카테고리 마무리 + 새 카테고리 시작
    3. 연속적인 1열 테이블: 여러 페이지에 걸친 동일 카테고리

    알고리즘:
    1. 이전 테이블의 마지막 카테고리 값 추적
    2. 현재 테이블의 열 구조 분석
    3. 빈 첫 번째 열을 적절한 카테고리로 채움
    4. 새 카테고리가 등장하면 추적 값 업데이트

    Args:
        all_tables: 모든 페이지의 테이블 정보 리스트

    Returns:
        연속성이 처리된 테이블 정보 리스트
    """
    import copy

    if not all_tables:
        return all_tables

    result = copy.deepcopy(all_tables)

    # 전역 카테고리 추적 (여러 페이지에 걸친 연속성 처리)
    global_last_category = None

    for i in range(len(result)):
        curr_table = result[i]
        curr_data = curr_table['data']

        if not curr_data:
            continue

        curr_cols = max(len(row) for row in curr_data) if curr_data else 0
        curr_bbox = curr_table.get('bbox', (0, 0, 0, 0))
        curr_page_height = curr_table.get('page_height', 1000)

        # 현재 테이블이 페이지 상단에 있는지 (연속 가능성)
        curr_at_top = curr_bbox[1] < curr_page_height * 0.15

        # 이전 테이블 정보
        prev_table = result[i - 1] if i > 0 else None
        prev_data = prev_table['data'] if prev_table else None
        prev_cols = max(len(row) for row in prev_data) if prev_data else 0
        prev_bbox = prev_table.get('bbox', (0, 0, 0, 0)) if prev_table else (0, 0, 0, 0)
        prev_page_height = prev_table.get('page_height', 1000) if prev_table else 1000

        # 이전 테이블이 페이지 하단에 있는지
        prev_at_bottom = prev_bbox[3] > prev_page_height * 0.85 if prev_table else False

        # 다른 페이지 여부
        is_different_page = prev_table and curr_table['page_num'] > prev_table['page_num']

        # === Case 1: 첫 번째 테이블 (기준 설정) ===
        if i == 0:
            # 첫 번째 테이블에서 마지막 카테고리 추출
            global_last_category = _extract_last_category(curr_data)
            logger.debug(f"[continuity] Page {curr_table['page_num'] + 1}: Initial category = {global_last_category}")
            continue

        # === Case 2: 열 수 감소 (2열 → 1열) ===
        # 카테고리 열이 누락된 경우 - 열 추가 필요
        if curr_cols < prev_cols and is_different_page and prev_at_bottom and curr_at_top:
            logger.info(f"[continuity] Page {curr_table['page_num'] + 1}: Column reduction detected ({prev_cols} → {curr_cols})")

            if global_last_category:
                # 모든 행에 첫 번째 열 추가
                for row in curr_data:
                    row.insert(0, '')  # 빈 값으로 초기화, 아래에서 채움

                # 이제 2열이 되었으므로 curr_cols 업데이트
                curr_cols = max(len(row) for row in curr_data)

        # === Case 3: 첫 번째 열이 비어있는 행 처리 ===
        # 연속성이 있거나 (다른 페이지에서 이어짐) 같은 테이블 내 rowspan인 경우
        if curr_cols >= 2:
            current_category = global_last_category

            for row_idx, row in enumerate(curr_data):
                if len(row) < 2:
                    continue

                first_col = row[0]
                first_col_empty = not first_col or not str(first_col).strip()
                second_col = row[1] if len(row) > 1 else ""
                second_col_has_value = second_col and str(second_col).strip()

                if first_col_empty and second_col_has_value:
                    # 빈 첫 번째 열 + 값 있는 두 번째 열 = 이전 카테고리 연속
                    if current_category:
                        row[0] = current_category
                        logger.debug(f"[continuity] Page {curr_table['page_num'] + 1}, Row {row_idx}: Filled with '{current_category}'")
                elif not first_col_empty:
                    # 새 카테고리 발견 - 추적 값 업데이트
                    current_category = first_col
                    logger.debug(f"[continuity] Page {curr_table['page_num'] + 1}, Row {row_idx}: New category '{current_category}'")

            # 전역 카테고리 업데이트 (이 테이블의 마지막 유효 카테고리)
            new_last_category = _extract_last_category(curr_data)
            if new_last_category:
                global_last_category = new_last_category

        # === Case 4: 1열 테이블이 계속되는 경우 ===
        # 여러 페이지에 걸쳐 1열 테이블이 연속되면, 모두 같은 카테고리
        elif curr_cols == 1 and is_different_page:
            if global_last_category and prev_at_bottom and curr_at_top:
                logger.info(f"[continuity] Page {curr_table['page_num'] + 1}: Single column continuation, adding category '{global_last_category}'")

                # 모든 행에 카테고리 열 추가
                for row in curr_data:
                    row.insert(0, '')

                # 빈 첫 번째 열 채우기
                for row in curr_data:
                    if len(row) >= 2 and (not row[0] or not str(row[0]).strip()):
                        row[0] = global_last_category

        # 업데이트된 데이터 저장
        result[i]['data'] = curr_data

    return result


# === 테두리 없는 테이블 (Open Border Table) 복구 ===

def _recover_open_border_table_columns(
    page,
    table_bbox: Tuple[float, float, float, float],
    table_data: List[List[Optional[str]]],
    row_boundaries: List[float]
) -> Tuple[List[List[Optional[str]]], Tuple[float, float, float, float]]:
    """
    테두리가 없는(열린) 테이블의 누락된 열을 복구합니다.

    **문제 상황**:
    PDF 테이블의 좌우 테두리가 없는 경우, pdfplumber가 해당 열을
    테이블의 일부로 인식하지 못하고 일반 텍스트로 추출합니다.

    예시:
    +------+------+------+------+
    | 등급 |  S   |  A   |  B   |  D    <- D는 테두리 없음
    +------+------+------+------+
    | 구간 | S+   | A+   | B+   |  D+   <- D+는 테두리 없음
    +------+------+------+------+

    **해결 방법**:
    1. pdfplumber로 추출한 테이블의 bbox 확인
    2. 페이지의 모든 텍스트에서 테이블 bbox 외부이면서 같은 Y 좌표에 있는 텍스트 찾기
    3. X 좌표 기준으로 왼쪽/오른쪽 누락 열 구분
    4. 각 행에 누락된 텍스트를 적절한 위치에 삽입

    Args:
        page: pdfplumber Page 객체
        table_bbox: 테이블의 bbox (x0, y0, x1, y1)
        table_data: 테이블 데이터 (2D 리스트)
        row_boundaries: 테이블의 행 경계 Y 좌표 리스트

    Returns:
        (복구된 테이블 데이터, 확장된 bbox) 또는 원본 데이터
    """
    if not table_data or not row_boundaries:
        return table_data, table_bbox

    try:
        t_x0, t_y0, t_x1, t_y1 = table_bbox
        num_rows = len(table_data)
        num_cols = max(len(row) for row in table_data) if table_data else 0
        table_width = t_x1 - t_x0
        table_height = t_y1 - t_y0

        # === 안전 검증: 페이지 대비 테이블 크기 ===
        # 테이블이 너무 크면 복구 시도하지 않음 (페이지 테두리 오인식 방지)
        page_width = page.width
        page_height = page.height
        table_area_ratio = (table_width * table_height) / (page_width * page_height)

        if table_area_ratio > 0.7:
            logger.debug(
                f"[OpenBorder] Skipping recovery: table covers {table_area_ratio:.1%} of page"
            )
            return table_data, table_bbox

        # === 안전 검증: 테이블 구조 ===
        # 열이 1-2개뿐인 간단한 테이블은 복구 대상이 아님
        if num_cols <= 2:
            logger.debug(
                f"[OpenBorder] Skipping recovery: table has only {num_cols} columns"
            )
            return table_data, table_bbox

        # 행 경계가 부족하면 균등 분할
        if len(row_boundaries) < num_rows + 1:
            row_height = table_height / num_rows
            row_boundaries = [t_y0 + i * row_height for i in range(num_rows + 1)]

        # 페이지에서 모든 텍스트(words) 추출
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            extra_attrs=['fontname', 'size']
        )

        if not words:
            return table_data, table_bbox

        y_tolerance = 5   # 행 경계 판단 허용 오차

        # === 1단계: 테이블 Y 범위 내의 모든 외부 텍스트 수집 ===
        left_texts = []   # 테이블 왼쪽의 텍스트 [(row_idx, x0, x1, text), ...]
        right_texts = []  # 테이블 오른쪽의 텍스트 [(row_idx, x0, x1, text), ...]

        for word in words:
            word_x0 = word['x0']
            word_x1 = word['x1']
            word_y0 = word['top']
            word_y1 = word['bottom']
            word_y_center = (word_y0 + word_y1) / 2
            word_text = word['text'].strip()

            if not word_text:
                continue

            # 텍스트가 테이블 Y 범위 내인지 확인
            if word_y_center < t_y0 - y_tolerance or word_y_center > t_y1 + y_tolerance:
                continue

            # 이미 테이블 내부에 있는 텍스트는 제외
            if word_x0 >= t_x0 - y_tolerance and word_x1 <= t_x1 + y_tolerance:
                continue

            # 행 인덱스 찾기
            row_idx = _find_row_index_for_y(word_y_center, row_boundaries, y_tolerance)
            if row_idx < 0 or row_idx >= num_rows:
                continue

            # 왼쪽 또는 오른쪽 분류
            if word_x1 < t_x0:
                left_texts.append((row_idx, word_x0, word_x1, word_text))
            elif word_x0 > t_x1:
                right_texts.append((row_idx, word_x0, word_x1, word_text))

        # 누락된 열이 없으면 원본 반환
        if not left_texts and not right_texts:
            return table_data, table_bbox

        # === 2단계: 열 그룹화 및 별도 테이블 감지 ===
        left_columns = _group_texts_into_columns(left_texts, num_rows)
        right_columns = _group_texts_into_columns(right_texts, num_rows)

        logger.debug(f"[OpenBorder] Raw columns - Left: {len(left_columns)}, Right: {len(right_columns)}")

        # === 핵심 검증 1: 별도 테이블 감지 ===
        # 2개 이상의 열 구조가 발견되면 별도 테이블로 판단
        if len(left_columns) >= 2:
            logger.debug(f"[OpenBorder] Left side has {len(left_columns)} columns - likely separate table, skipping")
            left_columns = []

        if len(right_columns) >= 2:
            logger.debug(f"[OpenBorder] Right side has {len(right_columns)} columns - likely separate table, skipping")
            right_columns = []

        # === 핵심 검증 2: 근접성 검증 ===
        # 테이블과 "바로 인접한" 열만 복구 (간격이 너무 크면 별도 객체)
        max_gap = min(50, table_width * 0.2)  # 최대 50pt 또는 테이블 너비의 20%

        # 왼쪽 열 검증: 열의 오른쪽 끝이 테이블 왼쪽과 가까워야 함
        valid_left_columns = []
        for col in left_columns:
            # 열의 최대 x1 (오른쪽 끝)
            col_x1_max = max(t[2] for row_idx, (row_idx2, x0, x1, text) in enumerate(left_texts)
                           if row_idx2 in col for t in [(row_idx2, x0, x1, text)])
            # 다시 계산
            col_x1_values = []
            for row_idx in col.keys():
                for (ridx, x0, x1, txt) in left_texts:
                    if ridx == row_idx:
                        col_x1_values.append(x1)
                        break
            if col_x1_values:
                col_x1_max = max(col_x1_values)
                gap = t_x0 - col_x1_max
                if gap <= max_gap:
                    valid_left_columns.append(col)
                else:
                    logger.debug(f"[OpenBorder] Left column gap {gap:.1f} > max_gap {max_gap:.1f}, skipping")

        # 오른쪽 열 검증: 열의 왼쪽 끝이 테이블 오른쪽과 가까워야 함
        valid_right_columns = []
        for col in right_columns:
            col_x0_values = []
            for row_idx in col.keys():
                for (ridx, x0, x1, txt) in right_texts:
                    if ridx == row_idx:
                        col_x0_values.append(x0)
                        break
            if col_x0_values:
                col_x0_min = min(col_x0_values)
                gap = col_x0_min - t_x1
                if gap <= max_gap:
                    valid_right_columns.append(col)
                else:
                    logger.debug(f"[OpenBorder] Right column gap {gap:.1f} > max_gap {max_gap:.1f}, skipping")

        left_columns = valid_left_columns
        right_columns = valid_right_columns

        # === 핵심 검증 3: 최소 행 매칭 비율 ===
        min_row_ratio = 0.5
        min_rows_required = max(2, int(num_rows * min_row_ratio))

        left_columns = [col for col in left_columns if len(col) >= min_rows_required]
        right_columns = [col for col in right_columns if len(col) >= min_rows_required]

        # 검증 후 열이 없으면 원본 반환
        if not left_columns and not right_columns:
            logger.debug(f"[OpenBorder] No valid columns after filtering (min {min_rows_required} rows required)")
            return table_data, table_bbox

        logger.debug(f"[OpenBorder] Valid columns - Left: {len(left_columns)}, Right: {len(right_columns)}")

        # 새 테이블 데이터 생성
        new_table_data = []
        for row_idx, row in enumerate(table_data):
            new_row = []

            # 왼쪽 열들 추가 (X 좌표 순서대로)
            for col_texts in left_columns:
                cell_text = col_texts.get(row_idx, '')
                new_row.append(cell_text)

            # 기존 열들 추가
            new_row.extend(row)

            # 오른쪽 열들 추가 (X 좌표 순서대로)
            for col_texts in right_columns:
                cell_text = col_texts.get(row_idx, '')
                new_row.append(cell_text)

            new_table_data.append(new_row)

        # 확장된 bbox 계산
        new_x0 = t_x0
        new_x1 = t_x1

        if left_texts:
            new_x0 = min(t[1] for t in left_texts) - 5
        if right_texts:
            new_x1 = max(t[2] for t in right_texts) + 5

        new_bbox = (new_x0, t_y0, new_x1, t_y1)

        logger.info(f"[OpenBorder] Recovered table: {len(table_data[0])} cols -> {len(new_table_data[0])} cols")

        return new_table_data, new_bbox

    except Exception as e:
        logger.warning(f"[OpenBorder] Failed to recover columns: {e}")
        return table_data, table_bbox


def _find_row_index_for_y(y_coord: float, row_boundaries: List[float], tolerance: float = 5.0) -> int:
    """
    Y 좌표가 속하는 행 인덱스를 찾습니다.

    Args:
        y_coord: Y 좌표
        row_boundaries: 행 경계 리스트 [y0, y1, y2, ..., yn] (n+1개 = n행)
        tolerance: 허용 오차

    Returns:
        행 인덱스 (0-based) 또는 -1 (범위 밖)
    """
    if len(row_boundaries) < 2:
        return -1

    for i in range(len(row_boundaries) - 1):
        row_top = row_boundaries[i] - tolerance
        row_bottom = row_boundaries[i + 1] + tolerance

        if row_top <= y_coord <= row_bottom:
            return i

    return -1


def _group_texts_into_columns(
    texts: List[Tuple[int, float, float, str]],
    num_rows: int,
    x_tolerance: float = 15.0
) -> List[Dict[int, str]]:
    """
    텍스트들을 X 좌표 기준으로 열로 그룹화합니다.

    같은 X 좌표(허용 오차 내)에 있는 텍스트들을 하나의 열로 묶습니다.

    Args:
        texts: [(row_idx, x0, x1, text), ...]
        num_rows: 테이블 행 수
        x_tolerance: X 좌표 허용 오차

    Returns:
        [col1_dict, col2_dict, ...]
        각 col_dict = {row_idx: text, ...}
    """
    if not texts:
        return []

    # X 좌표로 정렬
    sorted_texts = sorted(texts, key=lambda t: t[1])

    # X 좌표 그룹화
    columns = []
    current_col = {}
    current_x = None

    for row_idx, x0, x1, text in sorted_texts:
        if current_x is None:
            current_x = x0
            current_col = {row_idx: text}
        elif abs(x0 - current_x) <= x_tolerance:
            # 같은 열에 속함
            if row_idx in current_col:
                # 같은 행에 이미 텍스트가 있으면 합치기
                current_col[row_idx] = current_col[row_idx] + ' ' + text
            else:
                current_col[row_idx] = text
        else:
            # 새 열 시작
            if current_col:
                columns.append(current_col)
            current_x = x0
            current_col = {row_idx: text}

    # 마지막 열 추가
    if current_col:
        columns.append(current_col)

    return columns


def _extract_row_boundaries_from_table(table) -> List[float]:
    """
    pdfplumber Table 객체에서 행 경계 Y 좌표를 추출합니다.

    Args:
        table: pdfplumber Table 객체

    Returns:
        행 경계 Y 좌표 리스트 [y0, y1, y2, ..., yn]
    """
    row_boundaries = []

    try:
        if hasattr(table, 'rows') and table.rows:
            for row in table.rows:
                if hasattr(row, 'bbox'):
                    row_boundaries.append(row.bbox[1])  # y0

            # 마지막 행의 y1 추가
            if table.rows and hasattr(table.rows[-1], 'bbox'):
                row_boundaries.append(table.rows[-1].bbox[3])
    except Exception:
        pass

    return row_boundaries


def _extract_last_category(table_data: List[List[Optional[str]]]) -> Optional[str]:
    """
    테이블 데이터에서 마지막 유효한 카테고리(첫 번째 열) 값을 추출합니다.

    Args:
        table_data: 테이블 데이터

    Returns:
        마지막 유효한 첫 번째 열 값 또는 None
    """
    if not table_data:
        return None

    for row in reversed(table_data):
        if row and len(row) >= 1:
            first_col = row[0]
            if first_col and str(first_col).strip():
                return first_col

    return None


def _extract_pdfplumber_cells(table) -> Optional[Dict]:
    """
    pdfplumber Table 객체에서 셀의 물리적 정보를 추출합니다.

    **핵심 원리**:
    pdfplumber의 Table.rows와 Table.cells를 사용하여 정확한 그리드 구조를 파악합니다.

    **중요**: Row.cells에서 None은 두 가지 의미가 있습니다:
    1. 왼쪽 셀의 colspan 연속 (가로 병합)
    2. 위쪽 셀의 rowspan 연속 (세로 병합)

    이를 구분하기 위해 **물리적 bbox 좌표**를 분석합니다:
    - colspan: bbox.x1이 어느 열 경계까지 확장되는지
    - rowspan: bbox.y1이 어느 행 경계까지 확장되는지

    Args:
        table: pdfplumber Table 객체

    Returns:
        {
            'cells': List of {'row': int, 'col': int, 'rowspan': int, 'colspan': int, 'bbox': tuple},
            'num_rows': int,
            'num_cols': int,
            'col_boundaries': List[float],  # 열 경계선 x 좌표
            'row_boundaries': List[float],  # 행 경계선 y 좌표
        }
        또는 실패 시 None
    """
    try:
        if not hasattr(table, 'rows') or not table.rows:
            return None

        rows = table.rows
        num_rows = len(rows)

        if num_rows == 0:
            return None

        # 열 수는 첫 행의 cells 수로 결정
        num_cols = len(rows[0].cells) if hasattr(rows[0], 'cells') else 0

        if num_cols == 0:
            return None

        # === 1단계: 열 경계선(x 좌표) 수집 ===
        # 모든 셀의 x0 좌표를 수집하여 열 경계 결정
        col_x_coords = set()
        for row in rows:
            if hasattr(row, 'cells'):
                for cell_bbox in row.cells:
                    if cell_bbox is not None:
                        col_x_coords.add(round(cell_bbox[0], 1))  # x0
                        col_x_coords.add(round(cell_bbox[2], 1))  # x1

        col_boundaries = sorted(col_x_coords)

        # === 2단계: 행 경계선(y 좌표) 수집 ===
        row_boundaries = []
        for row in rows:
            if hasattr(row, 'bbox'):
                row_boundaries.append(row.bbox[1])  # y0

        # 마지막 행의 y1 추가
        if rows and hasattr(rows[-1], 'bbox'):
            row_boundaries.append(rows[-1].bbox[3])

        # === 3단계: 각 셀의 rowspan/colspan 계산 ===
        cells_info = []

        for row_idx, row in enumerate(rows):
            if not hasattr(row, 'cells'):
                continue

            col_idx = 0
            while col_idx < len(row.cells):
                cell_bbox = row.cells[col_idx]

                if cell_bbox is None:
                    # None이면 다른 셀에 의해 병합된 영역
                    col_idx += 1
                    continue

                # 실제 셀 발견 - bbox 좌표로 정확한 span 계산
                cell_x0, cell_y0, cell_x1, cell_y1 = cell_bbox

                # 열 시작 인덱스 찾기 (cell_x0가 col_boundaries에서 몇 번째인지)
                col_start_idx = 0
                for i, bound in enumerate(col_boundaries):
                    if abs(cell_x0 - bound) <= 3.0:
                        col_start_idx = i
                        break

                # colspan 계산: bbox.x1이 어느 열 경계까지 확장되는지
                colspan = _calculate_span_from_coord(
                    cell_x1, col_boundaries, col_start_idx, tolerance=3.0
                )

                # rowspan 계산: bbox.y1이 어느 행 경계까지 확장되는지
                rowspan = _calculate_span_from_coord(
                    cell_y1, row_boundaries, row_idx, tolerance=3.0
                )

                cells_info.append({
                    'row': row_idx,
                    'col': col_idx,
                    'rowspan': max(1, rowspan),
                    'colspan': max(1, colspan),
                    'bbox': cell_bbox,
                })

                # 다음 실제 셀로 이동 (None들 건너뛰기)
                col_idx += 1
                while col_idx < len(row.cells) and row.cells[col_idx] is None:
                    col_idx += 1

        return {
            'cells': cells_info,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'col_boundaries': col_boundaries,
            'row_boundaries': row_boundaries,
        }

    except Exception as e:
        logger.debug(f"Failed to extract pdfplumber cells: {e}")
        return None


def _calculate_span_from_coord(
    end_coord: float,
    boundaries: List[float],
    start_idx: int,
    tolerance: float = 3.0
) -> int:
    """
    좌표와 경계선 리스트를 비교하여 span 수를 계산합니다.

    예시:
    - boundaries = [0, 100, 200, 300]  (열 경계선)
    - start_idx = 0  (첫 번째 열에서 시작)
    - end_coord = 200  (셀의 x1 좌표)
    - 결과: colspan = 2  (0~100, 100~200 두 열을 차지)

    Args:
        end_coord: 셀의 끝 좌표 (x1 또는 y1)
        boundaries: 경계선 좌표 리스트
        start_idx: 시작 인덱스
        tolerance: 좌표 비교 허용 오차

    Returns:
        span 수 (최소 1)
    """
    if not boundaries or start_idx >= len(boundaries) - 1:
        return 1

    span = 1

    # start_idx+1부터 경계선을 순회하며 end_coord가 넘는 경계 수 카운트
    for i in range(start_idx + 1, len(boundaries)):
        boundary = boundaries[i]

        # end_coord가 이 경계를 넘으면 span 증가
        if end_coord > boundary + tolerance:
            span += 1
        elif abs(end_coord - boundary) <= tolerance:
            # end_coord가 이 경계와 일치하면 여기서 끝
            break
        else:
            # end_coord가 이 경계보다 작으면 이전 영역에서 끝
            break

    return span


def _convert_table_to_html_with_cell_spans(
    table_data: List[List[Optional[str]]],
    cells_info: Optional[Dict] = None
) -> str:
    """
    테이블 데이터를 HTML로 변환합니다.
    pdfplumber의 물리적 셀 정보를 사용하여 정확한 rowspan/colspan을 계산합니다.

    알고리즘 (물리적 셀 bbox 기반 - 100% 정확):
    1. Table.cells에서 각 물리적 셀의 bbox 추출
    2. bbox의 시작/끝 좌표를 그리드 인덱스로 변환
    3. rowspan = (y_end - y_start), colspan = (x_end - x_start)

    이 방식은 PDF의 실제 셀 구조를 그대로 반영합니다.

    Args:
        table_data: 테이블 데이터 (2D 리스트, 텍스트 값)
        cells_info: _extract_pdfplumber_cells()의 반환값

    Returns:
        HTML 테이블 문자열
    """
    if not table_data or len(table_data) == 0:
        return ""

    num_rows = len(table_data)
    num_cols = max(len(row) for row in table_data) if table_data else 0

    if num_cols == 0:
        return ""

    # rowspan/colspan 맵 초기화
    span_map = [[{'rowspan': 1, 'colspan': 1} for _ in range(num_cols)] for _ in range(num_rows)]
    skip_map = [[False for _ in range(num_cols)] for _ in range(num_rows)]

    # 물리적 셀 정보가 있으면 이를 기반으로 span 계산
    if cells_info and 'cells' in cells_info:
        _apply_physical_spans(span_map, skip_map, cells_info, num_rows, num_cols)

    # HTML 생성
    return _generate_html_table(table_data, span_map, skip_map, num_rows, num_cols)


def _apply_physical_spans(
    span_map: List[List[Dict]],
    skip_map: List[List[bool]],
    cells_info: Dict,
    num_rows: int,
    num_cols: int
) -> None:
    """
    물리적 셀 bbox 정보를 기반으로 정확한 rowspan/colspan을 계산합니다.

    알고리즘:
    1. _extract_pdfplumber_cells()에서 각 셀의 row, col, rowspan, colspan 추출
    2. 해당 정보를 span_map에 적용
    3. 병합된 영역의 다른 셀들을 skip_map에 표시

    이 방식의 장점:
    - 물리적 bbox에서 직접 계산하므로 100% 정확
    - rowspan/colspan 혼재 상황도 완벽 처리
    - 휴리스틱 없음

    Args:
        span_map: rowspan/colspan 정보를 저장할 맵
        skip_map: 건너뛸 셀을 표시할 맵
        cells_info: _extract_pdfplumber_cells()의 반환값
        num_rows: 행 수
        num_cols: 열 수
    """
    cells = cells_info.get('cells', [])

    # 먼저 모든 셀의 span 정보 적용
    for cell in cells:
        row = cell.get('row', 0)
        col = cell.get('col', 0)
        rowspan = cell.get('rowspan', 1)
        colspan = cell.get('colspan', 1)

        # 범위 체크
        if row >= num_rows or col >= num_cols:
            continue

        # span 정보 설정
        span_map[row][col] = {'rowspan': rowspan, 'colspan': colspan}

        # 병합된 영역의 다른 셀들을 skip 처리
        if rowspan > 1 or colspan > 1:
            for r in range(row, min(row + rowspan, num_rows)):
                for c in range(col, min(col + colspan, num_cols)):
                    if r != row or c != col:
                        skip_map[r][c] = True


def _generate_html_table(
    table_data: List[List[Optional[str]]],
    span_map: List[List[Dict]],
    skip_map: List[List[bool]],
    num_rows: int,
    num_cols: int
) -> str:
    """
    span 정보를 적용하여 HTML 테이블을 생성합니다.

    Args:
        table_data: 테이블 데이터
        span_map: rowspan/colspan 정보
        skip_map: 건너뛸 셀 표시
        num_rows: 행 수
        num_cols: 열 수

    Returns:
        HTML 테이블 문자열
    """
    html_parts = ["<table border='1'>"]

    for row_idx in range(num_rows):
        row = table_data[row_idx] if row_idx < len(table_data) else []
        row_cells = []

        for col_idx in range(num_cols):
            # 건너뛸 셀인지 확인
            if skip_map[row_idx][col_idx]:
                continue

            cell_value = row[col_idx] if col_idx < len(row) else ""
            cell_text = str(cell_value).strip() if cell_value else ""
            cell_text = cell_text.replace("&", "&amp;")
            cell_text = cell_text.replace("<", "&lt;")
            cell_text = cell_text.replace(">", "&gt;")
            cell_text = cell_text.replace("\n", "<br>")

            tag = "th" if row_idx == 0 else "td"

            span_info = span_map[row_idx][col_idx]
            rowspan = span_info.get('rowspan', 1)
            colspan = span_info.get('colspan', 1)

            attrs = []
            if rowspan > 1:
                attrs.append(f"rowspan='{rowspan}'")
            if colspan > 1:
                attrs.append(f"colspan='{colspan}'")

            attr_str = " " + " ".join(attrs) if attrs else ""
            row_cells.append(f"<{tag}{attr_str}>{cell_text}</{tag}>")

        if row_cells:
            html_parts.append("<tr>" + "".join(row_cells) + "</tr>")

    html_parts.append("</table>")

    return "\n".join(html_parts)


def _clean_pdfplumber_table(table_data: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """
    pdfplumber에서 추출한 테이블 데이터를 정제합니다.

    - None 값을 빈 문자열로 변환
    - 앞뒤 공백 제거
    - 빈 행 제거
    - 열 수 통일

    Args:
        table_data: pdfplumber extract_tables() 결과

    Returns:
        정제된 테이블 데이터
    """
    if not table_data:
        return []

    # 열 수 통일
    max_cols = max(len(row) for row in table_data if row) if table_data else 0

    cleaned = []
    for row in table_data:
        cleaned_row = []

        if not row:
            # 빈 행도 유지 (물리적 셀 정보와 매핑 유지를 위해)
            cleaned_row = [""] * max_cols
        else:
            for i in range(max_cols):
                if i < len(row):
                    cell = row[i]
                    if cell is None:
                        cleaned_row.append("")
                    else:
                        # 줄바꿈을 공백으로 대체하고 정제
                        cell_str = str(cell).replace('\n', ' ').strip()
                        cleaned_row.append(cell_str)
                else:
                    cleaned_row.append("")

        # 빈 행도 포함 (물리적 셀 정보와 행 인덱스 매핑을 위해)
        cleaned.append(cleaned_row)

    return cleaned


# === 테이블 추출 (PyMuPDF - 폴백) ===

def _find_tables_with_fallback(page) -> List[Any]:
    """
    PyMuPDF를 사용하여 다양한 전략으로 테이블을 찾습니다.
    (pdfplumber 폴백용)

    전략 순서:
    1. lines (기본) - 벡터 그래픽의 선과 사각형 테두리 사용
    2. lines_strict - 테두리 없는 사각형 무시
    3. text - 텍스트 위치 기반 가상 그리드 생성

    Args:
        page: PyMuPDF Page 객체

    Returns:
        발견된 테이블 리스트 (가장 많은 테이블을 찾은 전략의 결과)
    """
    if not hasattr(page, 'find_tables'):
        return []

    strategies = [
        ("lines", {}),
        ("lines_strict", {}),
        ("text", {"min_words_vertical": 2, "min_words_horizontal": 1}),
        # 텍스트 전략을 더 관대하게 시도
        ("text", {"min_words_vertical": 1, "min_words_horizontal": 1}),
    ]

    best_tables = []
    best_strategy = None
    best_cell_count = 0

    for strategy_name, extra_params in strategies:
        try:
            params = {"strategy": strategy_name, **extra_params}
            result = page.find_tables(**params)

            if result and result.tables:
                tables = result.tables

                # 테이블 품질 평가: 유효한 데이터가 있는 테이블만 계산
                valid_tables = []
                total_cells = 0
                for table in tables:
                    try:
                        data = table.extract()
                        if data and any(any(cell for cell in row if cell) for row in data):
                            valid_tables.append(table)
                            # 테이블의 총 셀 수 계산 (더 많은 셀 = 더 상세한 테이블)
                            total_cells += sum(len(row) for row in data)
                    except Exception:
                        continue

                # 테이블 수와 셀 수를 종합적으로 비교
                if len(valid_tables) > len(best_tables) or \
                   (len(valid_tables) == len(best_tables) and total_cells > best_cell_count):
                    best_tables = valid_tables
                    best_strategy = strategy_name
                    best_cell_count = total_cells

        except Exception as e:
            logger.debug(f"Strategy '{strategy_name}' failed: {e}")
            continue

    if best_strategy:
        logger.debug(f"Best table detection strategy: {best_strategy} (found {len(best_tables)} tables with {best_cell_count} cells)")

    return best_tables


def _merge_fragmented_columns(table_data: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """
    과도하게 분리된 열을 병합합니다.

    PyMuPDF의 텍스트 기반 테이블 감지가 단어 사이의 공백을 열 구분으로
    잘못 인식하여 열이 과도하게 분리되는 경우가 있습니다.

    예시:
    입력: [["방송", "중", "표현", "불가 리스트"], ["미래한정", "표", "현", "영원히"]]
    출력: [["방송 중 표현불가 리스트"], ["미래한정 표현", "영원히"]]

    병합 전략:
    1. 각 열이 거의 모든 행에서 1-2글자만 포함하면 조각 열로 판단
    2. 연속된 조각 열들을 병합하여 의미 있는 단어/구 복원
    3. 마지막 열(실제 데이터 열)은 보존

    Args:
        table_data: 테이블 데이터

    Returns:
        열이 병합된 테이블 데이터
    """
    import copy

    if not table_data or len(table_data) == 0:
        return table_data

    num_rows = len(table_data)
    num_cols = max(len(row) for row in table_data) if table_data else 0

    if num_cols <= 2:
        # 이미 2열 이하면 병합 불필요
        return table_data

    # 각 열의 특성 분석 - 더 엄격한 기준 적용
    col_is_fragment = []

    for col_idx in range(num_cols):
        short_or_empty_count = 0
        total_count = 0
        has_long_content = False

        for row in table_data:
            if col_idx < len(row):
                cell = row[col_idx]
                cell_str = str(cell).strip() if cell else ""
                total_count += 1

                if not cell_str or len(cell_str) <= 2:
                    short_or_empty_count += 1
                elif len(cell_str) >= 4:  # 4글자 이상의 내용이 있으면 실제 데이터 열
                    has_long_content = True

        # 열 특성 판단: 거의 모든 셀이 짧고, 긴 내용이 없으면 조각 열
        if total_count > 0 and not has_long_content:
            ratio = short_or_empty_count / total_count
            col_is_fragment.append(ratio > 0.8)  # 80% 이상이 짧으면 조각
        else:
            col_is_fragment.append(False)

    # 연속된 조각 열 그룹 찾기
    # 마지막 열은 보통 실제 데이터이므로 조각으로 취급하지 않음
    if num_cols > 0:
        col_is_fragment[-1] = False

    # 첫 번째 열부터 시작하여 연속된 조각 열 병합
    merge_ranges = []  # [(start, end), ...]

    i = 0
    while i < num_cols:
        if col_is_fragment[i]:
            # 조각 열 시작
            start = i
            while i < num_cols and col_is_fragment[i]:
                i += 1
            end = i - 1

            # 이전 비-조각 열이 있으면 함께 병합
            if start > 0 and not col_is_fragment[start - 1]:
                start -= 1

            merge_ranges.append((start, end))
        else:
            # 비-조각 열 다음에 조각 열이 있는지 확인
            if i + 1 < num_cols and col_is_fragment[i + 1]:
                start = i
                i += 1
                while i < num_cols and col_is_fragment[i]:
                    i += 1
                end = i - 1
                merge_ranges.append((start, end))
            else:
                i += 1

    if not merge_ranges:
        return table_data

    logger.debug(f"Fragment columns detected. Merge ranges: {merge_ranges}")
    logger.debug(f"Column fragment status: {col_is_fragment}")

    # 병합된 결과 생성
    result = []
    for row_idx, row in enumerate(table_data):
        new_row = []
        processed_cols = set()

        for col_idx in range(len(row)):
            if col_idx in processed_cols:
                continue

            # 현재 열이 병합 범위에 속하는지 확인
            in_merge = False
            for start, end in merge_ranges:
                if start <= col_idx <= end:
                    # 이 범위의 모든 셀 병합
                    merged_parts = []
                    for merge_col in range(start, min(end + 1, len(row))):
                        if merge_col not in processed_cols:
                            cell = row[merge_col]
                            cell_str = str(cell).strip() if cell else ""
                            if cell_str:
                                merged_parts.append(cell_str)
                            processed_cols.add(merge_col)

                    # 공백 없이 연결 (한글 단어 조각 복원)
                    merged_text = "".join(merged_parts)
                    if merged_text:
                        new_row.append(merged_text)

                    in_merge = True
                    break

            if not in_merge:
                cell = row[col_idx]
                new_row.append(cell)
                processed_cols.add(col_idx)

        if new_row:
            result.append(new_row)

    # 결과 검증: 모든 행의 열 수가 동일해야 함
    if result:
        max_cols = max(len(row) for row in result)
        for row in result:
            while len(row) < max_cols:
                row.append("")

    logger.debug(f"Merged columns: {num_cols} -> {len(result[0]) if result else 0}")
    return result


def _extract_all_tables(file_path: str) -> Dict[int, List[PageElement]]:
    """
    모든 페이지에서 테이블을 추출합니다.

    우선순위:
    1. pdfplumber 사용 (더 정확한 테이블 감지)
    2. PyMuPDF 폴백 (pdfplumber 실패 시)

    페이지 간 연속 테이블을 감지하고, rowspan 셀을 복원합니다.
    """
    # 1. pdfplumber로 먼저 시도
    if PDFPLUMBER_AVAILABLE:
        tables_by_page = _extract_tables_with_pdfplumber(file_path)
        if tables_by_page:
            logger.info(f"Using pdfplumber for table extraction (found tables in {len(tables_by_page)} pages)")
            return tables_by_page
        logger.debug("pdfplumber found no tables, falling back to PyMuPDF")

    # 2. PyMuPDF 폴백
    return _extract_all_tables_pymupdf(file_path)


def _extract_all_tables_pymupdf(file_path: str) -> Dict[int, List[PageElement]]:
    """
    PyMuPDF를 사용하여 모든 페이지에서 테이블을 추출합니다.
    (pdfplumber 폴백용)
    """
    tables_by_page: Dict[int, List[PageElement]] = {}

    if not PYMUPDF_AVAILABLE:
        return tables_by_page

    try:
        doc = fitz.open(file_path)

        # 모든 페이지의 테이블 정보를 먼저 수집
        all_table_infos: List[TableContinuationInfo] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_height = page.rect.height

            if not hasattr(page, 'find_tables'):
                logger.debug("PyMuPDF find_tables not available (requires PyMuPDF 1.23.0+)")
                continue

            try:
                # 다양한 전략으로 테이블 찾기
                tables = _find_tables_with_fallback(page)

                if not tables:
                    continue

                for table_idx, table in enumerate(tables):
                    try:
                        table_data = table.extract()

                        if not table_data or not any(any(cell for cell in row if cell) for row in table_data):
                            continue

                        # 과도하게 분리된 열 병합
                        merged_data = _merge_fragmented_columns(table_data)

                        # 테이블 정보 추출
                        table_info = _extract_table_info(table, page_num, table_idx, page_height)
                        # 병합된 데이터로 업데이트
                        table_info = TableContinuationInfo(
                            page_num=table_info.page_num,
                            table_idx=table_info.table_idx,
                            bbox=table_info.bbox,
                            col_count=len(merged_data[0]) if merged_data else table_info.col_count,  # 병합 후 열 수 업데이트
                            x_coords=table_info.x_coords,
                            data=merged_data,  # 병합된 데이터 사용
                            header_rows=table_info.header_rows,
                            table_obj=table_info.table_obj,
                            page_height=table_info.page_height
                        )
                        all_table_infos.append(table_info)

                    except Exception as table_e:
                        logger.warning(f"Error extracting table info {table_idx} from page {page_num}: {table_e}")
                        continue

            except Exception as page_e:
                logger.warning(f"Error finding tables on page {page_num}: {page_e}")
                continue

        # 연속 테이블 감지 및 처리
        processed_tables: List[Tuple[TableContinuationInfo, List[List[Optional[str]]]]] = []

        for i, table_info in enumerate(all_table_infos):
            table_data = table_info.data

            # 이전 테이블과의 연속성 확인 (이전 페이지의 마지막 테이블)
            if i > 0:
                prev_table = all_table_infos[i - 1]

                # 서로 다른 페이지에 있고 연속 테이블인 경우
                if prev_table.page_num < table_info.page_num:
                    if _is_table_continuation(prev_table, table_info):
                        logger.info(f"Detected table continuation: page {prev_table.page_num + 1} -> page {table_info.page_num + 1}")

                        # 빈 셀 채우기
                        table_data = _fill_continuation_cells(
                            table_data,
                            prev_table,
                            table_info.header_rows
                        )

            processed_tables.append((table_info, table_data))

        # HTML 변환 및 PageElement 생성
        for table_info, table_data in processed_tables:
            try:
                # HTML 테이블 생성
                html_table = _convert_table_to_html_with_spans(table_info.table_obj, table_data)

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
                    logger.debug(f"Extracted table {table_info.table_idx + 1} from page {page_num + 1}")

            except Exception as table_e:
                logger.warning(f"Error converting table to HTML: {table_e}")
                continue

        doc.close()

    except Exception as e:
        logger.warning(f"PyMuPDF table extraction error: {e}")

    return tables_by_page


def _convert_table_to_html_with_spans(table, table_data: List[List[Optional[str]]]) -> str:
    """
    PyMuPDF 테이블을 HTML로 변환합니다.
    셀 병합 정보를 처리합니다.

    변환 전략:
    1. 먼저 물리적 셀 bbox 기반으로 rowspan/colspan 계산 시도
    2. 물리적 셀 정보가 없거나 불충분하면, 값 기반 rowspan 감지 사용
    3. 최종 폴백: 기본 HTML (병합 없음)
    """
    if not table_data:
        return ""

    try:
        # 전략 1: PyMuPDF 테이블의 rows 속성으로 물리적 셀 병합 처리
        if hasattr(table, 'rows') and table.rows:
            result = _convert_table_with_cell_spans(table, table_data)
            if result and '<table' in result:
                # 결과 검증: 최소한 rowspan이나 colspan이 있거나 데이터가 있어야 함
                if 'rowspan' in result or 'colspan' in result or '<td>' in result or '<th>' in result:
                    return result

        # 전략 2: 값 기반 rowspan 감지 (동일 값이 연속되는 셀 병합)
        # 이 방법은 첫 번째 열이 카테고리 헤더인 테이블에 효과적
        result = _convert_table_to_html_with_value_spans(table_data)
        if result and '<table' in result:
            return result

        # 전략 3: 기본 HTML 생성 (병합 없음)
        return _convert_table_to_html_basic(table_data)

    except Exception as e:
        logger.warning(f"Table HTML conversion error: {e}")
        return _convert_table_to_html_basic(table_data)


def _normalize_y_coordinates(coords: List[float], min_gap: float = 12.0) -> List[float]:
    """
    Y 좌표들을 정규화합니다.

    PDF 테이블에서 헤더 영역의 셀들이 미세하게 다른 Y 좌표를 가지는 경우가 있습니다.
    이 함수는 min_gap 이하 간격의 인접 좌표들을 하나의 그리드 라인으로 병합합니다.

    Args:
        coords: 정렬할 Y 좌표 리스트
        min_gap: 병합할 최소 간격 (이 값 미만이면 같은 그리드 라인으로 취급)

    Returns:
        정규화된 Y 좌표 리스트
    """
    if not coords:
        return []

    sorted_coords = sorted(coords)
    normalized = [sorted_coords[0]]

    for coord in sorted_coords[1:]:
        if coord - normalized[-1] >= min_gap:
            normalized.append(coord)

    return normalized


def _convert_table_with_cell_spans(table, table_data: List[List[Optional[str]]]) -> str:
    """
    PyMuPDF 테이블을 HTML로 변환 (셀 병합 정보 포함).

    물리적 셀 기반 알고리즘:
    1. table.rows에서 물리적 셀(bbox != None)만 추출
    2. 동일 bbox를 가진 중복 셀 제거 및 텍스트 병합
    3. Y 좌표 정규화로 분산된 헤더 영역 처리
    4. 점유 매트릭스 기반 HTML 생성

    Args:
        table: PyMuPDF Table 객체 (rows 속성 필요)
        table_data: table.extract() 결과

    Returns:
        HTML 테이블 문자열
    """
    # table.rows 속성 확인
    if not hasattr(table, 'rows') or not table.rows:
        logger.debug("Table has no 'rows' attribute, using basic conversion")
        return _convert_table_to_html_basic(table_data)

    try:
        # 1. 물리적 셀 수집 (중복 제거 + 텍스트 병합)
        unique_cells = {}
        for row_idx, row in enumerate(table.rows):
            if not hasattr(row, 'cells'):
                continue

            for col_idx, cell_bbox in enumerate(row.cells):
                if cell_bbox is None:
                    # None = 병합된 위치, 건너뛰기
                    continue

                x0, y0, x1, y1 = cell_bbox[:4]
                key = (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))

                # 해당 위치의 텍스트
                text = ""
                if row_idx < len(table_data) and col_idx < len(table_data[row_idx]):
                    text = table_data[row_idx][col_idx] or ""

                if key not in unique_cells:
                    unique_cells[key] = {
                        'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                        'text': text
                    }
                else:
                    # 텍스트 병합 (빈 텍스트보다 실제 텍스트 선호)
                    if text and not unique_cells[key]['text']:
                        unique_cells[key]['text'] = text
                    elif text and unique_cells[key]['text'] and text not in unique_cells[key]['text']:
                        unique_cells[key]['text'] += '\n' + text

        cells_list = list(unique_cells.values())

        if not cells_list:
            logger.debug("No physical cells found, using basic conversion")
            return _convert_table_to_html_basic(table_data)

        logger.debug(f"Collected {len(cells_list)} unique physical cells")

        # 2. X, Y 좌표 수집
        raw_x = set()
        raw_y = set()
        for c in cells_list:
            raw_x.add(c['x0'])
            raw_x.add(c['x1'])
            raw_y.add(c['y0'])
            raw_y.add(c['y1'])

        # 3. 좌표 그리드 구축
        # X 좌표는 정확히 유지 (컬럼 구조가 명확)
        all_x = sorted(raw_x)
        # Y 좌표는 정규화 (헤더 영역의 분산된 Y 처리)
        all_y = _normalize_y_coordinates(list(raw_y), min_gap=12.0)

        logger.debug(f"X grid: {len(all_x)} points")
        logger.debug(f"Y grid: {len(all_y)} points (normalized from {len(raw_y)})")

        if len(all_x) < 2 or len(all_y) < 2:
            logger.debug("Grid too small, using basic conversion")
            return _convert_table_to_html_basic(table_data)

        # 그리드 인덱스 찾기 함수
        def find_x_index(val: float) -> int:
            """X 좌표에서 가장 가까운 그리드 인덱스 찾기"""
            for i, x in enumerate(all_x):
                if abs(x - val) < 1.0:
                    return i
            # 가장 가까운 값 찾기
            return min(range(len(all_x)), key=lambda i: abs(all_x[i] - val))

        def find_y_index(val: float) -> int:
            """Y 좌표에서 가장 가까운 그리드 인덱스 찾기 (정규화된 그리드 사용)"""
            min_dist = float('inf')
            min_idx = 0
            for i, y in enumerate(all_y):
                dist = abs(y - val)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            return min_idx

        # 4. 각 셀의 그리드 위치 및 span 계산
        for cell in cells_list:
            cell['col_start'] = find_x_index(cell['x0'])
            cell['col_end'] = find_x_index(cell['x1'])
            cell['row_start'] = find_y_index(cell['y0'])
            cell['row_end'] = find_y_index(cell['y1'])

            # row_end가 row_start보다 같거나 작으면 최소 1칸
            if cell['row_end'] <= cell['row_start']:
                cell['row_end'] = cell['row_start'] + 1

            cell['colspan'] = max(1, cell['col_end'] - cell['col_start'])
            cell['rowspan'] = max(1, cell['row_end'] - cell['row_start'])

        # 5. 그리드 크기 계산
        num_grid_rows = len(all_y) - 1
        num_grid_cols = len(all_x) - 1

        logger.debug(f"Grid dimensions: {num_grid_rows} rows x {num_grid_cols} cols")

        # 6. 점유 맵 생성 (어떤 셀이 어느 그리드 위치를 차지하는지)
        occupied = [[None] * num_grid_cols for _ in range(num_grid_rows)]

        for cell in cells_list:
            for r in range(cell['row_start'], min(cell['row_end'], num_grid_rows)):
                for c in range(cell['col_start'], min(cell['col_end'], num_grid_cols)):
                    occupied[r][c] = cell

        # 7. HTML 생성
        html_parts = ["<table border='1'>"]
        processed_cells = set()  # 이미 HTML로 출력한 셀

        for grid_row in range(num_grid_rows):
            row_cells = []
            grid_col = 0

            while grid_col < num_grid_cols:
                cell = occupied[grid_row][grid_col]

                if cell is None:
                    # 빈 셀
                    tag = "th" if grid_row == 0 else "td"
                    row_cells.append(f'<{tag}></{tag}>')
                    grid_col += 1
                    continue

                cell_id = id(cell)

                # 이 셀의 시작 위치인지 확인
                if cell['row_start'] == grid_row and cell['col_start'] == grid_col:
                    # 처음 등장하는 셀 - HTML 태그 생성
                    if cell_id not in processed_cells:
                        processed_cells.add(cell_id)

                        # HTML 이스케이프
                        cell_text = str(cell['text']).strip() if cell['text'] else ""
                        cell_text = cell_text.replace("&", "&amp;")
                        cell_text = cell_text.replace("<", "&lt;")
                        cell_text = cell_text.replace(">", "&gt;")
                        cell_text = cell_text.replace("\n", "<br>")

                        # 첫 행은 헤더
                        tag = "th" if grid_row == 0 else "td"

                        attrs = []
                        if cell['rowspan'] > 1:
                            attrs.append(f"rowspan='{cell['rowspan']}'")
                        if cell['colspan'] > 1:
                            attrs.append(f"colspan='{cell['colspan']}'")

                        attr_str = ' ' + ' '.join(attrs) if attrs else ''
                        row_cells.append(f'<{tag}{attr_str}>{cell_text}</{tag}>')

                    grid_col += cell['colspan']
                else:
                    # 이미 상위 행 또는 왼쪽 셀에서 처리됨 - 건너뜀
                    grid_col += 1

            if row_cells:
                html_parts.append('<tr>' + ''.join(row_cells) + '</tr>')

        html_parts.append('</table>')

        result_html = '\n'.join(html_parts)
        logger.debug(f"Generated HTML table with {len(processed_cells)} cells")

        return result_html

    except Exception as e:
        logger.warning(f"Cell span conversion error: {e}, falling back to basic")
        return _convert_table_to_html_basic(table_data)


def _convert_table_to_html_basic(table_data: List[List[Optional[str]]]) -> str:
    """
    테이블 데이터를 기본 HTML 형식으로 변환합니다 (병합 없음).
    """
    if not table_data:
        return ""

    html_parts = ["<table border='1'>"]

    for row_idx, row in enumerate(table_data):
        if not row:
            continue

        html_parts.append("<tr>")

        for cell in row:
            cell_text = str(cell).strip() if cell else ""
            cell_text = cell_text.replace("&", "&amp;")
            cell_text = cell_text.replace("<", "&lt;")
            cell_text = cell_text.replace(">", "&gt;")
            cell_text = cell_text.replace("\n", "<br>")

            tag = "th" if row_idx == 0 else "td"
            html_parts.append(f"<{tag}>{cell_text}</{tag}>")

        html_parts.append("</tr>")

    html_parts.append("</table>")

    return "\n".join(html_parts)


def _convert_table_to_html_with_value_spans(table_data: List[List[Optional[str]]], max_merge_cols: int = 2) -> str:
    """
    테이블 데이터에서 연속으로 동일한 값을 가진 셀을 rowspan으로 변환합니다.

    주의: 카테고리 열(첫 1-2개 열)에만 rowspan을 적용합니다.
    숫자 데이터 열에는 적용하지 않습니다.

    이 함수는 PyMuPDF가 셀 병합을 감지하지 못했을 때 사용됩니다.
    연속으로 동일한 값이 나타나는 첫 번째 열(헤더 열)의 셀만 병합합니다.

    예시:
    입력: [["미래한정 표현", "영원히"], ["미래한정 표현", "전무후무"], ["허황/과장 표현", "우주 최고의"]]
    출력: <table> with rowspan='2' for "미래한정 표현"

    Args:
        table_data: 테이블 데이터
        max_merge_cols: rowspan을 적용할 최대 열 수 (기본: 첫 2개 열만)

    Returns:
        HTML 테이블 문자열
    """
    if not table_data or len(table_data) == 0:
        return ""

    num_rows = len(table_data)
    num_cols = max(len(row) for row in table_data) if table_data else 0

    if num_cols == 0:
        return ""

    # 병합 가능한 열 감지 (카테고리 열만)
    mergeable_cols = _detect_mergeable_columns(table_data, max_merge_cols)

    # 각 셀에 대해 rowspan 값 계산
    rowspan_map = [[1 for _ in range(num_cols)] for _ in range(num_rows)]
    skip_map = [[False for _ in range(num_cols)] for _ in range(num_rows)]

    # 병합 가능한 열에서만 연속 동일 값 찾기
    for col_idx in mergeable_cols:
        row_idx = 1  # 헤더 행(0) 스킵
        while row_idx < num_rows:
            current_value = table_data[row_idx][col_idx] if col_idx < len(table_data[row_idx]) else ""

            if not current_value or not str(current_value).strip():
                row_idx += 1
                continue

            # 숫자 값은 병합하지 않음
            if _is_numeric_value(current_value):
                row_idx += 1
                continue

            # 현재 값과 동일한 연속 셀 수 계산
            span_count = 1
            for next_row in range(row_idx + 1, num_rows):
                next_value = table_data[next_row][col_idx] if col_idx < len(table_data[next_row]) else ""

                if str(next_value).strip() == str(current_value).strip():
                    span_count += 1
                else:
                    break

            # rowspan > 1이면 적용
            if span_count > 1:
                rowspan_map[row_idx][col_idx] = span_count
                for skip_row in range(row_idx + 1, row_idx + span_count):
                    skip_map[skip_row][col_idx] = True

            row_idx += span_count

    # HTML 생성
    html_parts = ["<table border='1'>"]

    for row_idx, row in enumerate(table_data):
        row_cells = []

        for col_idx in range(num_cols):
            # 건너뛸 셀인지 확인
            if skip_map[row_idx][col_idx]:
                continue

            cell_value = row[col_idx] if col_idx < len(row) else ""
            cell_text = str(cell_value).strip() if cell_value else ""
            cell_text = cell_text.replace("&", "&amp;")
            cell_text = cell_text.replace("<", "&lt;")
            cell_text = cell_text.replace(">", "&gt;")
            cell_text = cell_text.replace("\n", "<br>")

            tag = "th" if row_idx == 0 else "td"
            rowspan = rowspan_map[row_idx][col_idx]

            if rowspan > 1:
                row_cells.append(f"<{tag} rowspan='{rowspan}'>{cell_text}</{tag}>")
            else:
                row_cells.append(f"<{tag}>{cell_text}</{tag}>")

        if row_cells:
            html_parts.append("<tr>" + "".join(row_cells) + "</tr>")

    html_parts.append("</table>")

    return "\n".join(html_parts)


def _detect_mergeable_columns(table_data: List[List[Optional[str]]], max_cols: int = 2) -> List[int]:
    """
    rowspan 병합이 적절한 열을 감지합니다.

    병합 가능 조건:
    - 텍스트 값 위주 (숫자 열은 제외)
    - 연속 동일 값이 있음 (실제 rowspan 패턴)
    - 테이블 왼쪽에 위치 (첫 1-2개 열)

    Args:
        table_data: 테이블 데이터
        max_cols: 검사할 최대 열 수

    Returns:
        병합 가능한 열 인덱스 리스트
    """
    if not table_data or len(table_data) < 2:
        return []

    num_rows = len(table_data)
    num_cols = max(len(row) for row in table_data) if table_data else 0

    mergeable_cols = []

    for col_idx in range(min(max_cols, num_cols)):
        numeric_count = 0
        text_count = 0
        has_consecutive_same = False

        prev_value = None
        for row_idx in range(1, num_rows):  # 헤더 스킵
            if col_idx >= len(table_data[row_idx]):
                continue

            cell_value = table_data[row_idx][col_idx]
            if cell_value is None or (isinstance(cell_value, str) and not cell_value.strip()):
                continue

            # 숫자 여부 체크
            if _is_numeric_value(cell_value):
                numeric_count += 1
            else:
                text_count += 1

            # 연속 동일 값 체크
            if prev_value and str(cell_value).strip() == str(prev_value).strip():
                has_consecutive_same = True

            prev_value = cell_value

        total = numeric_count + text_count
        if total == 0:
            continue

        # 병합 가능 조건: 텍스트 위주 + 연속 동일 값 있음
        text_ratio = text_count / total
        if text_ratio >= 0.5 and has_consecutive_same:
            mergeable_cols.append(col_idx)
            logger.debug(f"Column {col_idx} is mergeable (text ratio: {text_ratio:.1%})")

    return mergeable_cols


def _is_numeric_value(value) -> bool:
    """
    값이 숫자인지 확인합니다.
    """
    if value is None:
        return False

    if isinstance(value, (int, float)):
        return True

    if isinstance(value, str):
        clean_val = value.strip().replace(',', '').replace('.', '').replace('-', '').replace('+', '')
        return clean_val.isdigit() if clean_val else False

    return False


# === 요소 병합 ===

def _merge_page_elements(elements: List[PageElement]) -> str:
    """
    요소들을 병합하여 최종 텍스트를 생성합니다.

    테이블 영역과 겹치는 텍스트는 제거합니다 (중복 방지).
    """
    if not elements:
        return ""

    # 테이블 영역 수집 (여유 마진 포함)
    table_bboxes = []
    table_bboxes_expanded = []  # 확장된 bbox (마진 포함)

    for e in elements:
        if e.element_type == ElementType.TABLE:
            table_bboxes.append(e.bbox)
            # 테이블 bbox를 상하좌우로 확장 (경계 텍스트 포함)
            x0, y0, x1, y1 = e.bbox
            margin = 5.0  # 5 포인트 마진
            table_bboxes_expanded.append((x0 - margin, y0 - margin, x1 + margin, y1 + margin))

    # 나란히 있는 테이블들을 그룹화하여 전체 영역 계산
    table_group_bboxes = _group_adjacent_tables(table_bboxes)

    result_parts = []

    for element in elements:
        if element.element_type == ElementType.TABLE:
            # 테이블은 그대로 추가
            result_parts.append("\n" + element.content + "\n")

        elif element.element_type == ElementType.IMAGE:
            # 이미지는 그대로 추가
            result_parts.append(element.content)

        elif element.element_type == ElementType.TEXT:
            # 텍스트가 테이블 영역과 겹치는지 확인
            if _is_overlapping_with_tables(element.bbox, table_bboxes_expanded):
                # 테이블과 겹치면 건너뛰기 (테이블에서 이미 추출됨)
                logger.debug(f"Skipping text overlapping with table: {element.content[:50]}...")
                continue

            # 추가 검사: 테이블 그룹 영역 내부에 있는지 (나란히 있는 테이블 사이 텍스트)
            if _is_inside_table_group(element.bbox, table_group_bboxes):
                logger.debug(f"Skipping text inside table group: {element.content[:50]}...")
                continue

            # 추가 검사: 테이블과 같은 Y 범위에 있고, 테이블 내용과 유사한지
            if _is_likely_table_text(element, table_bboxes):
                logger.debug(f"Skipping likely table text: {element.content[:50]}...")
                continue

            result_parts.append(element.content + "\n")

    return "".join(result_parts)


def _group_adjacent_tables(
    table_bboxes: List[Tuple[float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    """
    Y 좌표가 겹치는 테이블들을 그룹화하여 전체 bbox를 반환합니다.

    나란히 배치된 테이블들(같은 행에 있는 테이블들)을 하나의 그룹으로 묶어
    그룹 전체를 포함하는 bbox를 계산합니다.

    예: 테이블A(x:50-200, y:100-300)와 테이블B(x:250-400, y:100-300)가 있으면
        그룹 bbox는 (50, 100, 400, 300)이 됩니다.

    Args:
        table_bboxes: 테이블 bbox 리스트

    Returns:
        그룹화된 테이블 영역 bbox 리스트
    """
    if not table_bboxes:
        return []

    if len(table_bboxes) == 1:
        return list(table_bboxes)

    # Y 범위가 겹치는 테이블들을 그룹화
    groups = []
    used = set()

    for i, bbox1 in enumerate(table_bboxes):
        if i in used:
            continue

        # 현재 테이블과 Y가 겹치는 테이블들 찾기
        group = [bbox1]
        used.add(i)

        x0_min, y0_min, x1_max, y1_max = bbox1

        for j, bbox2 in enumerate(table_bboxes):
            if j in used:
                continue

            # Y 범위가 겹치는지 확인
            by0, by1 = bbox2[1], bbox2[3]
            y_overlap = not (by1 < y0_min or by0 > y1_max)

            if y_overlap:
                group.append(bbox2)
                used.add(j)

                # 그룹 bbox 업데이트
                x0_min = min(x0_min, bbox2[0])
                y0_min = min(y0_min, bbox2[1])
                x1_max = max(x1_max, bbox2[2])
                y1_max = max(y1_max, bbox2[3])

        # 그룹의 전체 bbox 추가
        groups.append((x0_min, y0_min, x1_max, y1_max))

    return groups


def _is_inside_table_group(
    text_bbox: Tuple[float, float, float, float],
    group_bboxes: List[Tuple[float, float, float, float]]
) -> bool:
    """
    텍스트가 테이블 그룹 영역 내부에 있는지 확인합니다.

    나란히 있는 테이블들 사이의 화살표나 구분자를 필터링합니다.
    """
    tx0, ty0, tx1, ty1 = text_bbox

    for gx0, gy0, gx1, gy1 in group_bboxes:
        # 텍스트가 그룹 영역 내부에 있는지 (약간의 마진 포함)
        margin = 10.0
        if (tx0 >= gx0 - margin and tx1 <= gx1 + margin and
            ty0 >= gy0 - margin and ty1 <= gy1 + margin):
            return True

    return False


def _is_likely_table_text(
    text_element: 'PageElement',
    table_bboxes: List[Tuple[float, float, float, float]]
) -> bool:
    """
    텍스트가 테이블에서 추출된 것인지 추가 판단합니다.

    테이블과 같은 Y 범위에 있고, 내용이 짧거나 특수문자만 있으면
    테이블 관련 텍스트로 판단합니다.
    """
    if not table_bboxes:
        return False

    tx0, ty0, tx1, ty1 = text_element.bbox
    text_content = text_element.content.strip()

    # 빈 텍스트는 건너뛰기
    if not text_content:
        return True

    for bx0, by0, bx1, by1 in table_bboxes:
        # Y 범위가 테이블과 겹치는지 확인
        y_overlap = not (ty1 < by0 or ty0 > by1)

        if y_overlap:
            # X가 테이블 범위 내이거나 바로 옆에 있는지
            x_near_table = (tx0 >= bx0 - 50 and tx1 <= bx1 + 50)

            if x_near_table:
                # 짧은 텍스트(화살표, 구분자 등)는 테이블 관련으로 처리
                # 특수문자만 있거나, 줄바꿈 없이 짧은 텍스트
                lines = text_content.split('\n')

                # 각 줄이 매우 짧은 단어들의 나열이면 (테이블 셀 텍스트 중복)
                is_short_fragments = all(len(line.strip()) < 20 for line in lines if line.strip())

                # 화살표, 구분자 등 특수문자 패턴
                special_chars = {'⇒', '→', '↓', '↑', '◆', '●', '○', '■', '□', '※'}
                is_special = any(c in text_content for c in special_chars)

                if is_special or (is_short_fragments and len(lines) > 2):
                    return True

    return False


def _is_overlapping_with_tables(
    text_bbox: Tuple[float, float, float, float],
    table_bboxes: List[Tuple[float, float, float, float]],
    overlap_threshold: float = 0.2  # 20%로 낮춤 (더 적극적으로 필터링)
) -> bool:
    """
    텍스트 영역이 테이블 영역과 겹치는지 확인합니다.

    overlap_threshold를 낮춰서 조금만 겹쳐도 테이블 텍스트로 판단합니다.
    """
    tx0, ty0, tx1, ty1 = text_bbox
    text_area = max((tx1 - tx0) * (ty1 - ty0), 1)

    for table_bbox in table_bboxes:
        bx0, by0, bx1, by1 = table_bbox

        # 겹치는 영역 계산
        ix0 = max(tx0, bx0)
        iy0 = max(ty0, by0)
        ix1 = min(tx1, bx1)
        iy1 = min(ty1, by1)

        if ix0 < ix1 and iy0 < iy1:
            intersection_area = (ix1 - ix0) * (iy1 - iy0)
            overlap_ratio = intersection_area / text_area

            if overlap_ratio > overlap_threshold:
                return True

        # 추가: 텍스트가 테이블 bbox 완전히 내부에 있는지 확인
        # (작은 텍스트가 큰 테이블 안에 있는 경우)
        if tx0 >= bx0 and ty0 >= by0 and tx1 <= bx1 and ty1 <= by1:
            return True

    return False
