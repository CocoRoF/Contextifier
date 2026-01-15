# service/document_processor/processor/pdf_handler_v3.py
"""
PDF Handler V3 - 최고 수준의 PDF 테이블 처리기

=============================================================================
V3 핵심 목표:
=============================================================================
1. 얇은 테두리 선 처리 - 0.1pt ~ 1pt 선도 정확히 인식
2. 불완전 테두리 복구 - 상단/좌우 열린 테이블 복구
3. 상이한 선 두께 처리 - 테두리와 내부선 두께가 다른 경우
4. 이중선 처리 - 디자인적 이중선을 단일 구분선으로 처리
5. 주석/각주/미주 통합 - 테이블 관련 텍스트 요소 포함
6. 복잡한 병합셀 정밀 처리 - 물리적 bbox 기반 rowspan/colspan

=============================================================================
V3 아키텍처:
=============================================================================
┌─────────────────────────────────────────────────────────────────────────┐
│                         PDF Document Input                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 1: Line Analysis Engine                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Thin Lines    │  │ Double Lines  │  │ Incomplete    │                │
│  │ Detection     │  │ Merger        │  │ Border Fix    │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 2: Table Detection Engine                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ PyMuPDF       │  │ pdfplumber    │  │ Line-Based    │                │
│  │ Strategy      │  │ Strategy      │  │ Strategy      │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
│                    ↓ Confidence Scoring & Selection ↓                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 3: Cell Analysis Engine                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Physical Cell │  │ Text Position │  │ Merge Cell    │                │
│  │ Detection     │  │ Analysis      │  │ Calculation   │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 4: Annotation Integration                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │ Footnote      │  │ Endnote       │  │ Table Note    │                │
│  │ Detection     │  │ Detection     │  │ Integration   │                │
│  └───────────────┘  └───────────────┘  └───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Phase 5: HTML Generation                              │
│  ┌───────────────────────────────────────────────────────────┐          │
│  │ Semantic HTML with rowspan/colspan/accessibility          │          │
│  └───────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘

=============================================================================
핵심 알고리즘:
=============================================================================
1. 선 분석 (Line Analysis):
   - drawings/rects에서 모든 선 추출
   - 선 두께별 분류 (thin < 0.5pt, normal 0.5-2pt, thick > 2pt)
   - 인접 이중선 병합 (간격 < 5pt)
   - 불완전 테두리 복구 (3면 이상 존재시 4면 완성)

2. 테이블 감지 (Table Detection):
   - Strategy 1: PyMuPDF find_tables() - 신뢰도 점수 계산
   - Strategy 2: pdfplumber - 신뢰도 점수 계산
   - Strategy 3: 선 분석 기반 그리드 구성 - 신뢰도 점수 계산
   - 최고 신뢰도 전략 선택 또는 결과 병합

3. 셀 분석 (Cell Analysis):
   - 물리적 셀 bbox 추출
   - 그리드 라인 매핑 (tolerance 기반)
   - rowspan/colspan 정밀 계산
   - 텍스트 위치 기반 병합 검증

4. 주석 통합 (Annotation Integration):
   - 테이블 직후 주석행 감지 (예: "주) ...")
   - 각주/미주 텍스트 수집
   - 테이블 데이터에 적절히 통합
"""
import logging
import copy
import traceback
import math
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from libs.core.processor.pdf_helpers.pdf_helper import (
    upload_image_to_minio,
    extract_pdf_metadata,
    format_metadata,
    escape_html,
    is_inside_any_bbox,
    find_image_position,
    get_text_lines_with_positions,
)

# V3.1 모듈화된 컴포넌트 import
from libs.core.processor.pdf_helpers.v3_types import (
    TableDetectionStrategy as TableDetectionStrategyType,
    ElementType,
    V3Config as V3ConfigBase,
    LineInfo,
    GridInfo,
    CellInfo,
    PageElement,
    PageBorderInfo,
)
from libs.core.processor.pdf_helpers.vector_text_ocr import (
    VectorTextOCREngine,
)
from libs.core.processor.pdf_helpers.table_detection import (
    TableDetectionEngine,
)
from libs.core.processor.pdf_helpers.cell_analysis import (
    CellAnalysisEngine,
)
from libs.core.processor.pdf_helpers.text_quality_analyzer import (
    TextQualityAnalyzer,
    QualityAwareTextExtractor,
    TextQualityConfig,
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

# pytesseract import (for outlined/vector text OCR)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available - outlined text OCR will be disabled")


# ============================================================================
# V3 설정 확장 (v3_types.V3Config 기반)
# ============================================================================

class V3Config(V3ConfigBase):
    """V3 설정 상수 - 기본값 + 추가 설정"""
    # 선 분석
    THIN_LINE_THRESHOLD = 0.5      # pt
    THICK_LINE_THRESHOLD = 2.0     # pt
    DOUBLE_LINE_GAP = 5.0          # pt - 이중선으로 판단하는 최대 간격
    LINE_MERGE_TOLERANCE = 3.0     # pt - 같은 위치로 판단하는 허용 오차

    # 테이블 감지 추가 설정
    MIN_CELL_SIZE = 10.0           # pt - 최소 셀 크기
    PAGE_BORDER_MARGIN = 0.1       # 페이지 크기 대비 테두리 마진 비율
    PAGE_SPANNING_RATIO = 0.85     # 페이지를 가로지르는 것으로 판단하는 비율

    # 불완전 테두리 복구
    BORDER_EXTENSION_MARGIN = 20.0  # pt - 테두리 연장 시 마진
    INCOMPLETE_BORDER_MIN_SIDES = 3  # 불완전 테두리로 판단하는 최소 변 수

    # 주석 감지
    ANNOTATION_Y_MARGIN = 30.0     # pt - 테이블 하단에서 주석 탐색 범위
    ANNOTATION_PATTERNS = ['주)', '주 )', '※', '*', '†', '‡', '¹', '²', '³']

    # 벡터 텍스트 OCR 설정 (Outlined Text / Path Text)
    VECTOR_TEXT_MIN_ITEMS = 20     # 벡터 텍스트로 판단하는 최소 drawing items 수
    VECTOR_TEXT_MAX_HEIGHT = 30.0  # pt - 벡터 텍스트로 판단하는 최대 높이
    VECTOR_TEXT_OCR_DPI = 300      # OCR용 이미지 렌더링 DPI
    VECTOR_TEXT_OCR_SCALE = 4      # OCR용 이미지 확대 배율
    VECTOR_TEXT_OCR_LANG = 'kor+eng'  # Tesseract 언어 설정

    # 그리드 규칙성 검증 (Grid Regularity Validation)
    GRID_VARIANCE_THRESHOLD = 0.5          # 셀 크기 분산 임계값 (낮을수록 규칙적)
    GRID_MIN_ORTHOGONAL_RATIO = 0.7        # 직교선(수평/수직) 최소 비율

    # 이미지/일러스트 영역 보호
    IMAGE_AREA_MARGIN = 5.0               # 이미지 주변 마진 (pt)


# Enum aliases for backward compatibility
# Enum aliases for backward compatibility
TableDetectionStrategy = TableDetectionStrategyType


# ============================================================================
# V3 전용 타입 정의 (기존 코드와의 호환성을 위해)
# ============================================================================

@dataclass
class TableCandidateV3:
    """테이블 후보 - V3 내부 사용 (확장된 버전)"""
    strategy: TableDetectionStrategy
    confidence: float
    bbox: Tuple[float, float, float, float]
    grid: Optional[GridInfo]
    cells: List[CellInfo]
    data: List[List[Optional[str]]]
    raw_table: Any = None  # 원본 테이블 객체

    @property
    def row_count(self) -> int:
        return len(self.data)

    @property
    def col_count(self) -> int:
        return max(len(row) for row in self.data) if self.data else 0


@dataclass
class AnnotationInfoV3:
    """주석/각주/미주 정보 - V3 내부 사용"""
    text: str
    bbox: Tuple[float, float, float, float]
    type: str  # 'footnote', 'endnote', 'table_note'
    related_table_idx: Optional[int] = None


@dataclass
class PageElementV3(PageElement):
    """페이지 내 요소 - V3 확장"""

    @property
    def sort_key(self) -> Tuple[float, float]:
        """정렬 키: (y0, x0)"""
        return (self.bbox[1], self.bbox[0])


@dataclass
class TableInfo:
    """최종 테이블 정보"""
    page_num: int
    table_idx: int
    bbox: Tuple[float, float, float, float]
    data: List[List[Optional[str]]]
    col_count: int
    row_count: int
    page_height: float
    cells_info: Optional[List[Dict]] = None
    annotations: Optional[List[AnnotationInfoV3]] = None
    detection_strategy: Optional[TableDetectionStrategy] = None
    confidence: float = 1.0


# ============================================================================
# 메인 함수
# ============================================================================

async def extract_text_from_pdf_v3(
    file_path: str,
    current_config: Dict[str, Any] = None,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    PDF 텍스트 추출 V3 (최고 수준의 테이블 처리).

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

    logger.info(f"[PDF-V3] Processing: {file_path}")
    return await _extract_pdf_enhanced_v3(file_path, current_config, app_db, extract_default_metadata)


# ============================================================================
# 핵심 처리 로직
# ============================================================================

async def _extract_pdf_enhanced_v3(
    file_path: str,
    current_config: Dict[str, Any],
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    고도화된 PDF 처리 V3.

    처리 순서:
    1. 문서 열기 및 메타데이터 추출
    2. 각 페이지에 대해:
       a. 선 분석 (Line Analysis Engine)
       b. 테이블 감지 (Table Detection Engine - 다중 전략)
       c. 셀 분석 (Cell Analysis Engine)
       d. 주석 통합 (Annotation Integration)
       e. 텍스트/이미지 추출
    3. 페이지 간 테이블 연속성 처리
    4. 최종 HTML 생성 및 통합
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

        # 전체 문서 테이블 추출 (V3 엔진 사용)
        all_tables = _extract_all_tables_v3(doc, file_path)

        # 페이지별 처리
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_elements: List[PageElement] = []

            logger.debug(f"[PDF-V3] Processing page {page_num + 1}")

            # 1. 페이지 테두리 분석
            border_info = _detect_page_border_v3(page)

            # 1.5. 벡터 텍스트(Outlined/Path Text) 감지 및 OCR
            vector_text_engine = VectorTextOCREngine(page, page_num)
            vector_text_regions = vector_text_engine.detect_and_extract()

            # 벡터 텍스트 OCR 결과를 PageElement로 변환하여 추가
            for region in vector_text_regions:
                if region.ocr_text and region.confidence > 0.3:
                    page_elements.append(PageElement(
                        element_type=ElementType.TEXT,
                        content=region.ocr_text,
                        bbox=region.bbox,
                        page_num=page_num
                    ))

            # 2. 해당 페이지의 테이블 가져오기
            page_tables = all_tables.get(page_num, [])

            for table_element in page_tables:
                page_elements.append(table_element)

            # 3. 테이블 영역 계산 (텍스트 필터링용)
            table_bboxes = [elem.bbox for elem in page_tables]

            # 4. 텍스트 추출 (테이블 영역 제외)
            text_elements = _extract_text_blocks_v3(page, page_num, table_bboxes, border_info)
            page_elements.extend(text_elements)

            # 5. 이미지 추출
            image_elements = await _extract_images_from_page_v3(
                page, page_num, doc, app_db, processed_images, table_bboxes
            )
            page_elements.extend(image_elements)

            # 6. 요소 정렬 및 병합
            page_text = _merge_page_elements_v3(page_elements)
            if page_text.strip():
                all_pages_text.append(f"<Page {page_num + 1}>\n{page_text}\n</Page {page_num + 1}>")

        doc.close()

        final_text = "\n\n".join(all_pages_text)
        logger.info(f"[PDF-V3] Extracted {len(final_text)} chars from {file_path}")

        return final_text

    except Exception as e:
        logger.error(f"[PDF-V3] Error processing {file_path}: {e}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# 테이블 추출 함수
# ============================================================================

def _extract_all_tables_v3(doc, file_path: str) -> Dict[int, List[PageElement]]:
    """
    문서 전체에서 테이블을 추출합니다 (V3).

    전략:
    1. 다중 전략 테이블 감지
    2. 신뢰도 기반 최선의 결과 선택
    3. 셀 분석 및 병합셀 처리
    4. 주석 통합
    5. 페이지 간 연속성 처리
    """
    tables_by_page: Dict[int, List[PageElement]] = {}
    all_table_infos: List[TableInfo] = []

    # 1단계: 각 페이지에서 테이블 감지
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height

        # 페이지 테두리 감지
        border_info = _detect_page_border_v3(page)

        try:
            # 테이블 감지 엔진 사용
            detection_engine = TableDetectionEngine(page, page_num, file_path)
            candidates = detection_engine.detect_tables()

            for idx, candidate in enumerate(candidates):
                # 페이지 테두리와 겹치는지 확인
                if border_info.has_border and _is_table_likely_border_v3(candidate.bbox, border_info, page):
                    logger.debug(f"[PDF-V3] Skipping page border table: {candidate.bbox}")
                    continue

                # 셀 정보를 딕셔너리로 변환
                cells_info = None
                if candidate.cells:
                    cells_info = [
                        {
                            'row': cell.row,
                            'col': cell.col,
                            'rowspan': cell.rowspan,
                            'colspan': cell.colspan,
                            'bbox': cell.bbox
                        }
                        for cell in candidate.cells
                    ]

                table_info = TableInfo(
                    page_num=page_num,
                    table_idx=idx,
                    bbox=candidate.bbox,
                    data=candidate.data,
                    col_count=candidate.col_count,
                    row_count=candidate.row_count,
                    page_height=page_height,
                    cells_info=cells_info,
                    detection_strategy=candidate.strategy,
                    confidence=candidate.confidence
                )

                all_table_infos.append(table_info)

        except Exception as e:
            logger.debug(f"[PDF-V3] Error detecting tables on page {page_num}: {e}")
            continue

    # 2단계: 인접 테이블 병합
    merged_tables = _merge_adjacent_tables_v3(all_table_infos)

    # 3단계: 주석 행 찾아서 삽입
    merged_tables = _find_and_insert_annotations_v3(doc, merged_tables)

    # 4단계: 테이블 연속성 처리
    processed_tables = _process_table_continuity_v3(merged_tables)

    # 5단계: HTML 변환 및 PageElement 생성
    for table_info in processed_tables:
        try:
            html_table = _convert_table_to_html_v3(table_info)

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
            logger.debug(f"[PDF-V3] Error converting table to HTML: {e}")
            continue

    logger.info(f"[PDF-V3] Extracted tables from {len(tables_by_page)} pages")
    return tables_by_page


# ============================================================================
# Phase 4: Annotation Integration
# ============================================================================

def _find_and_insert_annotations_v3(doc, tables: List[TableInfo]) -> List[TableInfo]:
    """
    테이블 내부 및 직후에 주석/각주/미주를 찾아서 통합합니다.

    감지 패턴:
    1. 테이블 직후 "주)" 등으로 시작하는 행
    2. 테이블 내부의 서브헤더 행 (예: (A), (B))
    3. 각주/미주 표시 (※, *, †, ‡ 등)
    """
    if not tables:
        return tables

    result = []
    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)

    for table in tables:
        tables_by_page[table.page_num].append(table)

    for page_num, page_tables in tables_by_page.items():
        page = doc[page_num]
        page_height = page.rect.height

        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])
        text_lines = get_text_lines_with_positions(page)

        for i, table in enumerate(sorted_tables):
            table_top = table.bbox[1]
            table_bottom = table.bbox[3]
            table_left = table.bbox[0]
            table_right = table.bbox[2]

            next_table_top = sorted_tables[i + 1].bbox[1] if i + 1 < len(sorted_tables) else page_height

            # 1. 테이블 직후 주석 행 찾기
            annotation_lines = []
            for line in text_lines:
                # 테이블 바로 아래, 다음 테이블 전
                if table_bottom - 3 <= line['y0'] <= table_bottom + V3Config.ANNOTATION_Y_MARGIN:
                    if line['x0'] >= table_left - 10 and line['x1'] <= table_right + 10:
                        if line['y0'] < next_table_top - 20:
                            # 주석 패턴 확인
                            for pattern in V3Config.ANNOTATION_PATTERNS:
                                if line['text'].startswith(pattern):
                                    annotation_lines.append(line)
                                    break

            if annotation_lines:
                table = _add_annotation_to_table(table, annotation_lines, 'footer')
                logger.debug(f"[PDF-V3] Added annotation to table on page {page_num + 1}")

            # 2. 서브헤더 행 찾기 (예: (A), (B)) - 이미 서브헤더가 없을 때만
            has_subheader = False
            if table.row_count >= 2 and table.data and len(table.data) >= 2:
                # 두 번째 행이 서브헤더 패턴인지 확인
                second_row = table.data[1] if len(table.data) > 1 else []
                for cell in second_row:
                    if cell and ('(A)' in str(cell) or '(B)' in str(cell)):
                        has_subheader = True
                        break

            if not has_subheader and table.row_count >= 2 and table.data:
                row_height_estimate = (table_bottom - table_top) / table.row_count
                header_bottom_estimate = table_top + row_height_estimate
                second_row_top_estimate = table_top + row_height_estimate * 2

                subheader_lines = []
                for line in text_lines:
                    if header_bottom_estimate - 5 <= line['y0'] <= second_row_top_estimate - 5:
                        if line['x0'] >= table_left - 10 and line['x1'] <= table_right + 10:
                            # (A), (B) 패턴 확인
                            if '(A)' in line['text'] or '(B)' in line['text']:
                                subheader_lines.append(line)

                if subheader_lines:
                    table = _add_annotation_to_table(table, subheader_lines, 'subheader')
                    logger.debug(f"[PDF-V3] Added subheader to table on page {page_num + 1}")

            result.append(table)

    result.sort(key=lambda t: (t.page_num, t.bbox[1]))
    return result


def _add_annotation_to_table(table: TableInfo, text_lines: List[Dict], position: str) -> TableInfo:
    """주석 행을 테이블에 추가"""
    if not text_lines:
        return table

    text_lines_sorted = sorted(text_lines, key=lambda l: l['x0'])

    table_width = table.bbox[2] - table.bbox[0]
    col_width = table_width / table.col_count if table.col_count > 0 else table_width

    new_row = [''] * table.col_count

    for line in text_lines_sorted:
        relative_x = line['x0'] - table.bbox[0]
        col_idx = min(int(relative_x / col_width), table.col_count - 1)
        col_idx = max(0, col_idx)

        if new_row[col_idx]:
            new_row[col_idx] += " " + line['text']
        else:
            new_row[col_idx] = line['text']

    non_empty_cols = sum(1 for c in new_row if c)
    if non_empty_cols == 1 and new_row[0]:
        combined_text = " ".join(line['text'] for line in text_lines_sorted)
        new_row = [combined_text] + [''] * (table.col_count - 1)

    new_data = list(table.data)

    # 셀 정보 업데이트
    new_cells_info = None
    if table.cells_info:
        new_cells_info = list(table.cells_info)
    else:
        new_cells_info = []

    if position == 'subheader':
        if len(new_data) > 0:
            new_data.insert(1, new_row)
            # 기존 셀 정보의 row 인덱스 조정 (row >= 1인 경우 +1)
            adjusted_cells = []
            for cell in new_cells_info:
                if cell['row'] >= 1:
                    adjusted_cell = dict(cell)
                    adjusted_cell['row'] = cell['row'] + 1
                    adjusted_cells.append(adjusted_cell)
                else:
                    adjusted_cells.append(cell)
            new_cells_info = adjusted_cells
            # 새 서브헤더 행에 대한 셀 정보 추가 (각 셀은 colspan=1)
            for col_idx in range(table.col_count):
                new_cells_info.append({
                    'row': 1,
                    'col': col_idx,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })
        else:
            new_data.append(new_row)
    else:
        new_data.append(new_row)
        # footer 행에 대한 셀 정보는 _generate_html_from_cells에서 처리됨

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
        cells_info=new_cells_info if new_cells_info else None,
        annotations=table.annotations,
        detection_strategy=table.detection_strategy,
        confidence=table.confidence
    )


# ============================================================================
# Phase 5: HTML Generation
# ============================================================================

def _convert_table_to_html_v3(table_info: TableInfo) -> str:
    """
    테이블을 HTML로 변환 (V3).

    특징:
    1. CellAnalysisEngine을 사용한 셀 분석
    2. 물리적 셀 정보 기반 정확한 rowspan/colspan
    3. 값 기반 병합 추론 (셀 정보 없을 때)
    4. 주석 행 전체 colspan 처리
    5. 접근성 고려한 시맨틱 HTML
    """
    data = table_info.data

    if not data:
        return ""

    num_rows = len(data)
    num_cols = max(len(row) for row in data) if data else 0

    if num_cols == 0:
        return ""

    # CellAnalysisEngine을 사용하여 셀 분석 수행
    cell_engine = CellAnalysisEngine(table_info, None)
    analyzed_cells = cell_engine.analyze()

    # 분석된 셀 정보로 HTML 생성
    return _generate_html_from_cells(data, analyzed_cells, num_rows, num_cols)


def _generate_html_from_cells(
    data: List[List[Optional[str]]],
    cells_info: List[Dict],
    num_rows: int,
    num_cols: int
) -> str:
    """분석된 셀 정보를 사용하여 HTML 생성"""
    span_map: Dict[Tuple[int, int], Dict] = {}

    for cell in cells_info:
        key = (cell['row'], cell['col'])
        span_map[key] = {
            'rowspan': cell.get('rowspan', 1),
            'colspan': cell.get('colspan', 1)
        }

    skip_set: Set[Tuple[int, int]] = set()

    for (row, col), spans in span_map.items():
        rowspan = spans['rowspan']
        colspan = spans['colspan']

        for r in range(row, min(row + rowspan, num_rows)):
            for c in range(col, min(col + colspan, num_cols)):
                if (r, c) != (row, col):
                    skip_set.add((r, c))

    # 주석 행 감지 및 전체 colspan 처리
    for row_idx, row in enumerate(data):
        if not row:
            continue
        first_val = str(row[0]).strip() if row[0] else ""

        is_annotation = False
        for pattern in V3Config.ANNOTATION_PATTERNS:
            if first_val.startswith(pattern):
                is_annotation = True
                break

        if is_annotation:
            span_map[(row_idx, 0)] = {'rowspan': 1, 'colspan': num_cols}
            for col_idx in range(1, num_cols):
                skip_set.add((row_idx, col_idx))

    html_parts = ["<table>"]

    for row_idx, row in enumerate(data):
        html_parts.append("  <tr>")

        for col_idx in range(num_cols):
            if (row_idx, col_idx) in skip_set:
                continue

            content = row[col_idx] if col_idx < len(row) else ""
            content = escape_html(str(content).strip() if content else "")

            spans = span_map.get((row_idx, col_idx), {'rowspan': 1, 'colspan': 1})
            attrs = []

            if spans['rowspan'] > 1:
                attrs.append(f'rowspan="{spans["rowspan"]}"')
            if spans['colspan'] > 1:
                attrs.append(f'colspan="{spans["colspan"]}"')

            attr_str = " " + " ".join(attrs) if attrs else ""

            tag = "th" if row_idx == 0 else "td"
            html_parts.append(f"    <{tag}{attr_str}>{content}</{tag}>")

        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


# ============================================================================
# 테이블 병합 및 연속성 처리
# ============================================================================

def _merge_adjacent_tables_v3(tables: List[TableInfo]) -> List[TableInfo]:
    """인접 테이블 병합"""
    if not tables:
        return tables

    tables_by_page: Dict[int, List[TableInfo]] = defaultdict(list)
    for table in tables:
        tables_by_page[table.page_num].append(table)

    merged_result = []

    for page_num, page_tables in tables_by_page.items():
        sorted_tables = sorted(page_tables, key=lambda t: t.bbox[1])

        i = 0
        while i < len(sorted_tables):
            current = sorted_tables[i]

            merged = current
            while i + 1 < len(sorted_tables):
                next_table = sorted_tables[i + 1]

                if _should_merge_tables_v3(merged, next_table):
                    merged = _do_merge_tables_v3(merged, next_table)
                    i += 1
                    logger.debug(f"[PDF-V3] Merged adjacent tables on page {page_num + 1}")
                else:
                    break

            merged_result.append(merged)
            i += 1

    merged_result.sort(key=lambda t: (t.page_num, t.bbox[1]))
    return merged_result


def _should_merge_tables_v3(t1: TableInfo, t2: TableInfo) -> bool:
    """두 테이블 병합 여부 판단"""
    if t1.page_num != t2.page_num:
        return False

    y_gap = t2.bbox[1] - t1.bbox[3]
    if y_gap < 0 or y_gap > 30:
        return False

    x_overlap_start = max(t1.bbox[0], t2.bbox[0])
    x_overlap_end = min(t1.bbox[2], t2.bbox[2])
    x_overlap = max(0, x_overlap_end - x_overlap_start)

    t1_width = t1.bbox[2] - t1.bbox[0]
    t2_width = t2.bbox[2] - t2.bbox[0]

    overlap_ratio = x_overlap / max(t1_width, t2_width, 1)
    if overlap_ratio < 0.8:
        return False

    if t1.col_count == t2.col_count:
        return True
    if t1.row_count == 1 and t1.col_count < t2.col_count:
        return True

    return False


def _do_merge_tables_v3(t1: TableInfo, t2: TableInfo) -> TableInfo:
    """두 테이블 병합 수행"""
    merged_bbox = (
        min(t1.bbox[0], t2.bbox[0]),
        t1.bbox[1],
        max(t1.bbox[2], t2.bbox[2]),
        t2.bbox[3]
    )

    merged_col_count = max(t1.col_count, t2.col_count)

    merged_data = []
    merged_cells = []

    if t1.col_count < merged_col_count and t1.row_count == 1 and t1.data:
        extra_cols = merged_col_count - t1.col_count
        header_row = list(t1.data[0])

        new_header = []
        col_position = 0

        for orig_col_idx, value in enumerate(header_row):
            new_header.append(value)

            if orig_col_idx == 1 and extra_cols > 0:
                colspan = 1 + extra_cols
                merged_cells.append({
                    'row': 0,
                    'col': col_position,
                    'rowspan': 1,
                    'colspan': colspan,
                    'bbox': None
                })
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
        for row in t1.data:
            if len(row) < merged_col_count:
                adjusted_row = list(row) + [''] * (merged_col_count - len(row))
            else:
                adjusted_row = list(row)
            merged_data.append(adjusted_row)

        if t1.cells_info:
            merged_cells.extend(t1.cells_info)

    for row in t2.data:
        if len(row) < merged_col_count:
            adjusted_row = list(row) + [''] * (merged_col_count - len(row))
        else:
            adjusted_row = list(row)
        merged_data.append(adjusted_row)

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
        cells_info=merged_cells if merged_cells else None,
        detection_strategy=t1.detection_strategy,
        confidence=max(t1.confidence, t2.confidence)
    )


def _process_table_continuity_v3(all_tables: List[TableInfo]) -> List[TableInfo]:
    """페이지 간 테이블 연속성 처리"""
    if not all_tables:
        return all_tables

    result = []
    last_category = None

    for i, table_info in enumerate(all_tables):
        table_info = TableInfo(
            page_num=table_info.page_num,
            table_idx=table_info.table_idx,
            bbox=table_info.bbox,
            data=copy.deepcopy(table_info.data),
            col_count=table_info.col_count,
            row_count=table_info.row_count,
            page_height=table_info.page_height,
            cells_info=table_info.cells_info,
            annotations=table_info.annotations,
            detection_strategy=table_info.detection_strategy,
            confidence=table_info.confidence
        )

        curr_data = table_info.data

        if i == 0:
            last_category = _extract_last_category_v3(curr_data)
            result.append(table_info)
            continue

        prev_table = all_tables[i - 1]

        is_continuation = (
            table_info.page_num > prev_table.page_num and
            prev_table.bbox[3] > prev_table.page_height * 0.7 and
            table_info.bbox[1] < table_info.page_height * 0.3 and
            table_info.col_count == prev_table.col_count
        )

        if is_continuation and last_category:
            for row in curr_data:
                if len(row) >= 2:
                    first_col = row[0]
                    second_col = row[1] if len(row) > 1 else ""

                    if (not first_col or not str(first_col).strip()) and second_col and str(second_col).strip():
                        row[0] = last_category
                    elif first_col and str(first_col).strip():
                        last_category = first_col
        else:
            new_last = _extract_last_category_v3(curr_data)
            if new_last:
                last_category = new_last

        result.append(table_info)

    return result


def _extract_last_category_v3(table_data: List[List[Optional[str]]]) -> Optional[str]:
    """테이블에서 마지막 카테고리 추출"""
    if not table_data:
        return None

    last_category = None

    for row in table_data:
        if len(row) >= 1 and row[0] and str(row[0]).strip():
            last_category = str(row[0]).strip()

    return last_category


# ============================================================================
# 페이지 테두리 감지 (V3)
# ============================================================================

def _detect_page_border_v3(page) -> PageBorderInfo:
    """
    페이지 테두리(장식용)를 감지합니다 (V3).

    개선점:
    1. 얇은 선도 감지
    2. 이중선 처리
    3. 더 정확한 테두리 판별
    """
    result = PageBorderInfo()

    drawings = page.get_drawings()
    if not drawings:
        return result

    page_width = page.rect.width
    page_height = page.rect.height

    edge_margin = min(page_width, page_height) * V3Config.PAGE_BORDER_MARGIN
    page_spanning_ratio = V3Config.PAGE_SPANNING_RATIO

    border_lines = {
        'top': False,
        'bottom': False,
        'left': False,
        'right': False
    }

    for drawing in drawings:
        rect = drawing.get('rect')
        if not rect:
            continue

        w = rect.width
        h = rect.height

        # 얇은 선도 감지 (두께 제한 완화)
        # 가로선 (높이가 작고 너비가 큼)
        if h <= 10 and w > page_width * page_spanning_ratio:
            if rect.y0 < edge_margin:
                border_lines['top'] = True
            elif rect.y1 > page_height - edge_margin:
                border_lines['bottom'] = True

        # 세로선 (너비가 작고 높이가 큼)
        if w <= 10 and h > page_height * page_spanning_ratio:
            if rect.x0 < edge_margin:
                border_lines['left'] = True
            elif rect.x1 > page_width - edge_margin:
                border_lines['right'] = True

    # 4면 모두 있으면 페이지 테두리
    if all(border_lines.values()):
        result.has_border = True
        result.border_bbox = (edge_margin, edge_margin, page_width - edge_margin, page_height - edge_margin)
        result.border_lines = border_lines

    return result


def _is_table_likely_border_v3(
    table_bbox: Tuple[float, float, float, float],
    border_info: PageBorderInfo,
    page
) -> bool:
    """테이블이 페이지 테두리인지 확인 (V3)"""
    if not border_info.has_border or not border_info.border_bbox:
        return False

    page_width = page.rect.width
    page_height = page.rect.height

    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]

    if table_width > page_width * 0.85 and table_height > page_height * 0.85:
        return True

    return False


# ============================================================================
# 텍스트 추출 (V3)
# ============================================================================

def _extract_text_blocks_v3(
    page,
    page_num: int,
    table_bboxes: List[Tuple[float, float, float, float]],
    border_info: PageBorderInfo,
    use_quality_check: bool = True
) -> List[PageElement]:
    """
    테이블 영역을 제외한 텍스트 블록 추출 (V3)

    개선 사항:
    1. 텍스트 품질 분석 (깨진 텍스트 감지)
    2. 품질이 낮은 경우 OCR 폴백
    """
    elements = []

    # 텍스트 품질 분석
    if use_quality_check:
        analyzer = TextQualityAnalyzer(page, page_num)
        page_analysis = analyzer.analyze_page()

        # 품질이 너무 낮으면 전체 페이지 OCR 폴백
        if page_analysis.quality_result.needs_ocr:
            logger.info(
                f"[PDF-V3] Page {page_num + 1}: Low text quality "
                f"({page_analysis.quality_result.quality_score:.2f}), "
                f"PUA={page_analysis.quality_result.pua_count}, "
                f"using OCR fallback"
            )

            extractor = QualityAwareTextExtractor(page, page_num)
            ocr_text, _ = extractor.extract()

            if ocr_text.strip():
                # OCR 텍스트를 블록별로 분리하여 반환
                # 테이블 영역 제외
                ocr_blocks = _split_ocr_text_to_blocks(ocr_text, page, table_bboxes)
                return ocr_blocks

    # 기존 로직: 일반 텍스트 추출
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_bbox = block.get("bbox", (0, 0, 0, 0))

        if is_inside_any_bbox(block_bbox, table_bboxes):
            continue

        text_parts = []
        block_quality_ok = True

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                text_parts.append(line_text.strip())

        if text_parts:
            full_text = "\n".join(text_parts)

            # 개별 블록 품질 체크 (use_quality_check가 True인 경우)
            if use_quality_check:
                analyzer = TextQualityAnalyzer(page, page_num)
                block_quality = analyzer.analyze_text(full_text)

                if block_quality.needs_ocr:
                    # 해당 블록만 OCR
                    from libs.core.processor.pdf_helpers.text_quality_analyzer import PageOCRFallbackEngine
                    ocr_engine = PageOCRFallbackEngine(page, page_num)
                    ocr_text = ocr_engine.ocr_region(block_bbox)
                    if ocr_text.strip():
                        full_text = ocr_text
                        logger.debug(f"[PDF-V3] Block OCR: '{ocr_text[:50]}...'")

            elements.append(PageElement(
                element_type=ElementType.TEXT,
                content=full_text,
                bbox=block_bbox,
                page_num=page_num
            ))

    return elements


def _split_ocr_text_to_blocks(
    ocr_text: str,
    page,
    table_bboxes: List[Tuple[float, float, float, float]]
) -> List[PageElement]:
    """
    OCR 텍스트를 페이지 요소로 변환

    OCR은 위치 정보가 없으므로, 전체 텍스트를 하나의 블록으로 처리합니다.
    테이블 영역은 제외됩니다.
    """
    if not ocr_text.strip():
        return []

    # 테이블 영역을 제외한 페이지 영역 계산
    page_width = page.rect.width
    page_height = page.rect.height

    # OCR 텍스트를 단일 블록으로 반환 (위치는 페이지 전체)
    # 실제 위치 정보가 필요하면 pytesseract의 image_to_data 사용 가능
    return [PageElement(
        element_type=ElementType.TEXT,
        content=ocr_text,
        bbox=(0, 0, page_width, page_height),
        page_num=page.number
    )]


# ============================================================================
# 이미지 추출 (V3)
# ============================================================================

async def _extract_images_from_page_v3(
    page,
    page_num: int,
    doc,
    app_db,
    processed_images: Set[int],
    table_bboxes: List[Tuple[float, float, float, float]],
    min_image_size: int = 50,
    min_image_area: int = 2500
) -> List[PageElement]:
    """페이지에서 이미지 추출 및 MinIO 업로드 (V3)"""
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
                logger.debug(f"[PDF-V3] Error extracting image xref={xref}: {e}")
                continue

    except Exception as e:
        logger.warning(f"[PDF-V3] Error extracting images: {e}")

    return elements


# ============================================================================
# 요소 병합 (V3)
# ============================================================================

def _merge_page_elements_v3(elements: List[PageElement]) -> str:
    """페이지 요소들을 위치 기반으로 정렬하여 병합 (V3)"""
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
