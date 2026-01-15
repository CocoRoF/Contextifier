# your_package/document_processor/pdf_handler_default.py
"""
PDF Handler (Default) - PyMuPDF 전용 고속 PDF 처리기

**설계 철학**:
- PyMuPDF(fitz)만 사용하여 최대 속도와 안정성 확보
- 복잡한 테이블 구조 완벽 지원 (열린 테두리, 병합 셀, 다중 페이지 연속 테이블)
- 의미론적 텍스트 분석으로 테이블 구조 복원

**핵심 알고리즘**:
1. 벡터 그래픽 분석: PDF의 lines/rects에서 테이블 그리드 구조 추출
2. 텍스트 공간 분석: 텍스트 블록 위치로 열린 테두리 테이블 감지
3. 병합 셀 복원: 물리적 셀 bbox에서 rowspan/colspan 계산
4. 페이지 연속성: 다중 페이지 테이블의 카테고리 셀 복원

**테이블 처리 전략** (우선순위 순):
1. lines 전략: 벡터 선으로 형성된 명시적 테이블
2. lines_strict 전략: 닫힌 사각형만 사용
3. text 전략: 텍스트 위치 기반 암시적 테이블
4. 하이브리드 전략: 부분 테두리 + 텍스트 분석
"""

import copy
import hashlib
import io
import logging
import os
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from PIL import Image  # noqa: F401

from service.storage.minio_client import DEFAULT_BUCKET_NAME  # noqa: F401
from libs.core.functions.utils import upload_image_to_minio

logger = logging.getLogger("document-processor")

# PyMuPDF import
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.error("PyMuPDF is required for PDF processing but not available")


# === 타입 정의 ===

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
        """정렬 키: (y0, x0) - 위에서 아래, 왼쪽에서 오른쪽"""
        return (self.bbox[1], self.bbox[0])


@dataclass
class TableCell:
    """
    테이블 셀 정보

    물리적 bbox와 논리적 그리드 위치 모두 저장
    """
    row: int
    col: int
    rowspan: int
    colspan: int
    bbox: Tuple[float, float, float, float]
    text: str = ""


@dataclass
class TableGrid:
    """
    테이블 그리드 구조

    X/Y 경계선 목록과 셀 정보를 저장
    """
    x_boundaries: List[float]  # 열 경계선 X 좌표
    y_boundaries: List[float]  # 행 경계선 Y 좌표
    cells: List[TableCell] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)

    @property
    def num_rows(self) -> int:
        return max(0, len(self.y_boundaries) - 1)

    @property
    def num_cols(self) -> int:
        return max(0, len(self.x_boundaries) - 1)


@dataclass
class LineSegment:
    """선분 정보"""
    x0: float
    y0: float
    x1: float
    y1: float
    is_horizontal: bool

    @property
    def length(self) -> float:
        return ((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2) ** 0.5


# === 메인 함수 ===

async def extract_text_from_pdf_default(
    file_path: str,
    current_config: Dict[str, Any] = None,
    app_db=None,
    extract_default_metadata: bool = True,
) -> str:
    """
    PyMuPDF 전용 고속 PDF 텍스트 추출.

    Args:
        file_path: PDF 파일 경로
        current_config: 설정 딕셔너리
        app_db: 데이터베이스 연결 (이미지 메타데이터 저장용)
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (메타데이터, 테이블 HTML 포함)
    """
    if current_config is None:
        current_config = {}

    logger.info(f"[Default Handler] PDF processing: {file_path}")

    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available, cannot process PDF")
        return "[PDF 파일 처리 실패: PyMuPDF가 설치되지 않음]"

    try:
        result_parts = []
        processed_images: Set[int] = set()

        # PDF 열기
        doc = fitz.open(file_path)
        total_pages = len(doc)
        logger.info(f"[Default Handler] PDF has {total_pages} pages")

        # 1. 메타데이터 추출 및 추가 (extract_default_metadata가 True인 경우에만)
        if extract_default_metadata:
            metadata = _extract_pdf_metadata(doc)
            if metadata:
                metadata_str = _format_metadata(metadata)
                result_parts.append(metadata_str)

        # 2. 모든 페이지에서 테이블 추출 (연속성 처리 포함)
        all_tables_by_page = _extract_all_tables_robust(doc)
        logger.info(f"[Default Handler] Extracted tables from {len(all_tables_by_page)} pages")

        # 3. 각 페이지 처리
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_height = page.rect.height

            # 페이지 헤더 추가
            result_parts.append(f"\n<페이지 번호> {page_num + 1} </페이지 번호>\n")

            # 페이지의 모든 요소 수집
            elements: List[PageElement] = []

            # 텍스트 블록 추출
            text_elements = _extract_text_blocks(page, page_num, page_height)
            elements.extend(text_elements)

            # 이미지 추출
            image_elements = await _extract_images_from_page(
                page, page_num, page_height, doc, app_db, processed_images
            )
            elements.extend(image_elements)

            # 테이블 추가
            if page_num in all_tables_by_page:
                elements.extend(all_tables_by_page[page_num])

            # 요소들을 위치 기준으로 정렬
            elements.sort(key=lambda e: e.sort_key)

            # 중복 제거 및 통합
            merged_content = _merge_page_elements(elements)
            result_parts.append(merged_content)

        doc.close()

        result = "".join(result_parts)
        logger.info(f"[Default Handler] Processing completed: {len(result)} characters extracted")

        return result

    except Exception as e:
        logger.error(f"[Default Handler] Error processing PDF: {e}")
        logger.debug(traceback.format_exc())
        return f"[PDF 파일 처리 실패: {str(e)}]"


# === 메타데이터 추출 ===

def _extract_pdf_metadata(doc) -> Dict[str, Any]:
    """PDF 문서에서 메타데이터를 추출합니다."""
    metadata = {}

    try:
        raw = doc.metadata
        if not raw:
            return metadata

        if raw.get("title"):
            metadata["title"] = raw["title"]
        if raw.get("author"):
            metadata["author"] = raw["author"]
        if raw.get("subject"):
            metadata["subject"] = raw["subject"]
        if raw.get("keywords"):
            metadata["keywords"] = raw["keywords"]

        if raw.get("creationDate"):
            create_time = _parse_pdf_date(raw["creationDate"])
            if create_time:
                metadata["create_time"] = create_time.strftime("%Y-%m-%d %H:%M:%S")

        if raw.get("modDate"):
            mod_time = _parse_pdf_date(raw["modDate"])
            if mod_time:
                metadata["last_saved_time"] = mod_time.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata: {e}")

    return metadata


def _parse_pdf_date(date_str: str) -> Optional[datetime]:
    """PDF 날짜 문자열을 datetime 객체로 변환합니다."""
    if not date_str:
        return None

    try:
        # PDF 날짜 형식: D:YYYYMMDDHHmmSSOHH'mm'
        cleaned = date_str
        if cleaned.startswith("D:"):
            cleaned = cleaned[2:]

        # 시간대 정보 제거
        for sep in ["+", "-", "Z"]:
            if sep in cleaned:
                cleaned = cleaned.split(sep)[0]

        cleaned = cleaned.replace("'", "")

        formats = [
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M",
            "%Y%m%d",
            "%Y%m",
            "%Y"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(cleaned[:len(fmt.replace('%', ''))], fmt)
            except ValueError:
                continue

    except Exception as e:
        logger.debug(f"Could not parse PDF date: {date_str}, error: {e}")

    return None


def _format_metadata(metadata: Dict[str, Any]) -> str:
    """메타데이터를 포맷된 문자열로 변환합니다."""
    if not metadata:
        return ""

    lines = ["<Document-Metadata>"]

    field_names = {
        'title': '제목',
        'subject': '주제',
        'author': '작성자',
        'keywords': '키워드',
        'create_time': '작성일',
        'last_saved_time': '마지막 수정일'
    }

    for key, label in field_names.items():
        if key in metadata and metadata[key]:
            lines.append(f"  <{label}>{metadata[key]}</{label}>")

    lines.append("</Document-Metadata>\n")

    return "\n".join(lines)


# === 텍스트 추출 ===

def _extract_text_blocks(
    page,
    page_num: int,
    page_height: float
) -> List[PageElement]:
    """페이지에서 텍스트 블록을 추출합니다."""
    elements = []

    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # 텍스트 블록
                bbox = block.get("bbox", (0, 0, 0, 0))

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
    processed_images: Set[int],
    min_image_size: int = 50,
    min_image_area: int = 2500
) -> List[PageElement]:
    """페이지에서 이미지를 추출하고 MinIO에 업로드합니다."""
    elements = []

    try:
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]

                if xref in processed_images:
                    continue

                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_data = base_image.get("image")
                if not image_data:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < min_image_size or height < min_image_size:
                    continue

                if width * height < min_image_area:
                    continue

                bbox = _find_image_position(page, xref)
                if not bbox:
                    bbox = (0, page_height / 2, page.rect.width, page_height / 2 + 1)

                minio_path = upload_image_to_minio(image_data, app_db=app_db)

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


def _find_image_position(page, xref: int) -> Optional[Tuple[float, float, float, float]]:
    """이미지의 페이지 내 위치를 찾습니다."""
    try:
        for img in page.get_images():
            if img[0] == xref:
                rects = page.get_image_rects(img)
                if rects:
                    rect = rects[0]
                    return (rect.x0, rect.y0, rect.x1, rect.y1)

        image_rects = page.get_image_rects(xref)
        if image_rects:
            rect = image_rects[0]
            return (rect.x0, rect.y0, rect.x1, rect.y1)

    except Exception as e:
        logger.debug(f"Could not find image position for xref {xref}: {e}")

    return None


# === 강건한 테이블 추출 ===

def _extract_all_tables_robust(doc) -> Dict[int, List[PageElement]]:
    """
    모든 페이지에서 테이블을 강건하게 추출합니다.

    **알고리즘**:
    1. 각 페이지에서 벡터 그래픽 분석 (lines, rects)
    2. 테이블 그리드 구조 식별
    3. 텍스트 매핑으로 셀 내용 채우기
    4. 열린 테두리 테이블 감지 및 복원
    5. 페이지 간 연속 테이블 처리

    Returns:
        페이지별 테이블 요소 딕셔너리
    """
    tables_by_page: Dict[int, List[PageElement]] = {}

    # 모든 페이지의 raw 테이블 정보 수집
    all_raw_tables: List[Dict] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        try:
            # 1단계: PyMuPDF find_tables 사용 (다양한 전략)
            page_tables = _find_tables_multi_strategy(page, page_num)

            # 2단계: 열린 테두리 테이블 탐지 (텍스트 기반)
            if not page_tables:
                page_tables = _detect_open_border_tables(page, page_num)

            all_raw_tables.extend(page_tables)

        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")
            continue

    if not all_raw_tables:
        return tables_by_page

    # 3단계: 페이지 간 연속 테이블 처리
    processed_tables = _process_table_continuity(all_raw_tables)

    # 4단계: HTML 변환 및 PageElement 생성
    for table_info in processed_tables:
        page_num = table_info['page_num']
        table_data = table_info['data']
        bbox = table_info['bbox']
        cell_spans = table_info.get('cell_spans', {})

        html_table = _convert_table_to_html(table_data, cell_spans)

        if html_table:
            if page_num not in tables_by_page:
                tables_by_page[page_num] = []

            tables_by_page[page_num].append(PageElement(
                element_type=ElementType.TABLE,
                content=html_table,
                bbox=bbox,
                page_num=page_num
            ))

    return tables_by_page


def _find_tables_multi_strategy(page, page_num: int) -> List[Dict]:
    """
    다양한 전략으로 테이블을 찾습니다.

    **전략 우선순위**:
    1. lines: 벡터 선 기반 (가장 정확)
    2. lines_strict: 닫힌 테두리만
    3. text: 텍스트 위치 기반 (열린 테두리용)
    """
    if not hasattr(page, 'find_tables'):
        return []

    strategies = [
        ("lines", {}),
        ("lines_strict", {}),
        ("text", {"min_words_vertical": 2, "min_words_horizontal": 1}),
        ("text", {"min_words_vertical": 1, "min_words_horizontal": 1}),
    ]

    page_height = page.rect.height
    best_result = []
    best_score = 0

    for strategy_name, extra_params in strategies:
        try:
            params = {"strategy": strategy_name, **extra_params}
            tab_finder = page.find_tables(**params)
            tables = tab_finder.tables

            if not tables:
                continue

            # 테이블 품질 점수 계산
            score = 0
            for table in tables:
                # 셀 수에 비례한 점수
                cell_count = table.row_count * table.col_count
                score += cell_count

                # 데이터가 있는 셀 비율에 따른 보너스
                data = table.extract()
                non_empty = sum(1 for row in data for cell in row if cell and str(cell).strip())
                if cell_count > 0:
                    fill_ratio = non_empty / cell_count
                    score += fill_ratio * 10

            if score > best_score:
                best_score = score
                best_result = []

                for table_idx, table in enumerate(tables):
                    data = table.extract()

                    # 빈 테이블 건너뛰기
                    if not data or not any(any(cell for cell in row if cell) for row in data):
                        continue

                    # 병합 셀 정보 추출
                    cell_spans = _extract_cell_spans_from_table(table)

                    best_result.append({
                        'page_num': page_num,
                        'table_idx': table_idx,
                        'data': _clean_table_data(data),
                        'bbox': table.bbox,
                        'page_height': page_height,
                        'cell_spans': cell_spans,
                        'strategy': strategy_name,
                    })

        except Exception as e:
            logger.debug(f"Strategy {strategy_name} failed on page {page_num}: {e}")
            continue

    if best_result:
        logger.debug(f"Page {page_num + 1}: Found {len(best_result)} tables using best strategy")

    return best_result


def _extract_cell_spans_from_table(table) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    PyMuPDF 테이블에서 셀 병합 정보를 추출합니다.

    **알고리즘**:
    1. table.cells에서 각 셀의 물리적 bbox 추출
    2. Y 좌표를 행 인덱스로, X 좌표를 열 인덱스로 매핑
    3. 셀 bbox가 여러 그리드 셀을 차지하면 rowspan/colspan 계산

    Returns:
        {(row, col): {'rowspan': n, 'colspan': m}, ...}
    """
    cell_spans: Dict[Tuple[int, int], Dict[str, int]] = {}

    if not hasattr(table, 'cells') or not table.cells:
        return cell_spans

    try:
        cells = table.cells
        if not cells:
            return cell_spans

        # X, Y 경계선 추출
        x_coords = sorted(set([c[0] for c in cells] + [c[2] for c in cells]))
        y_coords = sorted(set([c[1] for c in cells] + [c[3] for c in cells]))

        # 좌표를 그리드 인덱스로 매핑
        def coord_to_index(coord: float, coords: List[float], tolerance: float = 3.0) -> int:
            for i, c in enumerate(coords):
                if abs(coord - c) <= tolerance:
                    return i
            # 가장 가까운 인덱스 반환
            return min(range(len(coords)), key=lambda i: abs(coords[i] - coord))

        # 처리된 그리드 위치 추적
        processed_positions = set()

        for cell in cells:
            x0, y0, x1, y1 = cell[:4]

            col_start = coord_to_index(x0, x_coords)
            col_end = coord_to_index(x1, x_coords)
            row_start = coord_to_index(y0, y_coords)
            row_end = coord_to_index(y1, y_coords)

            colspan = max(1, col_end - col_start)
            rowspan = max(1, row_end - row_start)

            if (row_start, col_start) in processed_positions:
                continue

            processed_positions.add((row_start, col_start))

            if rowspan > 1 or colspan > 1:
                cell_spans[(row_start, col_start)] = {
                    'rowspan': rowspan,
                    'colspan': colspan
                }

                # 병합된 영역의 다른 셀들 마킹
                for r in range(row_start, row_start + rowspan):
                    for c in range(col_start, col_start + colspan):
                        if (r, c) != (row_start, col_start):
                            processed_positions.add((r, c))

    except Exception as e:
        logger.debug(f"Error extracting cell spans: {e}")

    return cell_spans


def _detect_open_border_tables(page, page_num: int) -> List[Dict]:
    """
    열린 테두리(open border) 테이블을 텍스트 위치 분석으로 감지합니다.

    **원리**:
    1. 텍스트 블록을 수집하고 Y 좌표로 그룹화 (행)
    2. 각 행 내에서 X 좌표로 그룹화 (열)
    3. 일관된 열 구조가 있으면 테이블로 판단
    4. 최소 3행 이상이어야 테이블로 인정

    Returns:
        감지된 테이블 정보 리스트
    """
    try:
        # 텍스트 블록 추출
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        # 텍스트 스팬 수집: [(x0, y0, x1, y1, text), ...]
        text_spans = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox")
                    text = span.get("text", "").strip()
                    if text and bbox:
                        text_spans.append((bbox[0], bbox[1], bbox[2], bbox[3], text))

        if len(text_spans) < 6:  # 최소 셀 수
            return []

        # Y 좌표로 행 그룹화 (tolerance: 5pt)
        y_tolerance = 5.0
        rows: List[List[Tuple]] = []
        current_row: List[Tuple] = []
        current_y = None

        # Y 좌표로 정렬
        sorted_spans = sorted(text_spans, key=lambda s: (s[1], s[0]))

        for span in sorted_spans:
            y = span[1]
            if current_y is None or abs(y - current_y) <= y_tolerance:
                current_row.append(span)
                current_y = y
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [span]
                current_y = y

        if current_row:
            rows.append(current_row)

        if len(rows) < 3:  # 최소 3행
            return []

        # X 좌표로 열 경계 추출
        all_x_coords = []
        for row in rows:
            for span in row:
                all_x_coords.append(span[0])  # x0
                all_x_coords.append(span[2])  # x1

        # X 좌표 클러스터링
        x_clusters = _cluster_coordinates(all_x_coords, tolerance=15.0)

        if len(x_clusters) < 3:  # 최소 2열 (3개의 경계선)
            return []

        # 행별 열 수 일관성 확인
        col_counts = []
        for row in rows:
            row_x_starts = [span[0] for span in row]
            row_col_count = 0
            for x_cluster in x_clusters[:-1]:  # 마지막 경계 제외
                # 이 열 경계 근처에 시작하는 스팬이 있는지
                if any(abs(x - x_cluster) < 20 for x in row_x_starts):
                    row_col_count += 1
            col_counts.append(row_col_count)

        # 열 수가 일관적인지 확인 (80% 이상이 같은 열 수)
        if not col_counts:
            return []

        from collections import Counter
        col_counter = Counter(col_counts)
        most_common_count = col_counter.most_common(1)[0][0]
        consistency = sum(1 for c in col_counts if c == most_common_count) / len(col_counts)

        if consistency < 0.7 or most_common_count < 2:
            return []

        logger.debug(f"Page {page_num + 1}: Detected open-border table with ~{most_common_count} columns")

        # 테이블 데이터 구성
        num_cols = len(x_clusters) - 1
        table_data = []

        for row in rows:
            row_data = [""] * num_cols
            row_spans = sorted(row, key=lambda s: s[0])

            for span in row_spans:
                x = span[0]
                text = span[4]

                # 가장 가까운 열 찾기
                col_idx = 0
                for i in range(len(x_clusters) - 1):
                    if x_clusters[i] <= x < x_clusters[i + 1]:
                        col_idx = i
                        break
                    elif x >= x_clusters[-1]:
                        col_idx = num_cols - 1

                if col_idx < num_cols:
                    if row_data[col_idx]:
                        row_data[col_idx] += " " + text
                    else:
                        row_data[col_idx] = text

            table_data.append(row_data)

        # bbox 계산
        min_x = min(span[0] for row in rows for span in row)
        min_y = min(span[1] for row in rows for span in row)
        max_x = max(span[2] for row in rows for span in row)
        max_y = max(span[3] for row in rows for span in row)

        return [{
            'page_num': page_num,
            'table_idx': 0,
            'data': table_data,
            'bbox': (min_x, min_y, max_x, max_y),
            'page_height': page.rect.height,
            'cell_spans': {},
            'strategy': 'open_border',
        }]

    except Exception as e:
        logger.debug(f"Error detecting open-border tables on page {page_num}: {e}")
        return []


def _cluster_coordinates(coords: List[float], tolerance: float = 10.0) -> List[float]:
    """좌표들을 클러스터링하여 대표값 리스트 반환"""
    if not coords:
        return []

    sorted_coords = sorted(set(coords))
    clusters = []
    current_cluster = [sorted_coords[0]]

    for coord in sorted_coords[1:]:
        if coord - current_cluster[-1] <= tolerance:
            current_cluster.append(coord)
        else:
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [coord]

    if current_cluster:
        clusters.append(sum(current_cluster) / len(current_cluster))

    return clusters


# === 페이지 간 테이블 연속성 처리 ===

def _process_table_continuity(all_tables: List[Dict]) -> List[Dict]:
    """
    페이지 간 테이블 연속성을 처리합니다.

    **알고리즘**:
    1. 연속 페이지의 테이블 구조(열 수, X 좌표) 비교
    2. 이전 테이블이 페이지 하단, 현재 테이블이 페이지 상단에 있는지 확인
    3. 연속으로 판단되면 빈 카테고리 셀을 이전 값으로 채움

    **핵심 휴리스틱**:
    - 열 수가 같고 X 좌표가 유사하면 연속 가능성 높음
    - 첫 번째 열이 비어있고 두 번째 열에 데이터가 있으면 rowspan 연속
    - 카테고리 열(첫 1-2개 열)의 빈 셀만 채움
    """
    if not all_tables:
        return []

    result = copy.deepcopy(all_tables)

    # 이전 테이블의 마지막 카테고리 값 추적
    last_category: Dict[int, str] = {}  # col_idx -> last_value
    prev_table_info: Optional[Dict] = None

    for i, table_info in enumerate(result):
        data = table_info['data']
        page_num = table_info['page_num']
        bbox = table_info['bbox']
        page_height = table_info['page_height']

        if not data:
            prev_table_info = table_info
            continue

        # 연속 테이블 판단
        is_continuation = False
        if prev_table_info and _is_table_continuation(prev_table_info, table_info):
            is_continuation = True
            logger.debug(f"Page {page_num + 1}: Table is continuation from page {prev_table_info['page_num'] + 1}")

        # 연속 테이블인 경우 빈 카테고리 셀 채우기
        if is_continuation and last_category:
            data = _fill_empty_category_cells(data, last_category)
            result[i]['data'] = data

        # 현재 테이블의 카테고리 값 업데이트
        _update_last_category(data, last_category)

        prev_table_info = table_info

    return result


def _is_table_continuation(prev: Dict, curr: Dict) -> bool:
    """두 테이블이 페이지 간 연속인지 판단합니다."""
    # 연속 페이지여야 함
    if curr['page_num'] != prev['page_num'] + 1:
        return False

    # 열 수가 같아야 함
    prev_data = prev['data']
    curr_data = curr['data']

    if not prev_data or not curr_data:
        return False

    prev_cols = max(len(row) for row in prev_data) if prev_data else 0
    curr_cols = max(len(row) for row in curr_data) if curr_data else 0

    if prev_cols != curr_cols:
        return False

    # 이전 테이블이 페이지 하단에 있어야 함 (y1 > 70% of page)
    if prev['bbox'][3] < prev['page_height'] * 0.7:
        return False

    # 현재 테이블이 페이지 상단에 있어야 함 (y0 < 30% of page)
    if curr['bbox'][1] > curr['page_height'] * 0.3:
        return False

    return True


def _fill_empty_category_cells(
    data: List[List[str]],
    last_category: Dict[int, str],
    max_category_cols: int = 2
) -> List[List[str]]:
    """빈 카테고리 셀을 이전 값으로 채웁니다."""
    result = copy.deepcopy(data)

    for row_idx, row in enumerate(result):
        for col_idx in range(min(max_category_cols, len(row))):
            if not row[col_idx] or not row[col_idx].strip():
                if col_idx in last_category:
                    result[row_idx][col_idx] = last_category[col_idx]
            else:
                # 새 값이 있으면 업데이트
                last_category[col_idx] = row[col_idx]

    return result


def _update_last_category(data: List[List[str]], last_category: Dict[int, str], max_cols: int = 2):
    """테이블에서 마지막 카테고리 값을 업데이트합니다."""
    for row in data:
        for col_idx in range(min(max_cols, len(row))):
            if row[col_idx] and row[col_idx].strip():
                last_category[col_idx] = row[col_idx]


# === 테이블 데이터 정제 ===

def _clean_table_data(data: List[List[Optional[str]]]) -> List[List[str]]:
    """테이블 데이터를 정제합니다."""
    if not data:
        return []

    max_cols = max(len(row) for row in data if row) if data else 0
    cleaned = []

    for row in data:
        cleaned_row = []
        for i in range(max_cols):
            if row and i < len(row):
                cell = row[i]
                if cell is None:
                    cleaned_row.append("")
                else:
                    cleaned_row.append(str(cell).replace('\n', ' ').strip())
            else:
                cleaned_row.append("")
        cleaned.append(cleaned_row)

    return cleaned


# === HTML 변환 ===

def _convert_table_to_html(
    data: List[List[str]],
    cell_spans: Dict[Tuple[int, int], Dict[str, int]]
) -> str:
    """
    테이블 데이터를 HTML로 변환합니다.

    **병합 셀 처리**:
    1. cell_spans에서 물리적 rowspan/colspan 정보 사용
    2. 정보가 없으면 값 기반 rowspan 감지 시도
    """
    if not data or len(data) == 0:
        return ""

    num_rows = len(data)
    num_cols = max(len(row) for row in data) if data else 0

    if num_cols == 0:
        return ""

    # 물리적 셀 정보가 없으면 값 기반 감지
    if not cell_spans:
        cell_spans = _detect_value_based_spans(data)

    # skip 맵 생성
    skip_map = [[False] * num_cols for _ in range(num_rows)]

    for (row, col), spans in cell_spans.items():
        rowspan = spans.get('rowspan', 1)
        colspan = spans.get('colspan', 1)

        for r in range(row, min(row + rowspan, num_rows)):
            for c in range(col, min(col + colspan, num_cols)):
                if (r, c) != (row, col):
                    skip_map[r][c] = True

    # HTML 생성
    html_parts = ["<table border='1'>"]

    for row_idx in range(num_rows):
        row = data[row_idx] if row_idx < len(data) else []
        row_cells = []

        for col_idx in range(num_cols):
            if skip_map[row_idx][col_idx]:
                continue

            cell_value = row[col_idx] if col_idx < len(row) else ""
            cell_text = _escape_html(cell_value)

            tag = "th" if row_idx == 0 else "td"

            attrs = []
            if (row_idx, col_idx) in cell_spans:
                spans = cell_spans[(row_idx, col_idx)]
                if spans.get('rowspan', 1) > 1:
                    attrs.append(f"rowspan='{spans['rowspan']}'")
                if spans.get('colspan', 1) > 1:
                    attrs.append(f"colspan='{spans['colspan']}'")

            attr_str = " " + " ".join(attrs) if attrs else ""
            row_cells.append(f"<{tag}{attr_str}>{cell_text}</{tag}>")

        if row_cells:
            html_parts.append("<tr>" + "".join(row_cells) + "</tr>")

    html_parts.append("</table>")

    return "\n".join(html_parts)


def _detect_value_based_spans(data: List[List[str]], max_merge_cols: int = 2) -> Dict[Tuple[int, int], Dict[str, int]]:
    """
    값이 연속으로 동일한 셀을 rowspan으로 감지합니다.

    **주의**: 카테고리 열(첫 1-2개 열)에만 적용
    숫자 데이터 열에는 적용하지 않음
    """
    cell_spans: Dict[Tuple[int, int], Dict[str, int]] = {}

    if not data or len(data) < 2:
        return cell_spans

    num_rows = len(data)
    num_cols = max(len(row) for row in data) if data else 0

    # 각 열에 대해 연속 동일 값 감지
    for col_idx in range(min(max_merge_cols, num_cols)):
        # 숫자 열인지 확인
        if _is_numeric_column(data, col_idx):
            continue

        row_idx = 0
        while row_idx < num_rows:
            cell_value = data[row_idx][col_idx] if col_idx < len(data[row_idx]) else ""

            if not cell_value or not cell_value.strip():
                row_idx += 1
                continue

            # 연속으로 같은 값인 행 수 세기
            span = 1
            for next_row in range(row_idx + 1, num_rows):
                next_value = data[next_row][col_idx] if col_idx < len(data[next_row]) else ""
                if next_value == cell_value:
                    span += 1
                else:
                    break

            if span > 1:
                cell_spans[(row_idx, col_idx)] = {'rowspan': span, 'colspan': 1}

            row_idx += span

    return cell_spans


def _is_numeric_column(data: List[List[str]], col_idx: int, threshold: float = 0.7) -> bool:
    """열이 주로 숫자로 구성되어 있는지 확인합니다."""
    numeric_count = 0
    total_count = 0

    for row in data[1:]:  # 헤더 제외
        if col_idx < len(row):
            value = row[col_idx]
            if value and value.strip():
                total_count += 1
                # 숫자, 쉼표, 점, 백분율 등 제거 후 숫자인지 확인
                cleaned = value.strip().replace(',', '').replace('.', '').replace('%', '').replace('-', '').replace('+', '').replace(' ', '')
                if cleaned.isdigit():
                    numeric_count += 1

    if total_count == 0:
        return False

    return numeric_count / total_count >= threshold


def _escape_html(text: str) -> str:
    """HTML 특수문자를 이스케이프합니다."""
    if not text:
        return ""
    text = str(text).strip()
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    return text


# === 요소 병합 ===

def _merge_page_elements(elements: List[PageElement]) -> str:
    """요소들을 병합하여 최종 텍스트를 생성합니다."""
    if not elements:
        return ""

    # 테이블 영역 수집
    table_bboxes = []
    table_bboxes_expanded = []

    for e in elements:
        if e.element_type == ElementType.TABLE:
            table_bboxes.append(e.bbox)
            x0, y0, x1, y1 = e.bbox
            margin = 5.0
            table_bboxes_expanded.append((x0 - margin, y0 - margin, x1 + margin, y1 + margin))

    result_parts = []

    for element in elements:
        if element.element_type == ElementType.TABLE:
            result_parts.append("\n" + element.content + "\n")

        elif element.element_type == ElementType.IMAGE:
            result_parts.append(element.content)

        elif element.element_type == ElementType.TEXT:
            # 테이블과 겹치면 건너뛰기
            if _is_overlapping_with_tables(element.bbox, table_bboxes_expanded):
                continue

            result_parts.append(element.content + "\n")

    return "".join(result_parts)


def _is_overlapping_with_tables(
    text_bbox: Tuple[float, float, float, float],
    table_bboxes: List[Tuple[float, float, float, float]],
    overlap_threshold: float = 0.2
) -> bool:
    """텍스트 영역이 테이블 영역과 겹치는지 확인합니다."""
    tx0, ty0, tx1, ty1 = text_bbox
    text_area = max((tx1 - tx0) * (ty1 - ty0), 1)

    for bx0, by0, bx1, by1 in table_bboxes:
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

        # 텍스트가 테이블 내부에 완전히 포함되어 있는지
        if tx0 >= bx0 and ty0 >= by0 and tx1 <= bx1 and ty1 <= by1:
            return True

    return False
