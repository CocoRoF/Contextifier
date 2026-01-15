# your_package/document_processor/pdf_handler_ocr.py
"""
PDF Handler (OCR Enhanced) - 고도화된 OCR 기반 PDF 처리기

**핵심 설계 원칙**:
1. 텍스트는 텍스트로: PyMuPDF로 순수 텍스트만 추출
2. 비정형은 이미지로: 테이블, 차트, 그래프, 이미지 등은 영역을 이미지로 캡처
3. 인라인 참조: [image:{minio_path}] 형태로 원본 위치에 삽입
4. 위치 기반 정렬: 모든 요소를 페이지 내 Y 좌표 기준으로 정렬

**처리 흐름**:
1. PDF 페이지 분석 → 텍스트 블록 / 비정형 영역 분리
2. 비정형 영역 → 고해상도 이미지로 렌더링
3. 이미지 → MinIO 업로드 → minio_path 획득
4. 텍스트 + 이미지 참조 → 위치 순서대로 병합
5. 최종 문서 출력

**비정형 요소 감지 기준**:
- 이미지: PDF 내장 이미지 객체
- 테이블: 격자 구조 (rects/lines로 형성된 그리드)
- 차트/그래프: 벡터 드로잉 (paths, curves)이 밀집된 영역
- 다이어그램: 복합 그래픽 요소가 있는 영역
"""

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

from PIL import Image

from libs.core.processor.pdf_helpers.pdf_helper import (
    upload_image_to_minio,
    extract_pdf_metadata,
    format_metadata,
    find_image_position,
)

logger = logging.getLogger("document-processor")

# PyMuPDF import
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.error("PyMuPDF is required for PDF OCR processing but not available")

# pdfplumber import (테이블 영역 감지용)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available for table detection")


# === 타입 정의 ===

class ElementType(Enum):
    """페이지 요소 타입"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    VECTOR_GRAPHIC = "vector_graphic"


@dataclass
class PageRegion:
    """페이지 내 영역 정보"""
    region_type: ElementType
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    content: str = ""  # 텍스트 요소의 경우 실제 텍스트
    minio_path: Optional[str] = None  # 이미지 요소의 경우 MinIO 경로
    confidence: float = 1.0  # 감지 신뢰도

    @property
    def sort_key(self) -> Tuple[int, float, float]:
        """정렬 키: (페이지, Y좌표, X좌표)"""
        return (self.page_num, self.bbox[1], self.bbox[0])

    @property
    def area(self) -> float:
        """영역 넓이"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def height(self) -> float:
        """영역 높이"""
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> float:
        """영역 너비"""
        return self.bbox[2] - self.bbox[0]


@dataclass
class PageAnalysis:
    """페이지 분석 결과"""
    page_num: int
    page_width: float
    page_height: float
    text_regions: List[PageRegion] = field(default_factory=list)
    non_text_regions: List[PageRegion] = field(default_factory=list)

    @property
    def all_regions(self) -> List[PageRegion]:
        """모든 영역을 위치 순서대로 반환"""
        all_regs = self.text_regions + self.non_text_regions
        return sorted(all_regs, key=lambda r: r.sort_key)


@dataclass
class PageStructureAnalysis:
    """
    페이지 구조 분석 결과 (pdf_handler.py의 로직 재사용)

    **기계적 정보만 사용하여 판단**:
    1. 페이지 테두리: 페이지 가장자리에서 85% 이상 길이를 차지하는 선
    2. 테이블 구조: 내부에 격자를 형성하는 교차 선들
    """
    has_page_border: bool = False  # 장식용 페이지 테두리 존재 여부
    has_table_structure: bool = False  # 실제 테이블 구조 존재 여부
    table_regions: List[Tuple[float, float, float, float]] = field(default_factory=list)  # 테이블 bbox 리스트
    page_border_bbox: Optional[Tuple[float, float, float, float]] = None  # 페이지 테두리 bbox


def _analyze_page_structure_ocr(plumber_page, tolerance: float = 5.0) -> PageStructureAnalysis:
    """
    pdfplumber 페이지에서 페이지 구조를 분석합니다.

    **일반화된 기계적 접근**:
    - 페이지 테두리: 페이지 가장자리(상하좌우)에서 85% 이상 길이를 차지하는 선
    - 테이블 구조: 내부에서 교차하는 격자 패턴 (최소 3개 이상의 가로선/세로선 Y/X 위치)

    Args:
        plumber_page: pdfplumber Page 객체
        tolerance: 좌표 그룹화 허용 오차

    Returns:
        PageStructureAnalysis 결과
    """
    result = PageStructureAnalysis()

    page_width = float(plumber_page.width)
    page_height = float(plumber_page.height)

    # 모든 선 수집 (lines + rects 분해)
    all_lines = []

    # 1. lines 직접 수집
    if plumber_page.lines:
        for line in plumber_page.lines:
            x0, y0, x1, y1 = float(line["x0"]), float(line["top"]), float(line["x1"]), float(line["bottom"])
            all_lines.append((x0, y0, x1, y1))

    # 2. rects를 4개의 선으로 분해
    if plumber_page.rects:
        for rect in plumber_page.rects:
            x0, y0, x1, y1 = float(rect["x0"]), float(rect["top"]), float(rect["x1"]), float(rect["bottom"])
            # 상단 선
            all_lines.append((x0, y0, x1, y0))
            # 하단 선
            all_lines.append((x0, y1, x1, y1))
            # 좌측 선
            all_lines.append((x0, y0, x0, y1))
            # 우측 선
            all_lines.append((x1, y0, x1, y1))

    if not all_lines:
        return result

    # 선을 가로/세로로 분류
    h_lines = []  # (y, x0, x1) - 가로선
    v_lines = []  # (x, y0, y1) - 세로선

    for x0, y0, x1, y1 in all_lines:
        if abs(y1 - y0) < tolerance:  # 가로선
            h_lines.append((y0, min(x0, x1), max(x0, x1)))
        elif abs(x1 - x0) < tolerance:  # 세로선
            v_lines.append((x0, min(y0, y1), max(y0, y1)))

    # === 페이지 테두리 판별 ===
    # 기준: 페이지 가장자리에 위치하고 길이가 페이지 크기의 85% 이상
    edge_tolerance = page_width * 0.05  # 가장자리 판정 허용 범위 (5%)
    length_threshold = 0.85  # 85% 이상 길이

    border_lines = {
        "top": False, "bottom": False, "left": False, "right": False
    }
    border_coords = {"top": None, "bottom": None, "left": None, "right": None}

    for y, x0, x1 in h_lines:
        line_length = x1 - x0
        length_ratio = line_length / page_width

        # 상단 테두리
        if y < edge_tolerance and length_ratio >= length_threshold:
            border_lines["top"] = True
            border_coords["top"] = y
        # 하단 테두리
        elif y > page_height - edge_tolerance and length_ratio >= length_threshold:
            border_lines["bottom"] = True
            border_coords["bottom"] = y

    for x, y0, y1 in v_lines:
        line_length = y1 - y0
        length_ratio = line_length / page_height

        # 좌측 테두리
        if x < edge_tolerance and length_ratio >= length_threshold:
            border_lines["left"] = True
            border_coords["left"] = x
        # 우측 테두리
        elif x > page_width - edge_tolerance and length_ratio >= length_threshold:
            border_lines["right"] = True
            border_coords["right"] = x

    # 4변 중 최소 3변 이상 테두리가 있으면 페이지 테두리로 판정
    border_count = sum(border_lines.values())
    if border_count >= 3:
        result.has_page_border = True

        # 페이지 테두리 bbox 계산
        result.page_border_bbox = (
            border_coords["left"] or 0,
            border_coords["top"] or 0,
            border_coords["right"] or page_width,
            border_coords["bottom"] or page_height
        )
        logger.debug(f"Detected page border: {border_count} edges")

    # === 테이블 구조 판별 ===
    # 페이지 테두리를 제외한 내부 선들만 분석
    internal_h_lines = []
    internal_v_lines = []

    for y, x0, x1 in h_lines:
        # 페이지 테두리 선은 제외
        if result.has_page_border:
            if (border_coords["top"] and abs(y - border_coords["top"]) < tolerance):
                continue
            if (border_coords["bottom"] and abs(y - border_coords["bottom"]) < tolerance):
                continue
        internal_h_lines.append((y, x0, x1))

    for x, y0, y1 in v_lines:
        # 페이지 테두리 선은 제외
        if result.has_page_border:
            if (border_coords["left"] and abs(x - border_coords["left"]) < tolerance):
                continue
            if (border_coords["right"] and abs(x - border_coords["right"]) < tolerance):
                continue
        internal_v_lines.append((x, y0, y1))

    # 내부 선으로 격자 패턴 분석
    if len(internal_h_lines) >= 2 and len(internal_v_lines) >= 2:
        # Y 좌표 그룹화 (허용 오차 내 동일 Y는 같은 행)
        y_positions = sorted(set(round(y / tolerance) * tolerance for y, _, _ in internal_h_lines))
        x_positions = sorted(set(round(x / tolerance) * tolerance for x, _, _ in internal_v_lines))

        # 최소 3개 이상의 가로선 Y위치 OR 세로선 X위치가 있어야 테이블
        if len(y_positions) >= 3 or len(x_positions) >= 3:
            result.has_table_structure = True

            # 테이블 영역 bbox 계산
            min_x = min(x0 for _, x0, _ in internal_h_lines)
            max_x = max(x1 for _, _, x1 in internal_h_lines)
            min_y = min(y for y, _, _ in internal_h_lines)
            max_y = max(y for y, _, _ in internal_h_lines)

            result.table_regions.append((min_x, min_y, max_x, max_y))
            logger.debug(f"Detected table structure: {len(y_positions)} h-lines, {len(x_positions)} v-lines")

    return result


# === 메인 함수 ===

async def extract_text_from_pdf_ocr(
    file_path: str,
    current_config: Dict[str, Any] = None,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    고도화된 OCR 기반 PDF 텍스트 추출.

    **처리 방식**:
    - 텍스트: 그대로 추출
    - 비정형(이미지/테이블/차트): 영역을 이미지로 캡처 → MinIO 업로드 → [image:path] 참조

    Args:
        file_path: PDF 파일 경로
        current_config: 설정 딕셔너리
        app_db: 데이터베이스 연결 (이미지 메타데이터 저장용)
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (비정형 요소는 [image:minio_path] 형태로 포함)
    """
    if current_config is None:
        current_config = {}

    logger.info(f"PDF OCR processing started: {file_path}")

    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF not available, cannot process PDF")
        return "[PDF 파일 처리 실패: PyMuPDF가 설치되지 않음]"

    try:
        result_parts = []

        # PDF 열기
        doc = fitz.open(file_path)
        total_pages = len(doc)
        logger.info(f"PDF has {total_pages} pages")

        # 메타데이터 추출 (extract_default_metadata가 True인 경우에만)
        if extract_default_metadata:
            metadata = extract_pdf_metadata(doc)
            if metadata:
                metadata_str = format_metadata(metadata)
                result_parts.append(metadata_str)

        # 이미지 중복 방지용 세트
        processed_image_hashes: Set[str] = set()

        # 페이지별 처리
        for page_num in range(total_pages):
            page = doc.load_page(page_num)

            # 페이지 분석
            page_analysis = await _analyze_page(
                doc, page, page_num, file_path, app_db, processed_image_hashes
            )

            # 페이지 콘텐츠 생성
            page_content = _render_page_content(page_analysis)

            # 페이지 헤더 추가
            result_parts.append(f"\n<페이지 번호> {page_num + 1} </페이지 번호>\n")
            result_parts.append(page_content)

        doc.close()

        result = "".join(result_parts)
        logger.info(f"PDF OCR processing completed: {len(result)} characters extracted")

        return result

    except Exception as e:
        logger.error(f"Error in PDF OCR processing: {e}")
        logger.debug(traceback.format_exc())
        return f"[PDF 파일 처리 실패: {str(e)}]"


# === 페이지 분석 ===

async def _analyze_page(
    doc,
    page,
    page_num: int,
    file_path: str,
    app_db,
    processed_image_hashes: Set[str]
) -> PageAnalysis:
    """
    페이지를 분석하여 텍스트 영역과 비정형 영역을 분리합니다.

    **분석 순서**:
    1. 내장 이미지 감지 및 추출
    2. 테이블 영역 감지
    3. 벡터 그래픽/차트 영역 감지
    4. 텍스트 블록 추출 (비정형 영역과 겹치지 않는 부분만)

    Args:
        doc: PyMuPDF Document
        page: PyMuPDF Page
        page_num: 페이지 번호 (0-indexed)
        file_path: PDF 파일 경로
        app_db: 데이터베이스 연결
        processed_image_hashes: 처리된 이미지 해시 세트

    Returns:
        PageAnalysis 객체
    """
    page_width = page.rect.width
    page_height = page.rect.height

    analysis = PageAnalysis(
        page_num=page_num,
        page_width=page_width,
        page_height=page_height
    )

    # 1. 내장 이미지 감지 및 추출
    image_regions = await _detect_and_extract_images(
        doc, page, page_num, app_db, processed_image_hashes
    )
    analysis.non_text_regions.extend(image_regions)

    # 2. 테이블 영역 감지
    table_regions = await _detect_table_regions(
        doc, page, page_num, file_path, app_db, processed_image_hashes
    )
    # 이미지 영역과 겹치지 않는 테이블만 추가
    for table_reg in table_regions:
        if not _is_overlapping_with_regions(table_reg.bbox, [r.bbox for r in image_regions]):
            analysis.non_text_regions.append(table_reg)

    # 3. 벡터 그래픽/차트 영역 감지
    # NOTE: 벡터 그래픽 감지는 불확실한 추론을 유발하므로 비활성화
    # 차트/그래프 등은 별도의 확실한 메타데이터가 있을 때만 처리해야 함
    # vector_regions = await _detect_vector_graphics(
    #     doc, page, page_num, app_db, processed_image_hashes
    # )
    # # 기존 영역과 겹치지 않는 것만 추가
    # existing_bboxes = [r.bbox for r in analysis.non_text_regions]
    # for vec_reg in vector_regions:
    #     if not _is_overlapping_with_regions(vec_reg.bbox, existing_bboxes, threshold=0.5):
    #         analysis.non_text_regions.append(vec_reg)
    #         existing_bboxes.append(vec_reg.bbox)

    # 4. 텍스트 블록 추출 (비정형 영역과 겹치지 않는 부분만)
    text_regions = _extract_text_regions(page, page_num, analysis.non_text_regions)
    analysis.text_regions.extend(text_regions)

    logger.debug(
        f"Page {page_num + 1}: {len(analysis.text_regions)} text regions, "
        f"{len(analysis.non_text_regions)} non-text regions"
    )

    return analysis


# === 이미지 감지 및 추출 ===

async def _detect_and_extract_images(
    doc,
    page,
    page_num: int,
    app_db,
    processed_image_hashes: Set[str],
    min_size: int = 50,
    min_area: int = 2500
) -> List[PageRegion]:
    """
    페이지에서 내장 이미지를 감지하고 추출합니다.

    Args:
        doc: PyMuPDF Document
        page: PyMuPDF Page
        page_num: 페이지 번호
        app_db: 데이터베이스 연결
        processed_image_hashes: 처리된 이미지 해시 세트
        min_size: 최소 이미지 크기 (너비/높이)
        min_area: 최소 이미지 면적

    Returns:
        이미지 영역 리스트
    """
    regions = []

    try:
        image_list = page.get_images(full=True)

        for img_info in image_list:
            try:
                xref = img_info[0]

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

                if width < min_size or height < min_size:
                    continue
                if width * height < min_area:
                    continue

                # 중복 확인 (해시 기반)
                image_hash = hashlib.md5(image_data).hexdigest()
                if image_hash in processed_image_hashes:
                    continue

                # 이미지 위치 찾기
                bbox = find_image_position(page, xref)
                if not bbox:
                    # 위치를 찾지 못하면 건너뛰기
                    continue

                # MinIO에 업로드
                minio_path = upload_image_to_minio(image_data, app_db)

                if minio_path:
                    processed_image_hashes.add(image_hash)

                    regions.append(PageRegion(
                        region_type=ElementType.IMAGE,
                        bbox=bbox,
                        page_num=page_num,
                        minio_path=minio_path,
                        confidence=1.0
                    ))
                    logger.debug(f"Extracted image from page {page_num + 1}: {width}x{height}")

            except Exception as e:
                logger.warning(f"Error extracting image from page {page_num}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Error detecting images on page {page_num}: {e}")

    return regions


# === 테이블 영역 감지 ===

async def _detect_table_regions(
    doc,
    page,
    page_num: int,
    file_path: str,
    app_db,
    processed_image_hashes: Set[str],
    min_table_area: float = 5000
) -> List[PageRegion]:
    """
    페이지에서 테이블 영역을 감지하고 이미지로 캡처합니다.

    **핵심 원칙 (일반화된 기계적 접근)**:
    1. 먼저 페이지 구조를 분석하여 페이지 테두리와 실제 테이블 구분
    2. 페이지 테두리는 장식용이므로 제외
    3. 내부에 격자 구조가 있는 영역만 테이블로 인식

    Args:
        doc: PyMuPDF Document
        page: PyMuPDF Page
        page_num: 페이지 번호
        file_path: PDF 파일 경로
        app_db: 데이터베이스 연결
        processed_image_hashes: 처리된 이미지 해시 세트
        min_table_area: 최소 테이블 면적

    Returns:
        테이블 영역 리스트
    """
    regions = []
    table_bboxes = []

    # pdfplumber로 테이블 감지 + 페이지 구조 분석
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_num < len(pdf.pages):
                    plumber_page = pdf.pages[page_num]

                    # 1. 페이지 구조 분석 (페이지 테두리 vs 실제 테이블)
                    structure_analysis = _analyze_page_structure_ocr(plumber_page)

                    if structure_analysis.has_page_border:
                        logger.debug(f"Page {page_num + 1}: Detected page border, using structure analysis")

                    # 2. 테이블 감지
                    if structure_analysis.has_table_structure:
                        # 구조 분석에서 테이블 영역이 명확하게 식별된 경우 사용
                        for bbox in structure_analysis.table_regions:
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if area >= min_table_area:
                                table_bboxes.append(bbox)
                        logger.debug(f"Page {page_num + 1}: Using structure analysis for table: {table_bboxes}")
                    else:
                        # pdfplumber find_tables 사용
                        tables = plumber_page.find_tables()

                        page_width = float(plumber_page.width)
                        page_height = float(plumber_page.height)
                        page_area = page_width * page_height

                        for table in tables:
                            bbox = table.bbox  # (x0, y0, x1, y1)
                            table_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                            # 페이지 테두리인 경우 (페이지의 85% 이상 차지) 제외
                            if table_area > page_area * 0.85:
                                logger.debug(f"Page {page_num + 1}: Skipping table covering {table_area/page_area*100:.1f}% of page")
                                continue

                            if table_area >= min_table_area:
                                table_bboxes.append(bbox)

        except Exception as e:
            logger.debug(f"pdfplumber table detection failed: {e}")

    # 테이블 영역을 이미지로 캡처
    for bbox in table_bboxes:
        try:
            # 영역 이미지 렌더링
            image_data = _render_region_to_image(page, bbox)

            if image_data:
                # 중복 확인
                image_hash = hashlib.md5(image_data).hexdigest()
                if image_hash in processed_image_hashes:
                    continue

                # MinIO 업로드
                minio_path = upload_image_to_minio(image_data, app_db)

                if minio_path:
                    processed_image_hashes.add(image_hash)

                    regions.append(PageRegion(
                        region_type=ElementType.TABLE,
                        bbox=bbox,
                        page_num=page_num,
                        minio_path=minio_path,
                        confidence=0.9
                    ))
                    logger.debug(f"Captured table region on page {page_num + 1}: {bbox}")

        except Exception as e:
            logger.warning(f"Error capturing table region: {e}")

    return regions


def _detect_grid_patterns(page, min_lines: int = 3) -> List[Tuple[float, float, float, float]]:
    """
    페이지에서 격자 패턴(테이블)을 감지합니다.

    **감지 기준**:
    - 가로선과 세로선이 교차하는 패턴
    - 최소 min_lines개 이상의 가로선과 세로선

    Args:
        page: PyMuPDF Page
        min_lines: 최소 선 개수

    Returns:
        테이블 bbox 리스트
    """
    table_bboxes = []

    try:
        # 페이지의 drawings 가져오기
        drawings = page.get_drawings()

        if not drawings:
            return []

        # 선 분류 (가로/세로)
        h_lines = []  # (y, x0, x1)
        v_lines = []  # (x, y0, y1)

        for drawing in drawings:
            if drawing.get("type") == "l":  # line
                items = drawing.get("items", [])
                for item in items:
                    if item[0] == "l":  # line segment
                        p1, p2 = item[1], item[2]
                        x1, y1 = p1
                        x2, y2 = p2

                        # 가로선 (y 좌표 유사)
                        if abs(y2 - y1) < 3:
                            h_lines.append((y1, min(x1, x2), max(x1, x2)))
                        # 세로선 (x 좌표 유사)
                        elif abs(x2 - x1) < 3:
                            v_lines.append((x1, min(y1, y2), max(y1, y2)))

            elif drawing.get("type") == "re":  # rectangle
                rect = drawing.get("rect")
                if rect:
                    x0, y0, x1, y1 = rect
                    # 사각형을 4개의 선으로 분해
                    h_lines.append((y0, x0, x1))
                    h_lines.append((y1, x0, x1))
                    v_lines.append((x0, y0, y1))
                    v_lines.append((x1, y0, y1))

        # 격자 패턴 감지
        if len(h_lines) >= min_lines and len(v_lines) >= min_lines:
            # Y 좌표로 그룹화
            h_lines_sorted = sorted(h_lines, key=lambda l: l[0])
            v_lines_sorted = sorted(v_lines, key=lambda l: l[0])

            # 테이블 bbox 계산
            min_x = min(l[1] for l in h_lines_sorted)
            max_x = max(l[2] for l in h_lines_sorted)
            min_y = min(l[0] for l in h_lines_sorted)
            max_y = max(l[0] for l in h_lines_sorted)

            # 유효한 테이블 크기인지 확인
            if max_x - min_x > 50 and max_y - min_y > 30:
                table_bboxes.append((min_x, min_y, max_x, max_y))

    except Exception as e:
        logger.debug(f"Grid pattern detection error: {e}")

    return table_bboxes


# === 벡터 그래픽/차트 감지 ===

async def _detect_vector_graphics(
    doc,
    page,
    page_num: int,
    app_db,
    processed_image_hashes: Set[str],
    min_drawing_density: int = 20,
    min_area: float = 15000,
    min_curve_ratio: float = 0.2
) -> List[PageRegion]:
    """
    페이지에서 벡터 그래픽(차트, 다이어그램 등)을 감지합니다.

    **감지 기준**:
    - 곡선(curve), 원(circle), 호(arc) 등 비직선 요소가 일정 비율 이상
    - 다양한 색상의 채워진 도형이 밀집된 경우
    - 페이지 테두리나 단순 테이블 선은 제외

    **제외 기준**:
    - 페이지 크기의 80% 이상을 차지하는 영역
    - 직선만으로 구성된 영역 (테이블)
    - 단일 색상의 사각형 (배경, 테두리)

    Args:
        doc: PyMuPDF Document
        page: PyMuPDF Page
        page_num: 페이지 번호
        app_db: 데이터베이스 연결
        processed_image_hashes: 처리된 이미지 해시 세트
        min_drawing_density: 최소 드로잉 밀도
        min_area: 최소 영역 면적
        min_curve_ratio: 최소 곡선 비율

    Returns:
        벡터 그래픽 영역 리스트
    """
    regions = []

    try:
        drawings = page.get_drawings()

        if not drawings or len(drawings) < min_drawing_density:
            return []

        page_width = page.rect.width
        page_height = page.rect.height
        page_area = page_width * page_height

        # 드로잉 분석: 곡선 요소와 복잡한 채움 패턴 찾기
        curve_drawings = []  # 곡선이 포함된 드로잉
        filled_drawings = []  # 채워진 도형
        colors_used = set()  # 사용된 색상

        for drawing in drawings:
            items = drawing.get("items", [])
            rect = drawing.get("rect")

            if not rect:
                continue

            # 페이지 테두리 크기의 영역은 제외 (85% 이상)
            draw_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            if draw_area > page_area * 0.7:
                continue

            # 곡선 요소 확인
            has_curve = False
            for item in items:
                if item[0] in ("c", "qu", "v", "y"):  # curve types
                    has_curve = True
                    break

            if has_curve:
                curve_drawings.append(rect)

            # 채워진 도형 (다양한 색상 확인)
            fill = drawing.get("fill")
            if fill:
                # 색상 정보 추출 (RGB 튜플)
                if isinstance(fill, (list, tuple)) and len(fill) >= 3:
                    color_key = (int(fill[0]*255), int(fill[1]*255), int(fill[2]*255))
                    colors_used.add(color_key)
                    filled_drawings.append(rect)

        # 차트 조건 검증:
        # 1. 곡선 요소가 있어야 함 (테이블은 직선만 사용)
        # 2. 다양한 색상이 사용되어야 함 (테이블은 보통 1-2색)
        # 3. 충분한 밀도가 있어야 함

        total_drawings = len(curve_drawings) + len(filled_drawings)

        # 곡선 비율이 너무 낮으면 테이블/테두리일 가능성 높음
        if total_drawings > 0:
            curve_ratio = len(curve_drawings) / total_drawings
            if curve_ratio < min_curve_ratio and len(colors_used) < 3:
                logger.debug(f"Page {page_num + 1}: Skipping vector detection - low curve ratio ({curve_ratio:.2f})")
                return []

        # 색상 다양성이 낮으면 (2색 이하) 차트가 아닐 가능성
        if len(colors_used) < 3 and len(curve_drawings) < 5:
            logger.debug(f"Page {page_num + 1}: Skipping vector detection - low color diversity ({len(colors_used)})")
            return []

        # 클러스터링: 인접한 복잡한 드로잉들을 그룹화
        all_complex = curve_drawings + filled_drawings

        if len(all_complex) < 5:
            return []

        clusters = _cluster_drawings(all_complex, distance_threshold=30)

        for cluster_bbox in clusters:
            area = (cluster_bbox[2] - cluster_bbox[0]) * (cluster_bbox[3] - cluster_bbox[1])

            if area < min_area:
                continue

            # 페이지의 너무 큰 비율을 차지하면 제외
            if area > page_area * 0.6:
                continue

            # 영역 이미지 렌더링
            try:
                image_data = _render_region_to_image(page, cluster_bbox, scale=2.0)

                if image_data:
                    image_hash = hashlib.md5(image_data).hexdigest()
                    if image_hash in processed_image_hashes:
                        continue

                    minio_path = upload_image_to_minio(image_data, app_db)

                    if minio_path:
                        processed_image_hashes.add(image_hash)

                        regions.append(PageRegion(
                            region_type=ElementType.CHART,
                            bbox=cluster_bbox,
                            page_num=page_num,
                            minio_path=minio_path,
                            confidence=0.8
                        ))
                        logger.debug(f"Captured vector graphic on page {page_num + 1}: {cluster_bbox}")

            except Exception as e:
                logger.warning(f"Error capturing vector graphic: {e}")

    except Exception as e:
        logger.debug(f"Vector graphics detection error: {e}")

    return regions


def _cluster_drawings(
    rects: List[Tuple[float, float, float, float]],
    distance_threshold: float = 50
) -> List[Tuple[float, float, float, float]]:
    """
    인접한 드로잉 사각형들을 클러스터링합니다.

    Args:
        rects: 사각형 bbox 리스트
        distance_threshold: 클러스터링 거리 임계값

    Returns:
        클러스터 bbox 리스트
    """
    if not rects:
        return []

    # 단순 그리디 클러스터링
    clusters = []
    used = set()

    for i, rect in enumerate(rects):
        if i in used:
            continue

        # 새 클러스터 시작
        cluster_x0, cluster_y0, cluster_x1, cluster_y1 = rect
        used.add(i)

        # 인접한 사각형들 병합
        changed = True
        while changed:
            changed = False
            for j, other_rect in enumerate(rects):
                if j in used:
                    continue

                # 거리 계산 (bbox 간 최소 거리)
                dist = _rect_distance(
                    (cluster_x0, cluster_y0, cluster_x1, cluster_y1),
                    other_rect
                )

                if dist < distance_threshold:
                    # 클러스터에 병합
                    cluster_x0 = min(cluster_x0, other_rect[0])
                    cluster_y0 = min(cluster_y0, other_rect[1])
                    cluster_x1 = max(cluster_x1, other_rect[2])
                    cluster_y1 = max(cluster_y1, other_rect[3])
                    used.add(j)
                    changed = True

        clusters.append((cluster_x0, cluster_y0, cluster_x1, cluster_y1))

    return clusters


def _rect_distance(
    rect1: Tuple[float, float, float, float],
    rect2: Tuple[float, float, float, float]
) -> float:
    """두 사각형 간의 최소 거리를 계산합니다."""
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2

    # X 거리
    if x1_1 < x0_2:
        dx = x0_2 - x1_1
    elif x1_2 < x0_1:
        dx = x0_1 - x1_2
    else:
        dx = 0

    # Y 거리
    if y1_1 < y0_2:
        dy = y0_2 - y1_1
    elif y1_2 < y0_1:
        dy = y0_1 - y1_2
    else:
        dy = 0

    return (dx ** 2 + dy ** 2) ** 0.5


# === 텍스트 영역 추출 ===

def _extract_text_regions(
    page,
    page_num: int,
    non_text_regions: List[PageRegion]
) -> List[PageRegion]:
    """
    페이지에서 텍스트 블록을 추출합니다 (비정형 영역과 겹치지 않는 부분만).

    Args:
        page: PyMuPDF Page
        page_num: 페이지 번호
        non_text_regions: 비정형 영역 리스트

    Returns:
        텍스트 영역 리스트
    """
    regions = []
    non_text_bboxes = [r.bbox for r in non_text_regions]

    try:
        # 텍스트 블록 추출
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") != 0:  # 텍스트 블록만
                continue

            bbox = block["bbox"]

            # 비정형 영역과 겹치는지 확인
            if _is_overlapping_with_regions(bbox, non_text_bboxes, threshold=0.3):
                continue

            # 텍스트 추출
            text_parts = []
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                if line_text.strip():
                    text_parts.append(line_text)

            text = "\n".join(text_parts)

            if text.strip():
                regions.append(PageRegion(
                    region_type=ElementType.TEXT,
                    bbox=bbox,
                    page_num=page_num,
                    content=text
                ))

    except Exception as e:
        logger.warning(f"Error extracting text regions from page {page_num}: {e}")

    return regions


# === 영역 이미지 렌더링 ===

def _render_region_to_image(
    page,
    bbox: Tuple[float, float, float, float],
    scale: float = 2.0,
    margin: float = 5.0
) -> Optional[bytes]:
    """
    페이지의 특정 영역을 고해상도 이미지로 렌더링합니다.

    Args:
        page: PyMuPDF Page
        bbox: 영역 bbox (x0, y0, x1, y1)
        scale: 확대 배율 (기본 2.0 = 144 DPI)
        margin: 여백 (포인트)

    Returns:
        PNG 이미지 데이터 (bytes) 또는 None
    """
    try:
        x0, y0, x1, y1 = bbox

        # 여백 추가
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(page.rect.width, x1 + margin)
        y1 = min(page.rect.height, y1 + margin)

        # 클립 영역 설정
        clip_rect = fitz.Rect(x0, y0, x1, y1)

        # 변환 매트릭스 (확대)
        mat = fitz.Matrix(scale, scale)

        # 픽스맵 생성
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)

        # PNG로 변환
        png_data = pix.tobytes("png")

        return png_data

    except Exception as e:
        logger.warning(f"Error rendering region to image: {e}")
        return None


# === 겹침 확인 ===

def _is_overlapping_with_regions(
    bbox: Tuple[float, float, float, float],
    region_bboxes: List[Tuple[float, float, float, float]],
    threshold: float = 0.2
) -> bool:
    """
    bbox가 region_bboxes 중 하나와 겹치는지 확인합니다.

    Args:
        bbox: 확인할 bbox
        region_bboxes: 비교할 영역 bbox 리스트
        threshold: 겹침 비율 임계값

    Returns:
        겹침 여부
    """
    x0, y0, x1, y1 = bbox
    area = max((x1 - x0) * (y1 - y0), 1)

    for reg_bbox in region_bboxes:
        rx0, ry0, rx1, ry1 = reg_bbox

        # 교차 영역 계산
        ix0 = max(x0, rx0)
        iy0 = max(y0, ry0)
        ix1 = min(x1, rx1)
        iy1 = min(y1, ry1)

        if ix0 < ix1 and iy0 < iy1:
            intersection_area = (ix1 - ix0) * (iy1 - iy0)
            overlap_ratio = intersection_area / area

            if overlap_ratio > threshold:
                return True

        # 완전 내포 확인
        if x0 >= rx0 and y0 >= ry0 and x1 <= rx1 and y1 <= ry1:
            return True

    return False


# === 페이지 콘텐츠 렌더링 ===

def _render_page_content(page_analysis: PageAnalysis) -> str:
    """
    페이지 분석 결과를 최종 텍스트로 렌더링합니다.

    **렌더링 규칙**:
    - 모든 요소를 Y 좌표 순서대로 정렬
    - 텍스트: 그대로 출력
    - 비정형(이미지/테이블/차트): [image:minio_path] 형태로 출력

    Args:
        page_analysis: 페이지 분석 결과

    Returns:
        렌더링된 텍스트
    """
    parts = []

    # 모든 영역을 위치 순서대로 정렬
    all_regions = page_analysis.all_regions

    for region in all_regions:
        if region.region_type == ElementType.TEXT:
            # 텍스트 영역
            if region.content.strip():
                parts.append(region.content.strip())
        else:
            # 비정형 영역 (이미지 참조)
            if region.minio_path:
                parts.append(f"\n[image:{region.minio_path}]\n")

    return "\n".join(parts)
