"""
Layout Block Detector for PDF Handler V4

복잡한 다단 레이아웃(신문, 잡지 등)을 의미론적 블록 단위로 분할합니다.

=============================================================================
V4 고도화 핵심 개념:
=============================================================================
전체 페이지를 하나의 이미지로 처리하는 대신,
페이지를 **의미론적/논리적 블록 단위**로 분할하여 각각 PNG로 저장합니다.

이를 통해:
1. LLM이 각 블록을 **개별적으로** 해석 가능
2. 해상도 문제 해결 (블록별로 고해상도 유지)
3. 읽기 순서 보존
4. 컨텍스트 분리 (광고/기사/표 구분)

=============================================================================
레이아웃 분석 알고리즘:
=============================================================================

Phase 1: 기초 분석
┌─────────────────────────────────────────────────────────────────┐
│  1. 텍스트 블록 추출                                              │
│  2. 이미지/그래픽 영역 추출                                        │
│  3. 드로잉(선, 박스) 추출                                          │
│  4. 테이블 영역 식별                                               │
└─────────────────────────────────────────────────────────────────┘

Phase 2: 컬럼 감지 (다단 레이아웃)
┌─────────────────────────────────────────────────────────────────┐
│  1. X 좌표 기반 클러스터링                                         │
│  2. 컬럼 경계선 식별                                               │
│  3. 컬럼별 콘텐츠 그룹화                                           │
└─────────────────────────────────────────────────────────────────┘

Phase 3: 의미론적 블록 클러스터링
┌─────────────────────────────────────────────────────────────────┐
│  1. 인접 요소 연결 (거리 기반)                                      │
│  2. 헤드라인-본문 연결 (폰트 크기 분석)                             │
│  3. 이미지-캡션 연결 (위치 관계)                                    │
│  4. 구분선/박스 기반 영역 분리                                      │
└─────────────────────────────────────────────────────────────────┘

Phase 4: 블록 최적화 및 정렬
┌─────────────────────────────────────────────────────────────────┐
│  1. 작은 블록 병합                                                 │
│  2. 겹침 해결                                                      │
│  3. 읽기 순서 결정 (컬럼 → 상→하)                                   │
│  4. 블록 bbox 정규화                                               │
└─────────────────────────────────────────────────────────────────┘

=============================================================================
블록 유형:
=============================================================================
- ARTICLE: 기사 블록 (헤드라인 + 본문)
- IMAGE_WITH_CAPTION: 이미지 + 캡션
- TABLE: 테이블 영역
- ADVERTISEMENT: 광고 영역 (박스로 분리된)
- SIDEBAR: 사이드바/인포박스
- HEADER_FOOTER: 헤더/푸터
- UNKNOWN: 분류 불가
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto
from collections import defaultdict
import math

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class LayoutBlockType(Enum):
    """레이아웃 블록 유형"""
    ARTICLE = auto()            # 기사 (헤드라인 + 본문)
    IMAGE_WITH_CAPTION = auto() # 이미지 + 캡션
    STANDALONE_IMAGE = auto()   # 독립 이미지
    TABLE = auto()              # 테이블
    ADVERTISEMENT = auto()      # 광고
    SIDEBAR = auto()            # 사이드바/인포박스
    HEADER = auto()             # 페이지 헤더
    FOOTER = auto()             # 페이지 푸터
    COLUMN_BLOCK = auto()       # 컬럼 단위 블록
    UNKNOWN = auto()            # 분류 불가


@dataclass
class ContentElement:
    """페이지 내 콘텐츠 요소"""
    element_type: str  # 'text', 'image', 'drawing', 'table'
    bbox: Tuple[float, float, float, float]
    content: Optional[str] = None
    
    # 텍스트 속성
    font_size: float = 0.0
    is_bold: bool = False
    text_length: int = 0
    
    # 이미지 속성
    image_area: float = 0.0
    
    # 그룹 ID (클러스터링 후 할당)
    group_id: int = -1


@dataclass
class LayoutBlock:
    """의미론적 레이아웃 블록"""
    block_id: int
    block_type: LayoutBlockType
    bbox: Tuple[float, float, float, float]
    
    # 포함된 요소들
    elements: List[ContentElement] = field(default_factory=list)
    
    # 컬럼 정보
    column_index: int = 0
    
    # 읽기 순서 (0부터 시작)
    reading_order: int = 0
    
    # 신뢰도 (0.0 ~ 1.0)
    confidence: float = 1.0
    
    # 메타데이터
    metadata: Dict = field(default_factory=dict)
    
    @property
    def area(self) -> float:
        """블록 면적"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def center(self) -> Tuple[float, float]:
        """블록 중심점"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )


@dataclass
class ColumnInfo:
    """컬럼 정보"""
    index: int
    x_start: float
    x_end: float
    
    # 컬럼 내 블록들
    blocks: List[LayoutBlock] = field(default_factory=list)
    
    @property
    def width(self) -> float:
        return self.x_end - self.x_start


@dataclass
class LayoutAnalysisResult:
    """레이아웃 분석 결과"""
    page_num: int
    page_size: Tuple[float, float]
    
    # 컬럼 정보
    columns: List[ColumnInfo] = field(default_factory=list)
    column_count: int = 1
    
    # 레이아웃 블록 (읽기 순서대로 정렬됨)
    blocks: List[LayoutBlock] = field(default_factory=list)
    
    # 헤더/푸터 영역
    header_region: Optional[Tuple[float, float, float, float]] = None
    footer_region: Optional[Tuple[float, float, float, float]] = None
    
    # 통계
    total_text_elements: int = 0
    total_image_elements: int = 0
    
    # 분석 신뢰도
    confidence: float = 1.0


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LayoutDetectorConfig:
    """레이아웃 감지 설정"""
    
    # 컬럼 감지 설정
    MIN_COLUMN_GAP: float = 20.0        # 컬럼 간 최소 간격 (pt)
    COLUMN_CLUSTER_TOLERANCE: float = 30.0  # X 좌표 클러스터링 허용치 (pt)
    
    # 블록 클러스터링 설정
    ELEMENT_PROXIMITY_THRESHOLD: float = 15.0  # 요소 인접 판단 거리 (pt)
    VERTICAL_MERGE_THRESHOLD: float = 40.0     # 수직 병합 거리 (pt) - 더 적극적 병합
    HORIZONTAL_MERGE_THRESHOLD: float = 15.0   # 수평 병합 거리 (pt) - 더 적극적 병합
    
    # 헤드라인 감지 설정
    HEADLINE_FONT_RATIO: float = 1.3    # 본문 대비 헤드라인 폰트 비율
    HEADLINE_MIN_SIZE: float = 14.0     # 최소 헤드라인 폰트 크기 (pt)
    
    # 이미지-캡션 연결 설정
    CAPTION_MAX_DISTANCE: float = 30.0  # 이미지-캡션 최대 거리 (pt)
    CAPTION_MAX_HEIGHT: float = 50.0    # 캡션 최대 높이 (pt)
    
    # 헤더/푸터 설정
    HEADER_MAX_HEIGHT: float = 60.0     # 헤더 최대 높이 (pt)
    FOOTER_MAX_HEIGHT: float = 60.0     # 푸터 최대 높이 (pt)
    HEADER_FOOTER_MARGIN: float = 0.1   # 페이지 상/하단 마진 비율
    
    # 블록 최소 크기 (★ 핵심: 작은 블록은 병합 대상)
    MIN_BLOCK_WIDTH: float = 80.0       # 최소 블록 너비 (pt) - 상향
    MIN_BLOCK_HEIGHT: float = 60.0      # 최소 블록 높이 (pt) - 상향
    MIN_BLOCK_AREA: float = 15000.0     # 최소 블록 면적 (pt²) - 대폭 상향 (~100x150pt)
    
    # 블록 수 목표 (★ 신규: 너무 많은 블록 방지)
    TARGET_MIN_BLOCKS: int = 3          # 페이지당 최소 블록 수
    TARGET_MAX_BLOCKS: int = 10         # 페이지당 최대 블록 수 (5컬럼 신문 고려)
    AGGRESSIVE_MERGE_THRESHOLD: int = 15  # 이 이상이면 적극적 병합
    
    # 광고 감지
    AD_BOX_DETECTION: bool = True       # 박스로 둘러싸인 광고 감지
    AD_MIN_BOX_AREA: float = 10000.0    # 광고로 판단하는 최소 박스 면적
    
    # 구분선 감지
    SEPARATOR_MIN_LENGTH_RATIO: float = 0.3  # 구분선 최소 길이 (페이지 너비 대비)
    SEPARATOR_MAX_THICKNESS: float = 3.0     # 구분선 최대 두께 (pt)


# ============================================================================
# Layout Block Detector
# ============================================================================

class LayoutBlockDetector:
    """
    레이아웃 블록 감지기
    
    복잡한 다단 레이아웃을 의미론적 블록 단위로 분할합니다.
    """
    
    def __init__(
        self, 
        page, 
        page_num: int,
        config: Optional[LayoutDetectorConfig] = None
    ):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed)
            config: 감지 설정
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is required for LayoutBlockDetector")
        
        self.page = page
        self.page_num = page_num
        self.config = config or LayoutDetectorConfig()
        
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        
        # 캐시
        self._text_dict: Optional[Dict] = None
        self._drawings: Optional[List] = None
        self._images: Optional[List] = None
        
        # 내부 상태
        self._elements: List[ContentElement] = []
        self._separators: List[Tuple[float, float, float, float]] = []
        self._boxes: List[Tuple[float, float, float, float]] = []
    
    def detect(self) -> LayoutAnalysisResult:
        """
        레이아웃 블록을 감지합니다.
        
        Returns:
            LayoutAnalysisResult 객체
        """
        columns = [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        header_region = None
        footer_region = None
        blocks = []
        
        try:
            # Phase 1: 기초 분석
            try:
                self._extract_elements()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 1 (_extract_elements) failed: {e}")
                self._elements = []
            
            try:
                self._extract_separators_and_boxes()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 1 (_extract_separators_and_boxes) failed: {e}")
                self._separators = []
                self._boxes = []
            
            # Phase 2: 컬럼 감지
            try:
                columns = self._detect_columns()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 2 (_detect_columns) failed: {e}")
                columns = [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
            
            # Phase 3: 헤더/푸터 감지
            try:
                header_region, footer_region = self._detect_header_footer()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 3 (_detect_header_footer) failed: {e}")
                header_region = None
                footer_region = None
            
            # Phase 4: 의미론적 블록 클러스터링
            try:
                blocks = self._cluster_into_blocks(columns, header_region, footer_region)
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 4 (_cluster_into_blocks) failed: {e}")
                # 폴백: 컬럼 기반 단순 블록 생성
                blocks = self._create_column_based_blocks(columns)
            
            # Phase 5: 블록 분류
            try:
                self._classify_blocks(blocks)
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 5 (_classify_blocks) failed: {e}")
            
            # Phase 6: 블록 최적화 및 정렬
            try:
                blocks = self._optimize_and_sort_blocks(blocks, columns)
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 6 (_optimize_and_sort_blocks) failed: {e}")
        
        except Exception as e:
            logger.error(f"[LayoutBlockDetector] Critical error during detection: {e}")
            # 최소한 전체 페이지를 하나의 블록으로 반환
            blocks = [LayoutBlock(
                block_id=0,
                block_type=LayoutBlockType.UNKNOWN,
                bbox=(0, 0, self.page_width, self.page_height),
                elements=self._elements if self._elements else [],
                column_index=0,
                reading_order=0,
                confidence=0.1
            )]
        
        result = LayoutAnalysisResult(
            page_num=self.page_num,
            page_size=(self.page_width, self.page_height),
            columns=columns,
            column_count=len(columns),
            blocks=blocks,
            header_region=header_region,
            footer_region=footer_region,
            total_text_elements=sum(1 for e in self._elements if e.element_type == 'text'),
            total_image_elements=sum(1 for e in self._elements if e.element_type == 'image'),
            confidence=self._calculate_confidence(blocks, columns)
        )
        
        logger.info(f"[LayoutBlockDetector] Page {self.page_num + 1}: "
                   f"detected {len(blocks)} blocks in {len(columns)} columns")
        
        return result
    
    def _create_column_based_blocks(self, columns: List[ColumnInfo]) -> List[LayoutBlock]:
        """
        폴백: 컬럼 기반 단순 블록 생성
        
        클러스터링 실패 시 각 컬럼을 하나의 블록으로 처리합니다.
        """
        blocks = []
        block_id = 0
        
        for col in columns:
            # 이 컬럼에 속하는 요소들
            col_elements = [
                e for e in self._elements 
                if self._element_in_column(e, col)
            ]
            
            if col_elements:
                bbox = self._merge_bboxes([e.bbox for e in col_elements])
                blocks.append(LayoutBlock(
                    block_id=block_id,
                    block_type=LayoutBlockType.COLUMN_BLOCK,
                    bbox=bbox,
                    elements=col_elements,
                    column_index=col.index,
                    reading_order=block_id,
                    confidence=0.5
                ))
                block_id += 1
        
        # 요소가 없는 경우 전체 페이지를 하나의 블록으로
        if not blocks:
            blocks.append(LayoutBlock(
                block_id=0,
                block_type=LayoutBlockType.UNKNOWN,
                bbox=(0, 0, self.page_width, self.page_height),
                elements=[],
                column_index=0,
                reading_order=0,
                confidence=0.1
            ))
        
        return blocks
    
    # ========================================================================
    # Phase 1: 기초 분석
    # ========================================================================
    
    def _extract_elements(self):
        """페이지에서 모든 콘텐츠 요소 추출"""
        self._elements = []
        
        # 1. 텍스트 블록 추출
        text_dict = self._get_text_dict()
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 텍스트 블록만
                continue
            
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            
            # 폰트 정보 수집
            max_font_size = 0.0
            is_bold = False
            total_text = ""
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size = span.get("size", 0.0)
                    if font_size > max_font_size:
                        max_font_size = font_size
                    
                    flags = span.get("flags", 0)
                    if flags & 2**4:  # Bold flag
                        is_bold = True
                    
                    total_text += span.get("text", "")
            
            if total_text.strip():
                self._elements.append(ContentElement(
                    element_type='text',
                    bbox=bbox,
                    content=total_text.strip(),
                    font_size=max_font_size,
                    is_bold=is_bold,
                    text_length=len(total_text.strip())
                ))
        
        # 2. 이미지 추출
        images = self._get_images()
        for img_info in images:
            xref = img_info[0]
            try:
                # 이미지 위치 찾기
                img_bbox = self._find_image_position(xref)
                if img_bbox:
                    area = (img_bbox[2] - img_bbox[0]) * (img_bbox[3] - img_bbox[1])
                    self._elements.append(ContentElement(
                        element_type='image',
                        bbox=img_bbox,
                        image_area=area
                    ))
            except Exception:
                pass
    
    def _extract_separators_and_boxes(self):
        """구분선과 박스 추출"""
        self._separators = []
        self._boxes = []
        
        drawings = self._get_drawings()
        
        for drawing in drawings:
            try:
                rect = drawing.get("rect")
                if not rect:
                    continue
                
                # rect 속성 안전하게 접근
                try:
                    w = rect.width
                    h = rect.height
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                except (AttributeError, TypeError):
                    # rect가 튜플일 수 있음
                    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
                        x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
                        w = x1 - x0
                        h = y1 - y0
                    else:
                        continue
                
                # 구분선 (가로)
                if (h <= self.config.SEPARATOR_MAX_THICKNESS and 
                    w >= self.page_width * self.config.SEPARATOR_MIN_LENGTH_RATIO):
                    self._separators.append((x0, y0, x1, y1))
                
                # 구분선 (세로)
                elif (w <= self.config.SEPARATOR_MAX_THICKNESS and 
                      h >= self.page_height * self.config.SEPARATOR_MIN_LENGTH_RATIO * 0.5):
                    self._separators.append((x0, y0, x1, y1))
                
                # 박스 (광고/인포박스 후보)
                elif w > 50 and h > 50:
                    area = w * h
                    if area >= self.config.AD_MIN_BOX_AREA:
                        # 테두리가 있는 박스인지 확인
                        # NOTE: stroke_opacity가 None일 수 있으므로 안전하게 처리
                        stroke_opacity = drawing.get("stroke_opacity")
                        has_stroke = drawing.get("color") or (stroke_opacity is not None and stroke_opacity > 0)
                        if has_stroke:
                            self._boxes.append((x0, y0, x1, y1))
            except Exception as e:
                # 개별 드로잉 처리 실패 시 로깅하고 계속
                logger.debug(f"[LayoutBlockDetector] Error processing drawing: {e}")
                continue
    
    # ========================================================================
    # Phase 2: 컬럼 감지
    # ========================================================================
    
    def _detect_columns(self) -> List[ColumnInfo]:
        """컬럼 구조 감지"""
        if not self._elements:
            return [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        
        # 텍스트 요소의 X 시작점 수집
        x_starts = []
        for elem in self._elements:
            if elem.element_type == 'text' and elem.text_length > 20:  # 충분히 긴 텍스트만
                x_starts.append(elem.bbox[0])
        
        if not x_starts:
            return [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        
        # X 좌표 클러스터링
        x_starts.sort()
        clusters = self._cluster_x_positions(x_starts)
        
        if len(clusters) <= 1:
            return [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        
        # 클러스터 간 간격 분석
        cluster_centers = [sum(c) / len(c) for c in clusters]
        
        # 충분한 간격이 있는 클러스터만 컬럼으로 인정
        columns = []
        valid_boundaries = [0]
        
        for i in range(len(cluster_centers) - 1):
            gap = cluster_centers[i + 1] - cluster_centers[i]
            if gap >= self.config.MIN_COLUMN_GAP:
                # 컬럼 경계 = 두 클러스터 중간점
                boundary = (cluster_centers[i] + cluster_centers[i + 1]) / 2
                valid_boundaries.append(boundary)
        
        valid_boundaries.append(self.page_width)
        
        # 컬럼 생성
        for i in range(len(valid_boundaries) - 1):
            columns.append(ColumnInfo(
                index=i,
                x_start=valid_boundaries[i],
                x_end=valid_boundaries[i + 1]
            ))
        
        logger.debug(f"[LayoutBlockDetector] Detected {len(columns)} columns")
        return columns
    
    def _cluster_x_positions(self, x_positions: List[float]) -> List[List[float]]:
        """X 좌표 클러스터링 (밀도 기반)"""
        if not x_positions:
            return []
        
        clusters = []
        current_cluster = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_cluster[-1] <= self.config.COLUMN_CLUSTER_TOLERANCE:
                current_cluster.append(x)
            else:
                if len(current_cluster) >= 3:  # 최소 3개 요소
                    clusters.append(current_cluster)
                current_cluster = [x]
        
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)
        
        return clusters
    
    # ========================================================================
    # Phase 3: 헤더/푸터 감지
    # ========================================================================
    
    def _detect_header_footer(self) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """헤더와 푸터 영역 감지"""
        header_region = None
        footer_region = None
        
        header_boundary = self.page_height * self.config.HEADER_FOOTER_MARGIN
        footer_boundary = self.page_height * (1 - self.config.HEADER_FOOTER_MARGIN)
        
        # 상단 영역 분석
        header_elements = [
            e for e in self._elements 
            if e.bbox[3] <= header_boundary and e.element_type == 'text'
        ]
        
        if header_elements:
            min_y = min(e.bbox[1] for e in header_elements)
            max_y = max(e.bbox[3] for e in header_elements)
            
            if max_y - min_y <= self.config.HEADER_MAX_HEIGHT:
                header_region = (0, min_y, self.page_width, max_y)
        
        # 하단 영역 분석
        footer_elements = [
            e for e in self._elements 
            if e.bbox[1] >= footer_boundary and e.element_type == 'text'
        ]
        
        if footer_elements:
            min_y = min(e.bbox[1] for e in footer_elements)
            max_y = max(e.bbox[3] for e in footer_elements)
            
            if max_y - min_y <= self.config.FOOTER_MAX_HEIGHT:
                footer_region = (0, min_y, self.page_width, max_y)
        
        return header_region, footer_region
    
    # ========================================================================
    # Phase 4: 의미론적 블록 클러스터링
    # ========================================================================
    
    def _cluster_into_blocks(
        self, 
        columns: List[ColumnInfo],
        header_region: Optional[Tuple],
        footer_region: Optional[Tuple]
    ) -> List[LayoutBlock]:
        """요소들을 의미론적 블록으로 클러스터링"""
        blocks = []
        block_id = 0
        
        # 헤더/푸터 영역 제외한 요소들
        main_elements = []
        header_elements = []
        footer_elements = []
        
        for elem in self._elements:
            if header_region and self._is_inside(elem.bbox, header_region):
                header_elements.append(elem)
            elif footer_region and self._is_inside(elem.bbox, footer_region):
                footer_elements.append(elem)
            else:
                main_elements.append(elem)
        
        # 헤더 블록
        if header_elements:
            bbox = self._merge_bboxes([e.bbox for e in header_elements])
            blocks.append(LayoutBlock(
                block_id=block_id,
                block_type=LayoutBlockType.HEADER,
                bbox=bbox,
                elements=header_elements
            ))
            block_id += 1
        
        # 컬럼별로 처리
        for col in columns:
            # 이 컬럼에 속하는 요소들
            col_elements = [
                e for e in main_elements 
                if self._element_in_column(e, col)
            ]
            
            if not col_elements:
                continue
            
            # 구분선 기반 수직 분할
            vertical_groups = self._split_by_separators(col_elements, col)
            
            for group_elements in vertical_groups:
                if not group_elements:
                    continue
                
                # 인접 요소 클러스터링
                clusters = self._cluster_adjacent_elements(group_elements)
                
                for cluster in clusters:
                    if not cluster:
                        continue
                    
                    bbox = self._merge_bboxes([e.bbox for e in cluster])
                    
                    # 너무 작은 블록은 무시
                    if (bbox[2] - bbox[0] < self.config.MIN_BLOCK_WIDTH or
                        bbox[3] - bbox[1] < self.config.MIN_BLOCK_HEIGHT):
                        continue
                    
                    blocks.append(LayoutBlock(
                        block_id=block_id,
                        block_type=LayoutBlockType.UNKNOWN,  # 나중에 분류
                        bbox=bbox,
                        elements=cluster,
                        column_index=col.index
                    ))
                    block_id += 1
        
        # 푸터 블록
        if footer_elements:
            bbox = self._merge_bboxes([e.bbox for e in footer_elements])
            blocks.append(LayoutBlock(
                block_id=block_id,
                block_type=LayoutBlockType.FOOTER,
                bbox=bbox,
                elements=footer_elements
            ))
        
        return blocks
    
    def _element_in_column(self, elem: ContentElement, col: ColumnInfo) -> bool:
        """요소가 컬럼에 속하는지 확인"""
        elem_center_x = (elem.bbox[0] + elem.bbox[2]) / 2
        return col.x_start <= elem_center_x <= col.x_end
    
    def _split_by_separators(
        self, 
        elements: List[ContentElement], 
        col: ColumnInfo
    ) -> List[List[ContentElement]]:
        """구분선 기준으로 수직 분할"""
        if not elements:
            return []
        
        # 이 컬럼 내 수평 구분선 찾기
        col_separators = []
        for sep in self._separators:
            # 수평 구분선이고 이 컬럼과 겹치는지
            is_horizontal = abs(sep[3] - sep[1]) < 5
            if is_horizontal:
                sep_start_x = sep[0]
                sep_end_x = sep[2]
                if (sep_start_x <= col.x_end and sep_end_x >= col.x_start):
                    col_separators.append(sep[1])  # Y 좌표
        
        if not col_separators:
            return [elements]
        
        # 구분선 위치 정렬
        col_separators.sort()
        
        # 요소들을 구분선 기준으로 분할
        groups = []
        boundaries = [0] + col_separators + [self.page_height]
        
        for i in range(len(boundaries) - 1):
            y_start = boundaries[i]
            y_end = boundaries[i + 1]
            
            group = [
                e for e in elements
                if e.bbox[1] >= y_start - 5 and e.bbox[3] <= y_end + 5
            ]
            
            if group:
                groups.append(group)
        
        return groups if groups else [elements]
    
    def _cluster_adjacent_elements(
        self, 
        elements: List[ContentElement]
    ) -> List[List[ContentElement]]:
        """인접 요소 클러스터링"""
        if not elements:
            return []
        
        if len(elements) == 1:
            return [elements]
        
        # 요소를 Y 좌표로 정렬
        sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))
        
        # Union-Find 스타일 클러스터링
        clusters: List[List[ContentElement]] = []
        used = set()
        
        for elem in sorted_elements:
            if id(elem) in used:
                continue
            
            # 새 클러스터 시작
            cluster = [elem]
            used.add(id(elem))
            queue = [elem]
            
            while queue:
                current = queue.pop(0)
                
                for other in sorted_elements:
                    if id(other) in used:
                        continue
                    
                    if self._are_adjacent(current, other):
                        cluster.append(other)
                        used.add(id(other))
                        queue.append(other)
            
            clusters.append(cluster)
        
        return clusters
    
    def _are_adjacent(self, e1: ContentElement, e2: ContentElement) -> bool:
        """두 요소가 인접한지 확인"""
        # 수직 거리
        vertical_gap = max(0, e2.bbox[1] - e1.bbox[3], e1.bbox[1] - e2.bbox[3])
        
        # 수평 겹침
        x_overlap_start = max(e1.bbox[0], e2.bbox[0])
        x_overlap_end = min(e1.bbox[2], e2.bbox[2])
        has_x_overlap = x_overlap_start < x_overlap_end
        
        # 수직 인접 (같은 X 범위, 가까운 Y)
        if has_x_overlap and vertical_gap <= self.config.VERTICAL_MERGE_THRESHOLD:
            return True
        
        # 수평 인접 (같은 Y 범위)
        horizontal_gap = max(0, e2.bbox[0] - e1.bbox[2], e1.bbox[0] - e2.bbox[2])
        
        y_overlap_start = max(e1.bbox[1], e2.bbox[1])
        y_overlap_end = min(e1.bbox[3], e2.bbox[3])
        has_y_overlap = y_overlap_start < y_overlap_end
        
        if has_y_overlap and horizontal_gap <= self.config.HORIZONTAL_MERGE_THRESHOLD:
            return True
        
        return False
    
    # ========================================================================
    # Phase 5: 블록 분류
    # ========================================================================
    
    def _classify_blocks(self, blocks: List[LayoutBlock]):
        """블록 유형 분류"""
        for block in blocks:
            if block.block_type in (LayoutBlockType.HEADER, LayoutBlockType.FOOTER):
                continue
            
            block.block_type = self._determine_block_type(block)
    
    def _determine_block_type(self, block: LayoutBlock) -> LayoutBlockType:
        """블록 유형 결정"""
        text_elements = [e for e in block.elements if e.element_type == 'text']
        image_elements = [e for e in block.elements if e.element_type == 'image']
        
        has_text = len(text_elements) > 0
        has_image = len(image_elements) > 0
        
        # 이미지 + 텍스트 = IMAGE_WITH_CAPTION
        if has_image and has_text:
            # 텍스트가 이미지 아래/위에 있는지 확인
            for img_elem in image_elements:
                for txt_elem in text_elements:
                    if self._is_caption_of_image(txt_elem, img_elem):
                        return LayoutBlockType.IMAGE_WITH_CAPTION
            return LayoutBlockType.IMAGE_WITH_CAPTION  # 기본 가정
        
        # 이미지만 = STANDALONE_IMAGE
        if has_image and not has_text:
            return LayoutBlockType.STANDALONE_IMAGE
        
        # 텍스트만
        if has_text:
            # 헤드라인 감지 (큰 폰트 + 짧은 텍스트)
            avg_font_size = sum(e.font_size for e in text_elements) / len(text_elements)
            max_font_size = max(e.font_size for e in text_elements)
            
            # 폰트 크기 변화가 크면 ARTICLE (헤드라인 + 본문)
            if max_font_size >= self.config.HEADLINE_MIN_SIZE:
                if max_font_size >= avg_font_size * self.config.HEADLINE_FONT_RATIO:
                    return LayoutBlockType.ARTICLE
            
            # 박스 안에 있으면 SIDEBAR or ADVERTISEMENT
            if self._is_inside_box(block.bbox):
                # 텍스트가 짧으면 광고
                total_text_len = sum(e.text_length for e in text_elements)
                if total_text_len < 200:
                    return LayoutBlockType.ADVERTISEMENT
                return LayoutBlockType.SIDEBAR
            
            return LayoutBlockType.ARTICLE
        
        return LayoutBlockType.UNKNOWN
    
    def _is_caption_of_image(self, text_elem: ContentElement, img_elem: ContentElement) -> bool:
        """텍스트가 이미지의 캡션인지 확인"""
        # 이미지 바로 아래
        if (text_elem.bbox[1] > img_elem.bbox[3] - 5 and
            text_elem.bbox[1] < img_elem.bbox[3] + self.config.CAPTION_MAX_DISTANCE):
            # X 범위가 비슷
            if (text_elem.bbox[0] >= img_elem.bbox[0] - 20 and
                text_elem.bbox[2] <= img_elem.bbox[2] + 20):
                # 높이가 캡션 범위
                if text_elem.bbox[3] - text_elem.bbox[1] <= self.config.CAPTION_MAX_HEIGHT:
                    return True
        
        # 이미지 바로 위도 가능
        if (text_elem.bbox[3] < img_elem.bbox[1] + 5 and
            text_elem.bbox[3] > img_elem.bbox[1] - self.config.CAPTION_MAX_DISTANCE):
            if (text_elem.bbox[0] >= img_elem.bbox[0] - 20 and
                text_elem.bbox[2] <= img_elem.bbox[2] + 20):
                if text_elem.bbox[3] - text_elem.bbox[1] <= self.config.CAPTION_MAX_HEIGHT:
                    return True
        
        return False
    
    def _is_inside_box(self, bbox: Tuple) -> bool:
        """블록이 박스 안에 있는지 확인"""
        for box in self._boxes:
            if self._is_inside(bbox, box, margin=10):
                return True
        return False
    
    # ========================================================================
    # Phase 6: 블록 최적화 및 정렬
    # ========================================================================
    
    def _optimize_and_sort_blocks(
        self, 
        blocks: List[LayoutBlock],
        columns: List[ColumnInfo]
    ) -> List[LayoutBlock]:
        """블록 최적화 및 읽기 순서 정렬"""
        if not blocks:
            return []
        
        # 1. 작은 블록 병합
        blocks = self._merge_small_blocks(blocks)
        
        # 2. 겹침 해결
        blocks = self._resolve_overlaps(blocks)
        
        # 3. 읽기 순서 결정
        #    - 헤더 우선
        #    - 컬럼 순서 (좌 → 우)
        #    - 컬럼 내 상 → 하
        #    - 푸터 마지막
        
        header_blocks = [b for b in blocks if b.block_type == LayoutBlockType.HEADER]
        footer_blocks = [b for b in blocks if b.block_type == LayoutBlockType.FOOTER]
        main_blocks = [b for b in blocks if b.block_type not in (LayoutBlockType.HEADER, LayoutBlockType.FOOTER)]
        
        # 컬럼별로 정렬
        column_groups = defaultdict(list)
        for block in main_blocks:
            column_groups[block.column_index].append(block)
        
        # 각 컬럼 내에서 Y 좌표로 정렬
        for col_idx in column_groups:
            column_groups[col_idx].sort(key=lambda b: b.bbox[1])
        
        # 최종 순서: 헤더 → (컬럼별로) → 푸터
        sorted_blocks = []
        order = 0
        
        for block in header_blocks:
            block.reading_order = order
            sorted_blocks.append(block)
            order += 1
        
        for col_idx in sorted(column_groups.keys()):
            for block in column_groups[col_idx]:
                block.reading_order = order
                sorted_blocks.append(block)
                order += 1
        
        for block in footer_blocks:
            block.reading_order = order
            sorted_blocks.append(block)
            order += 1
        
        return sorted_blocks
    
    def _merge_small_blocks(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """너무 작은 인접 블록 병합"""
        if len(blocks) <= 1:
            return blocks
        
        # 블록 수가 목표 범위 내면 병합 스킵
        if len(blocks) <= self.config.TARGET_MAX_BLOCKS:
            return blocks
        
        result = []
        skip_ids = set()
        
        # 블록 수가 너무 많으면 적극적 병합
        aggressive_merge = len(blocks) > self.config.AGGRESSIVE_MERGE_THRESHOLD
        
        for block in blocks:
            if block.block_id in skip_ids:
                continue
            
            # 작은 블록인지 확인 (적극적 병합 시 기준 높임)
            min_area = self.config.MIN_BLOCK_AREA
            if aggressive_merge:
                min_area = self.config.MIN_BLOCK_AREA * 2  # 2배 기준
            
            if block.area >= min_area:
                result.append(block)
                continue
            
            # 인접한 블록 찾기
            merged = False
            for other in blocks:
                if other.block_id == block.block_id or other.block_id in skip_ids:
                    continue
                
                if self._should_merge_blocks(block, other, aggressive=aggressive_merge):
                    # 병합
                    merged_bbox = self._merge_bboxes([block.bbox, other.bbox])
                    other.bbox = merged_bbox
                    other.elements.extend(block.elements)
                    skip_ids.add(block.block_id)
                    merged = True
                    break
            
            if not merged:
                result.append(block)
        
        # 목표보다 많으면 추가 병합 시도
        if len(result) > self.config.TARGET_MAX_BLOCKS:
            result = self._force_merge_to_target(result)
        
        return result
    
    def _should_merge_blocks(self, b1: LayoutBlock, b2: LayoutBlock, aggressive: bool = False) -> bool:
        """두 블록을 병합해야 하는지 확인"""
        # 같은 컬럼 (적극적 병합 시 인접 컬럼도 허용)
        if not aggressive and b1.column_index != b2.column_index:
            return False
        if aggressive and abs(b1.column_index - b2.column_index) > 1:
            return False
        
        # 가까운 거리
        vertical_gap = max(0, b2.bbox[1] - b1.bbox[3], b1.bbox[1] - b2.bbox[3])
        threshold = self.config.VERTICAL_MERGE_THRESHOLD * (3 if aggressive else 2)
        if vertical_gap > threshold:
            return False
        
        return True
    
    def _force_merge_to_target(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        블록 수가 목표를 초과할 때 강제 병합.
        같은 컬럼 내에서 인접한 블록들을 병합합니다.
        """
        if len(blocks) <= self.config.TARGET_MAX_BLOCKS:
            return blocks
        
        # 컬럼별로 그룹화
        column_groups: Dict[int, List[LayoutBlock]] = defaultdict(list)
        for block in blocks:
            column_groups[block.column_index].append(block)
        
        result = []
        
        for col_idx in sorted(column_groups.keys()):
            col_blocks = sorted(column_groups[col_idx], key=lambda b: b.bbox[1])
            
            # 컬럼 내 블록 수가 2개 이상이면 병합 가능
            if len(col_blocks) >= 2:
                # 인접한 블록들을 병합
                merged_blocks = self._merge_adjacent_in_column(col_blocks)
                result.extend(merged_blocks)
            else:
                result.extend(col_blocks)
        
        logger.debug(f"[LayoutBlockDetector] Force merged: {len(blocks)} → {len(result)} blocks")
        return result
    
    def _merge_adjacent_in_column(self, col_blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        컬럼 내에서 인접한 블록들을 병합.
        최대 2~3개 블록으로 축소합니다.
        """
        if len(col_blocks) <= 2:
            return col_blocks
        
        # 블록들을 2~3개 그룹으로 나눔
        target_groups = max(2, min(3, len(col_blocks) // 2))
        blocks_per_group = max(1, len(col_blocks) // target_groups)
        
        result = []
        current_group = []
        
        for i, block in enumerate(col_blocks):
            current_group.append(block)
            
            # 그룹이 채워지면 병합
            if len(current_group) >= blocks_per_group and len(result) < target_groups - 1:
                merged = self._merge_block_group(current_group)
                result.append(merged)
                current_group = []
        
        # 남은 블록들 병합
        if current_group:
            merged = self._merge_block_group(current_group)
            result.append(merged)
        
        return result
    
    def _merge_block_group(self, blocks: List[LayoutBlock]) -> LayoutBlock:
        """블록 그룹을 하나로 병합"""
        if len(blocks) == 1:
            return blocks[0]
        
        merged_bbox = self._merge_bboxes([b.bbox for b in blocks])
        merged_elements = []
        for b in blocks:
            merged_elements.extend(b.elements)
        
        return LayoutBlock(
            block_id=blocks[0].block_id,
            block_type=blocks[0].block_type,
            bbox=merged_bbox,
            elements=merged_elements,
            column_index=blocks[0].column_index,
            reading_order=blocks[0].reading_order,
            confidence=min(b.confidence for b in blocks)
        )
    
    def _resolve_overlaps(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """블록 겹침 해결"""
        # 현재는 단순히 반환 (추후 개선 가능)
        return blocks
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_text_dict(self) -> Dict:
        """텍스트 딕셔너리 캐시"""
        if self._text_dict is None:
            self._text_dict = self.page.get_text("dict", sort=True)
        return self._text_dict
    
    def _get_drawings(self) -> List:
        """드로잉 캐시"""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings
    
    def _get_images(self) -> List:
        """이미지 캐시"""
        if self._images is None:
            self._images = self.page.get_images()
        return self._images
    
    def _find_image_position(self, xref: int) -> Optional[Tuple[float, float, float, float]]:
        """이미지 위치 찾기"""
        try:
            for img in self.page.get_image_rects(xref):
                return (img.x0, img.y0, img.x1, img.y1)
        except Exception:
            pass
        return None
    
    def _is_inside(
        self, 
        inner: Tuple[float, float, float, float], 
        outer: Tuple[float, float, float, float],
        margin: float = 0
    ) -> bool:
        """inner가 outer 안에 있는지 확인"""
        return (
            inner[0] >= outer[0] - margin and
            inner[1] >= outer[1] - margin and
            inner[2] <= outer[2] + margin and
            inner[3] <= outer[3] + margin
        )
    
    def _merge_bboxes(self, bboxes: List[Tuple]) -> Tuple[float, float, float, float]:
        """여러 bbox 병합"""
        if not bboxes:
            return (0, 0, 0, 0)
        
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        return (x0, y0, x1, y1)
    
    def _calculate_confidence(self, blocks: List[LayoutBlock], columns: List[ColumnInfo]) -> float:
        """분석 신뢰도 계산"""
        if not blocks:
            return 0.5
        
        # 요소 수 대비 블록 수 비율
        total_elements = len(self._elements)
        if total_elements == 0:
            return 0.5
        
        covered_elements = sum(len(b.elements) for b in blocks)
        coverage = covered_elements / total_elements
        
        # UNKNOWN 블록 비율
        unknown_ratio = sum(1 for b in blocks if b.block_type == LayoutBlockType.UNKNOWN) / max(1, len(blocks))
        
        confidence = coverage * (1 - unknown_ratio * 0.3)
        
        return min(1.0, max(0.0, confidence))


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'LayoutBlockType',
    'ContentElement',
    'LayoutBlock',
    'ColumnInfo',
    'LayoutAnalysisResult',
    'LayoutDetectorConfig',
    'LayoutBlockDetector',
]
