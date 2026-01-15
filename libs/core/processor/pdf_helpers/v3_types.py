"""
PDF Handler V3 Types and Configuration

V3 엔진에서 사용하는 모든 데이터 클래스와 설정값을 정의합니다.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any


# ============================================================================
# Enums
# ============================================================================

class LineThickness(Enum):
    """선 두께 분류"""
    THIN = auto()      # 테이블 내부선 (0.3-0.5pt)
    NORMAL = auto()    # 일반 테두리 (0.5-1.5pt)
    THICK = auto()     # 강조/헤더 구분선 (1.5pt+)


class TableDetectionStrategy(Enum):
    """테이블 감지 전략"""
    PYMUPDF_NATIVE = auto()        # PyMuPDF 내장 테이블 감지
    PDFPLUMBER_LINES = auto()      # pdfplumber 선 기반 감지
    HYBRID_ANALYSIS = auto()       # 선 분석 기반 하이브리드
    BORDERLESS_HEURISTIC = auto()  # 선 없는 테이블 휴리스틱


class ElementType(Enum):
    """페이지 요소 유형"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    ANNOTATION = "annotation"


# ============================================================================
# V3 Configuration Constants
# ============================================================================

class V3Config:
    """V3 엔진 설정 상수"""
    
    # 선 두께 임계값 (pt)
    THIN_LINE_MAX = 0.5
    NORMAL_LINE_MAX = 1.5
    
    # 테이블 감지 설정
    MIN_TABLE_ROWS = 2
    MIN_TABLE_COLS = 2
    TABLE_MERGE_TOLERANCE = 5.0  # 테이블 병합 허용 오차 (pt)
    
    # 이중선 병합 설정
    DOUBLE_LINE_TOLERANCE = 3.0  # 이중선 판단 거리 (pt)
    
    # 셀 분석 설정
    CELL_PADDING = 2.0
    MIN_CELL_WIDTH = 10.0
    MIN_CELL_HEIGHT = 8.0
    
    # 텍스트 추출 설정
    TEXT_BLOCK_TOLERANCE = 3.0
    
    # 신뢰도 임계값
    CONFIDENCE_THRESHOLD = 0.5
    
    # 페이지 테두리 감지 설정
    BORDER_MARGIN = 30.0        # 페이지 가장자리로부터의 최대 거리
    BORDER_LENGTH_RATIO = 0.8   # 페이지 크기 대비 최소 테두리 길이 비율
    
    # V3.1: 그래픽 영역 감지 설정
    GRAPHIC_CURVE_RATIO_THRESHOLD = 0.3   # 곡선 비율 임계값
    GRAPHIC_MIN_CURVE_COUNT = 10          # 최소 곡선 개수
    GRAPHIC_FILL_RATIO_THRESHOLD = 0.2    # 채우기 비율 임계값
    GRAPHIC_COLOR_VARIETY_THRESHOLD = 3   # 색상 다양성 임계값
    
    # V3.3: 테이블 품질 검증 설정 (완화됨)
    # 일반 테이블도 빈 셀이 많을 수 있음을 고려
    TABLE_MIN_FILLED_CELL_RATIO = 0.15    # 최소 채워진 셀 비율 (0.3 → 0.15)
    TABLE_MAX_EMPTY_ROW_RATIO = 0.7       # 최대 빈 행 비율 (0.5 → 0.7)
    TABLE_MIN_MEANINGFUL_CELLS = 2        # 최소 의미있는 셀 수 (3 → 2)
    TABLE_MIN_VALID_ROWS = 2              # 최소 유효 행 수
    TABLE_MIN_TEXT_DENSITY = 0.005        # 최소 텍스트 밀도 (0.01 → 0.005)
    
    # V3.3: 셀 텍스트 길이 설정 (완화됨)
    TABLE_MAX_CELL_TEXT_LENGTH = 300      # 셀당 최대 텍스트 길이 (200 → 300)
    TABLE_EXTREME_CELL_LENGTH = 800       # 극단적으로 긴 셀 (500 → 800)
    TABLE_MAX_LONG_CELLS_RATIO = 0.4      # 긴 셀 최대 비율 (0.3 → 0.4)


# ============================================================================
# Data Classes - Basic Types
# ============================================================================

@dataclass
class LineInfo:
    """선 정보"""
    x0: float
    y0: float
    x1: float
    y1: float
    thickness: float = 1.0
    thickness_class: LineThickness = LineThickness.NORMAL
    is_horizontal: bool = False
    is_vertical: bool = False
    
    @property
    def length(self) -> float:
        """선 길이"""
        import math
        return math.sqrt((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2)
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """중점"""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


@dataclass 
class GridInfo:
    """그리드 정보"""
    h_lines: List[float] = field(default_factory=list)  # Y 좌표
    v_lines: List[float] = field(default_factory=list)  # X 좌표
    cells: List['CellInfo'] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    is_complete: bool = False  # 완전한 테두리 여부
    reconstructed: bool = False  # 복구된 테두리 여부
    
    @property
    def row_count(self) -> int:
        """행 수 (가로선 사이의 영역 수)"""
        return max(0, len(self.h_lines) - 1)
    
    @property
    def col_count(self) -> int:
        """열 수 (세로선 사이의 영역 수)"""
        return max(0, len(self.v_lines) - 1)


@dataclass
class CellInfo:
    """셀 정보"""
    row: int
    col: int
    bbox: Tuple[float, float, float, float]
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    alignment: str = "left"


@dataclass
class AnnotationInfo:
    """주석 정보"""
    type: str
    bbox: Tuple[float, float, float, float]
    content: str = ""
    color: Optional[Tuple[float, float, float]] = None


# ============================================================================
# Data Classes - Vector Text OCR
# ============================================================================

@dataclass
class VectorTextRegion:
    """
    벡터 텍스트(Outlined/Path Text) 영역 정보
    """
    bbox: Tuple[float, float, float, float]
    drawing_count: int              # 포함된 드로잉 수
    curve_count: int                # 곡선 수 (c 아이템)
    fill_count: int                 # 채워진 경로 수
    ocr_text: str = ""              # OCR 결과
    confidence: float = 0.0         # 신뢰도
    is_vector_text: bool = False    # 벡터 텍스트 여부


# ============================================================================
# Data Classes - Graphic Region
# ============================================================================

@dataclass
class GraphicRegionInfo:
    """
    그래픽 영역 정보 (차트, 다이어그램, 아이콘 등)
    """
    bbox: Tuple[float, float, float, float]
    curve_count: int = 0            # 곡선 수
    line_count: int = 0             # 직선 수
    rect_count: int = 0             # 사각형 수
    fill_count: int = 0             # 채워진 도형 수
    color_count: int = 0            # 사용된 색상 수
    is_graphic: bool = False        # 그래픽 영역 여부
    confidence: float = 0.0         # 신뢰도
    reason: str = ""                # 판단 근거


# ============================================================================
# Data Classes - Table Detection
# ============================================================================

@dataclass
class TableCandidate:
    """테이블 후보"""
    strategy: TableDetectionStrategy
    confidence: float
    bbox: Tuple[float, float, float, float]
    grid: Optional[GridInfo] = None
    cells: List['CellInfo'] = field(default_factory=list)
    data: List[List[Optional[str]]] = field(default_factory=list)
    raw_table: Any = None  # 원본 테이블 객체
    
    @property
    def row_count(self) -> int:
        """행 수"""
        return len(self.data)
    
    @property
    def col_count(self) -> int:
        """열 수"""
        return max(len(row) for row in self.data) if self.data else 0


@dataclass
class PageElement:
    """페이지 요소"""
    element_type: ElementType
    content: str
    bbox: Tuple[float, float, float, float]
    page_num: int
    table_data: Optional[List[List]] = None
    cells_info: Optional[List[Dict]] = None
    annotations: Optional[List[AnnotationInfo]] = None
    detection_strategy: Optional[TableDetectionStrategy] = None
    confidence: float = 1.0


@dataclass
class PageBorderInfo:
    """페이지 테두리 정보"""
    has_border: bool = False
    border_bbox: Optional[Tuple[float, float, float, float]] = None
    border_lines: Dict[str, bool] = field(default_factory=lambda: {
        'top': False, 'bottom': False, 'left': False, 'right': False
    })


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Enums
    'LineThickness',
    'TableDetectionStrategy', 
    'ElementType',
    # Config
    'V3Config',
    # Data Classes
    'LineInfo',
    'GridInfo',
    'CellInfo',
    'AnnotationInfo',
    'VectorTextRegion',
    'GraphicRegionInfo',
    'TableCandidate',
    'PageElement',
    'PageBorderInfo',
]
