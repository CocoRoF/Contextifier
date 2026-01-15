"""
Complexity Analyzer for PDF Handler V4

페이지 및 영역의 복잡도를 분석하여 처리 전략을 결정합니다.

=============================================================================
V4 핵심 개념:
=============================================================================
1. 복잡도 점수 기반 처리 전략 결정
2. 복잡한 영역은 블록 이미지화 + OCR
3. 단순한 영역은 기존 텍스트 추출

복잡도 판단 기준:
- 드로잉 밀도 (곡선, 선, 채우기 수)
- 이미지 밀도
- 텍스트 품질 (깨진 텍스트 비율)
- 레이아웃 복잡도 (다단 컬럼)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class ComplexityLevel(Enum):
    """복잡도 수준"""
    SIMPLE = auto()        # 단순 텍스트 - 기존 추출
    MODERATE = auto()      # 중간 복잡도 - 기존 추출 + 품질 검증
    COMPLEX = auto()       # 복잡 - 블록 이미지화 권장
    EXTREME = auto()       # 극도로 복잡 - 전체 페이지 이미지화 권장


class ProcessingStrategy(Enum):
    """처리 전략"""
    TEXT_EXTRACTION = auto()       # 기존 텍스트 추출
    HYBRID = auto()                # 텍스트 + 부분 OCR
    BLOCK_IMAGE_OCR = auto()       # 블록 이미지화 + OCR
    FULL_PAGE_OCR = auto()         # 전체 페이지 OCR


@dataclass
class RegionComplexity:
    """영역별 복잡도 정보"""
    bbox: Tuple[float, float, float, float]
    complexity_level: ComplexityLevel
    complexity_score: float  # 0.0 ~ 1.0
    
    # 세부 점수
    drawing_density: float = 0.0
    image_density: float = 0.0
    text_quality: float = 1.0  # 1.0 = 완벽, 0.0 = 완전 깨짐
    layout_complexity: float = 0.0
    
    # 권장 전략
    recommended_strategy: ProcessingStrategy = ProcessingStrategy.TEXT_EXTRACTION
    
    # 상세 정보
    reasons: List[str] = field(default_factory=list)


@dataclass
class PageComplexity:
    """페이지 전체 복잡도 정보"""
    page_num: int
    page_size: Tuple[float, float]
    
    # 전체 복잡도
    overall_complexity: ComplexityLevel
    overall_score: float
    
    # 영역별 복잡도
    regions: List[RegionComplexity] = field(default_factory=list)
    
    # 복잡한 영역들
    complex_regions: List[Tuple[float, float, float, float]] = field(default_factory=list)
    
    # 통계
    total_drawings: int = 0
    total_images: int = 0
    total_text_blocks: int = 0
    column_count: int = 1
    
    # 권장 전략
    recommended_strategy: ProcessingStrategy = ProcessingStrategy.TEXT_EXTRACTION


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ComplexityConfig:
    """복잡도 분석 설정"""
    # 드로잉 밀도 임계값 (영역 1000pt² 당)
    DRAWING_DENSITY_MODERATE = 0.5
    DRAWING_DENSITY_COMPLEX = 2.0
    DRAWING_DENSITY_EXTREME = 5.0
    
    # 이미지 밀도 임계값
    IMAGE_DENSITY_MODERATE = 0.1
    IMAGE_DENSITY_COMPLEX = 0.3
    IMAGE_DENSITY_EXTREME = 0.5
    
    # 텍스트 품질 임계값
    TEXT_QUALITY_POOR = 0.7
    TEXT_QUALITY_BAD = 0.5
    
    # 레이아웃 복잡도 (다단 컬럼)
    # ★ 수정: 임계값 상향 - 다단이라고 무조건 EXTREME으로 판단하지 않음
    COLUMN_COUNT_MODERATE = 3   # 3단 이상 = MODERATE
    COLUMN_COUNT_COMPLEX = 5    # 5단 이상 = COMPLEX (신문급)
    COLUMN_COUNT_EXTREME = 7    # 7단 이상 = EXTREME (매우 복잡한 신문)
    
    # 전체 복잡도 임계값
    # ★ 수정: EXTREME 임계값 상향 - 너무 쉽게 FULL_PAGE_OCR로 가지 않도록
    COMPLEXITY_MODERATE = 0.35
    COMPLEXITY_COMPLEX = 0.65
    COMPLEXITY_EXTREME = 0.90  # 0.8 → 0.90으로 상향
    
    # 영역 분할 설정
    REGION_GRID_SIZE = 200  # pt - 분석 그리드 크기
    MIN_REGION_SIZE = 100   # pt - 최소 영역 크기
    
    # ★ 신규: 테이블 처리 가능성 분석 활성화
    ANALYZE_TABLE_QUALITY = True  # 테이블 품질 분석 활성화
    TABLE_QUALITY_THRESHOLD = 0.65  # 이 이상이면 테이블 추출 시도


# ============================================================================
# Complexity Analyzer
# ============================================================================

class ComplexityAnalyzer:
    """
    페이지 복잡도 분석기
    
    페이지의 복잡도를 분석하여 최적의 처리 전략을 결정합니다.
    """
    
    def __init__(self, page, page_num: int, config: Optional[ComplexityConfig] = None):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed)
            config: 분석 설정
        """
        self.page = page
        self.page_num = page_num
        self.config = config or ComplexityConfig()
        
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.page_area = self.page_width * self.page_height
        
        # 캐시
        self._drawings = None
        self._text_dict = None
        self._images = None
    
    def analyze(self) -> PageComplexity:
        """
        페이지 복잡도를 분석합니다.
        
        Returns:
            PageComplexity 객체
        """
        # 기본 데이터 수집
        drawings = self._get_drawings()
        text_dict = self._get_text_dict()
        images = self._get_images()
        
        text_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 0]
        
        # 1. 전체 통계
        total_drawings = len(drawings)
        total_images = len(images)
        total_text_blocks = len(text_blocks)
        
        # 2. 컬럼 수 분석
        column_count = self._analyze_columns(text_blocks)
        
        # 3. 드로잉 복잡도
        drawing_complexity = self._analyze_drawing_complexity(drawings)
        
        # 4. 이미지 복잡도
        image_complexity = self._analyze_image_complexity(images)
        
        # 5. 텍스트 품질
        text_quality = self._analyze_text_quality(text_blocks)
        
        # 6. 레이아웃 복잡도
        layout_complexity = self._analyze_layout_complexity(column_count, text_blocks)
        
        # 7. 전체 복잡도 점수 계산
        overall_score = self._calculate_overall_score(
            drawing_complexity, image_complexity, text_quality, layout_complexity
        )
        
        # 8. 복잡도 수준 결정
        overall_complexity = self._determine_complexity_level(overall_score)
        
        # 9. 영역별 분석
        regions = self._analyze_regions(drawings, text_blocks, images)
        
        # 10. 복잡한 영역 식별
        complex_regions = [
            r.bbox for r in regions 
            if r.complexity_level in (ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME)
        ]
        
        # 11. 처리 전략 결정
        recommended_strategy = self._determine_strategy(
            overall_complexity, overall_score, text_quality, complex_regions
        )
        
        result = PageComplexity(
            page_num=self.page_num,
            page_size=(self.page_width, self.page_height),
            overall_complexity=overall_complexity,
            overall_score=overall_score,
            regions=regions,
            complex_regions=complex_regions,
            total_drawings=total_drawings,
            total_images=total_images,
            total_text_blocks=total_text_blocks,
            column_count=column_count,
            recommended_strategy=recommended_strategy
        )
        
        logger.debug(f"[ComplexityAnalyzer] Page {self.page_num + 1}: "
                    f"complexity={overall_complexity.name}, score={overall_score:.2f}, "
                    f"strategy={recommended_strategy.name}, "
                    f"complex_regions={len(complex_regions)}")
        
        return result
    
    def _get_drawings(self) -> List[Dict]:
        """드로잉 캐시"""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings
    
    def _get_text_dict(self) -> Dict:
        """텍스트 딕셔너리 캐시"""
        if self._text_dict is None:
            self._text_dict = self.page.get_text("dict", sort=True)
        return self._text_dict
    
    def _get_images(self) -> List:
        """이미지 캐시"""
        if self._images is None:
            self._images = self.page.get_images()
        return self._images
    
    def _analyze_columns(self, text_blocks: List[Dict]) -> int:
        """컬럼 수 분석"""
        if not text_blocks:
            return 1
        
        x_positions = []
        for block in text_blocks:
            bbox = block.get("bbox", (0, 0, 0, 0))
            x_positions.append(bbox[0])
        
        if not x_positions:
            return 1
        
        x_positions.sort()
        
        # 클러스터링
        columns = []
        current_column = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_column[-1] < 50:  # 50pt 이내면 같은 컬럼
                current_column.append(x)
            else:
                columns.append(current_column)
                current_column = [x]
        columns.append(current_column)
        
        return len(columns)
    
    def _analyze_drawing_complexity(self, drawings: List[Dict]) -> float:
        """드로잉 복잡도 분석 (0.0 ~ 1.0)"""
        if not drawings:
            return 0.0
        
        # 아이템 수 계산
        total_items = 0
        curve_count = 0
        fill_count = 0
        
        for d in drawings:
            items = d.get("items", [])
            total_items += len(items)
            
            for item in items:
                if item[0] == 'c':  # 곡선
                    curve_count += 1
            
            if d.get("fill"):
                fill_count += 1
        
        # 밀도 계산 (1000pt² 당)
        density = total_items / (self.page_area / 1000) if self.page_area > 0 else 0
        
        # 곡선 비율 (차트/그래프 지표)
        curve_ratio = curve_count / max(1, total_items)
        
        # 채우기 비율 (색상 복잡도)
        fill_ratio = fill_count / max(1, len(drawings))
        
        # 복잡도 점수 계산
        score = 0.0
        
        if density >= self.config.DRAWING_DENSITY_EXTREME:
            score = 1.0
        elif density >= self.config.DRAWING_DENSITY_COMPLEX:
            score = 0.7
        elif density >= self.config.DRAWING_DENSITY_MODERATE:
            score = 0.4
        else:
            score = density / self.config.DRAWING_DENSITY_MODERATE * 0.4
        
        # 곡선과 채우기가 많으면 추가 점수
        score += curve_ratio * 0.2
        score += fill_ratio * 0.1
        
        return min(1.0, score)
    
    def _analyze_image_complexity(self, images: List) -> float:
        """이미지 복잡도 분석 (0.0 ~ 1.0)"""
        if not images:
            return 0.0
        
        # 이미지 밀도 (페이지 크기 대비)
        density = len(images) / (self.page_area / 10000)  # 100x100pt 당
        
        if density >= self.config.IMAGE_DENSITY_EXTREME:
            return 1.0
        elif density >= self.config.IMAGE_DENSITY_COMPLEX:
            return 0.7
        elif density >= self.config.IMAGE_DENSITY_MODERATE:
            return 0.4
        else:
            return density / self.config.IMAGE_DENSITY_MODERATE * 0.4
    
    def _analyze_text_quality(self, text_blocks: List[Dict]) -> float:
        """텍스트 품질 분석 (0.0 = 나쁨, 1.0 = 좋음)"""
        if not text_blocks:
            return 1.0
        
        total_chars = 0
        bad_chars = 0
        
        for block in text_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    total_chars += len(text)
                    
                    for char in text:
                        code = ord(char)
                        # PUA (Private Use Area) 문자
                        if 0xE000 <= code <= 0xF8FF:
                            bad_chars += 1
                        # 이상한 기호들
                        elif code in range(0x2400, 0x2500):  # Control Pictures
                            bad_chars += 1
        
        if total_chars == 0:
            return 1.0
        
        return 1.0 - (bad_chars / total_chars)
    
    def _analyze_layout_complexity(self, column_count: int, text_blocks: List[Dict]) -> float:
        """레이아웃 복잡도 분석 (0.0 ~ 1.0)
        
        ★ 수정: 다단 컬럼이라고 무조건 높은 점수를 부여하지 않음
        테이블이 처리 가능한 경우 TEXT_EXTRACTION이 더 효율적일 수 있음
        """
        score = 0.0
        
        # 컬럼 수 기반 - 임계값 완화
        if column_count >= getattr(self.config, 'COLUMN_COUNT_EXTREME', 7):
            # 7단 이상 = 매우 복잡한 신문 레이아웃
            score = 0.95
            logger.info(f"[ComplexityAnalyzer] Page {self.page_num + 1}: "
                       f"Detected very complex layout ({column_count} columns) → HIGH")
        elif column_count >= self.config.COLUMN_COUNT_COMPLEX:
            # 5-6단 = 신문급 레이아웃, 하지만 테이블 처리 가능할 수 있음
            score = 0.75
            logger.info(f"[ComplexityAnalyzer] Page {self.page_num + 1}: "
                       f"Detected multi-column layout ({column_count} columns) → COMPLEX")
        elif column_count >= self.config.COLUMN_COUNT_MODERATE:
            # 3-4단 = 중간 복잡도
            score = 0.5
        elif column_count >= 2:
            # 2단 = 낮은 복잡도
            score = 0.3
        
        # 텍스트 블록 분포 분석 - 같은 Y에 여러 블록이 있으면 다단 레이아웃
        if text_blocks:
            y_positions = [b.get("bbox", (0,0,0,0))[1] for b in text_blocks]
            unique_y = len(set(int(y/10) for y in y_positions))
            
            if unique_y < len(text_blocks) * 0.5 and len(text_blocks) > 5:
                # 같은 Y 라인에 여러 블록 = 다단 레이아웃 추가 증거
                score = max(score, 0.6)
        
        return min(1.0, score)
    
    def _calculate_overall_score(
        self, 
        drawing: float, 
        image: float, 
        text_quality: float, 
        layout: float
    ) -> float:
        """전체 복잡도 점수 계산
        
        ★ 수정: 레이아웃 복잡도만으로 EXTREME 판단하지 않음
        테이블이 처리 가능하면 TEXT_EXTRACTION이 더 효율적
        """
        # 레이아웃이 극도로 복잡하면 (7단 이상) 높은 점수
        if layout >= 0.95:
            return 0.9  # 0.9로 제한 (EXTREME이 되려면 다른 요소도 필요)
        
        # 일반적인 가중치 계산
        # ★ 수정: 레이아웃 가중치 감소 (0.35 → 0.25)
        w_drawing = 0.30
        w_image = 0.20
        w_text = 0.25
        w_layout = 0.25
        
        # 텍스트 품질은 역수 (낮을수록 복잡)
        text_complexity = 1.0 - text_quality
        
        score = (
            drawing * w_drawing +
            image * w_image +
            text_complexity * w_text +
            layout * w_layout
        )
        
        return min(1.0, score)
    
    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """복잡도 수준 결정"""
        if score >= self.config.COMPLEXITY_EXTREME:
            return ComplexityLevel.EXTREME
        elif score >= self.config.COMPLEXITY_COMPLEX:
            return ComplexityLevel.COMPLEX
        elif score >= self.config.COMPLEXITY_MODERATE:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE
    
    def _analyze_regions(
        self, 
        drawings: List[Dict], 
        text_blocks: List[Dict],
        images: List
    ) -> List[RegionComplexity]:
        """영역별 복잡도 분석"""
        regions = []
        grid_size = self.config.REGION_GRID_SIZE
        
        # 그리드 기반 분석
        for y in range(0, int(self.page_height), grid_size):
            for x in range(0, int(self.page_width), grid_size):
                x0, y0 = x, y
                x1 = min(x + grid_size, self.page_width)
                y1 = min(y + grid_size, self.page_height)
                
                bbox = (x0, y0, x1, y1)
                
                # 영역 내 드로잉 수
                region_drawings = [
                    d for d in drawings 
                    if d.get("rect") and self._bbox_overlaps(bbox, tuple(d["rect"]))
                ]
                
                # 영역 내 텍스트 블록 수
                region_texts = [
                    b for b in text_blocks
                    if self._bbox_overlaps(bbox, b.get("bbox", (0,0,0,0)))
                ]
                
                # 영역 복잡도 계산
                area = (x1 - x0) * (y1 - y0)
                drawing_density = len(region_drawings) / (area / 1000) if area > 0 else 0
                
                # 텍스트 품질
                text_quality = self._analyze_text_quality(region_texts)
                
                # 복잡도 점수
                region_score = min(1.0, drawing_density / 3.0 + (1.0 - text_quality) * 0.5)
                
                # 수준 결정
                if region_score >= 0.7:
                    level = ComplexityLevel.COMPLEX
                elif region_score >= 0.4:
                    level = ComplexityLevel.MODERATE
                else:
                    level = ComplexityLevel.SIMPLE
                
                # 전략 결정
                if level == ComplexityLevel.COMPLEX:
                    strategy = ProcessingStrategy.BLOCK_IMAGE_OCR
                elif text_quality < 0.7:
                    strategy = ProcessingStrategy.HYBRID
                else:
                    strategy = ProcessingStrategy.TEXT_EXTRACTION
                
                regions.append(RegionComplexity(
                    bbox=bbox,
                    complexity_level=level,
                    complexity_score=region_score,
                    drawing_density=drawing_density,
                    text_quality=text_quality,
                    recommended_strategy=strategy
                ))
        
        return regions
    
    def _determine_strategy(
        self,
        complexity: ComplexityLevel,
        score: float,
        text_quality: float,
        complex_regions: List[Tuple]
    ) -> ProcessingStrategy:
        """처리 전략 결정
        
        ★ 개선: 다단 컬럼이라도 테이블 처리가 가능하면 TEXT_EXTRACTION 권장
        테이블 품질이 좋으면 이미지화보다 텍스트 추출이 더 효율적
        """
        # 1. 텍스트 품질이 매우 낮으면 전체 페이지 이미지화
        if text_quality < 0.4:
            logger.info(f"[ComplexityAnalyzer] Page {self.page_num + 1}: "
                       f"Very low text quality ({text_quality:.2f}) → FULL_PAGE_OCR")
            return ProcessingStrategy.FULL_PAGE_OCR
        
        # 2. 극도로 복잡하고 (score >= 0.90) 텍스트 품질도 낮으면 전체 페이지 이미지화
        if complexity == ComplexityLevel.EXTREME and text_quality < 0.6:
            return ProcessingStrategy.FULL_PAGE_OCR
        
        # 3. 복잡한 영역이 50% 이상이고 텍스트 품질이 낮으면 전체 페이지 이미지화
        if len(complex_regions) > 0:
            complex_area = sum(
                (r[2] - r[0]) * (r[3] - r[1]) for r in complex_regions
            )
            if complex_area / self.page_area > 0.5 and text_quality < 0.7:
                return ProcessingStrategy.FULL_PAGE_OCR
        
        # 4. ★ 핵심 변경: COMPLEX 수준이라도 HYBRID로 처리 시도
        #    (블록별로 테이블/텍스트 처리 가능 여부 판단)
        if complexity == ComplexityLevel.COMPLEX:
            return ProcessingStrategy.HYBRID  # FULL_PAGE_OCR 대신 HYBRID
        
        # 5. 중간 복잡도면 하이브리드
        if complexity == ComplexityLevel.MODERATE:
            return ProcessingStrategy.HYBRID
        
        # 6. 단순하면 텍스트 추출
        return ProcessingStrategy.TEXT_EXTRACTION
    
    def _bbox_overlaps(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """두 bbox가 겹치는지 확인"""
        return not (
            bbox1[2] <= bbox2[0] or  # bbox1이 bbox2 왼쪽
            bbox1[0] >= bbox2[2] or  # bbox1이 bbox2 오른쪽
            bbox1[3] <= bbox2[1] or  # bbox1이 bbox2 위
            bbox1[1] >= bbox2[3]     # bbox1이 bbox2 아래
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ComplexityLevel',
    'ProcessingStrategy',
    'RegionComplexity',
    'PageComplexity',
    'ComplexityConfig',
    'ComplexityAnalyzer',
]
