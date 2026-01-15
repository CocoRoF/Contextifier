"""
Table Quality Analyzer for PDF Handler V4

테이블의 품질을 분석하여 텍스트 추출 가능 여부를 판단합니다.

=============================================================================
핵심 개념:
=============================================================================
모든 표를 이미지로 처리하는 것은 비효율적입니다.
정상적인 표(선이 완전하고, 그리드가 규칙적인 표)는 텍스트로 추출해야 합니다.

판단 기준:
1. 선의 완전성 - 표가 사방으로 닫혀 있는가?
2. 그리드 규칙성 - 직교하는 수평/수직선으로 구성되어 있는가?
3. 셀 구조 - 셀이 규칙적인 사각형인가?
4. 복잡한 요소 부재 - 곡선, 대각선, 복잡한 그래픽이 없는가?

=============================================================================
테이블 품질 등급:
=============================================================================
- EXCELLENT: 완벽한 표 → 반드시 텍스트 추출
- GOOD: 양호한 표 → 텍스트 추출 시도 권장
- MODERATE: 일부 문제 있는 표 → 텍스트 추출 시도, 실패 시 이미지화
- POOR: 문제가 많은 표 → 이미지화 권장
- UNPROCESSABLE: 처리 불가 → 반드시 이미지화
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
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

class TableQuality(Enum):
    """테이블 품질 등급"""
    EXCELLENT = auto()      # 완벽한 표 - 반드시 텍스트 추출
    GOOD = auto()           # 양호한 표 - 텍스트 추출 권장
    MODERATE = auto()       # 중간 - 시도 후 판단
    POOR = auto()           # 문제 있음 - 이미지화 권장
    UNPROCESSABLE = auto()  # 처리 불가 - 반드시 이미지화


class BlockProcessability(Enum):
    """블록 처리 가능성"""
    TEXT_EXTRACTABLE = auto()      # 텍스트 추출 가능
    TABLE_EXTRACTABLE = auto()     # 테이블 추출 가능
    NEEDS_OCR = auto()             # OCR 필요
    IMAGE_REQUIRED = auto()        # 이미지화 필수


@dataclass
class TableQualityResult:
    """테이블 품질 분석 결과"""
    bbox: Tuple[float, float, float, float]
    quality: TableQuality
    score: float  # 0.0 ~ 1.0 (높을수록 좋음)
    
    # 세부 점수
    border_completeness: float = 1.0  # 테두리 완전성
    grid_regularity: float = 1.0      # 그리드 규칙성
    cell_structure: float = 1.0       # 셀 구조 품질
    no_complex_elements: float = 1.0  # 복잡한 요소 부재
    
    # 처리 권장 사항
    recommended_action: BlockProcessability = BlockProcessability.TABLE_EXTRACTABLE
    
    # 문제점
    issues: List[str] = field(default_factory=list)


@dataclass
class BlockAnalysisResult:
    """블록 분석 결과"""
    bbox: Tuple[float, float, float, float]
    block_type: str  # 'table', 'text', 'image', 'mixed', 'unknown'
    
    # 처리 가능성
    processability: BlockProcessability
    
    # 테이블인 경우 품질 정보
    table_quality: Optional[TableQualityResult] = None
    
    # 텍스트인 경우 품질 정보
    text_quality: float = 1.0  # 0.0 ~ 1.0
    
    # 이미지화가 필요한 이유
    image_reasons: List[str] = field(default_factory=list)
    
    # 권장 전략
    recommended_strategy: str = "text"  # 'text', 'table', 'ocr', 'image'


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TableQualityConfig:
    """테이블 품질 분석 설정"""
    # 테두리 완전성
    BORDER_REQUIRED_SIDES: int = 4     # 완전한 표로 판단하는 최소 변 수
    BORDER_TOLERANCE: float = 5.0      # 테두리 정렬 허용 오차 (pt)
    
    # 그리드 규칙성
    LINE_ANGLE_TOLERANCE: float = 2.0  # 수평/수직 판단 각도 허용치 (도)
    GRID_ALIGNMENT_TOLERANCE: float = 3.0  # 그리드 정렬 허용 오차 (pt)
    MIN_ORTHOGONAL_RATIO: float = 0.9  # 직교선 최소 비율 (90% 이상이어야 정상 표)
    
    # 셀 구조
    MIN_CELL_SIZE: float = 10.0        # 최소 셀 크기 (pt)
    MAX_CELL_ASPECT_RATIO: float = 20.0  # 최대 셀 가로세로 비율
    
    # 복잡한 요소
    MAX_CURVE_RATIO: float = 0.05      # 곡선 비율 임계값 (5% 이하)
    MAX_DIAGONAL_RATIO: float = 0.05   # 대각선 비율 임계값
    
    # 품질 등급 임계값
    QUALITY_EXCELLENT: float = 0.95    # EXCELLENT 기준
    QUALITY_GOOD: float = 0.85         # GOOD 기준
    QUALITY_MODERATE: float = 0.65     # MODERATE 기준
    QUALITY_POOR: float = 0.40         # POOR 기준 (이하면 UNPROCESSABLE)


# ============================================================================
# Table Quality Analyzer
# ============================================================================

class TableQualityAnalyzer:
    """
    테이블 품질 분석기
    
    테이블 영역을 분석하여 텍스트 추출 가능 여부를 판단합니다.
    """
    
    def __init__(
        self, 
        page, 
        page_num: int = 0,
        config: Optional[TableQualityConfig] = None
    ):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed), 기본값 0
            config: 분석 설정
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF is required")
        
        self.page = page
        self.page_num = page_num
        self.config = config or TableQualityConfig()
        
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        
        # 캐시
        self._drawings = None
        self._text_dict = None
    
    def analyze_region(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> BlockAnalysisResult:
        """
        영역을 분석하여 처리 방법을 결정합니다.
        
        Args:
            bbox: 분석할 영역
            
        Returns:
            BlockAnalysisResult 객체
        """
        drawings = self._get_region_drawings(bbox)
        text_blocks = self._get_region_text_blocks(bbox)
        
        # 1. 테이블인지 확인
        is_table, table_quality = self._analyze_as_table(bbox, drawings)
        
        if is_table and table_quality:
            # 테이블 품질에 따른 처리 결정
            if table_quality.quality in (TableQuality.EXCELLENT, TableQuality.GOOD):
                return BlockAnalysisResult(
                    bbox=bbox,
                    block_type="table",
                    processability=BlockProcessability.TABLE_EXTRACTABLE,
                    table_quality=table_quality,
                    recommended_strategy="table"
                )
            elif table_quality.quality == TableQuality.MODERATE:
                return BlockAnalysisResult(
                    bbox=bbox,
                    block_type="table",
                    processability=BlockProcessability.TABLE_EXTRACTABLE,
                    table_quality=table_quality,
                    recommended_strategy="table",  # 시도 권장
                    image_reasons=table_quality.issues
                )
            else:
                return BlockAnalysisResult(
                    bbox=bbox,
                    block_type="table",
                    processability=BlockProcessability.IMAGE_REQUIRED,
                    table_quality=table_quality,
                    recommended_strategy="image",
                    image_reasons=table_quality.issues
                )
        
        # 2. 텍스트 블록인지 확인
        text_quality = self._analyze_text_quality(text_blocks)
        
        if text_blocks and text_quality >= 0.7:
            return BlockAnalysisResult(
                bbox=bbox,
                block_type="text",
                processability=BlockProcessability.TEXT_EXTRACTABLE,
                text_quality=text_quality,
                recommended_strategy="text"
            )
        elif text_blocks and text_quality >= 0.4:
            return BlockAnalysisResult(
                bbox=bbox,
                block_type="text",
                processability=BlockProcessability.NEEDS_OCR,
                text_quality=text_quality,
                recommended_strategy="ocr",
                image_reasons=["Low text quality"]
            )
        
        # 3. 이미지 영역이거나 혼합 영역
        if drawings and len(drawings) > 10:
            # 많은 드로잉 = 복잡한 그래픽
            return BlockAnalysisResult(
                bbox=bbox,
                block_type="mixed",
                processability=BlockProcessability.IMAGE_REQUIRED,
                recommended_strategy="image",
                image_reasons=["Complex graphics detected"]
            )
        
        # 4. 알 수 없는 영역 - 이미지화
        return BlockAnalysisResult(
            bbox=bbox,
            block_type="unknown",
            processability=BlockProcessability.IMAGE_REQUIRED,
            recommended_strategy="image",
            image_reasons=["Unknown content type"]
        )
    
    def analyze_table(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> TableQualityResult:
        """
        테이블 영역의 품질을 분석합니다.
        
        Args:
            bbox: 테이블 영역
            
        Returns:
            TableQualityResult 객체
        """
        drawings = self._get_region_drawings(bbox)
        
        issues = []
        
        # 1. 테두리 완전성 분석
        border_score, border_issues = self._analyze_border_completeness(bbox, drawings)
        issues.extend(border_issues)
        
        # 2. 그리드 규칙성 분석
        grid_score, grid_issues = self._analyze_grid_regularity(bbox, drawings)
        issues.extend(grid_issues)
        
        # 3. 셀 구조 분석
        cell_score, cell_issues = self._analyze_cell_structure(bbox, drawings)
        issues.extend(cell_issues)
        
        # 4. 복잡한 요소 분석
        simple_score, simple_issues = self._analyze_element_simplicity(bbox, drawings)
        issues.extend(simple_issues)
        
        # 전체 점수 계산 (가중 평균)
        total_score = (
            border_score * 0.30 +   # 테두리 완전성 30%
            grid_score * 0.30 +     # 그리드 규칙성 30%
            cell_score * 0.20 +     # 셀 구조 20%
            simple_score * 0.20     # 요소 단순성 20%
        )
        
        # 품질 등급 결정
        if total_score >= self.config.QUALITY_EXCELLENT:
            quality = TableQuality.EXCELLENT
            action = BlockProcessability.TABLE_EXTRACTABLE
        elif total_score >= self.config.QUALITY_GOOD:
            quality = TableQuality.GOOD
            action = BlockProcessability.TABLE_EXTRACTABLE
        elif total_score >= self.config.QUALITY_MODERATE:
            quality = TableQuality.MODERATE
            action = BlockProcessability.TABLE_EXTRACTABLE
        elif total_score >= self.config.QUALITY_POOR:
            quality = TableQuality.POOR
            action = BlockProcessability.IMAGE_REQUIRED
        else:
            quality = TableQuality.UNPROCESSABLE
            action = BlockProcessability.IMAGE_REQUIRED
        
        logger.debug(f"[TableQualityAnalyzer] Table at {bbox}: "
                    f"quality={quality.name}, score={total_score:.2f}, "
                    f"border={border_score:.2f}, grid={grid_score:.2f}, "
                    f"cell={cell_score:.2f}, simple={simple_score:.2f}")
        
        return TableQualityResult(
            bbox=bbox,
            quality=quality,
            score=total_score,
            border_completeness=border_score,
            grid_regularity=grid_score,
            cell_structure=cell_score,
            no_complex_elements=simple_score,
            recommended_action=action,
            issues=issues
        )
    
    def _get_region_drawings(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """영역 내 드로잉 추출"""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        
        result = []
        for d in self._drawings:
            rect = d.get("rect")
            if rect and self._bbox_overlaps(bbox, (rect.x0, rect.y0, rect.x1, rect.y1)):
                result.append(d)
        return result
    
    def _get_drawings_cached(self) -> List[Dict]:
        """전체 페이지의 드로잉을 캐시하여 반환"""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings
    
    def _get_region_text_blocks(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """영역 내 텍스트 블록 추출"""
        if self._text_dict is None:
            self._text_dict = self.page.get_text("dict", sort=True)
        
        result = []
        for block in self._text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_bbox = block.get("bbox", (0, 0, 0, 0))
            if self._bbox_overlaps(bbox, block_bbox):
                result.append(block)
        return result
    
    def _analyze_as_table(
        self, 
        bbox: Tuple[float, float, float, float],
        drawings: List[Dict]
    ) -> Tuple[bool, Optional[TableQualityResult]]:
        """영역이 테이블인지 분석"""
        # 선 추출
        lines = self._extract_lines(drawings)
        
        # 최소한의 선이 있어야 테이블
        if len(lines) < 4:  # 최소 4개 선 (사각형)
            return False, None
        
        # 수평선과 수직선 분리
        h_lines = [l for l in lines if l['is_horizontal']]
        v_lines = [l for l in lines if l['is_vertical']]
        
        # 수평선과 수직선이 모두 있어야 테이블
        if len(h_lines) < 2 or len(v_lines) < 2:
            return False, None
        
        # 테이블로 판단되면 품질 분석
        quality = self.analyze_table(bbox)
        return True, quality
    
    def _analyze_border_completeness(
        self, 
        bbox: Tuple[float, float, float, float],
        drawings: List[Dict]
    ) -> Tuple[float, List[str]]:
        """테두리 완전성 분석"""
        issues = []
        lines = self._extract_lines(drawings)
        
        if not lines:
            issues.append("No border lines detected")
            return 0.0, issues
        
        # 테두리 검출
        tolerance = self.config.BORDER_TOLERANCE
        x0, y0, x1, y1 = bbox
        
        has_top = False
        has_bottom = False
        has_left = False
        has_right = False
        
        for line in lines:
            if line['is_horizontal']:
                # 상단 테두리
                if abs(line['y1'] - y0) <= tolerance and line['x1'] >= x0 and line['x2'] <= x1:
                    has_top = True
                # 하단 테두리
                elif abs(line['y1'] - y1) <= tolerance and line['x1'] >= x0 and line['x2'] <= x1:
                    has_bottom = True
            
            if line['is_vertical']:
                # 좌측 테두리
                if abs(line['x1'] - x0) <= tolerance and line['y1'] >= y0 and line['y2'] <= y1:
                    has_left = True
                # 우측 테두리
                elif abs(line['x1'] - x1) <= tolerance and line['y1'] >= y0 and line['y2'] <= y1:
                    has_right = True
        
        sides = [has_top, has_bottom, has_left, has_right]
        complete_sides = sum(sides)
        
        if complete_sides < 4:
            missing = []
            if not has_top: missing.append("top")
            if not has_bottom: missing.append("bottom")
            if not has_left: missing.append("left")
            if not has_right: missing.append("right")
            issues.append(f"Missing borders: {', '.join(missing)}")
        
        return complete_sides / 4.0, issues
    
    def _analyze_grid_regularity(
        self, 
        bbox: Tuple[float, float, float, float],
        drawings: List[Dict]
    ) -> Tuple[float, List[str]]:
        """그리드 규칙성 분석"""
        issues = []
        lines = self._extract_lines(drawings)
        
        if not lines:
            return 0.0, ["No grid lines"]
        
        # 직교선 비율 계산
        orthogonal_count = sum(1 for l in lines if l['is_horizontal'] or l['is_vertical'])
        total_lines = len(lines)
        
        orthogonal_ratio = orthogonal_count / total_lines if total_lines > 0 else 0
        
        if orthogonal_ratio < self.config.MIN_ORTHOGONAL_RATIO:
            issues.append(f"Non-orthogonal lines: {(1-orthogonal_ratio)*100:.1f}%")
        
        # 선 정렬 분석
        h_lines = [l for l in lines if l['is_horizontal']]
        v_lines = [l for l in lines if l['is_vertical']]
        
        # 수평선들의 Y 좌표 정렬
        h_alignment = self._check_line_alignment([l['y1'] for l in h_lines])
        # 수직선들의 X 좌표 정렬
        v_alignment = self._check_line_alignment([l['x1'] for l in v_lines])
        
        alignment_score = (h_alignment + v_alignment) / 2
        
        if alignment_score < 0.8:
            issues.append("Misaligned grid lines")
        
        return (orthogonal_ratio * 0.6 + alignment_score * 0.4), issues
    
    def _analyze_cell_structure(
        self, 
        bbox: Tuple[float, float, float, float],
        drawings: List[Dict]
    ) -> Tuple[float, List[str]]:
        """셀 구조 분석"""
        issues = []
        lines = self._extract_lines(drawings)
        
        h_lines = sorted([l for l in lines if l['is_horizontal']], key=lambda l: l['y1'])
        v_lines = sorted([l for l in lines if l['is_vertical']], key=lambda l: l['x1'])
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            issues.append("Insufficient lines for cell structure")
            return 0.5, issues
        
        # 셀 크기 분석
        cell_heights = []
        for i in range(len(h_lines) - 1):
            height = h_lines[i+1]['y1'] - h_lines[i]['y1']
            if height > 0:
                cell_heights.append(height)
        
        cell_widths = []
        for i in range(len(v_lines) - 1):
            width = v_lines[i+1]['x1'] - v_lines[i]['x1']
            if width > 0:
                cell_widths.append(width)
        
        # 너무 작은 셀 확인
        tiny_cells = 0
        for h in cell_heights:
            if h < self.config.MIN_CELL_SIZE:
                tiny_cells += 1
        for w in cell_widths:
            if w < self.config.MIN_CELL_SIZE:
                tiny_cells += 1
        
        total_cells = len(cell_heights) + len(cell_widths)
        if total_cells > 0 and tiny_cells / total_cells > 0.1:
            issues.append("Too many tiny cells")
        
        # 극단적인 가로세로 비율 확인
        extreme_ratio_count = 0
        for h in cell_heights:
            for w in cell_widths:
                if h > 0 and w > 0:
                    ratio = max(h/w, w/h)
                    if ratio > self.config.MAX_CELL_ASPECT_RATIO:
                        extreme_ratio_count += 1
        
        if extreme_ratio_count > 0:
            issues.append("Extreme cell aspect ratios")
        
        # 점수 계산
        score = 1.0
        if tiny_cells > 0:
            score -= 0.2
        if extreme_ratio_count > 0:
            score -= 0.2
        
        return max(0.0, score), issues
    
    def _analyze_element_simplicity(
        self, 
        bbox: Tuple[float, float, float, float],
        drawings: List[Dict]
    ) -> Tuple[float, List[str]]:
        """요소 단순성 분석 (곡선, 대각선 등 복잡한 요소 부재)"""
        issues = []
        
        if not drawings:
            return 1.0, issues
        
        curve_count = 0
        diagonal_count = 0
        fill_count = 0
        total_items = 0
        
        for d in drawings:
            items = d.get("items", [])
            total_items += len(items)
            
            for item in items:
                item_type = item[0]
                if item_type == 'c':  # 곡선
                    curve_count += 1
                elif item_type == 'l':  # 선
                    # 대각선 확인
                    p1, p2 = item[1], item[2]
                    if not self._is_orthogonal_line(p1, p2):
                        diagonal_count += 1
            
            if d.get("fill"):
                fill_count += 1
        
        # 비율 계산
        curve_ratio = curve_count / max(1, total_items)
        diagonal_ratio = diagonal_count / max(1, total_items)
        fill_ratio = fill_count / max(1, len(drawings))
        
        # 문제 감지
        if curve_ratio > self.config.MAX_CURVE_RATIO:
            issues.append(f"Too many curves: {curve_ratio*100:.1f}%")
        
        if diagonal_ratio > self.config.MAX_DIAGONAL_RATIO:
            issues.append(f"Too many diagonals: {diagonal_ratio*100:.1f}%")
        
        if fill_ratio > 0.5:
            issues.append("Heavy fill patterns")
        
        # 점수 계산
        score = 1.0
        score -= min(0.3, curve_ratio * 3)
        score -= min(0.3, diagonal_ratio * 3)
        score -= min(0.2, fill_ratio * 0.4)
        
        return max(0.0, score), issues
    
    def _extract_lines(self, drawings: List[Dict]) -> List[Dict]:
        """드로잉에서 선 추출"""
        lines = []
        
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == 'l':  # 직선
                    p1, p2 = item[1], item[2]
                    x1, y1 = p1.x, p1.y
                    x2, y2 = p2.x, p2.y
                    
                    # 수평/수직 판단
                    angle_tolerance = self.config.LINE_ANGLE_TOLERANCE
                    is_horizontal = abs(y2 - y1) <= angle_tolerance
                    is_vertical = abs(x2 - x1) <= angle_tolerance
                    
                    lines.append({
                        'x1': min(x1, x2),
                        'y1': min(y1, y2),
                        'x2': max(x1, x2),
                        'y2': max(y1, y2),
                        'is_horizontal': is_horizontal,
                        'is_vertical': is_vertical,
                        'length': ((x2-x1)**2 + (y2-y1)**2) ** 0.5
                    })
                elif item[0] == 're':  # 사각형
                    rect = item[1]
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                    
                    # 사각형의 4변을 선으로 추가
                    lines.extend([
                        {'x1': x0, 'y1': y0, 'x2': x1, 'y2': y0, 'is_horizontal': True, 'is_vertical': False, 'length': x1-x0},  # top
                        {'x1': x0, 'y1': y1, 'x2': x1, 'y2': y1, 'is_horizontal': True, 'is_vertical': False, 'length': x1-x0},  # bottom
                        {'x1': x0, 'y1': y0, 'x2': x0, 'y2': y1, 'is_horizontal': False, 'is_vertical': True, 'length': y1-y0},  # left
                        {'x1': x1, 'y1': y0, 'x2': x1, 'y2': y1, 'is_horizontal': False, 'is_vertical': True, 'length': y1-y0},  # right
                    ])
        
        return lines
    
    def _is_orthogonal_line(self, p1, p2) -> bool:
        """선이 수평/수직인지 확인"""
        tolerance = self.config.LINE_ANGLE_TOLERANCE
        return abs(p2.x - p1.x) <= tolerance or abs(p2.y - p1.y) <= tolerance
    
    def _check_line_alignment(self, positions: List[float]) -> float:
        """선 정렬 품질 확인"""
        if len(positions) < 2:
            return 1.0
        
        # 클러스터링
        tolerance = self.config.GRID_ALIGNMENT_TOLERANCE
        sorted_pos = sorted(positions)
        
        clusters = []
        current_cluster = [sorted_pos[0]]
        
        for pos in sorted_pos[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                clusters.append(current_cluster)
                current_cluster = [pos]
        clusters.append(current_cluster)
        
        # 잘 정렬된 비율
        well_aligned = sum(len(c) for c in clusters if len(c) > 1)
        return well_aligned / len(positions) if positions else 1.0
    
    def _analyze_text_quality(self, text_blocks: List[Dict]) -> float:
        """텍스트 품질 분석"""
        if not text_blocks:
            return 0.0
        
        total_chars = 0
        bad_chars = 0
        
        for block in text_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    total_chars += len(text)
                    
                    for char in text:
                        code = ord(char)
                        if 0xE000 <= code <= 0xF8FF:  # PUA
                            bad_chars += 1
        
        if total_chars == 0:
            return 0.0
        
        return 1.0 - (bad_chars / total_chars)
    
    def _bbox_overlaps(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """두 bbox가 겹치는지 확인"""
        return not (
            bbox1[2] <= bbox2[0] or
            bbox1[0] >= bbox2[2] or
            bbox1[3] <= bbox2[1] or
            bbox1[1] >= bbox2[3]
        )
    
    def analyze_page_tables(self) -> Dict[str, Any]:
        """
        페이지의 모든 테이블 후보 영역을 분석합니다.
        
        Returns:
            Dict containing:
                - table_candidates: 테이블 후보 목록 (각각 품질 정보 포함)
                - has_processable_tables: 처리 가능한 테이블 존재 여부
                - summary: 분석 요약
        """
        # drawing에서 테이블 후보 영역 탐색
        drawings = self._get_drawings_cached()
        
        # 선 추출
        h_lines = []
        v_lines = []
        
        for d in drawings:
            items = d.get("items", [])
            for item in items:
                cmd = item[0] if item else None
                
                if cmd == "l":  # line
                    x0, y0, x1, y1 = item[1], item[2], item[3], item[4]
                    
                    if abs(y1 - y0) < 3:  # 수평선
                        h_lines.append((min(x0, x1), y0, max(x0, x1), y1))
                    elif abs(x1 - x0) < 3:  # 수직선
                        v_lines.append((x0, min(y0, y1), x1, max(y0, y1)))
                        
                elif cmd == "re":  # rect
                    x, y, w, h = item[1], item[2], item[3], item[4]
                    if w > 20 and h > 10:
                        # 사각형의 네 변을 선으로 추가
                        h_lines.append((x, y, x + w, y))  # top
                        h_lines.append((x, y + h, x + w, y + h))  # bottom
                        v_lines.append((x, y, x, y + h))  # left
                        v_lines.append((x + w, y, x + w, y + h))  # right
        
        # 테이블 후보 영역 찾기 (선들이 밀집된 영역)
        table_candidates = self._find_table_regions(h_lines, v_lines)
        
        results = []
        for bbox in table_candidates:
            quality_result = self.analyze_table(bbox)
            results.append({
                'bbox': bbox,
                'quality': quality_result.quality,
                'score': quality_result.score,
                'is_processable': quality_result.recommended_action == BlockProcessability.TABLE_EXTRACTABLE,
                'issues': quality_result.issues
            })
        
        has_processable = any(r['is_processable'] for r in results)
        
        summary = {
            'total_candidates': len(results),
            'processable': sum(1 for r in results if r['is_processable']),
            'unprocessable': sum(1 for r in results if not r['is_processable']),
        }
        
        logger.info(f"[TableQualityAnalyzer] Page {self.page_num + 1}: "
                   f"Found {summary['total_candidates']} table candidates, "
                   f"{summary['processable']} processable")
        
        return {
            'table_candidates': results,
            'has_processable_tables': has_processable,
            'summary': summary
        }
    
    def _find_table_regions(
        self, 
        h_lines: List[Tuple],
        v_lines: List[Tuple]
    ) -> List[Tuple[float, float, float, float]]:
        """
        수평선과 수직선이 교차하는 영역을 테이블 후보로 탐색
        """
        if not h_lines or not v_lines:
            return []
        
        # 선들의 bounding box 계산
        all_lines = h_lines + v_lines
        if not all_lines:
            return []
        
        # 선들을 클러스터링하여 테이블 영역 찾기
        clusters = []
        used = set()
        
        # 단순화된 접근: 선들이 교차하거나 가까운 영역을 그룹화
        tolerance = 50  # pixels
        
        for i, line1 in enumerate(all_lines):
            if i in used:
                continue
            
            cluster = [line1]
            used.add(i)
            
            for j, line2 in enumerate(all_lines):
                if j in used:
                    continue
                
                # 두 선이 가까우면 같은 클러스터
                if self._lines_are_close(line1, line2, tolerance):
                    cluster.append(line2)
                    used.add(j)
            
            if len(cluster) >= 4:  # 최소 4개 선이 있어야 테이블 후보
                clusters.append(cluster)
        
        # 클러스터를 bounding box로 변환
        table_regions = []
        for cluster in clusters:
            x0 = min(min(l[0], l[2]) for l in cluster)
            y0 = min(min(l[1], l[3]) for l in cluster)
            x1 = max(max(l[0], l[2]) for l in cluster)
            y1 = max(max(l[1], l[3]) for l in cluster)
            
            # 최소 크기 확인
            if (x1 - x0) > 100 and (y1 - y0) > 50:
                table_regions.append((x0, y0, x1, y1))
        
        return table_regions
    
    def _lines_are_close(
        self, 
        line1: Tuple, 
        line2: Tuple, 
        tolerance: float
    ) -> bool:
        """두 선이 가까운지 확인"""
        # line1과 line2의 끝점들 사이 거리 확인
        x1_min, y1_min = min(line1[0], line1[2]), min(line1[1], line1[3])
        x1_max, y1_max = max(line1[0], line1[2]), max(line1[1], line1[3])
        x2_min, y2_min = min(line2[0], line2[2]), min(line2[1], line2[3])
        x2_max, y2_max = max(line2[0], line2[2]), max(line2[1], line2[3])
        
        # 두 선의 bounding box가 겹치거나 가까우면 True
        return not (
            x1_max + tolerance < x2_min or
            x2_max + tolerance < x1_min or
            y1_max + tolerance < y2_min or
            y2_max + tolerance < y1_min
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'TableQuality',
    'BlockProcessability',
    'TableQualityResult',
    'BlockAnalysisResult',
    'TableQualityConfig',
    'TableQualityAnalyzer',
]
