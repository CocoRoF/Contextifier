"""
Line Analysis Engine for PDF Handler V3

PDF의 drawings에서 선을 추출하고 분석합니다.
- 얇은 선 감지
- 이중선 병합
- 불완전 테두리 복구
"""

import logging
import math
from typing import List, Optional, Tuple

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .v3_types import LineInfo, GridInfo, LineThickness, V3Config

logger = logging.getLogger(__name__)


# ============================================================================
# Line Analysis Engine
# ============================================================================

class LineAnalysisEngine:
    """
    선 분석 엔진
    
    PDF의 drawings에서 선을 추출하고 분석합니다.
    - 얇은 선 감지
    - 이중선 병합
    - 불완전 테두리 복구
    """
    
    # 설정 상수 (V3Config에서 가져오거나 기본값 사용)
    THIN_LINE_THRESHOLD = getattr(V3Config, 'THIN_LINE_THRESHOLD', 0.5)
    THICK_LINE_THRESHOLD = getattr(V3Config, 'THICK_LINE_THRESHOLD', 2.0)
    DOUBLE_LINE_GAP = getattr(V3Config, 'DOUBLE_LINE_GAP', 5.0)
    LINE_MERGE_TOLERANCE = getattr(V3Config, 'LINE_MERGE_TOLERANCE', 3.0)
    BORDER_EXTENSION_MARGIN = getattr(V3Config, 'BORDER_EXTENSION_MARGIN', 20.0)
    
    def __init__(self, page, page_width: float, page_height: float):
        """
        Args:
            page: PyMuPDF page 객체
            page_width: 페이지 너비
            page_height: 페이지 높이
        """
        self.page = page
        self.page_width = page_width
        self.page_height = page_height
        self.all_lines: List[LineInfo] = []
        self.h_lines: List[LineInfo] = []  # 가로선
        self.v_lines: List[LineInfo] = []  # 세로선
        
    def analyze(self) -> Tuple[List[LineInfo], List[LineInfo]]:
        """
        선 분석 수행
        
        Returns:
            (가로선 목록, 세로선 목록) 튜플
        """
        self._extract_all_lines()
        self._classify_lines()
        self._merge_double_lines()
        return self.h_lines, self.v_lines
    
    def _extract_all_lines(self):
        """모든 선 추출"""
        drawings = self.page.get_drawings()
        if not drawings:
            return
        
        for drawing in drawings:
            # 선 정보 추출
            items = drawing.get('items', [])
            rect = drawing.get('rect')
            
            if not rect:
                continue
            
            # rect 기반 선 분석
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            
            # 선 두께 추정
            stroke_width = drawing.get('width', 1.0) or 1.0
            
            # 선인지 판단 (가로선 또는 세로선)
            is_h_line = h <= max(3.0, stroke_width * 2) and w > 10
            is_v_line = w <= max(3.0, stroke_width * 2) and h > 10
            
            if not (is_h_line or is_v_line):
                # items에서 'l' (line) 추출 시도
                for item in items:
                    if item[0] == 'l':  # line
                        p1, p2 = item[1], item[2]
                        self._add_line_from_points(p1, p2, stroke_width)
                continue
            
            # 두께 분류
            thickness_class = self._classify_thickness(stroke_width)
            
            line_info = LineInfo(
                x0=x0,
                y0=y0 if is_h_line else y0,
                x1=x1,
                y1=y1 if is_h_line else y1,
                thickness=stroke_width,
                thickness_class=thickness_class,
                is_horizontal=is_h_line,
                is_vertical=is_v_line
            )
            
            self.all_lines.append(line_info)
    
    def _add_line_from_points(self, p1, p2, stroke_width: float):
        """두 점에서 선 생성"""
        x0, y0 = p1.x, p1.y
        x1, y1 = p2.x, p2.y
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # 선 방향 판단 (허용 오차 내에서)
        is_horizontal = dy < 3 and dx > 10
        is_vertical = dx < 3 and dy > 10
        
        if not (is_horizontal or is_vertical):
            return
        
        thickness_class = self._classify_thickness(stroke_width)
        
        line_info = LineInfo(
            x0=min(x0, x1),
            y0=min(y0, y1),
            x1=max(x0, x1),
            y1=max(y0, y1),
            thickness=stroke_width,
            thickness_class=thickness_class,
            is_horizontal=is_horizontal,
            is_vertical=is_vertical
        )
        
        self.all_lines.append(line_info)
    
    def _classify_thickness(self, thickness: float) -> LineThickness:
        """선 두께 분류"""
        if thickness < self.THIN_LINE_THRESHOLD:
            return LineThickness.THIN
        elif thickness > self.THICK_LINE_THRESHOLD:
            return LineThickness.THICK
        return LineThickness.NORMAL
    
    def _classify_lines(self):
        """가로선/세로선 분류"""
        for line in self.all_lines:
            if line.is_horizontal:
                self.h_lines.append(line)
            elif line.is_vertical:
                self.v_lines.append(line)
    
    def _merge_double_lines(self):
        """이중선 병합"""
        # 가로선 병합
        self.h_lines = self._merge_parallel_lines(self.h_lines, is_horizontal=True)
        # 세로선 병합
        self.v_lines = self._merge_parallel_lines(self.v_lines, is_horizontal=False)
    
    def _merge_parallel_lines(self, lines: List[LineInfo], is_horizontal: bool) -> List[LineInfo]:
        """평행한 이중선 병합"""
        if len(lines) < 2:
            return lines
        
        merged = []
        used = set()
        
        # 위치로 정렬
        if is_horizontal:
            sorted_lines = sorted(lines, key=lambda l: (l.y0, l.x0))
        else:
            sorted_lines = sorted(lines, key=lambda l: (l.x0, l.y0))
        
        for i, line1 in enumerate(sorted_lines):
            if i in used:
                continue
            
            merged_line = line1
            
            for j in range(i + 1, len(sorted_lines)):
                if j in used:
                    continue
                
                line2 = sorted_lines[j]
                
                # 이중선 판단
                if self._is_double_line(line1, line2, is_horizontal):
                    # 두 선을 병합 (중간 위치, 최대 범위)
                    merged_line = self._merge_two_lines(merged_line, line2, is_horizontal)
                    used.add(j)
            
            merged.append(merged_line)
            used.add(i)
        
        return merged
    
    def _is_double_line(self, line1: LineInfo, line2: LineInfo, is_horizontal: bool) -> bool:
        """두 선이 이중선인지 판단"""
        if is_horizontal:
            # Y 좌표 차이가 작고 X 범위가 겹치면 이중선
            y_gap = abs(line1.y0 - line2.y0)
            if y_gap > self.DOUBLE_LINE_GAP:
                return False
            
            # X 범위 겹침 확인
            x_overlap = min(line1.x1, line2.x1) - max(line1.x0, line2.x0)
            min_length = min(self._get_line_length(line1), self._get_line_length(line2))
            return x_overlap > min_length * 0.5
        else:
            # X 좌표 차이가 작고 Y 범위가 겹치면 이중선
            x_gap = abs(line1.x0 - line2.x0)
            if x_gap > self.DOUBLE_LINE_GAP:
                return False
            
            # Y 범위 겹침 확인
            y_overlap = min(line1.y1, line2.y1) - max(line1.y0, line2.y0)
            min_length = min(self._get_line_length(line1), self._get_line_length(line2))
            return y_overlap > min_length * 0.5
    
    def _get_line_length(self, line: LineInfo) -> float:
        """선 길이 계산"""
        return math.sqrt((line.x1 - line.x0) ** 2 + (line.y1 - line.y0) ** 2)
    
    def _merge_two_lines(self, line1: LineInfo, line2: LineInfo, is_horizontal: bool) -> LineInfo:
        """두 선 병합"""
        if is_horizontal:
            # 중간 Y, 최대 X 범위
            avg_y = (line1.y0 + line2.y0) / 2
            return LineInfo(
                x0=min(line1.x0, line2.x0),
                y0=avg_y,
                x1=max(line1.x1, line2.x1),
                y1=avg_y,
                thickness=max(line1.thickness, line2.thickness),
                thickness_class=line1.thickness_class if line1.thickness >= line2.thickness else line2.thickness_class,
                is_horizontal=True,
                is_vertical=False
            )
        else:
            # 중간 X, 최대 Y 범위
            avg_x = (line1.x0 + line2.x0) / 2
            return LineInfo(
                x0=avg_x,
                y0=min(line1.y0, line2.y0),
                x1=avg_x,
                y1=max(line1.y1, line2.y1),
                thickness=max(line1.thickness, line2.thickness),
                thickness_class=line1.thickness_class if line1.thickness >= line2.thickness else line2.thickness_class,
                is_horizontal=False,
                is_vertical=True
            )
    
    def build_grid(self, tolerance: float = None) -> Optional[GridInfo]:
        """
        선들로부터 그리드 구성
        
        불완전한 테두리를 복구하고 그리드 구조를 반환합니다.
        
        Args:
            tolerance: 위치 클러스터링 허용 오차
            
        Returns:
            GridInfo 또는 None
        """
        if tolerance is None:
            tolerance = self.LINE_MERGE_TOLERANCE
            
        if not self.h_lines and not self.v_lines:
            return None
        
        # Y 좌표 수집 (가로선)
        h_positions = self._cluster_positions(
            [line.y0 for line in self.h_lines],
            tolerance
        )
        
        # X 좌표 수집 (세로선)
        v_positions = self._cluster_positions(
            [line.x0 for line in self.v_lines],
            tolerance
        )
        
        if len(h_positions) < 2 or len(v_positions) < 2:
            return None
        
        # bbox 계산
        x0 = min(v_positions)
        y0 = min(h_positions)
        x1 = max(v_positions)
        y1 = max(h_positions)
        
        # 테두리 완전성 확인
        is_complete = self._check_border_completeness(h_positions, v_positions)
        
        return GridInfo(
            h_lines=sorted(h_positions),
            v_lines=sorted(v_positions),
            bbox=(x0, y0, x1, y1),
            is_complete=is_complete,
            reconstructed=False
        )
    
    def _cluster_positions(self, positions: List[float], tolerance: float) -> List[float]:
        """유사한 위치를 클러스터링"""
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        clusters = [[sorted_pos[0]]]
        
        for pos in sorted_pos[1:]:
            if pos - clusters[-1][-1] <= tolerance:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        # 각 클러스터의 중앙값 반환
        return [sum(c) / len(c) for c in clusters]
    
    def _check_border_completeness(self, h_positions: List[float], v_positions: List[float]) -> bool:
        """테두리 완전성 확인"""
        if len(h_positions) < 2 or len(v_positions) < 2:
            return False
        
        y_min, y_max = min(h_positions), max(h_positions)
        x_min, x_max = min(v_positions), max(v_positions)
        
        # 상단/하단에 충분한 가로선이 있는지
        has_top = any(line.y0 <= y_min + self.LINE_MERGE_TOLERANCE for line in self.h_lines)
        has_bottom = any(line.y0 >= y_max - self.LINE_MERGE_TOLERANCE for line in self.h_lines)
        
        # 좌측/우측에 충분한 세로선이 있는지
        has_left = any(line.x0 <= x_min + self.LINE_MERGE_TOLERANCE for line in self.v_lines)
        has_right = any(line.x0 >= x_max - self.LINE_MERGE_TOLERANCE for line in self.v_lines)
        
        return all([has_top, has_bottom, has_left, has_right])
    
    def reconstruct_incomplete_border(self, grid: GridInfo) -> GridInfo:
        """
        불완전한 테두리 복구
        
        3면 이상 존재할 경우 4면으로 완성합니다.
        
        Args:
            grid: 기존 GridInfo
            
        Returns:
            복구된 GridInfo
        """
        if grid.is_complete:
            return grid
        
        h_lines = list(grid.h_lines)
        v_lines = list(grid.v_lines)
        
        y_min, y_max = min(h_lines), max(h_lines)
        x_min, x_max = min(v_lines), max(v_lines)
        
        reconstructed = False
        
        # 상단 가로선 확인/추가
        has_top = any(abs(y - y_min) < self.LINE_MERGE_TOLERANCE for y in h_lines)
        if not has_top and len(h_lines) >= 2:
            # 상단 경계 추정
            h_lines.insert(0, y_min - self.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        
        # 하단 가로선 확인/추가
        has_bottom = any(abs(y - y_max) < self.LINE_MERGE_TOLERANCE for y in h_lines)
        if not has_bottom and len(h_lines) >= 2:
            h_lines.append(y_max + self.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        
        # 좌측 세로선 확인/추가
        has_left = any(abs(x - x_min) < self.LINE_MERGE_TOLERANCE for x in v_lines)
        if not has_left and len(v_lines) >= 2:
            v_lines.insert(0, x_min - self.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        
        # 우측 세로선 확인/추가
        has_right = any(abs(x - x_max) < self.LINE_MERGE_TOLERANCE for x in v_lines)
        if not has_right and len(v_lines) >= 2:
            v_lines.append(x_max + self.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        
        if not reconstructed:
            return grid
        
        new_x0 = min(v_lines)
        new_y0 = min(h_lines)
        new_x1 = max(v_lines)
        new_y1 = max(h_lines)
        
        return GridInfo(
            h_lines=sorted(h_lines),
            v_lines=sorted(v_lines),
            bbox=(new_x0, new_y0, new_x1, new_y1),
            is_complete=True,
            reconstructed=True
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'LineAnalysisEngine',
]
