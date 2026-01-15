"""
Graphic Region Detector for PDF Handler V3

PDF 페이지에서 그래픽 영역(차트, 다이어그램, 아이콘 등)을 감지합니다.
이 영역은 테이블로 오인되지 않도록 필터링됩니다.
"""

import logging
from typing import List, Dict, Tuple, Optional

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .v3_types import GraphicRegionInfo, V3Config

logger = logging.getLogger(__name__)


# ============================================================================
# Graphic Region Detector
# ============================================================================

class GraphicRegionDetector:
    """
    그래픽 영역 감지기
    
    PDF 페이지에서 차트, 다이어그램, 아이콘 등의 그래픽 영역을 감지합니다.
    이 영역은 테이블 감지에서 제외되어야 합니다.
    
    그래픽 판단 기준:
    1. 곡선(베지어 커브) 비율이 높음 - 표는 직선 위주
    2. 채워진 도형이 많음 - 색상으로 채워진 영역
    3. 다양한 색상 사용 - 표는 보통 단색
    4. 영역 내 곡선/선 밀도가 높음
    """
    
    def __init__(self, page, page_num: int):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed)
        """
        self.page = page
        self.page_num = page_num
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.graphic_regions: List[GraphicRegionInfo] = []
        self._drawings_cache: Optional[List[Dict]] = None
    
    def detect(self) -> List[GraphicRegionInfo]:
        """
        그래픽 영역 감지 수행
        
        Returns:
            GraphicRegionInfo 목록
        """
        drawings = self._get_drawings()
        if not drawings:
            return []
        
        # 드로잉 클러스터링
        regions = self._cluster_drawings(drawings)
        
        # 각 영역 분석
        for region in regions:
            self._analyze_region(region)
        
        # 그래픽으로 판단된 영역만 반환
        self.graphic_regions = [r for r in regions if r.is_graphic]
        
        logger.debug(f"[GraphicDetector] Page {self.page_num + 1}: Found {len(self.graphic_regions)} graphic regions")
        
        return self.graphic_regions
    
    def _get_drawings(self) -> List[Dict]:
        """드로잉 데이터 캐싱"""
        if self._drawings_cache is None:
            self._drawings_cache = self.page.get_drawings()
        return self._drawings_cache
    
    def _cluster_drawings(self, drawings: List[Dict]) -> List[GraphicRegionInfo]:
        """
        인접한 드로잉들을 하나의 영역으로 클러스터링
        """
        regions: List[Dict] = []
        
        for drawing in drawings:
            rect = drawing.get("rect", fitz.Rect())
            if rect.is_empty or rect.is_infinite:
                continue
            
            items = drawing.get("items", [])
            fill = drawing.get("fill")
            stroke = drawing.get("color")
            
            # 각 아이템 유형 카운트
            curve_count = sum(1 for item in items if item[0] == 'c')
            line_count = sum(1 for item in items if item[0] == 'l')
            rect_count = sum(1 for item in items if item[0] == 're')
            
            region_data = {
                'bbox': tuple(rect),
                'curve_count': curve_count,
                'line_count': line_count,
                'rect_count': rect_count,
                'fill_count': 1 if fill else 0,
                'colors': set()
            }
            
            # 색상 수집
            if fill:
                region_data['colors'].add(tuple(fill) if isinstance(fill, (list, tuple)) else fill)
            if stroke:
                region_data['colors'].add(tuple(stroke) if isinstance(stroke, (list, tuple)) else stroke)
            
            # 기존 영역과 병합 가능한지 확인
            merged = False
            for existing in regions:
                if self._should_merge_regions(existing['bbox'], region_data['bbox']):
                    self._merge_region_data(existing, region_data)
                    merged = True
                    break
            
            if not merged:
                regions.append(region_data)
        
        # 인접 영역 반복 병합
        regions = self._iterative_merge(regions)
        
        # GraphicRegionInfo로 변환
        result = []
        for r in regions:
            result.append(GraphicRegionInfo(
                bbox=r['bbox'],
                curve_count=r['curve_count'],
                line_count=r['line_count'],
                rect_count=r['rect_count'],
                fill_count=r['fill_count'],
                color_count=len(r['colors']),
                is_graphic=False,
                confidence=0.0
            ))
        
        return result
    
    def _should_merge_regions(self, bbox1: Tuple, bbox2: Tuple, margin: float = 20.0) -> bool:
        """두 영역이 병합되어야 하는지 확인"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        
        # 마진을 고려한 겹침 확인
        if (x0_1 - margin <= x1_2 and x1_1 + margin >= x0_2 and
            y0_1 - margin <= y1_2 and y1_1 + margin >= y0_2):
            return True
        return False
    
    def _merge_region_data(self, target: Dict, source: Dict):
        """두 영역 데이터 병합"""
        # bbox 병합
        x0 = min(target['bbox'][0], source['bbox'][0])
        y0 = min(target['bbox'][1], source['bbox'][1])
        x1 = max(target['bbox'][2], source['bbox'][2])
        y1 = max(target['bbox'][3], source['bbox'][3])
        target['bbox'] = (x0, y0, x1, y1)
        
        # 카운트 누적
        target['curve_count'] += source['curve_count']
        target['line_count'] += source['line_count']
        target['rect_count'] += source['rect_count']
        target['fill_count'] += source['fill_count']
        target['colors'].update(source['colors'])
    
    def _iterative_merge(self, regions: List[Dict], max_iterations: int = 5) -> List[Dict]:
        """반복적으로 인접 영역 병합"""
        for _ in range(max_iterations):
            merged_any = False
            new_regions = []
            used = set()
            
            for i, r1 in enumerate(regions):
                if i in used:
                    continue
                
                current = r1.copy()
                current['colors'] = r1['colors'].copy()
                
                for j, r2 in enumerate(regions):
                    if j <= i or j in used:
                        continue
                    
                    if self._should_merge_regions(current['bbox'], r2['bbox']):
                        self._merge_region_data(current, r2)
                        used.add(j)
                        merged_any = True
                
                new_regions.append(current)
            
            regions = new_regions
            
            if not merged_any:
                break
        
        return regions
    
    def _analyze_region(self, region: GraphicRegionInfo):
        """
        영역이 그래픽인지 분석
        
        그래픽 판단 기준:
        1. 곡선(베지어) 비율이 높음
        2. 채워진 도형이 많음
        3. 다양한 색상 사용
        4. 영역 크기 대비 선/곡선 밀도가 높음
        5. V3.2: 차트 패턴 감지 (곡선 + 채우기 조합)
        
        V3.2 개선: 테이블 셀(격자 형태의 사각형)은 그래픽에서 제외
        """
        total_items = region.curve_count + region.line_count + region.rect_count
        
        if total_items == 0:
            region.is_graphic = False
            region.confidence = 0.0
            return
        
        reasons = []
        score = 0.0
        
        # 1. 곡선 비율 체크 (원형 차트, 곡선 그래프 등)
        curve_ratio = region.curve_count / total_items if total_items > 0 else 0
        if curve_ratio >= V3Config.GRAPHIC_CURVE_RATIO_THRESHOLD:
            score += 0.4
            reasons.append(f"curve_ratio={curve_ratio:.2f}")
        
        # 2. 최소 곡선 수 체크
        if region.curve_count >= V3Config.GRAPHIC_MIN_CURVE_COUNT:
            score += 0.2
            reasons.append(f"curves={region.curve_count}")
        
        # 3. 채워진 도형 비율
        fill_ratio = region.fill_count / max(1, total_items // 10)  # 대략적 도형 수 추정
        if fill_ratio >= V3Config.GRAPHIC_FILL_RATIO_THRESHOLD:
            score += 0.2
            reasons.append(f"fills={region.fill_count}")
        
        # 4. 색상 다양성 (차트는 보통 여러 색상 사용)
        if region.color_count >= V3Config.GRAPHIC_COLOR_VARIETY_THRESHOLD:
            score += 0.2
            reasons.append(f"colors={region.color_count}")
        
        # 5. V3.2: 곡선이 있는 차트 패턴
        # 곡선이 있으면서 채우기가 많으면 차트일 가능성 높음
        if region.curve_count >= 5 and region.fill_count >= 3:
            score += 0.3
            reasons.append(f"chart_pattern(curves={region.curve_count}, fills={region.fill_count})")
        
        # 6. V3.2: 사각형만 있고 곡선이 없는 경우 - 테이블 셀일 가능성!
        # 테이블 셀은 그래픽이 아님
        if region.rect_count >= 5 and region.curve_count == 0 and region.line_count == 0:
            # 사각형만 있으면 테이블일 가능성이 높음
            # 색상 다양성이 높거나, 사각형이 불규칙한 크기면 차트일 수 있음
            if region.color_count >= 3:
                # 여러 색상 = 차트일 가능성
                score += 0.2
                reasons.append(f"colored_rects(rects={region.rect_count}, colors={region.color_count})")
            else:
                # 단일 색상의 사각형만 = 테이블 셀일 가능성 높음
                score -= 0.3
                reasons.append(f"likely_table_cells(rects={region.rect_count}, single_color)")
        
        # 7. 페이지 배경(전체 페이지 크기)은 제외
        bbox_width = region.bbox[2] - region.bbox[0]
        bbox_height = region.bbox[3] - region.bbox[1]
        if (bbox_width > self.page_width * 0.9 and 
            bbox_height > self.page_height * 0.9):
            score = 0.0
            reasons = ["page_background"]
        
        # 8. 너무 작은 영역은 그래픽이 아님 (아이콘 제외)
        area = bbox_width * bbox_height
        if area < 500:  # 약 22x22pt 미만
            score *= 0.5
        
        region.confidence = min(1.0, max(0.0, score))
        region.is_graphic = score >= 0.5
        region.reason = ", ".join(reasons) if reasons else "not_graphic"
        
        if region.is_graphic:
            logger.debug(f"[GraphicDetector] Graphic region detected: {region.bbox}, score={score:.2f}, {region.reason}")
    
    def is_bbox_in_graphic_region(self, bbox: Tuple[float, float, float, float], 
                                   threshold: float = 0.3) -> bool:
        """
        주어진 bbox가 그래픽 영역 내에 있는지 확인
        
        Args:
            bbox: 확인할 영역
            threshold: 겹침 비율 임계값
            
        Returns:
            그래픽 영역 내에 있으면 True
        """
        for graphic in self.graphic_regions:
            overlap = self._calculate_overlap_ratio(bbox, graphic.bbox)
            if overlap >= threshold:
                return True
        return False
    
    def _calculate_overlap_ratio(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """두 bbox의 겹침 비율 계산"""
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])
        
        if x1 <= x0 or y1 <= y0:
            return 0.0
        
        overlap_area = (x1 - x0) * (y1 - y0)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        
        if bbox1_area <= 0:
            return 0.0
        
        return overlap_area / bbox1_area


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'GraphicRegionDetector',
]
