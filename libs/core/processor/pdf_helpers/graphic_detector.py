"""
Graphic Region Detector for PDF Handler

Detects graphic regions (charts, diagrams, icons, etc.) in PDF pages.
These regions are filtered to avoid being misidentified as tables.
"""

import logging
from typing import List, Dict, Tuple, Optional

import fitz

from libs.core.processor.pdf_helpers.types import GraphicRegionInfo, PDFConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Graphic Region Detector
# ============================================================================

class GraphicRegionDetector:
    """
    Graphic Region Detector
    
    Detects graphic regions such as charts, diagrams, and icons in PDF pages.
    These regions should be excluded from table detection.
    
    Criteria for identifying graphics:
    1. High ratio of curves (Bezier curves) - tables are mostly straight lines
    2. Many filled shapes - areas filled with colors
    3. Use of various colors - tables are usually monochromatic
    4. High density of curves/lines within the region
    """
    
    def __init__(self, page, page_num: int):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
        """
        self.page = page
        self.page_num = page_num
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.graphic_regions: List[GraphicRegionInfo] = []
        self._drawings_cache: Optional[List[Dict]] = None
    
    def detect(self) -> List[GraphicRegionInfo]:
        """
        Perform graphic region detection
        
        Returns:
            List of GraphicRegionInfo
        """
        drawings = self._get_drawings()
        if not drawings:
            return []
        
        # Cluster drawings
        regions = self._cluster_drawings(drawings)
        
        # Analyze each region
        for region in regions:
            self._analyze_region(region)
        
        # Return only regions identified as graphics
        self.graphic_regions = [r for r in regions if r.is_graphic]
        
        logger.debug(f"[GraphicDetector] Page {self.page_num + 1}: Found {len(self.graphic_regions)} graphic regions")
        
        return self.graphic_regions
    
    def _get_drawings(self) -> List[Dict]:
        """Cache drawing data"""
        if self._drawings_cache is None:
            self._drawings_cache = self.page.get_drawings()
        return self._drawings_cache
    
    def _cluster_drawings(self, drawings: List[Dict]) -> List[GraphicRegionInfo]:
        """
        Cluster adjacent drawings into a single region
        """
        regions: List[Dict] = []
        
        for drawing in drawings:
            rect = drawing.get("rect", fitz.Rect())
            if rect.is_empty or rect.is_infinite:
                continue
            
            items = drawing.get("items", [])
            fill = drawing.get("fill")
            stroke = drawing.get("color")
            
            # Count each item type
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
            
            # Collect colors
            if fill:
                region_data['colors'].add(tuple(fill) if isinstance(fill, (list, tuple)) else fill)
            if stroke:
                region_data['colors'].add(tuple(stroke) if isinstance(stroke, (list, tuple)) else stroke)
            
            # Check if can be merged with existing regions
            merged = False
            for existing in regions:
                if self._should_merge_regions(existing['bbox'], region_data['bbox']):
                    self._merge_region_data(existing, region_data)
                    merged = True
                    break
            
            if not merged:
                regions.append(region_data)
        
        # Iteratively merge adjacent regions
        regions = self._iterative_merge(regions)
        
        # Convert to GraphicRegionInfo
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
        """Check if two regions should be merged"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        
        # Check overlap with margin consideration
        if (x0_1 - margin <= x1_2 and x1_1 + margin >= x0_2 and
            y0_1 - margin <= y1_2 and y1_1 + margin >= y0_2):
            return True
        return False
    
    def _merge_region_data(self, target: Dict, source: Dict):
        """Merge two region data"""
        # Merge bboxes
        x0 = min(target['bbox'][0], source['bbox'][0])
        y0 = min(target['bbox'][1], source['bbox'][1])
        x1 = max(target['bbox'][2], source['bbox'][2])
        y1 = max(target['bbox'][3], source['bbox'][3])
        target['bbox'] = (x0, y0, x1, y1)
        
        # Accumulate counts
        target['curve_count'] += source['curve_count']
        target['line_count'] += source['line_count']
        target['rect_count'] += source['rect_count']
        target['fill_count'] += source['fill_count']
        target['colors'].update(source['colors'])
    
    def _iterative_merge(self, regions: List[Dict], max_iterations: int = 5) -> List[Dict]:
        """Iteratively merge adjacent regions"""
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
        Analyze whether the region is a graphic
        
        Criteria for identifying graphics:
        1. High ratio of curves (Bezier)
        2. Many filled shapes
        3. Use of various colors
        4. High line/curve density relative to region size
        5. Chart pattern detection (curve + fill combination)
        
        Table cells (grid-shaped rectangles) are excluded from graphics.
        """
        total_items = region.curve_count + region.line_count + region.rect_count
        
        if total_items == 0:
            region.is_graphic = False
            region.confidence = 0.0
            return
        
        reasons = []
        score = 0.0
        
        # 1. Curve ratio check (pie charts, curved graphs, etc.)
        curve_ratio = region.curve_count / total_items if total_items > 0 else 0
        if curve_ratio >= PDFConfig.GRAPHIC_CURVE_RATIO_THRESHOLD:
            score += 0.4
            reasons.append(f"curve_ratio={curve_ratio:.2f}")
        
        # 2. Minimum curve count check
        if region.curve_count >= PDFConfig.GRAPHIC_MIN_CURVE_COUNT:
            score += 0.2
            reasons.append(f"curves={region.curve_count}")
        
        # 3. Filled shape ratio
        fill_ratio = region.fill_count / max(1, total_items // 10)  # Rough estimate of shape count
        if fill_ratio >= PDFConfig.GRAPHIC_FILL_RATIO_THRESHOLD:
            score += 0.2
            reasons.append(f"fills={region.fill_count}")
        
        # 4. Color diversity (charts usually use multiple colors)
        if region.color_count >= PDFConfig.GRAPHIC_COLOR_VARIETY_THRESHOLD:
            score += 0.2
            reasons.append(f"colors={region.color_count}")
        
        # 5. Chart pattern with curves
        # If curves exist with many fills, high probability of being a chart
        if region.curve_count >= 5 and region.fill_count >= 3:
            score += 0.3
            reasons.append(f"chart_pattern(curves={region.curve_count}, fills={region.fill_count})")
        
        # 6. Only rectangles with no curves - possibly table cells!
        # Table cells are not graphics
        if region.rect_count >= 5 and region.curve_count == 0 and region.line_count == 0:
            # Only rectangles = high probability of table
            # May be chart if high color diversity or irregular rectangle sizes
            if region.color_count >= 3:
                # Multiple colors = possibly a chart
                score += 0.2
                reasons.append(f"colored_rects(rects={region.rect_count}, colors={region.color_count})")
            else:
                # Single-colored rectangles only = high probability of table cells
                score -= 0.3
                reasons.append(f"likely_table_cells(rects={region.rect_count}, single_color)")
        
        # 7. Exclude page background (full page size)
        bbox_width = region.bbox[2] - region.bbox[0]
        bbox_height = region.bbox[3] - region.bbox[1]
        if (bbox_width > self.page_width * 0.9 and 
            bbox_height > self.page_height * 0.9):
            score = 0.0
            reasons = ["page_background"]
        
        # 8. Too small regions are not graphics (excluding icons)
        area = bbox_width * bbox_height
        if area < 500:  # Less than approximately 22x22pt
            score *= 0.5
        
        region.confidence = min(1.0, max(0.0, score))
        region.is_graphic = score >= 0.5
        region.reason = ", ".join(reasons) if reasons else "not_graphic"
        
        if region.is_graphic:
            logger.debug(f"[GraphicDetector] Graphic region detected: {region.bbox}, score={score:.2f}, {region.reason}")
    
    def is_bbox_in_graphic_region(self, bbox: Tuple[float, float, float, float], 
                                   threshold: float = 0.3) -> bool:
        """
        Check if the given bbox is within a graphic region
        
        Args:
            bbox: The region to check
            threshold: Overlap ratio threshold
            
        Returns:
            True if within a graphic region
        """
        for graphic in self.graphic_regions:
            overlap = self._calculate_overlap_ratio(bbox, graphic.bbox)
            if overlap >= threshold:
                return True
        return False
    
    def _calculate_overlap_ratio(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap ratio between two bboxes"""
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
