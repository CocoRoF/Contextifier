"""
Line Analysis Engine for PDF Handler

Extracts and analyzes lines from PDF drawings.
- Thin line detection
- Double line merging
- Incomplete border reconstruction
"""

import logging
import math
from typing import List, Optional, Tuple

import fitz

from libs.core.processor.pdf_helpers.types import LineInfo, GridInfo, LineThickness, PDFConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Line Analysis Engine
# ============================================================================

class LineAnalysisEngine:
    """
    Line Analysis Engine
    
    Extracts and analyzes lines from PDF drawings.
    - Thin line detection
    - Double line merging
    - Incomplete border reconstruction
    """
    
    # Configuration constants (from PDFConfig or default values)
    THIN_LINE_THRESHOLD = getattr(PDFConfig, 'THIN_LINE_THRESHOLD', 0.5)
    THICK_LINE_THRESHOLD = getattr(PDFConfig, 'THICK_LINE_THRESHOLD', 2.0)
    DOUBLE_LINE_GAP = getattr(PDFConfig, 'DOUBLE_LINE_GAP', 5.0)
    LINE_MERGE_TOLERANCE = getattr(PDFConfig, 'LINE_MERGE_TOLERANCE', 3.0)
    BORDER_EXTENSION_MARGIN = getattr(PDFConfig, 'BORDER_EXTENSION_MARGIN', 20.0)
    
    def __init__(self, page, page_width: float, page_height: float):
        """
        Args:
            page: PyMuPDF page object
            page_width: Page width
            page_height: Page height
        """
        self.page = page
        self.page_width = page_width
        self.page_height = page_height
        self.all_lines: List[LineInfo] = []
        self.h_lines: List[LineInfo] = []  # Horizontal lines
        self.v_lines: List[LineInfo] = []  # Vertical lines
        
    def analyze(self) -> Tuple[List[LineInfo], List[LineInfo]]:
        """
        Perform line analysis
        
        Returns:
            Tuple of (horizontal lines list, vertical lines list)
        """
        self._extract_all_lines()
        self._classify_lines()
        self._merge_double_lines()
        return self.h_lines, self.v_lines
    
    def _extract_all_lines(self):
        """Extract all lines"""
        drawings = self.page.get_drawings()
        if not drawings:
            return
        
        for drawing in drawings:
            # Extract line information
            items = drawing.get('items', [])
            rect = drawing.get('rect')
            
            if not rect:
                continue
            
            # Rect-based line analysis
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            
            # Estimate line thickness
            stroke_width = drawing.get('width', 1.0) or 1.0
            
            # Determine if it's a line (horizontal or vertical)
            is_h_line = h <= max(3.0, stroke_width * 2) and w > 10
            is_v_line = w <= max(3.0, stroke_width * 2) and h > 10
            
            if not (is_h_line or is_v_line):
                # Try to extract 'l' (line) from items
                for item in items:
                    if item[0] == 'l':  # line
                        p1, p2 = item[1], item[2]
                        self._add_line_from_points(p1, p2, stroke_width)
                continue
            
            # Classify thickness
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
        """Create a line from two points"""
        x0, y0 = p1.x, p1.y
        x1, y1 = p2.x, p2.y
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # Determine line direction (within tolerance)
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
        """Classify line thickness"""
        if thickness < self.THIN_LINE_THRESHOLD:
            return LineThickness.THIN
        elif thickness > self.THICK_LINE_THRESHOLD:
            return LineThickness.THICK
        return LineThickness.NORMAL
    
    def _classify_lines(self):
        """Classify horizontal/vertical lines"""
        for line in self.all_lines:
            if line.is_horizontal:
                self.h_lines.append(line)
            elif line.is_vertical:
                self.v_lines.append(line)
    
    def _merge_double_lines(self):
        """Merge double lines"""
        # Merge horizontal lines
        self.h_lines = self._merge_parallel_lines(self.h_lines, is_horizontal=True)
        # Merge vertical lines
        self.v_lines = self._merge_parallel_lines(self.v_lines, is_horizontal=False)
    
    def _merge_parallel_lines(self, lines: List[LineInfo], is_horizontal: bool) -> List[LineInfo]:
        """Merge parallel double lines"""
        if len(lines) < 2:
            return lines
        
        merged = []
        used = set()
        
        # Sort by position
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
                
                # Check if double line
                if self._is_double_line(line1, line2, is_horizontal):
                    # Merge two lines (middle position, maximum range)
                    merged_line = self._merge_two_lines(merged_line, line2, is_horizontal)
                    used.add(j)
            
            merged.append(merged_line)
            used.add(i)
        
        return merged
    
    def _is_double_line(self, line1: LineInfo, line2: LineInfo, is_horizontal: bool) -> bool:
        """Determine if two lines form a double line"""
        if is_horizontal:
            # Double line if Y coordinate difference is small and X ranges overlap
            y_gap = abs(line1.y0 - line2.y0)
            if y_gap > self.DOUBLE_LINE_GAP:
                return False
            
            # Check X range overlap
            x_overlap = min(line1.x1, line2.x1) - max(line1.x0, line2.x0)
            min_length = min(self._get_line_length(line1), self._get_line_length(line2))
            return x_overlap > min_length * 0.5
        else:
            # Double line if X coordinate difference is small and Y ranges overlap
            x_gap = abs(line1.x0 - line2.x0)
            if x_gap > self.DOUBLE_LINE_GAP:
                return False
            
            # Check Y range overlap
            y_overlap = min(line1.y1, line2.y1) - max(line1.y0, line2.y0)
            min_length = min(self._get_line_length(line1), self._get_line_length(line2))
            return y_overlap > min_length * 0.5
    
    def _get_line_length(self, line: LineInfo) -> float:
        """Calculate line length"""
        return math.sqrt((line.x1 - line.x0) ** 2 + (line.y1 - line.y0) ** 2)
    
    def _merge_two_lines(self, line1: LineInfo, line2: LineInfo, is_horizontal: bool) -> LineInfo:
        """Merge two lines"""
        if is_horizontal:
            # Middle Y, maximum X range
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
            # Middle X, maximum Y range
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
        Build grid from lines
        
        Reconstructs incomplete borders and returns grid structure.
        
        Args:
            tolerance: Position clustering tolerance
            
        Returns:
            GridInfo or None
        """
        if tolerance is None:
            tolerance = self.LINE_MERGE_TOLERANCE
            
        if not self.h_lines and not self.v_lines:
            return None
        
        # Collect Y coordinates (horizontal lines)
        h_positions = self._cluster_positions(
            [line.y0 for line in self.h_lines],
            tolerance
        )
        
        # Collect X coordinates (vertical lines)
        v_positions = self._cluster_positions(
            [line.x0 for line in self.v_lines],
            tolerance
        )
        
        if len(h_positions) < 2 or len(v_positions) < 2:
            return None
        
        # Calculate bbox
        x0 = min(v_positions)
        y0 = min(h_positions)
        x1 = max(v_positions)
        y1 = max(h_positions)
        
        # Check border completeness
        is_complete = self._check_border_completeness(h_positions, v_positions)
        
        return GridInfo(
            h_lines=sorted(h_positions),
            v_lines=sorted(v_positions),
            bbox=(x0, y0, x1, y1),
            is_complete=is_complete,
            reconstructed=False
        )
    
    def _cluster_positions(self, positions: List[float], tolerance: float) -> List[float]:
        """Cluster similar positions"""
        if not positions:
            return []
        
        sorted_pos = sorted(positions)
        clusters = [[sorted_pos[0]]]
        
        for pos in sorted_pos[1:]:
            if pos - clusters[-1][-1] <= tolerance:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        # Return the mean value of each cluster
        return [sum(c) / len(c) for c in clusters]
    
    def _check_border_completeness(self, h_positions: List[float], v_positions: List[float]) -> bool:
        """Check border completeness"""
        if len(h_positions) < 2 or len(v_positions) < 2:
            return False
        
        y_min, y_max = min(h_positions), max(h_positions)
        x_min, x_max = min(v_positions), max(v_positions)
        
        # Check if there are enough horizontal lines at top/bottom
        has_top = any(line.y0 <= y_min + self.LINE_MERGE_TOLERANCE for line in self.h_lines)
        has_bottom = any(line.y0 >= y_max - self.LINE_MERGE_TOLERANCE for line in self.h_lines)
        
        # Check if there are enough vertical lines at left/right
        has_left = any(line.x0 <= x_min + self.LINE_MERGE_TOLERANCE for line in self.v_lines)
        has_right = any(line.x0 >= x_max - self.LINE_MERGE_TOLERANCE for line in self.v_lines)
        
        return all([has_top, has_bottom, has_left, has_right])
    
    def reconstruct_incomplete_border(self, grid: GridInfo) -> GridInfo:
        """
        Reconstruct incomplete border
        
        Completes to 4 sides if 3 or more sides exist.
        
        Args:
            grid: Existing GridInfo
            
        Returns:
            Reconstructed GridInfo
        """
        if grid.is_complete:
            return grid
        
        h_lines = list(grid.h_lines)
        v_lines = list(grid.v_lines)
        
        y_min, y_max = min(h_lines), max(h_lines)
        x_min, x_max = min(v_lines), max(v_lines)
        
        reconstructed = False
        
        # Check/add top horizontal line
        has_top = any(abs(y - y_min) < self.LINE_MERGE_TOLERANCE for y in h_lines)
        if not has_top and len(h_lines) >= 2:
            # Estimate top border
            h_lines.insert(0, y_min - self.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        
        # Check/add bottom horizontal line
        has_bottom = any(abs(y - y_max) < self.LINE_MERGE_TOLERANCE for y in h_lines)
        if not has_bottom and len(h_lines) >= 2:
            h_lines.append(y_max + self.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        
        # Check/add left vertical line
        has_left = any(abs(x - x_min) < self.LINE_MERGE_TOLERANCE for x in v_lines)
        if not has_left and len(v_lines) >= 2:
            v_lines.insert(0, x_min - self.BORDER_EXTENSION_MARGIN)
            reconstructed = True
        
        # Check/add right vertical line
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
