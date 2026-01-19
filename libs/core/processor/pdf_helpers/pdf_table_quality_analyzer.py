"""
Table Quality Analyzer for PDF Handler

Analyzes table quality to determine whether text extraction is feasible.

=============================================================================
Core Concepts:
=============================================================================
Processing all tables as images is inefficient.
Normal tables (with complete borders and regular grids) should be extracted as text.

Evaluation Criteria:
1. Border Completeness - Is the table fully enclosed on all sides?
2. Grid Regularity - Is it composed of orthogonal horizontal/vertical lines?
3. Cell Structure - Are cells in regular rectangular shapes?
4. Absence of Complex Elements - No curves, diagonals, or complex graphics?

=============================================================================
Table Quality Grades:
=============================================================================
- EXCELLENT: Perfect table → Must use text extraction
- GOOD: Good table → Text extraction recommended
- MODERATE: Table with minor issues → Attempt text extraction, use image if it fails
- POOR: Table with major issues → Image conversion recommended
- UNPROCESSABLE: Cannot process → Must use image conversion
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import Enum, auto

import fitz

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class TableQuality(Enum):
    """Table quality grades"""
    EXCELLENT = auto()      # Perfect table - must use text extraction
    GOOD = auto()           # Good table - text extraction recommended
    MODERATE = auto()       # Medium - try and evaluate
    POOR = auto()           # Has issues - image conversion recommended
    UNPROCESSABLE = auto()  # Cannot process - must use image conversion


class BlockProcessability(Enum):
    """Block processability"""
    TEXT_EXTRACTABLE = auto()      # Text extraction possible
    TABLE_EXTRACTABLE = auto()     # Table extraction possible
    NEEDS_OCR = auto()             # OCR required
    IMAGE_REQUIRED = auto()        # Image conversion required


@dataclass
class TableQualityResult:
    """Table quality analysis result"""
    bbox: Tuple[float, float, float, float]
    quality: TableQuality
    score: float  # 0.0 ~ 1.0 (higher is better)
    
    # Detailed scores
    border_completeness: float = 1.0  # Border completeness
    grid_regularity: float = 1.0      # Grid regularity
    cell_structure: float = 1.0       # Cell structure quality
    no_complex_elements: float = 1.0  # Absence of complex elements
    
    # Recommended action
    recommended_action: BlockProcessability = BlockProcessability.TABLE_EXTRACTABLE
    
    # Issues
    issues: List[str] = field(default_factory=list)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TableQualityConfig:
    """Table quality analysis configuration"""
    # Border completeness
    BORDER_REQUIRED_SIDES: int = 4     # Minimum sides for a complete table
    BORDER_TOLERANCE: float = 5.0      # Border alignment tolerance (pt)
    
    # Grid regularity
    LINE_ANGLE_TOLERANCE: float = 2.0  # Horizontal/vertical angle tolerance (degrees)
    GRID_ALIGNMENT_TOLERANCE: float = 3.0  # Grid alignment tolerance (pt)
    MIN_ORTHOGONAL_RATIO: float = 0.9  # Minimum orthogonal line ratio (90%+ for normal table)
    
    # Cell structure
    MIN_CELL_SIZE: float = 10.0        # Minimum cell size (pt)
    MAX_CELL_ASPECT_RATIO: float = 20.0  # Maximum cell aspect ratio
    
    # Complex elements
    MAX_CURVE_RATIO: float = 0.05      # Curve ratio threshold (5% or less)
    MAX_DIAGONAL_RATIO: float = 0.05   # Diagonal line ratio threshold
    
    # Quality grade thresholds
    QUALITY_EXCELLENT: float = 0.95    # EXCELLENT threshold
    QUALITY_GOOD: float = 0.85         # GOOD threshold
    QUALITY_MODERATE: float = 0.65     # MODERATE threshold
    QUALITY_POOR: float = 0.40         # POOR threshold (below = UNPROCESSABLE)


# ============================================================================
# Table Quality Analyzer
# ============================================================================

class TableQualityAnalyzer:
    """
    Table Quality Analyzer
    
    Analyzes table regions to determine whether text extraction is feasible.
    """
    
    def __init__(
        self, 
        page, 
        page_num: int = 0,
        config: Optional[TableQualityConfig] = None
    ):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed), default 0
            config: Analysis configuration
        """
        self.page = page
        self.page_num = page_num
        self.config = config or TableQualityConfig()
        
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        
        # Cache
        self._drawings = None
        self._text_dict = None
    
    def analyze_table(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> TableQualityResult:
        """
        Analyzes the quality of a table region.
        
        Args:
            bbox: Table region bounding box
            
        Returns:
            TableQualityResult object
        """
        drawings = self._get_region_drawings(bbox)
        
        issues = []
        
        # 1. Analyze border completeness
        border_score, border_issues = self._analyze_border_completeness(bbox, drawings)
        issues.extend(border_issues)
        
        # 2. Analyze grid regularity
        grid_score, grid_issues = self._analyze_grid_regularity(bbox, drawings)
        issues.extend(grid_issues)
        
        # 3. Analyze cell structure
        cell_score, cell_issues = self._analyze_cell_structure(bbox, drawings)
        issues.extend(cell_issues)
        
        # 4. Analyze complex elements
        simple_score, simple_issues = self._analyze_element_simplicity(bbox, drawings)
        issues.extend(simple_issues)
        
        # Calculate total score (weighted average)
        total_score = (
            border_score * 0.30 +   # Border completeness 30%
            grid_score * 0.30 +     # Grid regularity 30%
            cell_score * 0.20 +     # Cell structure 20%
            simple_score * 0.20     # Element simplicity 20%
        )
        
        # Determine quality grade
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
        """Extract drawings within the region"""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        
        result = []
        for d in self._drawings:
            rect = d.get("rect")
            if rect and self._bbox_overlaps(bbox, (rect.x0, rect.y0, rect.x1, rect.y1)):
                result.append(d)
        return result
    
    def _get_drawings_cached(self) -> List[Dict]:
        """Return cached drawings for the entire page"""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings
    
    def _get_region_text_blocks(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """Extract text blocks within the region"""
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
        """Analyze if the region is a table"""
        # Extract lines
        lines = self._extract_lines(drawings)
        
        # Minimum lines required for a table
        if len(lines) < 4:  # At least 4 lines (rectangle)
            return False, None
        
        # Separate horizontal and vertical lines
        h_lines = [l for l in lines if l['is_horizontal']]
        v_lines = [l for l in lines if l['is_vertical']]
        
        # Both horizontal and vertical lines must exist for a table
        if len(h_lines) < 2 or len(v_lines) < 2:
            return False, None
        
        # If identified as table, analyze quality
        quality = self.analyze_table(bbox)
        return True, quality
    
    def _analyze_border_completeness(
        self, 
        bbox: Tuple[float, float, float, float],
        drawings: List[Dict]
    ) -> Tuple[float, List[str]]:
        """Analyze border completeness"""
        issues = []
        lines = self._extract_lines(drawings)
        
        if not lines:
            issues.append("No border lines detected")
            return 0.0, issues
        
        # Border detection
        tolerance = self.config.BORDER_TOLERANCE
        x0, y0, x1, y1 = bbox
        
        has_top = False
        has_bottom = False
        has_left = False
        has_right = False
        
        for line in lines:
            if line['is_horizontal']:
                # Top border
                if abs(line['y1'] - y0) <= tolerance and line['x1'] >= x0 and line['x2'] <= x1:
                    has_top = True
                # Bottom border
                elif abs(line['y1'] - y1) <= tolerance and line['x1'] >= x0 and line['x2'] <= x1:
                    has_bottom = True
            
            if line['is_vertical']:
                # Left border
                if abs(line['x1'] - x0) <= tolerance and line['y1'] >= y0 and line['y2'] <= y1:
                    has_left = True
                # Right border
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
        """Analyze grid regularity"""
        issues = []
        lines = self._extract_lines(drawings)
        
        if not lines:
            return 0.0, ["No grid lines"]
        
        # Calculate orthogonal line ratio
        orthogonal_count = sum(1 for l in lines if l['is_horizontal'] or l['is_vertical'])
        total_lines = len(lines)
        
        orthogonal_ratio = orthogonal_count / total_lines if total_lines > 0 else 0
        
        if orthogonal_ratio < self.config.MIN_ORTHOGONAL_RATIO:
            issues.append(f"Non-orthogonal lines: {(1-orthogonal_ratio)*100:.1f}%")
        
        # Analyze line alignment
        h_lines = [l for l in lines if l['is_horizontal']]
        v_lines = [l for l in lines if l['is_vertical']]
        
        # Check Y-coordinate alignment of horizontal lines
        h_alignment = self._check_line_alignment([l['y1'] for l in h_lines])
        # Check X-coordinate alignment of vertical lines
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
        """Analyze cell structure"""
        issues = []
        lines = self._extract_lines(drawings)
        
        h_lines = sorted([l for l in lines if l['is_horizontal']], key=lambda l: l['y1'])
        v_lines = sorted([l for l in lines if l['is_vertical']], key=lambda l: l['x1'])
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            issues.append("Insufficient lines for cell structure")
            return 0.5, issues
        
        # Analyze cell sizes
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
        
        # Check for cells that are too small
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
        
        # Check for extreme aspect ratios
        extreme_ratio_count = 0
        for h in cell_heights:
            for w in cell_widths:
                if h > 0 and w > 0:
                    ratio = max(h/w, w/h)
                    if ratio > self.config.MAX_CELL_ASPECT_RATIO:
                        extreme_ratio_count += 1
        
        if extreme_ratio_count > 0:
            issues.append("Extreme cell aspect ratios")
        
        # Calculate score
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
        """Analyze element simplicity (absence of curves, diagonals, and other complex elements)"""
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
                if item_type == 'c':  # curve
                    curve_count += 1
                elif item_type == 'l':  # line
                    # Check for diagonal
                    p1, p2 = item[1], item[2]
                    if not self._is_orthogonal_line(p1, p2):
                        diagonal_count += 1
            
            if d.get("fill"):
                fill_count += 1
        
        # Calculate ratios
        curve_ratio = curve_count / max(1, total_items)
        diagonal_ratio = diagonal_count / max(1, total_items)
        fill_ratio = fill_count / max(1, len(drawings))
        
        # Detect issues
        if curve_ratio > self.config.MAX_CURVE_RATIO:
            issues.append(f"Too many curves: {curve_ratio*100:.1f}%")
        
        if diagonal_ratio > self.config.MAX_DIAGONAL_RATIO:
            issues.append(f"Too many diagonals: {diagonal_ratio*100:.1f}%")
        
        if fill_ratio > 0.5:
            issues.append("Heavy fill patterns")
        
        # Calculate score
        score = 1.0
        score -= min(0.3, curve_ratio * 3)
        score -= min(0.3, diagonal_ratio * 3)
        score -= min(0.2, fill_ratio * 0.4)
        
        return max(0.0, score), issues
    
    def _extract_lines(self, drawings: List[Dict]) -> List[Dict]:
        """Extract lines from drawings"""
        lines = []
        
        for d in drawings:
            for item in d.get("items", []):
                if item[0] == 'l':  # straight line
                    p1, p2 = item[1], item[2]
                    x1, y1 = p1.x, p1.y
                    x2, y2 = p2.x, p2.y
                    
                    # Determine horizontal/vertical
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
                elif item[0] == 're':  # rectangle
                    rect = item[1]
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                    
                    # Add rectangle's 4 sides as lines
                    lines.extend([
                        {'x1': x0, 'y1': y0, 'x2': x1, 'y2': y0, 'is_horizontal': True, 'is_vertical': False, 'length': x1-x0},  # top
                        {'x1': x0, 'y1': y1, 'x2': x1, 'y2': y1, 'is_horizontal': True, 'is_vertical': False, 'length': x1-x0},  # bottom
                        {'x1': x0, 'y1': y0, 'x2': x0, 'y2': y1, 'is_horizontal': False, 'is_vertical': True, 'length': y1-y0},  # left
                        {'x1': x1, 'y1': y0, 'x2': x1, 'y2': y1, 'is_horizontal': False, 'is_vertical': True, 'length': y1-y0},  # right
                    ])
        
        return lines
    
    def _is_orthogonal_line(self, p1, p2) -> bool:
        """Check if the line is horizontal or vertical"""
        tolerance = self.config.LINE_ANGLE_TOLERANCE
        return abs(p2.x - p1.x) <= tolerance or abs(p2.y - p1.y) <= tolerance
    
    def _check_line_alignment(self, positions: List[float]) -> float:
        """Check line alignment quality"""
        if len(positions) < 2:
            return 1.0
        
        # Clustering
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
        
        # Ratio of well-aligned lines
        well_aligned = sum(len(c) for c in clusters if len(c) > 1)
        return well_aligned / len(positions) if positions else 1.0
    
    def _analyze_text_quality(self, text_blocks: List[Dict]) -> float:
        """Analyze text quality"""
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
        """Check if two bounding boxes overlap"""
        return not (
            bbox1[2] <= bbox2[0] or
            bbox1[0] >= bbox2[2] or
            bbox1[3] <= bbox2[1] or
            bbox1[1] >= bbox2[3]
        )
    
    def analyze_page_tables(self) -> Dict[str, Any]:
        """
        Analyzes all table candidate regions on the page.
        
        Returns:
            Dict containing:
                - table_candidates: List of table candidates (each with quality info)
                - has_processable_tables: Whether processable tables exist
                - summary: Analysis summary
        """
        # Search for table candidate regions from drawings
        drawings = self._get_drawings_cached()
        
        # Extract lines
        h_lines = []
        v_lines = []
        
        for d in drawings:
            items = d.get("items", [])
            for item in items:
                cmd = item[0] if item else None
                
                if cmd == "l":  # line
                    x0, y0, x1, y1 = item[1], item[2], item[3], item[4]
                    
                    if abs(y1 - y0) < 3:  # horizontal line
                        h_lines.append((min(x0, x1), y0, max(x0, x1), y1))
                    elif abs(x1 - x0) < 3:  # vertical line
                        v_lines.append((x0, min(y0, y1), x1, max(y0, y1)))
                        
                elif cmd == "re":  # rect
                    x, y, w, h = item[1], item[2], item[3], item[4]
                    if w > 20 and h > 10:
                        # Add rectangle's four sides as lines
                        h_lines.append((x, y, x + w, y))  # top
                        h_lines.append((x, y + h, x + w, y + h))  # bottom
                        v_lines.append((x, y, x, y + h))  # left
                        v_lines.append((x + w, y, x + w, y + h))  # right
        
        # Find table candidate regions (areas with dense lines)
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
        Search for table candidates in regions where horizontal and vertical lines intersect
        """
        if not h_lines or not v_lines:
            return []
        
        # Calculate bounding box of all lines
        all_lines = h_lines + v_lines
        if not all_lines:
            return []
        
        # Find table regions by clustering lines
        clusters = []
        used = set()
        
        # Simplified approach: group lines that intersect or are close to each other
        tolerance = 50  # pixels
        
        for i, line1 in enumerate(all_lines):
            if i in used:
                continue
            
            cluster = [line1]
            used.add(i)
            
            for j, line2 in enumerate(all_lines):
                if j in used:
                    continue
                
                # If two lines are close, put them in the same cluster
                if self._lines_are_close(line1, line2, tolerance):
                    cluster.append(line2)
                    used.add(j)
            
            if len(cluster) >= 4:  # At least 4 lines required for a table candidate
                clusters.append(cluster)
        
        # Convert clusters to bounding boxes
        table_regions = []
        for cluster in clusters:
            x0 = min(min(l[0], l[2]) for l in cluster)
            y0 = min(min(l[1], l[3]) for l in cluster)
            x1 = max(max(l[0], l[2]) for l in cluster)
            y1 = max(max(l[1], l[3]) for l in cluster)
            
            # Check minimum size
            if (x1 - x0) > 100 and (y1 - y0) > 50:
                table_regions.append((x0, y0, x1, y1))
        
        return table_regions
    
    def _lines_are_close(
        self, 
        line1: Tuple, 
        line2: Tuple, 
        tolerance: float
    ) -> bool:
        """Check if two lines are close to each other"""
        # Check distance between endpoints of line1 and line2
        x1_min, y1_min = min(line1[0], line1[2]), min(line1[1], line1[3])
        x1_max, y1_max = max(line1[0], line1[2]), max(line1[1], line1[3])
        x2_min, y2_min = min(line2[0], line2[2]), min(line2[1], line2[3])
        x2_max, y2_max = max(line2[0], line2[2]), max(line2[1], line2[3])
        
        # True if bounding boxes of the two lines overlap or are close
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
    'TableQualityConfig',
    'TableQualityAnalyzer',
]
