"""
Complexity Analyzer for PDF Handler

Analyzes page and region complexity to determine processing strategy.

Processing Strategy Based on Complexity Score:
1. Determine processing strategy based on complexity score
2. Complex regions use block image + OCR
3. Simple regions use standard text extraction

Complexity Criteria:
- Drawing density (curves, lines, fill count)
- Image density
- Text quality (broken text ratio)
- Layout complexity (multi-column)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto

import fitz

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class ComplexityLevel(Enum):
    """Complexity level"""
    SIMPLE = auto()        # Simple text - standard extraction
    MODERATE = auto()      # Moderate complexity - extraction + quality validation
    COMPLEX = auto()       # Complex - block image recommended
    EXTREME = auto()       # Extremely complex - full page image recommended


class ProcessingStrategy(Enum):
    """Processing strategy"""
    TEXT_EXTRACTION = auto()       # Standard text extraction
    HYBRID = auto()                # Text + partial OCR
    BLOCK_IMAGE_OCR = auto()       # Block image + OCR
    FULL_PAGE_OCR = auto()         # Full page OCR


@dataclass
class RegionComplexity:
    """Region complexity information"""
    bbox: Tuple[float, float, float, float]
    complexity_level: ComplexityLevel
    complexity_score: float  # 0.0 ~ 1.0
    
    # Detail scores
    drawing_density: float = 0.0
    image_density: float = 0.0
    text_quality: float = 1.0  # 1.0 = perfect, 0.0 = completely broken
    layout_complexity: float = 0.0
    
    # Recommended strategy
    recommended_strategy: ProcessingStrategy = ProcessingStrategy.TEXT_EXTRACTION
    
    # Detailed information
    reasons: List[str] = field(default_factory=list)


@dataclass
class PageComplexity:
    """Full page complexity information"""
    page_num: int
    page_size: Tuple[float, float]
    
    # Overall complexity
    overall_complexity: ComplexityLevel
    overall_score: float
    
    # Region-wise complexity
    regions: List[RegionComplexity] = field(default_factory=list)
    
    # Complex regions
    complex_regions: List[Tuple[float, float, float, float]] = field(default_factory=list)
    
    # Statistics
    total_drawings: int = 0
    total_images: int = 0
    total_text_blocks: int = 0
    column_count: int = 1
    
    # Recommended strategy
    recommended_strategy: ProcessingStrategy = ProcessingStrategy.TEXT_EXTRACTION


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ComplexityConfig:
    """Complexity analysis configuration"""
    # Drawing density threshold (per 1000pt² area)
    DRAWING_DENSITY_MODERATE = 0.5
    DRAWING_DENSITY_COMPLEX = 2.0
    DRAWING_DENSITY_EXTREME = 5.0
    
    # Image density threshold
    IMAGE_DENSITY_MODERATE = 0.1
    IMAGE_DENSITY_COMPLEX = 0.3
    IMAGE_DENSITY_EXTREME = 0.5
    
    # Text quality threshold
    TEXT_QUALITY_POOR = 0.7
    TEXT_QUALITY_BAD = 0.5
    
    # Layout complexity (multi-column)
    # Raised threshold - multi-column does not automatically mean EXTREME
    COLUMN_COUNT_MODERATE = 3   # 3+ columns = MODERATE
    COLUMN_COUNT_COMPLEX = 5    # 5+ columns = COMPLEX (newspaper-level)
    COLUMN_COUNT_EXTREME = 7    # 7+ columns = EXTREME (very complex newspaper)
    
    # Overall complexity threshold
    # Raised EXTREME threshold - avoid going to FULL_PAGE_OCR too easily
    COMPLEXITY_MODERATE = 0.35
    COMPLEXITY_COMPLEX = 0.65
    COMPLEXITY_EXTREME = 0.90  # Raised from 0.8 to 0.90
    
    # Region division settings
    REGION_GRID_SIZE = 200  # pt - analysis grid size
    MIN_REGION_SIZE = 100   # pt - minimum region size
    
    # Table quality analysis enabled
    ANALYZE_TABLE_QUALITY = True  # Enable table quality analysis
    TABLE_QUALITY_THRESHOLD = 0.65  # Attempt table extraction if above this


# ============================================================================
# Complexity Analyzer
# ============================================================================

class ComplexityAnalyzer:
    """
    Page complexity analyzer.
    
    Analyzes page complexity to determine the optimal processing strategy.
    """
    
    def __init__(self, page, page_num: int, config: Optional[ComplexityConfig] = None):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            config: Analysis configuration
        """
        self.page = page
        self.page_num = page_num
        self.config = config or ComplexityConfig()
        
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.page_area = self.page_width * self.page_height
        
        # Cache
        self._drawings = None
        self._text_dict = None
        self._images = None
    
    def analyze(self) -> PageComplexity:
        """
        Analyzes page complexity.
        
        Returns:
            PageComplexity object
        """
        # Collect base data
        drawings = self._get_drawings()
        text_dict = self._get_text_dict()
        images = self._get_images()
        
        text_blocks = [b for b in text_dict.get("blocks", []) if b.get("type") == 0]
        
        # 1. Overall statistics
        total_drawings = len(drawings)
        total_images = len(images)
        total_text_blocks = len(text_blocks)
        
        # 2. Analyze column count
        column_count = self._analyze_columns(text_blocks)
        
        # 3. Drawing complexity
        drawing_complexity = self._analyze_drawing_complexity(drawings)
        
        # 4. Image complexity
        image_complexity = self._analyze_image_complexity(images)
        
        # 5. Text quality
        text_quality = self._analyze_text_quality(text_blocks)
        
        # 6. Layout complexity
        layout_complexity = self._analyze_layout_complexity(column_count, text_blocks)
        
        # 7. Calculate overall complexity score
        overall_score = self._calculate_overall_score(
            drawing_complexity, image_complexity, text_quality, layout_complexity
        )
        
        # 8. Determine complexity level
        overall_complexity = self._determine_complexity_level(overall_score)
        
        # 9. Region-wise analysis
        regions = self._analyze_regions(drawings, text_blocks, images)
        
        # 10. Identify complex regions
        complex_regions = [
            r.bbox for r in regions 
            if r.complexity_level in (ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME)
        ]
        
        # 11. Determine processing strategy
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
        """Cached drawings retrieval"""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings
    
    def _get_text_dict(self) -> Dict:
        """Cached text dictionary retrieval"""
        if self._text_dict is None:
            self._text_dict = self.page.get_text("dict", sort=True)
        return self._text_dict
    
    def _get_images(self) -> List:
        """Cached images retrieval"""
        if self._images is None:
            self._images = self.page.get_images()
        return self._images
    
    def _analyze_columns(self, text_blocks: List[Dict]) -> int:
        """Analyze column count"""
        if not text_blocks:
            return 1
        
        x_positions = []
        for block in text_blocks:
            bbox = block.get("bbox", (0, 0, 0, 0))
            x_positions.append(bbox[0])
        
        if not x_positions:
            return 1
        
        x_positions.sort()
        
        # Clustering
        columns = []
        current_column = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_column[-1] < 50:  # Within 50pt means same column
                current_column.append(x)
            else:
                columns.append(current_column)
                current_column = [x]
        columns.append(current_column)
        
        return len(columns)
    
    def _analyze_drawing_complexity(self, drawings: List[Dict]) -> float:
        """Analyze drawing complexity (0.0 ~ 1.0)"""
        if not drawings:
            return 0.0
        
        # Count items
        total_items = 0
        curve_count = 0
        fill_count = 0
        
        for d in drawings:
            items = d.get("items", [])
            total_items += len(items)
            
            for item in items:
                if item[0] == 'c':  # Curve
                    curve_count += 1
            
            if d.get("fill"):
                fill_count += 1
        
        # Calculate density (per 1000pt²)
        density = total_items / (self.page_area / 1000) if self.page_area > 0 else 0
        
        # Curve ratio (chart/graph indicator)
        curve_ratio = curve_count / max(1, total_items)
        
        # Fill ratio (color complexity)
        fill_ratio = fill_count / max(1, len(drawings))
        
        # Calculate complexity score
        score = 0.0
        
        if density >= self.config.DRAWING_DENSITY_EXTREME:
            score = 1.0
        elif density >= self.config.DRAWING_DENSITY_COMPLEX:
            score = 0.7
        elif density >= self.config.DRAWING_DENSITY_MODERATE:
            score = 0.4
        else:
            score = density / self.config.DRAWING_DENSITY_MODERATE * 0.4
        
        # Add points for curves and fills
        score += curve_ratio * 0.2
        score += fill_ratio * 0.1
        
        return min(1.0, score)
    
    def _analyze_image_complexity(self, images: List) -> float:
        """Analyze image complexity (0.0 ~ 1.0)"""
        if not images:
            return 0.0
        
        # Image density (relative to page size)
        density = len(images) / (self.page_area / 10000)  # Per 100x100pt
        
        if density >= self.config.IMAGE_DENSITY_EXTREME:
            return 1.0
        elif density >= self.config.IMAGE_DENSITY_COMPLEX:
            return 0.7
        elif density >= self.config.IMAGE_DENSITY_MODERATE:
            return 0.4
        else:
            return density / self.config.IMAGE_DENSITY_MODERATE * 0.4
    
    def _analyze_text_quality(self, text_blocks: List[Dict]) -> float:
        """Analyze text quality (0.0 = poor, 1.0 = good)"""
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
                        # PUA (Private Use Area) characters
                        if 0xE000 <= code <= 0xF8FF:
                            bad_chars += 1
                        # Strange symbols
                        elif code in range(0x2400, 0x2500):  # Control Pictures
                            bad_chars += 1
        
        if total_chars == 0:
            return 1.0
        
        return 1.0 - (bad_chars / total_chars)
    
    def _analyze_layout_complexity(self, column_count: int, text_blocks: List[Dict]) -> float:
        """Analyze layout complexity (0.0 ~ 1.0).
        
        Does not automatically assign high score for multi-column layouts.
        TEXT_EXTRACTION may be more efficient when tables can be processed.
        """
        score = 0.0
        
        # Column count based - relaxed threshold
        if column_count >= getattr(self.config, 'COLUMN_COUNT_EXTREME', 7):
            # 7+ columns = very complex newspaper layout
            score = 0.95
            logger.info(f"[ComplexityAnalyzer] Page {self.page_num + 1}: "
                       f"Detected very complex layout ({column_count} columns) → HIGH")
        elif column_count >= self.config.COLUMN_COUNT_COMPLEX:
            # 5-6 columns = newspaper-level layout, but may be table-processable
            score = 0.75
            logger.info(f"[ComplexityAnalyzer] Page {self.page_num + 1}: "
                       f"Detected multi-column layout ({column_count} columns) → COMPLEX")
        elif column_count >= self.config.COLUMN_COUNT_MODERATE:
            # 3-4 columns = moderate complexity
            score = 0.5
        elif column_count >= 2:
            # 2 columns = low complexity
            score = 0.3
        
        # Text block distribution analysis - multiple blocks at same Y indicates multi-column
        if text_blocks:
            y_positions = [b.get("bbox", (0,0,0,0))[1] for b in text_blocks]
            unique_y = len(set(int(y/10) for y in y_positions))
            
            if unique_y < len(text_blocks) * 0.5 and len(text_blocks) > 5:
                # Multiple blocks on same Y line = additional evidence of multi-column layout
                score = max(score, 0.6)
        
        return min(1.0, score)
    
    def _calculate_overall_score(
        self, 
        drawing: float, 
        image: float, 
        text_quality: float, 
        layout: float
    ) -> float:
        """Calculate overall complexity score.
        
        Does not determine EXTREME based on layout complexity alone.
        TEXT_EXTRACTION is more efficient when tables can be processed.
        """
        # Extremely complex layout (7+ columns) gets high score
        if layout >= 0.95:
            return 0.9  # Limited to 0.9 (other factors needed for EXTREME)
        
        # Standard weighted calculation
        # Layout weight reduced (0.35 → 0.25)
        w_drawing = 0.30
        w_image = 0.20
        w_text = 0.25
        w_layout = 0.25
        
        # Text quality is inverse (lower = more complex)
        text_complexity = 1.0 - text_quality
        
        score = (
            drawing * w_drawing +
            image * w_image +
            text_complexity * w_text +
            layout * w_layout
        )
        
        return min(1.0, score)
    
    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level"""
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
        """Analyze complexity by region"""
        regions = []
        grid_size = self.config.REGION_GRID_SIZE
        
        # Grid-based analysis
        for y in range(0, int(self.page_height), grid_size):
            for x in range(0, int(self.page_width), grid_size):
                x0, y0 = x, y
                x1 = min(x + grid_size, self.page_width)
                y1 = min(y + grid_size, self.page_height)
                
                bbox = (x0, y0, x1, y1)
                
                # Number of drawings in region
                region_drawings = [
                    d for d in drawings 
                    if d.get("rect") and self._bbox_overlaps(bbox, tuple(d["rect"]))
                ]
                
                # Number of text blocks in region
                region_texts = [
                    b for b in text_blocks
                    if self._bbox_overlaps(bbox, b.get("bbox", (0,0,0,0)))
                ]
                
                # Calculate region complexity
                area = (x1 - x0) * (y1 - y0)
                drawing_density = len(region_drawings) / (area / 1000) if area > 0 else 0
                
                # Text quality
                text_quality = self._analyze_text_quality(region_texts)
                
                # Complexity score
                region_score = min(1.0, drawing_density / 3.0 + (1.0 - text_quality) * 0.5)
                
                # Determine level
                if region_score >= 0.7:
                    level = ComplexityLevel.COMPLEX
                elif region_score >= 0.4:
                    level = ComplexityLevel.MODERATE
                else:
                    level = ComplexityLevel.SIMPLE
                
                # Determine strategy
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
        """Determine processing strategy.
        
        Recommends TEXT_EXTRACTION even for multi-column layouts if table processing is possible.
        Text extraction is more efficient than image conversion when table quality is good.
        """
        # 1. Full page image conversion if text quality is very low
        if text_quality < 0.4:
            logger.info(f"[ComplexityAnalyzer] Page {self.page_num + 1}: "
                       f"Very low text quality ({text_quality:.2f}) → FULL_PAGE_OCR")
            return ProcessingStrategy.FULL_PAGE_OCR
        
        # 2. Full page image conversion if extremely complex (score >= 0.90) and low text quality
        if complexity == ComplexityLevel.EXTREME and text_quality < 0.6:
            return ProcessingStrategy.FULL_PAGE_OCR
        
        # 3. Full page image conversion if complex regions are 50%+ and text quality is low
        if len(complex_regions) > 0:
            complex_area = sum(
                (r[2] - r[0]) * (r[3] - r[1]) for r in complex_regions
            )
            if complex_area / self.page_area > 0.5 and text_quality < 0.7:
                return ProcessingStrategy.FULL_PAGE_OCR
        
        # 4. Try HYBRID processing even for COMPLEX level
        #    (Determine table/text processability per block)
        if complexity == ComplexityLevel.COMPLEX:
            return ProcessingStrategy.HYBRID  # HYBRID instead of FULL_PAGE_OCR
        
        # 5. Hybrid for moderate complexity
        if complexity == ComplexityLevel.MODERATE:
            return ProcessingStrategy.HYBRID
        
        # 6. Text extraction for simple
        return ProcessingStrategy.TEXT_EXTRACTION
    
    def _bbox_overlaps(self, bbox1: Tuple, bbox2: Tuple) -> bool:
        """Check if two bboxes overlap"""
        return not (
            bbox1[2] <= bbox2[0] or  # bbox1 is left of bbox2
            bbox1[0] >= bbox2[2] or  # bbox1 is right of bbox2
            bbox1[3] <= bbox2[1] or  # bbox1 is above bbox2
            bbox1[1] >= bbox2[3]     # bbox1 is below bbox2
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
