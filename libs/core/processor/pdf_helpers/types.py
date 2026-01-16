"""
PDF Handler Types and Configuration

Defines all data classes and configuration values used by the PDF engine.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any


# ============================================================================
# Enums
# ============================================================================

class LineThickness(Enum):
    """Line thickness classification"""
    THIN = auto()      # Table inner lines (0.3-0.5pt)
    NORMAL = auto()    # Regular borders (0.5-1.5pt)
    THICK = auto()     # Emphasis/header divider lines (1.5pt+)


class TableDetectionStrategy(Enum):
    """Table detection strategy"""
    PYMUPDF_NATIVE = auto()        # PyMuPDF built-in table detection
    PDFPLUMBER_LINES = auto()      # pdfplumber line-based detection
    HYBRID_ANALYSIS = auto()       # Line analysis-based hybrid
    BORDERLESS_HEURISTIC = auto()  # Borderless table heuristic


class ElementType(Enum):
    """Page element type"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    ANNOTATION = "annotation"


# ============================================================================
# Configuration Constants
# ============================================================================

class PDFConfig:
    """PDF engine configuration constants"""
    
    # Line thickness thresholds (pt)
    THIN_LINE_MAX = 0.5
    NORMAL_LINE_MAX = 1.5
    
    # Table detection settings
    MIN_TABLE_ROWS = 2
    MIN_TABLE_COLS = 2
    TABLE_MERGE_TOLERANCE = 5.0  # Table merge tolerance (pt)
    
    # Double line merge settings
    DOUBLE_LINE_TOLERANCE = 3.0  # Double line detection distance (pt)
    
    # Cell analysis settings
    CELL_PADDING = 2.0
    MIN_CELL_WIDTH = 10.0
    MIN_CELL_HEIGHT = 8.0
    
    # Text extraction settings
    TEXT_BLOCK_TOLERANCE = 3.0
    
    # Confidence threshold
    CONFIDENCE_THRESHOLD = 0.5
    
    # Page border detection settings
    BORDER_MARGIN = 30.0        # Maximum distance from page edge
    BORDER_LENGTH_RATIO = 0.8   # Minimum border length ratio relative to page size
    
    # Graphic region detection settings
    GRAPHIC_CURVE_RATIO_THRESHOLD = 0.3   # Curve ratio threshold
    GRAPHIC_MIN_CURVE_COUNT = 10          # Minimum curve count
    GRAPHIC_FILL_RATIO_THRESHOLD = 0.2    # Fill ratio threshold
    GRAPHIC_COLOR_VARIETY_THRESHOLD = 3   # Color variety threshold
    
    # Table quality validation settings
    TABLE_MIN_FILLED_CELL_RATIO = 0.15    # Minimum filled cell ratio
    TABLE_MAX_EMPTY_ROW_RATIO = 0.7       # Maximum empty row ratio
    TABLE_MIN_MEANINGFUL_CELLS = 2        # Minimum meaningful cell count
    TABLE_MIN_VALID_ROWS = 2              # Minimum valid row count
    TABLE_MIN_TEXT_DENSITY = 0.005        # Minimum text density
    
    # Cell text length settings
    TABLE_MAX_CELL_TEXT_LENGTH = 300      # Maximum text length per cell
    TABLE_EXTREME_CELL_LENGTH = 800       # Extremely long cell threshold
    TABLE_MAX_LONG_CELLS_RATIO = 0.4      # Maximum long cell ratio


# ============================================================================
# Data Classes - Basic Types
# ============================================================================

@dataclass
class LineInfo:
    """Line information"""
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
        """Line length"""
        import math
        return math.sqrt((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2)
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        """Midpoint"""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


@dataclass 
class GridInfo:
    """Grid information"""
    h_lines: List[float] = field(default_factory=list)  # Y coordinates
    v_lines: List[float] = field(default_factory=list)  # X coordinates
    cells: List['CellInfo'] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    is_complete: bool = False  # Whether border is complete
    reconstructed: bool = False  # Whether border was reconstructed
    
    @property
    def row_count(self) -> int:
        """Row count (number of regions between horizontal lines)"""
        return max(0, len(self.h_lines) - 1)
    
    @property
    def col_count(self) -> int:
        """Column count (number of regions between vertical lines)"""
        return max(0, len(self.v_lines) - 1)


@dataclass
class CellInfo:
    """Cell information"""
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
    """Annotation information"""
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
    Vector text (Outlined/Path Text) region information
    """
    bbox: Tuple[float, float, float, float]
    drawing_count: int              # Number of drawings contained
    curve_count: int                # Curve count (c items)
    fill_count: int                 # Filled path count
    ocr_text: str = ""              # OCR result
    confidence: float = 0.0         # Confidence score
    is_vector_text: bool = False    # Whether this is vector text


# ============================================================================
# Data Classes - Graphic Region
# ============================================================================

@dataclass
class GraphicRegionInfo:
    """
    Graphic region information (charts, diagrams, icons, etc.)
    """
    bbox: Tuple[float, float, float, float]
    curve_count: int = 0            # Curve count
    line_count: int = 0             # Straight line count
    rect_count: int = 0             # Rectangle count
    fill_count: int = 0             # Filled shape count
    color_count: int = 0            # Number of colors used
    is_graphic: bool = False        # Whether this is a graphic region
    confidence: float = 0.0         # Confidence score
    reason: str = ""                # Reasoning for determination


# ============================================================================
# Data Classes - Table Detection
# ============================================================================

@dataclass
class TableCandidate:
    """Table candidate"""
    strategy: TableDetectionStrategy
    confidence: float
    bbox: Tuple[float, float, float, float]
    grid: Optional[GridInfo] = None
    cells: List['CellInfo'] = field(default_factory=list)
    data: List[List[Optional[str]]] = field(default_factory=list)
    raw_table: Any = None  # Original table object
    
    @property
    def row_count(self) -> int:
        """Row count"""
        return len(self.data)
    
    @property
    def col_count(self) -> int:
        """Column count"""
        return max(len(row) for row in self.data) if self.data else 0


@dataclass
class PageElement:
    """Page element"""
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
    """Page border information"""
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
    'PDFConfig',
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
