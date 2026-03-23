# contextifier/handlers/pdf_plus/_types.py
"""
PDF Plus — type definitions for the adaptive PDF engine.

All enums, dataclasses, and configuration constants used by the
complexity analyzer, table pipeline, text quality engine, and
block image engine live here.

Mirrors the v1.0 ``pdf_helpers/types.py`` with improvements:
- ``PageElement`` uses a standard ``__lt__`` for sorting
- Configuration values are grouped in ``PdfPlusConfig``
- All numeric magic values are named constants

These types are internal to the ``pdf_plus`` package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# Enums
# ═════════════════════════════════════════════════════════════════════════════


class LineThickness(Enum):
    """Classification of a PDF drawing-line thickness."""
    THIN = auto()       # < 0.5 pt  — inner grid / hairline
    NORMAL = auto()     # 0.5–1.5 pt — regular border
    THICK = auto()      # > 1.5 pt  — emphasis / header divider


class TableDetectionStrategy(Enum):
    """Which strategy found this table candidate."""
    PYMUPDF_NATIVE = auto()        # fitz find_tables()
    PDFPLUMBER_LINES = auto()      # pdfplumber line-based
    HYBRID_ANALYSIS = auto()       # custom line-analysis grid
    BORDERLESS_HEURISTIC = auto()  # text-clustering heuristic


class ElementType(Enum):
    """Type of page element for the position-aware merger."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    ANNOTATION = "annotation"


class ComplexityLevel(Enum):
    """Page-level complexity classification."""
    SIMPLE = auto()      # score < 0.35 → text extraction
    MODERATE = auto()    # 0.35–0.65 → hybrid
    COMPLEX = auto()     # 0.65–0.90 → hybrid (not full OCR)
    EXTREME = auto()     # ≥ 0.90 → full-page OCR (with caveats)


class ProcessingStrategy(Enum):
    """Per-page processing strategy chosen by the complexity analyzer."""
    TEXT_EXTRACTION = auto()    # standard text + table + image
    HYBRID = auto()             # text + block-image for complex regions
    BLOCK_IMAGE_OCR = auto()    # block-image dominant + remaining text
    FULL_PAGE_OCR = auto()      # smart blocks / grid / full-page image


class TableQuality(Enum):
    """Processability grade for a table region."""
    EXCELLENT = auto()      # ≥ 0.95  → must text-extract
    GOOD = auto()           # ≥ 0.85  → recommended text
    MODERATE = auto()       # ≥ 0.65  → attempt text
    POOR = auto()           # ≥ 0.40  → image recommended
    UNPROCESSABLE = auto()  # < 0.40  → must image


class LayoutBlockType(Enum):
    """Semantic block type detected by the layout detector."""
    ARTICLE = auto()
    IMAGE_WITH_CAPTION = auto()
    STANDALONE_IMAGE = auto()
    TABLE = auto()
    ADVERTISEMENT = auto()
    SIDEBAR = auto()
    HEADER = auto()
    FOOTER = auto()
    COLUMN_BLOCK = auto()
    UNKNOWN = auto()


class BlockProcessingStrategy(Enum):
    """Strategy used by the block image engine."""
    SEMANTIC_BLOCKS = auto()    # layout detector → per-block images
    GRID_BLOCKS = auto()        # NxM grid division
    FULL_PAGE = auto()          # single full-page image


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════


class PdfPlusConfig:
    """
    All numeric constants for the pdf_plus engine.

    Centralised so they can be tuned without editing multiple files.
    """

    # ── Line analysis ────────────────────────────────────────────────────
    THIN_LINE_MAX: float = 0.5
    NORMAL_LINE_MAX: float = 1.5
    DOUBLE_LINE_TOLERANCE: float = 3.0      # merge lines within 5 pt
    DOUBLE_LINE_OVERLAP_RATIO: float = 0.5  # positional overlap threshold

    # ── Table detection ──────────────────────────────────────────────────
    MIN_TABLE_ROWS: int = 2
    MIN_TABLE_COLS: int = 2
    TABLE_MERGE_TOLERANCE: float = 5.0
    PYMUPDF_SNAP: float = 7.0
    PYMUPDF_JOIN: float = 7.0
    PYMUPDF_EDGE_MIN: float = 10.0
    PYMUPDF_INTERSECTION: float = 7.0
    PYMUPDF_BASE_CONFIDENCE: float = 0.6
    PDFPLUMBER_BASE_CONFIDENCE: float = 0.4
    LINE_ANALYSIS_BASE_CONFIDENCE: float = 0.3
    LINE_ANALYSIS_MIN_CONFIDENCE: float = 0.65  # when PyMuPDF found nothing
    NARROW_COLUMN_WIDTH: float = 15.0
    NARROW_COLUMN_EMPTY_RATIO: float = 0.9

    # ── Table validator (12-point) ───────────────────────────────────────
    MIN_FILLED_CELL_RATIO: float = 0.15
    MAX_EMPTY_ROW_RATIO: float = 0.7
    MIN_MEANINGFUL_CELLS: int = 2
    MIN_VALID_ROWS: int = 2
    MIN_TEXT_DENSITY: float = 0.005
    MAX_CELL_TEXT_LENGTH: int = 300
    EXTREME_CELL_LENGTH: int = 800
    MAX_LONG_CELLS_RATIO: float = 0.4
    MAX_PARAGRAPH_CELLS_RATIO: float = 0.25
    VALIDATOR_REJECT_THRESHOLD: float = 0.35

    # ── Table quality analyzer ───────────────────────────────────────────
    QUALITY_EXCELLENT: float = 0.95
    QUALITY_GOOD: float = 0.85
    QUALITY_MODERATE: float = 0.65
    QUALITY_POOR: float = 0.40

    QUALITY_WEIGHT_BORDER: float = 0.30
    QUALITY_WEIGHT_GRID: float = 0.30
    QUALITY_WEIGHT_CELL: float = 0.20
    QUALITY_WEIGHT_ELEMENT: float = 0.20

    # ── Table processor ──────────────────────────────────────────────────
    MERGE_Y_GAP: float = 30.0
    MERGE_X_OVERLAP: float = 0.80
    ANNOTATION_Y_MARGIN: float = 30.0
    ANNOTATION_PATTERNS: tuple = (
        "주)", "주 )", "※", "*", "†", "‡", "¹", "²", "³",
    )
    CONTINUITY_TOP_RATIO: float = 0.30
    CONTINUITY_BOTTOM_RATIO: float = 0.70

    # ── Cell analysis ────────────────────────────────────────────────────
    CELL_PADDING: float = 2.0
    MIN_CELL_WIDTH: float = 10.0
    MIN_CELL_HEIGHT: float = 8.0

    # ── Complexity analyzer ──────────────────────────────────────────────
    DRAWING_DENSITY_MODERATE: float = 0.5
    DRAWING_DENSITY_COMPLEX: float = 2.0
    DRAWING_DENSITY_EXTREME: float = 5.0
    IMAGE_DENSITY_MODERATE: float = 0.1
    IMAGE_DENSITY_COMPLEX: float = 0.3
    IMAGE_DENSITY_EXTREME: float = 0.5
    TEXT_QUALITY_POOR: float = 0.7
    TEXT_QUALITY_BAD: float = 0.5
    COLUMN_CLUSTER_TOLERANCE: float = 50.0
    COLUMN_COUNT_MODERATE: int = 3
    COLUMN_COUNT_COMPLEX: int = 5
    COLUMN_COUNT_EXTREME: int = 7
    COMPLEXITY_MODERATE: float = 0.35
    COMPLEXITY_COMPLEX: float = 0.65
    COMPLEXITY_EXTREME: float = 0.90
    REGION_GRID_SIZE: int = 200

    WEIGHT_DRAWING: float = 0.30
    WEIGHT_IMAGE: float = 0.20
    WEIGHT_TEXT: float = 0.25
    WEIGHT_LAYOUT: float = 0.25

    # ── Graphic detector ─────────────────────────────────────────────────
    GRAPHIC_CLUSTER_MARGIN: float = 20.0
    GRAPHIC_CURVE_RATIO: float = 0.3
    GRAPHIC_MIN_CURVES: int = 10
    GRAPHIC_FILL_RATIO: float = 0.2
    GRAPHIC_COLOR_VARIETY: int = 3
    GRAPHIC_MIN_AREA: float = 500.0
    GRAPHIC_THRESHOLD: float = 0.5

    # ── Page border detection ────────────────────────────────────────────
    PAGE_BORDER_MARGIN_RATIO: float = 0.10
    PAGE_SPANNING_RATIO: float = 0.85
    PAGE_BORDER_LINE_MAX_SIZE: float = 10.0

    # ── Block image engine ───────────────────────────────────────────────
    BLOCK_DPI: int = 300
    BLOCK_MAX_IMAGE_SIZE: int = 4096
    BLOCK_MIN_REGION_WIDTH: float = 80.0
    BLOCK_MIN_REGION_HEIGHT: float = 60.0
    BLOCK_MIN_AREA: float = 15000.0
    BLOCK_EMPTY_THRESHOLD: float = 0.95
    BLOCK_EMPTY_PIXEL_MIN: int = 240

    # ── Layout block detector ────────────────────────────────────────────
    LBD_GAP_THRESHOLD: float = 20.0
    LBD_HEADER_FOOTER_RATIO: float = 0.10
    LBD_HEADER_MAX_HEIGHT: float = 60.0
    LBD_VERT_CLUSTER_DIST: float = 40.0
    LBD_HORIZ_CLUSTER_DIST: float = 15.0
    LBD_MIN_BOX_AREA: float = 10000.0
    LBD_MAX_BLOCKS_BEFORE_MERGE: int = 15

    # ── Text quality ─────────────────────────────────────────────────────
    PUA_START: int = 0xE000
    PUA_END: int = 0xF8FF
    PUA_SUPP_START: int = 0xF0000
    PUA_SUPP_END: int = 0xFFFFD
    QUALITY_OCR_THRESHOLD: float = 0.7
    QUALITY_PUA_RATIO_THRESHOLD: float = 0.10

    # ── Vector text OCR ──────────────────────────────────────────────────
    VECTOR_MAX_HEIGHT: float = 50.0
    VECTOR_MIN_ITEMS: int = 5
    VECTOR_WIDTH_RATIO: float = 2.0
    VECTOR_MERGE_Y_TOLERANCE: float = 20.0
    VECTOR_MIN_CONFIDENCE: float = 0.3

    # ── Image extraction ─────────────────────────────────────────────────
    MIN_IMAGE_SIZE: int = 50
    MIN_IMAGE_AREA: int = 2500
    IMAGE_TABLE_OVERLAP: float = 0.7
    IMAGE_DPI: int = 150

    # ── OCR common ───────────────────────────────────────────────────────
    OCR_LANGUAGE: str = "kor+eng"
    BLOCK_IMAGE_DPI: int = 300             # alias for BLOCK_DPI

    # ═════════════════════════════════════════════════════════════════════
    # Aliases — used by modules that reference slightly different names.
    # Kept as class-level aliases so callers need not be updated.
    # ═════════════════════════════════════════════════════════════════════

    # _line_analysis.py
    LINE_MERGE_TOLERANCE: float = 5.0       # ≡ TABLE_MERGE_TOLERANCE
    BORDER_EXTENSION_MARGIN: float = 2.0    # page-border extension buffer
    THIN_LINE_THRESHOLD: float = 0.5        # ≡ THIN_LINE_MAX
    THICK_LINE_THRESHOLD: float = 1.5       # ≡ NORMAL_LINE_MAX
    DOUBLE_LINE_GAP: float = 3.0            # ≡ DOUBLE_LINE_TOLERANCE

    # _graphic_detector.py
    GRAPHIC_CURVE_RATIO_THRESHOLD: float = 0.3   # ≡ GRAPHIC_CURVE_RATIO
    GRAPHIC_MIN_CURVE_COUNT: int = 10             # ≡ GRAPHIC_MIN_CURVES
    GRAPHIC_FILL_RATIO_THRESHOLD: float = 0.2    # ≡ GRAPHIC_FILL_RATIO
    GRAPHIC_COLOR_VARIETY_THRESHOLD: int = 3      # ≡ GRAPHIC_COLOR_VARIETY

    # _table_detection.py
    TABLE_CONFIDENCE_THRESHOLD: float = 0.35      # minimum to accept a table
    PYMUPDF_SNAP_TOLERANCE: float = 7.0           # ≡ PYMUPDF_SNAP
    PYMUPDF_JOIN_TOLERANCE: float = 7.0           # ≡ PYMUPDF_JOIN
    PYMUPDF_EDGE_MIN_LENGTH: float = 10.0         # ≡ PYMUPDF_EDGE_MIN
    PYMUPDF_INTERSECTION_TOLERANCE: float = 7.0   # ≡ PYMUPDF_INTERSECTION
    PYMUPDF_CONFIDENCE_BASE: float = 0.6          # ≡ PYMUPDF_BASE_CONFIDENCE
    PDFPLUMBER_CONFIDENCE_BASE: float = 0.4       # ≡ PDFPLUMBER_BASE_CONFIDENCE
    LINE_CONFIDENCE_BASE: float = 0.3             # ≡ LINE_ANALYSIS_BASE_CONFIDENCE

    # _table_validator.py
    TABLE_MIN_FILLED_CELL_RATIO: float = 0.15     # ≡ MIN_FILLED_CELL_RATIO
    TABLE_MAX_EMPTY_ROW_RATIO: float = 0.7        # ≡ MAX_EMPTY_ROW_RATIO
    TABLE_MIN_MEANINGFUL_CELLS: int = 2            # ≡ MIN_MEANINGFUL_CELLS
    TABLE_MIN_VALID_ROWS: int = 2                  # ≡ MIN_VALID_ROWS
    TABLE_MIN_TEXT_DENSITY: float = 0.005          # ≡ MIN_TEXT_DENSITY
    TABLE_MAX_LONG_CELLS_RATIO: float = 0.4        # ≡ MAX_LONG_CELLS_RATIO
    TABLE_EXTREME_CELL_LENGTH: int = 800           # ≡ EXTREME_CELL_LENGTH
    TABLE_MAX_CELL_TEXT_LENGTH: int = 300           # ≡ MAX_CELL_TEXT_LENGTH

    # _table_quality_analyzer.py
    QUALITY_WEIGHT_SIMPLE: float = 0.20            # ≡ QUALITY_WEIGHT_ELEMENT
    QUALITY_BORDER_TOLERANCE: float = 3.0          # snap tolerance for border scoring
    QUALITY_MIN_ORTHOGONAL_RATIO: float = 0.7      # ratio of orthogonal lines
    QUALITY_MIN_CELL_SIZE: float = 8.0             # minimum cell dimension
    QUALITY_MAX_CELL_ASPECT_RATIO: float = 20.0    # reject excessively narrow cells
    QUALITY_LINE_ANGLE_TOLERANCE: float = 5.0      # degrees from hor/vert
    QUALITY_MAX_CURVE_RATIO: float = 0.3           # max curve-to-total-line ratio
    QUALITY_MAX_DIAGONAL_RATIO: float = 0.2        # max diagonal-to-total ratio
    QUALITY_EXCELLENT_THRESHOLD: float = 0.95      # ≡ QUALITY_EXCELLENT
    QUALITY_GOOD_THRESHOLD: float = 0.85           # ≡ QUALITY_GOOD
    QUALITY_MODERATE_THRESHOLD: float = 0.65       # ≡ QUALITY_MODERATE
    QUALITY_POOR_THRESHOLD: float = 0.40           # ≡ QUALITY_POOR

    # _cell_analysis.py
    CELL_GRID_TOLERANCE: float = 3.0       # snap cells to grid lines
    CELL_OVERLAP_THRESHOLD: float = 0.5    # overlap ratio for cell merging

    # _table_processor.py
    TABLE_CONTINUITY_BOTTOM_RATIO: float = 0.70   # ≡ CONTINUITY_BOTTOM_RATIO
    TABLE_CONTINUITY_TOP_RATIO: float = 0.30       # ≡ CONTINUITY_TOP_RATIO
    TABLE_ANNOTATION_GAP: float = 30.0             # ≡ ANNOTATION_Y_MARGIN

    # _text_quality_analyzer.py
    PUA_RANGES: list = [                   # for tuple-based iteration
        (0xE000, 0xF8FF),
        (0xF0000, 0xFFFFD),
    ]
    OCR_QUALITY_THRESHOLD: float = 0.7     # ≡ QUALITY_OCR_THRESHOLD

    # _vector_text_ocr.py
    VECTOR_TEXT_MAX_GLYPH_AREA: float = 2500.0  # max area for a single glyph path
    VECTOR_TEXT_MIN_GLYPH_CLUSTER: int = 5       # ≡ VECTOR_MIN_ITEMS


# ═════════════════════════════════════════════════════════════════════════════
# Data Classes — basic building blocks
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LineInfo:
    """A single detected line from page drawings."""
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
        return math.sqrt((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


@dataclass
class GridInfo:
    """Grid constructed from horizontal / vertical lines."""
    h_lines: List[float] = field(default_factory=list)   # Y positions
    v_lines: List[float] = field(default_factory=list)   # X positions
    cells: List["CellInfo"] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    is_complete: bool = False
    reconstructed: bool = False

    @property
    def row_count(self) -> int:
        return max(0, len(self.h_lines) - 1)

    @property
    def col_count(self) -> int:
        return max(0, len(self.v_lines) - 1)


@dataclass
class CellInfo:
    """A single table cell with grid position and span info."""
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
    """Annotation (footnote / endnote) associated with a table."""
    text: str = ""
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    annotation_type: str = "annotation"
    color: Optional[Tuple[float, float, float]] = None


@dataclass
class VectorTextRegion:
    """Region containing outlined / vector-path text."""
    bbox: Tuple[float, float, float, float]
    drawing_count: int
    curve_count: int
    fill_count: int
    ocr_text: str = ""
    confidence: float = 0.0
    is_vector_text: bool = False


@dataclass
class GraphicRegionInfo:
    """Detected graphic region (chart, diagram, icon)."""
    bbox: Tuple[float, float, float, float]
    curve_count: int = 0
    line_count: int = 0
    rect_count: int = 0
    fill_count: int = 0
    color_count: int = 0
    is_graphic: bool = False
    confidence: float = 0.0
    reason: str = ""


@dataclass
class TableCandidate:
    """A candidate table found by one of the detection strategies."""
    strategy: TableDetectionStrategy
    confidence: float
    bbox: Tuple[float, float, float, float]
    grid: Optional[GridInfo] = None
    cells: List[CellInfo] = field(default_factory=list)
    data: List[List[Optional[str]]] = field(default_factory=list)
    raw_table: Any = None

    @property
    def row_count(self) -> int:
        return len(self.data)

    @property
    def col_count(self) -> int:
        return max((len(r) for r in self.data), default=0)


@dataclass
class PageElement:
    """
    A typed, positioned element on a page.

    Sortable by ``(page_num, y0, x0)`` for reading-order assembly.
    """
    element_type: ElementType
    content: str
    bbox: Tuple[float, float, float, float]
    page_num: int
    table_data: Optional[List[List]] = None
    cells_info: Optional[List[Dict]] = None
    annotations: Optional[List[AnnotationInfo]] = None
    detection_strategy: Optional[TableDetectionStrategy] = None
    confidence: float = 1.0

    def __lt__(self, other: "PageElement") -> bool:
        return (self.page_num, self.bbox[1], self.bbox[0]) < (
            other.page_num, other.bbox[1], other.bbox[0]
        )


@dataclass
class PageBorderInfo:
    """Whether the page has a decorative border frame."""
    has_border: bool = False
    border_bbox: Optional[Tuple[float, float, float, float]] = None
    border_lines: Dict[str, bool] = field(
        default_factory=lambda: {
            "top": False, "bottom": False, "left": False, "right": False,
        }
    )


# ── Complexity types ─────────────────────────────────────────────────────────


@dataclass
class RegionComplexity:
    """Complexity analysis for a single page region (grid cell)."""
    bbox: Tuple[float, float, float, float]
    complexity_level: ComplexityLevel
    complexity_score: float
    drawing_density: float = 0.0
    image_density: float = 0.0
    text_quality: float = 1.0
    layout_complexity: float = 0.0
    recommended_strategy: ProcessingStrategy = ProcessingStrategy.TEXT_EXTRACTION
    reasons: List[str] = field(default_factory=list)


@dataclass
class PageComplexity:
    """Full-page complexity analysis result."""
    page_num: int
    page_size: Tuple[float, float]
    overall_complexity: ComplexityLevel
    overall_score: float
    regions: List[RegionComplexity] = field(default_factory=list)
    complex_regions: List[Tuple[float, float, float, float]] = field(
        default_factory=list,
    )
    total_drawings: int = 0
    total_images: int = 0
    total_text_blocks: int = 0
    column_count: int = 1
    recommended_strategy: ProcessingStrategy = ProcessingStrategy.TEXT_EXTRACTION


# ── Block image types ────────────────────────────────────────────────────────


@dataclass
class BlockResult:
    """Result of rendering one block as an image."""
    success: bool
    image_tag: Optional[str] = None
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    block_type: str = ""
    error: Optional[str] = None


@dataclass
class MultiBlockResult:
    """Result of rendering multiple blocks (smart processing)."""
    success: bool
    strategy_used: BlockProcessingStrategy = BlockProcessingStrategy.FULL_PAGE
    block_results: List[BlockResult] = field(default_factory=list)
    total_blocks: int = 0
    successful_blocks: int = 0


# ── Table quality types ──────────────────────────────────────────────────────


@dataclass
class TableQualityResult:
    """Assessment of a table region's processability."""
    quality: TableQuality
    score: float
    border_score: float = 0.0
    grid_score: float = 0.0
    cell_score: float = 0.0
    element_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


# ── Layout block types ───────────────────────────────────────────────────────


@dataclass
class LayoutBlock:
    """Semantic layout block detected by the layout detector."""
    block_type: LayoutBlockType
    bbox: Tuple[float, float, float, float]
    elements: List[Dict] = field(default_factory=list)
    confidence: float = 0.5
    column_index: int = 0


@dataclass
class LayoutAnalysisResult:
    """Full layout analysis for a page."""
    blocks: List[LayoutBlock] = field(default_factory=list)
    columns: List[Tuple[float, float]] = field(default_factory=list)
    has_header: bool = False
    has_footer: bool = False


# ── Text quality types ───────────────────────────────────────────────────────


@dataclass
class TextQualityResult:
    """Quality analysis for a text fragment."""
    quality_score: float         # 0.0 – 1.0
    total_chars: int = 0
    pua_chars: int = 0
    garbled_ratio: float = 0.0
    needs_ocr: bool = False
    details: str = "ok"


@dataclass
class PageTextAnalysis:
    """Quality-aware text extraction result for a single page."""
    text: str = ""
    quality: Optional[TextQualityResult] = None
    used_ocr: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Exports
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "LineThickness",
    "TableDetectionStrategy",
    "ElementType",
    "ComplexityLevel",
    "ProcessingStrategy",
    "TableQuality",
    "LayoutBlockType",
    "BlockProcessingStrategy",
    # Config
    "PdfPlusConfig",
    # Data classes
    "LineInfo",
    "GridInfo",
    "CellInfo",
    "AnnotationInfo",
    "VectorTextRegion",
    "GraphicRegionInfo",
    "TableCandidate",
    "PageElement",
    "PageBorderInfo",
    "RegionComplexity",
    "PageComplexity",
    "BlockResult",
    "MultiBlockResult",
    "TableQualityResult",
    "LayoutBlock",
    "LayoutAnalysisResult",
    "TextQualityResult",
    "PageTextAnalysis",
]
