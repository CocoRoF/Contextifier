"""
Layout Block Detector for PDF Handler

Divides complex multi-column layouts (newspapers, magazines, etc.) into semantic block units.

=============================================================================
=============================================================================
Instead of processing the entire page as a single image,
the page is divided into **semantic/logical block units** and saved as individual PNGs.

This enables:
1. LLM can interpret each block **individually**
2. Resolution issue resolved (maintains high resolution per block)
3. Preserves reading order
4. Context separation (distinguishes ads/articles/tables)

=============================================================================
Layout Analysis Algorithm:
=============================================================================

Phase 1: Basic Analysis
┌─────────────────────────────────────────────────────────────────┐
│  1. Extract text blocks                                         │
│  2. Extract image/graphic regions                               │
│  3. Extract drawings (lines, boxes)                             │
│  4. Identify table regions                                      │
└─────────────────────────────────────────────────────────────────┘

Phase 2: Column Detection (Multi-column Layout)
┌─────────────────────────────────────────────────────────────────┐
│  1. X-coordinate based clustering                               │
│  2. Identify column boundaries                                  │
│  3. Group content by column                                     │
└─────────────────────────────────────────────────────────────────┘

Phase 3: Semantic Block Clustering
┌─────────────────────────────────────────────────────────────────┐
│  1. Connect adjacent elements (distance-based)                  │
│  2. Connect headline-body (font size analysis)                  │
│  3. Connect image-caption (positional relationship)             │
│  4. Separate regions based on dividers/boxes                    │
└─────────────────────────────────────────────────────────────────┘

Phase 4: Block Optimization and Ordering
┌─────────────────────────────────────────────────────────────────┐
│  1. Merge small blocks                                          │
│  2. Resolve overlaps                                            │
│  3. Determine reading order (column → top-to-bottom)            │
│  4. Normalize block bboxes                                      │
└─────────────────────────────────────────────────────────────────┘

=============================================================================
Block Types:
=============================================================================
- ARTICLE: Article block (headline + body)
- IMAGE_WITH_CAPTION: Image + caption
- TABLE: Table region
- ADVERTISEMENT: Advertisement region (separated by box)
- SIDEBAR: Sidebar/infobox
- HEADER_FOOTER: Header/footer
- UNKNOWN: Unclassifiable
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto
from collections import defaultdict
import math

import fitz

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class LayoutBlockType(Enum):
    """Layout block type"""
    ARTICLE = auto()            # Article (headline + body)
    IMAGE_WITH_CAPTION = auto() # Image + caption
    STANDALONE_IMAGE = auto()   # Standalone image
    TABLE = auto()              # Table
    ADVERTISEMENT = auto()      # Advertisement
    SIDEBAR = auto()            # Sidebar/infobox
    HEADER = auto()             # Page header
    FOOTER = auto()             # Page footer
    COLUMN_BLOCK = auto()       # Column unit block
    UNKNOWN = auto()            # Unclassifiable


@dataclass
class ContentElement:
    """Content element within page"""
    element_type: str  # 'text', 'image', 'drawing', 'table'
    bbox: Tuple[float, float, float, float]
    content: Optional[str] = None
    
    # Text properties
    font_size: float = 0.0
    is_bold: bool = False
    text_length: int = 0
    
    # Image properties
    image_area: float = 0.0
    
    # Group ID (assigned after clustering)
    group_id: int = -1


@dataclass
class LayoutBlock:
    """Semantic layout block"""
    block_id: int
    block_type: LayoutBlockType
    bbox: Tuple[float, float, float, float]
    
    # Contained elements
    elements: List[ContentElement] = field(default_factory=list)
    
    # Column information
    column_index: int = 0
    
    # Reading order (starts from 0)
    reading_order: int = 0
    
    # Confidence (0.0 ~ 1.0)
    confidence: float = 1.0
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    @property
    def area(self) -> float:
        """Block area"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Block center point"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )


@dataclass
class ColumnInfo:
    """Column information"""
    index: int
    x_start: float
    x_end: float
    
    # Blocks within column
    blocks: List[LayoutBlock] = field(default_factory=list)
    
    @property
    def width(self) -> float:
        return self.x_end - self.x_start


@dataclass
class LayoutAnalysisResult:
    """Layout analysis result"""
    page_num: int
    page_size: Tuple[float, float]
    
    # Column information
    columns: List[ColumnInfo] = field(default_factory=list)
    column_count: int = 1
    
    # Layout blocks (sorted by reading order)
    blocks: List[LayoutBlock] = field(default_factory=list)
    
    # Header/footer region
    header_region: Optional[Tuple[float, float, float, float]] = None
    footer_region: Optional[Tuple[float, float, float, float]] = None
    
    # Statistics
    total_text_elements: int = 0
    total_image_elements: int = 0
    
    # Analysis confidence
    confidence: float = 1.0


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LayoutDetectorConfig:
    """Layout detection configuration."""
    
    # Column detection settings
    MIN_COLUMN_GAP: float = 20.0        # Minimum gap between columns (pt)
    COLUMN_CLUSTER_TOLERANCE: float = 30.0  # X-coordinate clustering tolerance (pt)
    
    # Block clustering settings
    ELEMENT_PROXIMITY_THRESHOLD: float = 15.0  # Element proximity threshold (pt)
    VERTICAL_MERGE_THRESHOLD: float = 40.0     # Vertical merge distance (pt) - aggressive merge
    HORIZONTAL_MERGE_THRESHOLD: float = 15.0   # Horizontal merge distance (pt) - aggressive merge
    
    # Headline detection settings
    HEADLINE_FONT_RATIO: float = 1.3    # Headline font ratio vs. body text
    HEADLINE_MIN_SIZE: float = 14.0     # Minimum headline font size (pt)
    
    # Image-caption connection settings
    CAPTION_MAX_DISTANCE: float = 30.0  # Max image-caption distance (pt)
    CAPTION_MAX_HEIGHT: float = 50.0    # Max caption height (pt)
    
    # Header/footer settings
    HEADER_MAX_HEIGHT: float = 60.0     # Max header height (pt)
    FOOTER_MAX_HEIGHT: float = 60.0     # Max footer height (pt)
    HEADER_FOOTER_MARGIN: float = 0.1   # Page top/bottom margin ratio
    
    # Minimum block size (small blocks are merge candidates)
    MIN_BLOCK_WIDTH: float = 80.0       # Minimum block width (pt)
    MIN_BLOCK_HEIGHT: float = 60.0      # Minimum block height (pt)
    MIN_BLOCK_AREA: float = 15000.0     # Minimum block area (pt²) (~100x150pt)
    
    # Block count target (prevents too many blocks)
    TARGET_MIN_BLOCKS: int = 3          # Minimum blocks per page
    TARGET_MAX_BLOCKS: int = 10         # Maximum blocks per page (considering 5-column newspapers)
    AGGRESSIVE_MERGE_THRESHOLD: int = 15  # Aggressive merge if more than this
    
    # Advertisement detection
    AD_BOX_DETECTION: bool = True       # Detect advertisements enclosed by boxes
    AD_MIN_BOX_AREA: float = 10000.0    # Minimum box area to be considered as advertisement
    
    # Separator detection
    SEPARATOR_MIN_LENGTH_RATIO: float = 0.3  # Minimum separator length (relative to page width)
    SEPARATOR_MAX_THICKNESS: float = 3.0     # Maximum separator thickness (pt)


# ============================================================================
# Layout Block Detector
# ============================================================================

class LayoutBlockDetector:
    """
    Layout Block Detector
    
    Divides complex multi-column layouts into semantic block units.
    """
    
    def __init__(
        self, 
        page, 
        page_num: int,
        config: Optional[LayoutDetectorConfig] = None
    ):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            config: Detection configuration
        """
        self.page = page
        self.page_num = page_num
        self.config = config or LayoutDetectorConfig()
        
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        
        # Cache
        self._text_dict: Optional[Dict] = None
        self._drawings: Optional[List] = None
        self._images: Optional[List] = None
        
        # Internal state
        self._elements: List[ContentElement] = []
        self._separators: List[Tuple[float, float, float, float]] = []
        self._boxes: List[Tuple[float, float, float, float]] = []
    
    def detect(self) -> LayoutAnalysisResult:
        """
        Detect layout blocks.
        
        Returns:
            LayoutAnalysisResult object
        """
        columns = [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        header_region = None
        footer_region = None
        blocks = []
        
        try:
            # Phase 1: Basic Analysis
            try:
                self._extract_elements()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 1 (_extract_elements) failed: {e}")
                self._elements = []
            
            try:
                self._extract_separators_and_boxes()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 1 (_extract_separators_and_boxes) failed: {e}")
                self._separators = []
                self._boxes = []
            
            # Phase 2: Column detection
            try:
                columns = self._detect_columns()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 2 (_detect_columns) failed: {e}")
                columns = [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
            
            # Phase 3: Header/footer detection
            try:
                header_region, footer_region = self._detect_header_footer()
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 3 (_detect_header_footer) failed: {e}")
                header_region = None
                footer_region = None
            
            # Phase 4: Semantic block clustering
            try:
                blocks = self._cluster_into_blocks(columns, header_region, footer_region)
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 4 (_cluster_into_blocks) failed: {e}")
                # Fallback: Create simple column-based blocks
                blocks = self._create_column_based_blocks(columns)
            
            # Phase 5: Block classification
            try:
                self._classify_blocks(blocks)
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 5 (_classify_blocks) failed: {e}")
            
            # Phase 6: Block optimization and sorting
            try:
                blocks = self._optimize_and_sort_blocks(blocks, columns)
            except Exception as e:
                logger.warning(f"[LayoutBlockDetector] Phase 6 (_optimize_and_sort_blocks) failed: {e}")
        
        except Exception as e:
            logger.error(f"[LayoutBlockDetector] Critical error during detection: {e}")
            # Return entire page as a single block as minimum fallback
            blocks = [LayoutBlock(
                block_id=0,
                block_type=LayoutBlockType.UNKNOWN,
                bbox=(0, 0, self.page_width, self.page_height),
                elements=self._elements if self._elements else [],
                column_index=0,
                reading_order=0,
                confidence=0.1
            )]
        
        result = LayoutAnalysisResult(
            page_num=self.page_num,
            page_size=(self.page_width, self.page_height),
            columns=columns,
            column_count=len(columns),
            blocks=blocks,
            header_region=header_region,
            footer_region=footer_region,
            total_text_elements=sum(1 for e in self._elements if e.element_type == 'text'),
            total_image_elements=sum(1 for e in self._elements if e.element_type == 'image'),
            confidence=self._calculate_confidence(blocks, columns)
        )
        
        logger.info(f"[LayoutBlockDetector] Page {self.page_num + 1}: "
                   f"detected {len(blocks)} blocks in {len(columns)} columns")
        
        return result
    
    def _create_column_based_blocks(self, columns: List[ColumnInfo]) -> List[LayoutBlock]:
        """
        Fallback: Create simple column-based blocks.
        
        When clustering fails, treats each column as a single block.
        """
        blocks = []
        block_id = 0
        
        for col in columns:
            # Elements belonging to this column
            col_elements = [
                e for e in self._elements 
                if self._element_in_column(e, col)
            ]
            
            if col_elements:
                bbox = self._merge_bboxes([e.bbox for e in col_elements])
                blocks.append(LayoutBlock(
                    block_id=block_id,
                    block_type=LayoutBlockType.COLUMN_BLOCK,
                    bbox=bbox,
                    elements=col_elements,
                    column_index=col.index,
                    reading_order=block_id,
                    confidence=0.5
                ))
                block_id += 1
        
        # If no elements, create entire page as a single block
        if not blocks:
            blocks.append(LayoutBlock(
                block_id=0,
                block_type=LayoutBlockType.UNKNOWN,
                bbox=(0, 0, self.page_width, self.page_height),
                elements=[],
                column_index=0,
                reading_order=0,
                confidence=0.1
            ))
        
        return blocks
    
    # ========================================================================
    # Phase 1: Basic Analysis
    # ========================================================================
    
    def _extract_elements(self):
        """Extract all content elements from the page."""
        self._elements = []
        
        # 1. Extract text blocks
        text_dict = self._get_text_dict()
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Text blocks only
                continue
            
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            
            # Collect font information
            max_font_size = 0.0
            is_bold = False
            total_text = ""
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_size = span.get("size", 0.0)
                    if font_size > max_font_size:
                        max_font_size = font_size
                    
                    flags = span.get("flags", 0)
                    if flags & 2**4:  # Bold flag
                        is_bold = True
                    
                    total_text += span.get("text", "")
            
            if total_text.strip():
                self._elements.append(ContentElement(
                    element_type='text',
                    bbox=bbox,
                    content=total_text.strip(),
                    font_size=max_font_size,
                    is_bold=is_bold,
                    text_length=len(total_text.strip())
                ))
        
        # 2. Extract images
        images = self._get_images()
        for img_info in images:
            xref = img_info[0]
            try:
                # Find image position
                img_bbox = self._find_image_position(xref)
                if img_bbox:
                    area = (img_bbox[2] - img_bbox[0]) * (img_bbox[3] - img_bbox[1])
                    self._elements.append(ContentElement(
                        element_type='image',
                        bbox=img_bbox,
                        image_area=area
                    ))
            except Exception:
                pass
    
    def _extract_separators_and_boxes(self):
        """Extract separators and boxes."""
        self._separators = []
        self._boxes = []
        
        drawings = self._get_drawings()
        
        for drawing in drawings:
            try:
                rect = drawing.get("rect")
                if not rect:
                    continue
                
                # Safely access rect attributes
                try:
                    w = rect.width
                    h = rect.height
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                except (AttributeError, TypeError):
                    # rect might be a tuple
                    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
                        x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
                        w = x1 - x0
                        h = y1 - y0
                    else:
                        continue
                
                # Horizontal separator
                if (h <= self.config.SEPARATOR_MAX_THICKNESS and 
                    w >= self.page_width * self.config.SEPARATOR_MIN_LENGTH_RATIO):
                    self._separators.append((x0, y0, x1, y1))
                
                # Vertical separator
                elif (w <= self.config.SEPARATOR_MAX_THICKNESS and 
                      h >= self.page_height * self.config.SEPARATOR_MIN_LENGTH_RATIO * 0.5):
                    self._separators.append((x0, y0, x1, y1))
                
                # Box (Advertisement/infobox candidate)
                elif w > 50 and h > 50:
                    area = w * h
                    if area >= self.config.AD_MIN_BOX_AREA:
                        # Check if it's a box with border
                        # NOTE: stroke_opacity can be None, so handle safely
                        stroke_opacity = drawing.get("stroke_opacity")
                        has_stroke = drawing.get("color") or (stroke_opacity is not None and stroke_opacity > 0)
                        if has_stroke:
                            self._boxes.append((x0, y0, x1, y1))
            except Exception as e:
                # Log and continue on individual drawing processing failure
                logger.debug(f"[LayoutBlockDetector] Error processing drawing: {e}")
                continue
    
    # ========================================================================
    # Phase 2: Column detection
    # ========================================================================
    
    def _detect_columns(self) -> List[ColumnInfo]:
        """Detect column structure."""
        if not self._elements:
            return [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        
        # Collect X start positions of text elements
        x_starts = []
        for elem in self._elements:
            if elem.element_type == 'text' and elem.text_length > 20:  # Only sufficiently long text
                x_starts.append(elem.bbox[0])
        
        if not x_starts:
            return [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        
        # X-coordinate clustering
        x_starts.sort()
        clusters = self._cluster_x_positions(x_starts)
        
        if len(clusters) <= 1:
            return [ColumnInfo(index=0, x_start=0, x_end=self.page_width)]
        
        # Analyze gaps between clusters
        cluster_centers = [sum(c) / len(c) for c in clusters]
        
        # Only clusters with sufficient gap are recognized as columns
        columns = []
        valid_boundaries = [0]
        
        for i in range(len(cluster_centers) - 1):
            gap = cluster_centers[i + 1] - cluster_centers[i]
            if gap >= self.config.MIN_COLUMN_GAP:
                # Column boundary = midpoint between two clusters
                boundary = (cluster_centers[i] + cluster_centers[i + 1]) / 2
                valid_boundaries.append(boundary)
        
        valid_boundaries.append(self.page_width)
        
        # Create columns
        for i in range(len(valid_boundaries) - 1):
            columns.append(ColumnInfo(
                index=i,
                x_start=valid_boundaries[i],
                x_end=valid_boundaries[i + 1]
            ))
        
        logger.debug(f"[LayoutBlockDetector] Detected {len(columns)} columns")
        return columns
    
    def _cluster_x_positions(self, x_positions: List[float]) -> List[List[float]]:
        """X-coordinate clustering (density-based)."""
        if not x_positions:
            return []
        
        clusters = []
        current_cluster = [x_positions[0]]
        
        for x in x_positions[1:]:
            if x - current_cluster[-1] <= self.config.COLUMN_CLUSTER_TOLERANCE:
                current_cluster.append(x)
            else:
                if len(current_cluster) >= 3:  # Minimum 3 elements
                    clusters.append(current_cluster)
                current_cluster = [x]
        
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)
        
        return clusters
    
    # ========================================================================
    # Phase 3: Header/footer detection
    # ========================================================================
    
    def _detect_header_footer(self) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Detect header and footer regions."""
        header_region = None
        footer_region = None
        
        header_boundary = self.page_height * self.config.HEADER_FOOTER_MARGIN
        footer_boundary = self.page_height * (1 - self.config.HEADER_FOOTER_MARGIN)
        
        # Analyze top region
        header_elements = [
            e for e in self._elements 
            if e.bbox[3] <= header_boundary and e.element_type == 'text'
        ]
        
        if header_elements:
            min_y = min(e.bbox[1] for e in header_elements)
            max_y = max(e.bbox[3] for e in header_elements)
            
            if max_y - min_y <= self.config.HEADER_MAX_HEIGHT:
                header_region = (0, min_y, self.page_width, max_y)
        
        # Analyze bottom region
        footer_elements = [
            e for e in self._elements 
            if e.bbox[1] >= footer_boundary and e.element_type == 'text'
        ]
        
        if footer_elements:
            min_y = min(e.bbox[1] for e in footer_elements)
            max_y = max(e.bbox[3] for e in footer_elements)
            
            if max_y - min_y <= self.config.FOOTER_MAX_HEIGHT:
                footer_region = (0, min_y, self.page_width, max_y)
        
        return header_region, footer_region
    
    # ========================================================================
    # Phase 4: Semantic block clustering
    # ========================================================================
    
    def _cluster_into_blocks(
        self, 
        columns: List[ColumnInfo],
        header_region: Optional[Tuple],
        footer_region: Optional[Tuple]
    ) -> List[LayoutBlock]:
        """Cluster elements into semantic blocks."""
        blocks = []
        block_id = 0
        
        # Elements excluding header/footer regions
        main_elements = []
        header_elements = []
        footer_elements = []
        
        for elem in self._elements:
            if header_region and self._is_inside(elem.bbox, header_region):
                header_elements.append(elem)
            elif footer_region and self._is_inside(elem.bbox, footer_region):
                footer_elements.append(elem)
            else:
                main_elements.append(elem)
        
        # Header block
        if header_elements:
            bbox = self._merge_bboxes([e.bbox for e in header_elements])
            blocks.append(LayoutBlock(
                block_id=block_id,
                block_type=LayoutBlockType.HEADER,
                bbox=bbox,
                elements=header_elements
            ))
            block_id += 1
        
        # Process by column
        for col in columns:
            # Elements belonging to this column
            col_elements = [
                e for e in main_elements 
                if self._element_in_column(e, col)
            ]
            
            if not col_elements:
                continue
            
            # Vertical split based on separators
            vertical_groups = self._split_by_separators(col_elements, col)
            
            for group_elements in vertical_groups:
                if not group_elements:
                    continue
                
                # Adjacent element clustering
                clusters = self._cluster_adjacent_elements(group_elements)
                
                for cluster in clusters:
                    if not cluster:
                        continue
                    
                    bbox = self._merge_bboxes([e.bbox for e in cluster])
                    
                    # Ignore too small blocks
                    if (bbox[2] - bbox[0] < self.config.MIN_BLOCK_WIDTH or
                        bbox[3] - bbox[1] < self.config.MIN_BLOCK_HEIGHT):
                        continue
                    
                    blocks.append(LayoutBlock(
                        block_id=block_id,
                        block_type=LayoutBlockType.UNKNOWN,  # Classify later
                        bbox=bbox,
                        elements=cluster,
                        column_index=col.index
                    ))
                    block_id += 1
        
        # Footer block
        if footer_elements:
            bbox = self._merge_bboxes([e.bbox for e in footer_elements])
            blocks.append(LayoutBlock(
                block_id=block_id,
                block_type=LayoutBlockType.FOOTER,
                bbox=bbox,
                elements=footer_elements
            ))
        
        return blocks
    
    def _element_in_column(self, elem: ContentElement, col: ColumnInfo) -> bool:
        """Check if element belongs to column."""
        elem_center_x = (elem.bbox[0] + elem.bbox[2]) / 2
        return col.x_start <= elem_center_x <= col.x_end
    
    def _split_by_separators(
        self, 
        elements: List[ContentElement], 
        col: ColumnInfo
    ) -> List[List[ContentElement]]:
        """Split vertically based on separators."""
        if not elements:
            return []
        
        # Find horizontal separators within this column
        col_separators = []
        for sep in self._separators:
            # Check if horizontal separator overlaps with this column
            is_horizontal = abs(sep[3] - sep[1]) < 5
            if is_horizontal:
                sep_start_x = sep[0]
                sep_end_x = sep[2]
                if (sep_start_x <= col.x_end and sep_end_x >= col.x_start):
                    col_separators.append(sep[1])  # Y coordinate
        
        if not col_separators:
            return [elements]
        
        # Sort separator positions
        col_separators.sort()
        
        # Split elements based on separators
        groups = []
        boundaries = [0] + col_separators + [self.page_height]
        
        for i in range(len(boundaries) - 1):
            y_start = boundaries[i]
            y_end = boundaries[i + 1]
            
            group = [
                e for e in elements
                if e.bbox[1] >= y_start - 5 and e.bbox[3] <= y_end + 5
            ]
            
            if group:
                groups.append(group)
        
        return groups if groups else [elements]
    
    def _cluster_adjacent_elements(
        self, 
        elements: List[ContentElement]
    ) -> List[List[ContentElement]]:
        """Adjacent element clustering."""
        if not elements:
            return []
        
        if len(elements) == 1:
            return [elements]
        
        # Sort elements by Y coordinate
        sorted_elements = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))
        
        # Union-Find style clustering
        clusters: List[List[ContentElement]] = []
        used = set()
        
        for elem in sorted_elements:
            if id(elem) in used:
                continue
            
            # Start new cluster
            cluster = [elem]
            used.add(id(elem))
            queue = [elem]
            
            while queue:
                current = queue.pop(0)
                
                for other in sorted_elements:
                    if id(other) in used:
                        continue
                    
                    if self._are_adjacent(current, other):
                        cluster.append(other)
                        used.add(id(other))
                        queue.append(other)
            
            clusters.append(cluster)
        
        return clusters
    
    def _are_adjacent(self, e1: ContentElement, e2: ContentElement) -> bool:
        """Check if two elements are adjacent."""
        # Vertical gap
        vertical_gap = max(0, e2.bbox[1] - e1.bbox[3], e1.bbox[1] - e2.bbox[3])
        
        # Horizontal overlap
        x_overlap_start = max(e1.bbox[0], e2.bbox[0])
        x_overlap_end = min(e1.bbox[2], e2.bbox[2])
        has_x_overlap = x_overlap_start < x_overlap_end
        
        # Vertically adjacent (same X range, close Y)
        if has_x_overlap and vertical_gap <= self.config.VERTICAL_MERGE_THRESHOLD:
            return True
        
        # Horizontally adjacent (same Y range)
        horizontal_gap = max(0, e2.bbox[0] - e1.bbox[2], e1.bbox[0] - e2.bbox[2])
        
        y_overlap_start = max(e1.bbox[1], e2.bbox[1])
        y_overlap_end = min(e1.bbox[3], e2.bbox[3])
        has_y_overlap = y_overlap_start < y_overlap_end
        
        if has_y_overlap and horizontal_gap <= self.config.HORIZONTAL_MERGE_THRESHOLD:
            return True
        
        return False
    
    # ========================================================================
    # Phase 5: Block classification
    # ========================================================================
    
    def _classify_blocks(self, blocks: List[LayoutBlock]):
        """Classify block types."""
        for block in blocks:
            if block.block_type in (LayoutBlockType.HEADER, LayoutBlockType.FOOTER):
                continue
            
            block.block_type = self._determine_block_type(block)
    
    def _determine_block_type(self, block: LayoutBlock) -> LayoutBlockType:
        """Determine block type."""
        text_elements = [e for e in block.elements if e.element_type == 'text']
        image_elements = [e for e in block.elements if e.element_type == 'image']
        
        has_text = len(text_elements) > 0
        has_image = len(image_elements) > 0
        
        # Image + text = IMAGE_WITH_CAPTION
        if has_image and has_text:
            # Check if text is above/below image
            for img_elem in image_elements:
                for txt_elem in text_elements:
                    if self._is_caption_of_image(txt_elem, img_elem):
                        return LayoutBlockType.IMAGE_WITH_CAPTION
            return LayoutBlockType.IMAGE_WITH_CAPTION  # Default assumption
        
        # Image only = STANDALONE_IMAGE
        if has_image and not has_text:
            return LayoutBlockType.STANDALONE_IMAGE
        
        # Text only
        if has_text:
            # Headline detection (large font + short text)
            avg_font_size = sum(e.font_size for e in text_elements) / len(text_elements)
            max_font_size = max(e.font_size for e in text_elements)
            
            # If font size variation is large, ARTICLE (headline + body)
            if max_font_size >= self.config.HEADLINE_MIN_SIZE:
                if max_font_size >= avg_font_size * self.config.HEADLINE_FONT_RATIO:
                    return LayoutBlockType.ARTICLE
            
            # If inside a box, SIDEBAR or ADVERTISEMENT
            if self._is_inside_box(block.bbox):
                # Short text means advertisement
                total_text_len = sum(e.text_length for e in text_elements)
                if total_text_len < 200:
                    return LayoutBlockType.ADVERTISEMENT
                return LayoutBlockType.SIDEBAR
            
            return LayoutBlockType.ARTICLE
        
        return LayoutBlockType.UNKNOWN
    
    def _is_caption_of_image(self, text_elem: ContentElement, img_elem: ContentElement) -> bool:
        """Check if text is a caption for the image."""
        # Directly below image
        if (text_elem.bbox[1] > img_elem.bbox[3] - 5 and
            text_elem.bbox[1] < img_elem.bbox[3] + self.config.CAPTION_MAX_DISTANCE):
            # Similar X range
            if (text_elem.bbox[0] >= img_elem.bbox[0] - 20 and
                text_elem.bbox[2] <= img_elem.bbox[2] + 20):
                # Height within caption range
                if text_elem.bbox[3] - text_elem.bbox[1] <= self.config.CAPTION_MAX_HEIGHT:
                    return True
        
        # Also possible directly above image
        if (text_elem.bbox[3] < img_elem.bbox[1] + 5 and
            text_elem.bbox[3] > img_elem.bbox[1] - self.config.CAPTION_MAX_DISTANCE):
            if (text_elem.bbox[0] >= img_elem.bbox[0] - 20 and
                text_elem.bbox[2] <= img_elem.bbox[2] + 20):
                if text_elem.bbox[3] - text_elem.bbox[1] <= self.config.CAPTION_MAX_HEIGHT:
                    return True
        
        return False
    
    def _is_inside_box(self, bbox: Tuple) -> bool:
        """Check if block is inside a box."""
        for box in self._boxes:
            if self._is_inside(bbox, box, margin=10):
                return True
        return False
    
    # ========================================================================
    # Phase 6: Block optimization and sorting
    # ========================================================================
    
    def _optimize_and_sort_blocks(
        self, 
        blocks: List[LayoutBlock],
        columns: List[ColumnInfo]
    ) -> List[LayoutBlock]:
        """Block optimization and reading order sorting."""
        if not blocks:
            return []
        
        # 1. Merge small blocks
        blocks = self._merge_small_blocks(blocks)
        
        # 2. Resolve overlaps
        blocks = self._resolve_overlaps(blocks)
        
        # 3. Determine reading order
        #    - Header first
        #    - Column order (left → right)
        #    - Top to bottom within column
        #    - Footer last
        
        header_blocks = [b for b in blocks if b.block_type == LayoutBlockType.HEADER]
        footer_blocks = [b for b in blocks if b.block_type == LayoutBlockType.FOOTER]
        main_blocks = [b for b in blocks if b.block_type not in (LayoutBlockType.HEADER, LayoutBlockType.FOOTER)]
        
        # Sort by column
        column_groups = defaultdict(list)
        for block in main_blocks:
            column_groups[block.column_index].append(block)
        
        # Sort by Y coordinate within each column
        for col_idx in column_groups:
            column_groups[col_idx].sort(key=lambda b: b.bbox[1])
        
        # Final order: Header → (by column) → Footer
        sorted_blocks = []
        order = 0
        
        for block in header_blocks:
            block.reading_order = order
            sorted_blocks.append(block)
            order += 1
        
        for col_idx in sorted(column_groups.keys()):
            for block in column_groups[col_idx]:
                block.reading_order = order
                sorted_blocks.append(block)
                order += 1
        
        for block in footer_blocks:
            block.reading_order = order
            sorted_blocks.append(block)
            order += 1
        
        return sorted_blocks
    
    def _merge_small_blocks(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """Merge adjacent blocks that are too small."""
        if len(blocks) <= 1:
            return blocks
        
        # Skip merge if block count is within target range
        if len(blocks) <= self.config.TARGET_MAX_BLOCKS:
            return blocks
        
        result = []
        skip_ids = set()
        
        # Aggressive merge if too many blocks
        aggressive_merge = len(blocks) > self.config.AGGRESSIVE_MERGE_THRESHOLD
        
        for block in blocks:
            if block.block_id in skip_ids:
                continue
            
            # Check if small block (raise threshold for aggressive merge)
            min_area = self.config.MIN_BLOCK_AREA
            if aggressive_merge:
                min_area = self.config.MIN_BLOCK_AREA * 2  # 2x threshold
            
            if block.area >= min_area:
                result.append(block)
                continue
            
            # Find adjacent block
            merged = False
            for other in blocks:
                if other.block_id == block.block_id or other.block_id in skip_ids:
                    continue
                
                if self._should_merge_blocks(block, other, aggressive=aggressive_merge):
                    # Merge
                    merged_bbox = self._merge_bboxes([block.bbox, other.bbox])
                    other.bbox = merged_bbox
                    other.elements.extend(block.elements)
                    skip_ids.add(block.block_id)
                    merged = True
                    break
            
            if not merged:
                result.append(block)
        
        # Try additional merge if still above target
        if len(result) > self.config.TARGET_MAX_BLOCKS:
            result = self._force_merge_to_target(result)
        
        return result
    
    def _should_merge_blocks(self, b1: LayoutBlock, b2: LayoutBlock, aggressive: bool = False) -> bool:
        """Check if two blocks should be merged."""
        # Same column (allow adjacent columns for aggressive merge)
        if not aggressive and b1.column_index != b2.column_index:
            return False
        if aggressive and abs(b1.column_index - b2.column_index) > 1:
            return False
        
        # Close distance
        vertical_gap = max(0, b2.bbox[1] - b1.bbox[3], b1.bbox[1] - b2.bbox[3])
        threshold = self.config.VERTICAL_MERGE_THRESHOLD * (3 if aggressive else 2)
        if vertical_gap > threshold:
            return False
        
        return True
    
    def _force_merge_to_target(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        Force merge when block count exceeds target.
        Merges adjacent blocks within the same column.
        """
        if len(blocks) <= self.config.TARGET_MAX_BLOCKS:
            return blocks
        
        # Group by column
        column_groups: Dict[int, List[LayoutBlock]] = defaultdict(list)
        for block in blocks:
            column_groups[block.column_index].append(block)
        
        result = []
        
        for col_idx in sorted(column_groups.keys()):
            col_blocks = sorted(column_groups[col_idx], key=lambda b: b.bbox[1])
            
            # If 2+ blocks in column, merge is possible
            if len(col_blocks) >= 2:
                # Merge adjacent blocks
                merged_blocks = self._merge_adjacent_in_column(col_blocks)
                result.extend(merged_blocks)
            else:
                result.extend(col_blocks)
        
        logger.debug(f"[LayoutBlockDetector] Force merged: {len(blocks)} → {len(result)} blocks")
        return result
    
    def _merge_adjacent_in_column(self, col_blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        Merge adjacent blocks within a column.
        Reduces to at most 2-3 blocks.
        """
        if len(col_blocks) <= 2:
            return col_blocks
        
        # Divide blocks into 2-3 groups
        target_groups = max(2, min(3, len(col_blocks) // 2))
        blocks_per_group = max(1, len(col_blocks) // target_groups)
        
        result = []
        current_group = []
        
        for i, block in enumerate(col_blocks):
            current_group.append(block)
            
            # When group is filled, merge
            if len(current_group) >= blocks_per_group and len(result) < target_groups - 1:
                merged = self._merge_block_group(current_group)
                result.append(merged)
                current_group = []
        
        # Merge remaining blocks
        if current_group:
            merged = self._merge_block_group(current_group)
            result.append(merged)
        
        return result
    
    def _merge_block_group(self, blocks: List[LayoutBlock]) -> LayoutBlock:
        """Merge a group of blocks into one."""
        if len(blocks) == 1:
            return blocks[0]
        
        merged_bbox = self._merge_bboxes([b.bbox for b in blocks])
        merged_elements = []
        for b in blocks:
            merged_elements.extend(b.elements)
        
        return LayoutBlock(
            block_id=blocks[0].block_id,
            block_type=blocks[0].block_type,
            bbox=merged_bbox,
            elements=merged_elements,
            column_index=blocks[0].column_index,
            reading_order=blocks[0].reading_order,
            confidence=min(b.confidence for b in blocks)
        )
    
    def _resolve_overlaps(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """Resolve block overlaps."""
        # Currently simply returns (can be improved later)
        return blocks
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_text_dict(self) -> Dict:
        """Cached text dictionary."""
        if self._text_dict is None:
            self._text_dict = self.page.get_text("dict", sort=True)
        return self._text_dict
    
    def _get_drawings(self) -> List:
        """Cached drawings."""
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings
    
    def _get_images(self) -> List:
        """Cached images."""
        if self._images is None:
            self._images = self.page.get_images()
        return self._images
    
    def _find_image_position(self, xref: int) -> Optional[Tuple[float, float, float, float]]:
        """Find image position."""
        try:
            for img in self.page.get_image_rects(xref):
                return (img.x0, img.y0, img.x1, img.y1)
        except Exception:
            pass
        return None
    
    def _is_inside(
        self, 
        inner: Tuple[float, float, float, float], 
        outer: Tuple[float, float, float, float],
        margin: float = 0
    ) -> bool:
        """Check if inner is inside outer."""
        return (
            inner[0] >= outer[0] - margin and
            inner[1] >= outer[1] - margin and
            inner[2] <= outer[2] + margin and
            inner[3] <= outer[3] + margin
        )
    
    def _merge_bboxes(self, bboxes: List[Tuple]) -> Tuple[float, float, float, float]:
        """Merge multiple bboxes."""
        if not bboxes:
            return (0, 0, 0, 0)
        
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        return (x0, y0, x1, y1)
    
    def _calculate_confidence(self, blocks: List[LayoutBlock], columns: List[ColumnInfo]) -> float:
        """Calculate analysis confidence."""
        if not blocks:
            return 0.5
        
        # Ratio of blocks to total elements
        total_elements = len(self._elements)
        if total_elements == 0:
            return 0.5
        
        covered_elements = sum(len(b.elements) for b in blocks)
        coverage = covered_elements / total_elements
        
        # Ratio of UNKNOWN blocks
        unknown_ratio = sum(1 for b in blocks if b.block_type == LayoutBlockType.UNKNOWN) / max(1, len(blocks))
        
        confidence = coverage * (1 - unknown_ratio * 0.3)
        
        return min(1.0, max(0.0, confidence))


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'LayoutBlockType',
    'ContentElement',
    'LayoutBlock',
    'ColumnInfo',
    'LayoutAnalysisResult',
    'LayoutDetectorConfig',
    'LayoutBlockDetector',
]
