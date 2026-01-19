"""
Block Image Engine for PDF Handler

Splits complex regions into semantic block units, renders them as images, and saves locally.

=============================================================================
Core Concepts:
=============================================================================
Previous: Upload entire page as single image
Improved: Split page into **semantic/logical block units** and save each as PNG

Benefits:
1. LLM can interpret each block **individually**
2. Resolution issues resolved (high resolution maintained per block)
3. Reading order preserved
4. Context separation (ads/articles/tables distinguished)

=============================================================================
Processing Strategies:
=============================================================================
1. SEMANTIC_BLOCKS: Semantic block-based splitting (recommended)
   - Block detection via LayoutBlockDetector
   - Convert each block to individual image
   - Generate [Image:path] tags in reading order

2. GRID_BLOCKS: Grid-based splitting (fallback)
   - Split page into NxM grid
   - Convert each grid cell to individual image

3. FULL_PAGE: Full page imaging (last resort)
   - Maintain existing approach

Rendering Settings:
- Default DPI: 300 (high resolution)
- Max image size: 4096px
- Image format: PNG (lossless)
"""

import logging
import io
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto

import fitz
from PIL import Image

# Image processing module
from libs.core.functions.img_processor import ImageProcessor

logger = logging.getLogger(__name__)


# ============================================================================
# Block Strategy Enum
# ============================================================================

class BlockStrategy(Enum):
    """Block processing strategy."""
    SEMANTIC_BLOCKS = auto()  # Semantic block units
    GRID_BLOCKS = auto()      # Grid-based splitting
    FULL_PAGE = auto()        # Full page


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BlockImageConfig:
    """Block image engine configuration."""
    # Rendering settings
    DEFAULT_DPI: int = 300
    MAX_IMAGE_SIZE: int = 4096

    # Image format
    IMAGE_FORMAT: str = "PNG"

    # Region settings
    REGION_PADDING: int = 5  # Region padding (pt)

    # Minimum size (below this is ignored)
    MIN_REGION_WIDTH: int = 80   # Increased
    MIN_REGION_HEIGHT: int = 60  # Increased

    # Block splitting strategy
    PREFERRED_STRATEGY: str = "semantic"  # semantic, grid, full_page

    # Grid splitting settings (for GRID_BLOCKS strategy)
    GRID_ROWS: int = 2
    GRID_COLS: int = 2

    # Block merging settings
    MERGE_SMALL_BLOCKS: bool = True
    MIN_BLOCK_AREA: float = 15000.0  # Minimum block area (pt²) - significantly increased

    # Empty block filtering
    SKIP_EMPTY_BLOCKS: bool = True
    EMPTY_THRESHOLD: float = 0.95  # Block is empty if white pixel ratio exceeds this


@dataclass
class BlockImageResult:
    """Block image processing result."""
    bbox: Tuple[float, float, float, float]

    # Image info
    image_size: Tuple[int, int]
    dpi: int

    # Image path
    image_path: Optional[str] = None

    # Inline tag ([Image:{path}] format)
    image_tag: Optional[str] = None

    # Success status
    success: bool = False
    error: Optional[str] = None

    # Block info (advanced)
    block_type: Optional[str] = None  # Block type (article, image, table, etc.)
    reading_order: int = 0            # Reading order
    column_index: int = 0             # Column index


@dataclass
class MultiBlockResult:
    """Multi-block processing result."""
    page_num: int
    strategy_used: BlockStrategy

    # Individual block results (in reading order)
    block_results: List[BlockImageResult] = field(default_factory=list)

    # Overall success status
    success: bool = False

    # Combined text output (includes all [Image:...] tags)
    combined_output: str = ""

    # Statistics
    total_blocks: int = 0
    successful_blocks: int = 0
    failed_blocks: int = 0


# ============================================================================
# Block Image Engine
# ============================================================================

class BlockImageEngine:
    """
    Block Image Engine

    Renders complex regions as images and saves locally.
    Results are returned in [image:{path}] format.
    """

    def __init__(
        self,
        page,
        page_num: int,
        image_processor: ImageProcessor,
        config: Optional[BlockImageConfig] = None
    ):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            image_processor: ImageProcessor instance for saving images
            config: Engine configuration (BlockImageConfig)
        """
        self.page = page
        self.page_num = page_num
        self.config = config or BlockImageConfig()

        self.page_width = page.rect.width
        self.page_height = page.rect.height

        self._image_processor = image_processor

        # Processed image hashes (duplicate prevention)
        self._processed_hashes: set = set()

    def process_region(
        self,
        bbox: Tuple[float, float, float, float],
        region_type: str = "complex_region"
    ) -> BlockImageResult:
        """
        Renders a specific region as an image and saves locally.

        Args:
            bbox: Region to process (x0, y0, x1, y1)
            region_type: Region type (for logging)

        Returns:
            BlockImageResult object (includes image_path, image_tag)
        """
        try:
            # Minimum size validation
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width < self.config.MIN_REGION_WIDTH or height < self.config.MIN_REGION_HEIGHT:
                return BlockImageResult(
                    bbox=bbox,
                    image_size=(0, 0),
                    dpi=0,
                    success=False,
                    error="Region too small"
                )

            # 1. Render region image
            image_bytes, actual_dpi, image_size = self._render_region(bbox)

            if image_bytes is None:
                return BlockImageResult(
                    bbox=bbox,
                    image_size=(0, 0),
                    dpi=self.config.DEFAULT_DPI,
                    success=False,
                    error="Failed to render region"
                )

            # 2. Duplicate check
            image_hash = hashlib.md5(image_bytes).hexdigest()
            if image_hash in self._processed_hashes:
                return BlockImageResult(
                    bbox=bbox,
                    image_size=image_size,
                    dpi=actual_dpi,
                    success=False,
                    error="Duplicate image"
                )
            self._processed_hashes.add(image_hash)

            # 3. Save locally (using ImageProcessor)
            image_tag = self._image_processor.save_image(image_bytes)

            if not image_tag:
                return BlockImageResult(
                    bbox=bbox,
                    image_size=image_size,
                    dpi=actual_dpi,
                    success=False,
                    error="Failed to save image"
                )

            # Extract path (from tag)
            image_path = image_tag.replace("[Image:", "").replace("]", "")

            logger.debug(f"[BlockImageEngine] Saved {region_type} at page {self.page_num + 1}: {image_path}")

            return BlockImageResult(
                bbox=bbox,
                image_size=image_size,
                dpi=actual_dpi,
                image_path=image_path,
                image_tag=image_tag,
                success=True
            )

        except Exception as e:
            logger.error(f"[BlockImageEngine] Error processing region {bbox}: {e}")
            return BlockImageResult(
                bbox=bbox,
                image_size=(0, 0),
                dpi=self.config.DEFAULT_DPI,
                success=False,
                error=str(e)
            )

    def process_full_page(self, region_type: str = "full_page") -> BlockImageResult:
        """
        Renders the entire page as an image and saves locally.

        Args:
            region_type: Region type (for logging)

        Returns:
            BlockImageResult object
        """
        bbox = (0, 0, self.page_width, self.page_height)
        return self.process_region(bbox, region_type)

    def process_regions(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        region_type: str = "complex_region"
    ) -> List[BlockImageResult]:
        """
        Processes multiple regions.

        Args:
            bboxes: List of regions to process
            region_type: Region type (for logging)

        Returns:
            List of BlockImageResult objects
        """
        results = []
        for bbox in bboxes:
            result = self.process_region(bbox, region_type)
            results.append(result)
        return results

    def _render_region(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[Optional[bytes], int, Tuple[int, int]]:
        """
        Renders a region to image bytes.

        Args:
            bbox: Region to render

        Returns:
            (image bytes, actual DPI, (width, height))
        """
        try:
            # Apply padding
            padding = self.config.REGION_PADDING
            x0 = max(0, bbox[0] - padding)
            y0 = max(0, bbox[1] - padding)
            x1 = min(self.page_width, bbox[2] + padding)
            y1 = min(self.page_height, bbox[3] + padding)

            # Create clip rect
            clip_rect = fitz.Rect(x0, y0, x1, y1)

            # Calculate DPI (considering max image size)
            dpi = self.config.DEFAULT_DPI

            region_width = x1 - x0
            region_height = y1 - y0

            max_dim = max(region_width, region_height)
            expected_size = max_dim * dpi / 72

            if expected_size > self.config.MAX_IMAGE_SIZE:
                # Adjust DPI
                dpi = int(self.config.MAX_IMAGE_SIZE * 72 / max_dim)

            # Create matrix (zoom = DPI / 72)
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)

            # Render
            pix = self.page.get_pixmap(matrix=matrix, clip=clip_rect)

            # Convert to PNG bytes
            image_bytes = pix.tobytes("png")
            image_size = (pix.width, pix.height)

            return image_bytes, dpi, image_size

        except Exception as e:
            logger.error(f"[BlockImageEngine] Render error: {e}")
            return None, 0, (0, 0)

    def render_to_bytes(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[bytes]:
        """
        Renders a region to image bytes (without saving).

        Args:
            bbox: Region to render

        Returns:
            Image bytes
        """
        image_bytes, _, _ = self._render_region(bbox)
        return image_bytes

    # ========================================================================
    # Advanced Block Processing
    # ========================================================================

    def process_page_as_semantic_blocks(self) -> MultiBlockResult:
        """
        Advanced processing: Splits page into semantic block units for processing.

        Unlike traditional FULL_PAGE_OCR:
        1. Detect semantic blocks with LayoutBlockDetector
        2. Render each block as individual image
        3. Generate [Image:path] tags in reading order

        Returns:
            MultiBlockResult object (contains all block results)
        """
        try:
            # 1. Layout block detection
            from libs.core.processor.pdf_helpers.pdf_layout_block_detector import (
                LayoutBlockDetector,
                LayoutBlock,
            )

            detector = LayoutBlockDetector(self.page, self.page_num)
            layout_result = detector.detect()

            if not layout_result.blocks:
                logger.warning(f"[BlockImageEngine] No blocks detected, falling back to full page")
                return self._fallback_to_full_page()

            logger.info(f"[BlockImageEngine] Page {self.page_num + 1}: "
                       f"Detected {len(layout_result.blocks)} semantic blocks in {layout_result.column_count} columns")

            # 2. Process each block as individual image
            block_results: List[BlockImageResult] = []

            for block in layout_result.blocks:
                # Filter out blocks that are too small (by area)
                # NOTE: Process if block region is valid even without elements
                if block.area < self.config.MIN_BLOCK_AREA:
                    logger.debug(f"[BlockImageEngine] Skipping small block: area={block.area:.0f}")
                    continue

                result = self.process_region(
                    block.bbox,
                    region_type=block.block_type.name if block.block_type else "unknown"
                )

                # Add block metadata
                result.block_type = block.block_type.name if block.block_type else "unknown"
                result.reading_order = block.reading_order
                result.column_index = block.column_index

                if result.success:
                    block_results.append(result)

            if not block_results:
                logger.warning(f"[BlockImageEngine] No valid blocks, falling back to full page")
                return self._fallback_to_full_page()

            # 3. Sort by reading order
            block_results.sort(key=lambda r: r.reading_order)

            # 4. Generate combined output
            combined_output = self._generate_combined_output(block_results)

            return MultiBlockResult(
                page_num=self.page_num,
                strategy_used=BlockStrategy.SEMANTIC_BLOCKS,
                block_results=block_results,
                success=True,
                combined_output=combined_output,
                total_blocks=len(layout_result.blocks),
                successful_blocks=len(block_results),
                failed_blocks=len(layout_result.blocks) - len(block_results)
            )

        except Exception as e:
            logger.error(f"[BlockImageEngine] Semantic block processing failed: {e}")
            return self._fallback_to_full_page()

    def process_page_as_grid_blocks(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None
    ) -> MultiBlockResult:
        """
        Processes the page by dividing into a grid.

        Used as fallback when semantic analysis fails.

        Args:
            rows: Number of rows (default: config.GRID_ROWS)
            cols: Number of columns (default: config.GRID_COLS)

        Returns:
            MultiBlockResult object
        """
        rows = rows or self.config.GRID_ROWS
        cols = cols or self.config.GRID_COLS

        try:
            cell_width = self.page_width / cols
            cell_height = self.page_height / rows

            block_results: List[BlockImageResult] = []
            reading_order = 0

            # Process left→right, top→bottom order
            for row in range(rows):
                for col in range(cols):
                    x0 = col * cell_width
                    y0 = row * cell_height
                    x1 = (col + 1) * cell_width
                    y1 = (row + 1) * cell_height

                    bbox = (x0, y0, x1, y1)

                    # Check if region is empty
                    if self.config.SKIP_EMPTY_BLOCKS and self._is_empty_region(bbox):
                        continue

                    result = self.process_region(bbox, region_type="grid_cell")
                    result.reading_order = reading_order
                    result.column_index = col

                    if result.success:
                        block_results.append(result)
                        reading_order += 1

            combined_output = self._generate_combined_output(block_results)

            return MultiBlockResult(
                page_num=self.page_num,
                strategy_used=BlockStrategy.GRID_BLOCKS,
                block_results=block_results,
                success=len(block_results) > 0,
                combined_output=combined_output,
                total_blocks=rows * cols,
                successful_blocks=len(block_results),
                failed_blocks=rows * cols - len(block_results)
            )

        except Exception as e:
            logger.error(f"[BlockImageEngine] Grid processing failed: {e}")
            return self._fallback_to_full_page()

    def process_page_smart(self) -> MultiBlockResult:
        """
        ★ Smart processing: Automatically selects optimal strategy.

        1. First try semantic block splitting
        2. If fails or results are poor, use grid splitting
        3. If still fails, fall back to full page imaging

        Returns:
            MultiBlockResult object
        """
        # 1. Try semantic block splitting
        result = self.process_page_as_semantic_blocks()

        if result.success and result.successful_blocks >= 1:
            # Use if sufficient blocks detected
            if result.successful_blocks >= 2 or result.block_results:
                logger.info(f"[BlockImageEngine] Smart: Using semantic blocks "
                           f"({result.successful_blocks} blocks)")
                return result

        # 2. If semantic analysis results are poor, use grid splitting
        logger.info(f"[BlockImageEngine] Smart: Semantic blocks insufficient, trying grid")

        # Determine grid based on column count
        try:
            from libs.core.processor.pdf_helpers.pdf_layout_block_detector import (
                LayoutBlockDetector,
            )
            detector = LayoutBlockDetector(self.page, self.page_num)
            layout_result = detector.detect()

            cols = max(2, layout_result.column_count)
            rows = max(2, int(self.page_height / self.page_width * cols))

            result = self.process_page_as_grid_blocks(rows=rows, cols=cols)

            if result.success and result.successful_blocks >= 2:
                logger.info(f"[BlockImageEngine] Smart: Using grid {rows}x{cols} "
                           f"({result.successful_blocks} blocks)")
                return result
        except Exception:
            pass

        # 3. Full page fallback
        logger.info(f"[BlockImageEngine] Smart: Falling back to full page")
        return self._fallback_to_full_page()

    def _fallback_to_full_page(self) -> MultiBlockResult:
        """Full page imaging fallback."""
        result = self.process_full_page()

        return MultiBlockResult(
            page_num=self.page_num,
            strategy_used=BlockStrategy.FULL_PAGE,
            block_results=[result] if result.success else [],
            success=result.success,
            combined_output=result.image_tag if result.success else "",
            total_blocks=1,
            successful_blocks=1 if result.success else 0,
            failed_blocks=0 if result.success else 1
        )

    def _is_empty_region(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if region is empty (mostly white)."""
        try:
            image_bytes, _, _ = self._render_region(bbox)
            if not image_bytes:
                return False

            # Analyze with PIL
            img = Image.open(io.BytesIO(image_bytes))

            # Calculate white pixel ratio
            if img.mode != 'RGB':
                img = img.convert('RGB')

            pixels = list(img.getdata())
            total_pixels = len(pixels)

            if total_pixels == 0:
                return True

            # Count nearly white pixels (R, G, B all > 240)
            white_pixels = sum(1 for p in pixels if p[0] > 240 and p[1] > 240 and p[2] > 240)
            white_ratio = white_pixels / total_pixels

            return white_ratio >= self.config.EMPTY_THRESHOLD

        except Exception:
            return False

    def _generate_combined_output(self, block_results: List[BlockImageResult]) -> str:
        """
        Converts block results to combined output string.

        Each block is arranged in reading order,
        with appropriate markup based on block type.
        """
        if not block_results:
            return ""

        output_parts = []

        for result in block_results:
            if not result.success or not result.image_tag:
                continue

            # Context hint based on block type
            block_type = result.block_type or "unknown"

            if block_type == "HEADER":
                output_parts.append(f"<!-- Page Header -->\n{result.image_tag}")
            elif block_type == "FOOTER":
                output_parts.append(f"<!-- Page Footer -->\n{result.image_tag}")
            elif block_type == "TABLE":
                output_parts.append(f"<!-- Table -->\n{result.image_tag}")
            elif block_type in ("IMAGE_WITH_CAPTION", "STANDALONE_IMAGE"):
                output_parts.append(f"<!-- Figure -->\n{result.image_tag}")
            elif block_type == "ADVERTISEMENT":
                output_parts.append(f"<!-- Advertisement -->\n{result.image_tag}")
            elif block_type == "SIDEBAR":
                output_parts.append(f"<!-- Sidebar -->\n{result.image_tag}")
            else:
                # General content block (ARTICLE, COLUMN_BLOCK, etc.)
                output_parts.append(result.image_tag)

        return "\n".join(output_parts)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'BlockStrategy',
    'BlockImageConfig',
    'BlockImageResult',
    'MultiBlockResult',
    'BlockImageEngine',
]
