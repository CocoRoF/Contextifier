# contextifier_new/handlers/pdf_plus/_block_image_engine.py
"""
PDF Plus — Block Image Engine.

Renders regions of a PDF page as high-resolution PNG images.
Three cascading strategies:

1. **Semantic blocks** — use ``LayoutBlockDetector`` to split the page
   into meaningful blocks (articles, images, tables) and render each.
2. **Grid blocks** — divide the page into an NxM grid and render cells.
3. **Full page** — render the entire page as a single image (last resort).

All rendered images are de-duplicated by MD5 hash and saved through the
``image_service``.
"""

from __future__ import annotations

import hashlib
import io
import logging
from typing import Any, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    BlockProcessingStrategy,
    BlockResult,
    MultiBlockResult,
    PdfPlusConfig,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment,misc]


class BlockImageEngine:
    """
    Render complex page regions as individual PNG images.

    Typical usage::

        engine = BlockImageEngine(page, page_num, image_service=svc)
        result = engine.process_page_smart()
        combined_text = _combine_block_output(result)
    """

    def __init__(
        self,
        page: Any,
        page_num: int,
        *,
        image_service: Any = None,
    ) -> None:
        self.page = page
        self.page_num = page_num
        self._image_service = image_service

        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height

        self._processed_hashes: set[str] = set()

    # ─────────────────────────────────────────────────────────────────────
    # Single-region rendering
    # ─────────────────────────────────────────────────────────────────────

    def process_region(
        self,
        bbox: Tuple[float, float, float, float],
        *,
        region_type: str = "region",
    ) -> BlockResult:
        """
        Render *bbox* at high DPI, de-duplicate, save via image_service.

        Returns a ``BlockResult`` with the ``[Image:…]`` tag on success.
        """
        try:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < CFG.BLOCK_MIN_REGION_WIDTH or h < CFG.BLOCK_MIN_REGION_HEIGHT:
                return BlockResult(success=False, bbox=bbox, error="too small")

            img_bytes, dpi, size = self._render(bbox)
            if img_bytes is None:
                return BlockResult(success=False, bbox=bbox, error="render failed")

            md5 = hashlib.md5(img_bytes).hexdigest()
            if md5 in self._processed_hashes:
                return BlockResult(success=False, bbox=bbox, error="duplicate")
            self._processed_hashes.add(md5)

            tag = self._save_image(img_bytes)
            if not tag:
                return BlockResult(success=False, bbox=bbox, error="save failed")

            return BlockResult(
                success=True,
                image_tag=tag,
                bbox=bbox,
                block_type=region_type,
            )
        except Exception as exc:
            logger.error(
                "[BlockImageEngine] Error processing %s on page %d: %s",
                region_type, self.page_num + 1, exc,
            )
            return BlockResult(success=False, bbox=bbox, error=str(exc))

    def process_full_page(self) -> BlockResult:
        """Render entire page as one image."""
        return self.process_region(
            (0, 0, self.page_width, self.page_height),
            region_type="full_page",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Multi-block strategies
    # ─────────────────────────────────────────────────────────────────────

    def process_page_smart(self) -> MultiBlockResult:
        """
        Automatically choose the best strategy:

        1. Try semantic blocks via ``LayoutBlockDetector``.
        2. Fall back to grid blocks if semantic is too coarse.
        3. Fall back to full-page image if all else fails.
        """
        result = self._try_semantic()
        if result.success and result.successful_blocks >= 1:
            return result

        result = self._try_grid()
        if result.success and result.successful_blocks >= 1:
            return result

        return self._fullpage_fallback()

    def process_semantic_blocks(self) -> MultiBlockResult:
        """Force semantic-block strategy."""
        return self._try_semantic()

    def process_grid_blocks(
        self,
        rows: int = 2,
        cols: int = 2,
    ) -> MultiBlockResult:
        """Force grid-block strategy."""
        return self._try_grid(rows=rows, cols=cols)

    # ─────────────────────────────────────────────────────────────────────
    # Strategy implementations
    # ─────────────────────────────────────────────────────────────────────

    def _try_semantic(self) -> MultiBlockResult:
        try:
            from contextifier.handlers.pdf_plus._layout_block_detector import (
                LayoutBlockDetector,
            )
            detector = LayoutBlockDetector(self.page, self.page_num)
            layout = detector.detect()

            if not layout.blocks:
                return self._fullpage_fallback()

            results: List[BlockResult] = []
            for block in layout.blocks:
                area = (
                    (block.bbox[2] - block.bbox[0])
                    * (block.bbox[3] - block.bbox[1])
                )
                if area < CFG.BLOCK_MIN_AREA:
                    continue
                br = self.process_region(
                    block.bbox,
                    region_type=block.block_type.name if block.block_type else "unknown",
                )
                if br.success:
                    results.append(br)

            if not results:
                return self._fullpage_fallback()

            return MultiBlockResult(
                success=True,
                strategy_used=BlockProcessingStrategy.SEMANTIC_BLOCKS,
                block_results=results,
                total_blocks=len(layout.blocks),
                successful_blocks=len(results),
            )
        except Exception as exc:
            logger.warning(
                "[BlockImageEngine] Semantic strategy failed on page %d: %s",
                self.page_num + 1, exc,
            )
            return self._fullpage_fallback()

    def _try_grid(
        self, rows: int = 2, cols: int = 2,
    ) -> MultiBlockResult:
        try:
            cell_w = self.page_width / cols
            cell_h = self.page_height / rows

            results: List[BlockResult] = []
            total = rows * cols
            for r in range(rows):
                for c in range(cols):
                    bbox = (
                        c * cell_w,
                        r * cell_h,
                        (c + 1) * cell_w,
                        (r + 1) * cell_h,
                    )
                    if self._is_empty_region(bbox):
                        continue
                    br = self.process_region(bbox, region_type="grid_cell")
                    if br.success:
                        results.append(br)

            return MultiBlockResult(
                success=len(results) > 0,
                strategy_used=BlockProcessingStrategy.GRID_BLOCKS,
                block_results=results,
                total_blocks=total,
                successful_blocks=len(results),
            )
        except Exception as exc:
            logger.warning(
                "[BlockImageEngine] Grid strategy failed on page %d: %s",
                self.page_num + 1, exc,
            )
            return self._fullpage_fallback()

    def _fullpage_fallback(self) -> MultiBlockResult:
        br = self.process_full_page()
        return MultiBlockResult(
            success=br.success,
            strategy_used=BlockProcessingStrategy.FULL_PAGE,
            block_results=[br] if br.success else [],
            total_blocks=1,
            successful_blocks=1 if br.success else 0,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Low-level rendering
    # ─────────────────────────────────────────────────────────────────────

    def _render(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[Optional[bytes], int, Tuple[int, int]]:
        """Render *bbox* to PNG bytes using PyMuPDF."""
        try:
            pad = 5
            x0 = max(0, bbox[0] - pad)
            y0 = max(0, bbox[1] - pad)
            x1 = min(self.page_width, bbox[2] + pad)
            y1 = min(self.page_height, bbox[3] + pad)

            clip = fitz.Rect(x0, y0, x1, y1)
            dpi = CFG.BLOCK_DPI  # 300
            max_dim = max(x1 - x0, y1 - y0)
            expected = max_dim * dpi / 72
            if expected > CFG.BLOCK_MAX_IMAGE_SIZE:
                dpi = int(CFG.BLOCK_MAX_IMAGE_SIZE * 72 / max_dim)

            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = self.page.get_pixmap(matrix=mat, clip=clip)
            return pix.tobytes("png"), dpi, (pix.width, pix.height)
        except Exception as exc:
            logger.error("[BlockImageEngine] Render error: %s", exc)
            return None, 0, (0, 0)

    def _save_image(self, img_bytes: bytes) -> Optional[str]:
        """Save through the image service, return ``[Image:…]`` tag."""
        if self._image_service is None:
            return None
        try:
            return self._image_service.save_and_tag(img_bytes)
        except Exception:
            try:
                return self._image_service.save_image(img_bytes)
            except Exception as exc:
                logger.error("[BlockImageEngine] Save error: %s", exc)
                return None

    def _is_empty_region(
        self, bbox: Tuple[float, float, float, float],
    ) -> bool:
        """Check if a region is nearly all white (skip rendering)."""
        try:
            img_bytes, _, _ = self._render(bbox)
            if img_bytes is None or Image is None:
                return False
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            pixels = list(img.getdata())
            if not pixels:
                return True
            thr = CFG.BLOCK_EMPTY_PIXEL_MIN  # 240
            whites = sum(
                1 for p in pixels if p[0] > thr and p[1] > thr and p[2] > thr
            )
            return whites / len(pixels) >= CFG.BLOCK_EMPTY_THRESHOLD  # 0.95
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Helper: combine block results into a single text
# ─────────────────────────────────────────────────────────────────────────────


def combine_block_output(result: MultiBlockResult) -> str:
    """
    Concatenate image tags from a ``MultiBlockResult`` with HTML
    comments indicating block type.
    """
    parts: list[str] = []
    for br in result.block_results:
        if not br.success or not br.image_tag:
            continue
        bt = br.block_type or "content"
        if bt in ("HEADER", "FOOTER", "TABLE"):
            parts.append(f"<!-- {bt} -->\n{br.image_tag}")
        elif bt in ("IMAGE_WITH_CAPTION", "STANDALONE_IMAGE"):
            parts.append(f"<!-- Figure -->\n{br.image_tag}")
        elif bt == "ADVERTISEMENT":
            parts.append(f"<!-- Ad -->\n{br.image_tag}")
        else:
            parts.append(br.image_tag)
    return "\n".join(parts)


__all__ = [
    "BlockImageEngine",
    "combine_block_output",
]
