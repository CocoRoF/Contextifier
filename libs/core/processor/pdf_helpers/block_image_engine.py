"""
Block Image Engine for PDF Handler V4 (고도화 버전)

복잡한 영역을 의미론적 블록 단위로 분할하여 이미지로 렌더링하고 로컬에 저장합니다.

=============================================================================
V4 고도화 핵심 개념:
=============================================================================
기존: 전체 페이지를 하나의 이미지로 업로드
개선: 페이지를 **의미론적/논리적 블록 단위**로 분할하여 각각 PNG로 저장

이를 통해:
1. LLM이 각 블록을 **개별적으로** 해석 가능
2. 해상도 문제 해결 (블록별로 고해상도 유지)
3. 읽기 순서 보존
4. 컨텍스트 분리 (광고/기사/표 구분)

=============================================================================
처리 전략:
=============================================================================
1. SEMANTIC_BLOCKS: 의미론적 블록 단위 분할 (권장)
   - LayoutBlockDetector로 블록 감지
   - 각 블록을 개별 이미지로 변환
   - 읽기 순서대로 [Image:path] 태그 생성

2. GRID_BLOCKS: 그리드 기반 분할 (폴백)
   - 페이지를 NxM 그리드로 분할
   - 각 그리드 셀을 개별 이미지로 변환

3. FULL_PAGE: 전체 페이지 이미지화 (최후 수단)
   - 기존 방식 유지

렌더링 설정:
- 기본 DPI: 300 (고해상도)
- 최대 이미지 크기: 4096px
- 이미지 포맷: PNG (무손실)
"""

import logging
import io
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto

import fitz
from PIL import Image

# 이미지 처리 모듈
from libs.core.functions.img_processor import ImageProcessor

logger = logging.getLogger(__name__)

# 모듈 레벨 이미지 프로세서
_image_processor = ImageProcessor(
    directory_path="temp/images",
    tag_prefix="[Image:",
    tag_suffix="]"
)


# ============================================================================
# Block Strategy Enum
# ============================================================================

class BlockStrategy(Enum):
    """블록 처리 전략"""
    SEMANTIC_BLOCKS = auto()  # 의미론적 블록 단위
    GRID_BLOCKS = auto()      # 그리드 기반 분할
    FULL_PAGE = auto()        # 전체 페이지


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BlockImageConfig:
    """블록 이미지 엔진 설정"""
    # 렌더링 설정
    DEFAULT_DPI: int = 300
    MAX_IMAGE_SIZE: int = 4096

    # 이미지 포맷
    IMAGE_FORMAT: str = "PNG"

    # 영역 설정
    REGION_PADDING: int = 5  # 영역 패딩 (pt)

    # 최소 크기 (이 이하는 무시)
    MIN_REGION_WIDTH: int = 80   # 상향 조정
    MIN_REGION_HEIGHT: int = 60  # 상향 조정

    # 블록 분할 전략
    PREFERRED_STRATEGY: str = "semantic"  # semantic, grid, full_page

    # 그리드 분할 설정 (GRID_BLOCKS 전략용)
    GRID_ROWS: int = 2
    GRID_COLS: int = 2

    # 블록 병합 설정
    MERGE_SMALL_BLOCKS: bool = True
    MIN_BLOCK_AREA: float = 15000.0  # 최소 블록 면적 (pt²) - 대폭 상향

    # 빈 블록 필터링
    SKIP_EMPTY_BLOCKS: bool = True
    EMPTY_THRESHOLD: float = 0.95  # 흰색 픽셀 비율이 이 이상이면 빈 블록


@dataclass
class BlockImageResult:
    """블록 이미지 처리 결과"""
    bbox: Tuple[float, float, float, float]

    # 이미지 정보
    image_size: Tuple[int, int]
    dpi: int

    # 이미지 경로
    image_path: Optional[str] = None

    # 인라인 태그 ([Image:{path}] 형태)
    image_tag: Optional[str] = None

    # 성공 여부
    success: bool = False
    error: Optional[str] = None

    # 블록 정보 (고도화)
    block_type: Optional[str] = None  # 블록 유형 (article, image, table 등)
    reading_order: int = 0            # 읽기 순서
    column_index: int = 0             # 컬럼 인덱스


@dataclass
class MultiBlockResult:
    """다중 블록 처리 결과"""
    page_num: int
    strategy_used: BlockStrategy

    # 개별 블록 결과들 (읽기 순서대로)
    block_results: List[BlockImageResult] = field(default_factory=list)

    # 전체 성공 여부
    success: bool = False

    # 통합된 텍스트 출력 (모든 [Image:...] 태그 포함)
    combined_output: str = ""

    # 통계
    total_blocks: int = 0
    successful_blocks: int = 0
    failed_blocks: int = 0


# ============================================================================
# Block Image Engine
# ============================================================================

class BlockImageEngine:
    """
    블록 이미지 엔진

    복잡한 영역을 이미지로 렌더링하고 로컬에 저장합니다.
    결과는 [image:{path}] 형태로 반환됩니다.
    """

    def __init__(
        self,
        page,
        page_num: int,
        config: Optional[BlockImageConfig] = None
    ):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed)
            config: 엔진 설정
        """
        self.page = page
        self.page_num = page_num
        self.config = config or BlockImageConfig()

        self.page_width = page.rect.width
        self.page_height = page.rect.height

        # 처리된 이미지 해시 (중복 방지)
        self._processed_hashes: set = set()

    def process_region(
        self,
        bbox: Tuple[float, float, float, float],
        region_type: str = "complex_region"
    ) -> BlockImageResult:
        """
        특정 영역을 이미지로 렌더링하고 로컬에 저장합니다.

        Args:
            bbox: 처리할 영역 (x0, y0, x1, y1)
            region_type: 영역 유형 (로깅용)

        Returns:
            BlockImageResult 객체 (image_path, image_tag 포함)
        """
        try:
            # 최소 크기 검증
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

            # 1. 영역 이미지 렌더링
            image_bytes, actual_dpi, image_size = self._render_region(bbox)

            if image_bytes is None:
                return BlockImageResult(
                    bbox=bbox,
                    image_size=(0, 0),
                    dpi=self.config.DEFAULT_DPI,
                    success=False,
                    error="Failed to render region"
                )

            # 2. 중복 체크
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

            # 3. 로컬에 저장 (ImageProcessor 사용)
            image_tag = _image_processor.save_image(image_bytes)

            if not image_tag:
                return BlockImageResult(
                    bbox=bbox,
                    image_size=image_size,
                    dpi=actual_dpi,
                    success=False,
                    error="Failed to save image"
                )

            # 경로 추출 (태그에서)
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
        전체 페이지를 이미지로 렌더링하고 로컬에 저장합니다.

        Args:
            region_type: 영역 유형 (로깅용)

        Returns:
            BlockImageResult 객체
        """
        bbox = (0, 0, self.page_width, self.page_height)
        return self.process_region(bbox, region_type)

    def process_regions(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        region_type: str = "complex_region"
    ) -> List[BlockImageResult]:
        """
        여러 영역을 처리합니다.

        Args:
            bboxes: 처리할 영역 목록
            region_type: 영역 유형 (로깅용)

        Returns:
            BlockImageResult 객체 목록
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
        영역을 이미지 바이트로 렌더링합니다.

        Args:
            bbox: 렌더링할 영역

        Returns:
            (이미지 바이트, 실제 DPI, (width, height))
        """
        try:
            # 패딩 적용
            padding = self.config.REGION_PADDING
            x0 = max(0, bbox[0] - padding)
            y0 = max(0, bbox[1] - padding)
            x1 = min(self.page_width, bbox[2] + padding)
            y1 = min(self.page_height, bbox[3] + padding)

            # 클립 영역 생성
            clip_rect = fitz.Rect(x0, y0, x1, y1)

            # DPI 계산 (최대 이미지 크기 고려)
            dpi = self.config.DEFAULT_DPI

            region_width = x1 - x0
            region_height = y1 - y0

            max_dim = max(region_width, region_height)
            expected_size = max_dim * dpi / 72

            if expected_size > self.config.MAX_IMAGE_SIZE:
                # DPI 조정
                dpi = int(self.config.MAX_IMAGE_SIZE * 72 / max_dim)

            # 매트릭스 생성 (줌 = DPI / 72)
            zoom = dpi / 72
            matrix = fitz.Matrix(zoom, zoom)

            # 렌더링
            pix = self.page.get_pixmap(matrix=matrix, clip=clip_rect)

            # PNG 바이트로 변환
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
        영역을 이미지 바이트로 렌더링합니다 (업로드 없이).

        Args:
            bbox: 렌더링할 영역

        Returns:
            이미지 바이트
        """
        image_bytes, _, _ = self._render_region(bbox)
        return image_bytes

    # ========================================================================
    # 고도화된 블록 처리 (V4)
    # ========================================================================

    def process_page_as_semantic_blocks(self) -> MultiBlockResult:
        """
        ★ 고도화된 처리: 페이지를 의미론적 블록 단위로 분할하여 처리합니다.

        기존 FULL_PAGE_OCR과 달리:
        1. LayoutBlockDetector로 의미론적 블록 감지
        2. 각 블록을 개별 이미지로 렌더링
        3. 읽기 순서대로 [Image:path] 태그 생성

        Returns:
            MultiBlockResult 객체 (모든 블록 결과 포함)
        """
        try:
            # 1. 레이아웃 블록 감지
            from libs.core.processor.pdf_helpers.layout_block_detector import (
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

            # 2. 각 블록을 개별 이미지로 처리
            block_results: List[BlockImageResult] = []

            for block in layout_result.blocks:
                # 너무 작은 블록 필터링 (면적 기준)
                # NOTE: 요소가 없어도 블록 영역 자체가 유효하면 처리
                if block.area < self.config.MIN_BLOCK_AREA:
                    logger.debug(f"[BlockImageEngine] Skipping small block: area={block.area:.0f}")
                    continue

                result = self.process_region(
                    block.bbox,
                    region_type=block.block_type.name if block.block_type else "unknown"
                )

                # 블록 메타데이터 추가
                result.block_type = block.block_type.name if block.block_type else "unknown"
                result.reading_order = block.reading_order
                result.column_index = block.column_index

                if result.success:
                    block_results.append(result)

            if not block_results:
                logger.warning(f"[BlockImageEngine] No valid blocks, falling back to full page")
                return self._fallback_to_full_page()

            # 3. 읽기 순서대로 정렬
            block_results.sort(key=lambda r: r.reading_order)

            # 4. 통합 출력 생성
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
        페이지를 그리드로 분할하여 처리합니다.

        의미론적 분석이 실패했을 때 폴백으로 사용합니다.

        Args:
            rows: 행 수 (기본값: config.GRID_ROWS)
            cols: 열 수 (기본값: config.GRID_COLS)

        Returns:
            MultiBlockResult 객체
        """
        rows = rows or self.config.GRID_ROWS
        cols = cols or self.config.GRID_COLS

        try:
            cell_width = self.page_width / cols
            cell_height = self.page_height / rows

            block_results: List[BlockImageResult] = []
            reading_order = 0

            # 좌→우, 상→하 순서로 처리
            for row in range(rows):
                for col in range(cols):
                    x0 = col * cell_width
                    y0 = row * cell_height
                    x1 = (col + 1) * cell_width
                    y1 = (row + 1) * cell_height

                    bbox = (x0, y0, x1, y1)

                    # 빈 영역인지 확인
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
        ★ 스마트 처리: 최적의 전략을 자동 선택합니다.

        1. 먼저 의미론적 블록 분할 시도
        2. 실패하거나 결과가 부실하면 그리드 분할
        3. 그래도 실패하면 전체 페이지 이미지화

        Returns:
            MultiBlockResult 객체
        """
        # 1. 의미론적 블록 분할 시도
        result = self.process_page_as_semantic_blocks()

        if result.success and result.successful_blocks >= 1:
            # 충분한 블록이 감지되었으면 사용
            if result.successful_blocks >= 2 or result.block_results:
                logger.info(f"[BlockImageEngine] Smart: Using semantic blocks "
                           f"({result.successful_blocks} blocks)")
                return result

        # 2. 의미론적 분석 결과가 부실하면 그리드 분할
        logger.info(f"[BlockImageEngine] Smart: Semantic blocks insufficient, trying grid")

        # 컬럼 수 기반으로 그리드 결정
        try:
            from libs.core.processor.pdf_helpers.layout_block_detector import (
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

        # 3. 전체 페이지 폴백
        logger.info(f"[BlockImageEngine] Smart: Falling back to full page")
        return self._fallback_to_full_page()

    def _fallback_to_full_page(self) -> MultiBlockResult:
        """전체 페이지 이미지화 폴백"""
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
        """영역이 비어있는지 확인 (흰색 위주인지)"""
        try:
            image_bytes, _, _ = self._render_region(bbox)
            if not image_bytes:
                return False

            # PIL로 분석
            img = Image.open(io.BytesIO(image_bytes))

            # 흰색 픽셀 비율 계산
            if img.mode != 'RGB':
                img = img.convert('RGB')

            pixels = list(img.getdata())
            total_pixels = len(pixels)

            if total_pixels == 0:
                return True

            # 거의 흰색인 픽셀 수 (R, G, B 모두 240 이상)
            white_pixels = sum(1 for p in pixels if p[0] > 240 and p[1] > 240 and p[2] > 240)
            white_ratio = white_pixels / total_pixels

            return white_ratio >= self.config.EMPTY_THRESHOLD

        except Exception:
            return False

    def _generate_combined_output(self, block_results: List[BlockImageResult]) -> str:
        """
        블록 결과들을 통합 출력 문자열로 변환합니다.

        각 블록은 읽기 순서대로 배치되며,
        블록 유형에 따라 적절한 마크업이 추가됩니다.
        """
        if not block_results:
            return ""

        output_parts = []

        for result in block_results:
            if not result.success or not result.image_tag:
                continue

            # 블록 유형에 따른 컨텍스트 힌트
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
                # 일반 콘텐츠 블록 (ARTICLE, COLUMN_BLOCK 등)
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
