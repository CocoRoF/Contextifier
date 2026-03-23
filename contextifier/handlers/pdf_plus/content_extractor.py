# contextifier/handlers/pdf_plus/content_extractor.py
"""
PdfPlusContentExtractor — adaptive, multi-strategy PDF extraction.

For every page the engine:
    1. Runs the **ComplexityAnalyzer** to score the page.
    2. Selects one of four strategies based on the score:

       =========  ====================================================
       Strategy   Processing
       =========  ====================================================
       TEXT       tables (TableProcessor) + text blocks + images
       HYBRID     tables + text blocks + images + vector-text OCR
       BLOCK_OCR  tables (quality-gated) + block-image for complex
       FULL_OCR   block-image engine (semantic → grid → full-page)
       =========  ====================================================

    3. Merges heterogeneous ``PageElement`` objects into page text via
       the **element merger**.

Key design principles (from v1.0):
    * Text extraction is preferred over image conversion.
    * PyMuPDF detection is trusted over alternatives.
    * Multiple fallback layers prevent silent data loss.
    * Korean-optimised OCR (``kor+eng``, PUA detection).
"""

from __future__ import annotations

import logging
from typing import Any, List, Set, Tuple

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    PreprocessedData,
    TableData,
)

from contextifier.handlers.pdf_plus._types import (
    ElementType,
    PageElement,
    PdfPlusConfig,
    ProcessingStrategy,
    TableQuality,
)
from contextifier.handlers.pdf_plus._complexity_analyzer import ComplexityAnalyzer
from contextifier.handlers.pdf_plus._table_processor import TableProcessor
from contextifier.handlers.pdf_plus._table_quality_analyzer import (
    TableQualityAnalyzer,
)
from contextifier.handlers.pdf_plus._text_extractor import extract_text_blocks
from contextifier.handlers.pdf_plus._image_extractor import extract_images
from contextifier.handlers.pdf_plus._element_merger import merge_page_elements
from contextifier.handlers.pdf_plus._vector_text_ocr import VectorTextOCREngine
from contextifier.handlers.pdf_plus._block_image_engine import (
    BlockImageEngine,
    combine_block_output,
)


logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class PdfPlusContentExtractor(BaseContentExtractor):
    """
    Full-featured adaptive PDF content extractor.

    Implements all five ``BaseContentExtractor`` methods.
    Receives a ``fitz.Document`` through ``PreprocessedData.content``.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        image_service: Any = None,
        tag_service: Any = None,
        chart_service: Any = None,
        table_service: Any = None,
    ) -> None:
        """
        Args:
            image_service: Service for saving raster images.
            tag_service:   Service for page/slide/sheet tags.
            chart_service: (unused — kept for interface compat.)
            table_service: Service for table formatting.
        """
        super().__init__(
            image_service=image_service,
            tag_service=tag_service,
            chart_service=chart_service,
            table_service=table_service,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Abstract method implementations
    # ─────────────────────────────────────────────────────────────────────

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        doc = self._unwrap_doc(preprocessed)
        if doc is None:
            return ""

        file_data = self._get_file_data(preprocessed)
        page_count = doc.page_count
        processed_images: Set[int] = set()
        parts: list[str] = []

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            page_text = self._process_page(
                doc, page, page_num, processed_images, file_data,
            )
            if page_text.strip():
                tag = self._make_page_tag(page_num + 1)
                parts.append(f"{tag}\n{page_text}")

        return "\n\n".join(parts)

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        doc = self._unwrap_doc(preprocessed)
        if doc is None:
            return []

        file_data = self._get_file_data(preprocessed)
        all_tables: list[TableData] = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            tp = TableProcessor(page, page_num, file_data)
            processed = tp.process()
            for pt in processed:
                if pt.data:
                    all_tables.append(
                        TableData(
                            rows=pt.data,
                            num_rows=len(pt.data),
                            num_cols=max((len(r) for r in pt.data), default=0),
                            metadata={"page_number": page_num + 1},
                        )
                    )
        return all_tables

    def get_format_name(self) -> str:
        return "pdf"

    # ─────────────────────────────────────────────────────────────────────
    # Core per-page processing
    # ─────────────────────────────────────────────────────────────────────

    def _process_page(
        self,
        doc: Any,
        page: Any,
        page_num: int,
        processed_images: Set[int],
        file_data: bytes,
    ) -> str:
        """Process one page according to its complexity."""
        try:
            # 1. Complexity analysis
            analyzer = ComplexityAnalyzer(page, page_num)
            complexity = analyzer.analyze()
            strategy = complexity.recommended_strategy

            logger.debug(
                "[PdfPlus] Page %d: strategy=%s  score=%.2f  level=%s",
                page_num + 1,
                strategy.name,
                complexity.overall_score,
                complexity.overall_complexity.name,
            )

            # 2. Dispatch to strategy
            if strategy == ProcessingStrategy.TEXT_EXTRACTION:
                return self._strategy_text(
                    doc, page, page_num, processed_images, file_data,
                )
            elif strategy == ProcessingStrategy.HYBRID:
                return self._strategy_hybrid(
                    doc, page, page_num, processed_images, file_data,
                    complexity,
                )
            elif strategy == ProcessingStrategy.BLOCK_IMAGE_OCR:
                return self._strategy_block_ocr(
                    doc, page, page_num, processed_images, file_data,
                    complexity,
                )
            else:  # FULL_PAGE_OCR
                return self._strategy_full_ocr(
                    page, page_num, processed_images,
                )

        except Exception as exc:
            logger.error(
                "[PdfPlus] Error on page %d, falling back to text: %s",
                page_num + 1, exc,
            )
            return self._strategy_text(
                doc, page, page_num, processed_images, file_data,
            )

    # ─────────────────────────────────────────────────────────────────────
    # TEXT_EXTRACTION strategy
    # ─────────────────────────────────────────────────────────────────────

    def _strategy_text(
        self,
        doc: Any,
        page: Any,
        page_num: int,
        processed_images: Set[int],
        file_data: bytes,
    ) -> str:
        elements: list[PageElement] = []

        # tables
        table_bboxes: list[Tuple[float, float, float, float]] = []
        tp = TableProcessor(page, page_num, file_data)
        for pt in tp.process():
            if pt.html:
                elements.append(PageElement(
                    element_type=ElementType.TABLE,
                    content=pt.html,
                    bbox=pt.bbox,
                    page_num=page_num,
                    table_data=pt.data,
                    confidence=pt.confidence,
                ))
                table_bboxes.append(pt.bbox)
            if pt.annotations:
                for ann in pt.annotations:
                    elements.append(PageElement(
                        element_type=ElementType.ANNOTATION,
                        content=ann.text,
                        bbox=ann.bbox,
                        page_num=page_num,
                    ))

        # text blocks
        elements.extend(
            extract_text_blocks(page, page_num, table_bboxes)
        )

        # images
        elements.extend(
            extract_images(
                page, page_num,
                table_bboxes=table_bboxes,
                image_service=self._image_service,
            )
        )

        return merge_page_elements(elements)

    # ─────────────────────────────────────────────────────────────────────
    # HYBRID strategy
    # ─────────────────────────────────────────────────────────────────────

    def _strategy_hybrid(
        self,
        doc: Any,
        page: Any,
        page_num: int,
        processed_images: Set[int],
        file_data: bytes,
        complexity: Any,
    ) -> str:
        elements: list[PageElement] = []

        # tables (with quality gating — image-render poor ones)
        table_bboxes: list[Tuple[float, float, float, float]] = []
        tp = TableProcessor(page, page_num, file_data)
        for pt in tp.process():
            if pt.html:
                elements.append(PageElement(
                    element_type=ElementType.TABLE,
                    content=pt.html,
                    bbox=pt.bbox,
                    page_num=page_num,
                    table_data=pt.data,
                    confidence=pt.confidence,
                ))
                table_bboxes.append(pt.bbox)
            if pt.annotations:
                for ann in pt.annotations:
                    elements.append(PageElement(
                        element_type=ElementType.ANNOTATION,
                        content=ann.text,
                        bbox=ann.bbox,
                        page_num=page_num,
                    ))

        # text blocks
        elements.extend(
            extract_text_blocks(page, page_num, table_bboxes)
        )

        # images
        elements.extend(
            extract_images(
                page, page_num,
                table_bboxes=table_bboxes,
                image_service=self._image_service,
            )
        )

        # vector text OCR (for outlined text)
        try:
            engine = VectorTextOCREngine(page, page_num)
            for region in engine.detect_and_ocr():
                if region.ocr_text.strip():
                    elements.append(PageElement(
                        element_type=ElementType.TEXT,
                        content=region.ocr_text,
                        bbox=region.bbox,
                        page_num=page_num,
                        confidence=region.confidence,
                    ))
        except Exception as exc:
            logger.debug(
                "[PdfPlus] Vector OCR skipped on page %d: %s",
                page_num + 1, exc,
            )

        # block-image complex regions (if significant complex area)
        try:
            complex_regions = complexity.complex_regions
            if complex_regions and self._image_service is not None:
                engine = BlockImageEngine(
                    page, page_num, image_service=self._image_service,
                )
                for bbox in complex_regions:
                    br = engine.process_region(bbox, region_type="complex")
                    if br.success and br.image_tag:
                        elements.append(PageElement(
                            element_type=ElementType.IMAGE,
                            content=br.image_tag,
                            bbox=br.bbox,
                            page_num=page_num,
                        ))
        except Exception as exc:
            logger.debug(
                "[PdfPlus] Block-image for complex regions skipped: %s", exc,
            )

        return merge_page_elements(elements)

    # ─────────────────────────────────────────────────────────────────────
    # BLOCK_IMAGE_OCR strategy
    # ─────────────────────────────────────────────────────────────────────

    def _strategy_block_ocr(
        self,
        doc: Any,
        page: Any,
        page_num: int,
        processed_images: Set[int],
        file_data: bytes,
        complexity: Any,
    ) -> str:
        elements: list[PageElement] = []

        # quality-gate tables: only text-extract high-quality ones
        table_bboxes: list[Tuple[float, float, float, float]] = []
        tqa = TableQualityAnalyzer(page, page_num)
        for tqr in tqa.analyze_page_tables():
            if tqr.quality in (
                TableQuality.EXCELLENT,
                TableQuality.GOOD,
                TableQuality.MODERATE,
            ):
                # text-extract this table
                tp = TableProcessor(page, page_num, file_data)
                for pt in tp.process():
                    if pt.html:
                        elements.append(PageElement(
                            element_type=ElementType.TABLE,
                            content=pt.html,
                            bbox=pt.bbox,
                            page_num=page_num,
                            table_data=pt.data,
                        ))
                        table_bboxes.append(pt.bbox)
                break  # processor handles all tables at once

        # remaining content → block-image
        if self._image_service is not None:
            engine = BlockImageEngine(
                page, page_num, image_service=self._image_service,
            )
            result = engine.process_page_smart()
            if result.success:
                combined = combine_block_output(result)
                if combined.strip():
                    elements.append(PageElement(
                        element_type=ElementType.IMAGE,
                        content=combined,
                        bbox=(0, 0, page.rect.width, page.rect.height),
                        page_num=page_num,
                    ))

        return merge_page_elements(elements)

    # ─────────────────────────────────────────────────────────────────────
    # FULL_PAGE_OCR strategy
    # ─────────────────────────────────────────────────────────────────────

    def _strategy_full_ocr(
        self,
        page: Any,
        page_num: int,
        processed_images: Set[int],
    ) -> str:
        if self._image_service is None:
            # cannot render without image service — fall back to raw text
            text = page.get_text("text").strip()
            return text

        engine = BlockImageEngine(
            page, page_num, image_service=self._image_service,
        )
        result = engine.process_page_smart()
        if result.success:
            return combine_block_output(result)
        return page.get_text("text").strip()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_doc(preprocessed: PreprocessedData) -> Any:
        """Get fitz.Document from PreprocessedData."""
        content = preprocessed.content
        if content is None:
            return None
        # could be doc directly or wrapped in resources
        if hasattr(content, "page_count"):
            return content
        resources = preprocessed.resources
        if resources and "document" in resources:
            return resources["document"]
        return None

    @staticmethod
    def _get_file_data(preprocessed: PreprocessedData) -> bytes:
        """Get raw PDF bytes from PreprocessedData (for pdfplumber)."""
        raw = preprocessed.raw_content
        if isinstance(raw, bytes):
            return raw
        return b""

    def _make_page_tag(self, page_number: int) -> str:
        if self._tag_service is not None:
            try:
                return self._tag_service.page_tag(page_number)
            except Exception:
                pass
        return f"[Page {page_number}]"


__all__ = ["PdfPlusContentExtractor"]
