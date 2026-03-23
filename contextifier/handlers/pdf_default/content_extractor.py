# contextifier/handlers/pdf_default/content_extractor.py
"""
PdfDefaultContentExtractor — simple page-by-page PDF extraction.

Strategy:
    For each page:
        1. Extract text via ``page.get_text("text")``
        2. Detect tables via PyMuPDF ``page.find_tables()``
        3. Convert tables to HTML (multi-col) or plain text (single-col)
        4. Extract embedded images via ``page.get_images()``
        5. Assemble: page tag + tables + text + images

This is deliberately simple — no complexity analysis, no OCR fallback,
no block imaging.  Use ``pdf_plus`` for the full pipeline.
"""

from __future__ import annotations

import html as html_mod
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    ExtractionResult,
    PreprocessedData,
    TableData,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _bbox_overlap_ratio(inner: Tuple, outer: Tuple) -> float:
    """Fraction of *inner* that overlaps with *outer*."""
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


def _is_inside_any(bbox: Tuple, targets: List[Tuple], threshold: float = 0.5) -> bool:
    return any(_bbox_overlap_ratio(bbox, t) >= threshold for t in targets)


def _escape(text: str) -> str:
    return html_mod.escape(text, quote=False)


# ── Simple table → HTML ──────────────────────────────────────────────────────


def _table_to_html(data: List[List[Optional[str]]]) -> str:
    """Convert 2D table data to an HTML ``<table>``."""
    if not data:
        return ""

    rows_out: list[str] = []
    for ri, row in enumerate(data):
        tag = "th" if ri == 0 else "td"
        cells = "".join(
            f"<{tag}>{_escape(cell or '')}</{tag}>"
            for cell in row
        )
        rows_out.append(f"<tr>{cells}</tr>")

    return "<table>" + "".join(rows_out) + "</table>"


def _table_to_text(data: List[List[Optional[str]]]) -> str:
    """Single-column table → plain newline-joined text."""
    parts: list[str] = []
    for row in data:
        text = (row[0] or "").strip() if row else ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


# ── Find image position ──────────────────────────────────────────────────────


def _find_image_bbox(
    page: Any,
    xref: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Return the bbox of an image identified by *xref* on *page*."""
    try:
        for info in page.get_image_info(xrefs=True):
            if info.get("xref") == xref:
                bbox = info.get("bbox")
                if bbox:
                    return tuple(bbox)  # type: ignore[return-value]
    except Exception:
        pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Content Extractor
# ═════════════════════════════════════════════════════════════════════════════


class PdfDefaultContentExtractor(BaseContentExtractor):
    """
    Simple / default PDF content extractor.

    Uses only PyMuPDF for text, tables, and images — no pdfplumber,
    no OCR, no block imaging, no complexity analysis.
    """

    # ── extract_text ─────────────────────────────────────────────────────

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        doc = self._get_doc(preprocessed)
        if doc is None:
            return ""

        page_count = doc.page_count
        processed_images: Set[int] = set()
        parts: list[str] = []

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            page_text = self._process_page(
                doc, page, page_num, processed_images,
            )
            if page_text.strip():
                tag = self._make_page_tag(page_num + 1)
                parts.append(f"{tag}\n{page_text}")

        return "\n\n".join(parts)

    # ── extract_tables ───────────────────────────────────────────────────

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """Return structured table data for the whole document."""
        doc = self._get_doc(preprocessed)
        if doc is None:
            return []

        all_tables: list[TableData] = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            tables = self._detect_tables(page)
            for tdata, _bbox in tables:
                all_tables.append(
                    TableData(
                        data=tdata,
                        page_number=page_num + 1,
                    )
                )
        return all_tables

    # ── format name ──────────────────────────────────────────────────────

    def get_format_name(self) -> str:
        return "pdf"

    # ═════════════════════════════════════════════════════════════════════
    # Internal
    # ═════════════════════════════════════════════════════════════════════

    def _get_doc(self, preprocessed: PreprocessedData) -> Any:
        """Unwrap the fitz.Document from PreprocessedData."""
        doc = preprocessed.content
        if hasattr(doc, "page_count"):
            return doc
        # Fallback to resources
        return (preprocessed.resources or {}).get("document")

    def _make_page_tag(self, page_number: int) -> str:
        if self._tag_service is not None:
            return self._tag_service.page_tag(page_number)
        return f"[Page Number: {page_number}]"

    # ── Per-page processing ──────────────────────────────────────────────

    def _process_page(
        self,
        doc: Any,
        page: Any,
        page_num: int,
        processed_images: Set[int],
    ) -> str:
        """Process a single page: tables → text → images → merge."""
        elements: list[Tuple[float, str]] = []   # (y_pos, content)

        # 1. Tables
        tables = self._detect_tables(page)
        table_bboxes = [bbox for _, bbox in tables]

        for tdata, tbbox in tables:
            cols = max((len(r) for r in tdata), default=0)
            if cols <= 1:
                content = _table_to_text(tdata)
            else:
                content = _table_to_html(tdata)
            if content.strip():
                elements.append((tbbox[1], content))

        # 2. Text (excluding table regions)
        text_elements = self._extract_text_blocks(page, table_bboxes)
        elements.extend(text_elements)

        # 3. Images
        image_elements = self._extract_images(
            doc, page, page_num, processed_images, table_bboxes,
        )
        elements.extend(image_elements)

        # Sort by Y position and join
        elements.sort(key=lambda e: e[0])
        return "\n\n".join(content for _, content in elements)

    # ── Table detection ──────────────────────────────────────────────────

    def _detect_tables(
        self, page: Any,
    ) -> List[Tuple[List[List[Optional[str]]], Tuple]]:
        """Detect tables on *page* via PyMuPDF find_tables()."""
        results: list[Tuple[List[List[Optional[str]]], Tuple]] = []
        try:
            tabs = page.find_tables()
            for tab in tabs:
                data = tab.extract()
                if not data or len(data) < 2:
                    continue
                bbox = tuple(tab.bbox)
                results.append((data, bbox))
        except Exception as exc:
            logger.warning("Table detection failed: %s", exc)
        return results

    # ── Text extraction ──────────────────────────────────────────────────

    def _extract_text_blocks(
        self,
        page: Any,
        table_bboxes: List[Tuple],
    ) -> List[Tuple[float, str]]:
        """Extract text blocks, skipping those inside tables."""
        elements: list[Tuple[float, str]] = []
        try:
            blocks = page.get_text("dict", sort=True).get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:  # text only
                    continue
                bbox = block.get("bbox", (0, 0, 0, 0))
                if _is_inside_any(bbox, table_bboxes, threshold=0.5):
                    continue
                lines: list[str] = []
                for line in block.get("lines", []):
                    spans_text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    )
                    if spans_text.strip():
                        lines.append(spans_text)
                text = "\n".join(lines)
                if text.strip():
                    elements.append((bbox[1], text))
        except Exception as exc:
            logger.warning("Text extraction failed: %s", exc)
            # Fallback: raw text
            raw = page.get_text("text")
            if raw.strip():
                elements.append((0.0, raw))
        return elements

    # ── Image extraction ─────────────────────────────────────────────────

    def _extract_images(
        self,
        doc: Any,
        page: Any,
        page_num: int,
        processed_images: Set[int],
        table_bboxes: List[Tuple],
        min_size: int = 50,
        min_area: int = 2500,
    ) -> List[Tuple[float, str]]:
        """Extract embedded images, save via image_service."""
        if self._image_service is None:
            return []

        elements: list[Tuple[float, str]] = []
        try:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in processed_images:
                    continue

                # Size filter
                width = img_info[2]
                height = img_info[3]
                if width < min_size or height < min_size:
                    continue
                if width * height < min_area:
                    continue

                # Position check — skip images inside tables
                img_bbox = _find_image_bbox(page, xref)
                if img_bbox and _is_inside_any(img_bbox, table_bboxes, threshold=0.7):
                    continue

                # Extract image bytes
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                    pix = None  # release
                except Exception:
                    continue

                # Save via ImageService
                tag = self._image_service.save_and_tag(
                    img_data,
                    custom_name=f"pdf_p{page_num + 1}_x{xref}.png",
                )
                if tag:
                    processed_images.add(xref)
                    y_pos = img_bbox[1] if img_bbox else 0.0
                    elements.append((y_pos, tag))
        except Exception as exc:
            logger.warning("Image extraction failed on page %d: %s", page_num + 1, exc)

        return elements


__all__ = ["PdfDefaultContentExtractor"]
