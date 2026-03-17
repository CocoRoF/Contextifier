"""
PptxContentExtractor — Stage 4: Extract text, tables, images, charts.

Iterates slides in order, and within each slide processes shapes
in visual reading order (sorted by ``top`` then ``left``).

Shape types handled:
- ``has_table``   → HTML table or simple text
- Picture shapes  → image save via ImageService
- ``has_chart``   → chart formatting via ChartService or pre-extracted data
- ``text_frame``  → bullet/numbering-aware text extraction
- Group shapes    → recursive processing of sub-shapes

Slide notes are appended after the slide content.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.types import (
    ChartData,
    ChartSeries,
    PreprocessedData,
    TableData,
)

from contextifier_new.handlers.pptx._constants import ElementType, SlideElement
from contextifier_new.handlers.pptx._bullet import extract_text_with_bullets
from contextifier_new.handlers.pptx._table import (
    is_simple_table,
    extract_simple_text,
    extract_table,
)

logger = logging.getLogger(__name__)


class PptxContentExtractor(BaseContentExtractor):
    """
    Content extractor for PPTX files.

    Processes all slides, extracting text, tables, images, and charts
    in visual reading order.
    """

    # ── Main extraction ───────────────────────────────────────────────────

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from all slides with slide tags and positioned elements.

        Returns:
            Full text with ``[Slide:N]`` tags, tables as HTML,
            image tags, chart blocks, and slide notes.
        """
        prs = self._get_presentation(preprocessed)
        if prs is None:
            return ""

        charts_by_slide: Dict[int, List[Any]] = preprocessed.resources.get(
            "charts_by_slide", {}
        )

        parts: List[str] = []
        processed_images: Set[str] = set()

        for slide_idx, slide in enumerate(prs.slides):
            # Slide tag
            slide_tag = self._make_slide_tag(slide_idx + 1)
            if slide_tag:
                parts.append(slide_tag)

            # Collect elements with positions
            elements: List[SlideElement] = []
            chart_queue = list(charts_by_slide.get(slide_idx, []))
            chart_ptr = 0

            for shape in slide.shapes:
                shape_elements, chart_ptr = self._process_shape(
                    shape,
                    slide_idx,
                    chart_queue,
                    chart_ptr,
                    processed_images,
                )
                elements.extend(shape_elements)

            # Sort by visual reading order (top → left)
            elements.sort(key=lambda e: e.sort_key)

            # Merge elements into text
            slide_text = _merge_elements(elements)
            if slide_text.strip():
                parts.append(slide_text.strip())
            else:
                parts.append("[Empty Slide]")

            # Slide notes
            notes = _extract_notes(slide)
            if notes:
                parts.append(f"[Notes]\n{notes}")

        result = "\n\n".join(parts)
        # Clean excessive whitespace
        import re
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """Extract all tables from all slides."""
        prs = self._get_presentation(preprocessed)
        if prs is None:
            return []

        tables: List[TableData] = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_table:
                    td = extract_table(shape.table)
                    if td.num_rows > 0:
                        tables.append(td)
                # Group shapes
                if hasattr(shape, "shapes"):
                    for sub in shape.shapes:
                        if hasattr(sub, "has_table") and sub.has_table:
                            td = extract_table(sub.table)
                            if td.num_rows > 0:
                                tables.append(td)
        return tables

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        """Extract and save all images from all slides."""
        prs = self._get_presentation(preprocessed)
        if prs is None:
            return []

        tags: List[str] = []
        processed: Set[str] = set()

        for slide_idx, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                tag = self._try_save_image(shape, slide_idx, processed)
                if tag:
                    tags.append(tag)
                # Group shapes
                if hasattr(shape, "shapes"):
                    for sub in shape.shapes:
                        tag = self._try_save_image(sub, slide_idx, processed)
                        if tag:
                            tags.append(tag)
        return tags

    def extract_charts(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[ChartData]:
        """Extract all charts from all slides as ``ChartData``."""
        prs = self._get_presentation(preprocessed)
        if prs is None:
            return []

        charts: List[ChartData] = []
        for slide in prs.slides:
            for shape in slide.shapes:
                cd = self._try_extract_chart_data(shape)
                if cd is not None:
                    charts.append(cd)
                if hasattr(shape, "shapes"):
                    for sub in shape.shapes:
                        cd = self._try_extract_chart_data(sub)
                        if cd is not None:
                            charts.append(cd)
        return charts

    def get_format_name(self) -> str:
        return "pptx"

    # ── Shape processing ──────────────────────────────────────────────────

    def _process_shape(
        self,
        shape: Any,
        slide_idx: int,
        chart_queue: List[Any],
        chart_ptr: int,
        processed_images: Set[str],
    ) -> Tuple[List[SlideElement], int]:
        """
        Process a single shape and return content elements + updated chart pointer.
        """
        elements: List[SlideElement] = []
        position = _get_position(shape)
        shape_id = getattr(shape, "shape_id", id(shape))

        # Table
        if shape.has_table:
            content = self._format_table_shape(shape.table)
            if content:
                etype = ElementType.TEXT if is_simple_table(shape.table) else ElementType.TABLE
                elements.append(SlideElement(
                    element_type=etype,
                    content=content,
                    position=position,
                    shape_id=shape_id,
                ))
            return elements, chart_ptr

        # Picture (image)
        if _is_picture(shape):
            tag = self._save_image_shape(shape, slide_idx, processed_images)
            if tag:
                elements.append(SlideElement(
                    element_type=ElementType.IMAGE,
                    content=tag,
                    position=position,
                    shape_id=shape_id,
                ))
            return elements, chart_ptr

        # Chart
        if getattr(shape, "has_chart", False):
            content = self._format_chart(chart_queue, chart_ptr)
            chart_ptr += 1
            if content:
                elements.append(SlideElement(
                    element_type=ElementType.CHART,
                    content=content,
                    position=position,
                    shape_id=shape_id,
                ))
            return elements, chart_ptr

        # Text frame (with bullets)
        if hasattr(shape, "text_frame") and shape.text_frame:
            text = extract_text_with_bullets(shape.text_frame)
            if text.strip():
                elements.append(SlideElement(
                    element_type=ElementType.TEXT,
                    content=text,
                    position=position,
                    shape_id=shape_id,
                ))
            return elements, chart_ptr

        # Group shape (recursive)
        if hasattr(shape, "shapes"):
            group_elems, chart_ptr = self._process_group(
                shape, slide_idx, chart_queue, chart_ptr, processed_images
            )
            elements.extend(group_elems)
            return elements, chart_ptr

        # Fallback: plain text
        if hasattr(shape, "text") and shape.text.strip():
            elements.append(SlideElement(
                element_type=ElementType.TEXT,
                content=shape.text.strip(),
                position=position,
                shape_id=shape_id,
            ))

        return elements, chart_ptr

    def _process_group(
        self,
        group_shape: Any,
        slide_idx: int,
        chart_queue: List[Any],
        chart_ptr: int,
        processed_images: Set[str],
    ) -> Tuple[List[SlideElement], int]:
        """Recursively process shapes inside a group."""
        elements: List[SlideElement] = []
        try:
            for sub_shape in group_shape.shapes:
                sub_elems, chart_ptr = self._process_shape(
                    sub_shape, slide_idx, chart_queue, chart_ptr, processed_images
                )
                elements.extend(sub_elems)
        except Exception as exc:
            logger.debug("Error processing group shape: %s", exc)
        return elements, chart_ptr

    # ── Table formatting ──────────────────────────────────────────────────

    def _format_table_shape(self, table: Any) -> str:
        """Format a table shape → HTML or plain text."""
        if is_simple_table(table):
            return extract_simple_text(table)

        table_data = extract_table(table)
        if table_data.num_rows == 0:
            return ""

        # 1×1 collapse
        if table_data.num_rows == 1 and table_data.num_cols == 1:
            if table_data.rows and table_data.rows[0]:
                return table_data.rows[0][0].content
            return ""

        # Single-column collapse
        if table_data.num_cols == 1:
            items = [
                row[0].content for row in table_data.rows
                if row and row[0].content
            ]
            return "\n\n".join(items) if items else ""

        # Use TableService if available
        if self._table_service is not None:
            try:
                return self._table_service.format_table(table_data)
            except Exception:
                pass

        # Fallback: HTML
        return _table_to_html(table_data)

    # ── Image processing ──────────────────────────────────────────────────

    def _save_image_shape(
        self,
        shape: Any,
        slide_idx: int,
        processed_images: Set[str],
    ) -> Optional[str]:
        """Save image from a picture shape and return tag."""
        if self._image_service is None:
            return None

        try:
            if not hasattr(shape, "image"):
                return None

            image_data = shape.image.blob
            if not image_data:
                return None

            # Deduplicate by content hash
            import hashlib
            content_hash = hashlib.md5(image_data).hexdigest()[:16]
            if content_hash in processed_images:
                return None

            shape_id = getattr(shape, "shape_id", 0)
            custom_name = f"pptx_slide{slide_idx + 1}_shape{shape_id}"

            tag = self._image_service.save_and_tag(
                image_bytes=image_data,
                custom_name=custom_name,
            )
            if tag:
                processed_images.add(content_hash)
                return tag

        except Exception as exc:
            logger.debug("Failed to save image shape: %s", exc)
        return None

    def _try_save_image(
        self,
        shape: Any,
        slide_idx: int,
        processed: Set[str],
    ) -> Optional[str]:
        """Try to save image from a picture shape (for ``extract_images``)."""
        if not _is_picture(shape):
            return None
        return self._save_image_shape(shape, slide_idx, processed)

    # ── Chart formatting ──────────────────────────────────────────────────

    def _format_chart(
        self,
        chart_queue: List[Any],
        chart_ptr: int,
    ) -> str:
        """Format a chart from the pre-extracted queue."""
        if chart_ptr >= len(chart_queue):
            return "[Chart]"

        chart = chart_queue[chart_ptr]
        chart_type = chart.get("chart_type", "Chart")
        title = chart.get("title")

        # Use ChartService if available
        if self._chart_service is not None:
            try:
                categories = chart.get("categories", [])
                series_raw = chart.get("series", [])
                series = [
                    ChartSeries(name=s.get("name"), values=s.get("values", []))
                    for s in series_raw
                ]
                cd = ChartData(
                    chart_type=chart_type,
                    title=title,
                    categories=categories,
                    series=series,
                )
                return self._chart_service.format_chart(cd)
            except Exception:
                pass

        # Fallback
        if title:
            return f"[Chart: {chart_type} - {title}]"
        return f"[Chart: {chart_type}]"

    def _try_extract_chart_data(self, shape: Any) -> Optional[ChartData]:
        """Extract ChartData from a chart shape."""
        if not getattr(shape, "has_chart", False):
            return None
        try:
            chart = shape.chart
            title = None
            try:
                if chart.has_title and chart.chart_title and chart.chart_title.has_text_frame:
                    title = chart.chart_title.text_frame.text.strip() or None
            except Exception:
                pass

            chart_type = "Chart"
            try:
                if hasattr(chart, "chart_type"):
                    t = str(chart.chart_type).split(".")[-1].split(" ")[0]
                    chart_type = t.replace("_", " ").title()
            except Exception:
                pass

            categories: List[str] = []
            try:
                if chart.plots:
                    for plot in chart.plots:
                        if hasattr(plot, "categories") and plot.categories:
                            categories = [str(c) for c in plot.categories]
                            break
            except Exception:
                pass

            series: List[ChartSeries] = []
            try:
                for idx, s in enumerate(chart.series):
                    name = f"Series {idx + 1}"
                    try:
                        if s.name:
                            name = str(s.name)
                    except Exception:
                        pass
                    vals: List[Any] = []
                    try:
                        if s.values:
                            vals = list(s.values)
                    except Exception:
                        pass
                    series.append(ChartSeries(name=name, values=vals))
            except Exception:
                pass

            return ChartData(
                chart_type=chart_type,
                title=title,
                categories=categories,
                series=series,
            )
        except Exception:
            return None

    # ── Slide tags ────────────────────────────────────────────────────────

    def _make_slide_tag(self, slide_number: int) -> Optional[str]:
        """Generate a ``[Slide:N]`` tag using TagService."""
        if self._tag_service is not None:
            try:
                return self._tag_service.make_slide_tag(slide_number)
            except Exception:
                pass
        return f"[Slide:{slide_number}]"

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_presentation(preprocessed: PreprocessedData) -> Any:
        """Resolve python-pptx Presentation from PreprocessedData."""
        for attr in ("content", "raw_content"):
            obj = getattr(preprocessed, attr, None)
            if obj is not None and hasattr(obj, "slides"):
                return obj
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_position(shape: Any) -> Tuple[int, int, int, int]:
    """Get (left, top, width, height) in EMU from a shape."""
    try:
        left = shape.left if hasattr(shape, "left") and shape.left else 0
        top = shape.top if hasattr(shape, "top") and shape.top else 0
        width = shape.width if hasattr(shape, "width") and shape.width else 0
        height = shape.height if hasattr(shape, "height") and shape.height else 0
        return (left, top, width, height)
    except Exception:
        return (0, 0, 0, 0)


def _is_picture(shape: Any) -> bool:
    """Check whether a shape is an image/picture."""
    try:
        from pptx.enum.shapes import MSO_SHAPE_TYPE
        if hasattr(shape, "shape_type") and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            return True
    except Exception:
        pass
    if hasattr(shape, "image"):
        try:
            _ = shape.image
            return True
        except Exception:
            pass
    return False


def _extract_notes(slide: Any) -> Optional[str]:
    """Extract slide notes text."""
    try:
        if hasattr(slide, "notes_slide") and slide.notes_slide:
            nf = slide.notes_slide.notes_text_frame
            if nf:
                text = nf.text.strip()
                if text:
                    return text
    except Exception:
        pass
    return None


def _merge_elements(elements: List[SlideElement]) -> str:
    """Merge sorted slide elements into a single text block."""
    if not elements:
        return ""

    parts: List[str] = []
    for elem in elements:
        if elem.element_type == ElementType.TABLE:
            parts.append("\n" + elem.content + "\n")
        elif elem.element_type == ElementType.IMAGE:
            parts.append("\n" + elem.content + "\n")
        elif elem.element_type == ElementType.CHART:
            parts.append("\n" + elem.content + "\n")
        elif elem.element_type == ElementType.TEXT:
            parts.append(elem.content + "\n")
    return "".join(parts)


def _table_to_html(table_data: TableData) -> str:
    """Fallback HTML table generation."""
    lines: List[str] = ["<table border='1'>"]
    for row in table_data.rows:
        lines.append("  <tr>")
        for cell in row:
            tag = "th" if cell.is_header else "td"
            attrs = ""
            if cell.row_span > 1:
                attrs += f" rowspan='{cell.row_span}'"
            if cell.col_span > 1:
                attrs += f" colspan='{cell.col_span}'"
            content = cell.content.replace("\n", "<br>") if cell.content else ""
            # Escape HTML
            content = (
                content.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            ) if content and "<br>" not in content else content
            lines.append(f"    <{tag}{attrs}>{content}</{tag}>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


__all__ = ["PptxContentExtractor"]
