"""
XlsxContentExtractor — Stage 4: Extract content from XLSX workbooks.

Iterates all sheets and for each one:
1. Detects layout regions using ``_layout.object_detect()``
2. Converts regions to tables (Markdown/HTML via ``_table``)
3. Processes charts from pre-extracted ZIP data
4. Extracts images from pre-extracted ZIP data (per-sheet via openpyxl)
5. Extracts textbox content from pre-extracted drawing XML

The output is organized with ``[Sheet: name]`` section tags.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.types import (
    ChartData,
    ChartSeries,
    PreprocessedData,
    TableData,
)
from contextifier_new.handlers.xlsx._layout import (
    LayoutRange,
    layout_detect_range,
    object_detect,
)
from contextifier_new.handlers.xlsx._table import (
    convert_sheet_to_text,
    convert_region_to_table,
)

logger = logging.getLogger(__name__)


class XlsxContentExtractor(BaseContentExtractor):
    """
    Content extractor for XLSX files.

    Uses openpyxl Workbook + pre-extracted ZIP resources to
    extract text, tables, charts, and images from all sheets.
    """

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from all sheets in the XLSX workbook.

        Returns text organized with ``[Sheet: name]`` tags and
        table content in Markdown or HTML format.
        """
        wb = preprocessed.content
        if wb is None:
            return ""

        charts = preprocessed.resources.get("charts", [])
        images = preprocessed.resources.get("images", {})
        textboxes = preprocessed.resources.get("textboxes", {})

        parts: List[str] = []
        chart_index = 0
        image_hashes: Set[str] = set()

        for sheet_name in wb.sheetnames:
            try:
                ws = wb[sheet_name]
            except Exception:
                continue

            sheet_parts: List[str] = []

            # Sheet tag
            sheet_tag = self._make_sheet_tag(sheet_name)
            if sheet_tag:
                sheet_parts.append(sheet_tag)

            # Detect and convert table regions
            layout = layout_detect_range(ws)
            if layout is not None:
                regions = object_detect(ws, layout)
                if regions:
                    table_texts = []
                    for region in regions:
                        text = convert_sheet_to_text(ws, region)
                        if text.strip():
                            table_texts.append(text)

                    if len(table_texts) > 1:
                        for i, tt in enumerate(table_texts, 1):
                            sheet_parts.append(f"[Table {i}]")
                            sheet_parts.append(tt)
                    elif table_texts:
                        sheet_parts.append(table_texts[0])

            # Process per-sheet charts
            try:
                ws_charts = getattr(ws, "_charts", [])
                for _ in ws_charts:
                    if chart_index < len(charts):
                        chart_text = self._format_chart(charts[chart_index])
                        if chart_text:
                            sheet_parts.append(chart_text)
                        chart_index += 1
            except Exception as exc:
                logger.debug("Error processing charts for sheet %s: %s", sheet_name, exc)

            # Process per-sheet images
            try:
                sheet_image_tags = self._extract_sheet_images(ws, images, image_hashes)
                if sheet_image_tags:
                    sheet_parts.extend(sheet_image_tags)
            except Exception as exc:
                logger.debug("Error processing images for sheet %s: %s", sheet_name, exc)

            # Textboxes
            sheet_textboxes = textboxes.get(sheet_name, [])
            for tb_text in sheet_textboxes:
                if tb_text.strip():
                    sheet_parts.append(tb_text)

            if sheet_parts:
                parts.append("\n\n".join(sheet_parts))

        # Append any remaining charts not matched to sheets
        while chart_index < len(charts):
            chart_text = self._format_chart(charts[chart_index])
            if chart_text:
                parts.append(chart_text)
            chart_index += 1

        return "\n\n".join(parts).strip()

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """Extract all tables from all sheets."""
        wb = preprocessed.content
        if wb is None:
            return []

        tables: List[TableData] = []

        for sheet_name in wb.sheetnames:
            try:
                ws = wb[sheet_name]
                layout = layout_detect_range(ws)
                if layout is None:
                    continue

                regions = object_detect(ws, layout)
                for region in regions:
                    table = convert_region_to_table(ws, region)
                    if table is not None:
                        tables.append(table)
            except Exception as exc:
                logger.debug("Error extracting tables from %s: %s", sheet_name, exc)

        return tables

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        """Extract all images from the workbook."""
        if self._image_service is None:
            return []

        images: Dict[str, bytes] = preprocessed.resources.get("images", {})
        if not images:
            return []

        tags: List[str] = []
        processed: Set[str] = set()

        for name, img_data in images.items():
            if not img_data:
                continue

            content_hash = hashlib.md5(img_data).hexdigest()[:16]
            if content_hash in processed:
                continue

            try:
                # Create a clean name from the ZIP path
                clean_name = name.replace("xl/media/", "").replace("/", "_")
                tag = self._image_service.save_and_tag(
                    image_bytes=img_data,
                    custom_name=f"excel_{clean_name}",
                )
                if tag:
                    tags.append(tag)
                    processed.add(content_hash)
            except Exception as exc:
                logger.debug("Failed to save image %s: %s", name, exc)

        return tags

    def extract_charts(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[ChartData]:
        """Extract all charts as ChartData objects."""
        charts_raw = preprocessed.resources.get("charts", [])
        result: List[ChartData] = []

        for chart_dict in charts_raw:
            try:
                series = [
                    ChartSeries(
                        name=s.get("name"),
                        values=s.get("values", []),
                    )
                    for s in chart_dict.get("series", [])
                ]
                result.append(ChartData(
                    chart_type=chart_dict.get("chart_type", "Unknown"),
                    title=chart_dict.get("title", ""),
                    categories=chart_dict.get("categories", []),
                    series=series,
                ))
            except Exception:
                pass

        return result

    def get_format_name(self) -> str:
        return "xlsx"

    # ── Internal helpers ──────────────────────────────────────────────────

    def _make_sheet_tag(self, sheet_name: str) -> Optional[str]:
        """Generate a ``[Sheet: name]`` tag using TagService."""
        if self._tag_service is not None:
            try:
                return self._tag_service.make_sheet_tag(sheet_name)
            except Exception:
                pass
        return f"[Sheet: {sheet_name}]"

    def _format_chart(self, chart_dict: dict) -> Optional[str]:
        """Format a chart dict using ChartService or fallback text."""
        chart_type = chart_dict.get("chart_type", "Chart")
        title = chart_dict.get("title", "")

        if self._chart_service is not None:
            try:
                series = [
                    ChartSeries(
                        name=s.get("name"),
                        values=s.get("values", []),
                    )
                    for s in chart_dict.get("series", [])
                ]
                chart_data = ChartData(
                    chart_type=chart_type,
                    title=title,
                    categories=chart_dict.get("categories", []),
                    series=series,
                )
                return self._chart_service.format_chart(chart_data)
            except Exception:
                pass

        # Fallback
        label = f"[Chart: {chart_type}"
        if title:
            label += f" - {title}"
        label += "]"
        return label

    def _extract_sheet_images(
        self,
        ws: object,
        all_images: Dict[str, bytes],
        processed_hashes: Set[str],
    ) -> List[str]:
        """Extract images belonging to a specific sheet."""
        if self._image_service is None:
            return []

        tags: List[str] = []

        try:
            # Use openpyxl's internal _images attribute
            ws_images = getattr(ws, "_images", [])
            for img in ws_images:
                try:
                    img_data = img._data()
                    if not img_data:
                        continue

                    content_hash = hashlib.md5(img_data).hexdigest()[:16]
                    if content_hash in processed_hashes:
                        continue

                    tag = self._image_service.save_and_tag(
                        image_bytes=img_data,
                        custom_name=f"excel_sheet_img",
                    )
                    if tag:
                        tags.append(tag)
                        processed_hashes.add(content_hash)
                except Exception:
                    pass
        except Exception:
            pass

        return tags


__all__ = ["XlsxContentExtractor"]
