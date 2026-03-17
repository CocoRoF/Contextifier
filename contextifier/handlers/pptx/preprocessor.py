"""
PptxPreprocessor — Stage 2: python-pptx Presentation → PreprocessedData.

Wraps the ``Presentation`` in ``PreprocessedData``, computes summary
statistics (slide count, dimensions), and pre-extracts charts from
slides so the ContentExtractor can consume them in slide order.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.errors import PreprocessingError

logger = logging.getLogger(__name__)


class PptxPreprocessor(BasePreprocessor):
    """
    Preprocessor for PPTX files.

    Light-weight pass-through that stores the Presentation and
    pre-extracts chart data in slide order for the ContentExtractor.
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """
        Wrap Presentation in ``PreprocessedData``.

        Args:
            converted_data: ``pptx.Presentation`` from the Converter.

        Returns:
            PreprocessedData with:
            - ``content``: the Presentation (primary source)
            - ``raw_content``: the Presentation (unchanged)
            - ``resources["charts"]``: list of pre-extracted chart dicts per slide
            - ``properties``: slide_count, slide_width, slide_height
        """
        prs = converted_data
        if prs is None:
            raise PreprocessingError(
                "Received None as converted data",
                stage="preprocess",
                handler="pptx",
            )

        # Summary statistics
        slide_count = len(prs.slides) if hasattr(prs, "slides") else 0
        slide_width = prs.slide_width if hasattr(prs, "slide_width") else None
        slide_height = prs.slide_height if hasattr(prs, "slide_height") else None

        # Pre-extract charts per slide
        charts_by_slide: Dict[int, List[Any]] = {}
        try:
            charts_by_slide = self._extract_charts(prs)
        except Exception as exc:
            logger.debug("Chart pre-extraction failed: %s", exc)

        return PreprocessedData(
            content=prs,
            raw_content=prs,
            encoding="utf-8",
            resources={
                "charts_by_slide": charts_by_slide,
            },
            properties={
                "slide_count": slide_count,
                "slide_width": slide_width,
                "slide_height": slide_height,
            },
        )

    def get_format_name(self) -> str:
        return "pptx"

    # ── Chart pre-extraction ──────────────────────────────────────────────

    @staticmethod
    def _extract_charts(prs: Any) -> Dict[int, List[Any]]:
        """
        Pre-extract chart data from all slides.

        Returns:
            Dict mapping slide_index (0-based) to a list of chart objects.
            Each chart object is a dict with keys:
            ``chart_type``, ``title``, ``categories``, ``series``.
        """
        charts_by_slide: Dict[int, List[Any]] = {}

        for slide_idx, slide in enumerate(prs.slides):
            slide_charts: List[Any] = []

            for shape in slide.shapes:
                chart_data = _try_extract_chart(shape)
                if chart_data is not None:
                    slide_charts.append(chart_data)

                # Check group shapes
                if hasattr(shape, "shapes"):
                    for sub_shape in shape.shapes:
                        chart_data = _try_extract_chart(sub_shape)
                        if chart_data is not None:
                            slide_charts.append(chart_data)

            if slide_charts:
                charts_by_slide[slide_idx] = slide_charts

        return charts_by_slide


def _try_extract_chart(shape: Any) -> Optional[Dict[str, Any]]:
    """
    Try to extract chart data from a single shape.

    Returns a dict with ``chart_type``, ``title``, ``categories``, ``series``
    or None if the shape is not a chart.
    """
    if not getattr(shape, "has_chart", False):
        return None

    try:
        chart = shape.chart
        result: Dict[str, Any] = {
            "chart_type": _get_chart_type(chart),
            "title": _get_chart_title(chart),
            "categories": _get_categories(chart),
            "series": _get_series(chart),
        }
        return result
    except Exception as exc:
        logger.debug("Failed to extract chart from shape: %s", exc)
        return None


def _get_chart_type(chart: Any) -> str:
    """Get chart type as a readable string."""
    try:
        if hasattr(chart, "chart_type"):
            type_str = str(chart.chart_type)
            type_name = type_str.split(".")[-1].split(" ")[0]
            return type_name.replace("_", " ").title()
    except Exception:
        pass
    return "Chart"


def _get_chart_title(chart: Any) -> Optional[str]:
    """Get chart title text."""
    try:
        if chart.has_title and chart.chart_title:
            if chart.chart_title.has_text_frame:
                text = chart.chart_title.text_frame.text
                if text:
                    return text.strip()
    except Exception:
        pass
    return None


def _get_categories(chart: Any) -> List[str]:
    """Get category labels from chart plots."""
    try:
        if hasattr(chart, "plots") and chart.plots:
            for plot in chart.plots:
                if hasattr(plot, "categories") and plot.categories:
                    return [str(c) for c in plot.categories]
    except Exception:
        pass
    return []


def _get_series(chart: Any) -> List[Dict[str, Any]]:
    """Get series data from chart."""
    series_list: List[Dict[str, Any]] = []
    try:
        for idx, series in enumerate(chart.series):
            name = f"Series {idx + 1}"
            try:
                if hasattr(series, "name") and series.name:
                    name = str(series.name)
            except Exception:
                pass

            values: List[Any] = []
            try:
                if hasattr(series, "values") and series.values:
                    values = list(series.values)
            except Exception:
                pass

            series_list.append({"name": name, "values": values})
    except Exception:
        pass
    return series_list


__all__ = ["PptxPreprocessor"]
