# tests/unit/handlers/test_pptx_group_shapes.py
"""
P2-5: PPTX group shape recursive extraction tests.

Verifies that extract_tables(), extract_images(), and extract_charts()
correctly recurse into nested group shapes (not just one level deep).
"""

from __future__ import annotations

from unittest import mock

import pytest

from contextifier.handlers.pptx.content_extractor import PptxContentExtractor
from contextifier.types import ChartData, ChartSeries, PreprocessedData, TableData, TableCell


def _make_shape(
    *,
    has_table: bool = False,
    has_chart: bool = False,
    is_picture: bool = False,
    sub_shapes: list | None = None,
    text: str = "",
    shape_id: int = 1,
):
    """Create a mock PPTX shape."""
    s = mock.MagicMock()
    s.has_table = has_table
    s.has_chart = has_chart
    s.shape_id = shape_id
    s.text = text

    if is_picture:
        s.shape_properties = mock.MagicMock()
        # Make _is_picture() detect it
        s.image = mock.MagicMock()
        s.image.blob = b"\x89PNG fake"
    else:
        # Remove image attribute so _is_picture returns False
        if hasattr(s, "image"):
            del s.image

    if sub_shapes is not None:
        s.shapes = sub_shapes
    else:
        del s.shapes

    # Position
    s.top = 0
    s.left = 0

    if has_table:
        # Create a minimal table mock compatible with pptx _table.extract_table()
        tbl = mock.MagicMock()
        tbl.rows = [mock.MagicMock(), mock.MagicMock()]
        tbl.columns = [mock.MagicMock(), mock.MagicMock()]

        def make_cell(text):
            c = mock.MagicMock()
            c.text = text
            c.is_merge_origin = False
            c.is_spanned = mock.MagicMock(return_value=False)
            c.span_height = 1
            c.span_width = 1
            return c

        cells = {
            (0, 0): make_cell("A"),
            (0, 1): make_cell("B"),
            (1, 0): make_cell("C"),
            (1, 1): make_cell("D"),
        }
        tbl.cell = lambda r, c: cells[(r, c)]
        s.table = tbl

    if has_chart:
        chart = mock.MagicMock()
        chart.has_title = False
        chart.chart_title = None
        chart.chart_type = "BAR_CLUSTERED"
        chart.plots = []
        chart.series = []
        s.chart = chart

    return s


def _make_presentation(shapes_per_slide: list[list]):
    """Create a mock Presentation with given shaped slides."""
    prs = mock.MagicMock()
    slides = []
    for shapes in shapes_per_slide:
        slide = mock.MagicMock()
        slide.shapes = shapes
        slides.append(slide)
    prs.slides = slides
    return prs


def _make_preprocessed(prs):
    return PreprocessedData(
        content=prs,
        raw_content=b"",
        resources={"charts_by_slide": {}},
    )


class TestRecursiveTableExtraction:
    """Tables inside nested group shapes must be found."""

    def test_table_in_top_level(self):
        shape = _make_shape(has_table=True, shape_id=1)
        prs = _make_presentation([[shape]])
        extractor = PptxContentExtractor()
        tables = extractor.extract_tables(_make_preprocessed(prs))
        assert len(tables) >= 1

    def test_table_in_group(self):
        inner = _make_shape(has_table=True, shape_id=2)
        group = _make_shape(sub_shapes=[inner], shape_id=10)
        prs = _make_presentation([[group]])
        extractor = PptxContentExtractor()
        tables = extractor.extract_tables(_make_preprocessed(prs))
        assert len(tables) >= 1

    def test_table_in_nested_group(self):
        """Table 2 levels deep in group shapes."""
        inner = _make_shape(has_table=True, shape_id=3)
        mid_group = _make_shape(sub_shapes=[inner], shape_id=20)
        outer_group = _make_shape(sub_shapes=[mid_group], shape_id=30)
        prs = _make_presentation([[outer_group]])
        extractor = PptxContentExtractor()
        tables = extractor.extract_tables(_make_preprocessed(prs))
        assert len(tables) >= 1


class TestRecursiveChartExtraction:
    """Charts inside nested group shapes must be found."""

    def test_chart_in_group(self):
        inner = _make_shape(has_chart=True, shape_id=5)
        group = _make_shape(sub_shapes=[inner], shape_id=50)
        prs = _make_presentation([[group]])
        extractor = PptxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed(prs))
        assert len(charts) >= 1

    def test_chart_in_nested_group(self):
        """Chart 2 levels deep."""
        inner = _make_shape(has_chart=True, shape_id=6)
        mid = _make_shape(sub_shapes=[inner], shape_id=60)
        outer = _make_shape(sub_shapes=[mid], shape_id=70)
        prs = _make_presentation([[outer]])
        extractor = PptxContentExtractor()
        charts = extractor.extract_charts(_make_preprocessed(prs))
        assert len(charts) >= 1


class TestRecursiveImageExtraction:
    """Images inside nested group shapes must be found."""

    def test_image_in_group(self):
        inner = _make_shape(is_picture=True, shape_id=7)
        group = _make_shape(sub_shapes=[inner], shape_id=80)
        prs = _make_presentation([[group]])

        mock_img_service = mock.MagicMock()
        mock_img_service.save_and_tag.return_value = "[Image:test.png]"

        extractor = PptxContentExtractor(image_service=mock_img_service)
        images = extractor.extract_images(_make_preprocessed(prs))
        assert len(images) >= 1

    def test_image_in_nested_group(self):
        """Image 2 levels deep."""
        inner = _make_shape(is_picture=True, shape_id=8)
        mid = _make_shape(sub_shapes=[inner], shape_id=90)
        outer = _make_shape(sub_shapes=[mid], shape_id=100)
        prs = _make_presentation([[outer]])

        mock_img_service = mock.MagicMock()
        mock_img_service.save_and_tag.return_value = "[Image:deep.png]"

        extractor = PptxContentExtractor(image_service=mock_img_service)
        images = extractor.extract_images(_make_preprocessed(prs))
        assert len(images) >= 1


class TestDepthLimit:
    """Excessive nesting is capped at _MAX_GROUP_DEPTH."""

    def test_depth_limit_stops_recursion(self):
        """Recursion stops at _MAX_GROUP_DEPTH."""
        extractor = PptxContentExtractor()

        # Build a chain deeper than _MAX_GROUP_DEPTH
        max_depth = extractor._MAX_GROUP_DEPTH
        shape = _make_shape(has_chart=True, shape_id=999)
        for i in range(max_depth + 5):
            shape = _make_shape(sub_shapes=[shape], shape_id=1000 + i)

        charts: list[ChartData] = []
        extractor._collect_charts(shape, charts)
        # Chart at the very bottom should NOT be found
        assert len(charts) == 0
