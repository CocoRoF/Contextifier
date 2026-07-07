# tests/unit/handlers/test_xls_images_charts.py
"""
Tests for XLS image extraction (OLE scanning) and chart sheet detection.

Covers:
- OLE image stream scanning
- Image format detection
- Chart sheet type detection
- Edge cases (no image service, no file_data, bad OLE)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from contextifier.handlers.xls.content_extractor import (
    XlsContentExtractor,
    _detect_image_format,
    _XL_CHART_SHEET,
)
from contextifier.types import PreprocessedData


# ═════════════════════════════════════════════════════════════════════════════
# Image format detection
# ═════════════════════════════════════════════════════════════════════════════

class TestImageFormatDetection:
    """Image signature detection works for all formats."""

    def test_png(self):
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert _detect_image_format(data) == "png"

    def test_jpeg(self):
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        assert _detect_image_format(data) == "jpeg"

    def test_gif87a(self):
        data = b"GIF87a" + b"\x00" * 100
        assert _detect_image_format(data) == "gif"

    def test_gif89a(self):
        data = b"GIF89a" + b"\x00" * 100
        assert _detect_image_format(data) == "gif"

    def test_bmp(self):
        data = b"BM" + b"\x00" * 100
        assert _detect_image_format(data) == "bmp"

    def test_tiff_le(self):
        data = b"II\x2a\x00" + b"\x00" * 100
        assert _detect_image_format(data) == "tiff"

    def test_tiff_be(self):
        data = b"MM\x00\x2a" + b"\x00" * 100
        assert _detect_image_format(data) == "tiff"

    def test_unknown(self):
        assert _detect_image_format(b"\x00\x01\x02\x03") is None

    def test_empty(self):
        assert _detect_image_format(b"") is None

    def test_too_short(self):
        assert _detect_image_format(b"\x89") is None


# ═════════════════════════════════════════════════════════════════════════════
# XLS image extraction via OLE
# ═════════════════════════════════════════════════════════════════════════════

class TestXlsImageExtraction:
    """OLE stream image scanning."""

    def test_no_image_service_returns_empty(self):
        preprocessed = PreprocessedData(
            content=None,
            resources={"file_data": b"\x00" * 100},
        )
        extractor = XlsContentExtractor()
        assert extractor.extract_images(preprocessed) == []

    def test_no_file_data_returns_empty(self):
        image_service = MagicMock()
        preprocessed = PreprocessedData(
            content=None,
            resources={},
        )
        extractor = XlsContentExtractor(image_service=image_service)
        assert extractor.extract_images(preprocessed) == []

    def test_invalid_ole_returns_empty(self):
        image_service = MagicMock()
        preprocessed = PreprocessedData(
            content=None,
            resources={"file_data": b"not an OLE file"},
        )
        extractor = XlsContentExtractor(image_service=image_service)
        assert extractor.extract_images(preprocessed) == []

    @patch("contextifier.handlers.xls.content_extractor.olefile")
    def test_ole_image_found_and_saved(self, mock_olefile_mod):
        """When OLE streams contain an image, save_and_tag is called."""
        # Set up mock OLE
        mock_ole = MagicMock()
        mock_olefile_mod.OleFileIO.return_value = mock_ole

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

        mock_ole.listdir.return_value = [
            ["Pictures", "image1.png"],
        ]
        mock_stream = MagicMock()
        mock_stream.read.return_value = png_data
        mock_ole.openstream.return_value = mock_stream

        image_service = MagicMock()
        image_service.save_and_tag.return_value = "[Image: xls_ole_abc.png]"

        preprocessed = PreprocessedData(
            content=None,
            resources={"file_data": b"\xd0\xcf\x11\xe0" + b"\x00" * 100},
        )
        extractor = XlsContentExtractor(image_service=image_service)
        tags = extractor.extract_images(preprocessed)

        assert len(tags) == 1
        assert "[Image:" in tags[0]
        image_service.save_and_tag.assert_called_once()

    @patch("contextifier.handlers.xls.content_extractor.olefile")
    def test_deduplication(self, mock_olefile_mod):
        """Identical images in different streams produce only one tag."""
        mock_ole = MagicMock()
        mock_olefile_mod.OleFileIO.return_value = mock_ole

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

        mock_ole.listdir.return_value = [
            ["Pictures", "img1"],
            ["Data", "img2"],
        ]
        mock_stream = MagicMock()
        mock_stream.read.return_value = png_data
        mock_ole.openstream.return_value = mock_stream

        image_service = MagicMock()
        image_service.save_and_tag.return_value = "[Image: dedup.png]"

        preprocessed = PreprocessedData(
            content=None,
            resources={"file_data": b"\xd0\xcf" + b"\x00" * 100},
        )
        extractor = XlsContentExtractor(image_service=image_service)
        extractor.extract_images(preprocessed)

        # Should only save once (dedup by md5)
        assert image_service.save_and_tag.call_count == 1


# ═════════════════════════════════════════════════════════════════════════════
# XLS chart sheet detection
# ═════════════════════════════════════════════════════════════════════════════

class TestXlsChartDetection:
    """Chart sheet type detection."""

    def test_chart_sheet_detected(self):
        book = MagicMock()
        book.nsheets = 3
        sheets = [MagicMock(), MagicMock(), MagicMock()]
        sheets[0].name = "Data"
        sheets[1].name = "Chart1"
        sheets[2].name = "Summary"
        book.sheet_by_index.side_effect = lambda i: sheets[i]
        book.sheet_type.side_effect = lambda i: [0, _XL_CHART_SHEET, 0][i]

        preprocessed = PreprocessedData(content=book, raw_content=book)
        extractor = XlsContentExtractor()
        charts = extractor.extract_charts(preprocessed)

        assert len(charts) == 1
        assert charts[0].title == "Chart1"
        assert charts[0].chart_type == "biff_chart_sheet"

    def test_no_chart_sheets(self):
        book = MagicMock()
        book.nsheets = 2
        sheets = [MagicMock(), MagicMock()]
        sheets[0].name = "Sheet1"
        sheets[1].name = "Sheet2"
        book.sheet_by_index.side_effect = lambda i: sheets[i]
        book.sheet_type.side_effect = lambda i: 0

        preprocessed = PreprocessedData(content=book, raw_content=book)
        extractor = XlsContentExtractor()
        charts = extractor.extract_charts(preprocessed)

        assert charts == []

    def test_none_book_returns_empty(self):
        preprocessed = PreprocessedData(content=None, raw_content=None)
        extractor = XlsContentExtractor()
        charts = extractor.extract_charts(preprocessed)
        assert charts == []
