# tests/unit/handlers/test_configurable_thresholds.py
"""
P2-8: Tests for configurable thresholds via format_options.

Verifies that hardcoded values in content extractors and preprocessors
can be overridden through ``ProcessingConfig.format_options``.

Thresholds tested:
1. PDF default: render_dpi, min_image_size, min_image_area
2. PPTX: max_group_depth
3. CSV: delimiter_candidates
4. DOC: min_text_fragment_length
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from contextifier.config import ProcessingConfig


# ═════════════════════════════════════════════════════════════════════════════
# 1. BaseContentExtractor — config propagation
# ═════════════════════════════════════════════════════════════════════════════


class TestBaseContentExtractorConfig(unittest.TestCase):
    """Verify BaseContentExtractor stores config correctly."""

    def test_config_default_none(self):
        """Config is None when not provided."""
        from contextifier.pipeline.content_extractor import BaseContentExtractor

        # Can't instantiate ABC directly — use a concrete subclass
        class _Stub(BaseContentExtractor):
            def extract_text(self, preprocessed, **kwargs):
                return ""
            def get_format_name(self):
                return "stub"

        ext = _Stub()
        self.assertIsNone(ext._config)

    def test_config_stored(self):
        """Config is stored when provided."""
        from contextifier.pipeline.content_extractor import BaseContentExtractor

        class _Stub(BaseContentExtractor):
            def extract_text(self, preprocessed, **kwargs):
                return ""
            def get_format_name(self):
                return "stub"

        config = ProcessingConfig()
        ext = _Stub(config=config)
        self.assertIs(ext._config, config)


# ═════════════════════════════════════════════════════════════════════════════
# 2. PDF Default — render_dpi, min_image_size, min_image_area
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfDefaultThresholds(unittest.TestCase):
    """Verify PdfDefaultContentExtractor reads thresholds from config."""

    def _make_extractor(self, **format_opts):
        from contextifier.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        config = ProcessingConfig()
        if format_opts:
            config = config.with_format_option("pdf", **format_opts)
        return PdfDefaultContentExtractor(
            image_service=MagicMock(),
            tag_service=MagicMock(),
            table_service=MagicMock(),
            config=config,
        )

    def test_default_render_dpi(self):
        """Default render_dpi is 150 when not configured."""
        ext = self._make_extractor()
        self.assertEqual(
            ext._config.get_format_option("pdf", "render_dpi", 150), 150,
        )

    def test_custom_render_dpi(self):
        """Custom render_dpi is read from config."""
        ext = self._make_extractor(render_dpi=200)
        self.assertEqual(
            ext._config.get_format_option("pdf", "render_dpi", 150), 200,
        )

    def test_custom_min_image_size(self):
        """Custom min_image_size is read from config."""
        ext = self._make_extractor(min_image_size=30)
        self.assertEqual(
            ext._config.get_format_option("pdf", "min_image_size", 50), 30,
        )

    def test_custom_min_image_area(self):
        """Custom min_image_area is read from config."""
        ext = self._make_extractor(min_image_area=1000)
        self.assertEqual(
            ext._config.get_format_option("pdf", "min_image_area", 2500), 1000,
        )

    @patch("contextifier.handlers.pdf_default.content_extractor.fitz")
    def test_render_dpi_used_in_scan(self, mock_fitz):
        """Configured render_dpi controls the zoom matrix in _extract_scan_pages."""
        ext = self._make_extractor(render_dpi=300)

        # Mock the doc object
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"PNG_DATA"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        ext._tag_service.page_tag.return_value = "[Page Number: 1]"
        ext._image_service.save_and_tag.return_value = "[Image: scan.png]"

        # Call _extract_scan_pages
        result = ext._extract_scan_pages(mock_doc)

        # Verify Matrix was called with custom DPI zoom (300/72 ≈ 4.167)
        zoom = 300 / 72.0
        mock_fitz.Matrix.assert_called_once_with(zoom, zoom)

    def test_min_image_size_filtering(self):
        """Images below min_image_size are filtered out."""
        ext = self._make_extractor(min_image_size=100)

        # Mock page with a small image (80x80)
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_images.return_value = [
            (1, 0, 80, 80, 8, "DeviceRGB", "", "", ""),  # xref=1, 80x80
        ]

        result = ext._extract_images(
            mock_doc, mock_page, 0, set(), [],
        )
        # Should be filtered out (80 < 100)
        self.assertEqual(result, [])


# ═════════════════════════════════════════════════════════════════════════════
# 3. PPTX — max_group_depth
# ═════════════════════════════════════════════════════════════════════════════


class TestPptxGroupDepthConfig(unittest.TestCase):
    """Verify PptxContentExtractor reads max_group_depth from config."""

    def test_default_depth(self):
        """Default max_group_depth is 20."""
        from contextifier.handlers.pptx.content_extractor import (
            PptxContentExtractor,
        )
        ext = PptxContentExtractor()
        self.assertEqual(ext._MAX_GROUP_DEPTH, 20)

    def test_custom_depth_from_config(self):
        """max_group_depth can be overridden via config."""
        from contextifier.handlers.pptx.content_extractor import (
            PptxContentExtractor,
        )
        config = ProcessingConfig().with_format_option(
            "pptx", max_group_depth=5,
        )
        ext = PptxContentExtractor(config=config)
        self.assertEqual(ext._MAX_GROUP_DEPTH, 5)

    def test_no_config_keeps_default(self):
        """Without config, the class attribute default is preserved."""
        from contextifier.handlers.pptx.content_extractor import (
            PptxContentExtractor,
        )
        ext = PptxContentExtractor(config=None)
        self.assertEqual(ext._MAX_GROUP_DEPTH, 20)


# ═════════════════════════════════════════════════════════════════════════════
# 4. CSV — delimiter_candidates
# ═════════════════════════════════════════════════════════════════════════════


class TestCsvDelimiterCandidatesConfig(unittest.TestCase):
    """Verify CSV delimiter_candidates is configurable."""

    def test_default_candidates(self):
        """Default candidates are [',', '\\t', ';', '|']."""
        from contextifier.handlers.csv.preprocessor import DELIMITER_CANDIDATES
        self.assertEqual(DELIMITER_CANDIDATES, [",", "\t", ";", "|"])

    def test_custom_candidates_used(self):
        """Custom delimiter_candidates are used in detection."""
        from contextifier.handlers.csv.preprocessor import _detect_delimiter

        # Data where ':' is the consistent delimiter
        data = "a:b:c\n1:2:3\n4:5:6"
        # Default candidates won't detect ':'
        result_default = _detect_delimiter(data)
        self.assertNotEqual(result_default, ":")

        # Custom candidates including ':' should detect it
        result_custom = _detect_delimiter(data, candidates=[":", ","])
        self.assertEqual(result_custom, ":")

    def test_preprocessor_stores_candidates(self):
        """CsvPreprocessor stores delimiter_candidates."""
        from contextifier.handlers.csv.preprocessor import CsvPreprocessor

        candidates = [",", ";"]
        prep = CsvPreprocessor(delimiter_candidates=candidates)
        self.assertEqual(prep._delimiter_candidates, candidates)

    def test_preprocessor_default_candidates_none(self):
        """CsvPreprocessor defaults delimiter_candidates to None."""
        from contextifier.handlers.csv.preprocessor import CsvPreprocessor

        prep = CsvPreprocessor()
        self.assertIsNone(prep._delimiter_candidates)

    def test_handler_passes_candidates_from_config(self):
        """CSVHandler reads delimiter_candidates from config."""
        from contextifier.handlers.csv.handler import CSVHandler

        config = ProcessingConfig().with_format_option(
            "csv", delimiter_candidates=[",", ";", ":"],
        )
        handler = CSVHandler(config)
        prep = handler._preprocessor
        self.assertEqual(prep._delimiter_candidates, [",", ";", ":"])


# ═════════════════════════════════════════════════════════════════════════════
# 5. DOC — min_text_fragment_length
# ═════════════════════════════════════════════════════════════════════════════


class TestDocFragmentLengthConfig(unittest.TestCase):
    """Verify DocContentExtractor reads min_text_fragment_length from config."""

    def test_default_fragment_length(self):
        """Default min_text_fragment_length is 4."""
        from contextifier.handlers.doc.content_extractor import (
            DocContentExtractor,
        )
        ext = DocContentExtractor()
        self.assertEqual(ext._min_text_fragment_length, 4)
        self.assertEqual(ext._min_unicode_bytes, 8)

    def test_custom_fragment_length(self):
        """Custom min_text_fragment_length is read from config."""
        from contextifier.handlers.doc.content_extractor import (
            DocContentExtractor,
        )
        config = ProcessingConfig().with_format_option(
            "doc", min_text_fragment_length=8,
        )
        ext = DocContentExtractor(config=config)
        self.assertEqual(ext._min_text_fragment_length, 8)
        self.assertEqual(ext._min_unicode_bytes, 16)

    def test_no_config_keeps_default(self):
        """Without config, uses module constant."""
        from contextifier.handlers.doc.content_extractor import (
            DocContentExtractor,
        )
        ext = DocContentExtractor(config=None)
        self.assertEqual(ext._min_text_fragment_length, 4)

    def test_handler_passes_config(self):
        """DOCHandler passes config to DocContentExtractor."""
        from contextifier.handlers.doc.handler import DOCHandler

        config = ProcessingConfig().with_format_option(
            "doc", min_text_fragment_length=6,
        )
        handler = DOCHandler(config)
        ext = handler._content_extractor
        self.assertEqual(ext._min_text_fragment_length, 6)
        self.assertEqual(ext._min_unicode_bytes, 12)


# ═════════════════════════════════════════════════════════════════════════════
# 6. PDF Handler — config propagation to extractors
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfHandlerConfigPropagation(unittest.TestCase):
    """Verify PDFHandler passes config to content extractors."""

    def test_default_mode_receives_config(self):
        """PdfDefaultContentExtractor receives config via PDFHandler."""
        from contextifier.handlers.pdf.handler import PDFHandler

        config = ProcessingConfig().with_format_option(
            "pdf", mode="default", render_dpi=200,
        )
        handler = PDFHandler(config)
        ext = handler._content_extractor
        self.assertIs(ext._config, config)
        self.assertEqual(
            ext._config.get_format_option("pdf", "render_dpi", 150), 200,
        )

    def test_plus_mode_receives_config(self):
        """PdfPlusContentExtractor receives config via PDFHandler."""
        from contextifier.handlers.pdf.handler import PDFHandler

        config = ProcessingConfig().with_format_option(
            "pdf", mode="plus", min_image_size=30,
        )
        handler = PDFHandler(config)
        ext = handler._content_extractor
        self.assertIs(ext._config, config)


if __name__ == "__main__":
    unittest.main()
