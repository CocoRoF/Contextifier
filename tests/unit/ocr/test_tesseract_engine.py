# tests/unit/ocr/test_tesseract_engine.py
"""
P2-3 / P2-4: Image OCR integration + Tesseract OCR engine tests.
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

from contextifier.ocr.engines.tesseract_engine import TesseractOCREngine


def _make_mock_pytesseract(return_value="Hello World"):
    """Create a mock pytesseract module."""
    m = mock.MagicMock()
    m.image_to_string.return_value = return_value
    m.pytesseract = mock.MagicMock()
    return m


def _make_mock_pil():
    """Create a mock PIL module with Image.open."""
    pil = mock.MagicMock()
    pil.Image.open.return_value = mock.MagicMock()
    return pil


class TestTesseractEngine:
    """Unit tests for TesseractOCREngine."""

    def test_provider_name(self):
        engine = TesseractOCREngine()
        assert engine.provider == "tesseract"

    def test_build_message_content_is_noop(self):
        engine = TesseractOCREngine()
        result = engine.build_message_content("base64data", "image/png", "prompt")
        assert result == []

    def test_llm_client_is_none(self):
        engine = TesseractOCREngine()
        assert engine.llm_client is None

    def test_repr(self):
        engine = TesseractOCREngine(lang="kor+eng")
        assert "kor+eng" in repr(engine)
        assert "TesseractOCREngine" in repr(engine)

    def test_convert_image_success(self):
        """Successful OCR returns [Figure:...] wrapped text."""
        mock_pt = _make_mock_pytesseract("Hello World")
        mock_pil = _make_mock_pil()

        engine = TesseractOCREngine(lang="eng")
        with mock.patch.dict(sys.modules, {
            "pytesseract": mock_pt,
            "PIL": mock_pil,
            "PIL.Image": mock_pil.Image,
        }):
            result = engine.convert_image_to_text("/fake/image.png")

        assert result == "[Figure:Hello World]"
        mock_pil.Image.open.assert_called_once_with("/fake/image.png")
        mock_pt.image_to_string.assert_called_once()

    def test_convert_image_empty_result(self):
        """Empty OCR result returns placeholder."""
        mock_pt = _make_mock_pytesseract("   ")
        mock_pil = _make_mock_pil()

        engine = TesseractOCREngine()
        with mock.patch.dict(sys.modules, {
            "pytesseract": mock_pt,
            "PIL": mock_pil,
            "PIL.Image": mock_pil.Image,
        }):
            result = engine.convert_image_to_text("/fake/empty.png")

        assert result == "[Figure: (no text detected)]"

    def test_convert_image_import_error(self):
        """Missing pytesseract returns install hint."""
        engine = TesseractOCREngine()

        # Remove pytesseract from sys.modules so import fails
        with mock.patch.dict(sys.modules, {"pytesseract": None}):
            result = engine.convert_image_to_text("/fake/img.png")

        assert "not installed" in result

    def test_convert_image_runtime_error(self):
        """Runtime error returns error message."""
        mock_pt = _make_mock_pytesseract()
        mock_pil = _make_mock_pil()
        mock_pil.Image.open.side_effect = RuntimeError("Tesseract not found")

        engine = TesseractOCREngine()
        with mock.patch.dict(sys.modules, {
            "pytesseract": mock_pt,
            "PIL": mock_pil,
            "PIL.Image": mock_pil.Image,
        }):
            result = engine.convert_image_to_text("/fake/img.png")

        assert "[Image conversion error:" in result
        assert "Tesseract not found" in result

    def test_custom_lang_passed_to_tesseract(self):
        """Custom language parameter is forwarded to pytesseract."""
        mock_pt = _make_mock_pytesseract("한글 텍스트")
        mock_pil = _make_mock_pil()

        engine = TesseractOCREngine(lang="kor+eng", config="--psm 6")
        with mock.patch.dict(sys.modules, {
            "pytesseract": mock_pt,
            "PIL": mock_pil,
            "PIL.Image": mock_pil.Image,
        }):
            engine.convert_image_to_text("/fake/korean.png")

        call_kwargs = mock_pt.image_to_string.call_args
        assert call_kwargs[1]["lang"] == "kor+eng"
        assert call_kwargs[1]["config"] == "--psm 6"

    def test_custom_tesseract_cmd(self):
        """Setting tesseract_cmd configures pytesseract path."""
        mock_pt = _make_mock_pytesseract()
        mock_pt.pytesseract = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"pytesseract": mock_pt}):
            TesseractOCREngine(tesseract_cmd="/usr/local/bin/tesseract")
        assert mock_pt.pytesseract.tesseract_cmd == "/usr/local/bin/tesseract"


class TestImageOCRIntegration:
    """P2-3: Verify Image handler → OCR pipeline integration."""

    def test_image_handler_produces_image_tag(self):
        """ImageContentExtractor produces [Image:...] tags for OCR."""
        from contextifier.handlers.image.content_extractor import ImageContentExtractor
        from contextifier.types import PreprocessedData

        mock_img_service = mock.MagicMock()
        mock_img_service.save_and_tag.return_value = "[Image:photo.png]"

        extractor = ImageContentExtractor(image_service=mock_img_service)
        preprocessed = PreprocessedData(
            content=b"\x89PNG\r\n\x1a\n fake png",
            raw_content=b"",
            properties={"file_extension": ".png"},
        )
        result = extractor.extract_text(preprocessed)
        assert result == "[Image:photo.png]"

    def test_image_handler_fallback_no_service(self):
        """Without ImageService, handler returns placeholder tag."""
        from contextifier.handlers.image.content_extractor import ImageContentExtractor
        from contextifier.types import PreprocessedData

        extractor = ImageContentExtractor()
        preprocessed = PreprocessedData(
            content=b"\x89PNG data",
            raw_content=b"",
            properties={"file_extension": ".png", "detected_format": "png"},
        )
        result = extractor.extract_text(preprocessed)
        assert "[Image:" in result

    def test_engine_registered_in_init(self):
        """TesseractOCREngine is accessible from engines package."""
        from contextifier.ocr.engines import TesseractOCREngine as Imported
        assert Imported is TesseractOCREngine
