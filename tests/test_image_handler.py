# tests/test_image_handler.py
"""
Comprehensive tests for contextifier_new.handlers.image

Covers:
    - _constants: magic bytes, detect_image_format, extension sets
    - converter: validation, ImageConvertedData, edge cases
    - preprocessor: format detection, property injection
    - metadata_extractor: minimal metadata generation
    - content_extractor: image saving via service, fallback tags
    - handler: full wiring, supported_extensions, handler_name
    - full pipeline: end-to-end processing
    - edge cases: empty data, corrupt data, unsupported formats
"""

from __future__ import annotations

import struct
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import (
    DocumentMetadata,
    ExtractionResult,
    FileContext,
    PreprocessedData,
)
from contextifier_new.errors import ConversionError

# ── Helpers ───────────────────────────────────────────────────────────────

# Real magic-byte headers for test data
JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 100
PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
GIF87_HEADER = b"GIF87a" + b"\x00" * 100
GIF89_HEADER = b"GIF89a" + b"\x00" * 100
BMP_HEADER = b"BM" + b"\x00" * 100
WEBP_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 100
TIFF_LE_HEADER = b"II\x2a\x00" + b"\x00" * 100
TIFF_BE_HEADER = b"MM\x00\x2a" + b"\x00" * 100


def _make_file_context(
    data: bytes,
    ext: str = "png",
    name: str = "test",
) -> FileContext:
    """Create a minimal FileContext dict for testing."""
    return {
        "file_path": f"/tmp/{name}.{ext}",
        "file_name": f"{name}.{ext}",
        "file_extension": ext,
        "file_category": "image",
        "file_data": data,
        "file_stream": None,
        "file_size": len(data),
    }


# ═════════════════════════════════════════════════════════════════════════
# 1. _constants
# ═════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.image._constants import (
    IMAGE_EXTENSIONS,
    MAGIC_VALIDATED_EXTENSIONS,
    MAGIC_TABLE,
    MAGIC_JPEG,
    MAGIC_PNG,
    MAGIC_GIF87,
    MAGIC_GIF89,
    MAGIC_BMP,
    MAGIC_WEBP_RIFF,
    MAGIC_WEBP_TAG,
    MAGIC_TIFF_LE,
    MAGIC_TIFF_BE,
    detect_image_format,
)


class TestConstants:
    """Tests for _constants module."""

    def test_image_extensions_is_frozenset(self):
        assert isinstance(IMAGE_EXTENSIONS, frozenset)

    def test_image_extensions_contains_known(self):
        for ext in ("jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif"):
            assert ext in IMAGE_EXTENSIONS, f"Missing: {ext}"

    def test_image_extensions_contains_special(self):
        for ext in ("svg", "ico", "heic", "heif"):
            assert ext in IMAGE_EXTENSIONS, f"Missing: {ext}"

    def test_magic_validated_subset_of_image_extensions(self):
        assert MAGIC_VALIDATED_EXTENSIONS.issubset(IMAGE_EXTENSIONS)

    def test_magic_table_is_nonempty(self):
        assert len(MAGIC_TABLE) > 0

    def test_magic_bytes_are_bytes(self):
        for magic in (MAGIC_JPEG, MAGIC_PNG, MAGIC_GIF87, MAGIC_GIF89,
                       MAGIC_BMP, MAGIC_WEBP_RIFF, MAGIC_TIFF_LE, MAGIC_TIFF_BE):
            assert isinstance(magic, bytes)


class TestDetectImageFormat:
    """Tests for detect_image_format()."""

    def test_detect_jpeg(self):
        assert detect_image_format(JPEG_HEADER) == "jpeg"

    def test_detect_png(self):
        assert detect_image_format(PNG_HEADER) == "png"

    def test_detect_gif87(self):
        assert detect_image_format(GIF87_HEADER) == "gif"

    def test_detect_gif89(self):
        assert detect_image_format(GIF89_HEADER) == "gif"

    def test_detect_bmp(self):
        assert detect_image_format(BMP_HEADER) == "bmp"

    def test_detect_webp(self):
        assert detect_image_format(WEBP_HEADER) == "webp"

    def test_detect_tiff_le(self):
        assert detect_image_format(TIFF_LE_HEADER) == "tiff"

    def test_detect_tiff_be(self):
        assert detect_image_format(TIFF_BE_HEADER) == "tiff"

    def test_detect_unknown_returns_none(self):
        assert detect_image_format(b"\x00\x01\x02\x03") is None

    def test_detect_empty_returns_none(self):
        assert detect_image_format(b"") is None

    def test_detect_short_data(self):
        # Single byte — should not crash
        assert detect_image_format(b"\xff") is None

    def test_detect_webp_wrong_tag(self):
        # RIFF header but not WEBP tag at offset 8
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 50
        assert detect_image_format(data) != "webp"


# ═════════════════════════════════════════════════════════════════════════
# 2. ImageConverter
# ═════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.image.converter import ImageConverter, ImageConvertedData


class TestImageConverter:
    """Tests for ImageConverter."""

    def setup_method(self):
        self.converter = ImageConverter()

    def test_convert_png(self):
        fc = _make_file_context(PNG_HEADER, ext="png")
        result = self.converter.convert(fc)
        assert isinstance(result, ImageConvertedData)
        assert result.image_data == PNG_HEADER
        assert result.detected_format == "png"

    def test_convert_jpeg(self):
        fc = _make_file_context(JPEG_HEADER, ext="jpg")
        result = self.converter.convert(fc)
        assert result.detected_format == "jpeg"

    def test_convert_gif(self):
        fc = _make_file_context(GIF89_HEADER, ext="gif")
        result = self.converter.convert(fc)
        assert result.detected_format == "gif"

    def test_convert_bmp(self):
        fc = _make_file_context(BMP_HEADER, ext="bmp")
        result = self.converter.convert(fc)
        assert result.detected_format == "bmp"

    def test_convert_webp(self):
        fc = _make_file_context(WEBP_HEADER, ext="webp")
        result = self.converter.convert(fc)
        assert result.detected_format == "webp"

    def test_convert_tiff_le(self):
        fc = _make_file_context(TIFF_LE_HEADER, ext="tiff")
        result = self.converter.convert(fc)
        assert result.detected_format == "tiff"

    def test_convert_tiff_be(self):
        fc = _make_file_context(TIFF_BE_HEADER, ext="tif")
        result = self.converter.convert(fc)
        assert result.detected_format == "tiff"

    def test_convert_empty_data_raises(self):
        fc = _make_file_context(b"", ext="png")
        with pytest.raises(ConversionError):
            self.converter.convert(fc)

    def test_convert_unknown_magic_still_succeeds(self):
        """SVG, ICO etc. won't match magic — should still convert."""
        fc = _make_file_context(b"<svg></svg>", ext="svg")
        result = self.converter.convert(fc)
        assert result.image_data == b"<svg></svg>"
        assert result.detected_format is None

    def test_convert_magic_mismatch_warns(self):
        """Extension claims jpg but data is random — warning, not error."""
        fc = _make_file_context(b"\x00\x01random", ext="jpg")
        result = self.converter.convert(fc)
        assert result.detected_format is None

    def test_validate_returns_true_for_data(self):
        fc = _make_file_context(PNG_HEADER)
        assert self.converter.validate(fc) is True

    def test_validate_returns_false_for_empty(self):
        fc = _make_file_context(b"")
        assert self.converter.validate(fc) is False

    def test_get_format_name(self):
        assert self.converter.get_format_name() == "image"

    def test_close_no_error(self):
        """close() should be a no-op for images."""
        self.converter.close(ImageConvertedData(b"data", "png"))


# ═════════════════════════════════════════════════════════════════════════
# 3. ImagePreprocessor
# ═════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.image.preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()

    def test_preprocess_from_converted_data(self):
        converted = ImageConvertedData(image_data=PNG_HEADER, detected_format="png")
        result = self.preprocessor.preprocess(converted)
        assert isinstance(result, PreprocessedData)
        assert result.content == PNG_HEADER
        assert result.encoding == "binary"
        assert result.properties["detected_format"] == "png"
        assert result.properties["file_size"] == len(PNG_HEADER)

    def test_preprocess_from_raw_bytes(self):
        """Fallback: raw bytes instead of ImageConvertedData."""
        result = self.preprocessor.preprocess(JPEG_HEADER)
        assert result.properties["detected_format"] == "jpeg"

    def test_preprocess_unknown_format(self):
        """File without magic detection falls back to extension via kwargs."""
        data = b"<svg>...</svg>"
        result = self.preprocessor.preprocess(data, file_extension="svg")
        assert result.properties["detected_format"] == "svg"

    def test_preprocess_no_format_no_ext(self):
        data = b"\x00\x01\x02"
        result = self.preprocessor.preprocess(data)
        assert result.properties["detected_format"] == "unknown"

    def test_preprocess_empty_bytes(self):
        result = self.preprocessor.preprocess(b"")
        assert result.properties["file_size"] == 0

    def test_get_format_name(self):
        assert self.preprocessor.get_format_name() == "image"


# ═════════════════════════════════════════════════════════════════════════
# 4. ImageMetadataExtractor
# ═════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.image.metadata_extractor import ImageMetadataExtractor


class TestImageMetadataExtractor:
    """Tests for ImageMetadataExtractor."""

    def setup_method(self):
        self.extractor = ImageMetadataExtractor()

    def test_extract_from_bytes(self):
        meta = self.extractor.extract(PNG_HEADER)
        assert isinstance(meta, DocumentMetadata)
        assert meta.page_count == 1

    def test_extract_from_preprocessed_data(self):
        ppd = PreprocessedData(
            content=PNG_HEADER,
            raw_content=PNG_HEADER,
            encoding="binary",
            resources={},
            properties={"detected_format": "png", "file_size": 108},
        )
        meta = self.extractor.extract(ppd)
        assert meta.page_count == 1
        assert meta.custom["format"] == "png"
        assert meta.custom["file_size"] == 108

    def test_extract_from_unknown_content(self):
        meta = self.extractor.extract("some string")
        assert meta.page_count == 1

    def test_get_format_name(self):
        assert self.extractor.get_format_name() == "image"


# ═════════════════════════════════════════════════════════════════════════
# 5. ImageContentExtractor
# ═════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.image.content_extractor import ImageContentExtractor


class TestContentExtractor:
    """Tests for ImageContentExtractor."""

    def _make_preprocessed(
        self,
        data: bytes = PNG_HEADER,
        ext: str = "png",
        fmt: str = "png",
    ) -> PreprocessedData:
        return PreprocessedData(
            content=data,
            raw_content=data,
            encoding="binary",
            resources={},
            properties={
                "detected_format": fmt,
                "file_extension": ext,
                "file_size": len(data),
            },
        )

    def test_extract_text_with_image_service(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:/saved/test.png]"
        extractor = ImageContentExtractor(image_service=svc)
        ppd = self._make_preprocessed()
        result = extractor.extract_text(ppd)
        assert result == "[Image:/saved/test.png]"
        svc.save_and_tag.assert_called_once()

    def test_extract_text_without_image_service(self):
        extractor = ImageContentExtractor()
        ppd = self._make_preprocessed()
        result = extractor.extract_text(ppd)
        assert "Image:" in result

    def test_extract_text_with_custom_name(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:custom.png]"
        extractor = ImageContentExtractor(image_service=svc)
        ppd = self._make_preprocessed()
        result = extractor.extract_text(ppd, file_name="custom.png")
        svc.save_and_tag.assert_called_once_with(
            PNG_HEADER, custom_name="custom.png",
        )
        assert "custom.png" in result

    def test_extract_text_empty_data(self):
        extractor = ImageContentExtractor()
        ppd = self._make_preprocessed(data=b"", ext="png", fmt="png")
        result = extractor.extract_text(ppd)
        assert result == ""

    def test_extract_text_service_returns_none(self):
        """If save_and_tag returns None (duplicate skipped), fallback tag."""
        svc = MagicMock()
        svc.save_and_tag.return_value = None
        extractor = ImageContentExtractor(image_service=svc)
        ppd = self._make_preprocessed()
        result = extractor.extract_text(ppd)
        assert "Image:" in result

    def test_extract_images_with_service(self):
        svc = MagicMock()
        svc.get_processed_paths.return_value = ["/saved/img1.png"]
        extractor = ImageContentExtractor(image_service=svc)
        ppd = self._make_preprocessed()
        result = extractor.extract_images(ppd)
        assert result == ["/saved/img1.png"]

    def test_extract_images_without_service(self):
        extractor = ImageContentExtractor()
        ppd = self._make_preprocessed()
        assert extractor.extract_images(ppd) == []

    def test_extract_tables_returns_empty(self):
        extractor = ImageContentExtractor()
        ppd = self._make_preprocessed()
        assert extractor.extract_tables(ppd) == []

    def test_extract_charts_returns_empty(self):
        extractor = ImageContentExtractor()
        ppd = self._make_preprocessed()
        assert extractor.extract_charts(ppd) == []

    def test_get_format_name(self):
        extractor = ImageContentExtractor()
        assert extractor.get_format_name() == "image"

    def test_extract_all_orchestration(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:result.png]"
        svc.get_processed_paths.return_value = ["/result.png"]
        extractor = ImageContentExtractor(image_service=svc)
        ppd = self._make_preprocessed()
        result = extractor.extract_all(ppd)
        assert isinstance(result, ExtractionResult)
        assert result.text == "[Image:result.png]"
        assert result.images == ["/result.png"]
        assert result.tables == []
        assert result.charts == []


# ═════════════════════════════════════════════════════════════════════════
# 6. ImageFileHandler
# ═════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.image.handler import ImageFileHandler


class TestImageFileHandler:
    """Tests for ImageFileHandler."""

    def _make_handler(self, **kwargs) -> ImageFileHandler:
        config = ProcessingConfig()
        return ImageFileHandler(config, **kwargs)

    def test_supported_extensions(self):
        h = self._make_handler()
        assert "png" in h.supported_extensions
        assert "jpg" in h.supported_extensions
        assert "jpeg" in h.supported_extensions
        assert "gif" in h.supported_extensions
        assert "bmp" in h.supported_extensions
        assert "webp" in h.supported_extensions
        assert "tiff" in h.supported_extensions
        assert "svg" in h.supported_extensions
        assert "heic" in h.supported_extensions

    def test_handler_name(self):
        h = self._make_handler()
        assert h.handler_name == "Image File Handler"

    def test_create_converter(self):
        h = self._make_handler()
        assert isinstance(h.create_converter(), ImageConverter)

    def test_create_preprocessor(self):
        h = self._make_handler()
        assert isinstance(h.create_preprocessor(), ImagePreprocessor)

    def test_create_metadata_extractor(self):
        h = self._make_handler()
        assert isinstance(h.create_metadata_extractor(), ImageMetadataExtractor)

    def test_create_content_extractor(self):
        h = self._make_handler()
        assert isinstance(h.create_content_extractor(), ImageContentExtractor)


# ═════════════════════════════════════════════════════════════════════════
# 7. Full Pipeline
# ═════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    """End-to-end tests — handler.process() with all stages."""

    def _run_pipeline(
        self,
        data: bytes,
        ext: str = "png",
        name: str = "test",
        **handler_kwargs,
    ):
        config = ProcessingConfig()
        handler = ImageFileHandler(config, **handler_kwargs)
        fc = _make_file_context(data, ext=ext, name=name)
        return handler.process(fc)

    def test_png_pipeline(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:test.png]"
        svc.get_processed_paths.return_value = ["/saved/test.png"]
        result = self._run_pipeline(PNG_HEADER, ext="png", image_service=svc)
        assert isinstance(result, ExtractionResult)
        assert "Image" in result.text
        svc.save_and_tag.assert_called_once()

    def test_jpeg_pipeline(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:photo.jpg]"
        svc.get_processed_paths.return_value = []
        result = self._run_pipeline(JPEG_HEADER, ext="jpg", image_service=svc)
        assert "Image" in result.text

    def test_gif_pipeline(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:anim.gif]"
        svc.get_processed_paths.return_value = []
        result = self._run_pipeline(GIF89_HEADER, ext="gif", image_service=svc)
        assert "Image" in result.text

    def test_pipeline_without_image_service(self):
        """Handler with no image_service still produces a fallback tag."""
        result = self._run_pipeline(PNG_HEADER, ext="png")
        assert "Image:" in result.text

    def test_bmp_pipeline(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:bitmap.bmp]"
        svc.get_processed_paths.return_value = []
        result = self._run_pipeline(BMP_HEADER, ext="bmp", image_service=svc)
        assert "Image" in result.text

    def test_webp_pipeline(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:pic.webp]"
        svc.get_processed_paths.return_value = []
        result = self._run_pipeline(WEBP_HEADER, ext="webp", image_service=svc)
        assert "Image" in result.text

    def test_tiff_pipeline(self):
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:scan.tiff]"
        svc.get_processed_paths.return_value = []
        result = self._run_pipeline(TIFF_LE_HEADER, ext="tiff", image_service=svc)
        assert "Image" in result.text

    def test_svg_pipeline(self):
        """SVG has no magic-byte validation — passes through on trust."""
        svg_data = b"<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>"
        svc = MagicMock()
        svc.save_and_tag.return_value = "[Image:icon.svg]"
        svc.get_processed_paths.return_value = []
        result = self._run_pipeline(svg_data, ext="svg", image_service=svc)
        assert "Image" in result.text


# ═════════════════════════════════════════════════════════════════════════
# 8. Edge Cases
# ═════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_empty_file_data_convert_raises(self):
        converter = ImageConverter()
        fc = _make_file_context(b"", ext="png")
        with pytest.raises(ConversionError):
            converter.convert(fc)

    def test_very_small_data(self):
        """Tiny data (2 bytes) — magic detection returns None, but should not crash."""
        converter = ImageConverter()
        fc = _make_file_context(b"\x89P", ext="png")
        result = converter.convert(fc)
        assert result.detected_format is None

    def test_preprocessor_with_non_bytes_converted(self):
        """Fallback: pass a non-ImageConvertedData, non-bytes object."""
        pp = ImagePreprocessor()
        result = pp.preprocess(42)  # type: ignore
        assert result.content == b""

    def test_metadata_extractor_with_none(self):
        extractor = ImageMetadataExtractor()
        meta = extractor.extract(None)
        assert meta.page_count == 1

    def test_content_extractor_non_bytes_content(self):
        extractor = ImageContentExtractor()
        ppd = PreprocessedData(
            content="not bytes",
            raw_content="not bytes",
            encoding="binary",
            resources={},
            properties={},
        )
        assert extractor.extract_text(ppd) == ""

    def test_converter_named_tuple_fields(self):
        result = ImageConvertedData(image_data=b"abc", detected_format="png")
        assert result.image_data == b"abc"
        assert result.detected_format == "png"
        assert result[0] == b"abc"
        assert result[1] == "png"

    def test_large_image_data(self):
        """1 MB of PNG data should not cause issues."""
        large_data = PNG_HEADER + b"\x00" * (1024 * 1024)
        converter = ImageConverter()
        fc = _make_file_context(large_data, ext="png")
        result = converter.convert(fc)
        assert result.detected_format == "png"
        assert len(result.image_data) == len(large_data)

    def test_handler_process_empty_raises(self):
        config = ProcessingConfig()
        handler = ImageFileHandler(config)
        fc = _make_file_context(b"", ext="png")
        with pytest.raises(Exception):
            handler.process(fc)

    def test_ico_no_magic_validation(self):
        """ICO extension is supported but not magic-validated."""
        from contextifier_new.handlers.image._constants import MAGIC_VALIDATED_EXTENSIONS
        assert "ico" not in MAGIC_VALIDATED_EXTENSIONS
        assert "ico" in IMAGE_EXTENSIONS

    def test_heic_no_magic_validation(self):
        assert "heic" not in MAGIC_VALIDATED_EXTENSIONS
        assert "heic" in IMAGE_EXTENSIONS

    def test_heif_in_extensions(self):
        assert "heif" in IMAGE_EXTENSIONS


# ═════════════════════════════════════════════════════════════════════════
# 9. ImageConvertedData NamedTuple
# ═════════════════════════════════════════════════════════════════════════


class TestImageConvertedData:
    """Tests for ImageConvertedData NamedTuple."""

    def test_fields(self):
        d = ImageConvertedData(image_data=b"x", detected_format="png")
        assert d._fields == ("image_data", "detected_format")

    def test_none_format(self):
        d = ImageConvertedData(image_data=b"data", detected_format=None)
        assert d.detected_format is None

    def test_unpacking(self):
        img, fmt = ImageConvertedData(b"data", "jpeg")
        assert img == b"data"
        assert fmt == "jpeg"


# ═════════════════════════════════════════════════════════════════════════
# 10. Package-level imports
# ═════════════════════════════════════════════════════════════════════════


class TestPackageImports:
    """Tests for contextifier_new.handlers.image __init__.py exports."""

    def test_all_exports(self):
        from contextifier_new.handlers.image import __all__ as exports
        expected = {
            "ImageFileHandler",
            "ImageConverter",
            "ImageConvertedData",
            "ImagePreprocessor",
            "ImageMetadataExtractor",
            "ImageContentExtractor",
            "IMAGE_EXTENSIONS",
            "MAGIC_VALIDATED_EXTENSIONS",
            "detect_image_format",
        }
        assert set(exports) == expected

    def test_import_handler(self):
        from contextifier_new.handlers.image import ImageFileHandler
        assert ImageFileHandler is not None

    def test_import_converter(self):
        from contextifier_new.handlers.image import ImageConverter, ImageConvertedData
        assert ImageConverter is not None
        assert ImageConvertedData is not None

    def test_import_preprocessor(self):
        from contextifier_new.handlers.image import ImagePreprocessor
        assert ImagePreprocessor is not None

    def test_import_metadata_extractor(self):
        from contextifier_new.handlers.image import ImageMetadataExtractor
        assert ImageMetadataExtractor is not None

    def test_import_content_extractor(self):
        from contextifier_new.handlers.image import ImageContentExtractor
        assert ImageContentExtractor is not None

    def test_import_detect_function(self):
        from contextifier_new.handlers.image import detect_image_format
        assert callable(detect_image_format)
