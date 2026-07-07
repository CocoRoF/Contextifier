# tests/unit/handlers/test_html_handler.py
"""Unit tests for HtmlHandler — specifically the preprocessor's base64 size limit."""

from __future__ import annotations

import base64
import logging

import pytest

from contextifier.handlers.html.converter import HtmlConvertedData
from contextifier.handlers.html.preprocessor import (
    HtmlPreprocessor,
)


@pytest.fixture()
def preprocessor() -> HtmlPreprocessor:
    return HtmlPreprocessor()


def _make_converted(html_text: str) -> HtmlConvertedData:
    return HtmlConvertedData(
        html_text=html_text,
        encoding="utf-8",
        file_extension="html",
    )


class TestBase64ImageSizeLimit:
    """P0-2: base64 embedded images exceeding the size threshold are skipped."""

    def test_small_image_is_extracted(self, preprocessor: HtmlPreprocessor) -> None:
        """A small base64 image should be extracted normally."""
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16  # tiny fake PNG
        b64 = base64.b64encode(raw).decode()
        html = f'<html><body><img src="data:image/png;base64,{b64}"></body></html>'

        result = preprocessor.preprocess(_make_converted(html))

        images = result.resources.get("images", [])
        assert len(images) == 1
        assert images[0]["format"] == "png"
        assert images[0]["data"] == raw

    def test_oversized_image_is_skipped(
        self, preprocessor: HtmlPreprocessor, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Base64 image whose decoded size exceeds _MAX_IMAGE_DECODE_BYTES is skipped."""
        # Create a base64 string that decodes to > threshold
        # base64 encoding: 3 bytes → 4 chars. To exceed threshold we need
        # a base64 string whose decoded length > _MAX_IMAGE_DECODE_BYTES.
        # Instead of allocating real huge data, we just need a b64 string
        # large enough. Using a small fake + manipulated length check.
        #
        # The preprocessor estimates: len(b64_str) * 3 // 4
        # We need: len(b64_str) * 3 // 4 > _MAX_IMAGE_DECODE_BYTES
        # So: len(b64_str) > _MAX_IMAGE_DECODE_BYTES * 4 // 3
        #
        # That would require ~67 MB of string in the test, which is too much.
        # Instead, we temporarily monkeypatch the threshold.
        import contextifier.handlers.html.preprocessor as mod

        original = mod._MAX_IMAGE_DECODE_BYTES
        try:
            mod._MAX_IMAGE_DECODE_BYTES = 10  # 10 bytes threshold

            raw = b"\x00" * 20  # 20 bytes > 10-byte threshold
            b64 = base64.b64encode(raw).decode()
            html = f'<html><body><img src="data:image/png;base64,{b64}"></body></html>'

            with caplog.at_level(logging.WARNING, logger="contextifier.handlers.html.preprocessor"):
                result = preprocessor.preprocess(_make_converted(html))

            images = result.resources.get("images", [])
            assert len(images) == 0
            assert "Skipping oversized base64 image" in caplog.text
        finally:
            mod._MAX_IMAGE_DECODE_BYTES = original

    def test_mix_of_small_and_oversized(
        self, preprocessor: HtmlPreprocessor,
    ) -> None:
        """Only images within the size limit are extracted; oversized ones are skipped."""
        import contextifier.handlers.html.preprocessor as mod

        original = mod._MAX_IMAGE_DECODE_BYTES
        try:
            mod._MAX_IMAGE_DECODE_BYTES = 10

            small_raw = b"\x01\x02\x03"  # 3 bytes, under limit
            big_raw = b"\x00" * 20       # 20 bytes, over limit
            small_b64 = base64.b64encode(small_raw).decode()
            big_b64 = base64.b64encode(big_raw).decode()

            html = (
                "<html><body>"
                f'<img src="data:image/png;base64,{small_b64}">'
                f'<img src="data:image/jpeg;base64,{big_b64}">'
                "</body></html>"
            )

            result = preprocessor.preprocess(_make_converted(html))

            images = result.resources.get("images", [])
            assert len(images) == 1
            assert images[0]["format"] == "png"
            assert images[0]["data"] == small_raw
        finally:
            mod._MAX_IMAGE_DECODE_BYTES = original
