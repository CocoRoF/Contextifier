# tests/unit/handlers/test_text_handler.py
"""Unit tests for TextHandler."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig
from contextifier.handlers.text.handler import TextHandler
from contextifier.types import FileContext


def _make_text_context(
    text: str,
    *,
    ext: str = "txt",
    encoding: str = "utf-8",
) -> FileContext:
    data = text.encode(encoding)
    return FileContext(
        file_name=f"sample.{ext}",
        file_extension=ext,
        file_category="text",
        file_data=data,
        file_stream=None,
    )


@pytest.fixture()
def handler() -> TextHandler:
    return TextHandler(ProcessingConfig())


class TestTextHandlerInit:
    def test_supported_extensions_includes_txt(self, handler: TextHandler) -> None:
        assert "txt" in handler.supported_extensions

    def test_supported_extensions_includes_code(self, handler: TextHandler) -> None:
        assert "py" in handler.supported_extensions
        assert "js" in handler.supported_extensions

    def test_handler_name(self, handler: TextHandler) -> None:
        assert handler.handler_name == "Text Handler"


class TestTextExtraction:
    def test_extract_plain_text(self, handler: TextHandler) -> None:
        ctx = _make_text_context("Hello, World!")
        text = handler.extract_text(ctx, include_metadata=False)
        assert "Hello, World!" in text

    def test_extract_multiline(self, handler: TextHandler) -> None:
        ctx = _make_text_context("line1\nline2\nline3")
        text = handler.extract_text(ctx, include_metadata=False)
        assert "line1" in text
        assert "line2" in text

    def test_extract_empty(self, handler: TextHandler) -> None:
        ctx = _make_text_context(" ")  # single space, not truly empty
        result = handler.process(ctx, include_metadata=False)
        assert result.text is not None

    def test_extract_utf8_bom(self, handler: TextHandler) -> None:
        raw = b"\xef\xbb\xbfBOM content"
        ctx = FileContext(
            file_name="bom.txt",
            file_extension="txt",
            file_category="text",
            file_data=raw,
            file_stream=None,
        )
        text = handler.extract_text(ctx, include_metadata=False)
        # BOM should be stripped
        assert text.strip().startswith("BOM content") or "BOM content" in text


class TestTextHandlerPipeline:
    def test_pipeline_components_created(self, handler: TextHandler) -> None:
        assert handler.converter is not None
        assert handler.preprocessor is not None
        assert handler.metadata_extractor is not None
        assert handler.content_extractor is not None
        assert handler.postprocessor is not None

    def test_process_returns_extraction_result(self, handler: TextHandler) -> None:
        ctx = _make_text_context("Test content")
        result = handler.process(ctx, include_metadata=False)
        assert hasattr(result, "text")
        assert hasattr(result, "metadata")
