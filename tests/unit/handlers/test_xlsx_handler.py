# tests/unit/handlers/test_xlsx_handler.py
"""Unit tests for XLSXHandler construction and pipeline setup."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig
from contextifier.handlers.xlsx.handler import XLSXHandler


@pytest.fixture()
def xlsx_handler() -> XLSXHandler:
    return XLSXHandler(ProcessingConfig())


class TestXLSXHandlerInit:
    def test_supported_extensions(self, xlsx_handler: XLSXHandler) -> None:
        assert xlsx_handler.supported_extensions == frozenset({"xlsx"})

    def test_handler_name(self, xlsx_handler: XLSXHandler) -> None:
        assert "XLSX" in xlsx_handler.handler_name or "xlsx" in xlsx_handler.handler_name.lower()

    def test_pipeline_components_created(self, xlsx_handler: XLSXHandler) -> None:
        assert xlsx_handler.converter is not None
        assert xlsx_handler.preprocessor is not None
        assert xlsx_handler.metadata_extractor is not None
        assert xlsx_handler.content_extractor is not None
        assert xlsx_handler.postprocessor is not None


class TestXLSXFormatOptions:
    def test_data_only_default(self) -> None:
        handler = XLSXHandler(ProcessingConfig())
        # Should not raise
        assert handler.converter is not None

    def test_include_hidden_sheets_option(self) -> None:
        config = ProcessingConfig(
            format_options={"xlsx": {"include_hidden_sheets": True}}
        )
        handler = XLSXHandler(config)
        assert handler.content_extractor is not None
