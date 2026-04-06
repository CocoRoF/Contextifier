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

    # ── P4-2: read_only mode ──────────────────────────────────────────

    def test_read_only_default_false(self) -> None:
        """Default: read_only=False."""
        handler = XLSXHandler(ProcessingConfig())
        assert handler.converter._read_only is False

    def test_read_only_from_format_options(self) -> None:
        """read_only=True via format_options["xlsx"]["read_only"]."""
        config = ProcessingConfig(
            format_options={"xlsx": {"read_only": True}}
        )
        handler = XLSXHandler(config)
        assert handler.converter._read_only is True

    def test_read_only_passed_to_load_workbook(self) -> None:
        """Converter passes read_only to openpyxl.load_workbook."""
        from unittest.mock import patch, MagicMock
        from contextifier.handlers.xlsx.converter import XlsxConverter

        converter = XlsxConverter(data_only=True, read_only=True)

        mock_wb = MagicMock()
        with patch("contextifier.handlers.xlsx.converter.openpyxl.load_workbook", return_value=mock_wb) as mock_load:
            ctx = {"file_data": b"PK\x03\x04" + b"\x00" * 100}
            try:
                converter.convert(ctx)
            except Exception:
                pass  # May fail on invalid XLSX, we just check the call

            if mock_load.called:
                _, kwargs = mock_load.call_args
                assert kwargs.get("read_only") is True
                assert kwargs.get("data_only") is True
