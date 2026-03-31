# tests/unit/handlers/test_pptx_handler.py
"""Unit tests for PPTXHandler construction and pipeline setup."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig
from contextifier.handlers.pptx.handler import PPTXHandler


@pytest.fixture()
def pptx_handler() -> PPTXHandler:
    return PPTXHandler(ProcessingConfig())


class TestPPTXHandlerInit:
    def test_supported_extensions(self, pptx_handler: PPTXHandler) -> None:
        assert pptx_handler.supported_extensions == frozenset({"pptx"})

    def test_handler_name(self, pptx_handler: PPTXHandler) -> None:
        assert "PPTX" in pptx_handler.handler_name or "pptx" in pptx_handler.handler_name.lower()

    def test_pipeline_components_created(self, pptx_handler: PPTXHandler) -> None:
        assert pptx_handler.converter is not None
        assert pptx_handler.preprocessor is not None
        assert pptx_handler.metadata_extractor is not None
        assert pptx_handler.content_extractor is not None
        assert pptx_handler.postprocessor is not None

    def test_services_passed_through(self) -> None:
        from unittest.mock import MagicMock
        mock_chart = MagicMock()
        mock_table = MagicMock()
        handler = PPTXHandler(
            ProcessingConfig(),
            chart_service=mock_chart,
            table_service=mock_table,
        )
        assert handler.chart_service is mock_chart
        assert handler.table_service is mock_table
