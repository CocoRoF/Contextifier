# tests/unit/handlers/test_docx_handler.py
"""Unit tests for DOCXHandler construction and pipeline setup."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig
from contextifier.handlers.docx.handler import DOCXHandler


@pytest.fixture()
def docx_handler() -> DOCXHandler:
    return DOCXHandler(ProcessingConfig())


class TestDOCXHandlerInit:
    def test_supported_extensions(self, docx_handler: DOCXHandler) -> None:
        assert docx_handler.supported_extensions == frozenset({"docx"})

    def test_handler_name(self, docx_handler: DOCXHandler) -> None:
        assert "DOCX" in docx_handler.handler_name or "docx" in docx_handler.handler_name.lower()

    def test_pipeline_components_created(self, docx_handler: DOCXHandler) -> None:
        assert docx_handler.converter is not None
        assert docx_handler.preprocessor is not None
        assert docx_handler.metadata_extractor is not None
        assert docx_handler.content_extractor is not None
        assert docx_handler.postprocessor is not None

    def test_services_passed_through(self) -> None:
        from unittest.mock import MagicMock
        mock_img = MagicMock()
        mock_tag = MagicMock()
        handler = DOCXHandler(
            ProcessingConfig(),
            image_service=mock_img,
            tag_service=mock_tag,
        )
        assert handler.image_service is mock_img
        assert handler.tag_service is mock_tag
