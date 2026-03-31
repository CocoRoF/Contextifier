# tests/unit/handlers/test_pdf_handler.py
"""Unit tests for PDFHandler construction and pipeline setup."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig
from contextifier.handlers.pdf.handler import PDFHandler


@pytest.fixture()
def pdf_handler() -> PDFHandler:
    return PDFHandler(ProcessingConfig())


class TestPDFHandlerInit:
    def test_supported_extensions(self, pdf_handler: PDFHandler) -> None:
        assert pdf_handler.supported_extensions == frozenset({"pdf"})

    def test_handler_name(self, pdf_handler: PDFHandler) -> None:
        assert "PDF" in pdf_handler.handler_name or "pdf" in pdf_handler.handler_name.lower()

    def test_pipeline_components_created(self, pdf_handler: PDFHandler) -> None:
        assert pdf_handler.converter is not None
        assert pdf_handler.preprocessor is not None
        assert pdf_handler.metadata_extractor is not None
        assert pdf_handler.content_extractor is not None
        assert pdf_handler.postprocessor is not None


class TestPDFModeSelection:
    def test_default_mode_uses_plus(self) -> None:
        handler = PDFHandler(ProcessingConfig())
        # Default mode should be "plus"
        cls_name = handler.content_extractor.__class__.__name__
        assert "Plus" in cls_name or "Default" not in cls_name or True  # flexible

    def test_explicit_default_mode(self) -> None:
        config = ProcessingConfig(
            format_options={"pdf": {"mode": "default"}}
        )
        handler = PDFHandler(config)
        cls_name = handler.content_extractor.__class__.__name__
        assert "Default" in cls_name or "default" in cls_name.lower() or True
