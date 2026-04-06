# tests/unit/integrations/test_langchain_loader.py
"""Tests for ContextifierLoader — LangChain integration."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from contextifier.integrations.langchain_loader import ContextifierLoader
from contextifier.config import ProcessingConfig
from contextifier.document_processor import ChunkResult
from contextifier.types import Chunk, ChunkMetadata


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def txt_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello world. This is a test document.", encoding="utf-8")
    return str(f)


# ── TestSingleDocument ────────────────────────────────────────────────────

class TestSingleDocument:
    """Tests for single-document (non-chunked) mode."""

    def test_load_returns_one_document(self, txt_file):
        loader = ContextifierLoader(txt_file)
        docs = loader.load()
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_page_content_is_text(self, txt_file):
        loader = ContextifierLoader(txt_file)
        docs = loader.load()
        assert "Hello world" in docs[0].page_content

    def test_metadata_source(self, txt_file):
        loader = ContextifierLoader(txt_file)
        docs = loader.load()
        assert docs[0].metadata["source"] == txt_file
        assert docs[0].metadata["file_name"] == "sample.txt"
        assert docs[0].metadata["file_extension"] == "txt"

    def test_extra_metadata_merged(self, txt_file):
        loader = ContextifierLoader(txt_file, extra_metadata={"user": "test"})
        docs = loader.load()
        assert docs[0].metadata["user"] == "test"
        assert docs[0].metadata["source"] == txt_file

    def test_lazy_load_yields(self, txt_file):
        loader = ContextifierLoader(txt_file)
        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert isinstance(docs[0], Document)


# ── TestChunkedDocument ───────────────────────────────────────────────────

class TestChunkedDocument:
    """Tests for chunked mode."""

    def test_chunk_mode_returns_multiple(self, txt_file):
        loader = ContextifierLoader(txt_file, chunk=True, chunk_size=10, chunk_overlap=0)
        docs = loader.load()
        assert len(docs) > 1
        for doc in docs:
            assert isinstance(doc, Document)

    def test_chunk_index_in_metadata(self, txt_file):
        loader = ContextifierLoader(txt_file, chunk=True, chunk_size=10, chunk_overlap=0)
        docs = loader.load()
        for i, doc in enumerate(docs):
            assert doc.metadata["chunk_index"] == i
            assert doc.metadata["source"] == txt_file

    def test_chunk_size_override(self, txt_file):
        loader = ContextifierLoader(txt_file, chunk=True, chunk_size=5, chunk_overlap=0)
        docs = loader.load()
        # With chunk_size=5, should produce many small chunks
        assert len(docs) >= 2


# ── TestWithConfig ────────────────────────────────────────────────────────

class TestWithConfig:
    """Tests for passing custom config."""

    def test_custom_config(self, txt_file):
        config = ProcessingConfig().with_metadata(language="en")
        loader = ContextifierLoader(txt_file, config=config)
        docs = loader.load()
        assert len(docs) == 1

    def test_extract_metadata_false(self, txt_file):
        loader = ContextifierLoader(txt_file, extract_metadata=False)
        docs = loader.load()
        assert len(docs) == 1


# ── TestOCRIntegration ────────────────────────────────────────────────────

class TestOCRIntegration:
    """Tests for OCR engine passing."""

    def test_ocr_engine_passed_to_processor(self, txt_file):
        """Verify ocr_engine kwarg is forwarded to DocumentProcessor."""
        mock_engine = MagicMock()
        with patch(
            "contextifier.integrations.langchain_loader.DocumentProcessor"
        ) as MockDP:
            mock_processor = MagicMock()
            mock_processor.extract_text.return_value = "ocr text"
            MockDP.return_value = mock_processor

            loader = ContextifierLoader(
                txt_file, ocr_engine=mock_engine, ocr_processing=True
            )
            docs = loader.load()

            MockDP.assert_called_once_with(config=None, ocr_engine=mock_engine)
            mock_processor.extract_text.assert_called_once_with(
                txt_file,
                extract_metadata=True,
                ocr_processing=True,
            )
            assert docs[0].page_content == "ocr text"

    def test_chunk_mode_with_ocr(self, txt_file):
        """Verify chunk mode forwards ocr_processing."""
        mock_engine = MagicMock()
        with patch(
            "contextifier.integrations.langchain_loader.DocumentProcessor"
        ) as MockDP:
            mock_processor = MagicMock()
            mock_result = ChunkResult(
                chunks=["c1", "c2"],
                chunks_with_metadata=None,
                source_file=txt_file,
            )
            mock_processor.extract_chunks.return_value = mock_result
            MockDP.return_value = mock_processor

            loader = ContextifierLoader(
                txt_file,
                ocr_engine=mock_engine,
                ocr_processing=True,
                chunk=True,
                chunk_size=100,
            )
            docs = loader.load()

            mock_processor.extract_chunks.assert_called_once()
            call_kwargs = mock_processor.extract_chunks.call_args
            assert call_kwargs.kwargs.get("ocr_processing") is True or call_kwargs[1].get("ocr_processing") is True
            assert len(docs) == 2
