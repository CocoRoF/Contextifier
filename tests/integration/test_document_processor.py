# tests/integration/test_document_processor.py
"""Integration tests for DocumentProcessor end-to-end flow.

These tests require minimal real files but exercise the full pipeline
from DocumentProcessor.extract_text() through handler → pipeline → result.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from contextifier.config import ProcessingConfig
from contextifier.document_processor import DocumentProcessor, ChunkResult
from contextifier.errors import (
    FileNotFoundError as ContextifyFileNotFoundError,
    UnsupportedFormatError,
)


@pytest.fixture()
def processor() -> DocumentProcessor:
    return DocumentProcessor()


@pytest.fixture()
def tmp_text_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("Integration test content.\nLine two.", encoding="utf-8")
    return p


@pytest.fixture()
def tmp_md_file(tmp_path: Path) -> Path:
    p = tmp_path / "readme.md"
    p.write_text("# Title\n\nParagraph text.", encoding="utf-8")
    return p


class TestExtractText:
    def test_extract_from_txt(
        self, processor: DocumentProcessor, tmp_text_file: Path
    ) -> None:
        text = processor.extract_text(tmp_text_file)
        assert "Integration test content" in text

    def test_extract_from_md(
        self, processor: DocumentProcessor, tmp_md_file: Path
    ) -> None:
        text = processor.extract_text(tmp_md_file)
        assert "Title" in text

    def test_file_not_found_raises(self, processor: DocumentProcessor) -> None:
        with pytest.raises(ContextifyFileNotFoundError):
            processor.extract_text("/nonexistent/file.txt")

    def test_unsupported_extension_raises(
        self, processor: DocumentProcessor, tmp_path: Path
    ) -> None:
        p = tmp_path / "file.unsupported_ext_xyz"
        p.write_bytes(b"data")
        with pytest.raises((UnsupportedFormatError, Exception)):
            processor.extract_text(p)


class TestChunkText:
    def test_chunk_text_returns_list(self, processor: DocumentProcessor) -> None:
        chunks = processor.chunk_text("Hello world " * 50)
        assert isinstance(chunks, (list, ChunkResult))

    def test_chunk_text_with_size(self, processor: DocumentProcessor) -> None:
        text = "word " * 500
        result = processor.chunk_text(text, chunk_size=500, chunk_overlap=50)
        if isinstance(result, ChunkResult):
            assert len(result.chunks) > 1
        else:
            assert len(result) > 1


class TestExtractChunks:
    def test_extract_chunks_txt(
        self, processor: DocumentProcessor, tmp_text_file: Path
    ) -> None:
        result = processor.extract_chunks(tmp_text_file, chunk_size=500)
        assert isinstance(result, ChunkResult)
        assert len(result) >= 1
        assert "Integration test content" in result[0]


class TestProcessorInit:
    def test_default_config(self) -> None:
        proc = DocumentProcessor()
        assert proc._config is not None

    def test_custom_config(self) -> None:
        config = ProcessingConfig()
        config = config.with_chunking(chunk_size=500)
        proc = DocumentProcessor(config=config)
        assert proc._config.chunking.chunk_size == 500

    def test_registry_populated(self) -> None:
        proc = DocumentProcessor()
        # Should have handlers for common formats
        assert proc._registry.is_supported("txt")
        assert proc._registry.is_supported("pdf")
