# tests/integration/test_document_processor.py
"""Integration tests for DocumentProcessor end-to-end flow.

These tests require minimal real files but exercise the full pipeline
from DocumentProcessor.extract_text() through handler → pipeline → result.
"""

from __future__ import annotations

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


# ═══════════════════════════════════════════════════════════════════════════════
# P3-7: Expanded integration test framework
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessEndToEnd:
    """Full pipeline via processor.process() — returns ExtractionResult."""

    def test_process_returns_extraction_result(
        self, processor: DocumentProcessor, tmp_text_file: Path
    ) -> None:
        result = processor.process(tmp_text_file)
        from contextifier.types import ExtractionResult
        assert isinstance(result, ExtractionResult)
        assert "Integration test content" in result.text

    def test_process_metadata_included(
        self, processor: DocumentProcessor, tmp_text_file: Path
    ) -> None:
        result = processor.process(tmp_text_file, extract_metadata=True)
        assert result.text  # non-empty


class TestRoundTripConsistency:
    """Same file should produce identical output across repeated calls."""

    def test_deterministic_extraction(
        self, processor: DocumentProcessor, tmp_text_file: Path
    ) -> None:
        r1 = processor.extract_text(tmp_text_file)
        r2 = processor.extract_text(tmp_text_file)
        assert r1 == r2

    def test_deterministic_chunks(
        self, processor: DocumentProcessor, tmp_text_file: Path
    ) -> None:
        c1 = processor.extract_chunks(tmp_text_file, chunk_size=500)
        c2 = processor.extract_chunks(tmp_text_file, chunk_size=500)
        assert len(c1.chunks) == len(c2.chunks)
        for a, b in zip(c1.chunks, c2.chunks):
            assert a == b


class TestConfigIntegration:
    """Verify config changes affect output end-to-end."""

    def test_metadata_language_ko(self, tmp_text_file: Path) -> None:
        config = ProcessingConfig().with_metadata(language="ko")
        proc = DocumentProcessor(config=config)
        text = proc.extract_text(tmp_text_file)
        assert isinstance(text, str)

    def test_metadata_language_en(self, tmp_text_file: Path) -> None:
        config = ProcessingConfig().with_metadata(language="en")
        proc = DocumentProcessor(config=config)
        text = proc.extract_text(tmp_text_file)
        assert isinstance(text, str)

    def test_chunk_size_affects_count(self, tmp_path: Path) -> None:
        f = tmp_path / "long.txt"
        f.write_text("word " * 2000, encoding="utf-8")
        proc = DocumentProcessor()
        small = proc.extract_chunks(f, chunk_size=200)
        large = proc.extract_chunks(f, chunk_size=2000)
        assert len(small.chunks) > len(large.chunks)


class TestMultiFormatSupport:
    """Verify extraction works across text-like formats."""

    def test_csv_extraction(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        proc = DocumentProcessor()
        text = proc.extract_text(f)
        assert "Alice" in text
        assert "Bob" in text

    def test_tsv_extraction(self, tmp_path: Path) -> None:
        f = tmp_path / "data.tsv"
        f.write_text("name\tage\nAlice\t30", encoding="utf-8")
        proc = DocumentProcessor()
        text = proc.extract_text(f)
        assert "Alice" in text

    def test_html_extraction(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text(
            "<html><body><p>Hello HTML</p></body></html>",
            encoding="utf-8",
        )
        proc = DocumentProcessor()
        text = proc.extract_text(f)
        assert "Hello HTML" in text

    def test_json_extraction(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        proc = DocumentProcessor()
        text = proc.extract_text(f)
        assert "key" in text or "value" in text


class TestSupportedExtensions:
    """Verify supported_extensions property."""

    def test_common_formats_supported(self) -> None:
        proc = DocumentProcessor()
        for ext in ("txt", "pdf", "docx", "xlsx", "pptx", "csv", "html"):
            assert proc.is_supported(ext), f"{ext} should be supported"

    def test_unknown_format_not_supported(self) -> None:
        proc = DocumentProcessor()
        assert not proc.is_supported("zzzzz")
