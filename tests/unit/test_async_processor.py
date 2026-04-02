# tests/unit/test_async_processor.py
"""P3-5: Unit tests for AsyncDocumentProcessor — async extraction, batch, concurrency."""

from __future__ import annotations

from pathlib import Path

import pytest

from contextifier.async_processor import AsyncDocumentProcessor
from contextifier.config import ProcessingConfig
from contextifier.types import ExtractionResult


@pytest.fixture()
def async_proc() -> AsyncDocumentProcessor:
    return AsyncDocumentProcessor()


# ── extract_text ─────────────────────────────────────────────────────────

class TestExtractText:
    @pytest.mark.asyncio
    async def test_extract_text_basic(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("Hello async", encoding="utf-8")
        proc = AsyncDocumentProcessor()
        result = await proc.extract_text(str(f))
        assert "Hello async" in result

    @pytest.mark.asyncio
    async def test_extract_text_missing_file(self) -> None:
        proc = AsyncDocumentProcessor()
        with pytest.raises(Exception):
            await proc.extract_text("/nonexistent/file.txt")


# ── process ──────────────────────────────────────────────────────────────

class TestProcess:
    @pytest.mark.asyncio
    async def test_process_returns_extraction_result(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Process test", encoding="utf-8")
        proc = AsyncDocumentProcessor()
        result = await proc.process(str(f))
        assert isinstance(result, ExtractionResult)
        assert "Process test" in result.text


# ── extract_chunks ───────────────────────────────────────────────────────

class TestExtractChunks:
    @pytest.mark.asyncio
    async def test_extract_chunks_basic(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.txt"
        f.write_text("A" * 500, encoding="utf-8")
        proc = AsyncDocumentProcessor()
        result = await proc.extract_chunks(str(f), chunk_size=200)
        assert len(result.chunks) >= 1


# ── extract_batch ────────────────────────────────────────────────────────

class TestExtractBatch:
    @pytest.mark.asyncio
    async def test_batch_all_success(self, tmp_path: Path) -> None:
        files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"Content {i}", encoding="utf-8")
            files.append(str(f))

        proc = AsyncDocumentProcessor()
        results = await proc.extract_batch(files, max_concurrent=2)
        assert len(results) == 3
        for path in files:
            assert isinstance(results[path], str)

    @pytest.mark.asyncio
    async def test_batch_with_failure(self, tmp_path: Path) -> None:
        good = tmp_path / "good.txt"
        good.write_text("OK", encoding="utf-8")
        bad = "/nonexistent/bad.txt"

        proc = AsyncDocumentProcessor()
        results = await proc.extract_batch([str(good), bad])
        assert isinstance(results[str(good)], str)
        assert isinstance(results[bad], Exception)

    @pytest.mark.asyncio
    async def test_batch_empty(self) -> None:
        proc = AsyncDocumentProcessor()
        results = await proc.extract_batch([])
        assert results == {}


# ── Utility ──────────────────────────────────────────────────────────────

class TestUtility:
    def test_is_supported(self, async_proc: AsyncDocumentProcessor) -> None:
        assert async_proc.is_supported("txt") is True
        assert async_proc.is_supported("zzz") is False

    def test_config_property(self) -> None:
        cfg = ProcessingConfig()
        proc = AsyncDocumentProcessor(config=cfg)
        assert proc.config is cfg
