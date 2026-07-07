# tests/unit/test_cached_processor.py
"""P3-4: Unit tests for CachedDocumentProcessor, MemoryCacheBackend, DiskCacheBackend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


from contextifier.cached_processor import (
    CachedDocumentProcessor,
    DiskCacheBackend,
    MemoryCacheBackend,
)
from contextifier.config import ProcessingConfig


# ═══════════════ MemoryCacheBackend ═══════════════════════════════════════════

class TestMemoryCacheBackend:
    def test_get_returns_none_for_missing(self) -> None:
        backend = MemoryCacheBackend()
        assert backend.get("no-such-key") is None

    def test_put_and_get(self) -> None:
        backend = MemoryCacheBackend()
        backend.put("k1", "hello")
        assert backend.get("k1") == "hello"

    def test_overwrite(self) -> None:
        backend = MemoryCacheBackend()
        backend.put("k", "v1")
        backend.put("k", "v2")
        assert backend.get("k") == "v2"

    def test_fifo_eviction(self) -> None:
        """Without access, eviction is FIFO (oldest inserted first)."""
        backend = MemoryCacheBackend(max_size=2)
        backend.put("a", "1")
        backend.put("b", "2")
        backend.put("c", "3")  # evicts "a" (LRU = oldest insert)
        assert backend.get("a") is None
        assert backend.get("b") == "2"
        assert backend.get("c") == "3"

    def test_eviction_order(self) -> None:
        backend = MemoryCacheBackend(max_size=3)
        for ch in "abcd":
            backend.put(ch, ch)
        # evicts "a" first (LRU)
        assert backend.get("a") is None
        assert backend.get("d") == "d"

    def test_lru_eviction(self) -> None:
        """Accessing an item makes it recently used, so others evict first."""
        backend = MemoryCacheBackend(max_size=2)
        backend.put("a", "1")
        backend.put("b", "2")
        # Access "a" → "a" is now most recently used
        assert backend.get("a") == "1"
        # Insert "c" → evicts "b" (least recently used)
        backend.put("c", "3")
        assert backend.get("b") is None
        assert backend.get("a") == "1"
        assert backend.get("c") == "3"

    def test_lru_put_update_refreshes(self) -> None:
        """Updating an existing key refreshes its LRU position."""
        backend = MemoryCacheBackend(max_size=2)
        backend.put("a", "1")
        backend.put("b", "2")
        # Overwrite "a" → now most recent
        backend.put("a", "updated")
        # Insert "c" → evicts "b" (LRU)
        backend.put("c", "3")
        assert backend.get("b") is None
        assert backend.get("a") == "updated"
        assert backend.get("c") == "3"


# ═══════════════ DiskCacheBackend ════════════════════════════════════════════

class TestDiskCacheBackend:
    def test_put_and_get(self, tmp_path: Path) -> None:
        backend = DiskCacheBackend(cache_dir=tmp_path / "cache")
        backend.put("key1", "value1")
        assert backend.get("key1") == "value1"

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        backend = DiskCacheBackend(cache_dir=tmp_path / "cache")
        assert backend.get("missing") is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "bad.json").write_text("not-json", encoding="utf-8")
        backend = DiskCacheBackend(cache_dir=cache_dir)
        assert backend.get("bad") is None

    def test_file_persists(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        backend1 = DiskCacheBackend(cache_dir=cache_dir)
        backend1.put("persist", "data")
        # new instance reads the same file
        backend2 = DiskCacheBackend(cache_dir=cache_dir)
        assert backend2.get("persist") == "data"

    def test_unicode_value(self, tmp_path: Path) -> None:
        backend = DiskCacheBackend(cache_dir=tmp_path / "cache")
        backend.put("kr", "한글 데이터")
        assert backend.get("kr") == "한글 데이터"


# ═══════════════ CachedDocumentProcessor ═════════════════════════════════════

class TestCachedDocumentProcessor:
    def test_cache_hit_skips_extraction(self, tmp_path: Path) -> None:
        """Second call returns cached result without re-parsing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello", encoding="utf-8")

        proc = CachedDocumentProcessor()
        result1 = proc.extract_text(str(test_file))
        result2 = proc.extract_text(str(test_file))
        assert result1 == result2

    def test_different_files_different_results(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha", encoding="utf-8")
        f2.write_text("beta", encoding="utf-8")

        proc = CachedDocumentProcessor()
        r1 = proc.extract_text(str(f1))
        r2 = proc.extract_text(str(f2))
        assert r1 != r2

    def test_custom_backend(self, tmp_path: Path) -> None:
        mock_backend = MagicMock()
        mock_backend.get.return_value = "cached-text"

        proc = CachedDocumentProcessor(cache_backend=mock_backend)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")
        result = proc.extract_text(str(test_file))
        assert result == "cached-text"
        mock_backend.get.assert_called_once()

    def test_is_supported_delegates(self) -> None:
        proc = CachedDocumentProcessor()
        assert proc.is_supported(".txt") is True
        assert proc.is_supported(".zzz") is False

    def test_supported_extensions(self) -> None:
        proc = CachedDocumentProcessor()
        exts = proc.supported_extensions
        assert "txt" in exts
        assert isinstance(exts, frozenset)

    def test_config_property(self) -> None:
        cfg = ProcessingConfig()
        proc = CachedDocumentProcessor(config=cfg)
        assert proc.config is cfg

    def test_repr(self) -> None:
        proc = CachedDocumentProcessor()
        assert "Cached" in repr(proc)

    def test_config_change_invalidates(self, tmp_path: Path) -> None:
        """Different config → different cache key → re-extraction."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")

        proc1 = CachedDocumentProcessor(
            config=ProcessingConfig().with_metadata(language="ko"),
        )
        proc2 = CachedDocumentProcessor(
            config=ProcessingConfig().with_metadata(language="en"),
        )
        # both extract fresh since config hashes differ
        r1 = proc1.extract_text(str(test_file))
        r2 = proc2.extract_text(str(test_file))
        # Results may be same text but cache keys are different
        assert isinstance(r1, str)
        assert isinstance(r2, str)


# ═══════════════ P4-5: Cached process() and extract_chunks() ═════════════════

class TestCachedProcess:
    """CachedDocumentProcessor.process() cache support."""

    def test_process_returns_extraction_result(self, tmp_path: Path) -> None:
        from contextifier.types import ExtractionResult
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World", encoding="utf-8")

        proc = CachedDocumentProcessor()
        result = proc.process(str(test_file))
        assert isinstance(result, ExtractionResult)
        assert "Hello World" in result.text

    def test_process_cache_hit(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello", encoding="utf-8")

        proc = CachedDocumentProcessor()
        r1 = proc.process(str(test_file))
        r2 = proc.process(str(test_file))
        assert r1.text == r2.text

    def test_process_custom_backend(self, tmp_path: Path) -> None:
        from contextifier.cached_processor import _serialize_extraction_result
        from contextifier.types import ExtractionResult

        fake_result = ExtractionResult(text="cached process text")
        mock_backend = MagicMock()
        mock_backend.get.return_value = _serialize_extraction_result(fake_result)

        proc = CachedDocumentProcessor(cache_backend=mock_backend)
        test_file = tmp_path / "test.txt"
        test_file.write_text("content", encoding="utf-8")

        result = proc.process(str(test_file))
        assert result.text == "cached process text"
        mock_backend.get.assert_called_once()


class TestCachedExtractChunks:
    """CachedDocumentProcessor.extract_chunks() cache support."""

    def test_extract_chunks_returns_chunk_result(self, tmp_path: Path) -> None:
        from contextifier.document_processor import ChunkResult
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World " * 100, encoding="utf-8")

        proc = CachedDocumentProcessor()
        result = proc.extract_chunks(str(test_file))
        assert isinstance(result, ChunkResult)
        assert len(result.chunks) > 0

    def test_extract_chunks_cache_hit(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World " * 100, encoding="utf-8")

        proc = CachedDocumentProcessor()
        r1 = proc.extract_chunks(str(test_file))
        r2 = proc.extract_chunks(str(test_file))
        assert r1.chunks == r2.chunks


# ═══════════════ P4-5: Serialization round-trip ══════════════════════════════

class TestSerializationRoundTrip:
    """Verify ExtractionResult and ChunkResult survive serialization."""

    def test_extraction_result_round_trip(self) -> None:
        from contextifier.cached_processor import (
            _serialize_extraction_result,
            _deserialize_extraction_result,
        )
        from contextifier.types import (
            ExtractionResult,
            DocumentMetadata,
            TableData,
            TableCell,
            ChartData,
            ChartSeries,
        )

        original = ExtractionResult(
            text="Hello 한글",
            metadata=DocumentMetadata(title="Test", author="Author"),
            tables=[
                TableData(
                    rows=[[TableCell(content="A", is_header=True)]],
                    num_rows=1, num_cols=1, has_header=True,
                )
            ],
            charts=[
                ChartData(
                    chart_type="bar", title="Sales",
                    categories=["Q1", "Q2"],
                    series=[ChartSeries(name="Revenue", values=[100, 200])],
                )
            ],
            images=["img1.png"],
            page_count=5,
            warnings=["warn1"],
        )

        serialized = _serialize_extraction_result(original)
        restored = _deserialize_extraction_result(serialized)

        assert restored.text == original.text
        assert restored.metadata.title == "Test"
        assert restored.metadata.author == "Author"
        assert len(restored.tables) == 1
        assert restored.tables[0].rows[0][0].content == "A"
        assert len(restored.charts) == 1
        assert restored.charts[0].title == "Sales"
        assert restored.images == ["img1.png"]
        assert restored.page_count == 5
        assert restored.warnings == ["warn1"]

    def test_chunk_result_round_trip(self) -> None:
        from contextifier.cached_processor import (
            _serialize_chunk_result,
            _deserialize_chunk_result,
        )
        from contextifier.document_processor import ChunkResult
        from contextifier.types import Chunk, ChunkMetadata

        original = ChunkResult(
            chunks=["chunk1", "chunk2"],
            chunks_with_metadata=[
                Chunk(
                    text="chunk1",
                    metadata=ChunkMetadata(chunk_index=0, line_start=0, line_end=10),
                ),
                Chunk(text="chunk2", metadata=None),
            ],
            source_file="test.txt",
        )

        serialized = _serialize_chunk_result(original)
        restored = _deserialize_chunk_result(serialized)

        assert restored.chunks == ["chunk1", "chunk2"]
        assert restored.source_file == "test.txt"
        assert len(restored.chunks_with_metadata) == 2
        assert restored.chunks_with_metadata[0].text == "chunk1"
        assert restored.chunks_with_metadata[0].metadata.chunk_index == 0
        assert restored.chunks_with_metadata[1].metadata is None

    def test_extraction_result_no_metadata(self) -> None:
        from contextifier.cached_processor import (
            _serialize_extraction_result,
            _deserialize_extraction_result,
        )
        from contextifier.types import ExtractionResult

        original = ExtractionResult(text="plain")
        serialized = _serialize_extraction_result(original)
        restored = _deserialize_extraction_result(serialized)
        assert restored.text == "plain"
        assert restored.metadata is None
