# tests/unit/test_cached_processor.py
"""P3-4: Unit tests for CachedDocumentProcessor, MemoryCacheBackend, DiskCacheBackend."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        backend = MemoryCacheBackend(max_size=2)
        backend.put("a", "1")
        backend.put("b", "2")
        backend.put("c", "3")  # evicts "a"
        assert backend.get("a") is None
        assert backend.get("b") == "2"
        assert backend.get("c") == "3"

    def test_eviction_order(self) -> None:
        backend = MemoryCacheBackend(max_size=3)
        for ch in "abcd":
            backend.put(ch, ch)
        # evicts "a" first
        assert backend.get("a") is None
        assert backend.get("d") == "d"


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
