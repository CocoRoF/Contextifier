# tests/unit/test_security.py
"""P3-8: Consolidated security tests for Contextifier.

Covers:
- File size limits (DoS prevention)
- ZIP bomb defense (decompression bomb prevention)
- Path traversal defense in storage
- Delegation depth limit (infinite loop prevention)
- Input boundary safety (null bytes, oversized metadata)
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextifier.config import ProcessingConfig
from contextifier.document_processor import DocumentProcessor
from contextifier.errors import ConversionError, FileReadError
from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import check_zip_bomb, MAX_ZIP_DECOMPRESSED_BYTES
from contextifier.services.storage.local import LocalStorageBackend
from contextifier.errors import StorageError


# ═══════════════════════════════════════════════════════════════════════════════
# File Size Limits
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileSizeLimits:
    def test_oversized_file_rejected(self, tmp_path: Path) -> None:
        """Files exceeding _MAX_FILE_SIZE are rejected before loading."""
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 100)

        # Use a tiny limit to trigger the check
        with pytest.raises(FileReadError, match="exceeds the limit"):
            DocumentProcessor._create_file_context(str(f), "txt", max_file_size=50)

    def test_file_within_limit_loads(self, tmp_path: Path) -> None:
        f = tmp_path / "small.txt"
        f.write_bytes(b"ok")
        ctx = DocumentProcessor._create_file_context(str(f), "txt", max_file_size=1000)
        assert ctx["file_data"] == b"ok"

    def test_zero_limit_disables_check(self, tmp_path: Path) -> None:
        f = tmp_path / "any.txt"
        f.write_bytes(b"x" * 1000)
        ctx = DocumentProcessor._create_file_context(str(f), "txt", max_file_size=0)
        assert len(ctx["file_data"]) == 1000


# ═══════════════════════════════════════════════════════════════════════════════
# ZIP Bomb Defense
# ═══════════════════════════════════════════════════════════════════════════════

def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestZipBombDefense:
    def test_small_archive_passes(self) -> None:
        raw = _make_zip({"a.txt": b"hi"})
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            check_zip_bomb(zf, handler="test")  # no raise

    def test_oversized_archive_blocked(self) -> None:
        raw = _make_zip({"big.bin": b"\x00" * 500})
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with pytest.raises(ConversionError, match="ZIP bomb detected"):
                check_zip_bomb(zf, max_bytes=100, handler="test")

    def test_cumulative_size_checked(self) -> None:
        """Multiple small entries that exceed limit in aggregate."""
        raw = _make_zip({f"f{i}.bin": b"\x00" * 40 for i in range(5)})
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with pytest.raises(ConversionError, match="ZIP bomb detected"):
                check_zip_bomb(zf, max_bytes=150, handler="test")

    def test_default_limit_is_1gb(self) -> None:
        assert MAX_ZIP_DECOMPRESSED_BYTES == 1 * 1024 * 1024 * 1024


# ═══════════════════════════════════════════════════════════════════════════════
# Path Traversal Defense
# ═══════════════════════════════════════════════════════════════════════════════

class TestPathTraversalDefense:
    def test_dotdot_blocked(self, tmp_path: Path) -> None:
        backend = LocalStorageBackend(str(tmp_path))
        with pytest.raises(StorageError, match="Path traversal blocked"):
            backend.save(b"data", "../escape.txt")

    def test_absolute_path_blocked(self, tmp_path: Path) -> None:
        backend = LocalStorageBackend(str(tmp_path))
        if os.name == "nt":
            evil_path = "C:\\Windows\\temp\\evil.txt"
        else:
            evil_path = "/tmp/evil.txt"
        with pytest.raises(StorageError, match="Path traversal blocked"):
            backend.save(b"data", evil_path)

    def test_nested_dotdot_blocked(self, tmp_path: Path) -> None:
        backend = LocalStorageBackend(str(tmp_path))
        with pytest.raises(StorageError, match="Path traversal blocked"):
            backend.save(b"data", "sub/../../escape.txt")

    def test_normal_subdir_allowed(self, tmp_path: Path) -> None:
        backend = LocalStorageBackend(str(tmp_path))
        full_path = str(tmp_path / "subdir" / "file.txt")
        result = backend.save(b"data", full_path)
        assert (tmp_path / "subdir" / "file.txt").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Delegation Depth Limit (Infinite Loop Prevention)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDelegationDepthSecurity:
    def test_max_depth_constant_reasonable(self) -> None:
        assert 1 <= BaseHandler._MAX_DELEGATION_DEPTH <= 10

    def test_depth_overflow_prevented(self) -> None:
        """Simulating depth at max should block further delegation."""
        from contextifier.errors import HandlerExecutionError

        class _Stub(BaseHandler):
            @property
            def supported_extensions(self):
                return frozenset({"stub"})
            @property
            def handler_name(self):
                return "Stub"
            def create_converter(self):
                return MagicMock()
            def create_preprocessor(self):
                return MagicMock()
            def create_metadata_extractor(self):
                return MagicMock()
            def create_content_extractor(self):
                return MagicMock()
            def create_postprocessor(self):
                return MagicMock()

        h = _Stub(ProcessingConfig())
        h.set_registry(MagicMock())
        BaseHandler._delegation_state.depth = BaseHandler._MAX_DELEGATION_DEPTH
        try:
            with pytest.raises(HandlerExecutionError, match="Delegation depth limit"):
                h._delegate_to("rtf", {"file_data": b"test"}, fallback_to_self=False)
        finally:
            BaseHandler._delegation_state.depth = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Input Boundary Safety
# ═══════════════════════════════════════════════════════════════════════════════

class TestInputBoundary:
    def test_null_bytes_in_text_handled(self, tmp_path: Path) -> None:
        """Files containing null bytes should not crash extraction."""
        f = tmp_path / "nulls.txt"
        f.write_bytes(b"Hello\x00World")
        proc = DocumentProcessor()
        text = proc.extract_text(str(f))
        assert isinstance(text, str)

    def test_empty_file_returns_empty_text(self, tmp_path: Path) -> None:
        """Empty files return empty text instead of crashing."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        proc = DocumentProcessor()
        text = proc.extract_text(str(f))
        assert isinstance(text, str)
        assert text.strip() == ""

    def test_binary_garbage_as_txt(self, tmp_path: Path) -> None:
        """Random binary data with .txt extension should not crash."""
        f = tmp_path / "garbage.txt"
        f.write_bytes(bytes(range(256)))
        proc = DocumentProcessor()
        text = proc.extract_text(str(f))
        assert isinstance(text, str)

    def test_nonexistent_file_raises(self) -> None:
        proc = DocumentProcessor()
        with pytest.raises(Exception):
            proc.extract_text("/absolutely/nonexistent/path.txt")
