# tests/unit/services/test_storage_local.py
"""Unit tests for LocalStorageBackend — specifically path-traversal defense."""

from __future__ import annotations

import os

import pytest

from contextifier.services.storage.local import LocalStorageBackend
from contextifier.errors import StorageError


@pytest.fixture()
def storage(tmp_path) -> LocalStorageBackend:
    """LocalStorageBackend rooted at a temporary directory."""
    base = str(tmp_path / "images")
    os.makedirs(base, exist_ok=True)
    return LocalStorageBackend(base_path=base)


class TestPathTraversal:
    """P0-3: file_path resolving outside base_path must be rejected."""

    def test_normal_save_succeeds(self, storage: LocalStorageBackend, tmp_path) -> None:
        """Saving directly under base_path works."""
        path = str(tmp_path / "images" / "photo.png")
        assert storage.save(b"PNG-DATA", path) is True
        assert os.path.isfile(path)

    def test_subdirectory_save_succeeds(self, storage: LocalStorageBackend, tmp_path) -> None:
        """Saving in a subdirectory under base_path works."""
        path = str(tmp_path / "images" / "sub" / "photo.png")
        assert storage.save(b"DATA", path) is True
        assert os.path.isfile(path)

    def test_traversal_blocked(self, storage: LocalStorageBackend, tmp_path) -> None:
        """A path containing ../ that escapes base_path is rejected."""
        evil_path = str(tmp_path / "images" / ".." / "evil.txt")
        with pytest.raises(StorageError, match="Path traversal blocked"):
            storage.save(b"EVIL", evil_path)

    def test_traversal_deep_blocked(self, storage: LocalStorageBackend, tmp_path) -> None:
        """Deeper traversal (../../) is also rejected."""
        evil_path = str(tmp_path / "images" / ".." / ".." / "evil.txt")
        with pytest.raises(StorageError, match="Path traversal blocked"):
            storage.save(b"EVIL", evil_path)

    def test_absolute_path_outside_base(self, storage: LocalStorageBackend, tmp_path) -> None:
        """An absolute path completely outside base_path is rejected."""
        outside_dir = str(tmp_path / "other_dir")
        os.makedirs(outside_dir, exist_ok=True)
        evil_path = os.path.join(outside_dir, "stolen.dat")
        with pytest.raises(StorageError, match="Path traversal blocked"):
            storage.save(b"EVIL", evil_path)
