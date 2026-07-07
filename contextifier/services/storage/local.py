# contextifier/services/storage/local.py
"""
LocalStorageBackend — Local Filesystem Storage

Concrete implementation that saves files to the local filesystem.
"""

from __future__ import annotations

import os

from contextifier.types import StorageType
from contextifier.errors import StorageError
from contextifier.services.storage.base import BaseStorageBackend


class LocalStorageBackend(BaseStorageBackend):
    """
    Stores files on the local filesystem.
    """

    def __init__(self, base_path: str = "temp/images") -> None:
        super().__init__(StorageType.LOCAL)
        self._base_path = base_path

    def save(self, data: bytes, file_path: str) -> bool:
        """Save data to a local file.

        The resolved *file_path* must reside under ``self._base_path``.
        If the path escapes the base directory (e.g. via ``../``),
        a :class:`StorageError` is raised to prevent path-traversal
        attacks.
        """
        try:
            resolved_base = os.path.realpath(self._base_path)
            resolved_path = os.path.realpath(file_path)

            # Path-traversal guard: resolved path must be inside base dir
            if (
                not resolved_path.startswith(resolved_base + os.sep)
                and resolved_path != resolved_base
            ):
                raise StorageError(
                    f"Path traversal blocked: {file_path!r} resolves outside base directory",
                )

            directory = os.path.dirname(resolved_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(resolved_path, "wb") as f:
                f.write(data)
            return True
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to save file: {file_path}",
                cause=e,
            )

    def delete(self, file_path: str) -> bool:
        """Delete a local file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            self._logger.warning(f"Failed to delete {file_path}: {e}")
            return False

    def exists(self, file_path: str) -> bool:
        """Check if a local file exists."""
        return os.path.exists(file_path)

    def ensure_ready(self, directory_path: str) -> None:
        """Create local directories if needed."""
        os.makedirs(directory_path, exist_ok=True)


__all__ = ["LocalStorageBackend"]
