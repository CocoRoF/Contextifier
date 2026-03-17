# contextifier_new/services/storage/local.py
"""
LocalStorageBackend — Local Filesystem Storage

Concrete implementation that saves files to the local filesystem.
"""

from __future__ import annotations

import os

from contextifier_new.types import StorageType
from contextifier_new.errors import StorageError
from contextifier_new.services.storage.base import BaseStorageBackend


class LocalStorageBackend(BaseStorageBackend):
    """
    Stores files on the local filesystem.
    """

    def __init__(self, base_path: str = "temp/images") -> None:
        super().__init__(StorageType.LOCAL)
        self._base_path = base_path

    def save(self, data: bytes, file_path: str) -> bool:
        """Save data to a local file."""
        try:
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(data)
            return True
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
