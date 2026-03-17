# contextifier_new/services/storage/base.py
"""
BaseStorageBackend — Abstract Storage Interface

Defines the contract for all storage backends:
- save(data, path) → bool
- delete(path) → bool
- exists(path) → bool
- ensure_ready(directory) → None

Improvements over old code:
- build_url() returns a proper URL/path for the stored file
- All methods have consistent error handling
- Storage type is always identifiable
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from contextifier_new.types import StorageType


class BaseStorageBackend(ABC):
    """
    Abstract base for all storage backends.
    """

    def __init__(self, storage_type: StorageType) -> None:
        self._storage_type = storage_type
        self._logger = logging.getLogger(
            f"contextifier.storage.{self.__class__.__name__}"
        )

    @abstractmethod
    def save(self, data: bytes, file_path: str) -> bool:
        """
        Save data to the specified path.

        Args:
            data: Binary data to save.
            file_path: Target path.

        Returns:
            True if saved successfully.

        Raises:
            StorageError: If save fails.
        """
        ...

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """
        Delete a file at the specified path.

        Args:
            file_path: Path of file to delete.

        Returns:
            True if deleted successfully.
        """
        ...

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """
        Check if a file exists at the specified path.

        Args:
            file_path: Path to check.

        Returns:
            True if file exists.
        """
        ...

    @abstractmethod
    def ensure_ready(self, directory_path: str) -> None:
        """
        Ensure the directory/bucket is ready for writing.

        For local: creates directories.
        For cloud: validates bucket/container access.

        Args:
            directory_path: Directory or bucket path.
        """
        ...

    def build_url(self, file_path: str) -> str:
        """
        Build a URL or normalized path for the stored file.

        Default: returns forward-slash normalized path.
        Override for cloud backends to return full URLs.

        Args:
            file_path: Storage path.

        Returns:
            URL or normalized path string.
        """
        return file_path.replace("\\", "/")

    @property
    def storage_type(self) -> StorageType:
        """The type of this storage backend."""
        return self._storage_type


__all__ = ["BaseStorageBackend"]
