# contextifier/services/image_service.py
"""
ImageService — Image Saving, Deduplication & Tag Generation

Replaces and unifies the old ImageProcessor (concrete class),
format-specific image processors (DOCXImageProcessor, PDFImageProcessor, etc.),
and the image-to-tag logic from ImageProcessor.save_image().

Design changes from old code:
1. Image SAVING (storage) is delegated to StorageBackend
2. Image TAGGING uses TagService for consistent tag format
3. Duplicate detection via content hashing
4. Format-specific image extraction logic stays in ContentExtractor,
   NOT in the service (separation of concerns)
5. The service is format-agnostic — it handles raw bytes

Separation of concerns:
    - ImageService handles:  save, deduplicate, generate filename, build tag
    - ContentExtractor handles: find/extract image bytes from format-specific source
    - StorageBackend handles: persist bytes to filesystem/cloud
    - TagService handles: tag format (prefix/suffix/pattern)

The format-specific image extraction (e.g., extracting images from
PDF pages, DOCX relationships, PPTX slides) is done by each
format's ContentExtractor. The ContentExtractor calls:
    path = image_service.save(image_data)       → str (saved path)
    tag  = image_service.save_and_tag(image_data) → str (complete tag)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from contextifier.config import ProcessingConfig, ImageConfig
from contextifier.types import NamingStrategy
from contextifier.errors import ImageServiceError

from contextifier.services.storage.base import BaseStorageBackend
from contextifier.services.storage.local import LocalStorageBackend

if TYPE_CHECKING:
    from contextifier.services.tag_service import TagService


class ImageService:
    """
    Shared service for saving images and generating image tags.

    Uses TagService for tag format consistency — the tag format
    is defined in ONE place (TagConfig) and applied uniformly.

    Thread-safe for concurrent handler use within one processor.

    Per-file state (hashes, paths, counter) is protected by a lock
    so that concurrent calls to save()/clear_state() do not race.
    """

    def __init__(
        self,
        config: ProcessingConfig,
        *,
        storage_backend: Optional[BaseStorageBackend] = None,
        tag_service: Optional["TagService"] = None,
    ) -> None:
        """
        Initialize ImageService.

        Args:
            config: Processing config containing ImageConfig and TagConfig.
            storage_backend: Storage backend for persisting images.
                             Defaults to LocalStorageBackend.
            tag_service: TagService for creating image tags.
                         If None, tags are built directly from config
                         (backward-compatible fallback).
        """
        self._config = config
        self._image_config: ImageConfig = config.images
        self._tag_config = config.tags
        self._tag_service = tag_service
        self._storage = storage_backend or LocalStorageBackend(
            base_path=self._image_config.directory_path
        )
        self._logger = logging.getLogger("contextifier.services.image")

        # Per-thread deduplication state.
        # Using threading.local ensures that concurrent calls from
        # different threads each get their own hash set / path list /
        # counter — without needing a global lock for every operation.
        self._local = threading.local()

    # ── Thread-local state helpers ────────────────────────────────────────

    def _ensure_state(self) -> None:
        """Lazily initialise per-thread state attributes."""
        local = self._local
        if not hasattr(local, "processed_hashes"):
            local.processed_hashes: Set[str] = set()
            local.hash_to_tag: Dict[str, str] = {}
            local.processed_paths: List[str] = []
            local.counter: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def save(
        self,
        image_data: bytes,
        *,
        custom_name: Optional[str] = None,
        skip_duplicate: Optional[bool] = None,
    ) -> Optional[str]:
        """
        Save image data and return the saved file path.

        This is the low-level method. Use save_and_tag() when you
        also need the formatted image tag.

        Args:
            image_data: Raw image bytes.
            custom_name: Override filename (instead of naming strategy).
            skip_duplicate: Override duplicate skipping. None = use config.

        Returns:
            Saved file path, or None if duplicate was skipped.

        Raises:
            ImageServiceError: If saving fails.
        """
        if not image_data:
            return None

        self._ensure_state()

        # Size limit check
        max_mb = self._image_config.max_file_size_mb
        if max_mb is not None and max_mb > 0:
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > max_mb:
                self._logger.warning(
                    "Image skipped: %.2f MB exceeds limit of %.2f MB",
                    size_mb, max_mb,
                )
                return None

        should_skip = (
            skip_duplicate if skip_duplicate is not None
            else self._image_config.skip_duplicate
        )

        # Duplicate check (per-thread state — no lock needed)
        if should_skip:
            content_hash = self._hash(image_data)
            if content_hash in self._local.processed_hashes:
                return None
            self._local.processed_hashes.add(content_hash)

        # Generate filename
        filename = custom_name or self._generate_filename(image_data)

        # Build full path
        file_path = f"{self._image_config.directory_path}/{filename}"

        # Save via storage backend
        try:
            self._storage.save(image_data, file_path)
        except Exception as e:
            raise ImageServiceError(
                f"Failed to save image: {e}",
                context={"path": file_path},
                cause=e,
            )

        self._local.processed_paths.append(file_path)
        return file_path

    def save_and_tag(
        self,
        image_data: bytes,
        *,
        custom_name: Optional[str] = None,
        skip_duplicate: Optional[bool] = None,
    ) -> Optional[str]:
        """
        Save image data and return the formatted image tag string.

        This is the primary method for ContentExtractors. It:
        1. Saves the image (with dedup check)
        2. Generates the tag using TagService (or fallback)

        Args:
            image_data: Raw image bytes.
            custom_name: Optional custom filename.
            skip_duplicate: Override duplicate skipping. None = use config.

        Returns:
            Image tag string (e.g., "[Image:path/to/img.png]"),
            or None if duplicate was skipped.
        """
        file_path = self.save(
            image_data,
            custom_name=custom_name,
            skip_duplicate=skip_duplicate,
        )
        if file_path is None:
            return None

        return self._build_tag(file_path)

    def get_processed_count(self) -> int:
        """Number of images processed in this session (current thread)."""
        self._ensure_state()
        return len(self._local.processed_paths)

    def get_processed_paths(self) -> List[str]:
        """List of all saved image paths (current thread)."""
        self._ensure_state()
        return list(self._local.processed_paths)

    def clear_state(self) -> None:
        """Reset deduplication state and counters for a new session (current thread)."""
        self._ensure_state()
        self._local.processed_hashes.clear()
        self._local.hash_to_tag.clear()
        self._local.processed_paths.clear()
        self._local.counter = 0

    def extract_and_deduplicate(
        self,
        image_bytes: bytes,
        source_hint: str,
    ) -> Optional[str]:
        """
        Unified content-hash dedup + save + tag generation.

        If the image was already saved (same content hash), returns the
        existing tag.  Otherwise saves, records the tag, and returns it.

        Args:
            image_bytes: Raw image bytes.
            source_hint: Short handler hint used in the filename
                         (e.g. ``"docx"``, ``"pptx_slide3"``).

        Returns:
            Image tag string, or None if the data is empty.
        """
        if not image_bytes:
            return None

        self._ensure_state()
        content_hash = self._hash(image_bytes)

        existing = self._local.hash_to_tag.get(content_hash)
        if existing is not None:
            return existing

        custom_name = f"{source_hint}_{content_hash[:8]}"
        tag = self.save_and_tag(
            image_bytes,
            custom_name=custom_name,
            skip_duplicate=True,
        )

        if tag:
            self._local.hash_to_tag[content_hash] = tag
        return tag

    # ── Private ───────────────────────────────────────────────────────────

    def _build_tag(self, file_path: str) -> str:
        """
        Build image tag for a saved file path.

        Delegates to TagService if available; otherwise falls back to
        direct prefix/suffix construction from config.
        """
        if self._tag_service is not None:
            return self._tag_service.create_image_tag(file_path)
        # Fallback: build directly from config (backward compat)
        return f"{self._tag_config.image_prefix}{file_path}{self._tag_config.image_suffix}"

    def _generate_filename(self, image_data: bytes) -> str:
        """Generate a filename using the configured naming strategy."""
        ext = self._image_config.default_format
        strategy = self._image_config.naming_strategy

        if strategy == NamingStrategy.HASH:
            name = hashlib.sha256(image_data).hexdigest()[:16]
        elif strategy == NamingStrategy.UUID:
            name = uuid.uuid4().hex[:16]
        elif strategy == NamingStrategy.SEQUENTIAL:
            self._ensure_state()
            self._local.counter += 1
            name = f"img_{self._local.counter:04d}"
        elif strategy == NamingStrategy.TIMESTAMP:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"img_{ts}"
        else:
            name = hashlib.sha256(image_data).hexdigest()[:16]

        return f"{name}.{ext}"

    @staticmethod
    def _hash(data: bytes) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(data).hexdigest()


__all__ = ["ImageService"]
