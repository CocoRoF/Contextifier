# tests/unit/services/test_image_service.py
"""Unit tests for ImageService — save, dedup, naming, tags, thread isolation."""

from __future__ import annotations

import threading
from typing import Optional
from unittest.mock import MagicMock

import pytest

from contextifier.config import ProcessingConfig
from contextifier.types import NamingStrategy
from contextifier.services.image_service import ImageService


@pytest.fixture()
def image_service(tmp_path) -> ImageService:
    """ImageService with a mock storage backend."""
    config = ProcessingConfig()
    storage = MagicMock()
    storage.save = MagicMock()
    tag_service = MagicMock()
    tag_service.create_image_tag.side_effect = lambda p: f"[Image:{p}]"
    return ImageService(
        config,
        storage_backend=storage,
        tag_service=tag_service,
    )


@pytest.fixture()
def sequential_service(tmp_path) -> ImageService:
    """ImageService with sequential naming strategy."""
    config = ProcessingConfig().with_images(naming_strategy=NamingStrategy.SEQUENTIAL)
    storage = MagicMock()
    storage.save = MagicMock()
    tag_service = MagicMock()
    tag_service.create_image_tag.side_effect = lambda p: f"[Image:{p}]"
    return ImageService(config, storage_backend=storage, tag_service=tag_service)


class TestBasicSave:
    def test_save_returns_path(self, image_service: ImageService) -> None:
        path = image_service.save(b"png-data", custom_name="test_img")
        assert path is not None
        assert "test_img" in path

    def test_save_and_tag_returns_tag(self, image_service: ImageService) -> None:
        tag = image_service.save_and_tag(b"png-data", custom_name="test_img")
        assert tag is not None
        assert "[Image:" in tag

    def test_empty_data_returns_none(self, image_service: ImageService) -> None:
        assert image_service.save(b"") is None
        assert image_service.save_and_tag(b"") is None

    def test_save_calls_storage_backend(self, image_service: ImageService) -> None:
        image_service.save(b"data", custom_name="file.png")
        image_service._storage.save.assert_called_once()
        call_args = image_service._storage.save.call_args
        assert call_args[0][0] == b"data"

    def test_save_storage_error_raises(self, image_service: ImageService) -> None:
        from contextifier.errors import ImageServiceError
        image_service._storage.save.side_effect = IOError("disk full")
        with pytest.raises(ImageServiceError, match="Failed to save"):
            image_service.save(b"data", custom_name="fail.png")


class TestDeduplication:
    def test_duplicate_skipped(self, image_service: ImageService) -> None:
        path1 = image_service.save(b"same-data", skip_duplicate=True)
        path2 = image_service.save(b"same-data", skip_duplicate=True)
        assert path1 is not None
        assert path2 is None  # duplicate

    def test_different_data_not_skipped(self, image_service: ImageService) -> None:
        path1 = image_service.save(b"data-A", skip_duplicate=True)
        path2 = image_service.save(b"data-B", skip_duplicate=True)
        assert path1 is not None
        assert path2 is not None

    def test_config_skip_duplicate_default(self, image_service: ImageService) -> None:
        """Default config has skip_duplicate=True."""
        path1 = image_service.save(b"dup-data")
        path2 = image_service.save(b"dup-data")
        assert path1 is not None
        assert path2 is None

    def test_skip_duplicate_false_allows_duplicates(self, image_service: ImageService) -> None:
        path1 = image_service.save(b"dup", skip_duplicate=False)
        path2 = image_service.save(b"dup", skip_duplicate=False)
        assert path1 is not None
        assert path2 is not None


class TestClearState:
    def test_clear_resets_counter(self, image_service: ImageService) -> None:
        image_service.save(b"data-1")
        assert image_service.get_processed_count() == 1
        image_service.clear_state()
        assert image_service.get_processed_count() == 0

    def test_clear_allows_duplicate_after(self, image_service: ImageService) -> None:
        image_service.save(b"dup-data", skip_duplicate=True)
        image_service.clear_state()
        path = image_service.save(b"dup-data", skip_duplicate=True)
        assert path is not None  # not a dup anymore

    def test_clear_resets_processed_paths(self, image_service: ImageService) -> None:
        image_service.save(b"data-1", custom_name="a.png")
        image_service.save(b"data-2", custom_name="b.png")
        assert len(image_service.get_processed_paths()) == 2
        image_service.clear_state()
        assert image_service.get_processed_paths() == []


class TestExtractAndDeduplicate:
    def test_returns_tag(self, image_service: ImageService) -> None:
        tag = image_service.extract_and_deduplicate(b"img-bytes", "docx")
        assert tag is not None
        assert "[Image:" in tag

    def test_duplicate_returns_cached_tag(self, image_service: ImageService) -> None:
        tag1 = image_service.extract_and_deduplicate(b"same", "docx")
        tag2 = image_service.extract_and_deduplicate(b"same", "docx")
        assert tag1 == tag2

    def test_empty_returns_none(self, image_service: ImageService) -> None:
        assert image_service.extract_and_deduplicate(b"", "docx") is None

    def test_different_data_different_tags(self, image_service: ImageService) -> None:
        tag1 = image_service.extract_and_deduplicate(b"img-A", "docx")
        tag2 = image_service.extract_and_deduplicate(b"img-B", "docx")
        assert tag1 != tag2


class TestNamingStrategy:
    def test_hash_naming_deterministic(self, image_service: ImageService) -> None:
        """Hash naming produces the same name for same data."""
        path1 = image_service.save(b"hash-test", skip_duplicate=False)
        image_service.clear_state()
        path2 = image_service.save(b"hash-test", skip_duplicate=False)
        assert path1 == path2

    def test_sequential_naming_increments(self, sequential_service: ImageService) -> None:
        path1 = sequential_service.save(b"d1", skip_duplicate=False)
        path2 = sequential_service.save(b"d2", skip_duplicate=False)
        assert "img_0001" in path1
        assert "img_0002" in path2

    def test_sequential_resets_on_clear(self, sequential_service: ImageService) -> None:
        sequential_service.save(b"d1", skip_duplicate=False)
        sequential_service.clear_state()
        path = sequential_service.save(b"d2", skip_duplicate=False)
        assert "img_0001" in path

    def test_custom_name_overrides_strategy(self, image_service: ImageService) -> None:
        path = image_service.save(b"data", custom_name="my_img.png")
        assert "my_img.png" in path


class TestTagBuilding:
    def test_tag_uses_tag_service(self, image_service: ImageService) -> None:
        tag = image_service.save_and_tag(b"data", custom_name="test.png")
        assert tag is not None
        image_service._tag_service.create_image_tag.assert_called()

    def test_tag_fallback_without_tag_service(self) -> None:
        """When no TagService, tags are built from config prefixes."""
        config = ProcessingConfig()
        storage = MagicMock()
        svc = ImageService(config, storage_backend=storage, tag_service=None)
        tag = svc.save_and_tag(b"data", custom_name="test.png")
        assert tag is not None
        assert "[Image:" in tag
        assert "test.png" in tag


class TestThreadIsolation:
    def test_threads_have_independent_state(self, image_service: ImageService) -> None:
        """Two threads saving with skip_duplicate should each see their own state."""
        results: dict[str, Optional[str]] = {}

        def worker(name: str) -> None:
            image_service.clear_state()
            path = image_service.save(b"shared-content", skip_duplicate=True)
            results[name] = path

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both threads should succeed (independent state)
        assert results["t1"] is not None
        assert results["t2"] is not None

    def test_thread_counter_independence(self) -> None:
        """Sequential counter is per-thread."""
        config = ProcessingConfig().with_images(naming_strategy=NamingStrategy.SEQUENTIAL)
        storage = MagicMock()
        svc = ImageService(config, storage_backend=storage, tag_service=None)

        paths: dict[str, str] = {}

        def worker(name: str) -> None:
            svc.clear_state()
            p = svc.save(b"data", skip_duplicate=False)
            paths[name] = p

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start(); t1.join()
        t2.start(); t2.join()

        # Both should get img_0001 (independent counters)
        assert "img_0001" in paths["t1"]
        assert "img_0001" in paths["t2"]
