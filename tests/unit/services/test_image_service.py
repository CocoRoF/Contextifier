# tests/unit/services/test_image_service.py
"""Unit tests for ImageService — especially thread-local state."""

from __future__ import annotations

import threading
from typing import Optional
from unittest.mock import MagicMock

import pytest

from contextifier.config import ProcessingConfig
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
