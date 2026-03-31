# tests/unit/services/test_tag_service.py
"""Unit tests for TagService."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig
from contextifier.services.tag_service import TagService


@pytest.fixture()
def tag_service() -> TagService:
    return TagService(ProcessingConfig())


class TestPageTags:
    def test_create_page_tag(self, tag_service: TagService) -> None:
        tag = tag_service.create_page_tag(1)
        assert "1" in tag

    def test_find_page_tags(self, tag_service: TagService) -> None:
        tag = tag_service.create_page_tag(3)
        text = f"Some text\n{tag}\nMore text"
        results = tag_service.find_page_tags(text)
        assert len(results) >= 1
        # result tuples contain (start, end, page_number)
        assert any(r[2] == 3 for r in results)


class TestSlideTags:
    def test_create_slide_tag(self, tag_service: TagService) -> None:
        tag = tag_service.create_slide_tag(5)
        assert "5" in tag


class TestSheetTags:
    def test_create_sheet_tag(self, tag_service: TagService) -> None:
        tag = tag_service.create_sheet_tag("Summary")
        assert "Summary" in tag


class TestRemoveMarkers:
    def test_remove_page_slide_sheet(self, tag_service: TagService) -> None:
        page = tag_service.create_page_tag(1)
        slide = tag_service.create_slide_tag(2)
        text = f"A{page}B{slide}C"
        cleaned = tag_service.remove_page_slide_sheet_markers(text)
        assert page not in cleaned
        assert slide not in cleaned
        assert "ABC" in cleaned.replace("\n", "").replace(" ", "")

    def test_deprecated_alias(self, tag_service: TagService) -> None:
        # Class-level alias: bound methods are not identical objects but
        # should produce the same result.
        text = tag_service.create_page_tag(1) + "hello"
        a = tag_service.remove_all_structural_markers(text)
        b = tag_service.remove_page_slide_sheet_markers(text)
        assert a == b
