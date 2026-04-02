# tests/unit/services/test_metadata_service.py
"""P3-3: Unit tests for MetadataService — formatting, language labels, date handling."""

from __future__ import annotations

from datetime import datetime

import pytest

from contextifier.config import ProcessingConfig
from contextifier.services.metadata_service import MetadataService
from contextifier.types import DocumentMetadata


@pytest.fixture()
def svc_ko() -> MetadataService:
    return MetadataService(ProcessingConfig())


@pytest.fixture()
def svc_en() -> MetadataService:
    return MetadataService(ProcessingConfig().with_metadata(language="en"))


# ── Basic formatting ─────────────────────────────────────────────────────

class TestFormatMetadata:
    def test_full_metadata_ko(self, svc_ko: MetadataService) -> None:
        meta = DocumentMetadata(
            title="테스트 문서",
            author="홍길동",
            create_time=datetime(2024, 3, 15, 10, 30, 0),
            page_count=42,
        )
        result = svc_ko.format_metadata(meta)
        assert "[Document-Metadata]" in result
        assert "[/Document-Metadata]" in result
        assert "제목: 테스트 문서" in result
        assert "작성자: 홍길동" in result
        assert "작성일: 2024-03-15 10:30:00" in result
        assert "페이지 수: 42" in result

    def test_full_metadata_en(self, svc_en: MetadataService) -> None:
        meta = DocumentMetadata(
            title="Test Doc",
            author="John",
            page_count=10,
        )
        result = svc_en.format_metadata(meta)
        assert "Title: Test Doc" in result
        assert "Author: John" in result
        assert "Page Count: 10" in result

    def test_none_returns_empty(self, svc_ko: MetadataService) -> None:
        assert svc_ko.format_metadata(None) == ""

    def test_empty_metadata_returns_empty(self, svc_ko: MetadataService) -> None:
        assert svc_ko.format_metadata(DocumentMetadata()) == ""

    def test_custom_fields_appended(self, svc_ko: MetadataService) -> None:
        meta = DocumentMetadata(
            title="Doc",
            custom={"소스": "내부 시스템", "priority": "high"},
        )
        result = svc_ko.format_metadata(meta)
        assert "소스: 내부 시스템" in result
        assert "priority: high" in result

    def test_field_order(self, svc_ko: MetadataService) -> None:
        """Standard fields appear before custom fields."""
        meta = DocumentMetadata(
            title="A",
            author="B",
            custom={"extra": "C"},
        )
        result = svc_ko.format_metadata(meta)
        title_pos = result.index("제목: A")
        author_pos = result.index("작성자: B")
        extra_pos = result.index("extra: C")
        assert title_pos < author_pos < extra_pos


# ── Value formatting ─────────────────────────────────────────────────────

class TestValueFormatting:
    def test_datetime_formatted(self, svc_ko: MetadataService) -> None:
        meta = DocumentMetadata(
            last_saved_time=datetime(2025, 1, 1, 0, 0, 0),
        )
        result = svc_ko.format_metadata(meta)
        assert "2025-01-01 00:00:00" in result

    def test_custom_date_format(self) -> None:
        svc = MetadataService(
            ProcessingConfig().with_metadata(date_format="%Y/%m/%d"),
        )
        meta = DocumentMetadata(create_time=datetime(2024, 6, 15, 12, 0))
        result = svc.format_metadata(meta)
        assert "2024/06/15" in result

    def test_integer_value(self, svc_ko: MetadataService) -> None:
        meta = DocumentMetadata(word_count=500)
        result = svc_ko.format_metadata(meta)
        assert "단어 수: 500" in result

    def test_whitespace_only_value_skipped(self, svc_ko: MetadataService) -> None:
        meta = DocumentMetadata(title="  ", author="Valid")
        result = svc_ko.format_metadata(meta)
        assert "제목" not in result
        assert "작성자: Valid" in result

    def test_zero_page_count_included(self, svc_ko: MetadataService) -> None:
        """Zero is a valid value, not None."""
        meta = DocumentMetadata(page_count=0)
        result = svc_ko.format_metadata(meta)
        assert "페이지 수: 0" in result


# ── format_metadata_dict ─────────────────────────────────────────────────

class TestFormatMetadataDict:
    def test_dict_input(self, svc_ko: MetadataService) -> None:
        data = {"title": "From Dict", "author": "Jane"}
        result = svc_ko.format_metadata_dict(data)
        assert "제목: From Dict" in result
        assert "작성자: Jane" in result

    def test_dict_with_iso_datetime(self, svc_ko: MetadataService) -> None:
        data = {"create_time": "2024-06-01T09:00:00"}
        result = svc_ko.format_metadata_dict(data)
        assert "작성일: 2024-06-01 09:00:00" in result

    def test_empty_dict(self, svc_ko: MetadataService) -> None:
        assert svc_ko.format_metadata_dict({}) == ""
