# tests/unit/handlers/test_pdf_scan_ocr.py
"""P1-1: Test PDF scan detection → image tag insertion for OCR bridge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from contextifier.handlers.pdf_default.content_extractor import (
    PdfDefaultContentExtractor,
)
from contextifier.types import PreprocessedData


def _make_fake_doc(page_count: int = 2) -> MagicMock:
    """Create a mock fitz.Document with pages that return pixmaps."""
    doc = MagicMock()
    doc.page_count = page_count

    pages = []
    for i in range(page_count):
        page = MagicMock()
        pix = MagicMock()
        pix.tobytes.return_value = b"PNG-FAKE-DATA"
        page.get_pixmap.return_value = pix
        page.get_text.return_value = ""
        page.get_images.return_value = []
        page.find_tables.return_value = []
        pages.append(page)

    doc.__getitem__ = lambda self, idx: pages[idx]
    doc.load_page = lambda idx: pages[idx]
    return doc


def _make_preprocessed(doc: MagicMock, needs_ocr: bool = True) -> PreprocessedData:
    return PreprocessedData(
        content=doc,
        raw_content=b"",
        encoding="binary",
        resources={"document": doc},
        properties={
            "page_count": doc.page_count,
            "needs_ocr": needs_ocr,
        },
    )


@pytest.fixture()
def image_service() -> MagicMock:
    svc = MagicMock()
    svc.save_and_tag.side_effect = lambda img_data, custom_name=None, skip_duplicate=None: (
        f"[Image:{custom_name}]"
    )
    return svc


@pytest.fixture()
def extractor(image_service: MagicMock) -> PdfDefaultContentExtractor:
    tag_service = MagicMock()
    tag_service.page_tag.side_effect = lambda n: f"[Page {n}]"

    ext = PdfDefaultContentExtractor(
        image_service=image_service,
        tag_service=tag_service,
    )
    return ext


class TestScanPageRendering:
    def test_needs_ocr_renders_pages_as_images(
        self, extractor: PdfDefaultContentExtractor, image_service: MagicMock,
    ) -> None:
        """When needs_ocr=True, extract_text should produce image tags instead of text."""
        doc = _make_fake_doc(page_count=3)
        preprocessed = _make_preprocessed(doc, needs_ocr=True)

        result = extractor.extract_text(preprocessed)

        # Should have image tags for all 3 pages
        assert "[Image:scan_page_1.png]" in result
        assert "[Image:scan_page_2.png]" in result
        assert "[Image:scan_page_3.png]" in result

        # Should have page tags
        assert "[Page 1]" in result
        assert "[Page 2]" in result
        assert "[Page 3]" in result

        # ImageService.save_and_tag called 3 times (one per page)
        assert image_service.save_and_tag.call_count == 3

    def test_normal_pdf_does_not_use_scan_path(
        self, extractor: PdfDefaultContentExtractor, image_service: MagicMock,
    ) -> None:
        """When needs_ocr=False, extract_text should use normal extraction."""
        doc = _make_fake_doc(page_count=1)
        # Make the page return some text so it's a normal PDF
        doc.load_page(0).get_text.return_value = "Hello World"
        preprocessed = _make_preprocessed(doc, needs_ocr=False)

        result = extractor.extract_text(preprocessed)

        # Should NOT have scan page image tags
        assert "scan_page" not in result
        # save_and_tag should not be called for scan rendering
        # (it may be called for embedded images, but not for scan pages)

    def test_scan_without_image_service_returns_empty(self) -> None:
        """When needs_ocr=True but no ImageService, should return empty string."""
        ext = PdfDefaultContentExtractor()
        doc = _make_fake_doc(page_count=1)
        preprocessed = _make_preprocessed(doc, needs_ocr=True)

        result = ext.extract_text(preprocessed)
        assert result == ""
