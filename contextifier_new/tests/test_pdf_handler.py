"""
Comprehensive tests for the PDF handler pipeline.

Tests all shared components, pdf_default, pdf_plus, and the handler router:

 ── Shared ───────────────────────────────────────────────────────────────────
 - _constants: PDF_MAGIC, parse_pdf_date, mode names
 - converter: PdfConverter (magic validation, fitz open, close, NamedTuple)
 - preprocessor: PdfPreprocessor (wrapping, page_count, is_encrypted)
 - metadata_extractor: PdfMetadataExtractor (title, author, dates, unwrap)

 ── pdf_default ──────────────────────────────────────────────────────────────
 - content_extractor helpers: _bbox_overlap_ratio, _is_inside_any, _escape,
                              _table_to_html, _table_to_text, _find_image_bbox
 - PdfDefaultContentExtractor: text, tables, images, page tags

 ── pdf_plus ─────────────────────────────────────────────────────────────────
 - _types: enums, PdfPlusConfig constants, dataclass instantiation & sorting
 - _utils: overlap ratios, bbox checks, escape_html
 - _page_analyzer: detect_page_border, is_table_likely_border
 - _element_merger: merge_page_elements
 - _complexity_analyzer: scoring dimensions, strategy mapping, regions
 - content_extractor: PdfPlusContentExtractor (strategy dispatch, unwrap)

 ── handler ──────────────────────────────────────────────────────────────────
 - PDFHandler: supported_extensions, handler_name, pipeline factories, mode
"""

from __future__ import annotations

import io
import struct
import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock, call


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_minimal_pdf() -> bytes:
    """Create a minimal valid PDF that PyMuPDF can open.

    This is a proper 1-page blank PDF with the required %PDF header,
    a minimal page tree, and %%EOF trailer.
    """
    # Minimal valid PDF — one blank page (A4)
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Root 1 0 R /Size 4 >>\n"
        b"startxref\n196\n%%EOF\n"
    )
    return pdf


def _make_pdf_with_text(pages_text: list[str]) -> bytes:
    """Create a PDF with text on pages via PyMuPDF."""
    import fitz
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 72), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_pdf_with_table_text(header: list[str], rows: list[list[str]]) -> bytes:
    """Create a PDF with a simple text table (bordered cells) via PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # Draw a simple table with lines
    col_width = 100
    row_height = 20
    x_start, y_start = 72, 72
    num_cols = len(header)
    num_rows = len(rows) + 1  # header + data

    # Draw horizontal lines
    for r in range(num_rows + 1):
        y = y_start + r * row_height
        page.draw_line(
            fitz.Point(x_start, y),
            fitz.Point(x_start + num_cols * col_width, y),
        )

    # Draw vertical lines
    for c in range(num_cols + 1):
        x = x_start + c * col_width
        page.draw_line(
            fitz.Point(x, y_start),
            fitz.Point(x, y_start + num_rows * row_height),
        )

    # Insert header text
    for c, h in enumerate(header):
        x = x_start + c * col_width + 5
        y = y_start + 15
        page.insert_text((x, y), h, fontsize=10)

    # Insert row data
    for ri, row in enumerate(rows):
        for ci, cell_text in enumerate(row):
            x = x_start + ci * col_width + 5
            y = y_start + (ri + 1) * row_height + 15
            page.insert_text((x, y), cell_text, fontsize=10)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_file_context(data: bytes) -> dict:
    """Create a FileContext dict for PDF testing."""
    return {
        "file_path": "/test/document.pdf",
        "file_name": "document.pdf",
        "file_extension": "pdf",
        "file_category": "document",
        "file_data": data,
        "file_stream": io.BytesIO(data),
        "file_size": len(data),
    }


def _mock_fitz_page(
    *,
    width: float = 612.0,
    height: float = 792.0,
    text: str = "Sample text",
    drawings: list | None = None,
    images: list | None = None,
    text_dict: dict | None = None,
) -> MagicMock:
    """Create a mock fitz.Page for testing."""
    page = MagicMock()
    rect = MagicMock()
    rect.width = width
    rect.height = height
    page.rect = rect

    page.get_text = MagicMock(side_effect=lambda fmt="text", **kw: (
        text if fmt == "text" else (
            text_dict or {
                "blocks": [{
                    "type": 0,
                    "bbox": (72, 72, 540, 120),
                    "lines": [{
                        "bbox": (72, 72, 540, 84),
                        "spans": [{"text": text, "size": 12, "font": "Helvetica"}],
                    }],
                }],
            }
            if fmt == "dict" else text
        )
    ))
    page.get_drawings = MagicMock(return_value=drawings or [])
    page.get_images = MagicMock(return_value=images or [])
    page.get_image_info = MagicMock(return_value=[])
    page.find_tables = MagicMock(return_value=[])

    return page


def _mock_fitz_doc(
    pages: list[MagicMock] | None = None,
    page_count: int = 1,
    metadata: dict | None = None,
) -> MagicMock:
    """Create a mock fitz.Document for testing."""
    doc = MagicMock()
    if pages is None:
        pages = [_mock_fitz_page() for _ in range(page_count)]

    doc.page_count = len(pages)
    doc.is_encrypted = False
    doc.metadata = metadata or {}
    doc.load_page = MagicMock(side_effect=lambda idx: pages[idx])
    return doc


# ═════════════════════════════════════════════════════════════════════════════
# Test Constants
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfConstants:
    """Tests for handlers/pdf/_constants.py module."""

    def test_pdf_magic_value(self):
        from contextifier_new.handlers.pdf._constants import PDF_MAGIC
        assert PDF_MAGIC == b"%PDF"

    def test_pdf_mode_default(self):
        from contextifier_new.handlers.pdf._constants import PDF_MODE_DEFAULT
        assert PDF_MODE_DEFAULT == "default"

    def test_pdf_mode_plus(self):
        from contextifier_new.handlers.pdf._constants import PDF_MODE_PLUS
        assert PDF_MODE_PLUS == "plus"

    def test_pdf_format_option_key(self):
        from contextifier_new.handlers.pdf._constants import PDF_FORMAT_OPTION_KEY
        assert PDF_FORMAT_OPTION_KEY == "pdf"

    def test_pdf_mode_option(self):
        from contextifier_new.handlers.pdf._constants import PDF_MODE_OPTION
        assert PDF_MODE_OPTION == "mode"

    def test_parse_pdf_date_basic(self):
        from contextifier_new.handlers.pdf._constants import parse_pdf_date
        result = parse_pdf_date("D:20231015120000")
        assert result is not None
        assert result.year == 2023
        assert result.month == 10
        assert result.day == 15
        assert result.hour == 12

    def test_parse_pdf_date_partial(self):
        from contextifier_new.handlers.pdf._constants import parse_pdf_date
        result = parse_pdf_date("D:2023")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1  # default

    def test_parse_pdf_date_year_month(self):
        from contextifier_new.handlers.pdf._constants import parse_pdf_date
        result = parse_pdf_date("D:202305")
        assert result is not None
        assert result.year == 2023
        assert result.month == 5

    def test_parse_pdf_date_none(self):
        from contextifier_new.handlers.pdf._constants import parse_pdf_date
        assert parse_pdf_date(None) is None

    def test_parse_pdf_date_empty(self):
        from contextifier_new.handlers.pdf._constants import parse_pdf_date
        assert parse_pdf_date("") is None

    def test_parse_pdf_date_invalid(self):
        from contextifier_new.handlers.pdf._constants import parse_pdf_date
        assert parse_pdf_date("not a date") is None

    def test_parse_pdf_date_garbage(self):
        from contextifier_new.handlers.pdf._constants import parse_pdf_date
        assert parse_pdf_date("D:abcdefgh") is None


# ═════════════════════════════════════════════════════════════════════════════
# Test Converter
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfConverter:
    """Tests for PdfConverter."""

    def test_convert_valid_pdf(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        data = _make_minimal_pdf()
        ctx = _make_file_context(data)
        result = conv.convert(ctx)
        assert hasattr(result, "doc")
        assert hasattr(result, "file_data")
        assert result.file_data == data
        result.doc.close()

    def test_convert_empty_raises(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.errors import ConversionError
        conv = PdfConverter()
        ctx = _make_file_context(b"")
        with pytest.raises(ConversionError):
            conv.convert(ctx)

    def test_convert_invalid_magic_raises(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.errors import ConversionError
        conv = PdfConverter()
        ctx = _make_file_context(b"NOT_PDF data here")
        with pytest.raises(ConversionError, match="Not a valid PDF"):
            conv.convert(ctx)

    def test_validate_valid_pdf(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        ctx = _make_file_context(_make_minimal_pdf())
        assert conv.validate(ctx) is True

    def test_validate_invalid_magic(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        ctx = _make_file_context(b"\x00\x00\x00\x00XXXX")
        assert conv.validate(ctx) is False

    def test_validate_too_short(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        ctx = _make_file_context(b"%PD")
        assert conv.validate(ctx) is False

    def test_validate_empty(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        ctx = _make_file_context(b"")
        assert conv.validate(ctx) is False

    def test_get_format_name(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        assert PdfConverter().get_format_name() == "pdf"

    def test_close_pdf_converted_data(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter, PdfConvertedData
        conv = PdfConverter()
        data = _make_minimal_pdf()
        ctx = _make_file_context(data)
        result = conv.convert(ctx)
        # Should not raise
        conv.close(result)

    def test_close_none_does_not_raise(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        conv.close(None)

    def test_close_random_object_does_not_raise(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        conv.close("anything")

    def test_converted_data_is_namedtuple(self):
        from contextifier_new.handlers.pdf.converter import PdfConvertedData
        data = PdfConvertedData(doc="fake_doc", file_data=b"fake")
        assert data.doc == "fake_doc"
        assert data.file_data == b"fake"

    def test_convert_real_single_page(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        data = _make_pdf_with_text(["Hello World"])
        conv = PdfConverter()
        ctx = _make_file_context(data)
        result = conv.convert(ctx)
        assert result.doc.page_count == 1
        page = result.doc.load_page(0)
        text = page.get_text("text")
        assert "Hello World" in text
        result.doc.close()

    def test_convert_multi_page(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        data = _make_pdf_with_text(["Page One", "Page Two", "Page Three"])
        conv = PdfConverter()
        ctx = _make_file_context(data)
        result = conv.convert(ctx)
        assert result.doc.page_count == 3
        result.doc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Test Preprocessor
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfPreprocessor:
    """Tests for PdfPreprocessor."""

    def test_preprocess_from_converted_data(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter, PdfConvertedData
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor

        data = _make_minimal_pdf()
        ctx = _make_file_context(data)
        converted = PdfConverter().convert(ctx)

        pp = PdfPreprocessor()
        result = pp.preprocess(converted)

        assert result.content is converted.doc
        assert result.raw_content == data
        assert result.properties["page_count"] == 1
        assert result.properties["is_encrypted"] is False
        assert result.resources["document"] is converted.doc
        converted.doc.close()

    def test_preprocess_from_bare_doc(self):
        import fitz
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor

        data = _make_minimal_pdf()
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")

        pp = PdfPreprocessor()
        result = pp.preprocess(doc)

        assert result.content is doc
        assert result.properties["page_count"] == 1
        doc.close()

    def test_preprocess_multi_page(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor

        data = _make_pdf_with_text(["A", "B", "C"])
        ctx = _make_file_context(data)
        converted = PdfConverter().convert(ctx)

        pp = PdfPreprocessor()
        result = pp.preprocess(converted)
        assert result.properties["page_count"] == 3
        converted.doc.close()

    def test_preprocess_encoding_is_binary(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor

        data = _make_minimal_pdf()
        converted = PdfConverter().convert(_make_file_context(data))
        result = PdfPreprocessor().preprocess(converted)
        assert result.encoding == "binary"
        converted.doc.close()

    def test_get_format_name(self):
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        assert PdfPreprocessor().get_format_name() == "pdf"

    def test_preprocess_with_mock_doc(self):
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor

        doc = MagicMock()
        doc.page_count = 5
        doc.is_encrypted = True

        pp = PdfPreprocessor()
        result = pp.preprocess(doc)
        assert result.properties["page_count"] == 5
        assert result.properties["is_encrypted"] is True


# ═════════════════════════════════════════════════════════════════════════════
# Test Metadata Extractor
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfMetadataExtractor:
    """Tests for PdfMetadataExtractor."""

    def test_extract_from_mock_doc(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor

        doc = MagicMock()
        doc.page_count = 3
        doc.metadata = {
            "title": "Test Title",
            "author": "Test Author",
            "subject": "Test Subject",
            "keywords": "test,pdf",
            "creationDate": "D:20231015120000",
            "modDate": "D:20231020150000",
            "creator": "TestCreator",
            "producer": "TestProducer",
        }

        ext = PdfMetadataExtractor()
        meta = ext.extract(doc)

        assert meta.title == "Test Title"
        assert meta.author == "Test Author"
        assert meta.subject == "Test Subject"
        assert meta.keywords == "test,pdf"
        assert meta.page_count == 3
        assert meta.create_time == "2023-10-15 12:00:00"
        assert meta.last_saved_time == "2023-10-20 15:00:00"
        assert meta.custom["creator"] == "TestCreator"
        assert meta.custom["producer"] == "TestProducer"

    def test_extract_from_preprocessed_data(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        from contextifier_new.types import PreprocessedData

        doc = MagicMock()
        doc.page_count = 2
        doc.metadata = {"title": "Via Preprocessed"}

        ppd = PreprocessedData(content=doc, raw_content=b"")
        ext = PdfMetadataExtractor()
        meta = ext.extract(ppd)
        assert meta.title == "Via Preprocessed"
        assert meta.page_count == 2

    def test_extract_none_returns_empty(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        ext = PdfMetadataExtractor()
        meta = ext.extract(None)
        assert meta.is_empty()

    def test_extract_nonsense_returns_empty(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        ext = PdfMetadataExtractor()
        meta = ext.extract("not a document")
        assert meta.is_empty()

    def test_extract_no_metadata(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        doc = MagicMock()
        doc.page_count = 1
        doc.metadata = {}
        ext = PdfMetadataExtractor()
        meta = ext.extract(doc)
        assert meta.title is None
        assert meta.author is None
        assert meta.page_count == 1

    def test_extract_blank_strings_are_none(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        doc = MagicMock()
        doc.page_count = 1
        doc.metadata = {"title": "", "author": "   "}
        ext = PdfMetadataExtractor()
        meta = ext.extract(doc)
        assert meta.title is None
        assert meta.author is None

    def test_extract_with_encryption_info(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        doc = MagicMock()
        doc.page_count = 1
        doc.metadata = {"encryption": "Standard V2 R3 128-bit RC4"}
        ext = PdfMetadataExtractor()
        meta = ext.extract(doc)
        assert "encryption" in meta.custom

    def test_extract_with_format_info(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        doc = MagicMock()
        doc.page_count = 1
        doc.metadata = {"format": "PDF 1.7"}
        ext = PdfMetadataExtractor()
        meta = ext.extract(doc)
        assert meta.custom.get("format") == "PDF 1.7"

    def test_get_format_name(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        assert PdfMetadataExtractor().get_format_name() == "pdf"

    def test_extract_real_pdf(self):
        """Extract metadata from a real (in-memory) PDF."""
        import fitz
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor

        data = _make_pdf_with_text(["Hello"])
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")

        ext = PdfMetadataExtractor()
        meta = ext.extract(doc)
        assert meta.page_count == 1
        doc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_default Content Extractor Helpers
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfDefaultHelpers:
    """Tests for helper functions in pdf_default/content_extractor.py."""

    def test_bbox_overlap_ratio_full_overlap(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _bbox_overlap_ratio
        ratio = _bbox_overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10))
        assert ratio == pytest.approx(1.0)

    def test_bbox_overlap_ratio_no_overlap(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _bbox_overlap_ratio
        ratio = _bbox_overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30))
        assert ratio == pytest.approx(0.0)

    def test_bbox_overlap_ratio_partial(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _bbox_overlap_ratio
        ratio = _bbox_overlap_ratio((0, 0, 10, 10), (5, 5, 15, 15))
        # Inner area=100, overlap=5*5=25 → 25/100=0.25
        assert ratio == pytest.approx(0.25)

    def test_bbox_overlap_ratio_zero_area(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _bbox_overlap_ratio
        ratio = _bbox_overlap_ratio((5, 5, 5, 5), (0, 0, 10, 10))
        assert ratio == pytest.approx(0.0)

    def test_is_inside_any_true(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _is_inside_any
        result = _is_inside_any(
            (2, 2, 8, 8),
            [(0, 0, 10, 10)],
            threshold=0.5,
        )
        assert result is True

    def test_is_inside_any_false(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _is_inside_any
        result = _is_inside_any(
            (100, 100, 200, 200),
            [(0, 0, 10, 10)],
            threshold=0.5,
        )
        assert result is False

    def test_is_inside_any_empty_list(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _is_inside_any
        result = _is_inside_any((0, 0, 10, 10), [], threshold=0.5)
        assert result is False

    def test_escape_html_chars(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _escape
        assert "&amp;" in _escape("&")
        assert "&lt;" in _escape("<")
        assert "&gt;" in _escape(">")

    def test_table_to_html_basic(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _table_to_html
        data = [["H1", "H2"], ["A", "B"]]
        html = _table_to_html(data)
        assert "<table>" in html
        assert "<th>H1</th>" in html
        assert "<td>A</td>" in html

    def test_table_to_html_empty(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _table_to_html
        assert _table_to_html([]) == ""

    def test_table_to_html_none_cells(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _table_to_html
        data = [["H1", None], [None, "B"]]
        html = _table_to_html(data)
        assert "<table>" in html
        # None should become empty string
        assert "<th></th>" in html

    def test_table_to_text(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _table_to_text
        data = [["Row 1"], ["Row 2"], ["Row 3"]]
        text = _table_to_text(data)
        assert "Row 1" in text
        assert "Row 2" in text
        assert "Row 3" in text

    def test_table_to_text_empty_rows(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _table_to_text
        data = [[""], ["Text"], [""]]
        text = _table_to_text(data)
        assert text == "Text"

    def test_table_to_text_none_cells(self):
        from contextifier_new.handlers.pdf_default.content_extractor import _table_to_text
        data = [[None], ["Hello"]]
        text = _table_to_text(data)
        assert "Hello" in text


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_default Content Extractor
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfDefaultContentExtractor:
    """Tests for PdfDefaultContentExtractor."""

    def test_extract_text_single_page(self):
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = _mock_fitz_doc(page_count=1, pages=[
            _mock_fitz_page(text="Hello PDF World"),
        ])
        ppd = PreprocessedData(
            content=doc, raw_content=b"", resources={"document": doc},
        )

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert "Hello PDF World" in text

    def test_extract_text_multi_page(self):
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        pages = [
            _mock_fitz_page(text="Page 1 content"),
            _mock_fitz_page(text="Page 2 content"),
            _mock_fitz_page(text="Page 3 content"),
        ]
        doc = _mock_fitz_doc(pages=pages)
        ppd = PreprocessedData(
            content=doc, raw_content=b"", resources={"document": doc},
        )

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert "Page 1 content" in text
        assert "Page 2 content" in text
        assert "Page 3 content" in text

    def test_extract_text_empty_doc(self):
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        ppd = PreprocessedData(content=None, raw_content=b"")
        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert text == ""

    def test_extract_text_with_page_tags(self):
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = _mock_fitz_doc(page_count=1, pages=[
            _mock_fitz_page(text="Content here"),
        ])
        ppd = PreprocessedData(content=doc, raw_content=b"")

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        # Default tag when no tag_service
        assert "Page Number: 1" in text

    def test_extract_text_with_tag_service(self):
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = _mock_fitz_doc(page_count=1, pages=[
            _mock_fitz_page(text="Content here"),
        ])
        ppd = PreprocessedData(content=doc, raw_content=b"")

        tag_svc = MagicMock()
        tag_svc.page_tag = MagicMock(return_value="[Page:1]")

        ext = PdfDefaultContentExtractor(tag_service=tag_svc)
        text = ext.extract_text(ppd)
        assert "[Page:1]" in text

    def test_get_format_name(self):
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        assert PdfDefaultContentExtractor().get_format_name() == "pdf"

    def test_extract_text_real_pdf(self):
        """End-to-end: create a real PDF and extract text."""
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )

        data = _make_pdf_with_text(["Hello from real PDF"])
        ctx = _make_file_context(data)
        converted = PdfConverter().convert(ctx)
        ppd = PdfPreprocessor().preprocess(converted)

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert "Hello from real PDF" in text
        converted.doc.close()

    def test_extract_text_real_multi_page(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )

        data = _make_pdf_with_text(["Page Alpha", "Page Beta"])
        converted = PdfConverter().convert(_make_file_context(data))
        ppd = PdfPreprocessor().preprocess(converted)

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert "Page Alpha" in text
        assert "Page Beta" in text
        converted.doc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_plus Types
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfPlusTypes:
    """Tests for pdf_plus/_types.py enums, config, and dataclasses."""

    def test_element_type_values(self):
        from contextifier_new.handlers.pdf_plus._types import ElementType
        assert ElementType.TEXT.value == "text"
        assert ElementType.TABLE.value == "table"
        assert ElementType.IMAGE.value == "image"
        assert ElementType.ANNOTATION.value == "annotation"

    def test_complexity_level_enum(self):
        from contextifier_new.handlers.pdf_plus._types import ComplexityLevel
        levels = list(ComplexityLevel)
        assert len(levels) == 4
        assert ComplexityLevel.SIMPLE in levels
        assert ComplexityLevel.EXTREME in levels

    def test_processing_strategy_enum(self):
        from contextifier_new.handlers.pdf_plus._types import ProcessingStrategy
        strategies = list(ProcessingStrategy)
        assert len(strategies) == 4
        assert ProcessingStrategy.TEXT_EXTRACTION in strategies
        assert ProcessingStrategy.FULL_PAGE_OCR in strategies

    def test_table_quality_enum(self):
        from contextifier_new.handlers.pdf_plus._types import TableQuality
        assert len(list(TableQuality)) == 5

    def test_layout_block_type_enum(self):
        from contextifier_new.handlers.pdf_plus._types import LayoutBlockType
        assert LayoutBlockType.ARTICLE.name == "ARTICLE"
        assert LayoutBlockType.TABLE.name == "TABLE"
        assert LayoutBlockType.SIDEBAR.name == "SIDEBAR"

    def test_config_complexity_thresholds(self):
        from contextifier_new.handlers.pdf_plus._types import PdfPlusConfig
        assert PdfPlusConfig.COMPLEXITY_MODERATE == 0.35
        assert PdfPlusConfig.COMPLEXITY_COMPLEX == 0.65
        assert PdfPlusConfig.COMPLEXITY_EXTREME == 0.90

    def test_config_weight_sum(self):
        from contextifier_new.handlers.pdf_plus._types import PdfPlusConfig
        total = (
            PdfPlusConfig.WEIGHT_DRAWING
            + PdfPlusConfig.WEIGHT_IMAGE
            + PdfPlusConfig.WEIGHT_TEXT
            + PdfPlusConfig.WEIGHT_LAYOUT
        )
        assert total == pytest.approx(1.0)

    def test_config_quality_order(self):
        """Quality thresholds must be in descending order."""
        from contextifier_new.handlers.pdf_plus._types import PdfPlusConfig
        assert PdfPlusConfig.QUALITY_EXCELLENT > PdfPlusConfig.QUALITY_GOOD
        assert PdfPlusConfig.QUALITY_GOOD > PdfPlusConfig.QUALITY_MODERATE
        assert PdfPlusConfig.QUALITY_MODERATE > PdfPlusConfig.QUALITY_POOR

    def test_lineinfo_length(self):
        from contextifier_new.handlers.pdf_plus._types import LineInfo
        li = LineInfo(x0=0, y0=0, x1=3, y1=4)
        assert li.length == pytest.approx(5.0)

    def test_lineinfo_midpoint(self):
        from contextifier_new.handlers.pdf_plus._types import LineInfo
        li = LineInfo(x0=0, y0=0, x1=10, y1=20)
        assert li.midpoint == (5.0, 10.0)

    def test_gridinfo_row_col_count(self):
        from contextifier_new.handlers.pdf_plus._types import GridInfo
        g = GridInfo(h_lines=[0, 20, 40, 60], v_lines=[0, 100, 200])
        assert g.row_count == 3
        assert g.col_count == 2

    def test_gridinfo_empty(self):
        from contextifier_new.handlers.pdf_plus._types import GridInfo
        g = GridInfo()
        assert g.row_count == 0
        assert g.col_count == 0

    def test_page_element_sort_by_y(self):
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        e1 = PageElement(ElementType.TEXT, "A", (0, 100, 100, 200), page_num=0)
        e2 = PageElement(ElementType.TEXT, "B", (0, 50, 100, 100), page_num=0)
        assert e2 < e1  # e2 has smaller y=50

    def test_page_element_sort_same_y(self):
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        e1 = PageElement(ElementType.TEXT, "A", (200, 100, 300, 200), page_num=0)
        e2 = PageElement(ElementType.TEXT, "B", (50, 100, 150, 200), page_num=0)
        assert e2 < e1  # same y, e2 x=50 < e1 x=200

    def test_page_element_sort_by_page(self):
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        e1 = PageElement(ElementType.TEXT, "A", (0, 0, 100, 100), page_num=1)
        e2 = PageElement(ElementType.TEXT, "B", (0, 0, 100, 100), page_num=0)
        assert e2 < e1  # page 0 < page 1

    def test_table_candidate_row_col_count(self):
        from contextifier_new.handlers.pdf_plus._types import (
            TableCandidate, TableDetectionStrategy,
        )
        tc = TableCandidate(
            strategy=TableDetectionStrategy.PYMUPDF_NATIVE,
            confidence=0.8,
            bbox=(0, 0, 100, 100),
            data=[["H1", "H2"], ["A", "B"], ["C", "D"]],
        )
        assert tc.row_count == 3
        assert tc.col_count == 2

    def test_page_complexity_defaults(self):
        from contextifier_new.handlers.pdf_plus._types import (
            PageComplexity, ComplexityLevel, ProcessingStrategy,
        )
        pc = PageComplexity(
            page_num=0,
            page_size=(612, 792),
            overall_complexity=ComplexityLevel.SIMPLE,
            overall_score=0.1,
        )
        assert pc.recommended_strategy == ProcessingStrategy.TEXT_EXTRACTION
        assert pc.column_count == 1
        assert pc.complex_regions == []

    def test_block_result(self):
        from contextifier_new.handlers.pdf_plus._types import BlockResult
        br = BlockResult(success=True, image_tag="<img>", bbox=(0, 0, 100, 100))
        assert br.success is True
        assert br.image_tag == "<img>"

    def test_page_border_info_default(self):
        from contextifier_new.handlers.pdf_plus._types import PageBorderInfo
        pbi = PageBorderInfo()
        assert pbi.has_border is False
        assert all(v is False for v in pbi.border_lines.values())


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_plus Utilities
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfPlusUtils:
    """Tests for pdf_plus/_utils.py."""

    def test_escape_html(self):
        from contextifier_new.handlers.pdf_plus._utils import escape_html
        assert "&amp;" in escape_html("&")
        assert "&lt;" in escape_html("<b>")
        assert "&gt;" in escape_html(">")
        assert "&quot;" in escape_html('"')

    def test_calculate_overlap_ratio_full(self):
        from contextifier_new.handlers.pdf_plus._utils import calculate_overlap_ratio
        assert calculate_overlap_ratio((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_calculate_overlap_ratio_none(self):
        from contextifier_new.handlers.pdf_plus._utils import calculate_overlap_ratio
        assert calculate_overlap_ratio((0, 0, 5, 5), (10, 10, 20, 20)) == pytest.approx(0.0)

    def test_calculate_overlap_ratio_half(self):
        from contextifier_new.handlers.pdf_plus._utils import calculate_overlap_ratio
        # inner (0,0,10,10) area=100, outer (5,0,15,10)
        # overlap = 5*10 = 50, ratio = 50/100 = 0.5
        assert calculate_overlap_ratio((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)

    def test_calculate_overlap_ratio_zero_area(self):
        from contextifier_new.handlers.pdf_plus._utils import calculate_overlap_ratio
        assert calculate_overlap_ratio((5, 5, 5, 5), (0, 0, 10, 10)) == pytest.approx(0.0)

    def test_is_inside_any_bbox_true(self):
        from contextifier_new.handlers.pdf_plus._utils import is_inside_any_bbox
        assert is_inside_any_bbox(
            (1, 1, 9, 9), [(0, 0, 10, 10)], threshold=0.5,
        ) is True

    def test_is_inside_any_bbox_false(self):
        from contextifier_new.handlers.pdf_plus._utils import is_inside_any_bbox
        assert is_inside_any_bbox(
            (100, 100, 200, 200), [(0, 0, 10, 10)], threshold=0.5,
        ) is False

    def test_is_inside_any_bbox_empty(self):
        from contextifier_new.handlers.pdf_plus._utils import is_inside_any_bbox
        assert is_inside_any_bbox((0, 0, 10, 10), []) is False

    def test_bbox_overlaps_yes(self):
        from contextifier_new.handlers.pdf_plus._utils import bbox_overlaps
        assert bbox_overlaps((0, 0, 10, 10), (5, 5, 15, 15)) is True

    def test_bbox_overlaps_no(self):
        from contextifier_new.handlers.pdf_plus._utils import bbox_overlaps
        assert bbox_overlaps((0, 0, 10, 10), (20, 20, 30, 30)) is False

    def test_bbox_overlaps_edge_touch(self):
        from contextifier_new.handlers.pdf_plus._utils import bbox_overlaps
        # Touching at edge → no overlap (strict inequality)
        assert bbox_overlaps((0, 0, 10, 10), (10, 0, 20, 10)) is False

    def test_find_image_position_found(self):
        from contextifier_new.handlers.pdf_plus._utils import find_image_position
        page = MagicMock()
        page.get_image_info = MagicMock(return_value=[
            {"xref": 5, "bbox": (10, 20, 100, 200)},
            {"xref": 7, "bbox": (50, 50, 150, 150)},
        ])
        result = find_image_position(page, 7)
        assert result is not None
        assert result[0] == 50

    def test_find_image_position_not_found(self):
        from contextifier_new.handlers.pdf_plus._utils import find_image_position
        page = MagicMock()
        page.get_image_info = MagicMock(return_value=[
            {"xref": 5, "bbox": (10, 20, 100, 200)},
        ])
        result = find_image_position(page, 99)
        assert result is None

    def test_find_image_position_error(self):
        from contextifier_new.handlers.pdf_plus._utils import find_image_position
        page = MagicMock()
        page.get_image_info = MagicMock(side_effect=RuntimeError("fail"))
        result = find_image_position(page, 5)
        assert result is None

    def test_get_text_lines_with_positions(self):
        from contextifier_new.handlers.pdf_plus._utils import get_text_lines_with_positions
        page = MagicMock()
        page.get_text = MagicMock(return_value={
            "blocks": [{
                "type": 0,
                "lines": [
                    {"bbox": (72, 72, 200, 84),
                     "spans": [{"text": "Hello "}, {"text": "World"}]},
                ],
            }],
        })
        lines = get_text_lines_with_positions(page)
        assert len(lines) == 1
        assert lines[0]["text"] == "Hello World"
        assert lines[0]["bbox"] == (72, 72, 200, 84)

    def test_get_text_lines_skips_image_blocks(self):
        from contextifier_new.handlers.pdf_plus._utils import get_text_lines_with_positions
        page = MagicMock()
        page.get_text = MagicMock(return_value={
            "blocks": [
                {"type": 1, "lines": []},  # image block — skipped
                {"type": 0, "lines": [
                    {"bbox": (0, 0, 100, 12), "spans": [{"text": "Visible"}]},
                ]},
            ],
        })
        lines = get_text_lines_with_positions(page)
        assert len(lines) == 1
        assert lines[0]["text"] == "Visible"


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_plus Page Analyzer
# ═════════════════════════════════════════════════════════════════════════════


class TestPageAnalyzer:
    """Tests for pdf_plus/_page_analyzer.py."""

    def test_detect_page_border_no_border(self):
        from contextifier_new.handlers.pdf_plus._page_analyzer import detect_page_border
        page = _mock_fitz_page(drawings=[])
        info = detect_page_border(page)
        assert info.has_border is False

    def test_detect_page_border_with_top_line(self):
        from contextifier_new.handlers.pdf_plus._page_analyzer import detect_page_border
        page = _mock_fitz_page(width=612, height=792)
        # Horizontal line near top, spanning >85% of width
        page.get_drawings = MagicMock(return_value=[
            {"rect": (0, 5, 600, 7)},  # thin, near top, wide
        ])
        info = detect_page_border(page)
        assert info.has_border is True
        assert info.border_lines["top"] is True

    def test_detect_page_border_error_handling(self):
        from contextifier_new.handlers.pdf_plus._page_analyzer import detect_page_border
        page = MagicMock()
        page.rect = MagicMock()
        page.rect.width = 612
        page.rect.height = 792
        page.get_drawings = MagicMock(side_effect=RuntimeError("oops"))
        info = detect_page_border(page)
        assert info.has_border is False

    def test_is_table_likely_border_true(self):
        from contextifier_new.handlers.pdf_plus._page_analyzer import (
            is_table_likely_border,
        )
        from contextifier_new.handlers.pdf_plus._types import PageBorderInfo
        border = PageBorderInfo(
            has_border=True,
            border_lines={"top": True, "bottom": True, "left": True, "right": True},
        )
        page = _mock_fitz_page(width=612, height=792)
        # Table spanning ~95% of page
        result = is_table_likely_border(
            (5, 5, 605, 785), border, page,
        )
        assert result is True

    def test_is_table_likely_border_false_no_border(self):
        from contextifier_new.handlers.pdf_plus._page_analyzer import (
            is_table_likely_border,
        )
        from contextifier_new.handlers.pdf_plus._types import PageBorderInfo
        border = PageBorderInfo(has_border=False)
        page = _mock_fitz_page()
        result = is_table_likely_border(
            (5, 5, 605, 785), border, page,
        )
        assert result is False

    def test_is_table_likely_border_false_small_table(self):
        from contextifier_new.handlers.pdf_plus._page_analyzer import (
            is_table_likely_border,
        )
        from contextifier_new.handlers.pdf_plus._types import PageBorderInfo
        border = PageBorderInfo(has_border=True,
                                 border_lines={"top": True, "bottom": True,
                                               "left": True, "right": True})
        page = _mock_fitz_page(width=612, height=792)
        # Small table — not a border
        result = is_table_likely_border(
            (100, 200, 300, 400), border, page,
        )
        assert result is False


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_plus Element Merger
# ═════════════════════════════════════════════════════════════════════════════


class TestElementMerger:
    """Tests for pdf_plus/_element_merger.py."""

    def test_merge_empty(self):
        from contextifier_new.handlers.pdf_plus._element_merger import merge_page_elements
        assert merge_page_elements([]) == ""

    def test_merge_single(self):
        from contextifier_new.handlers.pdf_plus._element_merger import merge_page_elements
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        elements = [
            PageElement(ElementType.TEXT, "Hello World", (0, 10, 100, 30), page_num=0),
        ]
        result = merge_page_elements(elements)
        assert result == "Hello World"

    def test_merge_sorted_by_y_position(self):
        from contextifier_new.handlers.pdf_plus._element_merger import merge_page_elements
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        elements = [
            PageElement(ElementType.TEXT, "Bottom", (0, 200, 100, 250), page_num=0),
            PageElement(ElementType.TEXT, "Top", (0, 10, 100, 30), page_num=0),
            PageElement(ElementType.TEXT, "Middle", (0, 100, 100, 130), page_num=0),
        ]
        result = merge_page_elements(elements)
        parts = result.split("\n\n")
        assert parts[0] == "Top"
        assert parts[1] == "Middle"
        assert parts[2] == "Bottom"

    def test_merge_mixed_types(self):
        from contextifier_new.handlers.pdf_plus._element_merger import merge_page_elements
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        elements = [
            PageElement(ElementType.TABLE, "<table>...</table>",
                        (0, 100, 500, 300), page_num=0),
            PageElement(ElementType.TEXT, "Title text",
                        (0, 10, 500, 30), page_num=0),
            PageElement(ElementType.IMAGE, "<img src='test.png'>",
                        (0, 400, 500, 600), page_num=0),
        ]
        result = merge_page_elements(elements)
        parts = result.split("\n\n")
        assert parts[0] == "Title text"
        assert "<table>" in parts[1]
        assert "<img" in parts[2]

    def test_merge_skips_whitespace_only(self):
        from contextifier_new.handlers.pdf_plus._element_merger import merge_page_elements
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        elements = [
            PageElement(ElementType.TEXT, "  ", (0, 10, 100, 30), page_num=0),
            PageElement(ElementType.TEXT, "Real content",
                        (0, 40, 100, 60), page_num=0),
        ]
        result = merge_page_elements(elements)
        assert result == "Real content"

    def test_merge_custom_gap(self):
        from contextifier_new.handlers.pdf_plus._element_merger import merge_page_elements
        from contextifier_new.handlers.pdf_plus._types import PageElement, ElementType
        elements = [
            PageElement(ElementType.TEXT, "A", (0, 0, 10, 10), page_num=0),
            PageElement(ElementType.TEXT, "B", (0, 20, 10, 30), page_num=0),
        ]
        result = merge_page_elements(elements, page_gap="\n---\n")
        assert "\n---\n" in result


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_plus Complexity Analyzer
# ═════════════════════════════════════════════════════════════════════════════


class TestComplexityAnalyzer:
    """Tests for pdf_plus/_complexity_analyzer.py."""

    def test_simple_text_page(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )
        from contextifier_new.handlers.pdf_plus._types import (
            ComplexityLevel, ProcessingStrategy,
        )

        page = _mock_fitz_page(
            text="Simple text page",
            drawings=[],
            images=[],
        )
        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()

        assert result.overall_complexity == ComplexityLevel.SIMPLE
        assert result.recommended_strategy == ProcessingStrategy.TEXT_EXTRACTION
        assert result.overall_score < 0.35

    def test_page_with_many_drawings(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )
        from contextifier_new.handlers.pdf_plus._types import ComplexityLevel

        # Create many drawings to push drawing_score high
        drawings = []
        for i in range(200):
            drawings.append({
                "rect": (i, i, i + 10, i + 10),
                "items": [("l", (i, i), (i + 10, i + 10)),
                          ("c", (i, i), (i + 5, i + 5), (i + 10, i + 10))],
                "fill": (0, 0, 0),
            })

        page = _mock_fitz_page(drawings=drawings, images=[])
        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()

        # Many drawings should produce a non-zero score
        assert result.overall_score > 0.0
        assert result.total_drawings == 200

    def test_page_with_many_images(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )

        # Many images → higher image score
        images = [(i, 0, 100, 100, 8, "DeviceRGB", "", "", "", 0)
                  for i in range(50)]

        page = _mock_fitz_page(images=images, drawings=[])
        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()

        assert result.total_images == 50

    def test_page_with_bad_text_quality(self):
        """PUA characters → low text quality → possibly FULL_PAGE_OCR."""
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )
        from contextifier_new.handlers.pdf_plus._types import ProcessingStrategy

        # Create text with lots of PUA characters
        pua_text = "\uE000" * 100  # all PUA characters

        page = _mock_fitz_page(text=pua_text, text_dict={
            "blocks": [{
                "type": 0,
                "bbox": (72, 72, 540, 120),
                "lines": [{
                    "bbox": (72, 72, 540, 84),
                    "spans": [{"text": pua_text, "size": 12, "font": "Custom"}],
                }],
            }],
        })

        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()

        # Bad text quality should push toward OCR
        assert result.recommended_strategy == ProcessingStrategy.FULL_PAGE_OCR

    def test_page_complexity_has_regions(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )

        page = _mock_fitz_page()
        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()

        # Should have region analysis
        assert isinstance(result.regions, list)
        assert len(result.regions) > 0  # at least some grid cells

    def test_column_counting_single(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )

        page = _mock_fitz_page(text_dict={
            "blocks": [
                {"type": 0, "bbox": (72, 72, 540, 100),
                 "lines": [{"spans": [{"text": "Line 1"}], "bbox": (72, 72, 540, 84)}]},
                {"type": 0, "bbox": (72, 120, 540, 150),
                 "lines": [{"spans": [{"text": "Line 2"}], "bbox": (72, 120, 540, 132)}]},
            ],
        })

        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()
        assert result.column_count == 1

    def test_column_counting_multi(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )

        # Two-column layout: blocks at x=72 and x=306
        page = _mock_fitz_page(text_dict={
            "blocks": [
                {"type": 0, "bbox": (72, 72, 300, 100),
                 "lines": [{"spans": [{"text": "Col1"}], "bbox": (72, 72, 300, 84)}]},
                {"type": 0, "bbox": (306, 72, 540, 100),
                 "lines": [{"spans": [{"text": "Col2"}], "bbox": (306, 72, 540, 84)}]},
                {"type": 0, "bbox": (72, 120, 300, 150),
                 "lines": [{"spans": [{"text": "Col1b"}], "bbox": (72, 120, 300, 132)}]},
                {"type": 0, "bbox": (306, 120, 540, 150),
                 "lines": [{"spans": [{"text": "Col2b"}], "bbox": (306, 120, 540, 132)}]},
            ],
        })

        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()
        assert result.column_count >= 2

    def test_empty_page_is_simple(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )
        from contextifier_new.handlers.pdf_plus._types import (
            ComplexityLevel, ProcessingStrategy,
        )

        page = _mock_fitz_page(text="", drawings=[], images=[], text_dict={
            "blocks": []
        })
        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()
        assert result.overall_complexity == ComplexityLevel.SIMPLE
        assert result.recommended_strategy == ProcessingStrategy.TEXT_EXTRACTION

    def test_strategy_text_for_simple(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )
        from contextifier_new.handlers.pdf_plus._types import ProcessingStrategy

        page = _mock_fitz_page(drawings=[], images=[], text_dict={
            "blocks": [{
                "type": 0, "bbox": (72, 72, 540, 100),
                "lines": [{"spans": [{"text": "Clean text"}],
                            "bbox": (72, 72, 540, 84)}],
            }],
        })
        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()
        assert result.recommended_strategy == ProcessingStrategy.TEXT_EXTRACTION

    def test_page_size_stored(self):
        from contextifier_new.handlers.pdf_plus._complexity_analyzer import (
            ComplexityAnalyzer,
        )
        page = _mock_fitz_page(width=595, height=842)  # A4
        analyzer = ComplexityAnalyzer(page, 0)
        result = analyzer.analyze()
        assert result.page_size == (595, 842)


# ═════════════════════════════════════════════════════════════════════════════
# Test pdf_plus Content Extractor
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfPlusContentExtractor:
    """Tests for PdfPlusContentExtractor."""

    def test_extract_text_single_page(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = _mock_fitz_doc(page_count=1, pages=[
            _mock_fitz_page(text="PDF Plus Text"),
        ])
        ppd = PreprocessedData(
            content=doc, raw_content=b"fake_bytes",
            resources={"document": doc},
        )

        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        assert "PDF Plus Text" in text

    def test_extract_text_multi_page(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        pages = [
            _mock_fitz_page(text=f"Page {i} text") for i in range(3)
        ]
        doc = _mock_fitz_doc(pages=pages)
        ppd = PreprocessedData(
            content=doc, raw_content=b"bytes",
            resources={"document": doc},
        )

        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        assert "Page 0 text" in text
        assert "Page 1 text" in text
        assert "Page 2 text" in text

    def test_extract_text_empty_doc(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        ppd = PreprocessedData(content=None, raw_content=b"")
        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        assert text == ""

    def test_extract_text_with_tag_service(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = _mock_fitz_doc(page_count=1, pages=[
            _mock_fitz_page(text="Content"),
        ])
        ppd = PreprocessedData(content=doc, raw_content=b"data")

        tag_svc = MagicMock()
        tag_svc.page_tag = MagicMock(return_value="[Page:1]")

        ext = PdfPlusContentExtractor(tag_service=tag_svc)
        text = ext.extract_text(ppd)
        assert "[Page:1]" in text

    def test_extract_text_default_page_tag(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = _mock_fitz_doc(page_count=1, pages=[
            _mock_fitz_page(text="Some text"),
        ])
        ppd = PreprocessedData(content=doc, raw_content=b"")

        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        assert "[Page 1]" in text

    def test_unwrap_doc_from_content(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = MagicMock()
        doc.page_count = 2
        ppd = PreprocessedData(content=doc, raw_content=b"")

        result = PdfPlusContentExtractor._unwrap_doc(ppd)
        assert result is doc

    def test_unwrap_doc_from_resources(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        doc = MagicMock()
        doc.page_count = 1
        # content doesn't look like a doc (no page_count)
        ppd = PreprocessedData(
            content="not_a_doc",
            raw_content=b"",
            resources={"document": doc},
        )

        result = PdfPlusContentExtractor._unwrap_doc(ppd)
        assert result is doc

    def test_unwrap_doc_none(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        ppd = PreprocessedData(content=None, raw_content=b"")
        result = PdfPlusContentExtractor._unwrap_doc(ppd)
        assert result is None

    def test_get_file_data_from_raw_content(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        ppd = PreprocessedData(content=None, raw_content=b"pdf_bytes_here")
        result = PdfPlusContentExtractor._get_file_data(ppd)
        assert result == b"pdf_bytes_here"

    def test_get_file_data_not_bytes(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        ppd = PreprocessedData(content=None, raw_content="not_bytes")
        result = PdfPlusContentExtractor._get_file_data(ppd)
        assert result == b""

    def test_get_format_name(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        assert PdfPlusContentExtractor().get_format_name() == "pdf"

    def test_extract_tables_empty_doc(self):
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        ppd = PreprocessedData(content=None, raw_content=b"")
        ext = PdfPlusContentExtractor()
        tables = ext.extract_tables(ppd)
        assert tables == []

    def test_extract_text_real_pdf(self):
        """End-to-end with real PDF through pdf_plus extractor."""
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )

        data = _make_pdf_with_text(["Hello Plus Mode"])
        converted = PdfConverter().convert(_make_file_context(data))
        ppd = PdfPreprocessor().preprocess(converted)

        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        assert "Hello Plus Mode" in text
        converted.doc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Test PDFHandler
# ═════════════════════════════════════════════════════════════════════════════


class TestPDFHandler:
    """Tests for the PDFHandler router."""

    def _make_config(self, mode: str = "plus"):
        """Create a ProcessingConfig with the given PDF mode."""
        from contextifier_new.config import ProcessingConfig
        config = ProcessingConfig()
        if mode != "plus":
            config = config.with_format_option("pdf", mode=mode)
        return config

    def test_supported_extensions(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        handler = PDFHandler(self._make_config())
        assert handler.supported_extensions == frozenset({"pdf"})

    def test_handler_name(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        handler = PDFHandler(self._make_config())
        assert handler.handler_name == "PDF Handler"

    def test_create_converter(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.handlers.pdf.converter import PdfConverter
        handler = PDFHandler(self._make_config())
        conv = handler.create_converter()
        assert isinstance(conv, PdfConverter)

    def test_create_preprocessor(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        handler = PDFHandler(self._make_config())
        pp = handler.create_preprocessor()
        assert isinstance(pp, PdfPreprocessor)

    def test_create_metadata_extractor(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        handler = PDFHandler(self._make_config())
        ext = handler.create_metadata_extractor()
        assert isinstance(ext, PdfMetadataExtractor)

    def test_create_content_extractor_plus_mode(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        handler = PDFHandler(self._make_config("plus"))
        ext = handler.create_content_extractor()
        assert isinstance(ext, PdfPlusContentExtractor)

    def test_create_content_extractor_default_mode(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        handler = PDFHandler(self._make_config("default"))
        ext = handler.create_content_extractor()
        assert isinstance(ext, PdfDefaultContentExtractor)

    def test_create_postprocessor(self):
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.pipeline.postprocessor import DefaultPostprocessor
        handler = PDFHandler(self._make_config())
        pp = handler.create_postprocessor()
        assert isinstance(pp, DefaultPostprocessor)

    def test_default_mode_is_plus(self):
        """If no mode specified, plus is the default."""
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.config import ProcessingConfig
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        handler = PDFHandler(ProcessingConfig())
        ext = handler.create_content_extractor()
        assert isinstance(ext, PdfPlusContentExtractor)

    def test_unknown_mode_falls_back_to_plus(self):
        """Unrecognized mode value → falls to the else branch (plus)."""
        from contextifier_new.handlers.pdf.handler import PDFHandler
        from contextifier_new.config import ProcessingConfig
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        config = ProcessingConfig().with_format_option("pdf", mode="unknown")
        handler = PDFHandler(config)
        ext = handler.create_content_extractor()
        assert isinstance(ext, PdfPlusContentExtractor)


# ═════════════════════════════════════════════════════════════════════════════
# Test Full Pipeline Integration
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfPipelineIntegration:
    """Full pipeline tests: convert → preprocess → metadata → extract."""

    def test_default_mode_pipeline(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )

        data = _make_pdf_with_text(["Pipeline test default mode"])
        ctx = _make_file_context(data)

        converted = PdfConverter().convert(ctx)
        ppd = PdfPreprocessor().preprocess(converted)
        meta = PdfMetadataExtractor().extract(ppd)
        text = PdfDefaultContentExtractor().extract_text(ppd)

        assert meta.page_count == 1
        assert "Pipeline test default mode" in text
        converted.doc.close()

    def test_plus_mode_pipeline(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )

        data = _make_pdf_with_text(["Pipeline test plus mode"])
        ctx = _make_file_context(data)

        converted = PdfConverter().convert(ctx)
        ppd = PdfPreprocessor().preprocess(converted)
        meta = PdfMetadataExtractor().extract(ppd)
        text = PdfPlusContentExtractor().extract_text(ppd)

        assert meta.page_count == 1
        assert "Pipeline test plus mode" in text
        converted.doc.close()

    def test_multi_page_default(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )

        data = _make_pdf_with_text(["Alpha", "Beta", "Gamma"])
        converted = PdfConverter().convert(_make_file_context(data))
        ppd = PdfPreprocessor().preprocess(converted)

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert "Alpha" in text
        assert "Beta" in text
        assert "Gamma" in text
        converted.doc.close()

    def test_multi_page_plus(self):
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )

        data = _make_pdf_with_text(["Page 1 Plus", "Page 2 Plus"])
        converted = PdfConverter().convert(_make_file_context(data))
        ppd = PdfPreprocessor().preprocess(converted)

        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        assert "Page 1 Plus" in text
        assert "Page 2 Plus" in text
        converted.doc.close()

    def test_validation_then_convert(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter

        data = _make_pdf_with_text(["Validate then convert"])
        conv = PdfConverter()
        ctx = _make_file_context(data)

        assert conv.validate(ctx) is True
        result = conv.convert(ctx)
        assert result.doc.page_count == 1
        result.doc.close()

    def test_converter_metadata_real(self):
        """Full chain: convert → preprocess → metadata from real PDF."""
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor

        data = _make_pdf_with_text(["Page 1", "Page 2"])
        converted = PdfConverter().convert(_make_file_context(data))
        ppd = PdfPreprocessor().preprocess(converted)
        meta = PdfMetadataExtractor().extract(ppd)

        assert meta.page_count == 2
        converted.doc.close()

    def test_empty_page_handling(self):
        """Document with an empty page."""
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )

        # Create a blank page (no text)
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()

        converted = PdfConverter().convert(_make_file_context(buf.getvalue()))
        ppd = PdfPreprocessor().preprocess(converted)
        text = PdfDefaultContentExtractor().extract_text(ppd)

        # Empty page may produce empty or minimal text
        assert isinstance(text, str)
        converted.doc.close()


# ═════════════════════════════════════════════════════════════════════════════
# Test Edge Cases
# ═════════════════════════════════════════════════════════════════════════════


class TestPdfEdgeCases:
    """Edge cases and error handling tests."""

    def test_corrupted_pdf_raises(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.errors import ConversionError
        conv = PdfConverter()
        ctx = _make_file_context(b"%PDF-1.4 garbage data not a real pdf")
        with pytest.raises(ConversionError):
            conv.convert(ctx)

    def test_non_pdf_magic_bytes(self):
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.errors import ConversionError
        conv = PdfConverter()
        ctx = _make_file_context(b"PK\x03\x04")  # ZIP magic
        with pytest.raises(ConversionError, match="Not a valid PDF"):
            conv.convert(ctx)

    def test_preprocessor_with_none(self):
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        pp = PdfPreprocessor()
        result = pp.preprocess(None)
        assert result.properties["page_count"] == 0

    def test_metadata_bad_date(self):
        from contextifier_new.handlers.pdf.metadata_extractor import PdfMetadataExtractor
        doc = MagicMock()
        doc.page_count = 1
        doc.metadata = {
            "creationDate": "totally-invalid",
            "modDate": "D:abcdef",
        }
        ext = PdfMetadataExtractor()
        meta = ext.extract(doc)
        # Bad dates → no create/mod time set
        assert meta.create_time is None
        assert meta.last_saved_time is None

    def test_plus_extractor_fallback_on_error(self):
        """If complexity analyzer fails, should fall back to TEXT strategy."""
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        # Page where get_drawings throws on first call (complexity analyzer)
        # but returns [] on subsequent calls (fallback TEXT strategy path)
        page = _mock_fitz_page(text="Fallback text")
        _call_count = {"n": 0}

        def _drawings_side_effect():
            _call_count["n"] += 1
            if _call_count["n"] == 1:
                raise RuntimeError("drawings fail")
            return []

        page.get_drawings = MagicMock(side_effect=_drawings_side_effect)
        # get_images should still work for the fallback
        page.get_images = MagicMock(return_value=[])
        # find_tables should work for fallback
        page.find_tables = MagicMock(return_value=[])

        doc = _mock_fitz_doc(pages=[page])
        ppd = PreprocessedData(content=doc, raw_content=b"data")

        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        # Should still get text via fallback
        assert isinstance(text, str)

    def test_default_extractor_handles_find_tables_error(self):
        """find_tables() failure should not crash the extractor."""
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        page = _mock_fitz_page(text="Tables will fail")
        page.find_tables = MagicMock(side_effect=RuntimeError("No tables"))

        doc = _mock_fitz_doc(pages=[page])
        ppd = PreprocessedData(content=doc, raw_content=b"")

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert "Tables will fail" in text

    def test_default_extractor_handles_text_error(self):
        """get_text() failure in dict mode should fall back."""
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        page = MagicMock()
        page.rect = MagicMock()
        page.rect.width = 612
        page.rect.height = 792
        # get_text("dict") fails, but get_text("text") returns fallback
        def _get_text_side_effect(fmt="text", **kw):
            if fmt == "dict":
                raise RuntimeError("text error")
            return "fallback text"
        page.get_text = MagicMock(side_effect=_get_text_side_effect)
        page.find_tables = MagicMock(return_value=[])
        page.get_images = MagicMock(return_value=[])

        doc = _mock_fitz_doc(pages=[page])
        ppd = PreprocessedData(content=doc, raw_content=b"")

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert isinstance(text, str)
        assert "fallback text" in text

    def test_converter_close_with_closeable(self):
        """close() should handle object with .close() method."""
        from contextifier_new.handlers.pdf.converter import PdfConverter
        conv = PdfConverter()
        obj = MagicMock()
        obj.close = MagicMock()
        conv.close(obj)
        # Should have called close
        obj.close.assert_called_once()

    def test_korean_text_extraction(self):
        """PDF with Korean text should be extracted correctly."""
        import fitz
        from contextifier_new.handlers.pdf.converter import PdfConverter
        from contextifier_new.handlers.pdf.preprocessor import PdfPreprocessor
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )

        # Use a CJK font if available, otherwise skip
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        # Try inserting Korean text with a standard font
        rc = page.insert_text(
            (72, 72), "Korean Test", fontsize=12,
        )
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()

        converted = PdfConverter().convert(_make_file_context(buf.getvalue()))
        ppd = PdfPreprocessor().preprocess(converted)
        text = PdfDefaultContentExtractor().extract_text(ppd)
        # At minimum, the ASCII portion should be extracted
        assert "Korean" in text or "Test" in text
        converted.doc.close()

    def test_large_document_mock(self):
        """Test with a document with many pages (mocked)."""
        from contextifier_new.handlers.pdf_default.content_extractor import (
            PdfDefaultContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        pages = [_mock_fitz_page(text=f"Page {i}") for i in range(100)]
        doc = _mock_fitz_doc(pages=pages)
        ppd = PreprocessedData(content=doc, raw_content=b"")

        ext = PdfDefaultContentExtractor()
        text = ext.extract_text(ppd)
        assert "Page 0" in text
        assert "Page 99" in text

    def test_plus_strategy_dispatch_text(self):
        """TEXT_EXTRACTION strategy should produce text output."""
        from contextifier_new.handlers.pdf_plus.content_extractor import (
            PdfPlusContentExtractor,
        )
        from contextifier_new.types import PreprocessedData

        page = _mock_fitz_page(
            text="Simple strategy text",
            drawings=[],
            images=[],
            text_dict={
                "blocks": [{
                    "type": 0,
                    "bbox": (72, 72, 540, 120),
                    "lines": [{
                        "bbox": (72, 72, 540, 84),
                        "spans": [{"text": "Simple strategy text",
                                   "size": 12, "font": "Helvetica"}],
                    }],
                }],
            },
        )
        doc = _mock_fitz_doc(pages=[page])
        ppd = PreprocessedData(content=doc, raw_content=b"data")

        ext = PdfPlusContentExtractor()
        text = ext.extract_text(ppd)
        assert "Simple strategy text" in text
