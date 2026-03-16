"""
Comprehensive tests for the PPT handler pipeline.

Tests cover:
1. Constants: Magic bytes, OLE stream names
2. PptConverter: OLE2 opening, validation, close
3. PptPreprocessor: Stream extraction, image parsing
4. PptMetadataExtractor: OLE summary properties
5. PptContentExtractor: Text record parsing, slide grouping, images
6. PPTHandler: Integration, delegation, pipeline
7. Binary parsing helpers: _parse_text_records, _group_into_slides, _clean_text
"""

from __future__ import annotations

import io
import struct
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Imports under test
# ═══════════════════════════════════════════════════════════════════════════════

from contextifier_new.handlers.ppt._constants import (
    OLE2_MAGIC,
    ZIP_MAGIC,
    PP_DOCUMENT_STREAM,
    CURRENT_USER_STREAM,
    SUMMARY_INFO_STREAM,
    DOC_SUMMARY_STREAM,
    PPT_KNOWN_STREAMS,
)
from contextifier_new.handlers.ppt.converter import (
    PptConverter,
    PptConvertedData,
)
from contextifier_new.handlers.ppt.preprocessor import PptPreprocessor
from contextifier_new.handlers.ppt.metadata_extractor import PptMetadataExtractor
from contextifier_new.handlers.ppt.content_extractor import (
    PptContentExtractor,
    _parse_text_records,
    _group_into_slides,
    _clean_text,
)
from contextifier_new.handlers.ppt.handler import PPTHandler
from contextifier_new.types import (
    DocumentMetadata,
    ExtractionResult,
    FileContext,
    PreprocessedData,
)
from contextifier_new.config import ProcessingConfig
from contextifier_new.errors import ConversionError, PreprocessingError


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ctx(data: bytes, ext: str = "ppt", name: str = "test.ppt") -> FileContext:
    """Create a minimal FileContext dict."""
    return {
        "file_data": data,
        "file_extension": ext,
        "file_name": name,
        "file_path": f"/tmp/{name}",
        "file_category": "presentation",
        "file_stream": io.BytesIO(data),
        "file_size": len(data),
    }


def _make_minimal_ole() -> bytes:
    """
    Build a minimal OLE2 compound binary file that olefile can open.

    We use olefile itself to create a valid OLE2 file in memory.
    """
    import olefile
    import tempfile
    import os

    # Create a temporary file because olefile.OleFileIO needs a file to write
    # We'll use a trick: create via writing a Word Document stream
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ppt")
    tmp_path = tmp.name
    tmp.close()

    try:
        # olefile can't easily create files, so we use a different strategy:
        # Build a minimal OLE2 structure by hand using the OLE2 header format.
        # Actually, let's just use a simple approach with olefile.
        # Since olefile has limited write support, let's craft manually.
        pass
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Alternative: use the built-in olefile write capabilities
    # Actually, olefile doesn't support creating new files easily.
    # We'll mock the OLE object instead for most tests.
    return b""


def _make_text_record(rec_type: int, text: str, *, encoding: str = "utf-16-le") -> bytes:
    """Build a single PPT binary record with given type and text."""
    if encoding == "utf-16-le":
        text_bytes = text.encode("utf-16-le")
    elif encoding == "cp1252":
        text_bytes = text.encode("cp1252")
    else:
        text_bytes = text.encode(encoding)

    rec_ver_instance = 0x0000  # version=0, instance=0
    header = struct.pack("<HHI", rec_ver_instance, rec_type, len(text_bytes))
    return header + text_bytes


def _make_pp_stream(*texts: str, record_type: int = 0x0FA8, gap: int = 100) -> bytes:
    """Build a fake PowerPoint Document stream with text records separated by gaps."""
    stream = b""
    for text in texts:
        stream += _make_text_record(record_type, text)
        # Add a gap of padding between records
        stream += b"\x00" * gap
    return stream


def _make_pp_stream_multi_slide(slides: List[List[str]], gap_between_slides: int = 2000) -> bytes:
    """Build a PP stream with distinct slide boundaries (large gaps between slides)."""
    stream = b""
    for slide_texts in slides:
        for text in slide_texts:
            stream += _make_text_record(0x0FA8, text)
            stream += b"\x00" * 50  # Small gap within slide
        stream += b"\x00" * gap_between_slides  # Large gap between slides
    return stream


def _make_image_record(image_data: bytes, rec_type: int = 0xF01E, instance: int = 0) -> bytes:
    """Build a PPT Pictures stream image record."""
    # Record header: recVer/recInstance (2 bytes), recType (2 bytes), recLen (4 bytes)
    rec_ver_instance = (instance << 4) | 0x0  # instance bits, version 0
    header_size = 25 if (instance & 1) else 17
    total_data = b"\x00" * header_size + image_data
    header = struct.pack("<HHI", rec_ver_instance, rec_type, len(total_data))
    return header + total_data


def _mock_ole(
    *,
    pp_stream: Optional[bytes] = None,
    pictures_stream: Optional[bytes] = None,
    metadata: Optional[object] = None,
) -> MagicMock:
    """Create a mock OLE2 file object."""
    ole = MagicMock()

    def exists_fn(name):
        if name == PP_DOCUMENT_STREAM and pp_stream is not None:
            return True
        if name == "Pictures" and pictures_stream is not None:
            return True
        return False

    ole.exists.side_effect = exists_fn

    def openstream_fn(name):
        if name == PP_DOCUMENT_STREAM and pp_stream is not None:
            return io.BytesIO(pp_stream)
        if name == "Pictures" and pictures_stream is not None:
            return io.BytesIO(pictures_stream)
        raise Exception(f"Stream not found: {name}")

    ole.openstream.side_effect = openstream_fn

    if metadata is not None:
        ole.get_metadata.return_value = metadata
    else:
        meta = MagicMock()
        meta.title = None
        meta.subject = None
        meta.author = None
        meta.keywords = None
        meta.comments = None
        meta.last_saved_by = None
        meta.create_time = None
        meta.last_saved_time = None
        meta.revision_number = None
        meta.category = None
        ole.get_metadata.return_value = meta

    return ole


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Tests for PPT constants."""

    def test_ole2_magic_length(self):
        assert len(OLE2_MAGIC) == 8

    def test_ole2_magic_value(self):
        assert OLE2_MAGIC == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

    def test_zip_magic_length(self):
        assert len(ZIP_MAGIC) == 4

    def test_zip_magic_value(self):
        assert ZIP_MAGIC == b"PK\x03\x04"

    def test_pp_document_stream(self):
        assert PP_DOCUMENT_STREAM == "PowerPoint Document"

    def test_current_user_stream(self):
        assert CURRENT_USER_STREAM == "Current User"

    def test_summary_info_stream(self):
        assert "\x05" in SUMMARY_INFO_STREAM

    def test_known_streams_frozen(self):
        assert isinstance(PPT_KNOWN_STREAMS, frozenset)

    def test_known_streams_contains_pp(self):
        assert PP_DOCUMENT_STREAM in PPT_KNOWN_STREAMS


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Converter
# ═══════════════════════════════════════════════════════════════════════════════

class TestPptConverter:
    """Tests for PptConverter (OLE2 opening)."""

    def test_format_name(self):
        c = PptConverter()
        assert c.get_format_name() == "ppt"

    def test_validate_with_ole2_magic(self):
        c = PptConverter()
        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        assert c.validate(ctx) is True

    def test_validate_with_zip_magic(self):
        c = PptConverter()
        ctx = _make_ctx(ZIP_MAGIC + b"\x00" * 100)
        assert c.validate(ctx) is False

    def test_validate_with_empty_data(self):
        c = PptConverter()
        ctx = _make_ctx(b"")
        assert c.validate(ctx) is False

    def test_validate_with_short_data(self):
        c = PptConverter()
        ctx = _make_ctx(b"\xd0\xcf\x11")
        assert c.validate(ctx) is False

    def test_convert_empty_raises(self):
        c = PptConverter()
        ctx = _make_ctx(b"")
        with pytest.raises(ConversionError, match="Empty"):
            c.convert(ctx)

    def test_convert_invalid_data_raises(self):
        c = PptConverter()
        ctx = _make_ctx(b"not an OLE2 file at all" * 10)
        with pytest.raises(ConversionError, match="OLE2"):
            c.convert(ctx)

    @patch("contextifier_new.handlers.ppt.converter.olefile.OleFileIO")
    def test_convert_success(self, mock_oleclass):
        """Successful conversion returns PptConvertedData."""
        mock_ole = MagicMock()
        mock_oleclass.return_value = mock_ole

        c = PptConverter()
        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = c.convert(ctx)

        assert isinstance(result, PptConvertedData)
        assert result.ole is mock_ole
        assert result.file_extension == "ppt"

    @patch("contextifier_new.handlers.ppt.converter.olefile.OleFileIO")
    def test_convert_preserves_extension(self, mock_oleclass):
        mock_oleclass.return_value = MagicMock()
        c = PptConverter()
        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100, ext="ppt")
        result = c.convert(ctx)
        assert result.file_extension == "ppt"

    def test_close_ppt_converted_data(self):
        c = PptConverter()
        mock_ole = MagicMock()
        cd = PptConvertedData(ole=mock_ole, file_extension="ppt")
        c.close(cd)
        mock_ole.close.assert_called_once()

    def test_close_ole_directly(self):
        import olefile
        c = PptConverter()
        mock_ole = MagicMock(spec=olefile.OleFileIO)
        c.close(mock_ole)
        mock_ole.close.assert_called_once()

    def test_close_none_no_error(self):
        c = PptConverter()
        c.close(None)  # Should not raise

    def test_close_exception_suppressed(self):
        c = PptConverter()
        mock_ole = MagicMock()
        mock_ole.close.side_effect = Exception("close error")
        cd = PptConvertedData(ole=mock_ole, file_extension="ppt")
        c.close(cd)  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Preprocessor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPptPreprocessor:
    """Tests for PptPreprocessor (stream extraction)."""

    def test_format_name(self):
        p = PptPreprocessor()
        assert p.get_format_name() == "ppt"

    def test_preprocess_none_raises(self):
        p = PptPreprocessor()
        with pytest.raises(PreprocessingError, match="None"):
            p.preprocess(None)

    def test_preprocess_with_pp_stream(self):
        ole = _mock_ole(pp_stream=b"\x01\x02\x03")
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        p = PptPreprocessor()
        result = p.preprocess(cd)

        assert isinstance(result, PreprocessedData)
        assert result.resources["pp_stream"] == b"\x01\x02\x03"
        assert result.properties["has_pp_stream"] is True
        assert result.properties["stream_size"] == 3

    def test_preprocess_no_pp_stream(self):
        ole = _mock_ole()  # No PP stream
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        p = PptPreprocessor()
        result = p.preprocess(cd)

        assert result.resources["pp_stream"] is None
        assert result.properties["has_pp_stream"] is False

    def test_preprocess_with_images(self):
        # Create a Pictures stream with one PNG image record
        fake_image = b"\x89PNG" + b"\x00" * 50
        pictures = _make_image_record(fake_image, rec_type=0xF01E)
        ole = _mock_ole(pp_stream=b"\x00" * 16, pictures_stream=pictures)
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        p = PptPreprocessor()
        result = p.preprocess(cd)

        assert len(result.resources["image_streams"]) >= 1
        assert result.properties["image_count"] >= 1

    def test_preprocess_no_pictures_stream(self):
        ole = _mock_ole(pp_stream=b"\x00" * 16)
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        p = PptPreprocessor()
        result = p.preprocess(cd)

        assert result.resources["image_streams"] == []
        assert result.properties["image_count"] == 0

    def test_preprocess_ole_directly(self):
        """Can pass ole directly if it has exists/openstream."""
        ole = _mock_ole(pp_stream=b"\xAA" * 10)
        # Pass without wrapping in PptConvertedData
        p = PptPreprocessor()
        # Should handle via the else branch
        result = p.preprocess(ole)
        # Should work because mock has .ole attribute checking

    def test_preprocess_pp_stream_read_error(self):
        """If PP stream read fails, should handle gracefully."""
        ole = MagicMock()
        ole.exists.return_value = True
        ole.openstream.side_effect = Exception("Stream read error")
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        p = PptPreprocessor()
        result = p.preprocess(cd)

        assert result.resources["pp_stream"] is None
        assert result.properties["has_pp_stream"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Metadata Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPptMetadataExtractor:
    """Tests for PptMetadataExtractor (OLE summary information)."""

    def test_format_name(self):
        m = PptMetadataExtractor()
        assert m.get_format_name() == "ppt"

    def test_extract_from_none(self):
        m = PptMetadataExtractor()
        result = m.extract(None)
        assert isinstance(result, DocumentMetadata)

    def test_extract_full_metadata(self):
        meta = MagicMock()
        meta.title = "My Presentation"
        meta.subject = "Demo Subject"
        meta.author = "Test Author"
        meta.keywords = "test, ppt"
        meta.comments = "A comment"
        meta.last_saved_by = "Editor"
        meta.create_time = datetime(2024, 1, 15, 10, 30)
        meta.last_saved_time = datetime(2024, 6, 20, 14, 0)
        meta.revision_number = 5
        meta.category = "Presentations"

        ole = MagicMock()
        ole.get_metadata.return_value = meta
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        m = PptMetadataExtractor()
        result = m.extract(cd)

        assert result.title == "My Presentation"
        assert result.subject == "Demo Subject"
        assert result.author == "Test Author"
        assert result.keywords == "test, ppt"
        assert result.comments == "A comment"
        assert result.last_saved_by == "Editor"
        assert result.create_time == datetime(2024, 1, 15, 10, 30)
        assert result.last_saved_time == datetime(2024, 6, 20, 14, 0)
        assert result.revision == "5"
        assert result.category == "Presentations"

    def test_extract_empty_metadata(self):
        meta = MagicMock()
        meta.title = None
        meta.subject = None
        meta.author = None
        meta.keywords = None
        meta.comments = None
        meta.last_saved_by = None
        meta.create_time = None
        meta.last_saved_time = None
        meta.revision_number = None
        meta.category = None

        ole = MagicMock()
        ole.get_metadata.return_value = meta
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        m = PptMetadataExtractor()
        result = m.extract(cd)

        assert result.title is None
        assert result.author is None

    def test_extract_bytes_metadata(self):
        """Metadata fields stored as bytes should be decoded."""
        meta = MagicMock()
        meta.title = b"Bytes Title"
        meta.subject = None
        meta.author = b"\xc3\xa9l\xc3\xa8ve"  # "élève" in utf-8
        meta.keywords = None
        meta.comments = None
        meta.last_saved_by = None
        meta.create_time = None
        meta.last_saved_time = None
        meta.revision_number = None
        meta.category = None

        ole = MagicMock()
        ole.get_metadata.return_value = meta
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        m = PptMetadataExtractor()
        result = m.extract(cd)

        assert result.title == "Bytes Title"
        assert result.author is not None

    def test_extract_from_preprocessed_data(self):
        """Can unwrap OLE from PreprocessedData."""
        import olefile

        ole = MagicMock(spec=olefile.OleFileIO)
        meta = MagicMock()
        meta.title = "From Preprocessed"
        meta.subject = None
        meta.author = None
        meta.keywords = None
        meta.comments = None
        meta.last_saved_by = None
        meta.create_time = None
        meta.last_saved_time = None
        meta.revision_number = None
        meta.category = None
        ole.get_metadata.return_value = meta

        ppd = PreprocessedData(
            content=ole,
            raw_content=ole,
            encoding="utf-8",
            resources={},
            properties={},
        )

        m = PptMetadataExtractor()
        result = m.extract(ppd)
        assert result.title == "From Preprocessed"

    def test_extract_metadata_error_returns_empty(self):
        """If get_metadata() fails, return empty DocumentMetadata."""
        ole = MagicMock()
        ole.get_metadata.side_effect = Exception("corrupt metadata")

        m = PptMetadataExtractor()
        result = m.extract(ole)
        assert isinstance(result, DocumentMetadata)

    def test_decode_empty_string(self):
        """Empty strings should decode to None."""
        assert PptMetadataExtractor._decode("") is None
        assert PptMetadataExtractor._decode("   ") is None

    def test_decode_normal_string(self):
        assert PptMetadataExtractor._decode("hello") == "hello"

    def test_to_datetime_none(self):
        assert PptMetadataExtractor._to_datetime(None) is None

    def test_to_datetime_datetime(self):
        dt = datetime(2024, 1, 1)
        assert PptMetadataExtractor._to_datetime(dt) == dt

    def test_to_datetime_nonstandard(self):
        """Non-datetime values return None."""
        assert PptMetadataExtractor._to_datetime("not a date") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Content Extractor — Binary parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryParsing:
    """Tests for PPT binary record parsing functions."""

    def test_parse_text_chars_atom(self):
        """TextCharsAtom (0x0FA8) should be decoded as UTF-16LE."""
        stream = _make_text_record(0x0FA8, "Hello Unicode")
        records = _parse_text_records(stream)
        assert len(records) == 1
        assert records[0][1] == "Hello Unicode"

    def test_parse_text_bytes_atom(self):
        """TextBytesAtom (0x0FA0) should be decoded as CP1252."""
        stream = _make_text_record(0x0FA0, "Hello ANSI", encoding="cp1252")
        records = _parse_text_records(stream)
        assert len(records) == 1
        assert records[0][1] == "Hello ANSI"

    def test_parse_cstring_atom(self):
        """CString (0x0FBA) should be decoded as UTF-16LE."""
        stream = _make_text_record(0x0FBA, "Header Title")
        records = _parse_text_records(stream)
        assert len(records) == 1
        assert records[0][1] == "Header Title"

    def test_parse_multiple_records(self):
        stream = (
            _make_text_record(0x0FA8, "First")
            + _make_text_record(0x0FA0, "Second", encoding="cp1252")
            + _make_text_record(0x0FBA, "Third")
        )
        records = _parse_text_records(stream)
        assert len(records) == 3
        assert records[0][1] == "First"
        assert records[1][1] == "Second"
        assert records[2][1] == "Third"

    def test_parse_preserves_offset_order(self):
        """Records should be in stream order with increasing offsets."""
        stream = (
            _make_text_record(0x0FA8, "A")
            + _make_text_record(0x0FA8, "B")
            + _make_text_record(0x0FA8, "C")
        )
        records = _parse_text_records(stream)
        offsets = [r[0] for r in records]
        assert offsets == sorted(offsets)

    def test_parse_skips_unknown_records(self):
        """Unknown record types should be skipped."""
        # Build a non-text record
        non_text = struct.pack("<HHI", 0, 0x1234, 4) + b"\xAA\xBB\xCC\xDD"
        stream = non_text + _make_text_record(0x0FA8, "Real Text")
        records = _parse_text_records(stream)
        assert len(records) == 1
        assert records[0][1] == "Real Text"

    def test_parse_empty_stream(self):
        records = _parse_text_records(b"")
        assert records == []

    def test_parse_short_stream(self):
        """Stream too short for a header should return empty."""
        records = _parse_text_records(b"\x00\x01\x02")
        assert records == []

    def test_parse_record_with_invalid_length(self):
        """Record with length extending past end should stop parsing."""
        # Header says 1000 bytes but only 10 are present
        header = struct.pack("<HHI", 0, 0x0FA8, 1000)
        stream = header + b"\x00" * 10
        records = _parse_text_records(stream)
        assert records == []

    def test_parse_empty_text_skipped(self):
        """Empty text records should be skipped."""
        stream = _make_text_record(0x0FA8, "")
        records = _parse_text_records(stream)
        # Empty string after clean should be skipped
        assert records == []

    def test_clean_text_removes_nulls(self):
        assert _clean_text("Hello\x00World") == "Hello World" or _clean_text("Hello\x00World") == "HelloWorld"

    def test_clean_text_normalizes_newlines(self):
        result = _clean_text("line1\r\nline2\rline3")
        assert "\r" not in result
        assert "line1" in result
        assert "line2" in result

    def test_clean_text_removes_control_chars(self):
        result = _clean_text("text\x01\x02\x03here")
        assert "\x01" not in result
        assert "text" in result

    def test_clean_text_strips_whitespace(self):
        result = _clean_text("  hello  ")
        assert result == "hello"

    def test_clean_text_empty(self):
        result = _clean_text("")
        assert result == ""


class TestSlideGrouping:
    """Tests for _group_into_slides heuristic."""

    def test_empty_records(self):
        assert _group_into_slides([]) == []

    def test_single_record(self):
        result = _group_into_slides([(0, "Only text")])
        assert result == [["Only text"]]

    def test_close_records_same_slide(self):
        """Records close together should be on the same slide."""
        records = [(100, "A"), (150, "B"), (200, "C")]
        result = _group_into_slides(records)
        # All close together → likely one slide
        assert len(result) == 1
        assert result[0] == ["A", "B", "C"]

    def test_large_gap_splits_slides(self):
        """Records with large gaps should be on different slides."""
        # Need enough records so median gap is small, then one big gap
        records = [
            (100, "Slide 1 Title"),
            (150, "Slide 1 Body"),
            (200, "Slide 1 Footer"),
            (250, "Slide 1 Notes"),
            (50000, "Slide 2 Title"),
            (50050, "Slide 2 Body"),
        ]
        result = _group_into_slides(records)
        # Gaps: [50, 50, 50, 49750, 50] → median=50, threshold=max(150,1000)=1000
        # 49750 > 1000 → should split
        assert len(result) >= 2

    def test_two_records(self):
        """Two records should always produce at least one slide."""
        records = [(0, "A"), (10, "B")]
        result = _group_into_slides(records)
        assert len(result) >= 1

    def test_consistent_gaps_one_slide(self):
        """Uniform small gaps → single slide."""
        records = [(i * 20, f"Text {i}") for i in range(10)]
        result = _group_into_slides(records)
        # Uniform gaps → everything in one slide
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Content Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPptContentExtractor:
    """Tests for PptContentExtractor."""

    def test_format_name(self):
        ext = PptContentExtractor()
        assert ext.get_format_name() == "ppt"

    def test_extract_text_empty_stream(self):
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": None, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert result == ""

    def test_extract_text_single_record(self):
        stream = _make_text_record(0x0FA8, "Hello World")
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": stream, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert "Hello World" in result

    def test_extract_text_multiple_records(self):
        stream = (
            _make_text_record(0x0FA8, "Title")
            + _make_text_record(0x0FA8, "Content")
        )
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": stream, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert "Title" in result
        assert "Content" in result

    def test_extract_text_has_slide_tags(self):
        """Output should contain [Slide:N] tags."""
        stream = _make_text_record(0x0FA8, "Slide content")
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": stream, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert "[Slide:1]" in result

    def test_extract_text_with_tag_service(self):
        """When tag_service is available, use it for slide tags."""
        tag_svc = MagicMock()
        tag_svc.make_slide_tag.return_value = "[SLIDE:1]"

        stream = _make_text_record(0x0FA8, "Text")
        ext = PptContentExtractor(tag_service=tag_svc)
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": stream, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert "[SLIDE:1]" in result

    def test_extract_text_mixed_record_types(self):
        stream = (
            _make_text_record(0x0FA8, "Unicode text")
            + _make_text_record(0x0FA0, "ANSI text", encoding="cp1252")
        )
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": stream, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert "Unicode text" in result
        assert "ANSI text" in result

    def test_extract_text_no_excessive_newlines(self):
        """Should not have more than 2 consecutive newlines."""
        stream = (
            _make_text_record(0x0FA8, "A")
            + _make_text_record(0x0FA8, "B")
        )
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": stream, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert "\n\n\n" not in result

    def test_extract_images_empty(self):
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": None, "image_streams": []},
            properties={},
        )
        result = ext.extract_images(ppd)
        assert result == []

    def test_extract_images_no_service(self):
        """Without image_service, should return empty."""
        ext = PptContentExtractor(image_service=None)
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": None, "image_streams": [b"\x89PNG" + b"\x00" * 20]},
            properties={},
        )
        result = ext.extract_images(ppd)
        assert result == []

    def test_extract_images_with_service(self):
        img_svc = MagicMock()
        img_svc.save_and_tag.return_value = "[IMG:ppt_image_0]"

        ext = PptContentExtractor(image_service=img_svc)
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": None, "image_streams": [b"\x89PNG" + b"\x00" * 20]},
            properties={},
        )
        result = ext.extract_images(ppd)
        assert len(result) == 1
        assert result[0] == "[IMG:ppt_image_0]"
        img_svc.save_and_tag.assert_called_once()

    def test_extract_images_deduplication(self):
        """Duplicate images (same hash) extracted only once."""
        img_svc = MagicMock()
        img_svc.save_and_tag.return_value = "[IMG:ppt_image_0]"

        same_data = b"\x89PNG" + b"\x00" * 50
        ext = PptContentExtractor(image_service=img_svc)
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": None, "image_streams": [same_data, same_data, same_data]},
            properties={},
        )
        result = ext.extract_images(ppd)
        assert len(result) == 1
        assert img_svc.save_and_tag.call_count == 1

    def test_extract_images_different_images(self):
        """Different images should all be extracted."""
        img_svc = MagicMock()
        img_svc.save_and_tag.side_effect = [
            "[IMG:img0]", "[IMG:img1]", "[IMG:img2]"
        ]

        ext = PptContentExtractor(image_service=img_svc)
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={
                "pp_stream": None,
                "image_streams": [b"\x01" * 30, b"\x02" * 30, b"\x03" * 30],
            },
            properties={},
        )
        result = ext.extract_images(ppd)
        assert len(result) == 3

    def test_extract_images_service_error(self):
        """If image_service.save_and_tag fails, skip that image."""
        img_svc = MagicMock()
        img_svc.save_and_tag.side_effect = Exception("save failed")

        ext = PptContentExtractor(image_service=img_svc)
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": None, "image_streams": [b"\x89PNG" + b"\x00" * 20]},
            properties={},
        )
        result = ext.extract_images(ppd)
        assert result == []

    def test_make_slide_tag_no_service(self):
        ext = PptContentExtractor(tag_service=None)
        tag = ext._make_slide_tag(3)
        assert tag == "[Slide:3]"

    def test_make_slide_tag_with_service(self):
        tag_svc = MagicMock()
        tag_svc.make_slide_tag.return_value = "[CUSTOM:3]"
        ext = PptContentExtractor(tag_service=tag_svc)
        tag = ext._make_slide_tag(3)
        assert tag == "[CUSTOM:3]"

    def test_make_slide_tag_service_error_fallback(self):
        tag_svc = MagicMock()
        tag_svc.make_slide_tag.side_effect = Exception("tag error")
        ext = PptContentExtractor(tag_service=tag_svc)
        tag = ext._make_slide_tag(5)
        assert tag == "[Slide:5]"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PPTHandler (handler + delegation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPPTHandler:
    """Tests for PPTHandler configuration and properties."""

    def test_supported_extensions(self):
        config = ProcessingConfig()
        handler = PPTHandler(config)
        assert handler.supported_extensions == frozenset({"ppt"})

    def test_handler_name(self):
        config = ProcessingConfig()
        handler = PPTHandler(config)
        assert handler.handler_name == "PPT Handler"

    def test_creates_ppt_converter(self):
        config = ProcessingConfig()
        handler = PPTHandler(config)
        converter = handler.create_converter()
        assert isinstance(converter, PptConverter)

    def test_creates_ppt_preprocessor(self):
        config = ProcessingConfig()
        handler = PPTHandler(config)
        preprocessor = handler.create_preprocessor()
        assert isinstance(preprocessor, PptPreprocessor)

    def test_creates_ppt_metadata_extractor(self):
        config = ProcessingConfig()
        handler = PPTHandler(config)
        extractor = handler.create_metadata_extractor()
        assert isinstance(extractor, PptMetadataExtractor)

    def test_creates_ppt_content_extractor(self):
        config = ProcessingConfig()
        handler = PPTHandler(config)
        extractor = handler.create_content_extractor()
        assert isinstance(extractor, PptContentExtractor)


class TestDelegation:
    """Tests for the PPTHandler delegation logic."""

    def test_check_delegation_zip_triggers(self):
        """ZIP content (misnamed PPTX) should trigger delegation."""
        config = ProcessingConfig()
        handler = PPTHandler(config)

        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.process.return_value = ExtractionResult(text="PPTX text")
        mock_registry.get_handler.return_value = mock_handler
        handler._handler_registry = mock_registry

        ctx = _make_ctx(ZIP_MAGIC + b"\x00" * 100)
        result = handler._check_delegation(ctx)

        assert result is not None

    def test_check_delegation_ole2_returns_none(self):
        """Genuine OLE2 should NOT delegate (returns None)."""
        config = ProcessingConfig()
        handler = PPTHandler(config)

        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = handler._check_delegation(ctx)
        assert result is None

    def test_check_delegation_empty_returns_none(self):
        """Empty data should NOT delegate."""
        config = ProcessingConfig()
        handler = PPTHandler(config)

        ctx = _make_ctx(b"")
        result = handler._check_delegation(ctx)
        assert result is None

    def test_check_delegation_short_returns_none(self):
        """Data shorter than 8 bytes should NOT delegate."""
        config = ProcessingConfig()
        handler = PPTHandler(config)

        ctx = _make_ctx(b"\x01\x02")
        result = handler._check_delegation(ctx)
        assert result is None

    def test_check_delegation_random_bytes_returns_none(self):
        """Unknown magic bytes should NOT delegate."""
        config = ProcessingConfig()
        handler = PPTHandler(config)

        ctx = _make_ctx(b"\xFF\xFE\xFD\xFC\xFB\xFA\xF9\xF8" + b"\x00" * 50)
        result = handler._check_delegation(ctx)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Full Pipeline (integration tests using mocked OLE)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Integration tests for the full PPT pipeline."""

    @patch("contextifier_new.handlers.ppt.converter.olefile.OleFileIO")
    def test_full_pipeline_basic(self, mock_oleclass):
        """Full pipeline: convert → preprocess → metadata → extract → postprocess."""
        pp_stream = _make_text_record(0x0FA8, "Slide 1 Content")
        mock_ole = _mock_ole(pp_stream=pp_stream)
        mock_oleclass.return_value = mock_ole

        config = ProcessingConfig()
        handler = PPTHandler(config)

        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = handler.process(ctx)

        assert isinstance(result, ExtractionResult)
        assert "Slide 1 Content" in result.text

    @patch("contextifier_new.handlers.ppt.converter.olefile.OleFileIO")
    def test_full_pipeline_with_metadata(self, mock_oleclass):
        """Pipeline should include metadata when extracted."""
        pp_stream = _make_text_record(0x0FA8, "Content")

        meta = MagicMock()
        meta.title = "Pipeline Title"
        meta.subject = None
        meta.author = "Author Name"
        meta.keywords = None
        meta.comments = None
        meta.last_saved_by = None
        meta.create_time = None
        meta.last_saved_time = None
        meta.revision_number = None
        meta.category = None

        mock_ole = _mock_ole(pp_stream=pp_stream, metadata=meta)
        mock_oleclass.return_value = mock_ole

        config = ProcessingConfig()
        handler = PPTHandler(config)

        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = handler.process(ctx)

        assert result.metadata is not None
        assert result.metadata.title == "Pipeline Title"
        assert result.metadata.author == "Author Name"

    @patch("contextifier_new.handlers.ppt.converter.olefile.OleFileIO")
    def test_full_pipeline_empty_pp_stream(self, mock_oleclass):
        """Pipeline with no PowerPoint Document stream should still work."""
        mock_ole = _mock_ole()  # No PP stream
        mock_oleclass.return_value = mock_ole

        config = ProcessingConfig()
        handler = PPTHandler(config)

        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = handler.process(ctx)

        assert isinstance(result, ExtractionResult)

    @patch("contextifier_new.handlers.ppt.converter.olefile.OleFileIO")
    def test_full_pipeline_with_images(self, mock_oleclass):
        """Pipeline with images should extract them via image service."""
        pp_stream = _make_text_record(0x0FA8, "Slide with images")
        fake_image = b"\x89PNG" + b"\x00" * 50
        pictures = _make_image_record(fake_image, rec_type=0xF01E)

        mock_ole = _mock_ole(pp_stream=pp_stream, pictures_stream=pictures)
        mock_oleclass.return_value = mock_ole

        img_svc = MagicMock()
        img_svc.save_and_tag.return_value = "[IMG:test]"

        config = ProcessingConfig()
        handler = PPTHandler(config, image_service=img_svc)

        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = handler.process(ctx)

        assert "Slide with images" in result.text


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_converter_rejects_non_ole2(self):
        c = PptConverter()
        ctx = _make_ctx(b"This is plain text, not OLE2")
        with pytest.raises(ConversionError):
            c.convert(ctx)

    def test_preprocessor_handles_corrupt_pictures(self):
        """Corrupt pictures stream should not crash."""
        ole = _mock_ole(pp_stream=b"\x00" * 16, pictures_stream=b"\xFF" * 5)
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        p = PptPreprocessor()
        result = p.preprocess(cd)
        # Should not crash, just empty images
        assert isinstance(result, PreprocessedData)

    def test_content_extractor_garbled_stream(self):
        """Garbled stream data should not crash extraction."""
        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": b"\xFF\xFE\xFD" * 100, "image_streams": []},
            properties={},
        )
        # May find some garbage text or nothing, but should not crash
        result = ext.extract_text(ppd)
        assert isinstance(result, str)

    def test_pictures_stream_with_multiple_images(self):
        """Multiple images in the Pictures stream."""
        img1 = b"\x89PNG" + b"\x01" * 40
        img2 = b"\xFF\xD8\xFF" + b"\x02" * 40  # JPEG-like
        pictures = (
            _make_image_record(img1, rec_type=0xF01E)
            + _make_image_record(img2, rec_type=0xF01D)
        )
        ole = _mock_ole(pp_stream=b"\x00" * 16, pictures_stream=pictures)
        cd = PptConvertedData(ole=ole, file_extension="ppt")

        p = PptPreprocessor()
        result = p.preprocess(cd)
        assert result.properties["image_count"] >= 2

    def test_extract_text_very_long_stream(self):
        """Large number of records should still work."""
        records = [_make_text_record(0x0FA8, f"Record {i}") for i in range(50)]
        stream = b"".join(records)

        ext = PptContentExtractor()
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": stream, "image_streams": []},
            properties={},
        )
        result = ext.extract_text(ppd)
        assert "Record 0" in result
        assert "Record 49" in result

    def test_empty_image_data_skipped(self):
        """Empty bytes in image_streams should be skipped."""
        img_svc = MagicMock()
        img_svc.save_and_tag.return_value = "[IMG:test]"

        ext = PptContentExtractor(image_service=img_svc)
        ppd = PreprocessedData(
            content=None,
            raw_content=None,
            encoding="utf-8",
            resources={"pp_stream": None, "image_streams": [b"", b"", b""]},
            properties={},
        )
        result = ext.extract_images(ppd)
        assert result == []
        img_svc.save_and_tag.assert_not_called()

    def test_ppt_converted_data_namedtuple(self):
        """PptConvertedData should be a proper NamedTuple."""
        ole = MagicMock()
        cd = PptConvertedData(ole=ole, file_extension="ppt")
        assert cd.ole is ole
        assert cd.file_extension == "ppt"
        assert cd[0] is ole
        assert cd[1] == "ppt"

    def test_text_record_with_whitespace_only(self):
        """Records with only whitespace should be skipped."""
        stream = _make_text_record(0x0FA8, "   \t  ")
        records = _parse_text_records(stream)
        assert records == []

    @patch("contextifier_new.handlers.ppt.converter.olefile.OleFileIO")
    def test_full_pipeline_multiple_text_records(self, mock_oleclass):
        """Pipeline with multiple text records."""
        stream = b"".join([
            _make_text_record(0x0FA8, "Title"),
            _make_text_record(0x0FA8, "Subtitle"),
            _make_text_record(0x0FA0, "Body text", encoding="cp1252"),
        ])
        mock_ole = _mock_ole(pp_stream=stream)
        mock_oleclass.return_value = mock_ole

        config = ProcessingConfig()
        handler = PPTHandler(config)
        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = handler.process(ctx)

        assert "Title" in result.text
        assert "Subtitle" in result.text
        assert "Body text" in result.text
