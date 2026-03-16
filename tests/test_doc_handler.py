# tests/test_doc_handler.py
"""
Comprehensive E2E tests for the DOC (OLE2) handler pipeline.

Tests cover all 6 modules:
  _constants, converter, preprocessor, metadata_extractor,
  content_extractor, handler

Because genuine OLE2 DOC files are binary and complex, many tests use
``olefile`` to build in-memory OLE2 containers or mock the OLE interface.
For the heuristic text extractor we craft raw byte sequences that simulate
WordDocument stream content.
"""

from __future__ import annotations

import io
import struct
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import olefile

# ── Module imports ────────────────────────────────────────────────────────

from contextifier_new.handlers.doc._constants import (
    OLE2_MAGIC,
    WORD_FIB_MAGIC,
    WORD_DOCUMENT_STREAM,
    TABLE_STREAM_NAMES,
    IMAGE_STREAM_KEYWORDS,
    IMAGE_SIGNATURES,
    OLE_STRING_ENCODINGS,
    MIN_TEXT_FRAGMENT_LENGTH,
    MIN_UNICODE_BYTES,
    CJK_HIGH_BYTE_RANGES,
)
from contextifier_new.handlers.doc.converter import DocConverter, DocConvertedData
from contextifier_new.handlers.doc.preprocessor import (
    DocPreprocessor,
    DocStreamData,
)
from contextifier_new.handlers.doc.metadata_extractor import DocMetadataExtractor
from contextifier_new.handlers.doc.content_extractor import DocContentExtractor
from contextifier_new.handlers.doc.handler import DOCHandler

from contextifier_new.types import (
    DocumentMetadata,
    ExtractionResult,
    FileContext,
    PreprocessedData,
    TableData,
)
from contextifier_new.config import ProcessingConfig
from contextifier_new.errors import ConversionError, PreprocessingError


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_ctx(
    data: bytes = b"",
    ext: str = "doc",
    name: str = "test.doc",
) -> FileContext:
    """Build a minimal FileContext dict."""
    return {
        "file_path": f"/tmp/{name}",
        "file_name": name,
        "file_extension": ext,
        "file_category": "document",
        "file_data": data,
        "file_stream": io.BytesIO(data),
        "file_size": len(data),
    }


def _make_word_stream(text: str = "Hello World") -> bytes:
    """
    Build a minimal WordDocument stream with a valid FIB header
    and UTF-16LE text payload.

    The first 2 bytes are the FIB magic (0xA5EC), followed by
    padding, then the text encoded as UTF-16LE.
    """
    fib_header = struct.pack("<H", 0xA5EC)
    padding = b"\x00" * 10  # 10 bytes padding to reach >= 12 bytes
    text_bytes = text.encode("utf-16-le")
    return fib_header + padding + text_bytes


def _make_ole_bytes_with_word_stream(text: str = "Hello World") -> bytes:
    """
    Create a real OLE2 file in memory with a WordDocument stream.

    Uses the olefile writer (OleFileIO) to create a valid compound file.
    """
    # olefile doesn't have a public write API for creating new files.
    # We'll create a mock OLE object instead for testing.
    # For converter tests, we need actual OLE2 bytes.
    # Use a minimal CFB approach: write the OLE2 magic + minimal structure.
    # Since building a valid CFB from scratch is complex, we'll test
    # the converter with real olefile by using olefile's write capability.
    #
    # Alternative: Use a tiny valid OLE2 file created with olefile.
    # olefile.OleFileIO has write_stream() on existing files only.
    # Instead, we'll create a real DOC-like OLE2 using the _OleDirectoryEntry
    # approach or just test with mocked objects.
    raise NotImplementedError("Use mock-based tests instead")


class MockOleMeta:
    """Mock for olefile metadata object."""

    def __init__(
        self,
        title=None,
        subject=None,
        author=None,
        keywords=None,
        comments=None,
        last_saved_by=None,
        create_time=None,
        last_saved_time=None,
        num_pages=None,
        num_words=None,
        revision_number=None,
    ):
        self.title = title
        self.subject = subject
        self.author = author
        self.keywords = keywords
        self.comments = comments
        self.last_saved_by = last_saved_by
        self.create_time = create_time
        self.last_saved_time = last_saved_time
        self.num_pages = num_pages
        self.num_words = num_words
        self.revision_number = revision_number


class MockOle:
    """Mock for olefile.OleFileIO."""

    def __init__(
        self,
        streams: Optional[Dict[str, bytes]] = None,
        metadata: Optional[MockOleMeta] = None,
        entries: Optional[List[List[str]]] = None,
    ):
        self._streams = streams or {}
        self._metadata = metadata or MockOleMeta()
        self._entries = entries or []
        self._closed = False

    def exists(self, name: str) -> bool:
        return name in self._streams

    def openstream(self, name):
        if isinstance(name, list):
            name = "/".join(name)
        if name not in self._streams:
            raise IOError(f"Stream not found: {name}")
        stream = io.BytesIO(self._streams[name])
        return stream

    def get_metadata(self):
        return self._metadata

    def listdir(self):
        return self._entries

    def close(self):
        self._closed = True


# ═══════════════════════════════════════════════════════════════════════════
# Test Classes
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Tests for _constants.py."""

    def test_ole2_magic_length(self):
        assert len(OLE2_MAGIC) == 8

    def test_ole2_magic_starts_with_d0cf(self):
        assert OLE2_MAGIC[:2] == b"\xd0\xcf"

    def test_word_fib_magic_contains_a5ec(self):
        assert 0xA5EC in WORD_FIB_MAGIC

    def test_word_fib_magic_contains_a5dc(self):
        assert 0xA5DC in WORD_FIB_MAGIC

    def test_table_stream_names(self):
        assert "1Table" in TABLE_STREAM_NAMES
        assert "0Table" in TABLE_STREAM_NAMES

    def test_image_stream_keywords(self):
        assert "pictures" in IMAGE_STREAM_KEYWORDS
        assert "data" in IMAGE_STREAM_KEYWORDS

    def test_image_signatures_have_png(self):
        assert "png" in IMAGE_SIGNATURES
        sig, min_len = IMAGE_SIGNATURES["png"]
        assert sig == b"\x89PNG\r\n\x1a\n"

    def test_image_signatures_have_jpeg(self):
        assert "jpeg" in IMAGE_SIGNATURES
        sig, _ = IMAGE_SIGNATURES["jpeg"]
        assert sig == b"\xff\xd8"

    def test_ole_string_encodings_order(self):
        assert OLE_STRING_ENCODINGS[0] == "utf-8"
        assert "cp949" in OLE_STRING_ENCODINGS

    def test_min_text_constants(self):
        assert MIN_TEXT_FRAGMENT_LENGTH == 4
        assert MIN_UNICODE_BYTES == 8

    def test_cjk_ranges_include_hangul(self):
        found = any(lo == 0xAC for lo, hi in CJK_HIGH_BYTE_RANGES)
        assert found


class TestConverter:
    """Tests for converter.py."""

    def test_validate_ole2_magic(self):
        """Valid OLE2 file passes validation."""
        conv = DocConverter()
        data = OLE2_MAGIC + b"\x00" * 100
        ctx = _make_ctx(data)
        assert conv.validate(ctx) is True

    def test_validate_non_ole2(self):
        """Non-OLE2 data fails validation."""
        conv = DocConverter()
        ctx = _make_ctx(b"{\\rtf1 hello}")
        assert conv.validate(ctx) is False

    def test_validate_empty(self):
        """Empty data fails validation."""
        conv = DocConverter()
        ctx = _make_ctx(b"")
        assert conv.validate(ctx) is False

    def test_validate_short(self):
        """Data shorter than OLE2 magic fails."""
        conv = DocConverter()
        ctx = _make_ctx(b"\xd0\xcf\x11")
        assert conv.validate(ctx) is False

    def test_get_format_name(self):
        conv = DocConverter()
        assert conv.get_format_name() == "doc"

    def test_convert_invalid_data_raises(self):
        """Non-OLE2 data should raise ConversionError."""
        conv = DocConverter()
        ctx = _make_ctx(b"not ole data at all")
        with pytest.raises(ConversionError):
            conv.convert(ctx)

    def test_convert_empty_raises(self):
        """Empty data should raise ConversionError."""
        conv = DocConverter()
        ctx = _make_ctx(b"")
        with pytest.raises(ConversionError):
            conv.convert(ctx)

    def test_close_doc_converted_data(self):
        """Close should handle DocConvertedData."""
        conv = DocConverter()
        mock_ole = MockOle()
        data = DocConvertedData(ole=mock_ole, file_extension="doc")
        conv.close(data)
        assert mock_ole._closed is True

    def test_close_raw_ole(self):
        """Close should handle raw OLE object."""
        conv = DocConverter()
        mock_ole = MockOle()
        conv.close(mock_ole)
        assert mock_ole._closed is True

    def test_close_none(self):
        """Close should not fail on None."""
        conv = DocConverter()
        conv.close(None)  # Should not raise


class TestPreprocessor:
    """Tests for preprocessor.py."""

    def test_get_format_name(self):
        pp = DocPreprocessor()
        assert pp.get_format_name() == "doc"

    def test_validate_doc_converted_data(self):
        pp = DocPreprocessor()
        data = DocConvertedData(ole=MockOle(), file_extension="doc")
        assert pp.validate(data) is True

    def test_validate_raw_ole(self):
        pp = DocPreprocessor()
        assert pp.validate(MockOle()) is True

    def test_validate_none(self):
        pp = DocPreprocessor()
        assert pp.validate(None) is False

    def test_preprocess_reads_word_stream(self):
        """Preprocessor should read and validate WordDocument stream."""
        word_data = _make_word_stream("Test content")
        ole = MockOle(
            streams={"WordDocument": word_data},
            entries=[],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        result = pp.preprocess(converted)

        assert isinstance(result, PreprocessedData)
        assert isinstance(result.content, DocStreamData)
        assert result.content.word_data == word_data
        assert result.content.fib_magic == 0xA5EC

    def test_preprocess_finds_table_stream(self):
        """Should detect 1Table stream if present."""
        word_data = _make_word_stream("Test")
        ole = MockOle(
            streams={
                "WordDocument": word_data,
                "1Table": b"\x00" * 50,
            },
            entries=[],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        result = pp.preprocess(converted)

        assert result.content.table_stream is not None
        assert result.content.table_stream_name == "1Table"

    def test_preprocess_no_table_stream(self):
        """Should handle missing table stream gracefully."""
        word_data = _make_word_stream("Test")
        ole = MockOle(
            streams={"WordDocument": word_data},
            entries=[],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        result = pp.preprocess(converted)

        assert result.content.table_stream is None
        assert result.content.table_stream_name is None

    def test_preprocess_detects_image_streams(self):
        """Should list OLE entries matching image keywords."""
        word_data = _make_word_stream("Test")
        ole = MockOle(
            streams={
                "WordDocument": word_data,
                "Pictures/image1": b"\xff\xd8" + b"\x00" * 100,
            },
            entries=[
                ["Pictures", "image1"],
                ["Data", "somedata"],
            ],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        result = pp.preprocess(converted)

        # "Pictures" and "Data" both match IMAGE_STREAM_KEYWORDS
        assert len(result.resources["image_streams"]) == 2

    def test_preprocess_missing_word_stream_raises(self):
        """Should raise PreprocessingError if WordDocument is missing."""
        ole = MockOle(streams={}, entries=[])
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        with pytest.raises(PreprocessingError):
            pp.preprocess(converted)

    def test_preprocess_short_word_stream_raises(self):
        """Should raise if WordDocument stream is too short."""
        ole = MockOle(
            streams={"WordDocument": b"\xa5\xec\x00"},  # Only 3 bytes
            entries=[],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        with pytest.raises(PreprocessingError):
            pp.preprocess(converted)

    def test_preprocess_properties_include_extension(self):
        """Properties should include file_extension."""
        word_data = _make_word_stream("Test")
        ole = MockOle(streams={"WordDocument": word_data}, entries=[])
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        result = pp.preprocess(converted)

        assert result.properties["file_extension"] == "doc"

    def test_preprocess_invalid_fib_warns(self):
        """Invalid FIB magic should not crash, just warn."""
        # Build a stream with wrong magic but enough bytes
        bad_magic = struct.pack("<H", 0xBEEF) + b"\x00" * 20
        ole = MockOle(streams={"WordDocument": bad_magic}, entries=[])
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        result = pp.preprocess(converted)
        # Should still work
        assert result.content.fib_magic == 0xBEEF


class TestMetadataExtractor:
    """Tests for metadata_extractor.py."""

    def test_get_format_name(self):
        ext = DocMetadataExtractor()
        assert ext.get_format_name() == "doc"

    def test_extract_empty_metadata(self):
        """Empty metadata should return DocumentMetadata with all None."""
        ext = DocMetadataExtractor()
        ole = MockOle(metadata=MockOleMeta())
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert isinstance(result, DocumentMetadata)
        assert result.is_empty()

    def test_extract_title_and_author(self):
        ext = DocMetadataExtractor()
        meta = MockOleMeta(title="My Document", author="John Doe")
        ole = MockOle(metadata=meta)
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert result.title == "My Document"
        assert result.author == "John Doe"

    def test_extract_all_string_fields(self):
        ext = DocMetadataExtractor()
        meta = MockOleMeta(
            title="Title",
            subject="Subject",
            author="Author",
            keywords="key1, key2",
            comments="A comment",
            last_saved_by="Editor",
        )
        ole = MockOle(metadata=meta)
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert result.title == "Title"
        assert result.subject == "Subject"
        assert result.author == "Author"
        assert result.keywords == "key1, key2"
        assert result.comments == "A comment"
        assert result.last_saved_by == "Editor"

    def test_extract_dates(self):
        now = datetime(2024, 1, 15, 10, 30, 0)
        ext = DocMetadataExtractor()
        meta = MockOleMeta(create_time=now, last_saved_time=now)
        ole = MockOle(metadata=meta)
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert result.create_time == now
        assert result.last_saved_time == now

    def test_extract_page_word_counts(self):
        ext = DocMetadataExtractor()
        meta = MockOleMeta(num_pages=5, num_words=1000)
        ole = MockOle(metadata=meta)
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert result.page_count == 5
        assert result.word_count == 1000

    def test_extract_revision(self):
        ext = DocMetadataExtractor()
        meta = MockOleMeta(revision_number="3")
        ole = MockOle(metadata=meta)
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert result.revision == "3"

    def test_extract_bytes_title_utf8(self):
        """Bytes title should be decoded."""
        ext = DocMetadataExtractor()
        meta = MockOleMeta(title=b"My Document")
        ole = MockOle(metadata=meta)
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert result.title == "My Document"

    def test_extract_bytes_title_cp949(self):
        """Korean bytes title should be decoded with cp949."""
        ext = DocMetadataExtractor()
        korean_title = "한글 문서"
        meta = MockOleMeta(title=korean_title.encode("cp949"))
        ole = MockOle(metadata=meta)
        result = ext.extract(DocConvertedData(ole=ole, file_extension="doc"))
        assert result.title == korean_title

    def test_extract_from_preprocessed_data(self):
        """Should accept PreprocessedData as input."""
        ext = DocMetadataExtractor()
        meta = MockOleMeta(title="From Preprocessed")
        ole = MockOle(metadata=meta)
        converted = DocConvertedData(ole=ole, file_extension="doc")
        preprocessed = PreprocessedData(
            content=None,
            raw_content=converted,
        )
        result = ext.extract(preprocessed)
        assert result.title == "From Preprocessed"

    def test_extract_from_raw_ole(self):
        """Should accept raw OLE object."""
        ext = DocMetadataExtractor()
        meta = MockOleMeta(title="Direct OLE")
        ole = MockOle(metadata=meta)
        result = ext.extract(ole)
        assert result.title == "Direct OLE"

    def test_extract_none_input(self):
        """None input should return empty metadata."""
        ext = DocMetadataExtractor()
        result = ext.extract(None)
        assert result.is_empty()

    def test_extract_metadata_exception_returns_empty(self):
        """If get_metadata() raises, return empty metadata."""
        ext = DocMetadataExtractor()

        class BadOle:
            def get_metadata(self):
                raise RuntimeError("broken")

        result = ext.extract(BadOle())
        assert result.is_empty()


class TestContentExtractor:
    """Tests for content_extractor.py."""

    def test_get_format_name(self):
        ext = DocContentExtractor()
        assert ext.get_format_name() == "doc"

    def test_extract_text_simple_ascii(self):
        """Should extract simple ASCII text from WordDocument stream."""
        text = "Hello World from DOC"
        word_data = _make_word_stream(text)
        stream_data = DocStreamData(
            word_data=word_data,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        assert "Hello World from DOC" in result

    def test_extract_text_korean(self):
        """Should extract Korean UTF-16LE text."""
        text = "안녕하세요 한글 문서입니다"
        word_data = _make_word_stream(text)
        stream_data = DocStreamData(
            word_data=word_data,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        assert "안녕하세요" in result

    def test_extract_text_empty_stream(self):
        """Should return empty string for no content."""
        stream_data = DocStreamData(
            word_data=struct.pack("<H", 0xA5EC) + b"\x00" * 10,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        assert result == ""

    def test_extract_text_none_content(self):
        """Should handle missing stream data gracefully."""
        preprocessed = PreprocessedData(content=None, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        assert result == ""

    def test_extract_tables_returns_empty(self):
        """DOC tables are not supported — should return empty list."""
        stream_data = DocStreamData(
            word_data=_make_word_stream("Test"),
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_tables(preprocessed)
        assert result == []

    def test_extract_images_no_service(self):
        """Without image service, should return empty list."""
        stream_data = DocStreamData(
            word_data=_make_word_stream("Test"),
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=["Pictures/img1"],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(
            content=stream_data,
            raw_content=None,
            resources={"image_streams": ["Pictures/img1"]},
        )

        ext = DocContentExtractor()  # No image_service
        result = ext.extract_images(preprocessed)
        assert result == []

    def test_extract_images_with_service(self):
        """With image service, should detect and save JPEG images."""
        jpeg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 200

        mock_image_service = MagicMock()
        mock_image_service.save_and_tag.return_value = "[Image: doc_ole_abc.jpeg]"

        ole = MockOle(
            streams={
                "WordDocument": _make_word_stream("Test"),
                "Pictures/img1": jpeg_data,
            },
            entries=[["Pictures", "img1"]],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")
        stream_data = DocStreamData(
            word_data=_make_word_stream("Test"),
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=["Pictures/img1"],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(
            content=stream_data,
            raw_content=converted,
            resources={"image_streams": ["Pictures/img1"]},
        )

        ext = DocContentExtractor(image_service=mock_image_service)
        result = ext.extract_images(preprocessed)
        assert len(result) == 1
        assert "[Image:" in result[0]

    def test_detect_image_format_png(self):
        """Should detect PNG format."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        fmt = DocContentExtractor._detect_image_format(data)
        assert fmt == "png"

    def test_detect_image_format_jpeg(self):
        """Should detect JPEG format."""
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        fmt = DocContentExtractor._detect_image_format(data)
        assert fmt == "jpeg"

    def test_detect_image_format_gif(self):
        """Should detect GIF format."""
        data = b"GIF89a" + b"\x00" * 100
        fmt = DocContentExtractor._detect_image_format(data)
        assert fmt == "gif"

    def test_detect_image_format_unknown(self):
        """Unknown data should return None."""
        fmt = DocContentExtractor._detect_image_format(b"\x00\x01\x02\x03")
        assert fmt is None

    def test_detect_image_format_empty(self):
        """Empty data should return None."""
        fmt = DocContentExtractor._detect_image_format(b"")
        assert fmt is None

    def test_extract_text_deduplicates(self):
        """Duplicate text fragments should be removed."""
        text = "Duplicate text here"
        text_bytes = text.encode("utf-16-le")
        # Repeat the same text in the WordDocument stream with gaps
        word_data = (
            struct.pack("<H", 0xA5EC) + b"\x00" * 10
            + text_bytes
            + b"\xff" * 20  # noise separator
            + text_bytes    # duplicate
        )
        stream_data = DocStreamData(
            word_data=word_data,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        # Text should appear only once
        assert result.count("Duplicate text here") == 1


class TestExtractAll:
    """Tests for extract_all() integration."""

    def test_extract_all_text_only(self):
        """extract_all should return ExtractionResult with text."""
        text = "Integration test document"
        word_data = _make_word_stream(text)
        stream_data = DocStreamData(
            word_data=word_data,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_all(preprocessed)

        assert isinstance(result, ExtractionResult)
        assert "Integration test document" in result.text
        assert result.tables == []
        assert result.images == []
        assert result.charts == []

    def test_extract_all_with_images(self):
        """extract_all should include image tags when service is available."""
        jpeg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 200
        mock_image_service = MagicMock()
        mock_image_service.save_and_tag.return_value = "[Image: test.jpeg]"

        ole = MockOle(
            streams={
                "WordDocument": _make_word_stream("Test"),
                "Pictures/img1": jpeg_data,
            },
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")
        stream_data = DocStreamData(
            word_data=_make_word_stream("Test text content"),
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=["Pictures/img1"],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(
            content=stream_data,
            raw_content=converted,
            resources={"image_streams": ["Pictures/img1"]},
        )

        ext = DocContentExtractor(image_service=mock_image_service)
        result = ext.extract_all(preprocessed)

        assert isinstance(result, ExtractionResult)
        assert len(result.images) >= 1


class TestFullPipeline:
    """Full pipeline integration tests using mocked OLE objects."""

    def test_converter_to_preprocessor(self):
        """Converter output feeds into Preprocessor correctly."""
        word_data = _make_word_stream("Pipeline test")
        ole = MockOle(
            streams={"WordDocument": word_data},
            entries=[],
        )

        # Simulate converter output
        converted = DocConvertedData(ole=ole, file_extension="doc")

        # Feed into preprocessor
        pp = DocPreprocessor()
        result = pp.preprocess(converted)

        assert isinstance(result.content, DocStreamData)
        assert result.content.word_data == word_data

    def test_preprocessor_to_metadata(self):
        """Preprocessor output feeds into MetadataExtractor correctly."""
        word_data = _make_word_stream("Metadata test")
        meta = MockOleMeta(title="Test Title", author="Test Author")
        ole = MockOle(
            streams={"WordDocument": word_data},
            metadata=meta,
            entries=[],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        preprocessed = pp.preprocess(converted)

        ext = DocMetadataExtractor()
        metadata = ext.extract(preprocessed)

        assert metadata.title == "Test Title"
        assert metadata.author == "Test Author"

    def test_preprocessor_to_content(self):
        """Preprocessor output feeds into ContentExtractor correctly."""
        word_data = _make_word_stream("Content extraction test ABC123")
        ole = MockOle(
            streams={"WordDocument": word_data},
            entries=[],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        pp = DocPreprocessor()
        preprocessed = pp.preprocess(converted)

        ext = DocContentExtractor()
        text = ext.extract_text(preprocessed)

        assert "Content extraction test" in text

    def test_full_pipeline(self):
        """Full Convert → Preprocess → Metadata → Content pipeline."""
        word_data = _make_word_stream("Full pipeline document text")
        meta = MockOleMeta(
            title="Full Pipeline",
            author="Test Author",
            create_time=datetime(2024, 6, 1),
        )
        ole = MockOle(
            streams={"WordDocument": word_data},
            metadata=meta,
            entries=[],
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")

        # Stage 2: Preprocess
        pp = DocPreprocessor()
        preprocessed = pp.preprocess(converted)

        # Stage 3: Metadata
        meta_ext = DocMetadataExtractor()
        metadata = meta_ext.extract(preprocessed)

        assert metadata.title == "Full Pipeline"
        assert metadata.create_time == datetime(2024, 6, 1)

        # Stage 4: Content
        content_ext = DocContentExtractor()
        result = content_ext.extract_all(
            preprocessed,
            extract_metadata_result=metadata,
        )

        assert isinstance(result, ExtractionResult)
        assert "Full pipeline document text" in result.text
        assert result.metadata is not None
        assert result.metadata.title == "Full Pipeline"


class TestDelegation:
    """Tests for the DOCHandler delegation logic."""

    def test_check_delegation_rtf(self):
        """RTF content should trigger delegation to RTF handler."""
        config = ProcessingConfig()
        handler = DOCHandler(config)

        # Mock registry
        mock_result = ExtractionResult(text="RTF text")
        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.process.return_value = mock_result
        mock_registry.get_handler.return_value = mock_handler
        handler._handler_registry = mock_registry

        ctx = _make_ctx(b"{\\rtf1 test content  " + b"\x00" * 50)
        result = handler._check_delegation(ctx)

        assert result is not None

    def test_check_delegation_zip(self):
        """ZIP content should trigger delegation to DOCX handler."""
        config = ProcessingConfig()
        handler = DOCHandler(config)

        mock_registry = MagicMock()
        mock_handler = MagicMock()
        mock_handler.process.return_value = ExtractionResult(text="DOCX text")
        mock_registry.get_handler.return_value = mock_handler
        handler._handler_registry = mock_registry

        ctx = _make_ctx(b"PK\x03\x04" + b"\x00" * 100)
        result = handler._check_delegation(ctx)

        assert result is not None

    def test_check_delegation_ole2_returns_none(self):
        """Genuine OLE2 should NOT delegate (returns None)."""
        config = ProcessingConfig()
        handler = DOCHandler(config)

        ctx = _make_ctx(OLE2_MAGIC + b"\x00" * 100)
        result = handler._check_delegation(ctx)

        assert result is None

    def test_check_delegation_empty_returns_none(self):
        """Empty data should NOT delegate."""
        config = ProcessingConfig()
        handler = DOCHandler(config)

        ctx = _make_ctx(b"")
        result = handler._check_delegation(ctx)

        assert result is None

    def test_check_delegation_short_returns_none(self):
        """Data shorter than 8 bytes should NOT delegate."""
        config = ProcessingConfig()
        handler = DOCHandler(config)

        ctx = _make_ctx(b"\x01\x02")
        result = handler._check_delegation(ctx)

        assert result is None


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_text_extraction_with_noise_bytes(self):
        """Noise between text runs should be skipped."""
        text1 = "First fragment here"
        text2 = "Second fragment here"
        noise = bytes(range(0x80, 0xC0)) * 2  # Non-text high bytes

        word_data = (
            struct.pack("<H", 0xA5EC) + b"\x00" * 10
            + text1.encode("utf-16-le")
            + noise
            + text2.encode("utf-16-le")
        )
        stream_data = DocStreamData(
            word_data=word_data,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        assert "First fragment here" in result
        assert "Second fragment here" in result

    def test_text_fragment_too_short_ignored(self):
        """Fragments shorter than MIN_TEXT_FRAGMENT_LENGTH are skipped."""
        # "Hi" is only 2 chars — below threshold
        short_frag = "Hi".encode("utf-16-le")
        # "This is long enough" is above threshold
        long_frag = "This is long enough".encode("utf-16-le")

        word_data = (
            struct.pack("<H", 0xA5EC) + b"\x00" * 10
            + short_frag
            + b"\xff" * 10
            + long_frag
        )
        stream_data = DocStreamData(
            word_data=word_data,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        assert "Hi" not in result.split("\n")  # Short fragment excluded
        assert "This is long enough" in result

    def test_image_deduplication(self):
        """Same image data in two streams should only produce one tag."""
        jpeg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 200

        mock_image_service = MagicMock()
        mock_image_service.save_and_tag.return_value = "[Image: dedup.jpeg]"

        ole = MockOle(
            streams={
                "WordDocument": _make_word_stream("Test"),
                "Pictures/img1": jpeg_data,
                "Data/img2": jpeg_data,  # Same content
            },
        )
        converted = DocConvertedData(ole=ole, file_extension="doc")
        stream_data = DocStreamData(
            word_data=_make_word_stream("Test"),
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=["Pictures/img1", "Data/img2"],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(
            content=stream_data,
            raw_content=converted,
            resources={"image_streams": ["Pictures/img1", "Data/img2"]},
        )

        ext = DocContentExtractor(image_service=mock_image_service)
        result = ext.extract_images(preprocessed)
        # Should only save once due to hash deduplication
        assert mock_image_service.save_and_tag.call_count == 1

    def test_ole_metadata_bytes_decode_fallback(self):
        """Bytes that aren't valid UTF-8 should fall through to cp949."""
        ext = DocMetadataExtractor()
        result = ext._decode_ole_value(b"\xc7\xd1\xb1\xdb")  # "한글" in cp949
        assert result == "한글"

    def test_ole_metadata_none_value(self):
        """None values should return empty string."""
        ext = DocMetadataExtractor()
        result = ext._decode_ole_value(None)
        assert result == ""

    def test_handler_supported_extensions(self):
        """DOCHandler should only support 'doc'."""
        config = ProcessingConfig()
        handler = DOCHandler(config)
        assert handler.supported_extensions == frozenset({"doc"})

    def test_handler_name(self):
        config = ProcessingConfig()
        handler = DOCHandler(config)
        assert handler.handler_name == "DOC Handler"

    def test_mixed_cjk_and_ascii(self):
        """Text with mixed Korean and ASCII should be extracted."""
        text = "Hello 안녕하세요 World"
        word_data = _make_word_stream(text)
        stream_data = DocStreamData(
            word_data=word_data,
            table_stream=None,
            table_stream_name=None,
            fib_magic=0xA5EC,
            image_streams=[],
            encoding="utf-16-le",
        )
        preprocessed = PreprocessedData(content=stream_data, raw_content=None)

        ext = DocContentExtractor()
        result = ext.extract_text(preprocessed)
        # The mixed text should be captured in one or more fragments
        assert "Hello" in result or "안녕하세요" in result
