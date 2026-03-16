# tests/test_hwp_handler.py
"""
Comprehensive tests for the HWP handler pipeline.

Covers:
- _constants: tag IDs, magic bytes, control characters
- _record: HwpRecord parsing and tree building
- _decoder: compression detection and decompression
- _docinfo: DocInfo BinData parsing
- _table: table grid building and HTML rendering
- _recovery: raw text extraction, zlib scanning, signature detection
- converter: OLE validation and opening
- preprocessor: unwrap + DocInfo pre-parsing
- metadata_extractor: OLE + HwpSummary metadata extraction
- content_extractor: section traversal, text/table/image assembly
- handler: delegation and full pipeline integration
"""

from __future__ import annotations

import io
import struct
import zlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import (
    DocumentMetadata,
    ExtractionResult,
    FileContext,
    PreprocessedData,
    TableData,
)

# ── Module imports ──────────────────────────────────────────────────────
from contextifier_new.handlers.hwp._constants import (
    BINDATA_EMBEDDING,
    BINDATA_LINK,
    COMPRESS_FLAG,
    CTRL_CHAR_DRAWING_TABLE_OBJECT,
    CTRL_CHAR_LINE_BREAK,
    CTRL_CHAR_PARA_BREAK,
    CTRL_CHAR_TAB,
    EXTENDED_CHAR_UNITS,
    FILE_HEADER_FLAGS_OFFSET,
    HWPTAG_BEGIN,
    HWPTAG_BIN_DATA,
    HWPTAG_CTRL_HEADER,
    HWPTAG_LIST_HEADER,
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_SHAPE_COMPONENT_PICTURE,
    HWPTAG_TABLE,
    OLE2_MAGIC,
    STREAM_BODY_TEXT,
    STREAM_DOC_INFO,
    STREAM_FILE_HEADER,
    STREAM_HWP_SUMMARY,
    ZIP_MAGIC,
)
from contextifier_new.handlers.hwp._record import HwpRecord
from contextifier_new.handlers.hwp._decoder import (
    decompress_section,
    decompress_stream,
    is_compressed,
)
from contextifier_new.handlers.hwp._docinfo import parse_doc_info, scan_bindata_folder
from contextifier_new.handlers.hwp._table import parse_table, render_table_html
from contextifier_new.handlers.hwp._recovery import (
    check_file_signature,
    extract_text_raw,
    find_zlib_streams,
)
from contextifier_new.handlers.hwp.converter import HwpConverter, HwpConvertedData
from contextifier_new.handlers.hwp.preprocessor import HwpPreprocessor
from contextifier_new.handlers.hwp.metadata_extractor import HwpMetadataExtractor
from contextifier_new.handlers.hwp.content_extractor import HwpContentExtractor
from contextifier_new.handlers.hwp.handler import HWPHandler


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_record_bytes(tag_id: int, level: int, payload: bytes) -> bytes:
    """Build a single HWP record as raw bytes."""
    size = len(payload)
    if size >= 0xFFF:
        header = (tag_id & 0x3FF) | ((level & 0x3FF) << 10) | (0xFFF << 20)
        return struct.pack("<I", header) + struct.pack("<I", size) + payload
    header = (tag_id & 0x3FF) | ((level & 0x3FF) << 10) | ((size & 0xFFF) << 20)
    return struct.pack("<I", header) + payload


def _make_para_text_payload(text: str) -> bytes:
    """Encode text as UTF-16LE HWP PARA_TEXT payload."""
    return text.encode("utf-16le")


def _make_file_context(data: bytes = b"", ext: str = "hwp") -> FileContext:
    return FileContext(
        file_path=f"/test/sample.{ext}",
        file_name=f"sample.{ext}",
        file_extension=ext,
        file_category="document",
        file_data=data,
        file_stream=io.BytesIO(data),
        file_size=len(data),
    )


def _mock_ole(**kwargs) -> MagicMock:
    """Create a mock olefile.OleFileIO."""
    ole = MagicMock()
    ole.listdir.return_value = kwargs.get("listdir", [])
    ole.exists = kwargs.get("exists", lambda x: False)

    def mock_openstream(path):
        streams = kwargs.get("streams", {})
        key = path if isinstance(path, str) else "/".join(path)
        s = MagicMock()
        s.read.return_value = streams.get(key, b"")
        return s

    ole.openstream = mock_openstream
    return ole


# ═══════════════════════════════════════════════════════════════════════════
# 1. Constants Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_ole2_magic(self):
        assert OLE2_MAGIC == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        assert len(OLE2_MAGIC) == 8

    def test_zip_magic(self):
        assert ZIP_MAGIC == b"PK\x03\x04"

    def test_tag_ids(self):
        assert HWPTAG_BEGIN == 0x10
        assert HWPTAG_BIN_DATA == 18
        assert HWPTAG_PARA_HEADER == 66
        assert HWPTAG_PARA_TEXT == 67
        assert HWPTAG_CTRL_HEADER == 71
        assert HWPTAG_LIST_HEADER == 72
        assert HWPTAG_TABLE == 77
        assert HWPTAG_SHAPE_COMPONENT_PICTURE == 85

    def test_control_chars(self):
        assert CTRL_CHAR_TAB == 0x09
        assert CTRL_CHAR_LINE_BREAK == 0x0A
        assert CTRL_CHAR_DRAWING_TABLE_OBJECT == 0x0B
        assert CTRL_CHAR_PARA_BREAK == 0x0D

    def test_extended_char_units(self):
        assert EXTENDED_CHAR_UNITS == 8

    def test_stream_names(self):
        assert STREAM_FILE_HEADER == "FileHeader"
        assert STREAM_DOC_INFO == "DocInfo"
        assert STREAM_BODY_TEXT == "BodyText"
        assert STREAM_HWP_SUMMARY == "\x05HwpSummaryInformation"


# ═══════════════════════════════════════════════════════════════════════════
# 2. HwpRecord Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHwpRecord:
    def test_basic_creation(self):
        rec = HwpRecord(tag_id=67, level=1, payload=b"\x00\x01")
        assert rec.tag_id == 67
        assert rec.level == 1
        assert rec.payload == b"\x00\x01"
        assert rec.children == []
        assert rec.parent is None

    def test_build_tree_empty(self):
        root = HwpRecord.build_tree(b"")
        assert root.tag_id == 0
        assert root.level == -1
        assert root.children == []

    def test_build_tree_single_record(self):
        payload = b"Hello"
        data = _make_record_bytes(HWPTAG_PARA_TEXT, 0, payload)
        root = HwpRecord.build_tree(data)
        assert len(root.children) == 1
        assert root.children[0].tag_id == HWPTAG_PARA_TEXT
        assert root.children[0].payload == payload

    def test_build_tree_nested(self):
        """Level 0 parent, level 1 child."""
        parent_data = _make_record_bytes(HWPTAG_PARA_HEADER, 0, b"\x00" * 4)
        child_data = _make_record_bytes(HWPTAG_PARA_TEXT, 1, b"A\x00B\x00")
        root = HwpRecord.build_tree(parent_data + child_data)
        assert len(root.children) == 1
        parent = root.children[0]
        assert parent.tag_id == HWPTAG_PARA_HEADER
        assert len(parent.children) == 1
        assert parent.children[0].tag_id == HWPTAG_PARA_TEXT

    def test_build_tree_siblings(self):
        """Two records at the same level."""
        r1 = _make_record_bytes(HWPTAG_PARA_HEADER, 0, b"\x00" * 4)
        r2 = _make_record_bytes(HWPTAG_PARA_HEADER, 0, b"\x01" * 4)
        root = HwpRecord.build_tree(r1 + r2)
        assert len(root.children) == 2

    def test_get_text_ascii(self):
        text = "Hello"
        rec = HwpRecord(tag_id=HWPTAG_PARA_TEXT, payload=text.encode("utf-16le"))
        assert rec.get_text() == "Hello"

    def test_get_text_korean(self):
        text = "안녕하세요"
        rec = HwpRecord(tag_id=HWPTAG_PARA_TEXT, payload=text.encode("utf-16le"))
        assert rec.get_text() == "안녕하세요"

    def test_get_text_control_chars(self):
        """Tab, line break, paragraph break are converted."""
        data = struct.pack("<HHH", 0x09, 0x0A, 0x0D)
        rec = HwpRecord(payload=data)
        assert rec.get_text() == "\t\n\n"

    def test_get_text_drawing_marker(self):
        """0x0B is kept as \\x0b."""
        data = struct.pack("<HHH", ord("A"), 0x0B, ord("B"))
        rec = HwpRecord(payload=data)
        assert rec.get_text() == "A\x0bB"

    def test_get_text_extended_char_skipped(self):
        """Extended control chars (code < 32, not tab/lf/cr/0b) skip 8 code units."""
        # Code 0x02 is an extended char → should skip 8*2=16 bytes total
        data = struct.pack("<H", 0x02) + b"\x00" * 14 + "X".encode("utf-16le")
        rec = HwpRecord(payload=data)
        assert rec.get_text() == "X"

    def test_get_next_siblings(self):
        root = HwpRecord(level=-1)
        c1 = HwpRecord(level=0, parent=root)
        c2 = HwpRecord(level=0, parent=root)
        c3 = HwpRecord(level=0, parent=root)
        root.children = [c1, c2, c3]
        assert c1.get_next_siblings(2) == [c2, c3]
        assert c2.get_next_siblings(1) == [c3]
        assert c3.get_next_siblings(1) == []

    def test_extended_size_record(self):
        """Record with size >= 0xFFF uses extended 4-byte size."""
        payload = b"X" * 5000
        data = _make_record_bytes(HWPTAG_PARA_TEXT, 0, payload)
        root = HwpRecord.build_tree(data)
        assert len(root.children) == 1
        assert len(root.children[0].payload) == 5000


# ═══════════════════════════════════════════════════════════════════════════
# 3. Decoder Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDecoder:
    def test_is_compressed_true(self):
        ole = _mock_ole(
            exists=lambda x: x == STREAM_FILE_HEADER,
            streams={
                STREAM_FILE_HEADER: b"\x00" * FILE_HEADER_FLAGS_OFFSET + struct.pack("<I", COMPRESS_FLAG)
            },
        )
        assert is_compressed(ole) is True

    def test_is_compressed_false(self):
        ole = _mock_ole(
            exists=lambda x: x == STREAM_FILE_HEADER,
            streams={
                STREAM_FILE_HEADER: b"\x00" * FILE_HEADER_FLAGS_OFFSET + struct.pack("<I", 0)
            },
        )
        assert is_compressed(ole) is False

    def test_is_compressed_missing_stream(self):
        ole = _mock_ole(exists=lambda x: False)
        assert is_compressed(ole) is False

    def test_decompress_stream_not_compressed(self):
        data = b"raw data"
        assert decompress_stream(data, False) == data

    def test_decompress_stream_raw_deflate(self):
        original = b"Hello World! " * 100
        compressed = zlib.compress(original)
        # Raw deflate (strip 2-byte header and 4-byte checksum)
        raw_deflate = compressed[2:-4]
        result = decompress_stream(raw_deflate, True)
        assert result == original

    def test_decompress_stream_standard_zlib(self):
        original = b"Hello World! " * 100
        compressed = zlib.compress(original)
        result = decompress_stream(compressed, True)
        assert result == original

    def test_decompress_stream_fallback(self):
        """Non-compressed data returned as-is when decompression fails."""
        data = b"not zlib at all"
        assert decompress_stream(data, True) == data

    def test_decompress_section_success(self):
        original = b"Section content " * 50
        compressed = zlib.compress(original)
        result, ok = decompress_section(compressed)
        assert ok is True
        assert result == original

    def test_decompress_section_empty(self):
        result, ok = decompress_section(b"")
        assert ok is False

    def test_decompress_section_uncompressed(self):
        """Uncompressed data returns successfully."""
        data = b"plain text"
        result, ok = decompress_section(data)
        assert ok is True
        assert result == data


# ═══════════════════════════════════════════════════════════════════════════
# 4. DocInfo Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDocInfo:
    def _make_bindata_record(self, storage_type: int, storage_id: int, ext: str = "png") -> bytes:
        """Build a minimal BIN_DATA record payload."""
        flags = storage_type & 0x0F
        ext_encoded = ext.encode("utf-16le")
        ext_len = len(ext)
        payload = struct.pack("<HHH", flags, storage_id, ext_len) + ext_encoded
        return payload

    def test_scan_bindata_folder_finds_entries(self):
        ole = _mock_ole(
            listdir=[
                ["BinData", "BIN0001.png"],
                ["BinData", "BIN0002.jpg"],
                ["BodyText", "Section0"],
            ],
        )
        by_id, ordered = scan_bindata_folder(ole)
        assert len(ordered) == 2
        assert by_id[1] == (1, "png")
        assert by_id[2] == (2, "jpg")

    def test_scan_bindata_folder_empty(self):
        ole = _mock_ole(listdir=[])
        by_id, ordered = scan_bindata_folder(ole)
        assert len(ordered) == 0

    def test_parse_doc_info_no_stream(self):
        ole = _mock_ole(exists=lambda x: False, listdir=[])
        by_id, ordered = parse_doc_info(ole)
        assert len(ordered) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Table Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTable:
    def _make_table_record(self, rows: int, cols: int) -> HwpRecord:
        """Build a minimal CTRL_HEADER → TABLE → LIST_HEADER structure."""
        ctrl = HwpRecord(tag_id=HWPTAG_CTRL_HEADER, level=0, payload=b"tbl "[::-1])

        table_payload = b"\x00" * 4 + struct.pack("<HH", rows, cols)
        table_rec = HwpRecord(tag_id=HWPTAG_TABLE, level=1, payload=table_payload, parent=ctrl)
        ctrl.children.append(table_rec)

        for r in range(rows):
            for c in range(cols):
                cell_payload = struct.pack(
                    "<HHHHHHHHH",
                    1,  # para_count
                    0, 0, 0,  # padding
                    c,  # col_idx
                    r,  # row_idx
                    1,  # col_span
                    1,  # row_span
                    0,  # padding
                )[:16]
                cell = HwpRecord(
                    tag_id=HWPTAG_LIST_HEADER, level=1,
                    payload=cell_payload, parent=ctrl,
                )
                # Add a child PARA_TEXT record for the cell
                text_payload = f"R{r}C{c}".encode("utf-16le")
                text_rec = HwpRecord(
                    tag_id=HWPTAG_PARA_TEXT, level=2,
                    payload=text_payload, parent=cell,
                )
                cell.children.append(text_rec)
                ctrl.children.append(cell)

        return ctrl

    def _traverse_stub(self, record, ole=None, bdm=None, pi=None):
        """Simple traversal stub that extracts text from PARA_TEXT."""
        if record.tag_id == HWPTAG_PARA_TEXT:
            return record.get_text()
        parts = []
        for child in record.children:
            parts.append(self._traverse_stub(child))
        return "".join(parts)

    def test_1x1_table(self):
        ctrl = self._make_table_record(1, 1)
        result = parse_table(ctrl, self._traverse_stub)
        assert "R0C0" in result

    def test_single_column_table(self):
        ctrl = self._make_table_record(3, 1)
        result = parse_table(ctrl, self._traverse_stub)
        assert "R0C0" in result
        assert "R1C0" in result
        assert "R2C0" in result

    def test_multi_column_table_html(self):
        ctrl = self._make_table_record(2, 2)
        result = parse_table(ctrl, self._traverse_stub)
        assert "<table" in result
        assert "<tr>" in result
        assert "<td>" in result
        assert "R0C0" in result
        assert "R1C1" in result

    def test_render_table_html(self):
        grid = {
            (0, 0): {"text": "A", "rowspan": 1, "colspan": 1},
            (0, 1): {"text": "B", "rowspan": 1, "colspan": 1},
            (1, 0): {"text": "C", "rowspan": 1, "colspan": 1},
            (1, 1): {"text": "D", "rowspan": 1, "colspan": 1},
        }
        html = render_table_html(grid, 2, 2)
        assert "<table" in html
        assert "A" in html and "D" in html

    def test_render_table_html_with_spans(self):
        grid = {
            (0, 0): {"text": "Merged", "rowspan": 2, "colspan": 1},
            (0, 1): {"text": "B", "rowspan": 1, "colspan": 1},
            (1, 1): {"text": "D", "rowspan": 1, "colspan": 1},
        }
        html = render_table_html(grid, 2, 2)
        assert "rowspan='2'" in html

    def test_no_table_record(self):
        ctrl = HwpRecord(tag_id=HWPTAG_CTRL_HEADER, level=0, payload=b"tbl "[::-1])
        result = parse_table(ctrl, self._traverse_stub)
        assert result == ""


# ═══════════════════════════════════════════════════════════════════════════
# 6. Recovery Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRecovery:
    def test_extract_text_raw_korean(self):
        text = "안녕하세요"
        data = text.encode("utf-16le")
        result = extract_text_raw(data)
        assert "안녕하세요" in result

    def test_extract_text_raw_ascii(self):
        text = "Hello World"
        data = text.encode("utf-16le")
        result = extract_text_raw(data)
        assert "Hello World" in result

    def test_extract_text_raw_mixed(self):
        text = "Hello 안녕"
        data = text.encode("utf-16le")
        result = extract_text_raw(data)
        assert "Hello" in result
        assert "안녕" in result

    def test_find_zlib_streams(self):
        original = b"some text content " * 20
        compressed = zlib.compress(original)
        padding = b"\x00" * 50
        data = padding + compressed + padding
        results = find_zlib_streams(data, min_size=10)
        assert len(results) >= 1
        assert results[0][1] == original

    def test_find_zlib_streams_none(self):
        data = b"\x00" * 100
        results = find_zlib_streams(data)
        assert results == []

    def test_check_file_signature_ole(self):
        assert check_file_signature(OLE2_MAGIC + b"\x00" * 100) == "OLE"

    def test_check_file_signature_zip(self):
        assert check_file_signature(ZIP_MAGIC + b"\x00" * 100) == "ZIP/HWPX"

    def test_check_file_signature_hwp3(self):
        assert check_file_signature(b"HWP Document File V3.0" + b"\x00" * 80) == "HWP3.0"

    def test_check_file_signature_unknown(self):
        assert check_file_signature(b"\x00" * 100) is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. Converter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHwpConverter:
    def test_get_format_name(self):
        conv = HwpConverter()
        assert conv.get_format_name() == "hwp"

    def test_validate_good_magic(self):
        conv = HwpConverter()
        fc = _make_file_context(OLE2_MAGIC + b"\x00" * 100)
        assert conv.validate(fc) is True

    def test_validate_bad_magic(self):
        conv = HwpConverter()
        fc = _make_file_context(b"not ole at all")
        assert conv.validate(fc) is False

    def test_validate_short_data(self):
        conv = HwpConverter()
        fc = _make_file_context(b"\xd0\xcf")
        assert conv.validate(fc) is False

    def test_validate_empty(self):
        conv = HwpConverter()
        fc = _make_file_context(b"")
        assert conv.validate(fc) is False

    @patch("contextifier_new.handlers.hwp.converter.olefile")
    def test_convert_success(self, mock_olefile):
        mock_ole = MagicMock()
        mock_olefile.OleFileIO.return_value = mock_ole

        conv = HwpConverter()
        fc = _make_file_context(OLE2_MAGIC + b"\x00" * 100)
        result = conv.convert(fc)

        assert isinstance(result, HwpConvertedData)
        assert result.ole is mock_ole
        assert result.file_data == fc["file_data"]
        mock_olefile.OleFileIO.assert_called_once()

    @patch("contextifier_new.handlers.hwp.converter.olefile")
    def test_convert_failure(self, mock_olefile):
        mock_olefile.OleFileIO.side_effect = Exception("bad file")

        conv = HwpConverter()
        fc = _make_file_context(OLE2_MAGIC + b"\x00" * 100)

        from contextifier_new.errors import ConversionError
        with pytest.raises(ConversionError):
            conv.convert(fc)

    def test_convert_empty_data(self):
        conv = HwpConverter()
        fc = _make_file_context(b"")

        from contextifier_new.errors import ConversionError
        with pytest.raises(ConversionError):
            conv.convert(fc)

    @patch("contextifier_new.handlers.hwp.converter.olefile")
    def test_close(self, mock_olefile):
        mock_ole = MagicMock()
        converted = HwpConvertedData(ole=mock_ole, file_data=b"test")

        conv = HwpConverter()
        conv.close(converted)
        mock_ole.close.assert_called_once()

    def test_close_non_converted(self):
        """close() handles arbitrary objects gracefully."""
        conv = HwpConverter()
        obj = MagicMock()
        conv.close(obj)
        obj.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Preprocessor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHwpPreprocessor:
    def test_get_format_name(self):
        pp = HwpPreprocessor()
        assert pp.get_format_name() == "hwp"

    @patch("contextifier_new.handlers.hwp.preprocessor.parse_doc_info")
    @patch("contextifier_new.handlers.hwp.preprocessor.is_compressed")
    def test_preprocess_success(self, mock_compressed, mock_docinfo):
        mock_compressed.return_value = True
        mock_docinfo.return_value = ({1: (1, "png")}, [(1, "png")])

        mock_ole = MagicMock()
        mock_ole.listdir.return_value = [
            ["BodyText", "Section0"],
            ["BodyText", "Section1"],
        ]

        converted = HwpConvertedData(ole=mock_ole, file_data=b"raw")
        pp = HwpPreprocessor()
        result = pp.preprocess(converted)

        assert isinstance(result, PreprocessedData)
        assert result.content is mock_ole
        assert result.resources["file_data"] == b"raw"
        assert result.properties["compressed"] is True
        assert result.properties["section_count"] == 2
        bdm = result.resources["bin_data_map"]
        assert bdm["by_storage_id"] == {1: (1, "png")}

    def test_preprocess_none(self):
        pp = HwpPreprocessor()
        from contextifier_new.errors import PreprocessingError
        with pytest.raises(PreprocessingError):
            pp.preprocess(None)

    @patch("contextifier_new.handlers.hwp.preprocessor.parse_doc_info")
    @patch("contextifier_new.handlers.hwp.preprocessor.is_compressed")
    def test_preprocess_bare_ole(self, mock_compressed, mock_docinfo):
        """Bare OLE object (not wrapped in HwpConvertedData)."""
        mock_compressed.return_value = False
        mock_docinfo.return_value = ({}, [])
        mock_ole = MagicMock()
        mock_ole.listdir.return_value = []

        pp = HwpPreprocessor()
        result = pp.preprocess(mock_ole)
        assert result.content is mock_ole
        assert result.resources["file_data"] == b""


# ═══════════════════════════════════════════════════════════════════════════
# 9. Metadata Extractor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHwpMetadataExtractor:
    def test_get_format_name(self):
        ext = HwpMetadataExtractor()
        assert ext.get_format_name() == "hwp"

    def test_extract_none(self):
        ext = HwpMetadataExtractor()
        result = ext.extract(None)
        assert isinstance(result, DocumentMetadata)
        assert result.is_empty()

    def test_extract_ole_metadata(self):
        """Standard OLE metadata extraction."""
        ole = MagicMock()
        ole.listdir.return_value = [["BodyText", "Section0"]]
        ole.exists.return_value = False  # no HwpSummary

        meta = MagicMock()
        meta.title = "Test Title"
        meta.subject = "Test Subject"
        meta.author = "Test Author"
        meta.keywords = "key1;key2"
        meta.comments = "A comment"
        meta.last_saved_by = "Editor"
        meta.create_time = datetime(2024, 1, 1)
        meta.last_saved_time = datetime(2024, 6, 1)
        ole.get_metadata.return_value = meta

        ext = HwpMetadataExtractor()
        result = ext.extract(ole)

        assert result.title == "Test Title"
        assert result.subject == "Test Subject"
        assert result.author == "Test Author"
        assert result.keywords == "key1;key2"
        assert result.comments == "A comment"
        assert result.last_saved_by == "Editor"
        assert result.create_time == datetime(2024, 1, 1)
        assert result.last_saved_time == datetime(2024, 6, 1)
        assert result.page_count == 1

    def test_extract_section_count(self):
        ole = MagicMock()
        ole.listdir.return_value = [
            ["BodyText", "Section0"],
            ["BodyText", "Section1"],
            ["BodyText", "Section2"],
        ]
        ole.exists.return_value = False
        ole.get_metadata.return_value = MagicMock(
            title=None, subject=None, author=None, keywords=None,
            comments=None, last_saved_by=None, create_time=None, last_saved_time=None,
        )

        ext = HwpMetadataExtractor()
        result = ext.extract(ole)
        assert result.page_count == 3

    def test_extract_handles_metadata_error(self):
        ole = MagicMock()
        ole.listdir.return_value = []
        ole.exists.return_value = False
        ole.get_metadata.side_effect = Exception("broken")

        ext = HwpMetadataExtractor()
        result = ext.extract(ole)
        assert isinstance(result, DocumentMetadata)

    def test_extract_empty_metadata(self):
        ole = MagicMock()
        ole.listdir.return_value = []
        ole.exists.return_value = False
        ole.get_metadata.return_value = MagicMock(
            title=None, subject=None, author=None, keywords=None,
            comments=None, last_saved_by=None, create_time=None, last_saved_time=None,
        )

        ext = HwpMetadataExtractor()
        result = ext.extract(ole)
        assert result.title is None
        assert result.author is None


# ═══════════════════════════════════════════════════════════════════════════
# 10. Content Extractor Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHwpContentExtractor:
    def test_get_format_name(self):
        ce = HwpContentExtractor()
        assert ce.get_format_name() == "hwp"

    def test_extract_text_empty_ole(self):
        """No sections → empty text."""
        ce = HwpContentExtractor()
        ole = MagicMock()
        ole.listdir.return_value = []
        pp = PreprocessedData(
            content=ole,
            resources={"bin_data_map": {"by_storage_id": {}, "by_index": []}},
        )
        result = ce.extract_text(pp)
        assert result == ""

    @patch("contextifier_new.handlers.hwp.content_extractor.decompress_section")
    def test_extract_text_simple_section(self, mock_decompress):
        """One section with a simple PARA_TEXT record."""
        text = "Hello HWP"
        payload = text.encode("utf-16le")
        record_data = _make_record_bytes(HWPTAG_PARA_TEXT, 0, payload)
        mock_decompress.return_value = (record_data, True)

        ole = MagicMock()
        ole.listdir.return_value = [["BodyText", "Section0"]]
        stream = MagicMock()
        stream.read.return_value = b"compressed_data"
        ole.openstream.return_value = stream

        ce = HwpContentExtractor()
        pp = PreprocessedData(
            content=ole,
            resources={"bin_data_map": {"by_storage_id": {}, "by_index": []}},
        )
        result = ce.extract_text(pp)
        assert "Hello HWP" in result

    @patch("contextifier_new.handlers.hwp.content_extractor.decompress_section")
    def test_extract_text_with_paragraph(self, mock_decompress):
        """PARA_HEADER with child PARA_TEXT."""
        text_payload = "Paragraph content".encode("utf-16le")
        header_data = _make_record_bytes(HWPTAG_PARA_HEADER, 0, b"\x00" * 4)
        text_data = _make_record_bytes(HWPTAG_PARA_TEXT, 1, text_payload)
        mock_decompress.return_value = (header_data + text_data, True)

        ole = MagicMock()
        ole.listdir.return_value = [["BodyText", "Section0"]]
        stream = MagicMock()
        stream.read.return_value = b"data"
        ole.openstream.return_value = stream

        ce = HwpContentExtractor()
        pp = PreprocessedData(
            content=ole,
            resources={"bin_data_map": {"by_storage_id": {}, "by_index": []}},
        )
        result = ce.extract_text(pp)
        assert "Paragraph content" in result

    @patch("contextifier_new.handlers.hwp.content_extractor.decompress_section")
    def test_extract_text_decompression_failure(self, mock_decompress):
        """If decompression fails, section is skipped."""
        mock_decompress.return_value = (b"", False)

        ole = MagicMock()
        ole.listdir.return_value = [["BodyText", "Section0"]]
        stream = MagicMock()
        stream.read.return_value = b"data"
        ole.openstream.return_value = stream

        ce = HwpContentExtractor()
        pp = PreprocessedData(
            content=ole,
            resources={"bin_data_map": {"by_storage_id": {}, "by_index": []}},
        )
        result = ce.extract_text(pp)
        assert result == ""

    @patch("contextifier_new.handlers.hwp.content_extractor.decompress_section")
    def test_extract_text_fallback_raw(self, mock_decompress):
        """If record parsing produces no text, fallback to raw extraction."""
        # Provide data that won't parse to valid records but has UTF-16LE text
        korean_text = "안녕하세요"
        raw_data = korean_text.encode("utf-16le")
        mock_decompress.return_value = (raw_data, True)

        ole = MagicMock()
        ole.listdir.return_value = [["BodyText", "Section0"]]
        stream = MagicMock()
        stream.read.return_value = b"compressed"
        ole.openstream.return_value = stream

        ce = HwpContentExtractor()
        pp = PreprocessedData(
            content=ole,
            resources={"bin_data_map": {"by_storage_id": {}, "by_index": []}},
        )
        result = ce.extract_text(pp)
        assert "안녕하세요" in result

    def test_extract_tables_empty(self):
        ce = HwpContentExtractor()
        pp = PreprocessedData(content=MagicMock())
        assert ce.extract_tables(pp) == []

    def test_extract_images_empty(self):
        ce = HwpContentExtractor()
        pp = PreprocessedData(content=MagicMock())
        assert ce.extract_images(pp) == []

    def test_extract_charts_empty(self):
        ce = HwpContentExtractor()
        pp = PreprocessedData(content=MagicMock())
        assert ce.extract_charts(pp) == []

    @patch("contextifier_new.handlers.hwp.content_extractor.decompress_section")
    def test_extract_text_multiple_sections(self, mock_decompress):
        """Multiple sections are extracted and joined."""
        text1 = "Section One".encode("utf-16le")
        text2 = "Section Two".encode("utf-16le")
        data1 = _make_record_bytes(HWPTAG_PARA_TEXT, 0, text1)
        data2 = _make_record_bytes(HWPTAG_PARA_TEXT, 0, text2)

        call_count = [0]
        def side_effect(data):
            call_count[0] += 1
            if call_count[0] == 1:
                return (data1, True)
            return (data2, True)

        mock_decompress.side_effect = side_effect

        ole = MagicMock()
        ole.listdir.return_value = [
            ["BodyText", "Section0"],
            ["BodyText", "Section1"],
        ]
        stream = MagicMock()
        stream.read.return_value = b"data"
        ole.openstream.return_value = stream

        ce = HwpContentExtractor()
        pp = PreprocessedData(
            content=ole,
            resources={"bin_data_map": {"by_storage_id": {}, "by_index": []}},
        )
        result = ce.extract_text(pp)
        assert "Section One" in result
        assert "Section Two" in result


# ═══════════════════════════════════════════════════════════════════════════
# 11. Handler Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHWPHandler:
    def _make_handler(self):
        config = ProcessingConfig()
        return HWPHandler(config)

    def test_supported_extensions(self):
        h = self._make_handler()
        assert h.supported_extensions == frozenset({"hwp"})

    def test_handler_name(self):
        h = self._make_handler()
        assert h.handler_name == "HWP Handler"

    def test_creates_converter(self):
        h = self._make_handler()
        assert isinstance(h._converter, HwpConverter)

    def test_creates_preprocessor(self):
        h = self._make_handler()
        assert isinstance(h._preprocessor, HwpPreprocessor)

    def test_creates_metadata_extractor(self):
        h = self._make_handler()
        assert isinstance(h._metadata_extractor, HwpMetadataExtractor)

    def test_creates_content_extractor(self):
        h = self._make_handler()
        assert isinstance(h._content_extractor, HwpContentExtractor)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Delegation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDelegation:
    def _make_handler(self):
        config = ProcessingConfig()
        return HWPHandler(config)

    def test_zip_delegates_to_hwpx(self):
        handler = self._make_handler()
        mock_registry = MagicMock()
        mock_target = MagicMock()
        mock_target.process.return_value = ExtractionResult(text="HWPX content")
        mock_registry.get_handler.return_value = mock_target
        handler._handler_registry = mock_registry

        fc = _make_file_context(b"PK\x03\x04" + b"\x00" * 100)
        result = handler.process(fc)
        mock_registry.get_handler.assert_called_once_with("hwpx")
        assert result.text == "HWPX content"

    def test_ole_does_not_delegate(self):
        """OLE2 data goes through the normal pipeline, not delegation."""
        handler = self._make_handler()
        fc = _make_file_context(OLE2_MAGIC + b"\x00" * 100)
        # _check_delegation should return None for OLE data
        result = handler._check_delegation(fc)
        assert result is None

    def test_empty_data_no_delegation(self):
        handler = self._make_handler()
        fc = _make_file_context(b"")
        result = handler._check_delegation(fc)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 13. Full Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFullPipeline:
    def _make_handler(self):
        config = ProcessingConfig()
        return HWPHandler(config)

    @patch("contextifier_new.handlers.hwp.converter.olefile")
    @patch("contextifier_new.handlers.hwp.preprocessor.parse_doc_info")
    @patch("contextifier_new.handlers.hwp.preprocessor.is_compressed")
    @patch("contextifier_new.handlers.hwp.content_extractor.decompress_section")
    def test_full_pipeline(self, mock_decompress, mock_compressed, mock_docinfo, mock_olefile):
        """Full pipeline from OLE bytes to extracted text."""
        # Set up OLE mock
        mock_ole = MagicMock()
        mock_olefile.OleFileIO.return_value = mock_ole
        mock_ole.listdir.return_value = [["BodyText", "Section0"]]
        mock_ole.exists.return_value = False
        mock_ole.get_metadata.return_value = MagicMock(
            title="Test", subject=None, author="Author", keywords=None,
            comments=None, last_saved_by=None, create_time=None, last_saved_time=None,
        )

        # Set up preprocessing
        mock_compressed.return_value = False
        mock_docinfo.return_value = ({}, [])

        # Set up section content
        text_payload = "Full Pipeline Test".encode("utf-16le")
        record_data = _make_record_bytes(HWPTAG_PARA_TEXT, 0, text_payload)
        mock_decompress.return_value = (record_data, True)

        stream = MagicMock()
        stream.read.return_value = b"section_data"
        mock_ole.openstream.return_value = stream

        handler = self._make_handler()
        fc = _make_file_context(OLE2_MAGIC + b"\x00" * 100)
        result = handler.process(fc)

        assert isinstance(result, ExtractionResult)
        assert "Full Pipeline Test" in result.text

    @patch("contextifier_new.handlers.hwp.converter.olefile")
    def test_pipeline_conversion_failure(self, mock_olefile):
        """Conversion failure raises appropriate error."""
        mock_olefile.OleFileIO.side_effect = Exception("corrupt file")

        handler = self._make_handler()
        fc = _make_file_context(OLE2_MAGIC + b"\x00" * 100)

        from contextifier_new.errors import ConversionError
        with pytest.raises(ConversionError):
            handler.process(fc)

    def test_pipeline_validation_failure(self):
        """Non-OLE data fails validation."""
        handler = self._make_handler()
        fc = _make_file_context(b"invalid data")

        from contextifier_new.errors import ConversionError
        with pytest.raises(ConversionError):
            handler.process(fc)


# ═══════════════════════════════════════════════════════════════════════════
# 14. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_record_truncated_payload(self):
        """Truncated payload is handled gracefully."""
        # Create a record header claiming 100 bytes, but only provide 10
        header = (HWPTAG_PARA_TEXT & 0x3FF) | (0 << 10) | (100 << 20)
        data = struct.pack("<I", header) + b"\x00" * 10
        root = HwpRecord.build_tree(data)
        assert len(root.children) == 1
        assert len(root.children[0].payload) == 10

    def test_record_empty_payload(self):
        data = _make_record_bytes(HWPTAG_PARA_TEXT, 0, b"")
        root = HwpRecord.build_tree(data)
        assert len(root.children) == 1
        assert root.children[0].payload == b""

    def test_get_text_empty(self):
        rec = HwpRecord(payload=b"")
        assert rec.get_text() == ""

    def test_get_text_single_char(self):
        rec = HwpRecord(payload=struct.pack("<H", ord("A")))
        assert rec.get_text() == "A"

    def test_content_extractor_sorted_sections(self):
        """Sections are sorted numerically, not lexicographically."""
        ole = MagicMock()
        ole.listdir.return_value = [
            ["BodyText", "Section2"],
            ["BodyText", "Section0"],
            ["BodyText", "Section10"],
            ["BodyText", "Section1"],
        ]
        sections = HwpContentExtractor._sorted_sections(ole)
        names = [s[1] for s in sections]
        assert names == ["Section0", "Section1", "Section2", "Section10"]

    def test_decompress_section_various_formats(self):
        """Both zlib and raw-deflate compressed data are handled."""
        original = b"test data " * 50
        # Standard zlib
        compressed = zlib.compress(original)
        result, ok = decompress_section(compressed)
        assert ok is True
        assert result == original

    def test_converted_data_namedtuple(self):
        """HwpConvertedData is a proper NamedTuple."""
        mock_ole = MagicMock()
        cd = HwpConvertedData(ole=mock_ole, file_data=b"test")
        assert cd.ole is mock_ole
        assert cd.file_data == b"test"
        assert cd[0] is mock_ole
        assert cd[1] == b"test"
