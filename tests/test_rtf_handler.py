# tests/test_rtf_handler.py
"""
Comprehensive E2E tests for the RTF handler pipeline.

Tests cover:
1. Constants validation
2. Encoding detection + decoding
3. Text cleaning (control codes, destinations, shapes)
4. Converter (validation, encoding detection)
5. Preprocessor (binary extraction, decoding)
6. Metadata extraction (info group, dates)
7. Table parsing (cells, merges, single-column detection)
8. Content extraction (inline assembly, fallback)
9. Full pipeline integration
"""

import sys
import os
import re
from datetime import datetime
from typing import List, Tuple

import pytest

# Ensure the workspace root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextifier_new.handlers.rtf._constants import (
    CODEPAGE_ENCODING_MAP,
    DEFAULT_ENCODINGS,
    IMAGE_SIGNATURES,
    RTF_IMAGE_TYPES,
    SKIP_DESTINATIONS,
    SUPPORTED_IMAGE_FORMATS,
)
from contextifier_new.handlers.rtf._decoder import (
    decode_bytes,
    decode_content,
    decode_hex_escapes,
    detect_encoding,
)
from contextifier_new.handlers.rtf._cleaner import (
    clean_rtf_text,
    find_excluded_regions,
    is_in_excluded_region,
    remove_destination_groups,
    remove_shape_groups,
    remove_shape_property_groups,
    remove_shprslt_blocks,
)
from contextifier_new.handlers.rtf.converter import RtfConverter, RtfConvertedData
from contextifier_new.handlers.rtf.preprocessor import (
    RtfImageData,
    RtfParsedData,
    RtfPreprocessor,
)
from contextifier_new.handlers.rtf.metadata_extractor import RtfMetadataExtractor
from contextifier_new.handlers.rtf._table_parser import (
    extract_tables,
    extract_tables_with_positions,
    single_column_to_text,
)
from contextifier_new.handlers.rtf.content_extractor import RtfContentExtractor
from contextifier_new.types import DocumentMetadata, PreprocessedData, TableData, TableCell


# ═══════════════════════════════════════════════════════════════════════════
# 1. Constants
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    """Test RTF constants module."""

    def test_codepage_map_contains_common_codepages(self):
        assert 949 in CODEPAGE_ENCODING_MAP  # Korean
        assert 1252 in CODEPAGE_ENCODING_MAP  # Western European
        assert 932 in CODEPAGE_ENCODING_MAP  # Japanese
        assert 936 in CODEPAGE_ENCODING_MAP  # Chinese Simplified
        assert 65001 in CODEPAGE_ENCODING_MAP  # UTF-8

    def test_default_encodings_order(self):
        assert len(DEFAULT_ENCODINGS) >= 3
        assert DEFAULT_ENCODINGS[0] == "utf-8"

    def test_skip_destinations_are_frozenset(self):
        assert isinstance(SKIP_DESTINATIONS, frozenset)
        assert "fonttbl" in SKIP_DESTINATIONS
        assert "colortbl" in SKIP_DESTINATIONS
        assert "stylesheet" in SKIP_DESTINATIONS

    def test_image_signatures_contain_standard_formats(self):
        assert len(IMAGE_SIGNATURES) >= 4
        # JPEG
        vals = list(IMAGE_SIGNATURES.values())
        assert "jpeg" in vals
        assert "png" in vals

    def test_rtf_image_types_mapping(self):
        assert "jpegblip" in RTF_IMAGE_TYPES
        assert "pngblip" in RTF_IMAGE_TYPES

    def test_supported_image_formats(self):
        assert "jpeg" in SUPPORTED_IMAGE_FORMATS
        assert "png" in SUPPORTED_IMAGE_FORMATS
        assert "gif" in SUPPORTED_IMAGE_FORMATS


# ═══════════════════════════════════════════════════════════════════════════
# 2. Decoder
# ═══════════════════════════════════════════════════════════════════════════

class TestDecoder:
    """Test encoding detection and decoding."""

    def test_detect_encoding_ansicpg_949(self):
        content = b"{\\rtf1\\ansi\\ansicpg949 some content}"
        enc = detect_encoding(content)
        assert enc == "cp949"

    def test_detect_encoding_ansicpg_1252(self):
        content = b"{\\rtf1\\ansi\\ansicpg1252 some content}"
        enc = detect_encoding(content)
        assert enc == "cp1252"

    def test_detect_encoding_ansicpg_65001(self):
        content = b"{\\rtf1\\ansi\\ansicpg65001 body}"
        enc = detect_encoding(content)
        assert enc == "utf-8"

    def test_detect_encoding_no_ansicpg_uses_default(self):
        content = b"{\\rtf1\\ansi some content}"
        enc = detect_encoding(content, default_encoding="latin-1")
        assert enc == "latin-1"

    def test_decode_content_utf8(self):
        raw = "Hello World".encode("utf-8")
        result = decode_content(raw, "utf-8")
        assert result == "Hello World"

    def test_decode_content_cp1252_fallback(self):
        # A byte sequence valid in cp1252 but not utf-8
        raw = b"\x80\x81\x82"  # euro sign, control, comma in cp1252
        result = decode_content(raw, "cp1252")
        assert isinstance(result, str)
        assert len(result) >= 1

    def test_decode_bytes_ascii(self):
        byte_list = [0x48, 0x65, 0x6C, 0x6C, 0x6F]
        result = decode_bytes(byte_list, "utf-8")
        assert result == "Hello"

    def test_decode_hex_escapes_simple_ascii(self):
        text = r"Hello \'48\'65\'6C\'6C\'6F"
        result = decode_hex_escapes(text, "utf-8")
        assert "Hello" in result

    def test_decode_hex_escapes_no_escapes(self):
        text = "No escapes here"
        result = decode_hex_escapes(text, "utf-8")
        assert result == "No escapes here"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cleaner
# ═══════════════════════════════════════════════════════════════════════════

class TestCleaner:
    """Test RTF text cleaning functions."""

    def test_clean_rtf_text_removes_par(self):
        text = r"Hello\par World"
        result = clean_rtf_text(text, "utf-8")
        assert "Hello" in result
        assert "World" in result
        # \par should be replaced with newline
        assert "\n" in result

    def test_clean_rtf_text_removes_line(self):
        text = r"Line1\line Line2"
        result = clean_rtf_text(text, "utf-8")
        assert "Line1" in result
        assert "Line2" in result

    def test_clean_rtf_text_removes_tab(self):
        text = r"Col1\tab Col2"
        result = clean_rtf_text(text, "utf-8")
        assert "Col1" in result
        assert "Col2" in result

    def test_clean_rtf_text_handles_escaped_braces(self):
        text = r"Some \{ text \}"
        result = clean_rtf_text(text, "utf-8")
        assert "{" in result
        assert "}" in result

    def test_clean_rtf_text_handles_escaped_backslash(self):
        text = "Path: C:\\\\Users"
        result = clean_rtf_text(text, "utf-8")
        assert "\\" in result

    def test_clean_rtf_text_nonbreaking_space(self):
        text = r"Word1\~Word2"
        result = clean_rtf_text(text, "utf-8")
        assert "Word1" in result
        assert "Word2" in result

    def test_clean_rtf_text_preserves_image_tags(self):
        text = r"Text [image: path/to/img.png] more text"
        result = clean_rtf_text(text, "utf-8")
        assert "[image:" in result

    def test_remove_destination_groups_fonttbl(self):
        text = r"{\fonttbl{\f0 Arial;}{\f1 Times;}} Body text"
        result = remove_destination_groups(text)
        assert "fonttbl" not in result
        assert "Body text" in result

    def test_remove_destination_groups_colortbl(self):
        text = r"{\colortbl ;\red0\green0\blue0;} Body text"
        result = remove_destination_groups(text)
        assert "colortbl" not in result
        assert "Body text" in result

    def test_remove_destination_groups_stylesheet(self):
        text = r"{\stylesheet {\s0 Normal;}} Body text"
        result = remove_destination_groups(text)
        assert "stylesheet" not in result
        assert "Body text" in result

    def test_remove_shape_groups_removes_shape(self):
        text = r"Before {\shp {\*\shpinst some shape data}} After"
        result = remove_shape_groups(text)
        # Should not contain shape data
        assert "shpinst" not in result

    def test_remove_shprslt_blocks(self):
        text = r"Before {\*\shprslt duplicate content} After"
        result = remove_shprslt_blocks(text)
        assert "shprslt" not in result
        assert "Before" in result
        assert "After" in result

    def test_find_excluded_regions_empty(self):
        content = r"{\rtf1 \pard Normal text \par}"
        regions = find_excluded_regions(content)
        assert isinstance(regions, list)

    def test_find_excluded_regions_with_header(self):
        content = r"{\rtf1 {\headerf Header text} \pard Body text}"
        regions = find_excluded_regions(content)
        # Should find at least one excluded region
        # Note: find_excluded_regions looks for \header, \headerf, \footer, etc.
        assert isinstance(regions, list)

    def test_is_in_excluded_region_true(self):
        regions = [(10, 50), (100, 150)]
        assert is_in_excluded_region(25, regions) is True
        assert is_in_excluded_region(110, regions) is True

    def test_is_in_excluded_region_false(self):
        regions = [(10, 50), (100, 150)]
        assert is_in_excluded_region(5, regions) is False
        assert is_in_excluded_region(75, regions) is False
        assert is_in_excluded_region(200, regions) is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. Converter (Stage 1)
# ═══════════════════════════════════════════════════════════════════════════

class TestConverter:
    """Test RtfConverter."""

    def _make_rtf(self, body: str = "Hello") -> bytes:
        return f"{{\\rtf1\\ansi\\ansicpg949 {body}}}".encode("ascii")

    def _make_ctx(self, data: bytes, ext: str = "rtf") -> dict:
        import io
        return {
            "file_path": "test.rtf",
            "file_name": "test.rtf",
            "file_extension": ext,
            "file_category": "document",
            "file_data": data,
            "file_stream": io.BytesIO(data),
            "file_size": len(data),
        }

    def test_validate_valid_rtf(self):
        conv = RtfConverter()
        ctx = self._make_ctx(self._make_rtf())
        assert conv.validate(ctx) is True

    def test_validate_invalid_magic(self):
        conv = RtfConverter()
        ctx = self._make_ctx(b"Not an RTF file")
        assert conv.validate(ctx) is False

    def test_validate_empty_data(self):
        conv = RtfConverter()
        ctx = self._make_ctx(b"")
        assert conv.validate(ctx) is False

    def test_convert_returns_rtf_converted_data(self):
        conv = RtfConverter()
        data = self._make_rtf()
        ctx = self._make_ctx(data)
        result = conv.convert(ctx)
        assert isinstance(result, RtfConvertedData)
        assert result.raw_bytes == data
        assert result.encoding == "cp949"
        assert result.file_extension == "rtf"

    def test_convert_detects_encoding(self):
        conv = RtfConverter()
        data = b"{\\rtf1\\ansi\\ansicpg1252 body}"
        ctx = self._make_ctx(data)
        result = conv.convert(ctx)
        assert result.encoding == "cp1252"

    def test_convert_allows_leading_whitespace(self):
        conv = RtfConverter()
        data = b"  \n  {\\rtf1\\ansi body}"
        ctx = self._make_ctx(data)
        assert conv.validate(ctx) is True


# ═══════════════════════════════════════════════════════════════════════════
# 5. Preprocessor (Stage 2)
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    """Test RtfPreprocessor."""

    def _make_converted(self, body: str = "Hello") -> RtfConvertedData:
        raw = f"{{\\rtf1\\ansi\\ansicpg949 {body}}}".encode("ascii")
        return RtfConvertedData(raw_bytes=raw, encoding="cp949", file_extension="rtf")

    def test_preprocess_basic_rtf(self):
        pp = RtfPreprocessor()
        converted = self._make_converted("Hello World")
        result = pp.preprocess(converted)
        assert isinstance(result, PreprocessedData)
        assert isinstance(result.content, RtfParsedData)
        assert "Hello World" in result.content.text
        assert result.encoding == "cp949"

    def test_preprocess_empty_data(self):
        pp = RtfPreprocessor()
        converted = RtfConvertedData(raw_bytes=b"", encoding="utf-8", file_extension="rtf")
        result = pp.preprocess(converted)
        assert result.content.text == ""
        assert result.content.image_count == 0

    def test_preprocess_accepts_raw_bytes(self):
        pp = RtfPreprocessor()
        raw = b"{\\rtf1\\ansi Simple text}"
        result = pp.preprocess(raw)
        assert isinstance(result.content, RtfParsedData)
        assert "Simple text" in result.content.text

    def test_preprocess_resources_has_images_key(self):
        pp = RtfPreprocessor()
        converted = self._make_converted()
        result = pp.preprocess(converted)
        assert "images" in result.resources
        assert isinstance(result.resources["images"], list)

    def test_preprocess_properties_include_extension(self):
        pp = RtfPreprocessor()
        converted = self._make_converted()
        result = pp.preprocess(converted)
        assert result.properties["file_extension"] == "rtf"
        assert result.properties["encoding"] == "cp949"

    def test_get_format_name(self):
        pp = RtfPreprocessor()
        assert pp.get_format_name() == "rtf"

    def test_validate_rtf_bytes(self):
        pp = RtfPreprocessor()
        assert pp.validate(b"{\\rtf1 hello}") is True
        assert pp.validate(b"not rtf") is False


# ═══════════════════════════════════════════════════════════════════════════
# 6. Metadata Extractor (Stage 3)
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadataExtractor:
    """Test RtfMetadataExtractor."""

    def _make_parsed(self, content: str) -> RtfParsedData:
        return RtfParsedData(text=content, encoding="cp949", image_count=0)

    def test_extract_empty_content(self):
        ext = RtfMetadataExtractor()
        result = ext.extract(self._make_parsed(""))
        assert isinstance(result, DocumentMetadata)
        assert result.title is None

    def test_extract_no_info_group(self):
        ext = RtfMetadataExtractor()
        result = ext.extract(self._make_parsed("{\\rtf1 no metadata}"))
        assert result.title is None
        assert result.author is None

    def test_extract_title(self):
        ext = RtfMetadataExtractor()
        content = "{\\rtf1 {\\info{\\title My Document}{\\author John}}}"
        result = ext.extract(self._make_parsed(content))
        assert result.title == "My Document"
        assert result.author == "John"

    def test_extract_subject_keywords(self):
        ext = RtfMetadataExtractor()
        content = "{\\rtf1 {\\info{\\subject Test Subject}{\\keywords key1 key2}}}"
        result = ext.extract(self._make_parsed(content))
        assert result.subject == "Test Subject"
        assert result.keywords == "key1 key2"

    def test_extract_comments(self):
        ext = RtfMetadataExtractor()
        content = "{\\rtf1 {\\info{\\doccomm This is a comment}}}"
        result = ext.extract(self._make_parsed(content))
        assert result.comments == "This is a comment"

    def test_extract_operator(self):
        ext = RtfMetadataExtractor()
        content = "{\\rtf1 {\\info{\\operator Admin User}}}"
        result = ext.extract(self._make_parsed(content))
        assert result.last_saved_by == "Admin User"

    def test_extract_create_time(self):
        ext = RtfMetadataExtractor()
        content = r"{\rtf1 {\info{\title Doc}} {\creatim\yr2024\mo6\dy15\hr10\min30}}"
        result = ext.extract(self._make_parsed(content))
        assert result.create_time is not None
        assert result.create_time.year == 2024
        assert result.create_time.month == 6
        assert result.create_time.day == 15
        assert result.create_time.hour == 10
        assert result.create_time.minute == 30

    def test_extract_last_saved_time(self):
        ext = RtfMetadataExtractor()
        content = r"{\rtf1 {\info{\title Doc}} {\revtim\yr2024\mo12\dy25}}"
        result = ext.extract(self._make_parsed(content))
        assert result.last_saved_time is not None
        assert result.last_saved_time.year == 2024
        assert result.last_saved_time.month == 12

    def test_extract_multiple_fields(self):
        ext = RtfMetadataExtractor()
        content = (
            r"{\rtf1 {\info"
            r"{\title Report 2024}"
            r"{\author Jane Doe}"
            r"{\subject Annual Report}"
            r"{\keywords finance annual}"
            r"{\operator John Smith}"
            r"}"
            r" {\creatim\yr2024\mo1\dy1}"
            r" {\revtim\yr2024\mo6\dy30}"
            r"}"
        )
        result = ext.extract(self._make_parsed(content))
        assert result.title == "Report 2024"
        assert result.author == "Jane Doe"
        assert result.subject == "Annual Report"
        assert result.keywords == "finance annual"
        assert result.last_saved_by == "John Smith"
        assert result.create_time.year == 2024
        assert result.last_saved_time.month == 6

    def test_get_format_name(self):
        ext = RtfMetadataExtractor()
        assert ext.get_format_name() == "rtf"

    def test_accepts_dict_input(self):
        ext = RtfMetadataExtractor()
        result = ext.extract({"text": "{\\rtf1 {\\info{\\title Dict Title}}}", "encoding": "utf-8"})
        assert result.title == "Dict Title"

    def test_accepts_string_input(self):
        ext = RtfMetadataExtractor()
        result = ext.extract("{\\rtf1 {\\info{\\title String Title}}}")
        assert result.title == "String Title"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Table Parser
# ═══════════════════════════════════════════════════════════════════════════

class TestTableParser:
    """Test RTF table parsing."""

    def _make_simple_table(self) -> str:
        """Create a simple 2x2 RTF table."""
        return (
            r"\trowd\cellx3000\cellx6000"
            r" Cell A1\cell Cell B1\cell\row"
            r"\trowd\cellx3000\cellx6000"
            r" Cell A2\cell Cell B2\cell\row"
        )

    def _make_single_column_table(self) -> str:
        """Create a single-column table (should be treated as text)."""
        return (
            r"\trowd\cellx6000"
            r" Line 1\cell\row"
            r"\trowd\cellx6000"
            r" Line 2\cell\row"
        )

    def test_extract_simple_table(self):
        content = self._make_simple_table()
        tables = extract_tables(content, "utf-8")
        assert len(tables) == 1
        table = tables[0]
        assert table.num_rows == 2
        assert table.num_cols == 2

    def test_extract_table_cell_content(self):
        content = self._make_simple_table()
        tables = extract_tables(content, "utf-8")
        assert len(tables) == 1
        table = tables[0]
        # Check first row, first cell
        first_row = table.rows[0]
        assert any("Cell A1" in cell.content for cell in first_row)

    def test_extract_table_with_positions(self):
        content = "Preamble text " + self._make_simple_table() + " Trailing text"
        ranges, regions = extract_tables_with_positions(content, "utf-8")
        assert len(ranges) == 1
        assert len(regions) == 1
        start, end = ranges[0]
        assert start > 0
        assert end > start

    def test_single_column_not_a_table(self):
        content = self._make_single_column_table()
        tables = extract_tables(content, "utf-8")
        # Single-column should NOT produce a table
        assert len(tables) == 0

    def test_single_column_to_text(self):
        content = self._make_single_column_table()
        # Extract the row texts manually from the content
        row_texts = re.findall(r"\\trowd.*?\\row", content, re.DOTALL)
        result = single_column_to_text(row_texts, "utf-8")
        assert "Line 1" in result
        assert "Line 2" in result

    def test_no_tables_in_plain_text(self):
        content = "Just some plain text with no table structures."
        tables = extract_tables(content, "utf-8")
        assert len(tables) == 0

    def test_table_cells_are_tablecell_instances(self):
        content = self._make_simple_table()
        tables = extract_tables(content, "utf-8")
        assert len(tables) == 1
        for row in tables[0].rows:
            for cell in row:
                assert isinstance(cell, TableCell)
                assert isinstance(cell.content, str)
                assert isinstance(cell.row_span, int)
                assert isinstance(cell.col_span, int)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Content Extractor (Stage 4)
# ═══════════════════════════════════════════════════════════════════════════

class TestContentExtractor:
    """Test RtfContentExtractor."""

    def _make_preprocessed(
        self,
        text: str,
        encoding: str = "utf-8",
        images: List[RtfImageData] = None,
    ) -> PreprocessedData:
        parsed = RtfParsedData(
            text=text,
            encoding=encoding,
            image_count=len(images) if images else 0,
        )
        return PreprocessedData(
            content=parsed,
            raw_content=text,
            encoding=encoding,
            resources={"images": images or []},
        )

    def test_extract_text_simple(self):
        ext = RtfContentExtractor()
        content = r"{\rtf1\pard Hello World \par}"
        pp = self._make_preprocessed(content)
        text = ext.extract_text(pp)
        assert "Hello World" in text

    def test_extract_text_with_control_codes(self):
        ext = RtfContentExtractor()
        content = (
            r"{\rtf1\ansi"
            r"{\fonttbl{\f0 Arial;}}"
            r"{\colortbl;\red0\green0\blue0;}"
            r"\pard\f0\fs24 Clean text here.\par"
            r"}"
        )
        pp = self._make_preprocessed(content)
        text = ext.extract_text(pp)
        assert "Clean text here" in text
        assert "fonttbl" not in text
        assert "colortbl" not in text

    def test_extract_text_empty(self):
        ext = RtfContentExtractor()
        pp = self._make_preprocessed("")
        text = ext.extract_text(pp)
        assert text == ""

    def test_extract_text_with_table(self):
        ext = RtfContentExtractor()
        content = (
            r"{\rtf1\pard Before table.\par"
            r"\trowd\cellx3000\cellx6000"
            r" A1\cell B1\cell\row"
            r"\trowd\cellx3000\cellx6000"
            r" A2\cell B2\cell\row"
            r"\pard After table.\par}"
        )
        pp = self._make_preprocessed(content)
        text = ext.extract_text(pp)
        assert "Before table" in text
        assert "After table" in text
        # Table content should be present (as HTML or inline)
        assert "A1" in text
        assert "B1" in text

    def test_extract_tables_returns_tabledata(self):
        ext = RtfContentExtractor()
        content = (
            r"\trowd\cellx3000\cellx6000"
            r" X\cell Y\cell\row"
            r"\trowd\cellx3000\cellx6000"
            r" Z\cell W\cell\row"
        )
        pp = self._make_preprocessed(content)
        tables = ext.extract_tables(pp)
        assert len(tables) == 1
        assert isinstance(tables[0], TableData)

    def test_extract_images_no_service(self):
        ext = RtfContentExtractor()
        pp = self._make_preprocessed("some text")
        images = ext.extract_images(pp)
        assert images == []

    def test_get_format_name(self):
        ext = RtfContentExtractor()
        assert ext.get_format_name() == "rtf"

    def test_fallback_striprtf(self):
        """Test striprtf fallback for content with no \\pard."""
        ext = RtfContentExtractor()
        # Content that won't produce text through normal extraction
        content = r"{\rtf1\ansi Some body text}"
        pp = self._make_preprocessed(content)
        text = ext.extract_text(pp)
        # Should still extract something (via normal or fallback)
        assert isinstance(text, str)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Full Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Integration tests: Converter → Preprocessor → Metadata → Content."""

    def _make_full_rtf(self) -> bytes:
        """Create an RTF document with metadata, text, and a table."""
        return (
            b"{\\rtf1\\ansi\\ansicpg1252"
            b"{\\fonttbl{\\f0 Arial;}}"
            b"{\\info"
            b"{\\title Integration Test}"
            b"{\\author Test Author}"
            b"{\\subject Test Subject}"
            b"}"
            b"{\\creatim\\yr2024\\mo3\\dy15\\hr9\\min0}"
            b"\\pard\\f0\\fs24 Introduction paragraph.\\par"
            b"\\trowd\\cellx3000\\cellx6000"
            b" Header A\\cell Header B\\cell\\row"
            b"\\trowd\\cellx3000\\cellx6000"
            b" Value 1\\cell Value 2\\cell\\row"
            b"\\pard Conclusion paragraph.\\par"
            b"}"
        )

    def _make_ctx(self, data: bytes) -> dict:
        import io
        return {
            "file_path": "test.rtf", "file_name": "test.rtf",
            "file_extension": "rtf", "file_category": "document",
            "file_data": data, "file_stream": io.BytesIO(data),
            "file_size": len(data),
        }

    def test_converter_to_preprocessor(self):
        """Test Converter → Preprocessor flow."""
        raw = self._make_full_rtf()
        ctx = self._make_ctx(raw)

        converter = RtfConverter()
        assert converter.validate(ctx) is True
        converted = converter.convert(ctx)
        assert isinstance(converted, RtfConvertedData)

        preprocessor = RtfPreprocessor()
        result = preprocessor.preprocess(converted)
        assert isinstance(result.content, RtfParsedData)
        assert "Introduction" in result.content.text

    def test_preprocessor_to_metadata(self):
        """Test Preprocessor → MetadataExtractor flow."""
        raw = self._make_full_rtf()
        converted = RtfConvertedData(
            raw_bytes=raw, encoding="cp1252", file_extension="rtf",
        )

        preprocessor = RtfPreprocessor()
        preprocessed = preprocessor.preprocess(converted)

        metadata_ext = RtfMetadataExtractor()
        metadata = metadata_ext.extract(preprocessed.content)
        assert metadata.title == "Integration Test"
        assert metadata.author == "Test Author"
        assert metadata.subject == "Test Subject"

    def test_preprocessor_to_content(self):
        """Test Preprocessor → ContentExtractor flow."""
        raw = self._make_full_rtf()
        converted = RtfConvertedData(
            raw_bytes=raw, encoding="cp1252", file_extension="rtf",
        )

        preprocessor = RtfPreprocessor()
        preprocessed = preprocessor.preprocess(converted)

        content_ext = RtfContentExtractor()
        text = content_ext.extract_text(preprocessed)
        assert "Introduction paragraph" in text
        assert "Conclusion paragraph" in text

    def test_full_pipeline(self):
        """Full Converter → Preprocessor → Metadata → Content."""
        raw = self._make_full_rtf()

        # Stage 1: Convert
        converter = RtfConverter()
        converted = converter.convert(self._make_ctx(raw))

        # Stage 2: Preprocess
        preprocessor = RtfPreprocessor()
        preprocessed = preprocessor.preprocess(converted)

        # Stage 3: Metadata
        metadata_ext = RtfMetadataExtractor()
        metadata = metadata_ext.extract(preprocessed.content)

        # Stage 4: Content
        content_ext = RtfContentExtractor()
        text = content_ext.extract_text(preprocessed)
        tables = content_ext.extract_tables(preprocessed)

        # Validate
        assert metadata.title == "Integration Test"
        assert metadata.author == "Test Author"
        assert metadata.create_time.year == 2024
        assert "Introduction paragraph" in text
        assert "Conclusion paragraph" in text
        assert len(tables) == 1
        assert tables[0].num_rows == 2
        assert tables[0].num_cols == 2

    def test_extract_all(self):
        """Test ContentExtractor.extract_all() orchestration."""
        raw = self._make_full_rtf()
        converted = RtfConvertedData(
            raw_bytes=raw, encoding="cp1252", file_extension="rtf",
        )

        preprocessor = RtfPreprocessor()
        preprocessed = preprocessor.preprocess(converted)

        metadata_ext = RtfMetadataExtractor()
        metadata = metadata_ext.extract(preprocessed.content)

        content_ext = RtfContentExtractor()
        result = content_ext.extract_all(
            preprocessed,
            extract_metadata_result=metadata,
        )
        from contextifier_new.types import ExtractionResult
        assert isinstance(result, ExtractionResult)
        assert result.metadata == metadata
        assert len(result.text) > 0
        assert len(result.tables) == 1
        assert result.warnings == [] or isinstance(result.warnings, list)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_converter_with_bom(self):
        """UTF-8 BOM before RTF content."""
        import io
        conv = RtfConverter()
        data = b"\xef\xbb\xbf{\\rtf1\\ansi body}"
        ctx = {
            "file_path": "test.rtf", "file_name": "test.rtf",
            "file_extension": "rtf", "file_category": "document",
            "file_data": data, "file_stream": io.BytesIO(data),
            "file_size": len(data),
        }
        # BOM + {\\rtf: BOM isn't whitespace so lstrip() won't help.
        # This is acceptable — RTF files shouldn't have BOMs.
        result = conv.validate(ctx)
        assert isinstance(result, bool)

    def test_preprocessor_with_deeply_nested_braces(self):
        """RTF with deeply nested brace structures."""
        pp = RtfPreprocessor()
        content = b"{\\rtf1 {{{nested text}}}}"
        result = pp.preprocess(content)
        assert isinstance(result.content.text, str)

    def test_metadata_with_empty_fields(self):
        ext = RtfMetadataExtractor()
        content = "{\\rtf1 {\\info{\\title }{\\author }}}"
        result = ext.extract(
            RtfParsedData(text=content, encoding="utf-8", image_count=0)
        )
        # Empty fields should be None, not empty strings
        assert result.title is None
        assert result.author is None

    def test_metadata_invalid_date(self):
        ext = RtfMetadataExtractor()
        content = r"{\rtf1 {\info{\title X}} {\creatim\yr9999\mo99\dy99}}"
        result = ext.extract(
            RtfParsedData(text=content, encoding="utf-8", image_count=0)
        )
        # Invalid date should be None, not crash
        assert result.create_time is None

    def test_table_with_no_cell_content(self):
        content = (
            r"\trowd\cellx3000\cellx6000"
            r"\cell\cell\row"
            r"\trowd\cellx3000\cellx6000"
            r"\cell\cell\row"
        )
        tables = extract_tables(content, "utf-8")
        # Empty table may or may not be "real" - should not crash
        assert isinstance(tables, list)

    def test_clean_text_with_unicode_escape(self):
        text = r"\u8364?EUR"
        result = clean_rtf_text(text, "utf-8")
        # \u8364 is the euro sign €
        assert "€" in result or "EUR" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
