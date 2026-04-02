# tests/unit/handlers/test_doc_fib.py
"""
Tests for DOC FIB + Piece Table text extraction and table detection.

Covers:
- FIB header validation
- Piece table parsing (compressed / Unicode)
- Full pipeline: FIB → Clx → pieces → text
- Table detection from cell markers
- Fallback to heuristic when FIB fails
"""

from __future__ import annotations

import struct
import pytest

from contextifier.handlers.doc._fib import (
    PieceDescriptor,
    _clean_doc_text,
    _parse_clx,
    _parse_plc_pcd,
    _read_pieces,
    detect_tables_from_text,
    parse_fib_text,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_word_data(
    *,
    wident: int = 0xA5EC,
    nfib: int = 0x00C1,
    ccp_text: int = 0,
    fc_clx: int = 0,
    lcb_clx: int = 0,
    body: bytes = b"",
    size: int = 0x0400,
) -> bytes:
    """Build a minimal WordDocument stream with FIB fields set."""
    buf = bytearray(max(size, 0x01AA))
    struct.pack_into("<H", buf, 0x0000, wident)    # wIdent
    struct.pack_into("<H", buf, 0x0002, nfib)      # nFib
    struct.pack_into("<I", buf, 0x004C, ccp_text)  # ccpText
    struct.pack_into("<I", buf, 0x01A2, fc_clx)    # fcClx
    struct.pack_into("<I", buf, 0x01A6, lcb_clx)   # lcbClx
    # Write body text at a fixed offset
    body_offset = 0x0200
    buf[body_offset : body_offset + len(body)] = body
    return bytes(buf)


def _build_clx_compressed(text: str, body_offset: int = 0x0200) -> tuple[bytes, int, int]:
    """
    Build a Clx structure for compressed (cp1252) text.

    Returns (table_stream_bytes, fc_clx, lcb_clx).
    The returned body_offset is where text should be in WordDocument stream.
    """
    text_bytes = text.encode("cp1252")
    char_count = len(text_bytes)

    # Build PlcPcd: 2 CPs (uint32) + 1 PCD (8 bytes)
    # CPs: [0, char_count]
    # PCD: flags=0x0000, fc= (body_offset * 2) | 0x40000000, prm=0x0000
    fc_raw = (body_offset * 2) | 0x40000000  # compressed flag + doubled offset
    plc_pcd = struct.pack("<II", 0, char_count)  # 2 CPs
    plc_pcd += struct.pack("<HIH", 0, fc_raw, 0)  # 1 PCD (2+4+2 = 8 bytes)

    # Build Pcdt: type 0x02 + cbPlcPcd + PlcPcd data
    pcdt = struct.pack("<BI", 0x02, len(plc_pcd)) + plc_pcd

    # Clx = Pcdt (no Prc entries)
    clx = pcdt

    # Build a table stream with Clx at offset 0
    table_stream = clx
    return table_stream, 0, len(clx)


def _build_clx_unicode(text: str, body_offset: int = 0x0200) -> tuple[bytes, int, int]:
    """
    Build a Clx structure for Unicode (UTF-16LE) text.

    Returns (table_stream_bytes, fc_clx, lcb_clx).
    """
    text_bytes = text.encode("utf-16-le")
    char_count = len(text)

    # Build PlcPcd: 2 CPs + 1 PCD
    # fc = body_offset (no compressed flag)
    plc_pcd = struct.pack("<II", 0, char_count)  # 2 CPs
    plc_pcd += struct.pack("<HIH", 0, body_offset, 0)  # 1 PCD
    pcdt = struct.pack("<BI", 0x02, len(plc_pcd)) + plc_pcd
    table_stream = pcdt
    return table_stream, 0, len(pcdt)


# ═════════════════════════════════════════════════════════════════════════════
# FIB validation tests
# ═════════════════════════════════════════════════════════════════════════════

class TestFibValidation:
    """FIB header checks reject invalid inputs."""

    def test_too_short_stream(self):
        result = parse_fib_text(b"\x00" * 10, b"\x00" * 10)
        assert result is None

    def test_bad_magic(self):
        wd = _build_word_data(wident=0xBEEF)
        result = parse_fib_text(wd, b"\x00" * 10)
        assert result is None

    def test_word95_nfib_rejected(self):
        wd = _build_word_data(nfib=0x006C)  # Word 6.0/95
        result = parse_fib_text(wd, b"\x00" * 10)
        assert result is None

    def test_no_table_stream(self):
        wd = _build_word_data(fc_clx=0, lcb_clx=100)
        result = parse_fib_text(wd, None)
        assert result is None

    def test_empty_table_stream(self):
        wd = _build_word_data(fc_clx=0, lcb_clx=100)
        result = parse_fib_text(wd, b"")
        assert result is None

    def test_clx_beyond_table_stream(self):
        wd = _build_word_data(fc_clx=9999, lcb_clx=100)
        result = parse_fib_text(wd, b"\x00" * 50)
        assert result is None

    def test_zero_lcb_clx(self):
        wd = _build_word_data(fc_clx=0, lcb_clx=0)
        result = parse_fib_text(wd, b"\x00" * 50)
        assert result is None


# ═════════════════════════════════════════════════════════════════════════════
# Piece table parsing tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPieceParsing:
    """PlcPcd parsing extracts correct piece descriptors."""

    def test_single_compressed_piece(self):
        text = "Hello World"
        table_stream, fc_clx, lcb_clx = _build_clx_compressed(text)

        body_offset = 0x0200
        word_data = _build_word_data(
            ccp_text=len(text),
            fc_clx=fc_clx,
            lcb_clx=lcb_clx,
            body=text.encode("cp1252"),
        )

        result = parse_fib_text(word_data, table_stream)
        assert result is not None
        assert "Hello World" in result

    def test_single_unicode_piece(self):
        text = "한글 테스트"
        table_stream, fc_clx, lcb_clx = _build_clx_unicode(text)

        body_offset = 0x0200
        word_data = _build_word_data(
            ccp_text=len(text),
            fc_clx=fc_clx,
            lcb_clx=lcb_clx,
            body=text.encode("utf-16-le"),
        )

        result = parse_fib_text(word_data, table_stream)
        assert result is not None
        assert "한글" in result

    def test_a5dc_magic_accepted(self):
        text = "Test"
        table_stream, fc_clx, lcb_clx = _build_clx_compressed(text)
        word_data = _build_word_data(
            wident=0xA5DC,  # Word 95 variant magic (but nFib >= 0xC1)
            ccp_text=len(text),
            fc_clx=fc_clx,
            lcb_clx=lcb_clx,
            body=text.encode("cp1252"),
        )
        result = parse_fib_text(word_data, table_stream)
        assert result is not None
        assert "Test" in result


# ═════════════════════════════════════════════════════════════════════════════
# Clx parser tests
# ═════════════════════════════════════════════════════════════════════════════

class TestClxParsing:
    """Clx structure parsing handles Prc + Pcdt correctly."""

    def test_pcdt_only(self):
        """Clx containing only a Pcdt (no Prc)."""
        plc = struct.pack("<II", 0, 10)  # 2 CPs
        plc += struct.pack("<HIH", 0, 0x40000200, 0)  # 1 PCD (compressed)
        pcdt = struct.pack("<BI", 0x02, len(plc)) + plc
        pieces = _parse_clx(pcdt)
        assert len(pieces) == 1
        assert pieces[0].is_compressed is True

    def test_prc_then_pcdt(self):
        """Clx with a Prc entry before Pcdt."""
        # Prc: type 0x01, cbGrpprl=2, 2 bytes of data
        prc = struct.pack("<BH", 0x01, 2) + b"\x00\x00"
        # Pcdt
        plc = struct.pack("<II", 0, 5)
        plc += struct.pack("<HIH", 0, 0x00000100, 0)  # Unicode piece
        pcdt = struct.pack("<BI", 0x02, len(plc)) + plc
        clx = prc + pcdt
        pieces = _parse_clx(clx)
        assert len(pieces) == 1
        assert pieces[0].is_compressed is False

    def test_empty_clx(self):
        pieces = _parse_clx(b"")
        assert pieces == []

    def test_unknown_type_aborts(self):
        pieces = _parse_clx(b"\xFF\x00\x00\x00")
        assert pieces == []


# ═════════════════════════════════════════════════════════════════════════════
# Table detection tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTableDetection:
    """Table detection from cell markers."""

    def test_simple_table(self):
        text = "Header\nA\x07B\x07C\x07\nD\x07E\x07F\x07\nFooter"
        tables = detect_tables_from_text(text)
        assert len(tables) == 1
        assert len(tables[0]) == 2  # 2 rows
        assert tables[0][0] == ["A", "B", "C"]
        assert tables[0][1] == ["D", "E", "F"]

    def test_no_cell_markers(self):
        text = "Just plain text\nwith no tables."
        tables = detect_tables_from_text(text)
        assert tables == []

    def test_single_row_rejected(self):
        """A single row doesn't make a table."""
        text = "A\x07B\x07C\x07"
        tables = detect_tables_from_text(text)
        assert tables == []

    def test_multiple_tables(self):
        text = "A\x07B\x07\nC\x07D\x07\nSeparator\nE\x07F\x07\nG\x07H\x07"
        tables = detect_tables_from_text(text)
        assert len(tables) == 2

    def test_korean_cell_content(self):
        text = "이름\x07나이\x07\n홍길동\x0730\x07"
        tables = detect_tables_from_text(text)
        assert len(tables) == 1
        assert tables[0][0] == ["이름", "나이"]
        assert tables[0][1] == ["홍길동", "30"]


# ═════════════════════════════════════════════════════════════════════════════
# Text cleaning tests
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanDocText:
    """Text cleaning handles DOC-specific characters."""

    def test_paragraph_marks(self):
        assert _clean_doc_text("Hello\rWorld") == "Hello\nWorld"

    def test_cr_lf(self):
        assert _clean_doc_text("Hello\r\nWorld") == "Hello\nWorld"

    def test_form_feeds(self):
        assert "Hello" in _clean_doc_text("Hello\x0cWorld")

    def test_control_chars_removed(self):
        result = _clean_doc_text("Hello\x01\x02\x03World")
        assert result == "HelloWorld"

    def test_excessive_newlines_collapsed(self):
        result = _clean_doc_text("A\n\n\n\n\nB")
        assert result == "A\n\nB"


__all__: list[str] = []
