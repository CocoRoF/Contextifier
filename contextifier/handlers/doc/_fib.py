# contextifier/handlers/doc/_fib.py
"""
FIB (File Information Block) and Piece Table parser for DOC files.

Implements MS-DOC Binary File Format text extraction via:
1. FIB header parsing → locate Clx offset in Table Stream
2. Clx parsing → extract PlcPcd (piece descriptor table)
3. Piece-by-piece text reconstruction from WordDocument stream

Reference: [MS-DOC] — Word (.doc) Binary File Format
https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-doc/

This module is used as the *primary* text extraction path in
``DocContentExtractor``.  The heuristic UTF-16LE scanner in
``_extract_text_from_word_stream()`` serves as fallback when
FIB parsing fails (e.g. corrupted FIB, Word 95 files).
"""

from __future__ import annotations

import re
import struct
import logging
from typing import List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

# ── FIB well-known offsets (Word 97+ / nFib >= 0x00C1) ──────────────────────

# FibBase fields
_OFFSET_WIDENT = 0x0000       # uint16 — magic (0xA5EC / 0xA5DC)
_OFFSET_NFIB = 0x0002         # uint16 — version number
_OFFSET_FLAGS = 0x000A        # uint16 — flags (bit 9 = fWhichTblStm)

# FibRgLw97 fields (absolute offsets assuming standard csw=14, cslw=22)
_OFFSET_CCPTEXT = 0x004C      # uint32 — character count of main body text
_OFFSET_CCPFTN = 0x0050       # uint32 — character count of footnote text

# FibRgFcLcb97: fcClx / lcbClx
# fcClx  is at absolute offset 0x01A2 (entry index 33 of FibRgFcLcb97)
# lcbClx is at absolute offset 0x01A6
_OFFSET_FC_CLX = 0x01A2
_OFFSET_LCB_CLX = 0x01A6

# Minimum FIB size to attempt piece-table parsing
_MIN_FIB_SIZE = 0x01AA        # must be able to read lcbClx

# nFib threshold — piece table parsing only reliable for Word 97+
_MIN_NFIB_FOR_PIECE_TABLE = 0x00C1   # Word 97


class PieceDescriptor(NamedTuple):
    """A single entry from the PlcPcd piece table."""
    cp_start: int         # Starting character position (in document order)
    cp_end: int           # Ending character position (exclusive)
    fc: int               # File Character offset in WordDocument stream
    is_compressed: bool   # True → cp1252 (1 byte/char), False → UTF-16LE


def parse_fib_text(
    word_data: bytes,
    table_stream: Optional[bytes],
) -> Optional[str]:
    """
    Extract text from a DOC file using FIB → Piece Table parsing.

    Args:
        word_data: Raw bytes of the WordDocument stream.
        table_stream: Raw bytes of the 0Table or 1Table stream.

    Returns:
        Extracted text as a string, or ``None`` if FIB parsing fails
        (caller should fall back to heuristic extraction).
    """
    if len(word_data) < _MIN_FIB_SIZE:
        logger.debug("WordDocument stream too short for FIB parsing (%d bytes)", len(word_data))
        return None

    if table_stream is None or len(table_stream) == 0:
        logger.debug("No table stream available for piece table parsing")
        return None

    # ── 1. Read FIB header ────────────────────────────────────────────────
    wident = struct.unpack_from("<H", word_data, _OFFSET_WIDENT)[0]
    nfib = struct.unpack_from("<H", word_data, _OFFSET_NFIB)[0]

    if wident not in (0xA5EC, 0xA5DC):
        logger.debug("Invalid FIB magic: 0x%04X", wident)
        return None

    if nfib < _MIN_NFIB_FOR_PIECE_TABLE:
        logger.debug("nFib 0x%04X too old for piece table parsing", nfib)
        return None

    # ── 2. Locate Clx in Table Stream ─────────────────────────────────────
    fc_clx = struct.unpack_from("<I", word_data, _OFFSET_FC_CLX)[0]
    lcb_clx = struct.unpack_from("<I", word_data, _OFFSET_LCB_CLX)[0]

    if lcb_clx == 0:
        logger.debug("lcbClx is 0 — no Clx structure present")
        return None

    if fc_clx + lcb_clx > len(table_stream):
        logger.debug(
            "Clx extends beyond table stream (fcClx=%d, lcbClx=%d, stream=%d)",
            fc_clx, lcb_clx, len(table_stream),
        )
        return None

    clx_data = table_stream[fc_clx : fc_clx + lcb_clx]

    # ── 3. Parse Clx → PlcPcd ─────────────────────────────────────────────
    pieces = _parse_clx(clx_data)
    if not pieces:
        logger.debug("Failed to parse Clx — no piece descriptors found")
        return None

    # ── 4. Read ccpText for validation ────────────────────────────────────
    ccp_text = struct.unpack_from("<I", word_data, _OFFSET_CCPTEXT)[0]

    # ── 5. Reconstruct text from pieces ───────────────────────────────────
    text = _read_pieces(word_data, pieces, ccp_text)
    if text is None:
        return None

    # ── 6. Clean up ───────────────────────────────────────────────────────
    text = _clean_doc_text(text)
    return text if text else None


def _parse_clx(clx_data: bytes) -> List[PieceDescriptor]:
    """
    Parse the Clx structure to extract piece descriptors.

    Clx = *Prc  Pcdt
    Prc:  clxt=0x01, PrcData (skip)
    Pcdt: clxt=0x02, cbPlcPcd (uint32), PlcPcd data
    """
    offset = 0
    length = len(clx_data)

    # Skip Prc entries (type 0x01)
    while offset < length:
        if clx_data[offset] == 0x01:
            # Prc: 1 byte type + 2 byte cbGrpprl + cbGrpprl bytes
            if offset + 3 > length:
                return []
            cb = struct.unpack_from("<H", clx_data, offset + 1)[0]
            offset += 3 + cb
        elif clx_data[offset] == 0x02:
            # Found Pcdt
            break
        else:
            # Unknown type — abort
            logger.debug("Unknown Clx entry type: 0x%02X at offset %d", clx_data[offset], offset)
            return []

    if offset >= length or clx_data[offset] != 0x02:
        return []

    offset += 1  # skip type byte

    if offset + 4 > length:
        return []

    cb_plc_pcd = struct.unpack_from("<I", clx_data, offset)[0]
    offset += 4

    if offset + cb_plc_pcd > length:
        logger.debug("PlcPcd extends beyond Clx data")
        return []

    plc_data = clx_data[offset : offset + cb_plc_pcd]
    return _parse_plc_pcd(plc_data)


def _parse_plc_pcd(plc_data: bytes) -> List[PieceDescriptor]:
    """
    Parse PlcPcd structure into piece descriptors.

    PlcPcd layout:
        CPs:  (n+1) uint32 values (character positions)
        PCDs: n entries of 8 bytes each

    n = (cbPlcPcd - 4) / 12
    (because: 4*(n+1) + 8*n = 12n + 4 = cbPlcPcd)
    """
    cb = len(plc_data)
    if cb < 16:  # minimum: 2 CPs (8 bytes) + 1 PCD (8 bytes) = 16
        return []

    # Calculate number of pieces
    n, remainder = divmod(cb - 4, 12)
    if remainder != 0 or n < 1:
        logger.debug("PlcPcd size %d does not produce integer piece count", cb)
        return []

    pieces: List[PieceDescriptor] = []

    # Read CPs
    cps: List[int] = []
    for i in range(n + 1):
        cp = struct.unpack_from("<I", plc_data, i * 4)[0]
        cps.append(cp)

    # Read PCDs (start after CPs)
    pcd_offset = (n + 1) * 4
    for i in range(n):
        # PCD: 2 bytes flags + 4 bytes fc + 2 bytes prm = 8 bytes
        fc_raw = struct.unpack_from("<I", plc_data, pcd_offset + i * 8 + 2)[0]

        is_compressed = bool(fc_raw & 0x40000000)  # bit 30
        fc = fc_raw & 0x3FFFFFFF  # clear bit 30 and 31

        if is_compressed:
            # ANSI: actual byte offset is fc / 2
            fc = fc // 2

        pieces.append(PieceDescriptor(
            cp_start=cps[i],
            cp_end=cps[i + 1],
            fc=fc,
            is_compressed=is_compressed,
        ))

    return pieces


def _read_pieces(
    word_data: bytes,
    pieces: List[PieceDescriptor],
    ccp_text: int,
) -> Optional[str]:
    """
    Read text from the WordDocument stream using piece descriptors.

    Only reads the main body text (cp < ccp_text).  Footnotes,
    headers, and other sub-documents are skipped.
    """
    parts: List[str] = []
    stream_len = len(word_data)

    for piece in pieces:
        cp_start = piece.cp_start
        cp_end = min(piece.cp_end, ccp_text) if ccp_text > 0 else piece.cp_end

        if cp_start >= cp_end:
            continue

        char_count = cp_end - cp_start

        if piece.is_compressed:
            # cp1252: 1 byte per character
            byte_offset = piece.fc
            byte_len = char_count
            if byte_offset + byte_len > stream_len:
                logger.debug(
                    "Piece extends beyond stream (offset=%d, len=%d, stream=%d)",
                    byte_offset, byte_len, stream_len,
                )
                continue
            raw = word_data[byte_offset : byte_offset + byte_len]
            try:
                text = raw.decode("cp1252", errors="replace")
            except Exception:
                continue
        else:
            # UTF-16LE: 2 bytes per character
            byte_offset = piece.fc
            byte_len = char_count * 2
            if byte_offset + byte_len > stream_len:
                logger.debug(
                    "Piece extends beyond stream (offset=%d, len=%d, stream=%d)",
                    byte_offset, byte_len, stream_len,
                )
                continue
            raw = word_data[byte_offset : byte_offset + byte_len]
            try:
                text = raw.decode("utf-16-le", errors="replace")
            except Exception:
                continue

        parts.append(text)

    if not parts:
        return None

    return "".join(parts)


def _clean_doc_text(text: str) -> str:
    """
    Clean text extracted from DOC piece table.

    Normalizes paragraph marks, removes binary control characters,
    and collapses excessive whitespace.
    """
    # Replace DOC paragraph marks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace section breaks (0x0C = form feed) with double newline
    text = text.replace("\x0c", "\n\n")

    # Remove non-printable control characters (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f]", "", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_tables_from_text(text: str) -> List[List[List[str]]]:
    """
    Detect table structures from cell markers in DOC text.

    In MS-DOC format, table cells are delimited by ``\\x07`` (BEL).
    A row typically ends with ``\\x07`` followed by paragraph end.

    Returns:
        List of tables, where each table is a list of rows,
        and each row is a list of cell strings.
        Returns empty list if no tables detected.
    """
    if "\x07" not in text:
        return []

    tables: List[List[List[str]]] = []
    current_rows: List[List[str]] = []

    # Split into lines; lines containing \x07 are table rows
    for line in text.split("\n"):
        if "\x07" not in line:
            # Non-table line — save accumulated rows
            if current_rows:
                tables.append(current_rows)
                current_rows = []
            continue

        # Split by cell markers
        cells = line.split("\x07")
        # Remove empty trailing entries from row-end markers
        cleaned = [c.strip() for c in cells]
        # Filter out empty strings but keep empty cells within a row
        # (row-end marker produces an extra empty entry at the end)
        if cleaned and cleaned[-1] == "":
            cleaned = cleaned[:-1]
        if cleaned:
            current_rows.append(cleaned)

    if current_rows:
        tables.append(current_rows)

    # Filter out "tables" with only 1 row or inconsistent column counts
    valid_tables: List[List[List[str]]] = []
    for table in tables:
        if len(table) < 2:
            continue
        valid_tables.append(table)

    return valid_tables


__all__ = [
    "parse_fib_text",
    "detect_tables_from_text",
    "PieceDescriptor",
]
