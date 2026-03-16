# contextifier_new/handlers/doc/_constants.py
"""
Constants for DOC (OLE2 Compound Binary Format) handler.

DOC files use the OLE2/CFBF (Compound File Binary Format) container.
The actual text lives in the WordDocument stream, with supplementary
data in 0Table/1Table and other streams.
"""

from __future__ import annotations

# ── Magic bytes ───────────────────────────────────────────────────────────

# OLE2 Compound File signature (first 8 bytes)
OLE2_MAGIC: bytes = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Word 97-2003 FIB (File Information Block) magic numbers
# 0xA5EC = Word 97-2003 document
# 0xA5DC = Word 95 document (older)
WORD_FIB_MAGIC: frozenset[int] = frozenset({0xA5EC, 0xA5DC})

# ── OLE stream names ─────────────────────────────────────────────────────

# Main document text stream
WORD_DOCUMENT_STREAM: str = "WordDocument"

# Formatting / style table streams (one will exist)
TABLE_STREAM_NAMES: tuple[str, ...] = ("1Table", "0Table")

# Streams that may contain embedded images
IMAGE_STREAM_KEYWORDS: frozenset[str] = frozenset({
    "pictures",
    "data",
    "object",
    "oleobject",
    "objectpool",
})

# ── Image format detection ────────────────────────────────────────────────

# Binary signatures for detecting image format in OLE streams.
# Mapping: format_name → (header_bytes, min_data_length)
IMAGE_SIGNATURES: dict[str, tuple[bytes, int]] = {
    "png":  (b"\x89PNG\r\n\x1a\n", 8),
    "jpeg": (b"\xff\xd8",          2),
    "gif87": (b"GIF87a",           6),
    "gif89": (b"GIF89a",           6),
    "bmp":  (b"BM",                2),
    "tiff_le": (b"II\x2a\x00",    4),
    "tiff_be": (b"MM\x00\x2a",    4),
    "emf":  (b"\x01\x00\x00\x00", 4),  # EMF header (Enhanced Metafile)
}

# ── Encoding fallback list ────────────────────────────────────────────────

# Ordered list of encodings to try when decoding OLE string properties.
# Korean encodings are prioritised because DOC is heavily used in Korea.
OLE_STRING_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "cp949",
    "euc-kr",
    "cp1252",
    "latin-1",
)

# ── Unicode extraction ────────────────────────────────────────────────────

# Minimum character count for a UTF-16LE fragment to be considered valid text.
# Fragments shorter than this are likely control structures or noise.
MIN_TEXT_FRAGMENT_LENGTH: int = 4

# Minimum byte length for a UTF-16LE sequence to be collected (= 2 * MIN chars).
MIN_UNICODE_BYTES: int = MIN_TEXT_FRAGMENT_LENGTH * 2

# Unicode ranges for CJK / Korean characters (high byte of UTF-16LE pair)
# AC00-D7AF: Hangul Syllables
# 3000-4DFF: CJK Symbols, Hiragana, Katakana, Bopomofo, CJK Unified
CJK_HIGH_BYTE_RANGES: tuple[tuple[int, int], ...] = (
    (0xAC, 0xD7),    # Hangul Syllables
    (0x30, 0x4E),    # CJK range (partial)
)


__all__ = [
    "OLE2_MAGIC",
    "WORD_FIB_MAGIC",
    "WORD_DOCUMENT_STREAM",
    "TABLE_STREAM_NAMES",
    "IMAGE_STREAM_KEYWORDS",
    "IMAGE_SIGNATURES",
    "OLE_STRING_ENCODINGS",
    "MIN_TEXT_FRAGMENT_LENGTH",
    "MIN_UNICODE_BYTES",
    "CJK_HIGH_BYTE_RANGES",
]
