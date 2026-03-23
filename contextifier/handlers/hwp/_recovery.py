# contextifier/handlers/hwp/_recovery.py
"""
Forensic-recovery utilities for corrupt / non-standard HWP files.

When the standard OLE-based pipeline cannot process a file, these
routines attempt to salvage text and images by scanning raw bytes.

Public API:
    extract_text_raw(data) -> str
    find_zlib_streams(data, min_size) -> list[(offset, bytes)]
    check_file_signature(data) -> str | None
"""

from __future__ import annotations

import struct
import zlib
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_text_raw(data: bytes) -> str:
    """
    Best-effort extraction of UTF-16LE text from raw binary data.

    Scans for valid Korean syllables (U+AC00-U+D7A3), ASCII printable
    characters, hangul jamo, CJK punctuation, and whitespace.
    """
    parts: List[str] = []
    run: List[str] = []

    for i in range(0, len(data) - 1, 2):
        val = struct.unpack_from("<H", data, i)[0]

        is_valid = (
            (0xAC00 <= val <= 0xD7A3)   # Korean syllables
            or (0x0020 <= val <= 0x007E) # ASCII printable
            or (0x3130 <= val <= 0x318F) # Hangul compat jamo
            or (0x1100 <= val <= 0x11FF) # Hangul jamo
            or (0x3000 <= val <= 0x303F) # CJK punctuation
            or val in (10, 13, 9)        # LF, CR, TAB
        )

        if is_valid:
            if val in (10, 13):
                if run:
                    parts.append("".join(run))
                    run = []
                parts.append("\n")
            elif val == 9:
                run.append("\t")
            else:
                run.append(chr(val))
        else:
            if run:
                parts.append("".join(run))
                run = []

    if run:
        parts.append("".join(run))

    return "".join(p for p in parts if p.strip())


def find_zlib_streams(
    data: bytes, min_size: int = 50
) -> List[Tuple[int, bytes]]:
    """
    Scan raw data for zlib-compressed streams and decompress them.

    Returns a list of ``(offset, decompressed_data)`` tuples for
    streams whose decompressed size exceeds *min_size*.
    """
    headers = (b"\x78\x9c", b"\x78\x01", b"\x78\xda")
    results: List[Tuple[int, bytes]] = []
    pos = 0
    length = len(data)

    while pos < length:
        # Find nearest zlib header
        best = -1
        for h in headers:
            idx = data.find(h, pos)
            if idx != -1 and (best == -1 or idx < best):
                best = idx
        if best == -1:
            break

        pos = best
        try:
            dobj = zlib.decompressobj()
            decompressed = dobj.decompress(data[pos:])
            if len(decompressed) > min_size:
                results.append((pos, decompressed))
            if dobj.unused_data:
                consumed = len(data[pos:]) - len(dobj.unused_data)
                pos += consumed
            else:
                pos += 1
        except (zlib.error, Exception):
            pos += 1

    return results


def check_file_signature(data: bytes) -> Optional[str]:
    """
    Identify the file format by magic bytes.

    Returns ``"OLE"``, ``"ZIP/HWPX"``, ``"HWP3.0"``, or ``None``.
    """
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "OLE"
    if data[:4] == b"PK\x03\x04":
        return "ZIP/HWPX"
    if b"HWP Document File" in data[:100]:
        return "HWP3.0"
    return None


__all__ = ["extract_text_raw", "find_zlib_streams", "check_file_signature"]
