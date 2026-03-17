# contextifier_new/handlers/rtf/_decoder.py
"""
RTF Decoding Utilities — Encoding detection and string decoding.

Internal module shared by RtfPreprocessor, RtfMetadataExtractor,
and RtfContentExtractor.

Functions:
- detect_encoding: Scan \\ansicpgNNNN in RTF header → encoding name
- decode_content: Try multiple encodings to decode bytes → str
- decode_bytes: Decode byte list → str (for hex escape sequences)
- decode_hex_escapes: Process \\'XX hex escapes in RTF text

Ported from v1.0 rtf_decoder.py with type annotations added.
"""

from __future__ import annotations

import logging
import re
from typing import List

from contextifier_new.handlers.rtf._constants import (
    CODEPAGE_ENCODING_MAP,
    DEFAULT_ENCODINGS,
)

_logger = logging.getLogger("contextifier.rtf.decoder")


def detect_encoding(content: bytes, default_encoding: str = "cp949") -> str:
    """
    Detect encoding from RTF content header.

    Scans the first 1000 bytes for ``\\ansicpgNNNN`` pattern.
    Maps the codepage number to a Python encoding name.

    Args:
        content: RTF binary data.
        default_encoding: Fallback if no codepage is found.

    Returns:
        Python encoding name (e.g. "cp949", "utf-8").
    """
    try:
        # Only need to scan the header area
        header = content[:1000].decode("ascii", errors="ignore")
        match = re.search(r"\\ansicpg(\d+)", header)
        if match:
            codepage = int(match.group(1))
            encoding = CODEPAGE_ENCODING_MAP.get(codepage, "cp1252")
            _logger.debug("RTF encoding: %s (codepage %d)", encoding, codepage)
            return encoding
    except Exception as e:
        _logger.debug("Encoding detection failed: %s", e)

    return default_encoding


def decode_content(content: bytes, encoding: str = "cp949") -> str:
    """
    Decode RTF binary data to string.

    Tries the given encoding first, then falls through the default
    encoding list. Returns replacement-error decoded text as last resort.

    Args:
        content: RTF binary data.
        encoding: Preferred encoding.

    Returns:
        Decoded string.
    """
    encodings = [encoding] + [e for e in DEFAULT_ENCODINGS if e != encoding]

    for enc in encodings:
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    return content.decode("cp1252", errors="replace")


def decode_bytes(byte_list: List[int], encoding: str = "cp949") -> str:
    """
    Decode a list of byte values to a string.

    Used for processing RTF hex escape sequences (\\' XX)
    where adjacent hex bytes form multi-byte characters.

    Args:
        byte_list: List of integer byte values (0–255).
        encoding: Encoding to use.

    Returns:
        Decoded string.
    """
    try:
        return bytes(byte_list).decode(encoding)
    except (UnicodeDecodeError, LookupError):
        try:
            return bytes(byte_list).decode("cp949")
        except Exception:
            return bytes(byte_list).decode("latin-1", errors="replace")


def decode_hex_escapes(text: str, encoding: str = "cp949") -> str:
    """
    Decode RTF hex escape sequences (\\' XX) in text.

    RTF uses ``\\'XX`` to represent non-ASCII bytes. Adjacent
    hex escapes may form multi-byte characters (e.g. Korean CP949
    uses 2-byte sequences).

    This function buffers consecutive hex escapes and decodes them
    together, supporting multi-byte encodings correctly.

    Args:
        text: RTF text containing hex escapes.
        encoding: Encoding for decoding bytes.

    Returns:
        Text with hex escapes decoded to characters.
    """
    if "\\'" not in text:
        return text

    result: List[str] = []
    byte_buffer: List[int] = []
    i = 0
    n = len(text)

    while i < n:
        if i + 3 < n and text[i : i + 2] == "\\'":
            try:
                hex_val = text[i + 2 : i + 4]
                byte_val = int(hex_val, 16)
                byte_buffer.append(byte_val)
                i += 4
                continue
            except ValueError:
                pass

        # Flush byte buffer when non-hex-escape encountered
        if byte_buffer:
            result.append(decode_bytes(byte_buffer, encoding))
            byte_buffer = []

        result.append(text[i])
        i += 1

    # Flush remaining bytes
    if byte_buffer:
        result.append(decode_bytes(byte_buffer, encoding))

    return "".join(result)


__all__ = [
    "detect_encoding",
    "decode_content",
    "decode_bytes",
    "decode_hex_escapes",
]
