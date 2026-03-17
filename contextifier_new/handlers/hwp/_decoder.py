# contextifier_new/handlers/hwp/_decoder.py
"""
HWP 5.0 compression / decompression utilities.

HWP files may store their body streams (BodyText/Section*, DocInfo)
in raw-deflate compressed form.  Whether compression is enabled is
indicated by a flag bit in the ``FileHeader`` OLE stream.

Public API:
    is_compressed(ole) -> bool
    decompress_stream(data, compressed) -> bytes
    decompress_section(data) -> tuple[bytes, bool]
"""

from __future__ import annotations

import struct
import zlib
import logging
from typing import Tuple

import olefile

from contextifier_new.handlers.hwp._constants import (
    COMPRESS_FLAG,
    FILE_HEADER_FLAGS_OFFSET,
    STREAM_FILE_HEADER,
)

logger = logging.getLogger(__name__)


def is_compressed(ole: olefile.OleFileIO) -> bool:
    """
    Check whether the HWP file uses compression.

    Reads the ``FileHeader`` stream and inspects the flags dword
    at byte offset 36.  Bit 0 indicates compression.

    Returns ``False`` if the stream is missing or unreadable.
    """
    try:
        if not ole.exists(STREAM_FILE_HEADER):
            return False
        stream = ole.openstream(STREAM_FILE_HEADER)
        header = stream.read()
        if len(header) < FILE_HEADER_FLAGS_OFFSET + 4:
            return False
        flags = struct.unpack_from(
            "<I", header, FILE_HEADER_FLAGS_OFFSET
        )[0]
        return bool(flags & COMPRESS_FLAG)
    except Exception:
        return False


def decompress_stream(data: bytes, compressed: bool) -> bytes:
    """
    Decompress a HWP OLE stream payload.

    If *compressed* is ``False`` the data is returned as-is.
    Otherwise tries raw-deflate first (``wbits=-15``), then
    standard zlib.

    Args:
        data: Raw stream bytes.
        compressed: Whether to attempt decompression.

    Returns:
        Decompressed (or original) bytes.
    """
    if not compressed or not data:
        return data

    # Strategy 1: raw deflate (most common in HWP)
    try:
        return zlib.decompress(data, -15)
    except zlib.error:
        pass

    # Strategy 2: standard zlib
    try:
        return zlib.decompress(data)
    except zlib.error:
        pass

    # Give up — return original bytes
    logger.debug("Decompression failed; returning raw data")
    return data


def decompress_section(data: bytes) -> Tuple[bytes, bool]:
    """
    Attempt to decompress a BodyText section payload.

    Tries raw-deflate, then standard zlib.  If both fail the
    original data is returned.

    Returns:
        (decompressed_bytes, success_flag)
    """
    if not data:
        return data, False

    try:
        return zlib.decompress(data, -15), True
    except zlib.error:
        pass

    try:
        return zlib.decompress(data), True
    except zlib.error:
        pass

    # Might be uncompressed already
    return data, True


__all__ = ["is_compressed", "decompress_stream", "decompress_section"]
