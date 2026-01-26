# contextifier/core/processor/hwp5_helper/hwp5_recovery.py
"""
HWP 5.0 File Recovery Utilities

Provides forensic recovery for corrupted or non-standard HWP files:
- extract_text_from_stream_raw: Extract UTF-16LE strings from binary
- find_zlib_streams: Find and decompress zlib chunks
- recover_images_from_raw: Extract images by signature scanning
- check_file_signature: Identify file type by signature
"""
import zlib
import struct
import logging
from typing import List, Tuple, Optional

from contextifier.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor.HWP5")


def extract_text_from_stream_raw(data: bytes) -> str:
    """
    Fallback text extraction from binary data without record parsing.
    
    Extracts valid UTF-16LE characters including:
    - Korean syllables (0xAC00-0xD7A3)
    - ASCII printable (0x0020-0x007E)
    - Korean Jamo (0x1100-0x11FF, 0x3130-0x318F)
    - CJK punctuation (0x3000-0x303F)
    - Control chars (newline, tab)
    
    Args:
        data: Binary data
        
    Returns:
        Extracted text string
    """
    text_parts = []
    current_run = []

    for i in range(0, len(data) - 1, 2):
        chunk = data[i:i+2]
        val = struct.unpack('<H', chunk)[0]

        is_valid = (
            (0xAC00 <= val <= 0xD7A3) or   # Korean syllables
            (0x0020 <= val <= 0x007E) or   # ASCII printable
            (0x3130 <= val <= 0x318F) or   # Korean compatibility Jamo
            (0x1100 <= val <= 0x11FF) or   # Korean Jamo
            (0x3000 <= val <= 0x303F) or   # CJK punctuation
            val in [10, 13, 9]              # LF, CR, Tab
        )

        if is_valid:
            if val in [10, 13]:
                if current_run:
                    text_parts.append("".join(current_run))
                    current_run = []
                text_parts.append("\n")
            elif val == 9:
                current_run.append("\t")
            else:
                current_run.append(chr(val))
        else:
            if len(current_run) > 0:
                text_parts.append("".join(current_run))
            current_run = []

    if current_run:
        text_parts.append("".join(current_run))

    final_parts = [p for p in text_parts if len(p.strip()) > 0]
    return "".join(final_parts)


def find_zlib_streams(raw_data: bytes, min_size: int = 50) -> List[Tuple[int, bytes]]:
    """
    Find and decompress zlib streams in binary data.
    
    Scans for zlib headers (0x78 0x9c, 0x78 0x01, 0x78 0xda)
    and attempts decompression.
    
    Args:
        raw_data: Binary data
        min_size: Minimum decompressed size to consider valid
        
    Returns:
        List of (offset, decompressed_data) tuples
    """
    zlib_headers = [b'\x78\x9c', b'\x78\x01', b'\x78\xda']

    decompressed_chunks = []
    start = 0
    file_len = len(raw_data)

    while start < file_len:
        next_header_pos = -1

        for h in zlib_headers:
            pos = raw_data.find(h, start)
            if pos != -1:
                if next_header_pos == -1 or pos < next_header_pos:
                    next_header_pos = pos

        if next_header_pos == -1:
            break

        start = next_header_pos

        try:
            dobj = zlib.decompressobj()
            decompressed = dobj.decompress(raw_data[start:])

            if len(decompressed) > min_size:
                decompressed_chunks.append((start, decompressed))

            if dobj.unused_data:
                compressed_size = len(raw_data[start:]) - len(dobj.unused_data)
                start += compressed_size
            else:
                start += 1

        except (zlib.error, Exception):
            start += 1

    return decompressed_chunks


def recover_images_from_raw(
    raw_data: bytes,
    image_processor: ImageProcessor
) -> str:
    """
    Extract images from raw binary data by signature scanning.
    
    Scans for JPEG (FFD8FF) and PNG (89504E47) signatures.
    
    Args:
        raw_data: Binary data
        image_processor: ImageProcessor instance for saving
        
    Returns:
        Combined image tags string
    """
    results = []

    # JPEG extraction
    start = 0
    while True:
        start = raw_data.find(b'\xff\xd8\xff', start)
        if start == -1:
            break

        end = raw_data.find(b'\xff\xd9', start)
        if end == -1:
            break

        end += 2

        size = end - start
        if 100 < size < 10 * 1024 * 1024:  # 100B to 10MB
            img_data = raw_data[start:end]

            image_tag = image_processor.save_image(img_data)
            if image_tag:
                results.append(image_tag)

        start = end

    # PNG extraction
    png_sig = b'\x89PNG\r\n\x1a\n'
    png_end = b'IEND\xae\x42\x60\x82'

    start = 0
    while True:
        start = raw_data.find(png_sig, start)
        if start == -1:
            break

        end = raw_data.find(png_end, start)
        if end == -1:
            break

        end += len(png_end)

        size = end - start
        if 100 < size < 10 * 1024 * 1024:
            img_data = raw_data[start:end]

            image_tag = image_processor.save_image(img_data)
            if image_tag:
                results.append(image_tag)

        start = end

    return "\n\n".join(results)


def check_file_signature(raw_data: bytes) -> Optional[str]:
    """
    Check file signature to identify file type.
    
    Args:
        raw_data: File binary data
        
    Returns:
        File type string or None if unknown
    """
    if len(raw_data) < 32:
        return None
    
    # OLE Compound Document (HWP 5.0)
    if raw_data[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return "OLE"
    
    # HWP 3.0/2.0 Legacy
    if b'HWP Document File V3' in raw_data[:32]:
        return "HWP3.0"
    
    if b'HWP Document File V2' in raw_data[:32]:
        return "HWP2.0"
    
    if b'HWP Document File' in raw_data[:32]:
        return "HWP_LEGACY"
    
    # ZIP (HWPX)
    if raw_data[:4] == b'PK\x03\x04':
        return "ZIP"
    
    return None


__all__ = [
    'extract_text_from_stream_raw',
    'find_zlib_streams',
    'recover_images_from_raw',
    'check_file_signature',
]
