# contextifier/core/processor/hwp5_helper/hwp5_decoder.py
"""
HWP 5.0 Compression/Encoding Utilities

Provides decompression utilities for HWP 5.0 OLE streams.

HWP 5.0 uses zlib Deflate compression for:
- DocInfo stream
- BodyText/Section streams
- BinData streams (images, OLE objects)

Compression flag is stored in FileHeader stream at bytes 36-40.
"""
import zlib
import struct
import logging
from typing import Tuple

import olefile

logger = logging.getLogger("document-processor.HWP5")


def is_compressed(ole: olefile.OleFileIO) -> bool:
    """
    Check if HWP file streams are compressed.
    
    Reads FileHeader stream and checks compression flag at bytes 36-40.
    
    Args:
        ole: OLE file object
        
    Returns:
        True if compressed (default: True for most HWP files)
    """
    try:
        if ole.exists("FileHeader"):
            stream = ole.openstream("FileHeader")
            header = stream.read()
            if len(header) >= 40:
                flags = struct.unpack('<I', header[36:40])[0]
                return bool(flags & 0x01)
    except Exception as e:
        logger.debug(f"Failed to read FileHeader: {e}")
    return True  # Default: compressed


def decompress_stream(data: bytes, is_compressed_flag: bool = True) -> bytes:
    """
    Decompress stream data if necessary.
    
    HWP uses zlib Deflate algorithm. Tries raw deflate (-15) first,
    then standard zlib (with header).
    
    Args:
        data: Stream binary data
        is_compressed_flag: Whether data should be decompressed
        
    Returns:
        Decompressed data (or original if not compressed/failed)
    """
    if not is_compressed_flag:
        return data
    
    # Try raw deflate (no header) - most common in HWP
    try:
        return zlib.decompress(data, -15)
    except zlib.error:
        pass
    
    # Try standard zlib (with header)
    try:
        return zlib.decompress(data)
    except zlib.error:
        pass
    
    return data


def decompress_section(data: bytes) -> Tuple[bytes, bool]:
    """
    Decompress BodyText section data.
    
    Args:
        data: Section binary data
        
    Returns:
        Tuple of (decompressed_data, success_flag)
    """
    # Try raw deflate
    try:
        decompressed = zlib.decompress(data, -15)
        return decompressed, True
    except zlib.error:
        pass
    
    # Try standard zlib
    try:
        decompressed = zlib.decompress(data)
        return decompressed, True
    except zlib.error:
        pass
    
    return data, False


def decompress_bindata(data: bytes) -> bytes:
    """
    Decompress BinData stream (images, OLE objects).
    
    Args:
        data: BinData binary data
        
    Returns:
        Decompressed data
    """
    # Try raw deflate
    try:
        return zlib.decompress(data, -15)
    except zlib.error:
        pass
    
    # Try standard zlib
    try:
        return zlib.decompress(data)
    except zlib.error:
        pass
    
    # Return original if not compressed
    return data


def get_file_header_info(ole: olefile.OleFileIO) -> dict:
    """
    Parse FileHeader stream for file information.
    
    FileHeader structure (simplified):
    - 0-31: Signature "HWP Document File" + version
    - 32-35: Version (major.minor)
    - 36-39: Flags (compression, password, etc.)
    
    Args:
        ole: OLE file object
        
    Returns:
        Dictionary with file info
    """
    info = {
        'signature': None,
        'version': None,
        'compressed': True,
        'password_encrypted': False,
        'distributed': False,
    }
    
    try:
        if not ole.exists("FileHeader"):
            return info
            
        stream = ole.openstream("FileHeader")
        header = stream.read()
        
        if len(header) < 32:
            return info
        
        # Signature (bytes 0-31)
        info['signature'] = header[:32].rstrip(b'\x00').decode('utf-8', errors='ignore')
        
        if len(header) >= 36:
            # Version (bytes 32-35)
            version_bytes = header[32:36]
            if len(version_bytes) >= 4:
                # HWP uses different version encoding
                info['version'] = f"{version_bytes[3]}.{version_bytes[2]}.{version_bytes[1]}.{version_bytes[0]}"
        
        if len(header) >= 40:
            # Flags (bytes 36-39)
            flags = struct.unpack('<I', header[36:40])[0]
            info['compressed'] = bool(flags & 0x01)
            info['password_encrypted'] = bool(flags & 0x02)
            info['distributed'] = bool(flags & 0x04)
            
    except Exception as e:
        logger.debug(f"Failed to parse FileHeader: {e}")
    
    return info


__all__ = [
    'is_compressed',
    'decompress_stream',
    'decompress_section',
    'decompress_bindata',
    'get_file_header_info',
]
