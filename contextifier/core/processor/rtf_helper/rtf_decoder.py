# contextifier/core/processor/rtf_helper/rtf_decoder.py
"""
RTF Decoding Utilities

Encoding detection and decoding functions for RTF content.
"""
import logging
import re
from typing import List

from contextifier.core.processor.rtf_helper.rtf_constants import (
    CODEPAGE_ENCODING_MAP,
    DEFAULT_ENCODINGS,
)

logger = logging.getLogger("contextify.rtf.decoder")


def detect_encoding(content: bytes, default_encoding: str = "cp949") -> str:
    """
    Detect encoding from RTF content.
    
    Looks for \\ansicpgXXXX pattern in the header.
    
    Args:
        content: RTF binary data
        default_encoding: Fallback encoding
        
    Returns:
        Detected encoding string
    """
    try:
        text = content[:1000].decode('ascii', errors='ignore')
        
        match = re.search(r'\\ansicpg(\d+)', text)
        if match:
            codepage = int(match.group(1))
            encoding = CODEPAGE_ENCODING_MAP.get(codepage, 'cp1252')
            logger.debug(f"RTF encoding detected: {encoding} (codepage {codepage})")
            return encoding
    except Exception as e:
        logger.debug(f"Encoding detection failed: {e}")
    
    return default_encoding


def decode_content(content: bytes, encoding: str = "cp949") -> str:
    """
    Decode RTF binary to string.
    
    Tries multiple encodings and returns first successful result.
    
    Args:
        content: RTF binary data
        encoding: Preferred encoding to try first
        
    Returns:
        Decoded string
    """
    encodings = [encoding] + [e for e in DEFAULT_ENCODINGS if e != encoding]
    
    for enc in encodings:
        try:
            return content.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    
    return content.decode('cp1252', errors='replace')


def decode_bytes(byte_list: List[int], encoding: str = "cp949") -> str:
    """
    Decode byte list to string.
    
    Args:
        byte_list: List of byte values
        encoding: Encoding to use
        
    Returns:
        Decoded string
    """
    try:
        return bytes(byte_list).decode(encoding)
    except (UnicodeDecodeError, LookupError):
        try:
            return bytes(byte_list).decode('cp949')
        except:
            return bytes(byte_list).decode('latin-1', errors='replace')


def decode_hex_escapes(text: str, encoding: str = "cp949") -> str:
    """
    Decode RTF hex escape sequences (\\'XX).
    
    Args:
        text: RTF text with hex escapes
        encoding: Encoding for decoding
        
    Returns:
        Decoded text
    """
    if "\\'" not in text:
        return text
    
    result = []
    byte_buffer = []
    i = 0
    n = len(text)
    
    while i < n:
        if i + 3 < n and text[i:i+2] == "\\'":
            try:
                hex_val = text[i+2:i+4]
                byte_val = int(hex_val, 16)
                byte_buffer.append(byte_val)
                i += 4
                continue
            except ValueError:
                pass
        
        # Flush byte buffer
        if byte_buffer:
            result.append(decode_bytes(byte_buffer, encoding))
            byte_buffer = []
        
        result.append(text[i])
        i += 1
    
    # Flush remaining bytes
    if byte_buffer:
        result.append(decode_bytes(byte_buffer, encoding))
    
    return ''.join(result)


__all__ = [
    'detect_encoding',
    'decode_content',
    'decode_bytes',
    'decode_hex_escapes',
]
