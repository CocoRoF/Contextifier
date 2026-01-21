# your_package/document_processor/utils.py
"""
Common utility module for document processing
"""
import io
import os
import hashlib
import tempfile
import logging
import re
import bisect
from typing import Any, Dict, List, Optional, Set

from PIL import Image

def sanitize_text_for_json(text: Optional[str]) -> str:
    """
    Sanitizes text to be safely encodable in a UTF-8 JSON response.

    Removes or replaces the following characters:
    - Invalid surrogate pairs (U+D800-U+DFFF): removes isolated high/low surrogates
    - Private Use Area characters (U+E000-U+F8FF, U+F0000 and above): removed
    - Non-character code points (U+FFFE, U+FFFF): removed
    - Problematic control characters (except tab, newline, carriage return)

    Args:
        text: Input text that may contain invalid characters

    Returns:
        Sanitized text safe for JSON encoding
    """
    if not text:
        return text if text is not None else ""

    result = []
    i = 0
    text_len = len(text)

    while i < text_len:
        char = text[i]
        code = ord(char)

        # Check for surrogate pairs (\uD800-\uDFFF)
        if 0xD800 <= code <= 0xDFFF:
            # High surrogate (\uD800-\uDBFF)
            if 0xD800 <= code <= 0xDBFF:
                # Check if followed by a valid low surrogate
                if i + 1 < text_len:
                    next_code = ord(text[i + 1])
                    if 0xDC00 <= next_code <= 0xDFFF:
                        # Valid surrogate pair, calculate actual code point
                        full_code = 0x10000 + ((code - 0xD800) << 10) + (next_code - 0xDC00)
                        # Supplementary Private Use Area-A: U+F0000 ~ U+FFFFF
                        # Supplementary Private Use Area-B: U+100000 ~ U+10FFFF
                        if full_code >= 0xF0000:
                            # Skip Private Use Supplementary characters
                            i += 2
                            continue
                        else:
                            # Valid supplementary character, keep it
                            result.append(char)
                            result.append(text[i + 1])
                            i += 2
                            continue
                # Invalid isolated high surrogate, skip it
                i += 1
                continue
            else:
                # Low surrogate without high surrogate, skip it
                i += 1
                continue

        # Check Basic Private Use Area (U+E000 ~ U+F8FF)
        if 0xE000 <= code <= 0xF8FF:
            # Skip Private Use characters
            i += 1
            continue

        # Check for problematic control characters
        # Keep: \t (9), \n (10), \r (13), space (32) and above
        # Remove: \x00-\x08, \x0B, \x0C, \x0E-\x1F (except those above)
        if code < 32 and code not in (9, 10, 13):
            # Skip problematic control characters
            i += 1
            continue

        # Check for non-characters (U+FFFE, U+FFFF)
        if code in (0xFFFE, 0xFFFF):
            i += 1
            continue

        # Valid character, keep it
        result.append(char)
        i += 1

    return ''.join(result)


def clean_text(text: Optional[str]) -> str:
   if not text:
       return ""
   text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
   return text.strip()

def clean_code_text(text: str) -> str:
    if not text:
        return ""
    text = text.rstrip().replace('\t', '    ')
    return text

def is_text_quality_sufficient(text: Optional[str], min_chars: int = 500, min_word_ratio: float = 0.6) -> bool:
    try:
        if not text or len(text) < min_chars:
            return False
        word_chars = re.findall(r"[\w\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", text)
        ratio = len(word_chars) / max(1, len(text))
        return ratio >= min_word_ratio
    except Exception:
        return False

def find_chunk_position(chunk: str, full_text: str, start_pos: int = 0) -> int:
    try:
        pos = full_text.find(chunk, start_pos)
        if pos != -1:
            return pos
        lines = chunk.strip().split('\n')
        if lines and len(lines[0]) >= 10:
            first_line = lines[0].strip()
            pos = full_text.find(first_line, start_pos)
            if pos != -1:
                chunk_start = full_text.find(chunk[:50] if len(chunk) > 50 else chunk, pos)
                return chunk_start if chunk_start != -1 else pos
        if len(chunk.strip()) >= 10:
            start = chunk.strip()[:50]
            pos = full_text.find(start, start_pos)
            if pos != -1:
                return pos
        return -1
    except Exception:
        return -1

def build_line_starts(text: str) -> List[int]:
    try:
        starts = [0]
        for i, ch in enumerate(text):
            if ch == '\n' and i + 1 < len(text):
                starts.append(i + 1)
        return starts
    except Exception:
        return [0]

def pos_to_line(pos: int, line_starts: List[int]) -> int:
    try:
        if pos < 0:
            return 1
        idx = bisect.bisect_right(line_starts, pos) - 1
        return max(1, idx + 1)
    except Exception:
        return 1
