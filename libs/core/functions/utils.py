# your_package/document_processor/utils.py
"""
문서 처리 공통 유틸리티 모듈
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
    텍스트를 UTF-8 JSON 응답에 안전하게 인코딩 가능하도록 정제합니다.

    다음 문자들을 제거하거나 교체합니다:
    - 잘못된 서로게이트 쌍 (U+D800-U+DFFF): 단독 high/low surrogate 제거
    - Private Use Area 문자 (U+E000-U+F8FF, U+F0000 이상): 제거
    - 비문자 코드 포인트 (U+FFFE, U+FFFF): 제거
    - 문제를 일으킬 수 있는 제어 문자 (탭, 개행, 캐리지 리턴 제외)

    Args:
        text: 잘못된 문자가 포함될 수 있는 입력 텍스트

    Returns:
        JSON 인코딩에 안전한 정제된 텍스트
    """
    if not text:
        return text if text is not None else ""

    result = []
    i = 0
    text_len = len(text)

    while i < text_len:
        char = text[i]
        code = ord(char)

        # 서로게이트 쌍 체크 (\uD800-\uDFFF)
        if 0xD800 <= code <= 0xDFFF:
            # High surrogate (\uD800-\uDBFF)
            if 0xD800 <= code <= 0xDBFF:
                # 유효한 low surrogate가 뒤따르는지 확인
                if i + 1 < text_len:
                    next_code = ord(text[i + 1])
                    if 0xDC00 <= next_code <= 0xDFFF:
                        # 유효한 서로게이트 쌍, 실제 코드 포인트 계산
                        full_code = 0x10000 + ((code - 0xD800) << 10) + (next_code - 0xDC00)
                        # Supplementary Private Use Area-A: U+F0000 ~ U+FFFFF
                        # Supplementary Private Use Area-B: U+100000 ~ U+10FFFF
                        if full_code >= 0xF0000:
                            # Private Use Supplementary 문자 건너뛰기
                            i += 2
                            continue
                        else:
                            # 유효한 supplementary 문자, 유지
                            result.append(char)
                            result.append(text[i + 1])
                            i += 2
                            continue
                # 잘못된 단독 high surrogate, 건너뛰기
                i += 1
                continue
            else:
                # High surrogate 없는 low surrogate, 건너뛰기
                i += 1
                continue

        # Basic Private Use Area 체크 (U+E000 ~ U+F8FF)
        if 0xE000 <= code <= 0xF8FF:
            # Private Use 문자 건너뛰기
            i += 1
            continue

        # 문제가 될 수 있는 제어 문자 체크
        # 유지: \t (9), \n (10), \r (13), 공백 (32) 이후
        # 제거: \x00-\x08, \x0B, \x0C, \x0E-\x1F (위 예외 제외)
        if code < 32 and code not in (9, 10, 13):
            # 문제 있는 제어 문자 건너뛰기
            i += 1
            continue

        # 비문자 체크 (U+FFFE, U+FFFF)
        if code in (0xFFFE, 0xFFFF):
            i += 1
            continue

        # 유효한 문자, 유지
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
