# service/document_processor/processor/doc_helpers/rtf_decoder.py
"""
RTF 디코딩 유틸리티

RTF 인코딩 감지 및 디코딩 관련 함수들을 제공합니다.
"""
import logging
import re
from typing import List, Tuple

from libs.core.processor.doc_helpers.rtf_constants import (
    CODEPAGE_ENCODING_MAP,
    DEFAULT_ENCODINGS,
)

logger = logging.getLogger("document-processor")


def detect_encoding(content: bytes, default_encoding: str = "cp949") -> str:
    """
    RTF 콘텐츠에서 인코딩을 감지합니다.

    Args:
        content: RTF 바이트 데이터
        default_encoding: 기본 인코딩

    Returns:
        감지된 인코딩 문자열
    """
    try:
        text = content[:1000].decode('ascii', errors='ignore')

        # \ansicpgXXXX 패턴 찾기
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
    RTF 바이너리를 문자열로 디코딩합니다.

    여러 인코딩을 시도하여 성공하는 첫 번째 결과를 반환합니다.

    Args:
        content: RTF 바이트 데이터
        encoding: 우선 시도할 인코딩

    Returns:
        디코딩된 문자열
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
    바이트 리스트를 문자열로 디코딩합니다.

    Args:
        byte_list: 바이트 값 리스트
        encoding: 사용할 인코딩

    Returns:
        디코딩된 문자열
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
    RTF hex escape (\'XX) 시퀀스를 디코딩합니다.

    Args:
        text: RTF 텍스트
        encoding: 사용할 인코딩

    Returns:
        디코딩된 텍스트
    """
    result = []
    byte_buffer = []
    i = 0

    while i < len(text):
        if text[i:i+2] == "\\'":
            # hex escape 발견
            try:
                hex_val = text[i+2:i+4]
                byte_val = int(hex_val, 16)
                byte_buffer.append(byte_val)
                i += 4
            except (ValueError, IndexError):
                # 잘못된 escape, 그대로 추가
                if byte_buffer:
                    result.append(decode_bytes(byte_buffer, encoding))
                    byte_buffer = []
                result.append(text[i])
                i += 1
        else:
            # 일반 문자
            if byte_buffer:
                result.append(decode_bytes(byte_buffer, encoding))
                byte_buffer = []
            result.append(text[i])
            i += 1

    # 남은 바이트 처리
    if byte_buffer:
        result.append(decode_bytes(byte_buffer, encoding))

    return ''.join(result)
