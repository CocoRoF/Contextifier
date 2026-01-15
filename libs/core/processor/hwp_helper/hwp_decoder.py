# service/document_processor/processor/hwp_helper/hwp_decoder.py
"""
HWP 압축/인코딩 유틸리티

HWP 5.0 OLE 파일의 스트림 압축 해제 및 관련 유틸리티를 제공합니다.
- is_compressed: FileHeader를 읽어 압축 여부 확인
- decompress_stream: zlib Deflate 압축 해제
"""
import zlib
import struct
import logging
from typing import Tuple

import olefile

logger = logging.getLogger("document-processor")


def is_compressed(ole: olefile.OleFileIO) -> bool:
    """
    FileHeader를 읽어 HWP 파일 스트림이 압축되어 있는지 확인합니다.
    
    HWP FileHeader의 36-40 바이트에 있는 플래그 필드를 읽어
    압축 비트(0x01)가 설정되어 있는지 확인합니다.
    
    Args:
        ole: OLE 파일 객체
        
    Returns:
        압축 여부 (기본값: True - 대부분의 HWP 파일은 압축됨)
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
    return True  # 기본값: 압축됨 (대부분의 HWP 파일)


def decompress_stream(data: bytes, is_compressed_flag: bool = True) -> bytes:
    """
    필요시 스트림 데이터를 압축 해제합니다.
    
    HWP는 zlib Deflate 알고리즘을 사용하며, raw deflate(-15)를 먼저 시도합니다.
    
    Args:
        data: 스트림 바이너리 데이터
        is_compressed_flag: 압축 여부 플래그
        
    Returns:
        압축 해제된 데이터 (또는 원본 데이터)
    """
    if not is_compressed_flag:
        return data
    
    # Raw deflate 시도 (헤더 없음)
    try:
        return zlib.decompress(data, -15)
    except zlib.error:
        pass
    
    # 표준 zlib 시도 (헤더 포함)
    try:
        return zlib.decompress(data)
    except zlib.error:
        pass
    
    return data


def decompress_section(data: bytes) -> Tuple[bytes, bool]:
    """
    BodyText 섹션 데이터를 압축 해제합니다.
    
    Args:
        data: 섹션 바이너리 데이터
        
    Returns:
        (압축 해제된 데이터, 성공 여부) 튜플
    """
    # Raw deflate 시도
    try:
        decompressed = zlib.decompress(data, -15)
        return decompressed, True
    except zlib.error:
        pass
    
    # 표준 zlib 시도
    try:
        decompressed = zlib.decompress(data)
        return decompressed, True
    except zlib.error:
        pass
    
    return data, False


__all__ = [
    'is_compressed',
    'decompress_stream',
    'decompress_section',
]
