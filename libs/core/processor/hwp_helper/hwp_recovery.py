# service/document_processor/processor/hwp_helper/hwp_recovery.py
"""
HWP 손상 파일 복구 유틸리티

손상되었거나 비-OLE HWP 파일에서 텍스트와 이미지를 복구합니다.
- extract_text_from_stream_raw: 바이너리에서 UTF-16LE 문자열 추출
- recover_images_from_raw: 이미지 시그니처 스캔 후 추출
- find_zlib_streams: zlib 압축 스트림 찾기 및 해제
"""
import zlib
import struct
import logging
from typing import List, Tuple, Optional

from libs.core.functions.img_processor import ImageProcessor

# 모듈 레벨 이미지 프로세서
_image_processor = ImageProcessor(
    directory_path="temp/images",
    tag_prefix="[image:",
    tag_suffix="]"
)

logger = logging.getLogger("document-processor")


def extract_text_from_stream_raw(data: bytes) -> str:
    """
    Fallback: 레코드 파싱 없이 바이너리 데이터에서 UTF-16LE 문자열을 추출합니다.

    한글 완성형(0xAC00-0xD7A3), ASCII 인쇄 가능 문자, 한글 자모,
    CJK 구두점 등 유효한 문자만 추출합니다.

    Args:
        data: 바이너리 데이터

    Returns:
        추출된 텍스트 문자열
    """
    text_parts = []
    current_run = []

    for i in range(0, len(data) - 1, 2):
        chunk = data[i:i+2]
        val = struct.unpack('<H', chunk)[0]

        is_valid = (
            (0xAC00 <= val <= 0xD7A3) or   # 한글 완성형
            (0x0020 <= val <= 0x007E) or   # ASCII 인쇄 가능
            (0x3130 <= val <= 0x318F) or   # 한글 호환 자모
            (0x1100 <= val <= 0x11FF) or   # 한글 자모
            (0x3000 <= val <= 0x303F) or   # CJK 구두점
            val in [10, 13, 9]              # 줄바꿈, 탭
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
    바이너리 데이터에서 zlib 압축 스트림을 찾아 압축 해제합니다.

    zlib 헤더(0x78 0x9c, 0x78 0x01, 0x78 0xda)를 스캔하고
    압축 해제를 시도합니다.

    Args:
        raw_data: 바이너리 데이터
        min_size: 유효한 스트림으로 인정할 최소 압축 해제 크기

    Returns:
        (시작 오프셋, 압축 해제된 데이터) 튜플 리스트
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


def recover_images_from_raw(raw_data: bytes) -> str:
    """
    raw 바이너리 데이터에서 이미지 시그니처(JPEG, PNG)를 스캔하여 로컬에 저장합니다.

    Args:
        raw_data: 바이너리 데이터

    Returns:
        이미지 태그들을 결합한 문자열
    """
    results = []

    # JPEG 추출
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
        if 100 < size < 10 * 1024 * 1024:
            img_data = raw_data[start:end]

            image_tag = _image_processor.save_image(img_data)
            if image_tag:
                results.append(image_tag)

        start = end

    # PNG 추출
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

            image_tag = _image_processor.save_image(img_data)
            if image_tag:
                results.append(image_tag)

        start = end

    return "\n\n".join(results)


def check_file_signature(raw_data: bytes) -> Optional[str]:
    """
    파일 시그니처를 확인하여 파일 형식을 식별합니다.

    Args:
        raw_data: 바이너리 데이터

    Returns:
        파일 형식 문자열 또는 None
    """
    if raw_data.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'):
        return "OLE"
    elif raw_data.startswith(b'PK\x03\x04'):
        return "ZIP/HWPX"
    elif b'HWP Document File' in raw_data[:100]:
        return "HWP3.0"
    return None


__all__ = [
    'extract_text_from_stream_raw',
    'find_zlib_streams',
    'recover_images_from_raw',
    'check_file_signature',
]
