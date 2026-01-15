# service/document_processor/processor/hwp_helper/hwp_image.py
"""
HWP 이미지 처리 유틸리티

HWP 5.0 OLE 파일에서 이미지를 추출하고 MinIO에 업로드합니다.
- try_decompress_image: zlib 압축 이미지 해제
- find_bindata_stream: BinData 스트림 경로 찾기
- extract_bindata_index: SHAPE_COMPONENT_PICTURE에서 BinData 인덱스 추출
- extract_and_upload_image: 이미지 추출 및 MinIO 업로드
- process_images_from_bindata: BinData에서 모든 이미지 추출
"""
import io
import os
import zlib
import struct
import logging
import traceback
from typing import Optional, List, Dict, Set

import olefile
from PIL import Image

from libs.core.functions.utils import upload_image_to_minio as _upload_image_to_minio

logger = logging.getLogger("document-processor")


def try_decompress_image(data: bytes) -> bytes:
    """
    HWP 이미지 데이터 압축 해제를 시도합니다.

    HWP 파일에서 이미지가 zlib으로 압축되어 있을 수 있으므로,
    다양한 전략으로 압축 해제를 시도합니다.

    Args:
        data: 원본 이미지 데이터 (압축되었을 수 있음)

    Returns:
        압축 해제된 이미지 데이터 (또는 원본 데이터)
    """
    # 1. zlib 헤더가 있으면 zlib 압축 해제 시도
    if data.startswith(b'\x78'):
        try:
            return zlib.decompress(data)
        except Exception:
            pass

    # 2. 이미 유효한 이미지인지 확인
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return data  # 유효한 이미지
    except Exception:
        pass

    # 3. raw deflate (헤더 없음) 시도
    try:
        return zlib.decompress(data, -15)
    except Exception:
        pass

    return data


def upload_image_to_minio(image_data: bytes, app_db=None) -> Optional[str]:
    """
    이미지를 MinIO에 업로드합니다.

    Args:
        image_data: 이미지 바이너리 데이터
        app_db: 데이터베이스 연결 (메타데이터 저장용)

    Returns:
        업로드된 이미지의 MinIO 경로 또는 None
    """
    return _upload_image_to_minio(image_data, app_db=app_db, processed_images=None)


def find_bindata_stream(ole: olefile.OleFileIO, storage_id: int, ext: str) -> Optional[List[str]]:
    """
    OLE 컨테이너에서 storage_id와 확장자로 BinData 스트림을 찾습니다.

    Args:
        ole: OLE 파일 객체
        storage_id: BinData 스토리지 ID
        ext: 파일 확장자

    Returns:
        찾은 스트림 경로 또는 None
    """
    ole_dirs = ole.listdir()

    candidates = [
        f"BinData/BIN{storage_id:04X}.{ext}",
        f"BinData/BIN{storage_id:04x}.{ext}",
        f"BinData/Bin{storage_id:04X}.{ext}",
        f"BinData/Bin{storage_id:04x}.{ext}",
        f"BinData/BIN{storage_id:04X}.{ext.lower()}",
        f"BinData/BIN{storage_id:04x}.{ext.lower()}",
    ]

    # 패턴 매칭으로 찾기
    for entry in ole_dirs:
        if entry[0] == "BinData" and len(entry) > 1:
            fname = entry[1].lower()
            expected_patterns = [
                f"bin{storage_id:04x}",
                f"bin{storage_id:04X}",
            ]
            for pattern in expected_patterns:
                if pattern.lower() in fname.lower():
                    logger.debug(f"Found stream by pattern match: {entry}")
                    return entry

    # 정확한 경로 매칭
    for candidate in candidates:
        candidate_parts = candidate.split('/')
        if candidate_parts in ole_dirs:
            return candidate_parts

    # 대소문자 무시 매칭
    for entry in ole_dirs:
        if entry[0] == "BinData" and len(entry) > 1:
            fname = entry[1]
            for candidate in candidates:
                if fname.lower() == candidate.split('/')[-1].lower():
                    return entry

    return None


def extract_bindata_index(payload: bytes, bin_data_list_len: int) -> Optional[int]:
    """
    SHAPE_COMPONENT_PICTURE 레코드 payload에서 BinData 인덱스를 추출합니다.

    여러 HWP 버전 호환을 위해 다양한 오프셋 전략을 시도합니다.

    Args:
        payload: SHAPE_COMPONENT_PICTURE 레코드의 payload
        bin_data_list_len: bin_data_list의 길이 (유효 범위 검증용)

    Returns:
        BinData 인덱스 (1-based) 또는 None
    """
    if bin_data_list_len == 0:
        return None

    bindata_index = None

    # Strategy 1: 오프셋 79 (HWP 5.0.3.x+ 스펙)
    if len(payload) >= 81:
        test_id = struct.unpack('<H', payload[79:81])[0]
        if 0 < test_id <= bin_data_list_len:
            bindata_index = test_id
            logger.debug(f"Found BinData index at offset 79: {bindata_index}")
            return bindata_index

    # Strategy 2: 오프셋 8 (구 버전)
    if len(payload) >= 10:
        test_id = struct.unpack('<H', payload[8:10])[0]
        if 0 < test_id <= bin_data_list_len:
            bindata_index = test_id
            logger.debug(f"Found BinData index at offset 8: {bindata_index}")
            return bindata_index

    # Strategy 3: 일반적인 오프셋 스캔
    for offset in [4, 6, 10, 12, 14, 16, 18, 20, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]:
        if len(payload) >= offset + 2:
            test_id = struct.unpack('<H', payload[offset:offset+2])[0]
            if 0 < test_id <= bin_data_list_len:
                bindata_index = test_id
                logger.debug(f"Found potential BinData index at offset {offset}: {bindata_index}")
                return bindata_index

    # Strategy 4: 범위 내 첫 번째 non-zero 2바이트 값 스캔
    for i in range(0, min(len(payload) - 1, 100), 2):
        test_id = struct.unpack('<H', payload[i:i+2])[0]
        if 0 < test_id <= bin_data_list_len:
            bindata_index = test_id
            logger.debug(f"Found BinData index by scanning at offset {i}: {bindata_index}")
            return bindata_index

    return None


def extract_and_upload_image(
    ole: olefile.OleFileIO,
    target_stream: List[str],
    app_db=None,
    processed_images: Optional[Set[str]] = None
) -> Optional[str]:
    """
    OLE 스트림에서 이미지를 추출하여 MinIO에 업로드합니다.

    Args:
        ole: OLE 파일 객체
        target_stream: 스트림 경로
        app_db: 데이터베이스 연결
        processed_images: 처리된 이미지 경로 집합

    Returns:
        이미지 태그 문자열 또는 None
    """
    try:
        stream = ole.openstream(target_stream)
        image_data = stream.read()
        image_data = try_decompress_image(image_data)

        minio_path = upload_image_to_minio(image_data, app_db)
        if minio_path:
            if processed_images is not None:
                processed_images.add("/".join(target_stream))
            logger.info(f"Successfully extracted inline image: {minio_path}")
            return f"\n[image:{minio_path}]\n"
    except Exception as e:
        logger.warning(f"Failed to process inline HWP image {target_stream}: {e}")
        logger.debug(traceback.format_exc())

    return None


async def process_images_from_bindata(
    ole: olefile.OleFileIO,
    app_db=None,
    processed_images: Optional[Set[str]] = None
) -> str:
    """
    BinData 스토리지에서 이미지를 추출하여 MinIO에 업로드합니다.

    Args:
        ole: OLE 파일 객체
        app_db: 데이터베이스 연결
        processed_images: 이미 처리된 이미지 경로 집합 (스킵용)

    Returns:
        이미지 태그들을 결합한 문자열
    """
    results = []

    try:
        bindata_streams = [
            entry for entry in ole.listdir()
            if entry[0] == "BinData"
        ]

        for stream_path in bindata_streams:
            if processed_images and "/".join(stream_path) in processed_images:
                continue

            stream_name = stream_path[-1]
            ext = os.path.splitext(stream_name)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                stream = ole.openstream(stream_path)
                image_data = stream.read()
                image_data = try_decompress_image(image_data)

                minio_path = upload_image_to_minio(image_data, app_db=app_db)
                if minio_path:
                    results.append(f"[image:{minio_path}]")

    except Exception as e:
        logger.warning(f"Error processing HWP images: {e}")

    return "\n\n".join(results)


# 하위 호환성을 위한 클래스 래퍼
class ImageHelper:
    """HWP 이미지 처리 유틸리티 (하위 호환성)"""

    @staticmethod
    def try_decompress_image(data: bytes) -> bytes:
        return try_decompress_image(data)

    @staticmethod
    def upload_image_to_minio(image_data: bytes, app_db=None) -> Optional[str]:
        return upload_image_to_minio(image_data, app_db)


__all__ = [
    'try_decompress_image',
    'upload_image_to_minio',
    'find_bindata_stream',
    'extract_bindata_index',
    'extract_and_upload_image',
    'process_images_from_bindata',
    'ImageHelper',
]
