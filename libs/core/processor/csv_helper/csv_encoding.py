# csv_helper/csv_encoding.py
"""
CSV 인코딩 감지 및 파일 읽기

파일의 인코딩을 자동 감지하고 올바르게 디코딩합니다.
BOM 감지, chardet 라이브러리, 휴리스틱 방식을 사용합니다.
"""
import logging
from typing import Optional, Tuple
import chardet

from libs.core.processor.csv_helper.csv_constants import ENCODING_CANDIDATES

logger = logging.getLogger("document-processor")


def detect_bom(data: bytes) -> Optional[str]:
    """
    BOM(Byte Order Mark)을 감지합니다.

    Args:
        data: 파일의 바이너리 데이터

    Returns:
        감지된 인코딩 또는 None
    """
    if data.startswith(b'\xef\xbb\xbf'):
        return 'utf-8-sig'
    elif data.startswith(b'\xff\xfe\x00\x00'):
        return 'utf-32-le'
    elif data.startswith(b'\x00\x00\xfe\xff'):
        return 'utf-32-be'
    elif data.startswith(b'\xff\xfe'):
        return 'utf-16-le'
    elif data.startswith(b'\xfe\xff'):
        return 'utf-16-be'
    return None


def read_file_with_encoding(
    file_path: str,
    preferred_encoding: str = None
) -> Tuple[str, str]:
    """
    파일을 읽고 인코딩을 자동 감지합니다.

    감지 순서:
    1. BOM 확인
    2. 선호 인코딩 시도 (지정된 경우)
    3. chardet 라이브러리 사용 (가능한 경우)
    4. 인코딩 후보 목록 순차 시도
    5. latin-1 폴백 (항상 성공)

    Args:
        file_path: 파일 경로
        preferred_encoding: 선호 인코딩 (None이면 자동 감지)

    Returns:
        (content, detected_encoding) 튜플
    """
    # 바이너리로 먼저 읽기
    with open(file_path, mode='rb') as f:
        raw_data = f.read()

    # BOM 확인
    bom_encoding = detect_bom(raw_data)
    if bom_encoding:
        logger.debug(f"BOM detected: {bom_encoding}")
        try:
            return raw_data.decode(bom_encoding), bom_encoding
        except UnicodeDecodeError:
            pass

    # 선호 인코딩 시도
    if preferred_encoding:
        try:
            return raw_data.decode(preferred_encoding), preferred_encoding
        except UnicodeDecodeError:
            logger.debug(f"Preferred encoding {preferred_encoding} failed")

    # chardet 사용
    detected = chardet.detect(raw_data[:10000])  # 처음 10KB만 분석
    if detected and detected.get('encoding'):
        detected_enc = detected['encoding']
        confidence = detected.get('confidence', 0)
        logger.debug(f"chardet detected: {detected_enc} (confidence: {confidence})")

        if confidence > 0.7:
            try:
                return raw_data.decode(detected_enc), detected_enc
            except UnicodeDecodeError:
                pass

    # 인코딩 후보 시도
    for enc in ENCODING_CANDIDATES:
        try:
            content = raw_data.decode(enc)
            logger.debug(f"Successfully decoded with: {enc}")
            return content, enc
        except UnicodeDecodeError:
            continue

    # 최후의 수단: latin-1 (항상 성공)
    return raw_data.decode('latin-1', errors='replace'), 'latin-1'
