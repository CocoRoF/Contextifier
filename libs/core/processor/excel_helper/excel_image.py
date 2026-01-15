"""
XLSX 이미지 추출 모듈

Excel 파일에서 임베디드 이미지를 추출합니다.
"""

import os
import logging
import zipfile
from typing import Dict, List, Tuple

logger = logging.getLogger("document-processor")

# PIL에서 지원하는 이미지 형식만 추출
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']

# 지원하지 않는 형식 (EMF, WMF 등)
UNSUPPORTED_IMAGE_EXTENSIONS = ['.emf', '.wmf']


def extract_images_from_xlsx(file_path: str) -> Dict[str, bytes]:
    """
    XLSX 파일에서 이미지를 추출합니다 (ZIP 직접 접근).
    EMF, WMF 등 PIL에서 지원하지 않는 형식은 제외합니다.

    Args:
        file_path: XLSX 파일 경로

    Returns:
        {이미지 경로: 이미지 바이트} 딕셔너리
    """
    images = {}

    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            for name in zf.namelist():
                if name.startswith('xl/media/'):
                    # 이미지 파일
                    ext = os.path.splitext(name)[1].lower()
                    if ext in SUPPORTED_IMAGE_EXTENSIONS:
                        images[name] = zf.read(name)
                    elif ext in UNSUPPORTED_IMAGE_EXTENSIONS:
                        logger.debug(f"Skipping unsupported image format: {name}")

        return images

    except Exception as e:
        logger.warning(f"Error extracting images from XLSX: {e}")
        return {}


def get_sheet_images(ws, images_data: Dict[str, bytes], file_path: str) -> List[Tuple[bytes, str]]:
    """
    시트에 포함된 이미지를 가져옵니다.

    Args:
        ws: openpyxl Worksheet 객체
        images_data: extract_images_from_xlsx에서 추출한 이미지 딕셔너리
        file_path: XLSX 파일 경로

    Returns:
        [(이미지 바이트, 앵커 정보)] 리스트
    """
    result = []

    try:
        # openpyxl의 _images 속성 사용
        if hasattr(ws, '_images') and ws._images:
            for img in ws._images:
                try:
                    # 이미지 데이터 접근
                    if hasattr(img, '_data') and callable(img._data):
                        img_data = img._data()
                        anchor = str(img.anchor) if hasattr(img, 'anchor') else ""
                        result.append((img_data, anchor))
                except Exception as e:
                    logger.debug(f"Error accessing image data: {e}")

        # 직접 추출한 이미지 사용 (위에서 못 가져온 경우)
        if not result and images_data:
            for name, data in images_data.items():
                result.append((data, name))

        return result

    except Exception as e:
        logger.warning(f"Error getting sheet images: {e}")
        return []
