# hwpx_helper/hwpx_image.py
"""
HWPX 이미지 처리

HWPX 문서의 이미지를 추출하고 MinIO에 업로드합니다.
"""
import logging
import os
import zipfile
from typing import List

from .hwpx_constants import SUPPORTED_IMAGE_EXTENSIONS

# hwp_helper에서 ImageHelper import
try:
    from ..hwp_helper import ImageHelper
    IMAGE_HELPER_AVAILABLE = True
except ImportError:
    IMAGE_HELPER_AVAILABLE = False

logger = logging.getLogger("document-processor")


async def process_hwpx_images(
    zf: zipfile.ZipFile,
    image_files: List[str],
    app_db=None
) -> str:
    """
    HWPX zip에서 이미지를 추출하고 MinIO에 업로드합니다.

    Args:
        zf: 열린 ZipFile 객체
        image_files: 처리할 이미지 파일 경로 목록
        app_db: 데이터베이스 연결

    Returns:
        이미지 태그 문자열들을 줄바꿈으로 연결한 결과
    """
    if not IMAGE_HELPER_AVAILABLE:
        logger.warning("ImageHelper not available, skipping image processing")
        return ""

    results = []

    for img_path in image_files:
        ext = os.path.splitext(img_path)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            try:
                with zf.open(img_path) as f:
                    image_data = f.read()

                minio_path = ImageHelper.upload_image_to_minio(image_data, app_db=app_db)
                if minio_path:
                    results.append(f"[image:{minio_path}]")

            except Exception as e:
                logger.warning(f"Error processing HWPX image {img_path}: {e}")

    return "\n\n".join(results)


def get_remaining_images(
    zf: zipfile.ZipFile,
    processed_images: set
) -> List[str]:
    """
    아직 처리되지 않은 이미지 파일 목록을 반환합니다.

    Args:
        zf: 열린 ZipFile 객체
        processed_images: 이미 처리된 이미지 경로 집합

    Returns:
        처리되지 않은 이미지 파일 경로 목록
    """
    image_files = [
        f for f in zf.namelist()
        if f.startswith("BinData/") and not f.endswith("/")
    ]

    remaining_images = []
    for img in image_files:
        if img not in processed_images:
            remaining_images.append(img)

    return remaining_images
