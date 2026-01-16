# hwpx_helper/hwpx_image.py
"""
HWPX 이미지 처리

HWPX 문서의 이미지를 추출하고 로컬에 저장합니다.
"""
import logging
import os
import zipfile
from typing import List

from libs.core.processor.hwpx_helper.hwpx_constants import SUPPORTED_IMAGE_EXTENSIONS
from libs.core.functions.img_processor import ImageProcessor

_image_processor = ImageProcessor(
    directory_path="temp/images",
    tag_prefix="[image:",
    tag_suffix="]"
)

logger = logging.getLogger("document-processor")


def process_hwpx_images(
    zf: zipfile.ZipFile,
    image_files: List[str]
) -> str:
    """
    HWPX zip에서 이미지를 추출하여 로컬에 저장합니다.

    Args:
        zf: 열린 ZipFile 객체
        image_files: 처리할 이미지 파일 경로 목록

    Returns:
        이미지 태그 문자열들을 줄바꿈으로 연결한 결과
    """
    results = []

    for img_path in image_files:
        ext = os.path.splitext(img_path)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            try:
                with zf.open(img_path) as f:
                    image_data = f.read()

                image_tag = _image_processor.save_image(image_data)
                if image_tag:
                    results.append(image_tag)

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
