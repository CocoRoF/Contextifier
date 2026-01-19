# libs/core/functions/__init__.py
"""
Functions - 공통 유틸리티 함수 모듈

문서 처리에 사용되는 공통 유틸리티 함수들을 제공합니다.

모듈 구성:
- utils: 텍스트 정리, 코드 정리, JSON 정리 등 유틸리티 함수
- img_processor: 이미지 처리 및 저장 (ImageProcessor 클래스)
- ppt2pdf: PPT를 PDF로 변환하는 함수

사용 예시:
    from libs.core.functions import clean_text, clean_code_text
    from libs.core.functions import ImageProcessor, save_image_to_file
    from libs.core.functions.utils import sanitize_text_for_json
"""

from libs.core.functions.utils import (
    clean_text,
    clean_code_text,
    sanitize_text_for_json,
)

# 이미지 처리 모듈
from libs.core.functions.img_processor import (
    ImageProcessor,
    ImageProcessorConfig,
    ImageFormat,
    NamingStrategy,
    save_image_to_file,
    create_image_processor,
)

__all__ = [
    # 텍스트 유틸리티
    "clean_text",
    "clean_code_text",
    "sanitize_text_for_json",
    # 이미지 처리
    "ImageProcessor",
    "ImageProcessorConfig",
    "ImageFormat",
    "NamingStrategy",
    "save_image_to_file",
    "create_image_processor",
]
