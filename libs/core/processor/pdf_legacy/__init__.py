# libs/core/processor/pdf_legacy/__init__.py
"""
PDF Legacy 모듈

레거시 PDF 처리 핸들러들을 제공합니다.

모듈 구성:
- pdf_handler_default: 기본 PDF 핸들러
- pdf_handler_ocr: OCR 기반 PDF 핸들러
- pdf_handler_unstructured: Unstructured 라이브러리 기반 핸들러
- pdf_handler_v2: PDF 핸들러 v2
- pdf_handler_v3: PDF 핸들러 v3

참고: 이 모듈들은 레거시 호환성을 위해 유지됩니다.
새로운 코드에서는 pdf_handler 또는 pdf_handler_v4를 사용하세요.
"""

from .pdf_handler_default import *
from .pdf_handler_ocr import *
