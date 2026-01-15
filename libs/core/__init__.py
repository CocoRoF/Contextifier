# libs/core/__init__.py
"""
Core - 문서 처리 핵심 모듈

이 패키지는 다양한 문서 형식을 처리하는 핵심 기능을 제공합니다.

모듈 구조:
- document_processor: 메인 DocumentProcessor 클래스 (레거시)
- document_processor_new: 신규 DocumentProcessor 클래스 (권장)
- processor/: 개별 문서 타입별 핸들러
    - pdf_handler: PDF 문서 처리
    - docx_handler: DOCX 문서 처리
    - doc_handler: DOC 문서 처리
    - ppt_handler: PPT/PPTX 문서 처리
    - excel_handler: Excel 문서 처리
    - hwp_processor: HWP 문서 처리
    - hwpx_processor: HWPX 문서 처리
    - csv_handler: CSV 파일 처리
    - text_handler: 텍스트 파일 처리
    - ocr_processor: OCR 처리
- functions/: 유틸리티 함수
    - utils: 텍스트 정리, 코드 정리 등 공통 유틸리티
    - ppt2pdf: PPT to PDF 변환

사용 예시:
    from libs.core import DocumentProcessor
    from libs.core.processor import extract_text_from_pdf, extract_text_from_docx
    from libs.core.functions import clean_text, clean_code_text
"""

# === 메인 클래스 ===
from libs.core.document_processor import DocumentProcessor

# === 유틸리티 함수 ===
from libs.core.functions.utils import (
    clean_text,
    clean_code_text,
    sanitize_text_for_json,
)

# === 서브패키지 명시적 import ===
from libs.core import processor
from libs.core import functions

__all__ = [
    # 메인 클래스
    "DocumentProcessor",
    # 유틸리티 함수
    "clean_text",
    "clean_code_text",
    "sanitize_text_for_json",
    # 서브패키지
    "processor",
    "functions",
]
