# libs/core/processor/__init__.py
"""
Processor - 문서 타입별 핸들러 모듈

개별 문서 형식을 처리하는 핸들러들을 제공합니다.

핸들러 목록:
- pdf_handler: PDF 문서 처리 (pdfminer 기반)
- pdf_handler_v4: PDF 문서 처리 v4 (고급 레이아웃 분석)
- docx_handler: DOCX 문서 처리
- doc_handler: DOC 문서 처리 (RTF 포함)
- ppt_handler: PPT/PPTX 문서 처리
- excel_handler: Excel (XLSX/XLS) 문서 처리
- hwp_processor: HWP 문서 처리
- hwpx_processor: HWPX 문서 처리
- csv_handler: CSV 파일 처리
- text_handler: 텍스트 파일 처리
- html_reprocessor: HTML 재처리
- ocr_processor: OCR 처리

헬퍼 모듈 (하위 디렉토리):
- csv_helper/: CSV 처리 헬퍼
- docx_helper/: DOCX 처리 헬퍼
- doc_helpers/: DOC/RTF 처리 헬퍼
- excel_helper/: Excel 처리 헬퍼
- hwp_helper/: HWP 처리 헬퍼
- hwpx_helper/: HWPX 처리 헬퍼
- pdf_helpers/: PDF 처리 헬퍼
- ppt_helper/: PPT 처리 헬퍼

사용 예시:
    from libs.core.processor import extract_text_from_pdf
    from libs.core.processor import extract_text_from_docx
    from libs.core.processor.pdf_helpers import extract_pdf_metadata
"""

# === PDF 핸들러 ===
from libs.core.processor.pdf_handler import extract_text_from_pdf
from libs.core.processor.pdf_handler_v4 import extract_text_from_pdf_v4

# === 문서 핸들러 ===
from libs.core.processor.docx_handler import extract_text_from_docx
from libs.core.processor.doc_handler import extract_text_from_doc
from libs.core.processor.ppt_handler import extract_text_from_ppt

# === 데이터 핸들러 ===
from libs.core.processor.excel_handler import extract_text_from_excel
from libs.core.processor.csv_handler import extract_text_from_csv
from libs.core.processor.text_handler import extract_text_from_text_file

# === HWP 핸들러 ===
from libs.core.processor.hwp_processor import extract_text_from_hwp
from libs.core.processor.hwpx_processor import extract_text_from_hwpx

# === 기타 프로세서 ===
# from libs.core.processor.ocr_processor import ...  # OCR 함수들
# from libs.core.processor.html_reprocessor import ...  # HTML 재처리

# === 헬퍼 모듈 (서브패키지) ===
from libs.core.processor import csv_helper
from libs.core.processor import docx_helper
from libs.core.processor import excel_helper
from libs.core.processor import hwp_helper
from libs.core.processor import hwpx_helper
from libs.core.processor import pdf_helpers
from libs.core.processor import ppt_helper

__all__ = [
    # PDF 핸들러
    "extract_text_from_pdf",
    "extract_text_from_pdf_v4",
    # 문서 핸들러
    "extract_text_from_docx",
    "extract_text_from_doc",
    "extract_text_from_ppt",
    # 데이터 핸들러
    "extract_text_from_excel",
    "extract_text_from_csv",
    "extract_text_from_text_file",
    # HWP 핸들러
    "extract_text_from_hwp",
    "extract_text_from_hwpx",
    # 헬퍼 서브패키지
    "csv_helper",
    "docx_helper",
    "excel_helper",
    "hwp_helper",
    "hwpx_helper",
    "pdf_helpers",
    "ppt_helper",
]
