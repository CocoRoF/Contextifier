# libs/core/processor/__init__.py
"""
Processor - Document Type-specific Handler Module

Provides handlers for processing individual document formats.

Handler List:
- pdf_handler: PDF document processing (adaptive complexity-based)
- docx_handler: DOCX document processing
- doc_handler: DOC document processing (including RTF)
- ppt_handler: PPT/PPTX document processing
- excel_handler: Excel (XLSX/XLS) document processing
- hwp_processor: HWP document processing
- hwpx_processor: HWPX document processing
- csv_handler: CSV file processing
- text_handler: Text file processing
- html_reprocessor: HTML reprocessing

Helper Modules (subdirectories):
- csv_helper/: CSV processing helper
- docx_helper/: DOCX processing helper
- doc_helpers/: DOC/RTF processing helper
- excel_helper/: Excel processing helper
- hwp_helper/: HWP processing helper
- hwpx_helper/: HWPX processing helper
- pdf_helpers/: PDF processing helper
- ppt_helper/: PPT processing helper

Usage Example:
    from libs.core.processor import extract_text_from_pdf
    from libs.core.processor import extract_text_from_docx
    from libs.core.processor.pdf_helpers import extract_pdf_metadata
"""

# === PDF Handler ===
from libs.core.processor.pdf_handler import extract_text_from_pdf

# === Document Handlers ===
from libs.core.processor.docx_handler import extract_text_from_docx
from libs.core.processor.doc_handler import extract_text_from_doc
from libs.core.processor.ppt_handler import extract_text_from_ppt

# === Data Handlers ===
from libs.core.processor.excel_handler import extract_text_from_excel
from libs.core.processor.csv_handler import extract_text_from_csv
from libs.core.processor.text_handler import extract_text_from_text_file

# === HWP Handlers ===
from libs.core.processor.hwp_processor import extract_text_from_hwp
from libs.core.processor.hwpx_processor import extract_text_from_hwpx

# === Other Processors ===
# from libs.core.processor.html_reprocessor import ...  # HTML reprocessing

# === Helper Modules (subpackages) ===
from libs.core.processor import csv_helper
from libs.core.processor import docx_helper
from libs.core.processor import excel_helper
from libs.core.processor import hwp_helper
from libs.core.processor import hwpx_helper
from libs.core.processor import pdf_helpers
from libs.core.processor import ppt_helper

__all__ = [
    # PDF Handler
    "extract_text_from_pdf",
    # Document Handlers
    "extract_text_from_docx",
    "extract_text_from_doc",
    "extract_text_from_ppt",
    # Data Handlers
    "extract_text_from_excel",
    "extract_text_from_csv",
    "extract_text_from_text_file",
    # HWP Handlers
    "extract_text_from_hwp",
    "extract_text_from_hwpx",
    # Helper subpackages
    "csv_helper",
    "docx_helper",
    "excel_helper",
    "hwp_helper",
    "hwpx_helper",
    "pdf_helpers",
    "ppt_helper",
]
