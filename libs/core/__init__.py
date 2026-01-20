# libs/core/__init__.py
"""
Core - Document Processing Core Module

This package provides core functionality for processing various document formats.

Module Structure:
- document_processor: Main DocumentProcessor class
- processor/: Individual document type handlers
    - pdf_handler: PDF document processing
    - docx_handler: DOCX document processing
    - doc_handler: DOC document processing
    - ppt_handler: PPT/PPTX document processing
    - excel_handler: Excel document processing
    - hwp_handler: HWP document processing
    - hwpx_handler: HWPX document processing
    - csv_handler: CSV file processing
    - text_handler: Text file processing
- functions/: Utility functions
    - utils: Text cleaning, code cleaning, and common utilities
    - img_processor: Image processing and saving (ImageProcessor class)
    - ppt2pdf: PPT to PDF conversion

Usage:
    from contextifier import DocumentProcessor
    from contextifier.core.processor import PDFHandler, DocxHandler
    from contextifier.core.functions import clean_text, ImageProcessor
"""

# === Main Class ===
from libs.core.document_processor import DocumentProcessor

# === Utility Functions ===
from libs.core.functions.utils import (
    clean_text,
    clean_code_text,
    sanitize_text_for_json,
)

# === Image Processing ===
from libs.core.functions.img_processor import (
    ImageProcessor,
    save_image_to_file,
)

# === Explicit Subpackage Imports ===
from libs.core import processor
from libs.core import functions

__all__ = [
    # Main Class
    "DocumentProcessor",
    # Utility Functions
    "clean_text",
    "clean_code_text",
    "sanitize_text_for_json",
    # Image Processing
    "ImageProcessor",
    "save_image_to_file",
    # Subpackages
    "processor",
    "functions",
]
