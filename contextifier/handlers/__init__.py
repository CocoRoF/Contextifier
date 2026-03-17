# contextifier_new/handlers/__init__.py
"""
Handlers — Format-Specific Document Processing

Each handler manages exactly ONE file extension (for document formats)
or a logical category (for text/image formats) and orchestrates the
5-stage pipeline + Stage 0 delegation check.

Principle: ONE EXTENSION PER HANDLER (document formats)
    Different binary formats must NEVER share a handler.
    e.g., PPT (OLE2) ≠ PPTX (OOXML), XLS (BIFF) ≠ XLSX (OOXML)

Architecture:
    BaseHandler (abstract, enforced pipeline)
      │
      │  Document formats — one extension each
      ├── PDFHandler        → .pdf  (via PyMuPDF/fitz)
      ├── DOCXHandler       → .docx (via python-docx)
      ├── DOCHandler        → .doc  (OLE2, with delegation for RTF/DOCX/HTML)
      ├── PPTXHandler       → .pptx (OOXML, via python-pptx)
      ├── PPTHandler        → .ppt  (OLE2, via LibreOffice conversion)
      ├── XLSXHandler       → .xlsx (OOXML, via openpyxl)
      ├── XLSHandler        → .xls  (BIFF, via xlrd)
      ├── CSVHandler        → .csv  (comma-delimited)
      ├── TSVHandler        → .tsv  (tab-delimited)
      ├── HWPHandler        → .hwp  (Korean OLE binary)
      ├── HWPXHandler       → .hwpx (Korean XML/ZIP)
      ├── RTFHandler        → .rtf
      │
      │  Category handlers — multiple extensions by design
      ├── TextHandler       → .txt/.md/.py/.json/.yaml/...
      └── ImageFileHandler  → .jpg/.png/.gif/.bmp/.webp/... (OCR)

Handler contract:
    - Every handler provides 5 pipeline component factories
    - Stage 0 (_check_delegation) runs before pipeline for format detection
    - Processing: process() → ExtractionResult
    - Text-only: extract_text() → str
    - Handlers are registered via HandlerRegistry
"""

from contextifier.handlers.base import BaseHandler
from contextifier.handlers.registry import HandlerRegistry

# Concrete handlers — imported for convenient access
from contextifier.handlers.pdf import PDFHandler
from contextifier.handlers.docx import DOCXHandler
from contextifier.handlers.doc import DOCHandler
from contextifier.handlers.pptx import PPTXHandler
from contextifier.handlers.ppt import PPTHandler
from contextifier.handlers.xlsx import XLSXHandler
from contextifier.handlers.xls import XLSHandler
from contextifier.handlers.csv import CSVHandler
from contextifier.handlers.tsv import TSVHandler
from contextifier.handlers.hwp import HWPHandler
from contextifier.handlers.hwpx import HWPXHandler
from contextifier.handlers.rtf import RTFHandler
from contextifier.handlers.text import TextHandler
from contextifier.handlers.image import ImageFileHandler

__all__ = [
    "BaseHandler",
    "HandlerRegistry",
    # Document formats (one extension each)
    "PDFHandler",
    "DOCXHandler",
    "DOCHandler",
    "PPTXHandler",
    "PPTHandler",
    "XLSXHandler",
    "XLSHandler",
    "CSVHandler",
    "TSVHandler",
    "HWPHandler",
    "HWPXHandler",
    "RTFHandler",
    # Category handlers
    "TextHandler",
    "ImageFileHandler",
]
