# contextifier/handlers/pdf_default/__init__.py
"""
pdf_default — Simple / basic PDF content extraction.

Provides a straightforward extraction pipeline:
- Page-by-page text via ``page.get_text("text")``
- Tables via PyMuPDF ``page.find_tables()`` → HTML
- Embedded images via ``page.get_images()``
- No complexity analysis, no OCR fallback, no block imaging

Use ``pdf_plus`` for the full adaptive, complexity-driven pipeline.
"""

from contextifier.handlers.pdf_default.content_extractor import (
    PdfDefaultContentExtractor,
)

__all__ = ["PdfDefaultContentExtractor"]
