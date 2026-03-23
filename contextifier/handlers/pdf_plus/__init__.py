# contextifier/handlers/pdf_plus/__init__.py
"""
PDF Plus — adaptive, multi-strategy PDF extraction engine.

Public API:
    ``PdfPlusContentExtractor``  — implements ``BaseContentExtractor``

Internal modules (imported lazily by the extractor):
    _types, _utils, _page_analyzer, _element_merger,
    _line_analysis, _graphic_detector,
    _table_detection, _table_validator, _table_quality_analyzer,
    _cell_analysis, _table_processor,
    _text_quality_analyzer, _text_extractor, _vector_text_ocr,
    _image_extractor,
    _layout_block_detector, _block_image_engine,
    _complexity_analyzer
"""

from contextifier.handlers.pdf_plus.content_extractor import (
    PdfPlusContentExtractor,
)

__all__ = ["PdfPlusContentExtractor"]
