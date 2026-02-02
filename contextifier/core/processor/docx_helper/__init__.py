# contextifier/core/processor/docx_helper/__init__.py
"""
DOCX Helper Module

Utility modules for DOCX document processing.

Module structure:
- docx_constants: Constants, Enum, dataclasses (ElementType, NAMESPACES, etc.)
- docx_metadata: Metadata extraction (DOCXMetadataExtractor)
- docx_chart_extractor: Chart extraction (DOCXChartExtractor)
- docx_image_processor: Image/drawing processing (DOCXImageProcessor)
- docx_table_extractor: Table extraction (DOCXTableExtractor) - BaseTableExtractor interface
- docx_table_processor: Table formatting (DOCXTableProcessor) - TableProcessor interface
- docx_paragraph: Paragraph processing and page breaks
"""

# Constants
from contextifier.core.processor.docx_helper.docx_constants import (
    ElementType,
    DocxElement,
    NAMESPACES,
    CHART_TYPE_MAP,
)

# Metadata
from contextifier.core.processor.docx_helper.docx_metadata import (
    DOCXMetadataExtractor,
)

# Chart Extractor
from contextifier.core.processor.docx_helper.docx_chart_extractor import (
    DOCXChartExtractor,
)

# Image Processor (replaces docx_image.py utility functions)
from contextifier.core.processor.docx_helper.docx_image_processor import (
    DOCXImageProcessor,
)

# Table Extractor (BaseTableExtractor interface)
from contextifier.core.processor.docx_helper.docx_table_extractor import (
    DOCXTableExtractor,
    create_docx_table_extractor,
)

# Table Processor (TableProcessor interface)
from contextifier.core.processor.docx_helper.docx_table_processor import (
    DOCXTableProcessor,
    DOCXTableProcessorConfig,
    create_docx_table_processor,
    format_table_as_html,
)

# Paragraph
from contextifier.core.processor.docx_helper.docx_paragraph import (
    process_paragraph_element,
    has_page_break_element,
)


__all__ = [
    # Constants
    'ElementType',
    'DocxElement',
    'NAMESPACES',
    'CHART_TYPE_MAP',
    # Metadata
    'DOCXMetadataExtractor',
    # Chart Extractor
    'DOCXChartExtractor',
    # Image Processor
    'DOCXImageProcessor',
    # Table Extractor (BaseTableExtractor interface)
    'DOCXTableExtractor',
    'create_docx_table_extractor',
    # Table Processor (TableProcessor interface)
    'DOCXTableProcessor',
    'DOCXTableProcessorConfig',
    'create_docx_table_processor',
    'format_table_as_html',
    # Paragraph
    'process_paragraph_element',
    'has_page_break_element',
]
