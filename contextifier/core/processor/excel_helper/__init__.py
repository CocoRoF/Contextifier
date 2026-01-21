"""
Excel Helper Module

Handles extraction of elements (textboxes, charts, images, tables, etc.) from XLSX/XLS files.

Module Structure:
- excel_chart_constants: Chart type mapping constants
- excel_chart_extractor: Chart extraction (ChartExtractor)
- excel_table_xlsx: XLSX table conversion
- excel_table_xls: XLS table conversion
- excel_textbox: Textbox extraction
- excel_metadata: Metadata extraction
- excel_image: Image extraction
- excel_layout_detector: Layout detection
"""

# === Textbox ===
from contextifier.core.processor.excel_helper.excel_textbox import extract_textboxes_from_xlsx

# === Metadata ===
from contextifier.core.processor.excel_helper.excel_metadata import (
    extract_xlsx_metadata,
    extract_xls_metadata,
    format_metadata,
)

# === Chart Constants ===
from contextifier.core.processor.excel_helper.excel_chart_constants import (
    CHART_TYPE_MAP,
    CHART_NAMESPACES,
)

# === Chart Extractor ===
from contextifier.core.processor.excel_helper.excel_chart_extractor import (
    ExcelChartExtractor,
)

# === Image ===
from contextifier.core.processor.excel_helper.excel_image import (
    extract_images_from_xlsx,
    get_sheet_images,
    SUPPORTED_IMAGE_EXTENSIONS,
)

# === Table XLSX ===
from contextifier.core.processor.excel_helper.excel_table_xlsx import (
    has_merged_cells_xlsx,
    convert_xlsx_sheet_to_table,
    convert_xlsx_sheet_to_markdown,
    convert_xlsx_sheet_to_html,
    convert_xlsx_objects_to_tables,
)

# === Table XLS ===
from contextifier.core.processor.excel_helper.excel_table_xls import (
    has_merged_cells_xls,
    convert_xls_sheet_to_table,
    convert_xls_sheet_to_markdown,
    convert_xls_sheet_to_html,
    convert_xls_objects_to_tables,
)

# === Layout Detector ===
from contextifier.core.processor.excel_helper.excel_layout_detector import (
    layout_detect_range_xlsx,
    layout_detect_range_xls,
    object_detect_xlsx,
    object_detect_xls,
    LayoutRange,
)


__all__ = [
    # Textbox
    'extract_textboxes_from_xlsx',
    # Metadata
    'extract_xlsx_metadata',
    'extract_xls_metadata',
    'format_metadata',
    # Chart Constants
    'CHART_TYPE_MAP',
    'CHART_NAMESPACES',
    # Chart Extractor
    'ExcelChartExtractor',
    # Image
    'extract_images_from_xlsx',
    'get_sheet_images',
    'SUPPORTED_IMAGE_EXTENSIONS',
    # Table XLSX
    'has_merged_cells_xlsx',
    'convert_xlsx_sheet_to_table',
    'convert_xlsx_sheet_to_markdown',
    'convert_xlsx_sheet_to_html',
    'convert_xlsx_objects_to_tables',
    # Table XLS
    'has_merged_cells_xls',
    'convert_xls_sheet_to_table',
    'convert_xls_sheet_to_markdown',
    'convert_xls_sheet_to_html',
    'convert_xls_objects_to_tables',
    # Layout Detector
    'layout_detect_range_xlsx',
    'layout_detect_range_xls',
    'object_detect_xlsx',
    'object_detect_xls',
    'LayoutRange',
]
