# libs/chunking/__init__.py
"""
Chunking - Text Chunking Module

This package provides functionality to split document text into appropriately sized chunks.

Module Structure:
- chunking: Main chunking functions (split_text_preserving_html_blocks, etc.)
- constants: Constants, patterns, and data classes
- table_parser: HTML table parsing
- table_chunker: Table chunking core logic
- protected_regions: Protected region handling
- page_chunker: Page-based chunking
- text_chunker: Text chunking
- sheet_processor: Sheet and metadata processing

Usage:
    from contextifier.chunking import split_text_preserving_html_blocks, chunk_plain_text
    from contextifier.chunking import TableRow, ParsedTable
"""

# === Main Chunking Functions (chunking.py) ===
from libs.chunking.chunking import (
    # Primary API
    create_chunks,
    # Backward compatibility (deprecated)
    split_text_preserving_html_blocks,
    split_table_based_content,
    is_table_based_file_type,
)

# constants
from libs.chunking.constants import (
    # Constants
    LANGCHAIN_CODE_LANGUAGE_MAP,
    HTML_TABLE_PATTERN,
    CHART_BLOCK_PATTERN,
    TEXTBOX_BLOCK_PATTERN,
    IMAGE_TAG_PATTERN,
    MARKDOWN_TABLE_PATTERN,
    TABLE_WRAPPER_OVERHEAD,
    CHUNK_INDEX_OVERHEAD,
    TABLE_SIZE_THRESHOLD_MULTIPLIER,
    TABLE_BASED_FILE_TYPES,
    # Data classes
    TableRow,
    ParsedTable,
)

# table_parser
from libs.chunking.table_parser import (
    parse_html_table,
    extract_cell_spans,
    extract_cell_spans_with_positions,
    has_complex_spans,
)

# table_chunker
from libs.chunking.table_chunker import (
    calculate_available_space,
    adjust_rowspan_in_chunk,
    build_table_chunk,
    update_chunk_metadata,
    split_table_into_chunks,
    split_table_preserving_rowspan,
    chunk_large_table,
)

# protected_regions
from libs.chunking.protected_regions import (
    find_protected_regions,
    get_protected_region_positions,
    ensure_protected_region_integrity,
    split_with_protected_regions,
    split_large_chunk_with_protected_regions,
    # Backward compatibility aliases
    ensure_table_integrity,
    split_large_chunk_with_table_protection,
)

# page_chunker
from libs.chunking.page_chunker import (
    split_into_pages,
    merge_pages,
    get_overlap_content,
    chunk_by_pages,
)

# text_chunker
from libs.chunking.text_chunker import (
    chunk_plain_text,
    chunk_text_without_tables,
    chunk_with_row_protection,
    chunk_with_row_protection_simple,
    clean_chunks,
    chunk_code_text,
    reconstruct_text_from_chunks,
    find_overlap_length,
    estimate_chunks_count,
)

# sheet_processor
from libs.chunking.sheet_processor import (
    extract_document_metadata,
    prepend_metadata_to_chunks,
    extract_sheet_sections,
    extract_content_segments,
    chunk_multi_sheet_content,
    chunk_single_table_content,
)


__all__ = [
    # === Primary API ===
    "create_chunks",
    # === Backward Compatibility (deprecated) ===
    "split_text_preserving_html_blocks",
    "split_table_based_content",
    "is_table_based_file_type",
    # constants
    "LANGCHAIN_CODE_LANGUAGE_MAP",
    "HTML_TABLE_PATTERN",
    "CHART_BLOCK_PATTERN",
    "TEXTBOX_BLOCK_PATTERN",
    "IMAGE_TAG_PATTERN",
    "MARKDOWN_TABLE_PATTERN",
    "TABLE_WRAPPER_OVERHEAD",
    "CHUNK_INDEX_OVERHEAD",
    "TABLE_SIZE_THRESHOLD_MULTIPLIER",
    "TABLE_BASED_FILE_TYPES",
    "TableRow",
    "ParsedTable",
    # table_parser
    "parse_html_table",
    "extract_cell_spans",
    "extract_cell_spans_with_positions",
    "has_complex_spans",
    # table_chunker
    "calculate_available_space",
    "adjust_rowspan_in_chunk",
    "build_table_chunk",
    "update_chunk_metadata",
    "split_table_into_chunks",
    "split_table_preserving_rowspan",
    "chunk_large_table",
    # protected_regions
    "find_protected_regions",
    "get_protected_region_positions",
    "ensure_protected_region_integrity",
    "split_with_protected_regions",
    "split_large_chunk_with_protected_regions",
    "ensure_table_integrity",
    "split_large_chunk_with_table_protection",
    # page_chunker
    "split_into_pages",
    "merge_pages",
    "get_overlap_content",
    "chunk_by_pages",
    # text_chunker
    "chunk_plain_text",
    "chunk_text_without_tables",
    "chunk_with_row_protection",
    "chunk_with_row_protection_simple",
    "clean_chunks",
    "chunk_code_text",
    "reconstruct_text_from_chunks",
    "find_overlap_length",
    "estimate_chunks_count",
    # sheet_processor
    "extract_document_metadata",
    "prepend_metadata_to_chunks",
    "extract_sheet_sections",
    "extract_content_segments",
    "chunk_multi_sheet_content",
    "chunk_single_table_content",
]
