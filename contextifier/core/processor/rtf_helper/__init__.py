# contextifier/core/processor/rtf_helper/__init__.py
"""
RTF Helper Module

Provides RTF parsing and extraction utilities with proper interface separation.

Architecture:
    - RTFPreprocessor: Binary preprocessing (image extraction, \\bin handling)
    - RTFFileConverter: Pass through (RTF uses raw binary)
    - RTFMetadataExtractor: Metadata extraction
    - Table extraction: extract_tables_with_positions()
    - Content extraction: extract_inline_content(), extract_text_only()

Usage:
    from contextifier.core.processor.rtf_helper import (
        RTFFileConverter,
        RTFConvertedData,
        RTFPreprocessor,
        RTFMetadataExtractor,
        RTFSourceInfo,
        extract_tables_with_positions,
        extract_inline_content,
        extract_text_only,
    )
"""

# Converter
from contextifier.core.processor.rtf_helper.rtf_file_converter import (
    RTFFileConverter,
    RTFConvertedData,
)

# Preprocessor
from contextifier.core.processor.rtf_helper.rtf_preprocessor import (
    RTFPreprocessor,
)

# Metadata
from contextifier.core.processor.rtf_helper.rtf_metadata_extractor import (
    RTFMetadataExtractor,
    RTFSourceInfo,
)

# Table extraction
from contextifier.core.processor.rtf_helper.rtf_table_extractor import (
    RTFCellInfo,
    RTFTable,
    extract_tables_with_positions,
)

# Content extraction
from contextifier.core.processor.rtf_helper.rtf_content_extractor import (
    extract_inline_content,
    extract_text_only,
)

# Decoder utilities
from contextifier.core.processor.rtf_helper.rtf_decoder import (
    detect_encoding,
    decode_content,
    decode_bytes,
    decode_hex_escapes,
)

# Text cleaning utilities
from contextifier.core.processor.rtf_helper.rtf_text_cleaner import (
    clean_rtf_text,
    remove_destination_groups,
    remove_shape_groups,
    remove_shape_property_groups,
    remove_shprslt_blocks,
)

# Region finder utilities
from contextifier.core.processor.rtf_helper.rtf_region_finder import (
    find_excluded_regions,
    is_in_excluded_region,
)

# Constants
from contextifier.core.processor.rtf_helper.rtf_constants import (
    SHAPE_PROPERTY_NAMES,
    SKIP_DESTINATIONS,
    EXCLUDE_DESTINATION_KEYWORDS,
    IMAGE_DESTINATIONS,
    CODEPAGE_ENCODING_MAP,
    DEFAULT_ENCODINGS,
)


__all__ = [
    # Converter
    'RTFFileConverter',
    'RTFConvertedData',
    # Preprocessor
    'RTFPreprocessor',
    # Metadata
    'RTFMetadataExtractor',
    'RTFSourceInfo',
    # Table
    'RTFCellInfo',
    'RTFTable',
    'extract_tables_with_positions',
    # Content
    'extract_inline_content',
    'extract_text_only',
    # Decoder
    'detect_encoding',
    'decode_content',
    'decode_bytes',
    'decode_hex_escapes',
    # Text cleaner
    'clean_rtf_text',
    'remove_destination_groups',
    'remove_shape_groups',
    'remove_shape_property_groups',
    'remove_shprslt_blocks',
    # Region finder
    'find_excluded_regions',
    'is_in_excluded_region',
    # Constants
    'SHAPE_PROPERTY_NAMES',
    'SKIP_DESTINATIONS',
    'EXCLUDE_DESTINATION_KEYWORDS',
    'IMAGE_DESTINATIONS',
    'CODEPAGE_ENCODING_MAP',
    'DEFAULT_ENCODINGS',
]
