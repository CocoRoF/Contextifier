# contextifier/core/processor/hwp5_helper/__init__.py
"""
HWP 5.0 OLE Format Helper Module

Provides utilities for processing HWP 5.0 OLE format files.

HWP 5.0 is a binary format based on OLE (Object Linking and Embedding) 
compound document structure. The file contains:
- FileHeader: Version, compression flags, etc.
- DocInfo: BinData mappings, fonts, styles
- BodyText: Sections containing paragraph/text records
- BinData: Embedded images and OLE objects

File structure:
- hwp5_constants.py: HWP 5.0 tag IDs and constants
- hwp5_record.py: Binary record parsing (tag/level/size/payload)
- hwp5_decoder.py: Compression/encoding utilities
- hwp5_handler.py: Main handler class
- hwp5_table_extractor.py: BaseTableExtractor implementation
- hwp5_table_processor.py: Table format conversion
- hwp5_file_converter.py: OLE file conversion
- hwp5_preprocessor.py: Document preprocessing
- hwp5_metadata.py: Metadata extraction
- hwp5_docinfo.py: DocInfo stream parsing
- hwp5_image_processor.py: Image extraction
- hwp5_chart_extractor.py: Chart extraction
- hwp5_recovery.py: Corrupted file recovery
"""

# Constants
from contextifier.core.processor.hwp5_helper.hwp5_constants import (
    HWPTAG_BEGIN,
    HWPTAG_BIN_DATA,
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_CTRL_HEADER,
    HWPTAG_LIST_HEADER,
    HWPTAG_SHAPE_COMPONENT,
    HWPTAG_SHAPE_COMPONENT_PICTURE,
    HWPTAG_TABLE,
    HWPTAG_SHAPE_COMPONENT_OLE,
    HWPTAG_CHART_DATA,
    CHART_TYPES,
    CTRL_CHAR_DRAWING_TABLE_OBJECT,
)

# Record Parser
from contextifier.core.processor.hwp5_helper.hwp5_record import HwpRecord

# Decoder
from contextifier.core.processor.hwp5_helper.hwp5_decoder import (
    is_compressed,
    decompress_stream,
    decompress_section,
)

# Table Extractor and Processor
from contextifier.core.processor.hwp5_helper.hwp5_table_extractor import HWP5TableExtractor
from contextifier.core.processor.hwp5_helper.hwp5_table_processor import HWP5TableProcessor

# DocInfo
from contextifier.core.processor.hwp5_helper.hwp5_docinfo import (
    parse_doc_info,
    scan_bindata_folder,
)

# Recovery
from contextifier.core.processor.hwp5_helper.hwp5_recovery import (
    extract_text_from_stream_raw,
    find_zlib_streams,
    recover_images_from_raw,
    check_file_signature,
)


__all__ = [
    # Constants
    'HWPTAG_BEGIN',
    'HWPTAG_BIN_DATA',
    'HWPTAG_PARA_HEADER',
    'HWPTAG_PARA_TEXT',
    'HWPTAG_CTRL_HEADER',
    'HWPTAG_LIST_HEADER',
    'HWPTAG_SHAPE_COMPONENT',
    'HWPTAG_SHAPE_COMPONENT_PICTURE',
    'HWPTAG_TABLE',
    'HWPTAG_SHAPE_COMPONENT_OLE',
    'HWPTAG_CHART_DATA',
    'CHART_TYPES',
    'CTRL_CHAR_DRAWING_TABLE_OBJECT',
    # Record
    'HwpRecord',
    # Decoder
    'is_compressed',
    'decompress_stream',
    'decompress_section',
    # Table Extractor/Processor
    'HWP5TableExtractor',
    'HWP5TableProcessor',
    # DocInfo
    'parse_doc_info',
    'scan_bindata_folder',
    # Recovery
    'extract_text_from_stream_raw',
    'find_zlib_streams',
    'recover_images_from_raw',
    'check_file_signature',
]
