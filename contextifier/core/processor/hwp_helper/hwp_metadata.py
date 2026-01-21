# contextifier/core/processor/hwp_helper/hwp_metadata.py
"""
HWP Metadata Extraction Module

Provides HWPMetadataExtractor class for extracting metadata from HWP 5.0 OLE files.
Implements BaseMetadataExtractor interface.

Extraction methods:
1. olefile's get_metadata() - OLE standard metadata
2. HwpSummaryInformation stream direct parsing - HWP-specific metadata

Note: HWP is a Korean-native document format, so Korean metadata labels
are preserved in output for proper display.
"""
import struct
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import olefile

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)

logger = logging.getLogger("document-processor")


class HWPMetadataExtractor(BaseMetadataExtractor):
    """
    HWP Metadata Extractor.
    
    Extracts metadata from olefile OleFileIO objects.
    Supports both OLE standard metadata and HWP-specific HwpSummaryInformation.
    
    Supported fields:
    - title, subject, author, keywords, comments
    - last_saved_by, create_time, last_saved_time
    
    Usage:
        extractor = HWPMetadataExtractor()
        metadata = extractor.extract(ole_file)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: olefile.OleFileIO) -> DocumentMetadata:
        """
        Extract metadata from HWP file.
        
        Args:
            source: olefile OleFileIO object
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        metadata_dict: Dict[str, Any] = {}
        
        # Method 1: Use olefile's get_metadata()
        try:
            ole_meta = source.get_metadata()
            
            if ole_meta:
                if ole_meta.title:
                    metadata_dict['title'] = ole_meta.title
                if ole_meta.subject:
                    metadata_dict['subject'] = ole_meta.subject
                if ole_meta.author:
                    metadata_dict['author'] = ole_meta.author
                if ole_meta.keywords:
                    metadata_dict['keywords'] = ole_meta.keywords
                if ole_meta.comments:
                    metadata_dict['comments'] = ole_meta.comments
                if ole_meta.last_saved_by:
                    metadata_dict['last_saved_by'] = ole_meta.last_saved_by
                if ole_meta.create_time:
                    metadata_dict['create_time'] = ole_meta.create_time
                if ole_meta.last_saved_time:
                    metadata_dict['last_saved_time'] = ole_meta.last_saved_time
            
            self.logger.debug(f"Extracted OLE metadata: {list(metadata_dict.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Failed to extract OLE metadata: {e}")
        
        # Method 2: Parse HwpSummaryInformation stream directly
        try:
            hwp_summary_stream = '\x05HwpSummaryInformation'
            if source.exists(hwp_summary_stream):
                self.logger.debug("Found HwpSummaryInformation stream, attempting to parse...")
                stream = source.openstream(hwp_summary_stream)
                data = stream.read()
                hwp_meta = parse_hwp_summary_information(data)
                
                # HWP-specific metadata takes priority
                for key, value in hwp_meta.items():
                    if value:
                        metadata_dict[key] = value
                        
        except Exception as e:
            self.logger.debug(f"Failed to parse HwpSummaryInformation: {e}")
        
        return DocumentMetadata(
            title=metadata_dict.get('title'),
            subject=metadata_dict.get('subject'),
            author=metadata_dict.get('author'),
            keywords=metadata_dict.get('keywords'),
            comments=metadata_dict.get('comments'),
            last_saved_by=metadata_dict.get('last_saved_by'),
            create_time=metadata_dict.get('create_time'),
            last_saved_time=metadata_dict.get('last_saved_time'),
        )


def parse_hwp_summary_information(data: bytes) -> Dict[str, Any]:
    """
    Parse HwpSummaryInformation stream (OLE Property Set format).
    
    OLE Property Set structure:
    - Header (28 bytes)
    - Section(s) containing property ID/offset pairs
    - Property values (string, datetime, etc.)
    
    Args:
        data: HwpSummaryInformation stream binary data
        
    Returns:
        Dictionary containing parsed metadata.
    """
    metadata = {}
    
    try:
        if len(data) < 28:
            return metadata
        
        pos = 0
        _byte_order = struct.unpack('<H', data[pos:pos+2])[0]  # noqa: F841
        pos = 28  # Skip header
        
        if len(data) < pos + 20:
            return metadata
        
        # Section Header: FMTID (16 bytes) + Offset (4 bytes)
        section_offset = struct.unpack('<I', data[pos+16:pos+20])[0]
        
        if section_offset >= len(data):
            return metadata
        
        # Parse section
        pos = section_offset
        if len(data) < pos + 8:
            return metadata
        
        _section_size = struct.unpack('<I', data[pos:pos+4])[0]  # noqa: F841
        num_properties = struct.unpack('<I', data[pos+4:pos+8])[0]
        pos += 8
        
        # Read property ID/offset pairs
        properties = []
        for _ in range(min(num_properties, 50)):
            if len(data) < pos + 8:
                break
            prop_id = struct.unpack('<I', data[pos:pos+4])[0]
            prop_offset = struct.unpack('<I', data[pos+4:pos+8])[0]
            properties.append((prop_id, prop_offset))
            pos += 8
        
        # Read property values
        for prop_id, prop_offset in properties:
            abs_offset = section_offset + prop_offset
            if abs_offset + 4 >= len(data):
                continue
            
            prop_type = struct.unpack('<I', data[abs_offset:abs_offset+4])[0]
            value_offset = abs_offset + 4
            
            value = None
            
            if prop_type == 0x1E:  # ANSI String
                if value_offset + 4 < len(data):
                    str_len = struct.unpack('<I', data[value_offset:value_offset+4])[0]
                    if str_len > 0 and value_offset + 4 + str_len <= len(data):
                        try:
                            value = data[value_offset+4:value_offset+4+str_len].decode('cp949', errors='ignore').rstrip('\x00')
                        except Exception:
                            value = data[value_offset+4:value_offset+4+str_len].decode('utf-8', errors='ignore').rstrip('\x00')
            
            elif prop_type == 0x1F:  # Unicode String
                if value_offset + 4 < len(data):
                    str_len = struct.unpack('<I', data[value_offset:value_offset+4])[0]
                    byte_len = str_len * 2
                    if str_len > 0 and value_offset + 4 + byte_len <= len(data):
                        value = data[value_offset+4:value_offset+4+byte_len].decode('utf-16le', errors='ignore').rstrip('\x00')
            
            elif prop_type == 0x40:  # FILETIME
                if value_offset + 8 <= len(data):
                    filetime = struct.unpack('<Q', data[value_offset:value_offset+8])[0]
                    if filetime > 0:
                        try:
                            seconds = filetime / 10000000
                            epoch_diff = 11644473600
                            unix_time = seconds - epoch_diff
                            if 0 < unix_time < 2000000000:
                                value = datetime.fromtimestamp(unix_time)
                        except Exception:
                            pass
            
            # Property ID mapping
            if value:
                if prop_id == 0x02:
                    metadata['title'] = value
                elif prop_id == 0x03:
                    metadata['subject'] = value
                elif prop_id == 0x04:
                    metadata['author'] = value
                elif prop_id == 0x05:
                    metadata['keywords'] = value
                elif prop_id == 0x06:
                    metadata['comments'] = value
                elif prop_id == 0x08:
                    metadata['last_saved_by'] = value
                elif prop_id == 0x0C:
                    metadata['create_time'] = value
                elif prop_id == 0x0D:
                    metadata['last_saved_time'] = value
    
    except Exception as e:
        logger.debug(f"Error parsing HWP summary information: {e}")
    
    return metadata


__all__ = [
    'HWPMetadataExtractor',
    'parse_hwp_summary_information',
]
