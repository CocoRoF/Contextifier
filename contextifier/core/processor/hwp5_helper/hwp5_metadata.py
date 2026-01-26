# contextifier/core/processor/hwp5_helper/hwp5_metadata.py
"""
HWP 5.0 Metadata Extraction

Extracts metadata from HWP 5.0 OLE files using:
1. olefile's get_metadata() - OLE standard metadata
2. HwpSummaryInformation stream - HWP-specific metadata
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

logger = logging.getLogger("document-processor.HWP5")


class HWP5MetadataExtractor(BaseMetadataExtractor):
    """
    HWP 5.0 Metadata Extractor.
    
    Extracts metadata from olefile OleFileIO objects.
    Supports both OLE standard metadata and HWP-specific HwpSummaryInformation.
    
    Supported fields:
    - title, subject, author, keywords, comments
    - last_saved_by, create_time, last_saved_time
    """
    
    def extract(self, source: olefile.OleFileIO) -> DocumentMetadata:
        """
        Extract metadata from HWP 5.0 file.
        
        Args:
            source: olefile OleFileIO object
            
        Returns:
            DocumentMetadata instance
        """
        metadata_dict: Dict[str, Any] = {}
        
        # Method 1: olefile's get_metadata()
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
        
        # Method 2: Parse HwpSummaryInformation stream
        try:
            hwp_summary_stream = '\x05HwpSummaryInformation'
            if source.exists(hwp_summary_stream):
                self.logger.debug("Found HwpSummaryInformation stream")
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
    
    Args:
        data: HwpSummaryInformation stream binary data
        
    Returns:
        Dictionary containing parsed metadata
    """
    metadata = {}
    
    try:
        if len(data) < 28:
            return metadata
        
        pos = 0
        _byte_order = struct.unpack('<H', data[pos:pos+2])[0]
        pos = 28  # Skip header
        
        if len(data) < pos + 20:
            return metadata
        
        # Section Header: FMTID (16 bytes) + Offset (4 bytes)
        section_offset = struct.unpack('<I', data[pos+16:pos+20])[0]
        
        if section_offset >= len(data):
            return metadata
        
        # Parse section
        pos = section_offset
        
        if pos + 8 > len(data):
            return metadata
        
        section_size = struct.unpack('<I', data[pos:pos+4])[0]
        prop_count = struct.unpack('<I', data[pos+4:pos+8])[0]
        
        pos += 8
        
        # Property ID to name mapping (OLE standard)
        prop_id_map = {
            0x02: 'title',
            0x03: 'subject',
            0x04: 'author',
            0x05: 'keywords',
            0x06: 'comments',
            0x08: 'last_saved_by',
            0x0C: 'create_time',
            0x0D: 'last_saved_time',
        }
        
        # Parse property entries
        for _ in range(min(prop_count, 50)):  # Limit iterations
            if pos + 8 > len(data):
                break
            
            prop_id = struct.unpack('<I', data[pos:pos+4])[0]
            prop_offset = struct.unpack('<I', data[pos+4:pos+8])[0]
            pos += 8
            
            if prop_id not in prop_id_map:
                continue
            
            prop_pos = section_offset + prop_offset
            if prop_pos + 4 > len(data):
                continue
            
            prop_type = struct.unpack('<I', data[prop_pos:prop_pos+4])[0]
            prop_pos += 4
            
            value = None
            
            if prop_type == 0x1F:  # VT_LPWSTR (Unicode string)
                if prop_pos + 4 > len(data):
                    continue
                str_len = struct.unpack('<I', data[prop_pos:prop_pos+4])[0]
                prop_pos += 4
                if prop_pos + str_len * 2 <= len(data):
                    value = data[prop_pos:prop_pos+str_len*2].decode('utf-16le', errors='ignore').rstrip('\x00')
            
            elif prop_type == 0x40:  # VT_FILETIME
                if prop_pos + 8 <= len(data):
                    ft = struct.unpack('<Q', data[prop_pos:prop_pos+8])[0]
                    if ft > 0:
                        # Convert FILETIME to datetime
                        try:
                            # FILETIME is 100-nanosecond intervals since 1601-01-01
                            epoch_diff = 116444736000000000
                            timestamp = (ft - epoch_diff) / 10000000.0
                            value = datetime.fromtimestamp(timestamp)
                        except (ValueError, OSError):
                            pass
            
            if value:
                metadata[prop_id_map[prop_id]] = value
        
    except Exception as e:
        logger.debug(f"Error parsing HwpSummaryInformation: {e}")
    
    return metadata


__all__ = [
    'HWP5MetadataExtractor',
    'parse_hwp_summary_information',
]
