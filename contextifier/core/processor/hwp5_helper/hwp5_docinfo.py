# contextifier/core/processor/hwp5_helper/hwp5_docinfo.py
"""
HWP 5.0 DocInfo Stream Parser

Parses DocInfo stream to extract BinData mappings and document properties.

BinData records map storage IDs to embedded files (images, OLE objects).
Each BIN_DATA record contains:
- Storage type (LINK, EMBEDDING, STORAGE)
- Storage ID (reference to BinData/BINxxxx.ext)
- Extension
"""
import re
import struct
import logging
import traceback
from typing import Dict, List, Tuple

import olefile

from contextifier.core.processor.hwp5_helper.hwp5_constants import (
    HWPTAG_BIN_DATA,
    BINDATA_LINK,
    BINDATA_EMBEDDING,
    BINDATA_STORAGE,
)
from contextifier.core.processor.hwp5_helper.hwp5_record import HwpRecord
from contextifier.core.processor.hwp5_helper.hwp5_decoder import (
    is_compressed,
    decompress_stream,
)

logger = logging.getLogger("document-processor.HWP5")


def parse_doc_info(ole: olefile.OleFileIO) -> Tuple[Dict[int, Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Parse DocInfo stream to extract BinData record mappings.
    
    BinData records contain references to embedded files in BinData/ folder.
    Storage ID maps to BinData/BINxxxx.ext where xxxx is hex storage ID.
    
    Args:
        ole: OLE file object
        
    Returns:
        Tuple of:
        - bin_data_by_storage_id: storage_id -> (storage_id, extension) mapping
        - bin_data_list: (storage_id, extension) ordered list (1-based index)
    """
    bin_data_by_storage_id = {}
    bin_data_list = []

    try:
        if not ole.exists("DocInfo"):
            logger.warning("DocInfo stream not found in OLE file")
            return bin_data_by_storage_id, bin_data_list

        compressed = is_compressed(ole)
        logger.debug(f"HWP 5.0 file compressed: {compressed}")

        stream = ole.openstream("DocInfo")
        data = stream.read()
        original_size = len(data)

        data = decompress_stream(data, compressed)
        logger.debug(f"DocInfo stream: original={original_size}, decompressed={len(data)}")

        root = HwpRecord.build_tree(data)
        logger.debug(f"DocInfo tree built with {len(root.children)} top-level records")

        # Log tag distribution for debugging
        tag_counts = {}
        for child in root.children:
            tag_counts[child.tag_id] = tag_counts.get(child.tag_id, 0) + 1
        logger.debug(f"DocInfo tag distribution: {tag_counts}")

        for child in root.children:
            if child.tag_id == HWPTAG_BIN_DATA:
                payload = child.payload
                logger.debug(
                    f"Found BIN_DATA record, payload size: {len(payload)}, "
                    f"hex: {payload[:20].hex() if len(payload) >= 20 else payload.hex()}"
                )

                if len(payload) < 2:
                    continue

                flags = struct.unpack('<H', payload[0:2])[0]
                storage_type = flags & 0x0F
                logger.debug(f"BIN_DATA flags: {flags:#06x}, storage_type: {storage_type}")

                if storage_type in [BINDATA_EMBEDDING, BINDATA_STORAGE]:
                    if len(payload) < 4:
                        bin_data_list.append((0, ""))
                        continue
                        
                    storage_id = struct.unpack('<H', payload[2:4])[0]

                    ext = ""
                    if len(payload) >= 6:
                        ext_len = struct.unpack('<H', payload[4:6])[0]
                        if ext_len > 0 and len(payload) >= 6 + ext_len * 2:
                            ext = payload[6:6+ext_len*2].decode('utf-16le', errors='ignore')

                    bin_data_by_storage_id[storage_id] = (storage_id, ext)
                    bin_data_list.append((storage_id, ext))
                    logger.debug(f"DocInfo BIN_DATA #{len(bin_data_list)}: storage_id={storage_id}, ext='{ext}'")

                elif storage_type == BINDATA_LINK:
                    bin_data_list.append((0, ""))
                    logger.debug(f"DocInfo BIN_DATA #{len(bin_data_list)}: LINK type (external)")

                else:
                    # Unknown type - try to extract anyway
                    storage_id = 0
                    ext = ""
                    if len(payload) >= 4:
                        storage_id = struct.unpack('<H', payload[2:4])[0]
                        if len(payload) >= 6:
                            ext_len = struct.unpack('<H', payload[4:6])[0]
                            if ext_len > 0 and ext_len < 20 and len(payload) >= 6 + ext_len * 2:
                                ext = payload[6:6+ext_len*2].decode('utf-16le', errors='ignore')
                    if storage_id > 0:
                        bin_data_by_storage_id[storage_id] = (storage_id, ext)
                    bin_data_list.append((storage_id, ext))
                    logger.debug(f"DocInfo BIN_DATA #{len(bin_data_list)}: unknown type {storage_type}, storage_id={storage_id}")

        logger.info(f"DocInfo parsed: {len(bin_data_list)} BIN_DATA records, {len(bin_data_by_storage_id)} with storage_id")

        # Fallback: scan BinData folder directly if no records found
        if len(bin_data_list) == 0:
            logger.info("No BIN_DATA in DocInfo, scanning BinData folder directly...")
            bin_data_by_storage_id, bin_data_list = scan_bindata_folder(ole)

    except Exception as e:
        logger.warning(f"Failed to parse DocInfo: {e}")
        logger.debug(traceback.format_exc())
        try:
            bin_data_by_storage_id, bin_data_list = scan_bindata_folder(ole)
        except Exception:
            pass

    return bin_data_by_storage_id, bin_data_list


def scan_bindata_folder(ole: olefile.OleFileIO) -> Tuple[Dict[int, Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Fallback: Scan BinData folder directly for embedded files.
    
    Used when DocInfo parsing fails or contains no BIN_DATA records.
    
    Args:
        ole: OLE file object
        
    Returns:
        Tuple of:
        - bin_data_by_storage_id: storage_id -> (storage_id, extension) mapping
        - bin_data_list: (storage_id, extension) ordered list
    """
    bin_data_by_storage_id = {}
    bin_data_list = []

    try:
        all_entries = ole.listdir()
        
        # Find BinData entries
        bindata_entries = [
            e for e in all_entries
            if len(e) >= 2 and e[0] == "BinData"
        ]
        
        # Sort by name to maintain order
        bindata_entries.sort(key=lambda x: x[-1] if x else "")
        
        # Pattern: BINxxxx.ext where xxxx is hex storage ID
        pattern = re.compile(r'BIN([0-9A-Fa-f]{4})\.(\w+)', re.IGNORECASE)
        
        for entry in bindata_entries:
            name = entry[-1] if entry else ""
            match = pattern.match(name)
            
            if match:
                storage_id = int(match.group(1), 16)
                ext = match.group(2)
                
                bin_data_by_storage_id[storage_id] = (storage_id, ext)
                bin_data_list.append((storage_id, ext))
                logger.debug(f"Scanned BinData: {name} -> storage_id={storage_id}, ext={ext}")
            else:
                # Non-standard naming, use index
                bin_data_list.append((0, ""))
                logger.debug(f"Scanned BinData (non-standard): {name}")
        
        logger.info(f"BinData folder scan: {len(bin_data_list)} files found")
        
    except Exception as e:
        logger.warning(f"Failed to scan BinData folder: {e}")
    
    return bin_data_by_storage_id, bin_data_list


__all__ = [
    'parse_doc_info',
    'scan_bindata_folder',
]
