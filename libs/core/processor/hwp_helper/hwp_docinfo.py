# service/document_processor/processor/hwp_helper/hwp_docinfo.py
"""
HWP DocInfo 파싱 유틸리티

HWP 5.0 OLE 파일의 DocInfo 스트림을 파싱하여 BinData 매핑 정보를 추출합니다.
- parse_doc_info: DocInfo 스트림에서 BinData 레코드 매핑 추출
- scan_bindata_folder: BinData 폴더 직접 스캔 (폴백)
"""
import re
import struct
import logging
import traceback
from typing import Dict, List, Tuple

import olefile

from libs.core.processor.hwp_helper.hwp_constants import HWPTAG_BIN_DATA
from libs.core.processor.hwp_helper.hwp_record import HwpRecord
from libs.core.processor.hwp_helper.hwp_decoder import (
    is_compressed,
    decompress_stream,
)

logger = logging.getLogger("document-processor")


def parse_doc_info(ole: olefile.OleFileIO) -> Tuple[Dict[int, Tuple[int, str]], List[Tuple[int, str]]]:
    """
    DocInfo 스트림을 파싱하여 BinData 레코드를 매핑합니다.

    HWP의 DocInfo 스트림에는 BinData 레코드들이 포함되어 있으며,
    각 레코드는 storage_id와 확장자 정보를 가지고 있습니다.

    Args:
        ole: OLE 파일 객체

    Returns:
        튜플:
        - bin_data_by_storage_id: storage_id -> (storage_id, extension) 매핑
        - bin_data_list: (storage_id, extension) 순서 리스트 (1-based index lookup)
    """
    bin_data_by_storage_id = {}
    bin_data_list = []

    try:
        if not ole.exists("DocInfo"):
            logger.warning("DocInfo stream not found in OLE file")
            return bin_data_by_storage_id, bin_data_list

        compressed = is_compressed(ole)
        logger.info(f"HWP file compressed: {compressed}")

        stream = ole.openstream("DocInfo")
        data = stream.read()
        original_size = len(data)

        data = decompress_stream(data, compressed)
        logger.info(f"DocInfo stream: original={original_size}, decompressed={len(data)}")

        root = HwpRecord.build_tree(data)
        logger.info(f"DocInfo tree built with {len(root.children)} top-level records")

        # 디버그: 모든 태그 ID 로깅
        tag_counts = {}
        for child in root.children:
            tag_counts[child.tag_id] = tag_counts.get(child.tag_id, 0) + 1
        logger.info(f"DocInfo tag distribution: {tag_counts}")

        for child in root.children:
            if child.tag_id == HWPTAG_BIN_DATA:
                payload = child.payload
                logger.debug(f"Found BIN_DATA record, payload size: {len(payload)}, hex: {payload[:20].hex() if len(payload) >= 20 else payload.hex()}")

                if len(payload) < 2:
                    continue

                flags = struct.unpack('<H', payload[0:2])[0]
                storage_type = flags & 0x0F
                logger.debug(f"BIN_DATA flags: {flags:#06x}, storage_type: {storage_type}")

                if storage_type in [1, 2]:  # EMBEDDING or STORAGE
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

                elif storage_type == 0:  # LINK
                    bin_data_list.append((0, ""))
                    logger.debug(f"DocInfo BIN_DATA #{len(bin_data_list)}: LINK type (external)")

                else:
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

        # Fallback: DocInfo에 BIN_DATA가 없으면 BinData 폴더 직접 스캔
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
    Fallback: BinData 폴더를 직접 스캔하여 임베디드 파일을 찾습니다.

    DocInfo 파싱에 실패했거나 BIN_DATA 레코드가 없는 경우 사용합니다.

    Args:
        ole: OLE 파일 객체

    Returns:
        튜플:
        - bin_data_by_storage_id: storage_id -> (storage_id, extension) 매핑
        - bin_data_list: (storage_id, extension) 순서 리스트
    """
    bin_data_by_storage_id = {}
    bin_data_list = []

    try:
        for entry in ole.listdir():
            if len(entry) >= 2 and entry[0] == "BinData":
                filename = entry[1]
                match = re.match(r'BIN([0-9A-Fa-f]{4})\.(\w+)', filename)
                if match:
                    storage_id = int(match.group(1), 16)
                    ext = match.group(2)
                    bin_data_by_storage_id[storage_id] = (storage_id, ext)
                    bin_data_list.append((storage_id, ext))
                    logger.debug(f"Found BinData stream: {filename} -> storage_id={storage_id}, ext={ext}")

        if bin_data_list:
            bin_data_list.sort(key=lambda x: x[0])
            logger.info(f"BinData folder scan: found {len(bin_data_list)} files")
    except Exception as e:
        logger.warning(f"Failed to scan BinData folder: {e}")

    return bin_data_by_storage_id, bin_data_list


__all__ = [
    'parse_doc_info',
    'scan_bindata_folder',
]
