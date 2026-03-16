# contextifier_new/handlers/hwp/_docinfo.py
"""
HWP DocInfo stream parser.

The ``DocInfo`` OLE stream contains metadata about embedded binary
objects (images, OLE, etc.).  Each ``HWPTAG_BIN_DATA`` record carries
a storage ID and file extension that together identify a BinData
stream in the OLE container (e.g. ``BinData/BIN0001.png``).

Public API:
    parse_doc_info(ole, compressed) -> (by_id_dict, ordered_list)
    scan_bindata_folder(ole)        -> (by_id_dict, ordered_list)
"""

from __future__ import annotations

import re
import struct
import logging
from typing import Dict, List, Tuple

import olefile

from contextifier_new.handlers.hwp._constants import (
    BINDATA_EMBEDDING,
    BINDATA_LINK,
    BINDATA_STORAGE,
    HWPTAG_BIN_DATA,
    STREAM_DOC_INFO,
)
from contextifier_new.handlers.hwp._decoder import decompress_stream, is_compressed
from contextifier_new.handlers.hwp._record import HwpRecord

logger = logging.getLogger(__name__)

# Type alias: storage_id → (storage_id, extension)
BinDataEntry = Tuple[int, str]
BinDataMap = Dict[int, BinDataEntry]
BinDataList = List[BinDataEntry]


def parse_doc_info(
    ole: olefile.OleFileIO,
) -> Tuple[BinDataMap, BinDataList]:
    """
    Parse the ``DocInfo`` stream for ``BIN_DATA`` records.

    Returns:
        (by_storage_id, ordered_list)
        - by_storage_id: ``{sid: (sid, ext)}``
        - ordered_list:  ``[(sid, ext), ...]`` in record order
    """
    by_id: BinDataMap = {}
    ordered: BinDataList = []

    try:
        if not ole.exists(STREAM_DOC_INFO):
            logger.warning("DocInfo stream not found")
            return by_id, ordered

        compressed = is_compressed(ole)
        raw = ole.openstream(STREAM_DOC_INFO).read()
        data = decompress_stream(raw, compressed)

        root = HwpRecord.build_tree(data)

        for child in root.children:
            if child.tag_id != HWPTAG_BIN_DATA:
                continue
            payload = child.payload
            if len(payload) < 2:
                continue

            flags = struct.unpack_from("<H", payload, 0)[0]
            storage_type = flags & 0x0F

            if storage_type in (BINDATA_EMBEDDING, BINDATA_STORAGE):
                if len(payload) < 4:
                    ordered.append((0, ""))
                    continue
                sid = struct.unpack_from("<H", payload, 2)[0]
                ext = ""
                if len(payload) >= 6:
                    ext_len = struct.unpack_from("<H", payload, 4)[0]
                    if 0 < ext_len < 50 and len(payload) >= 6 + ext_len * 2:
                        ext = payload[6: 6 + ext_len * 2].decode(
                            "utf-16le", errors="ignore"
                        )
                by_id[sid] = (sid, ext)
                ordered.append((sid, ext))

            elif storage_type == BINDATA_LINK:
                ordered.append((0, ""))

            else:
                # Unknown type — best-effort
                sid = 0
                ext = ""
                if len(payload) >= 4:
                    sid = struct.unpack_from("<H", payload, 2)[0]
                    if len(payload) >= 6:
                        ext_len = struct.unpack_from("<H", payload, 4)[0]
                        if 0 < ext_len < 50 and len(payload) >= 6 + ext_len * 2:
                            ext = payload[6: 6 + ext_len * 2].decode(
                                "utf-16le", errors="ignore"
                            )
                if sid > 0:
                    by_id[sid] = (sid, ext)
                ordered.append((sid, ext))

        # Fallback: if no BIN_DATA records found, scan BinData folder
        if not ordered:
            by_id, ordered = scan_bindata_folder(ole)

    except Exception as exc:
        logger.warning("Failed to parse DocInfo: %s", exc)
        try:
            by_id, ordered = scan_bindata_folder(ole)
        except Exception:
            pass

    return by_id, ordered


def scan_bindata_folder(
    ole: olefile.OleFileIO,
) -> Tuple[BinDataMap, BinDataList]:
    """
    Fallback: scan the ``BinData`` OLE folder directly.

    Returns the same tuple shape as :func:`parse_doc_info`.
    """
    by_id: BinDataMap = {}
    ordered: BinDataList = []

    try:
        for entry in ole.listdir():
            if len(entry) >= 2 and entry[0] == "BinData":
                match = re.match(r"BIN([0-9A-Fa-f]{4})\.(\w+)", entry[1])
                if match:
                    sid = int(match.group(1), 16)
                    ext = match.group(2)
                    by_id[sid] = (sid, ext)
                    ordered.append((sid, ext))

        if ordered:
            ordered.sort(key=lambda x: x[0])
    except Exception as exc:
        logger.warning("Failed to scan BinData folder: %s", exc)

    return by_id, ordered


__all__ = ["parse_doc_info", "scan_bindata_folder"]
