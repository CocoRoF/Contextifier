# contextifier_new/handlers/hwp/content_extractor.py
"""
HwpContentExtractor — Stage 4: OLE → text / tables / images

Walks each ``BodyText/SectionN`` stream:
1. (optionally) decompresses the stream
2. builds a record tree via ``HwpRecord.build_tree()``
3. traverses the tree, emitting text, table HTML, and image tags

The traversal strategy matches the v1.0 HWP handler:
- PARA_HEADER → drives paragraph output (text + inline controls)
- CTRL_HEADER(tbl ) → table parsing via ``_table.parse_table``
- CTRL_HEADER(gso ) → graphic shape object → image extraction
- SHAPE_COMPONENT_PICTURE → binary-image extraction via BinData
"""

from __future__ import annotations

import struct
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    ChartData,
    ExtractionResult,
    PreprocessedData,
    TableData,
)
from contextifier.handlers.hwp._constants import (
    HWPTAG_CTRL_HEADER,
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_SHAPE_COMPONENT_PICTURE,
    HWPTAG_TABLE,
    STREAM_BODY_TEXT,
)
from contextifier.handlers.hwp._record import HwpRecord
from contextifier.handlers.hwp._decoder import decompress_section
from contextifier.handlers.hwp._table import parse_table
from contextifier.handlers.hwp._recovery import extract_text_raw

if TYPE_CHECKING:
    import olefile
    from contextifier.services.image_service import ImageService
    from contextifier.services.tag_service import TagService
    from contextifier.services.chart_service import ChartService
    from contextifier.services.table_service import TableService

logger = logging.getLogger(__name__)


class HwpContentExtractor(BaseContentExtractor):
    """
    Extract text, tables and images from an HWP5 OLE container.
    """

    # ── BaseContentExtractor interface ────────────────────────────────

    def extract_text(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> str:
        ole, bin_data_map = self._unwrap(preprocessed)
        if ole is None:
            return ""

        processed_images: Set[str] = set()
        parts: List[str] = []

        sections = self._sorted_sections(ole)
        for section_path in sections:
            raw = ole.openstream(section_path).read()
            data, ok = decompress_section(raw)
            if not ok:
                continue

            text = self._parse_section(data, ole, bin_data_map, processed_images)
            if not text or not text.strip():
                text = extract_text_raw(data)
            if text and text.strip():
                parts.append(text)

        return "\n".join(parts)

    def extract_tables(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[TableData]:
        # Tables are rendered inline as HTML/text during extract_text
        return []

    def extract_images(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[str]:
        # Images are embedded as tags during extract_text
        return []

    def extract_charts(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[ChartData]:
        # Charts are not separately extracted; they live inside BinData
        return []

    def get_format_name(self) -> str:
        return "hwp"

    # ── Section / tree parsing ────────────────────────────────────────

    def _parse_section(
        self,
        data: bytes,
        ole: Any,
        bin_data_map: Optional[Dict],
        processed_images: Set[str],
    ) -> str:
        try:
            root = HwpRecord.build_tree(data)
            return self._traverse(root, ole, bin_data_map, processed_images)
        except Exception as exc:
            logger.warning("Error parsing HWP section: %s", exc)
            return ""

    def _traverse(
        self,
        record: HwpRecord,
        ole: Any,
        bin_data_map: Optional[Dict],
        processed_images: Set[str],
    ) -> str:
        parts: List[str] = []

        if record.tag_id == HWPTAG_PARA_HEADER:
            return self._process_paragraph(record, ole, bin_data_map, processed_images)

        if record.tag_id == HWPTAG_CTRL_HEADER:
            result = self._process_control(record, ole, bin_data_map, processed_images)
            if result:
                return result

        if record.tag_id == HWPTAG_SHAPE_COMPONENT_PICTURE:
            result = self._process_picture(record, ole, bin_data_map, processed_images)
            if result:
                return result

        if record.tag_id == HWPTAG_PARA_TEXT:
            text = record.get_text().replace("\x0b", "")
            if text:
                parts.append(text)

        for child in record.children:
            t = self._traverse(child, ole, bin_data_map, processed_images)
            if t:
                parts.append(t)

        return "".join(parts)

    # ── Paragraph handling ────────────────────────────────────────────

    def _process_paragraph(
        self,
        record: HwpRecord,
        ole: Any,
        bin_data_map: Optional[Dict],
        processed_images: Set[str],
    ) -> str:
        parts: List[str] = []

        text_rec = next(
            (c for c in record.children if c.tag_id == HWPTAG_PARA_TEXT), None
        )
        text_content = text_rec.get_text() if text_rec else ""

        control_tags = {HWPTAG_CTRL_HEADER, HWPTAG_TABLE}
        controls = [c for c in record.children if c.tag_id in control_tags]

        if "\x0b" in text_content:
            segments = text_content.split("\x0b")
            for i, seg in enumerate(segments):
                parts.append(seg)
                if i < len(controls):
                    parts.append(self._traverse(controls[i], ole, bin_data_map, processed_images))
            for k in range(len(segments) - 1, len(controls)):
                parts.append(self._traverse(controls[k], ole, bin_data_map, processed_images))
        else:
            parts.append(text_content)
            for c in controls:
                parts.append(self._traverse(c, ole, bin_data_map, processed_images))

        parts.append("\n")
        return "".join(parts)

    # ── Control handling ──────────────────────────────────────────────

    def _process_control(
        self,
        record: HwpRecord,
        ole: Any,
        bin_data_map: Optional[Dict],
        processed_images: Set[str],
    ) -> Optional[str]:
        if len(record.payload) < 4:
            return None

        ctrl_id = record.payload[:4][::-1]

        if ctrl_id == b"tbl ":
            return parse_table(
                record, self._traverse, ole, bin_data_map, processed_images
            )

        if ctrl_id == b"gso ":
            return self._process_gso(record, ole, bin_data_map, processed_images)

        return None

    def _process_gso(
        self,
        record: HwpRecord,
        ole: Any,
        bin_data_map: Optional[Dict],
        processed_images: Set[str],
    ) -> Optional[str]:
        """Process Graphic Shape Object — find embedded pictures."""
        pictures = self._find_pictures(record)
        if not pictures:
            return None

        image_parts: List[str] = []
        for pic in pictures:
            tag = self._process_picture(pic, ole, bin_data_map, processed_images)
            if tag:
                image_parts.append(tag)

        return "".join(image_parts) if image_parts else None

    def _find_pictures(self, record: HwpRecord) -> List[HwpRecord]:
        results: List[HwpRecord] = []
        if record.tag_id == HWPTAG_SHAPE_COMPONENT_PICTURE:
            results.append(record)
        for child in record.children:
            results.extend(self._find_pictures(child))
        return results

    # ── Image extraction ──────────────────────────────────────────────

    def _process_picture(
        self,
        record: HwpRecord,
        ole: Any,
        bin_data_map: Optional[Dict],
        processed_images: Set[str],
    ) -> Optional[str]:
        if not bin_data_map or ole is None:
            return None

        bin_list = bin_data_map.get("by_index", [])
        if not bin_list:
            return None

        idx = _extract_bindata_index(record.payload, len(bin_list))
        if idx is not None and 0 < idx <= len(bin_list):
            return self._save_bindata_image(
                ole, bin_list[idx - 1], processed_images
            )

        # Fallback: if only one BinData, use it
        if len(bin_list) == 1:
            return self._save_bindata_image(
                ole, bin_list[0], processed_images
            )

        return None

    def _save_bindata_image(
        self,
        ole: Any,
        entry: tuple,
        processed_images: Set[str],
    ) -> Optional[str]:
        """Locate, decompress, and save an image from BinData."""
        storage_id, ext = entry
        if storage_id <= 0:
            return None

        stream_path = _find_bindata_stream(ole, storage_id, ext)
        if stream_path is None:
            return None

        path_key = "/".join(stream_path)
        if path_key in processed_images:
            return None

        try:
            import zlib
            raw = ole.openstream(stream_path).read()

            # Try decompress
            try:
                img_data = zlib.decompress(raw, -15)
            except zlib.error:
                try:
                    img_data = zlib.decompress(raw)
                except zlib.error:
                    img_data = raw

            if self._image_service is not None:
                tag = self._image_service.save_image(
                    img_data, custom_name=f"hwp_{stream_path[-1]}"
                )
                if tag:
                    processed_images.add(path_key)
                    return f"\n{tag}\n"
        except Exception as exc:
            logger.debug("Failed to extract HWP image: %s", exc)

        return None

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(preprocessed: PreprocessedData):
        """Return (ole, bin_data_map) from PreprocessedData."""
        ole = preprocessed.content
        bdm = preprocessed.resources.get("bin_data_map") if preprocessed.resources else None
        return ole, bdm

    @staticmethod
    def _sorted_sections(ole: Any) -> List:
        """Return BodyText/Section* entries sorted by number."""
        sections = [
            e for e in ole.listdir()
            if len(e) >= 2 and e[0] == STREAM_BODY_TEXT and e[1].startswith("Section")
        ]
        sections.sort(key=lambda x: int(x[1].replace("Section", "")))
        return sections


# ══════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════


def _extract_bindata_index(payload: bytes, list_len: int) -> Optional[int]:
    """
    Heuristically find the 1-based BinData index inside a
    SHAPE_COMPONENT_PICTURE payload.
    """
    if list_len == 0:
        return None

    # Strategy 1: offset 79 (HWP 5.0.3+)
    if len(payload) >= 81:
        val = struct.unpack_from("<H", payload, 79)[0]
        if 0 < val <= list_len:
            return val

    # Strategy 2: offset 8 (older versions)
    if len(payload) >= 10:
        val = struct.unpack_from("<H", payload, 8)[0]
        if 0 < val <= list_len:
            return val

    # Strategy 3: scan common offsets
    for off in (4, 6, 10, 12, 14, 16, 18, 20, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80):
        if len(payload) >= off + 2:
            val = struct.unpack_from("<H", payload, off)[0]
            if 0 < val <= list_len:
                return val

    return None


def _find_bindata_stream(ole: Any, storage_id: int, ext: str) -> Optional[list]:
    """Locate a BinData stream in the OLE container."""
    dirs = ole.listdir()

    # Pattern-match first
    for entry in dirs:
        if entry[0] == "BinData" and len(entry) > 1:
            fname = entry[1].lower()
            expected = f"bin{storage_id:04x}"
            if expected in fname:
                return entry

    # Exact candidates
    for candidate in (
        f"BIN{storage_id:04X}.{ext}",
        f"BIN{storage_id:04x}.{ext}",
        f"BIN{storage_id:04X}.{ext.lower()}",
    ):
        parts = ["BinData", candidate]
        if parts in dirs:
            return parts

    return None


__all__ = ["HwpContentExtractor"]
