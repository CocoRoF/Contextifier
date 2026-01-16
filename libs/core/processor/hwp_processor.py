"""
HWP (5.0 OLE 형식) 파일 프로세서

HWP 5.0 OLE 형식 파일을 처리하여 텍스트, 테이블, 이미지, 차트를 추출합니다.
HWPX(ZIP 형식)는 hwpx_processor.py에서 처리합니다.

리팩터링된 구조:
- 기능별 로직은 hwp_helper/ 모듈로 분리
- HwpProcessor는 조합 및 조율 역할
"""
import os
import zlib
import logging
import traceback
import zipfile
from typing import List, Dict, Any, Optional, Set

import olefile

from libs.core.processor.hwp_helper import (
    # Constants
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_CTRL_HEADER,
    HWPTAG_SHAPE_COMPONENT_PICTURE,
    HWPTAG_TABLE,
    # Record Parser
    HwpRecord,
    # Decoder
    decompress_section,
    # Metadata
    extract_metadata,
    format_metadata,
    # Image
    find_bindata_stream,
    extract_bindata_index,
    extract_and_upload_image,
    process_images_from_bindata,
    # Chart
    ChartHelper,
    # DocInfo
    parse_doc_info,
    # Table
    parse_table,
    # Recovery
    extract_text_from_stream_raw,
    find_zlib_streams,
    recover_images_from_raw,
    check_file_signature,
)
from libs.core.processor.hwpx_processor import extract_text_from_hwpx

logger = logging.getLogger("document-processor")


class HwpProcessor:
    """HWP 5.0 OLE 형식 파일 프로세서"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    # ==========================================================================
    # Main Entry Point
    # ==========================================================================

    async def process(self, file_path: str, extract_default_metadata: bool = True) -> str:
        """
        HWP (5.0 OLE) 파일을 처리합니다.

        Args:
            file_path: HWP 파일 경로
            extract_default_metadata: 기본 메타데이터 추출 여부 (기본값: True)

        Returns:
            추출된 텍스트 내용
        """
        # OLE 파일인지 확인
        if not olefile.isOleFile(file_path):
            return await self._handle_non_ole_file(file_path, extract_default_metadata)

        text_content = []
        processed_images: Set[str] = set()

        try:
            with olefile.OleFileIO(file_path) as ole:
                # 1. 메타데이터 추출
                if extract_default_metadata:
                    metadata_text = self._extract_metadata(ole)
                    if metadata_text:
                        text_content.append(metadata_text)
                        text_content.append("")

                # 2. DocInfo 파싱
                bin_data_map = self._parse_docinfo(ole)

                # 3. BodyText에서 텍스트 추출
                section_texts = self._extract_body_text(ole, bin_data_map, processed_images)
                text_content.extend(section_texts)

                # 4. BinData에서 이미지 추출
                image_text = await process_images_from_bindata(ole, processed_images=processed_images)
                if image_text:
                    text_content.append("\n\n=== Extracted Images (Not Inline) ===\n")
                    text_content.append(image_text)

                # 5. BinData에서 차트 추출
                chart_texts = self._extract_charts_from_bindata(ole, processed_images)
                if chart_texts:
                    text_content.extend(chart_texts)

        except Exception as e:
            logger.error(f"Error processing HWP file: {e}")
            return f"Error processing HWP file: {str(e)}"

        final_result = "\n".join(text_content)
        logger.info(f"--- Final Extracted Text ({file_path}) ---\n{final_result}\n-----------------------------------")
        return final_result

    # ==========================================================================
    # File Type Detection
    # ==========================================================================

    async def _handle_non_ole_file(self, file_path: str, extract_default_metadata: bool) -> str:
        """비-OLE 파일 처리 (HWPX 또는 손상된 파일)"""
        # HWPX(ZIP) 형식인지 확인 (.hwp 확장자이지만 실제로는 HWPX인 경우)
        if zipfile.is_zipfile(file_path):
            logger.info(f"File {file_path} has .hwp extension but is a Zip file. Processing as HWPX.")
            return await extract_text_from_hwpx(file_path, self.config, extract_default_metadata)

        # HWP 3.0 확인
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
            if b'HWP Document File' in header:
                return "[HWP 3.0 Format - Not Supported]"
        except Exception:
            pass

        logger.error(f"Not a valid OLE file: {file_path}")
        # Fallback: 손상된 파일 복구 시도
        return await self._process_corrupted_hwp(file_path)

    # ==========================================================================
    # Metadata Extraction
    # ==========================================================================

    def _extract_metadata(self, ole: olefile.OleFileIO) -> str:
        """메타데이터 추출"""
        metadata = extract_metadata(ole)
        metadata_text = format_metadata(metadata)
        return metadata_text

    # ==========================================================================
    # DocInfo Parsing
    # ==========================================================================

    def _parse_docinfo(self, ole: olefile.OleFileIO) -> Dict:
        """DocInfo 파싱 및 BinData 매핑 생성"""
        bin_data_by_storage_id, bin_data_list = parse_doc_info(ole)
        return {
            'by_storage_id': bin_data_by_storage_id,
            'by_index': bin_data_list  # 1-based index lookup
        }

    # ==========================================================================
    # BodyText Extraction
    # ==========================================================================

    def _extract_body_text(
        self,
        ole: olefile.OleFileIO,
        bin_data_map: Dict,
        processed_images: Set[str]
    ) -> List[str]:
        """BodyText에서 텍스트 추출"""
        text_content = []

        body_text_sections = [
            entry for entry in ole.listdir()
            if entry[0] == "BodyText" and entry[1].startswith("Section")
        ]
        # 섹션 번호순 정렬
        body_text_sections.sort(key=lambda x: int(x[1].replace("Section", "")))

        for section in body_text_sections:
            stream = ole.openstream(section)
            data = stream.read()

            # 압축 해제
            decompressed_data, success = decompress_section(data)
            if not success:
                logger.warning(f"Failed to decompress section {section}")
                continue

            section_text = self._parse_section(decompressed_data, ole, bin_data_map, processed_images)

            # 표준 파싱 실패 시 raw 스캔 시도
            if not section_text or not section_text.strip():
                logger.warning(f"Standard parsing yielded no text for section {section}, trying raw scan.")
                section_text = extract_text_from_stream_raw(decompressed_data)

            text_content.append(section_text)

        return text_content

    # ==========================================================================
    # Section Parsing
    # ==========================================================================

    def _parse_section(
        self,
        data: bytes,
        ole: olefile.OleFileIO = None,
        bin_data_map: Dict = None,
        processed_images: Set[str] = None
    ) -> str:
        """HWP 5.0 Section 바이너리 데이터를 트리 구조로 파싱합니다."""
        try:
            root = HwpRecord.build_tree(data)
            self._log_section_tags(root)
            return self._traverse_tree(root, ole, bin_data_map, processed_images)
        except Exception as e:
            logger.error(f"Error parsing HWP section: {e}")
            logger.debug(traceback.format_exc())
            return ""

    def _log_section_tags(self, root: HwpRecord):
        """섹션의 태그 분포를 로깅합니다."""
        def count_tags(rec, counts):
            counts[rec.tag_id] = counts.get(rec.tag_id, 0) + 1
            for child in rec.children:
                count_tags(child, counts)

        tag_counts = {}
        count_tags(root, tag_counts)
        logger.info(f"Section tag distribution: {dict(sorted(tag_counts.items()))}")

        if 85 in tag_counts:
            logger.info(f"Found {tag_counts[85]} SHAPE_COMPONENT_PICTURE (Tag 85) records")
        else:
            logger.info("No SHAPE_COMPONENT_PICTURE (Tag 85) found. Checking related tags...")
            for tag_id in [71, 76, 85, 74]:
                if tag_id in tag_counts:
                    logger.info(f"  Tag {tag_id}: {tag_counts[tag_id]} records")

    def _traverse_tree(
        self,
        record: HwpRecord,
        ole: olefile.OleFileIO = None,
        bin_data_map: Dict = None,
        processed_images: Set[str] = None
    ) -> str:
        """HWP 레코드 트리를 순회하며 컨텐츠를 추출합니다."""
        parts = []

        # Paragraph Header 처리 (컨테이너)
        if record.tag_id == HWPTAG_PARA_HEADER:
            return self._process_paragraph(record, ole, bin_data_map, processed_images)

        # Control Header (테이블, GSO 등)
        if record.tag_id == HWPTAG_CTRL_HEADER:
            result = self._process_control(record, ole, bin_data_map, processed_images)
            if result:
                return result

        # Picture (Inline Image)
        if record.tag_id == HWPTAG_SHAPE_COMPONENT_PICTURE:
            result = self._process_picture(record, ole, bin_data_map, processed_images)
            if result:
                return result

        # Text (PARA_HEADER에서 처리되지 않은 경우의 fallback)
        if record.tag_id == HWPTAG_PARA_TEXT:
            text = record.get_text()
            text = text.replace('\x0b', '')
            if text:
                parts.append(text)

        # 재귀
        for child in record.children:
            child_text = self._traverse_tree(child, ole, bin_data_map, processed_images)
            if child_text:
                parts.append(child_text)

        if record.tag_id == HWPTAG_PARA_HEADER:
            parts.append("\n")

        return "".join(parts)

    def _process_paragraph(
        self,
        record: HwpRecord,
        ole: olefile.OleFileIO,
        bin_data_map: Dict,
        processed_images: Set[str]
    ) -> str:
        """PARA_HEADER 레코드를 처리합니다."""
        parts = []

        # 1. Text Record 찾기
        text_rec = next((c for c in record.children if c.tag_id == HWPTAG_PARA_TEXT), None)
        text_content = text_rec.get_text() if text_rec else ""

        # 2. Control Records 찾기 (순서대로)
        control_tags = [HWPTAG_CTRL_HEADER, HWPTAG_TABLE]
        controls = [c for c in record.children if c.tag_id in control_tags]

        # 3. \x0b로 텍스트 분할 및 인터리브
        if '\x0b' in text_content:
            segments = text_content.split('\x0b')
            logger.debug(f"PARA_HEADER: Found {len(segments)-1} placeholders in text")

            for i, segment in enumerate(segments):
                parts.append(segment)
                if i < len(controls):
                    control_text = self._traverse_tree(controls[i], ole, bin_data_map, processed_images)
                    parts.append(control_text)

            # 남은 컨트롤 처리
            for k in range(len(segments) - 1, len(controls)):
                control_text = self._traverse_tree(controls[k], ole, bin_data_map, processed_images)
                parts.append(control_text)
        else:
            parts.append(text_content)
            for c in controls:
                parts.append(self._traverse_tree(c, ole, bin_data_map, processed_images))

        parts.append("\n")
        return "".join(parts)

    def _process_control(
        self,
        record: HwpRecord,
        ole: olefile.OleFileIO,
        bin_data_map: Dict,
        processed_images: Set[str]
    ) -> Optional[str]:
        """CTRL_HEADER 레코드를 처리합니다."""
        if len(record.payload) < 4:
            return None

        ctrl_id = record.payload[:4][::-1]
        logger.debug(f"CTRL_HEADER found with ID: {ctrl_id}, has {len(record.children)} children")

        # 테이블 처리 - hwp_helper의 parse_table 사용
        if ctrl_id == b'tbl ':
            return parse_table(
                record,
                self._traverse_tree,
                ole,
                bin_data_map,
                processed_images
            )

        # GSO (Graphic Shape Object) 처리
        if ctrl_id == b'gso ':
            return self._process_gso(record, ole, bin_data_map, processed_images)

        return None

    def _process_gso(
        self,
        record: HwpRecord,
        ole: olefile.OleFileIO,
        bin_data_map: Dict,
        processed_images: Set[str]
    ) -> Optional[str]:
        """GSO (Graphic Shape Object) 처리"""
        logger.debug(f"Found GSO with {len(record.children)} children")

        def find_pictures(rec, depth=0):
            results = []
            if rec.tag_id == HWPTAG_SHAPE_COMPONENT_PICTURE:
                results.append(rec)
            for child in rec.children:
                results.extend(find_pictures(child, depth + 1))
            return results

        pictures = find_pictures(record)
        logger.debug(f"Found {len(pictures)} SHAPE_COMPONENT_PICTURE in GSO")

        if pictures:
            image_parts = []
            for pic_rec in pictures:
                img_result = self._process_picture(pic_rec, ole, bin_data_map, processed_images)
                if img_result:
                    image_parts.append(img_result)
            if image_parts:
                return "".join(image_parts)

        return None

    def _process_picture(
        self,
        record: HwpRecord,
        ole: olefile.OleFileIO,
        bin_data_map: Dict,
        processed_images: Set[str]
    ) -> Optional[str]:
        """SHAPE_COMPONENT_PICTURE 레코드를 처리합니다."""
        logger.debug(f"Processing SHAPE_COMPONENT_PICTURE (Tag 85), payload size: {len(record.payload)}")

        if not bin_data_map or not ole:
            logger.debug("Missing bin_data_map or ole, cannot process picture")
            return None

        bin_data_list = bin_data_map.get('by_index', [])
        if not bin_data_list:
            return self._try_fallback_image(ole, bin_data_map, processed_images)

        # BinData 인덱스 추출
        bindata_index = extract_bindata_index(record.payload, len(bin_data_list))

        if bindata_index and 0 < bindata_index <= len(bin_data_list):
            storage_id, ext = bin_data_list[bindata_index - 1]
            logger.debug(f"BinData index {bindata_index} -> storage_id={storage_id}, ext='{ext}'")

            if storage_id > 0:
                target_stream = find_bindata_stream(ole, storage_id, ext)
                if target_stream:
                    return extract_and_upload_image(ole, target_stream, processed_images)
                else:
                    logger.warning(f"Could not find stream for storage_id={storage_id}, ext={ext}")
        else:
            logger.warning("No valid BinData index found in SHAPE_COMPONENT_PICTURE")

            # Fallback: bin_data_list에 하나만 있으면 사용
            if len(bin_data_list) == 1:
                storage_id, ext = bin_data_list[0]
                logger.info(f"Only one BinData available, using it: storage_id={storage_id}")
                if storage_id > 0:
                    target_stream = find_bindata_stream(ole, storage_id, ext)
                    if target_stream:
                        return extract_and_upload_image(ole, target_stream, processed_images)

        return None

    def _try_fallback_image(
        self,
        ole: olefile.OleFileIO,
        bin_data_map: Dict,
        processed_images: Set[str]
    ) -> Optional[str]:
        """bin_data_list가 비어있을 때 대체 방법으로 이미지 추출"""
        logger.debug("bin_data_list is empty, trying direct stream scan")
        bin_data_by_storage_id = bin_data_map.get('by_storage_id', {})
        if bin_data_by_storage_id:
            first_storage_id = list(bin_data_by_storage_id.keys())[0]
            storage_id, ext = bin_data_by_storage_id[first_storage_id]
            target_stream = find_bindata_stream(ole, storage_id, ext)
            if target_stream:
                return extract_and_upload_image(ole, target_stream, processed_images)
        return None

    # ==========================================================================
    # Chart Extraction
    # ==========================================================================

    def _extract_charts_from_bindata(
        self,
        ole: olefile.OleFileIO,
        processed_images: Set[str] = None
    ) -> List[str]:
        """BinData 스토리지에서 차트가 포함된 OLE 객체를 스캔합니다."""
        chart_results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.wmf', '.emf'}

        try:
            bindata_streams = [
                entry for entry in ole.listdir()
                if len(entry) >= 2 and entry[0] == "BinData"
            ]

            for stream_path in bindata_streams:
                stream_name = stream_path[-1]
                ext = os.path.splitext(stream_name)[1].lower()

                # 이미지 형식은 스킵
                if ext in image_extensions:
                    continue

                chart_text = self._process_chart_stream(ole, stream_path, processed_images)
                if chart_text:
                    chart_results.append(chart_text)
                    logger.info(f"Extracted chart from {stream_name}")

        except Exception as e:
            logger.warning(f"Error extracting charts from BinData: {e}")

        return chart_results

    def _process_chart_stream(
        self,
        ole: olefile.OleFileIO,
        stream_path: List[str],
        processed_images: Set[str]
    ) -> Optional[str]:
        """차트 스트림을 처리합니다."""
        try:
            stream = ole.openstream(stream_path)
            ole_data = stream.read()

            # 압축 해제 시도
            ole_data = self._try_decompress(ole_data)

            chart_data = ChartHelper.extract_chart_from_ole_stream(ole_data)
            if chart_data:
                return ChartHelper.process_chart(chart_data, processed_images)

        except Exception as e:
            logger.debug(f"Failed to extract chart from {stream_path[-1]}: {e}")

        return None

    def _try_decompress(self, data: bytes) -> bytes:
        """zlib 압축 해제 시도"""
        try:
            return zlib.decompress(data, -15)
        except Exception:
            pass
        try:
            return zlib.decompress(data)
        except Exception:
            pass
        return data

    # ==========================================================================
    # Corrupted File Recovery
    # ==========================================================================

    async def _process_corrupted_hwp(self, file_path: str) -> str:
        """손상되었거나 비-OLE HWP 파일에서 텍스트와 이미지 복구를 시도합니다."""
        logger.info(f"Starting forensic recovery for: {file_path}")
        text_content = []

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()

            logger.info(f"File size: {len(raw_data)} bytes")

            # 파일 시그니처 확인
            file_type = check_file_signature(raw_data)
            if file_type:
                logger.info(f"File signature: {file_type}")
                if file_type == "HWP3.0":
                    return "[HWP 3.0 Format - Not Supported by this processor]"

            # 텍스트 복구
            text_chunks = self._recover_text_from_raw(raw_data)
            if text_chunks:
                text_content.append("[Forensic Recovery: Found potential text streams]")
                text_content.extend(text_chunks)
            else:
                logger.warning("Forensic text scan found no valid text.")

            # 이미지 복구
            image_text = await self._recover_images_from_raw(raw_data)
            if image_text:
                text_content.append(image_text)

        except Exception as e:
            logger.error(f"Forensic recovery failed completely: {e}")
            return f"Forensic recovery failed: {str(e)}"

        if not text_content:
            return "[Forensic Recovery: No text found]"

        return "\n".join(text_content)

    def _recover_text_from_raw(self, raw_data: bytes) -> List[str]:
        """raw 데이터에서 텍스트 복구"""
        zlib_chunks = find_zlib_streams(raw_data, min_size=50)
        logger.info(f"Found {len(zlib_chunks)} zlib streams")

        decompressed_chunks = []
        for offset, decompressed in zlib_chunks:
            parsed_text = self._parse_section(decompressed)

            if not parsed_text or not parsed_text.strip():
                parsed_text = extract_text_from_stream_raw(decompressed)
                if parsed_text:
                    logger.info(f"Stream at {offset}: Parsed via raw scan ({len(parsed_text)} chars)")
            else:
                logger.info(f"Stream at {offset}: Parsed via record tree ({len(parsed_text)} chars)")

            if parsed_text and len(parsed_text.strip()) > 0:
                decompressed_chunks.append(parsed_text)

        if not decompressed_chunks:
            logger.info("No zlib streams found. Attempting plain text scan.")
            plain_text = extract_text_from_stream_raw(raw_data)
            if plain_text and len(plain_text) > 100:
                logger.info(f"Found plain text content ({len(plain_text)} chars).")
                decompressed_chunks.append(plain_text)

        return decompressed_chunks

    async def _recover_images_from_raw(self, raw_data: bytes) -> Optional[str]:
        """raw 데이터에서 이미지 복구"""
        logger.info("Starting forensic image recovery...")
        try:
            image_text = await recover_images_from_raw(raw_data)
            if image_text:
                return f"\n\n=== Forensically Recovered Images ===\n{image_text}"
            logger.info("Forensic image recovery complete.")
        except Exception as e:
            logger.error(f"Forensic image recovery failed (skipping): {e}")
            return "\n[Forensic Image Recovery Failed]"
        return None


# ==========================================================================
# Convenience Functions
# ==========================================================================

async def extract_text_from_hwp(
    file_path: str,
    config: Dict[str, Any] = None,
    extract_default_metadata: bool = True
) -> str:
    """HWP 파일에서 텍스트를 추출하는 편의 함수"""
    processor = HwpProcessor(config)
    return await processor.process(file_path, extract_default_metadata=extract_default_metadata)
