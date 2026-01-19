# libs/core/processor/hwp_processor.py
"""
HWP Handler - HWP 5.0 OLE Format File Processor

Class-based handler for HWP files inheriting from BaseHandler.
"""
import io
import os
import zlib
import logging
import traceback
import zipfile
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

import olefile

from libs.core.processor.base_handler import BaseHandler
from libs.core.processor.hwp_helper import (
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_CTRL_HEADER,
    HWPTAG_SHAPE_COMPONENT_PICTURE,
    HWPTAG_TABLE,
    HwpRecord,
    decompress_section,
    extract_metadata,
    format_metadata,
    find_bindata_stream,
    extract_bindata_index,
    extract_and_upload_image,
    process_images_from_bindata,
    ChartHelper,
    parse_doc_info,
    parse_table,
    extract_text_from_stream_raw,
    find_zlib_streams,
    recover_images_from_raw,
    check_file_signature,
)

if TYPE_CHECKING:
    from libs.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


class HWPHandler(BaseHandler):
    """HWP 5.0 OLE Format File Processing Handler Class"""
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from HWP file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        # Check if it's an OLE file using bytes
        if not self._is_ole_file(file_data):
            return self._handle_non_ole_file(current_file, extract_metadata)
        
        text_content = []
        processed_images: Set[str] = set()
        
        try:
            # Open OLE file from stream
            file_stream = self.get_file_stream(current_file)
            with olefile.OleFileIO(file_stream) as ole:
                if extract_metadata:
                    metadata_text = self._extract_metadata(ole)
                    if metadata_text:
                        text_content.append(metadata_text)
                        text_content.append("")
                
                bin_data_map = self._parse_docinfo(ole)
                section_texts = self._extract_body_text(ole, bin_data_map, processed_images)
                text_content.extend(section_texts)
                
                image_text = process_images_from_bindata(ole, processed_images=processed_images, image_processor=self.image_processor)
                if image_text:
                    text_content.append("\n\n=== Extracted Images (Not Inline) ===\n")
                    text_content.append(image_text)
                
                chart_texts = self._extract_charts_from_bindata(ole, processed_images)
                if chart_texts:
                    text_content.extend(chart_texts)
        
        except Exception as e:
            self.logger.error(f"Error processing HWP file: {e}")
            return f"Error processing HWP file: {str(e)}"
        
        return "\n".join(text_content)
    
    def _is_ole_file(self, file_data: bytes) -> bool:
        """Check if file data is OLE format."""
        # OLE file signature: D0 CF 11 E0 A1 B1 1A E1
        ole_signature = b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'
        return file_data[:8] == ole_signature
    
    def _handle_non_ole_file(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """Handle non-OLE file."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        # Check if it's a ZIP file (HWPX)
        if file_data[:4] == b'PK\x03\x04':
            self.logger.info(f"File {file_path} is a Zip file. Processing as HWPX.")
            from libs.core.processor.hwps_handler import HWPXHandler
            hwpx_handler = HWPXHandler(config=self.config, image_processor=self.image_processor)
            return hwpx_handler.extract_text(current_file, extract_metadata=extract_metadata)
        
        # Check HWP 3.0 format
        if b'HWP Document File' in file_data[:32]:
            return "[HWP 3.0 Format - Not Supported]"
        
        return self._process_corrupted_hwp(current_file)
    
    def _extract_metadata(self, ole: olefile.OleFileIO) -> str:
        """Extract metadata from OLE file."""
        metadata = extract_metadata(ole)
        return format_metadata(metadata)
    
    def _parse_docinfo(self, ole: olefile.OleFileIO) -> Dict:
        """Parse DocInfo stream."""
        bin_data_by_storage_id, bin_data_list = parse_doc_info(ole)
        return {'by_storage_id': bin_data_by_storage_id, 'by_index': bin_data_list}
    
    def _extract_body_text(self, ole: olefile.OleFileIO, bin_data_map: Dict, processed_images: Set[str]) -> List[str]:
        """Extract text from BodyText sections."""
        text_content = []
        
        body_text_sections = [
            entry for entry in ole.listdir()
            if entry[0] == "BodyText" and entry[1].startswith("Section")
        ]
        body_text_sections.sort(key=lambda x: int(x[1].replace("Section", "")))
        
        for section in body_text_sections:
            stream = ole.openstream(section)
            data = stream.read()
            
            decompressed_data, success = decompress_section(data)
            if not success:
                continue
            
            section_text = self._parse_section(decompressed_data, ole, bin_data_map, processed_images)
            
            if not section_text or not section_text.strip():
                section_text = extract_text_from_stream_raw(decompressed_data)
            
            text_content.append(section_text)
        
        return text_content
    
    def _parse_section(self, data: bytes, ole=None, bin_data_map=None, processed_images=None) -> str:
        """Parse a section."""
        try:
            root = HwpRecord.build_tree(data)
            return self._traverse_tree(root, ole, bin_data_map, processed_images)
        except Exception as e:
            self.logger.error(f"Error parsing HWP section: {e}")
            return ""
    
    def _traverse_tree(self, record: 'HwpRecord', ole=None, bin_data_map=None, processed_images=None) -> str:
        """Traverse record tree."""
        parts = []
        
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
            text = record.get_text().replace('\x0b', '')
            if text:
                parts.append(text)
        
        for child in record.children:
            child_text = self._traverse_tree(child, ole, bin_data_map, processed_images)
            if child_text:
                parts.append(child_text)
        
        if record.tag_id == HWPTAG_PARA_HEADER:
            parts.append("\n")
        
        return "".join(parts)
    
    def _process_paragraph(self, record: 'HwpRecord', ole, bin_data_map, processed_images) -> str:
        """Process PARA_HEADER record."""
        parts = []
        
        text_rec = next((c for c in record.children if c.tag_id == HWPTAG_PARA_TEXT), None)
        text_content = text_rec.get_text() if text_rec else ""
        
        control_tags = [HWPTAG_CTRL_HEADER, HWPTAG_TABLE]
        controls = [c for c in record.children if c.tag_id in control_tags]
        
        if '\x0b' in text_content:
            segments = text_content.split('\x0b')
            for i, segment in enumerate(segments):
                parts.append(segment)
                if i < len(controls):
                    parts.append(self._traverse_tree(controls[i], ole, bin_data_map, processed_images))
            for k in range(len(segments) - 1, len(controls)):
                parts.append(self._traverse_tree(controls[k], ole, bin_data_map, processed_images))
        else:
            parts.append(text_content)
            for c in controls:
                parts.append(self._traverse_tree(c, ole, bin_data_map, processed_images))
        
        parts.append("\n")
        return "".join(parts)
    
    def _process_control(self, record: 'HwpRecord', ole, bin_data_map, processed_images) -> Optional[str]:
        """Process CTRL_HEADER record."""
        if len(record.payload) < 4:
            return None
        
        ctrl_id = record.payload[:4][::-1]
        
        if ctrl_id == b'tbl ':
            return parse_table(record, self._traverse_tree, ole, bin_data_map, processed_images)
        
        if ctrl_id == b'gso ':
            return self._process_gso(record, ole, bin_data_map, processed_images)
        
        return None
    
    def _process_gso(self, record: 'HwpRecord', ole, bin_data_map, processed_images) -> Optional[str]:
        """Process GSO (Graphic Shape Object) record."""
        def find_pictures(rec):
            results = []
            if rec.tag_id == HWPTAG_SHAPE_COMPONENT_PICTURE:
                results.append(rec)
            for child in rec.children:
                results.extend(find_pictures(child))
            return results
        
        pictures = find_pictures(record)
        if pictures:
            image_parts = []
            for pic_rec in pictures:
                img_result = self._process_picture(pic_rec, ole, bin_data_map, processed_images)
                if img_result:
                    image_parts.append(img_result)
            if image_parts:
                return "".join(image_parts)
        
        return None
    
    def _process_picture(self, record: 'HwpRecord', ole, bin_data_map, processed_images) -> Optional[str]:
        """Process SHAPE_COMPONENT_PICTURE record."""
        if not bin_data_map or not ole:
            return None
        
        bin_data_list = bin_data_map.get('by_index', [])
        if not bin_data_list:
            return None
        
        bindata_index = extract_bindata_index(record.payload, len(bin_data_list))
        
        if bindata_index and 0 < bindata_index <= len(bin_data_list):
            storage_id, ext = bin_data_list[bindata_index - 1]
            if storage_id > 0:
                target_stream = find_bindata_stream(ole, storage_id, ext)
                if target_stream:
                    return extract_and_upload_image(ole, target_stream, processed_images, image_processor=self.image_processor)
        
        if len(bin_data_list) == 1:
            storage_id, ext = bin_data_list[0]
            if storage_id > 0:
                target_stream = find_bindata_stream(ole, storage_id, ext)
                if target_stream:
                    return extract_and_upload_image(ole, target_stream, processed_images, image_processor=self.image_processor)
        
        return None
    
    def _extract_charts_from_bindata(self, ole: olefile.OleFileIO, processed_images: Set[str]) -> List[str]:
        """Extract charts from BinData streams."""
        chart_results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.wmf', '.emf'}
        
        try:
            bindata_streams = [e for e in ole.listdir() if len(e) >= 2 and e[0] == "BinData"]
            
            for stream_path in bindata_streams:
                stream_name = stream_path[-1]
                ext = os.path.splitext(stream_name)[1].lower()
                
                if ext in image_extensions:
                    continue
                
                chart_text = self._process_chart_stream(ole, stream_path, processed_images)
                if chart_text:
                    chart_results.append(chart_text)
        except Exception as e:
            self.logger.warning(f"Error extracting charts: {e}")
        
        return chart_results
    
    def _process_chart_stream(self, ole, stream_path, processed_images) -> Optional[str]:
        """Process chart stream."""
        try:
            stream = ole.openstream(stream_path)
            ole_data = stream.read()
            
            try:
                ole_data = zlib.decompress(ole_data, -15)
            except:
                try:
                    ole_data = zlib.decompress(ole_data)
                except:
                    pass
            
            chart_data = ChartHelper.extract_chart_from_ole_stream(ole_data)
            if chart_data:
                return ChartHelper.process_chart(chart_data, processed_images, image_processor=self.image_processor)
        except:
            pass
        
        return None
    
    def _process_corrupted_hwp(self, current_file: "CurrentFile") -> str:
        """Attempt forensic recovery of corrupted HWP file."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"Starting forensic recovery for: {file_path}")
        text_content = []
        
        try:
            raw_data = file_data
            
            file_type = check_file_signature(raw_data)
            if file_type == "HWP3.0":
                return "[HWP 3.0 Format - Not Supported]"
            
            zlib_chunks = find_zlib_streams(raw_data, min_size=50)
            
            for offset, decompressed in zlib_chunks:
                parsed_text = self._parse_section(decompressed)
                if not parsed_text or not parsed_text.strip():
                    parsed_text = extract_text_from_stream_raw(decompressed)
                if parsed_text and len(parsed_text.strip()) > 0:
                    text_content.append(parsed_text)
            
            if not text_content:
                plain_text = extract_text_from_stream_raw(raw_data)
                if plain_text and len(plain_text) > 100:
                    text_content.append(plain_text)
            
            image_text = recover_images_from_raw(raw_data, image_processor=self.image_processor)
            if image_text:
                text_content.append(f"\n\n=== Recovered Images ===\n{image_text}")
        
        except Exception as e:
            self.logger.error(f"Forensic recovery failed: {e}")
            return f"Forensic recovery failed: {str(e)}"
        
        if not text_content:
            return "[Forensic Recovery: No text found]"
        
        return "\n".join(text_content)
