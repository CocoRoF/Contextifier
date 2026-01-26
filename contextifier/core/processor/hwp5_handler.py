# contextifier/core/processor/hwp5_handler.py
"""
HWP 5.0 Handler - HWP 5.0 OLE Format File Processor

Class-based handler for HWP 5.0 OLE format files.
Inherits from BaseHandler and provides HWP 5.0-specific processing.

HWP 5.0 is a binary format based on OLE compound document structure.
Key streams:
- FileHeader: Version, compression flags
- DocInfo: Document properties, BinData mappings
- BodyText/Section0, Section1, ...: Document content
- BinData/: Embedded images and OLE objects
"""
import io
import os
import zlib
import logging
import traceback
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

import olefile

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.chart_extractor import BaseChartExtractor
from contextifier.core.processor.hwp5_helper import (
    HWPTAG_PARA_HEADER,
    HWPTAG_PARA_TEXT,
    HWPTAG_CTRL_HEADER,
    HWPTAG_SHAPE_COMPONENT_PICTURE,
    HWPTAG_TABLE,
    HwpRecord,
    decompress_section,
    parse_doc_info,
    extract_text_from_stream_raw,
    find_zlib_streams,
    recover_images_from_raw,
    check_file_signature,
)
from contextifier.core.processor.hwp5_helper.hwp5_chart_extractor import HWP5ChartExtractor
from contextifier.core.processor.hwp5_helper.hwp5_metadata import HWP5MetadataExtractor
from contextifier.core.processor.hwp5_helper.hwp5_image_processor import HWP5ImageProcessor
from contextifier.core.processor.hwp5_helper.hwp5_table_extractor import HWP5TableExtractor
from contextifier.core.processor.hwp5_helper.hwp5_table_processor import HWP5TableProcessor

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile
    from contextifier.core.functions.chart_extractor import ChartData

logger = logging.getLogger("document-processor.HWP5")


class HWP5Handler(BaseHandler):
    """HWP 5.0 OLE Format File Processing Handler Class"""
    
    def __init__(self, *args, **kwargs):
        """Initialize HWP 5.0 handler."""
        super().__init__(*args, **kwargs)
        self._table_extractor: Optional[HWP5TableExtractor] = None
        self._table_processor: Optional[HWP5TableProcessor] = None

    def _create_file_converter(self):
        """Create HWP 5.0-specific file converter."""
        from contextifier.core.processor.hwp5_helper.hwp5_file_converter import HWP5FileConverter
        return HWP5FileConverter()

    def _create_preprocessor(self):
        """Create HWP 5.0-specific preprocessor."""
        from contextifier.core.processor.hwp5_helper.hwp5_preprocessor import HWP5Preprocessor
        return HWP5Preprocessor()

    def _create_chart_extractor(self) -> BaseChartExtractor:
        """Create HWP 5.0-specific chart extractor."""
        return HWP5ChartExtractor(self._chart_processor)

    def _create_metadata_extractor(self):
        """Create HWP 5.0-specific metadata extractor."""
        return HWP5MetadataExtractor()

    def _create_format_image_processor(self):
        """Create HWP 5.0-specific image processor."""
        return HWP5ImageProcessor(
            directory_path=self._image_processor.config.directory_path,
            tag_prefix=self._image_processor.config.tag_prefix,
            tag_suffix=self._image_processor.config.tag_suffix,
            storage_backend=self._image_processor.storage_backend,
        )
    
    @property
    def table_extractor(self) -> HWP5TableExtractor:
        """Get HWP 5.0 table extractor."""
        if self._table_extractor is None:
            self._table_extractor = HWP5TableExtractor()
        return self._table_extractor
    
    @property
    def table_processor(self) -> HWP5TableProcessor:
        """Get HWP 5.0 table processor."""
        if self._table_processor is None:
            self._table_processor = HWP5TableProcessor()
        return self._table_processor

    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from HWP 5.0 file.

        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options

        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")

        # Validate OLE format
        if not self.file_converter.validate(file_data):
            return self._handle_non_ole_file(current_file)

        text_content = []
        processed_images: Set[str] = set()

        try:
            # Step 1: Open OLE file
            file_stream = self.get_file_stream(current_file)

            # Pre-extract all charts
            chart_data_list = self.chart_extractor.extract_all_from_file(file_stream)

            # Convert binary to OLE object
            ole = self.file_converter.convert(file_data, file_stream)

            # Step 2: Preprocess
            preprocessed = self.preprocess(ole)
            ole = preprocessed.clean_content  # TRUE SOURCE

            try:
                if extract_metadata:
                    metadata_text = self._extract_metadata(ole)
                    if metadata_text:
                        text_content.append(metadata_text)
                        text_content.append("")

                bin_data_map = self._parse_docinfo(ole)
                section_texts = self._extract_body_text(ole, bin_data_map, processed_images)
                text_content.extend(section_texts)

                # Process images from BinData
                image_processor = self.format_image_processor
                if hasattr(image_processor, 'process_images_from_bindata'):
                    image_text = image_processor.process_images_from_bindata(ole, processed_images=processed_images)
                else:
                    image_text = ""
                if image_text:
                    text_content.append("\n\n=== Extracted Images (Not Inline) ===\n")
                    text_content.append(image_text)

                # Add pre-extracted charts
                for chart_data in chart_data_list:
                    chart_text = self._format_chart_data(chart_data)
                    if chart_text:
                        text_content.append(chart_text)
            finally:
                # Close OLE object
                self.file_converter.close(ole)

        except Exception as e:
            self.logger.error(f"Error processing HWP 5.0 file: {e}")
            traceback.print_exc()
            return f"Error processing HWP 5.0 file: {str(e)}"

        return "\n".join(text_content)

    def _format_chart_data(self, chart_data: "ChartData") -> str:
        """Format ChartData using ChartProcessor."""
        from contextifier.core.functions.chart_extractor import ChartData

        if not isinstance(chart_data, ChartData):
            return ""

        if chart_data.has_data():
            return self.chart_processor.format_chart_data(
                chart_type=chart_data.chart_type,
                title=chart_data.title,
                categories=chart_data.categories,
                series=chart_data.series
            )
        else:
            return self.chart_processor.format_chart_fallback(
                chart_type=chart_data.chart_type,
                title=chart_data.title
            )

    def _handle_non_ole_file(self, current_file: "CurrentFile") -> str:
        """Handle non-OLE HWP file (corrupted or unknown format)."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")

        # Check file signature
        file_type = check_file_signature(file_data)
        
        if file_type == "ZIP":
            # HWPX format - redirect to HWPX handler
            self.logger.info(f"File {file_path} is HWPX format")
            from contextifier.core.processor.hwpx_handler import HWPXHandler
            hwpx_handler = HWPXHandler(config=self.config, image_processor=self.format_image_processor)
            return hwpx_handler.extract_text(current_file)
        
        if file_type in ("HWP3.0", "HWP2.0", "HWP_LEGACY"):
            # HWP legacy format - redirect to legacy handler
            self.logger.info(f"File {file_path} is HWP legacy format: {file_type}")
            from contextifier.core.processor.hwp_handler import HWPHandler
            hwp_handler = HWPHandler(config=self.config, image_processor=self.format_image_processor)
            return hwp_handler.extract_text(current_file)
        
        # Attempt forensic recovery
        return self._process_corrupted_hwp(current_file)

    def _extract_metadata(self, ole: olefile.OleFileIO) -> str:
        """Extract metadata from OLE file."""
        return self.extract_and_format_metadata(ole)

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
            self.logger.error(f"Error parsing HWP 5.0 section: {e}")
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
            return self._process_table(record, ole, bin_data_map, processed_images)

        if ctrl_id == b'gso ':
            return self._process_gso(record, ole, bin_data_map, processed_images)

        return None

    def _process_table(self, record: 'HwpRecord', ole, bin_data_map, processed_images) -> Optional[str]:
        """Process table using HWP5TableExtractor and HWP5TableProcessor.
        
        Uses the standardized table_extractor/table_processor interface
        instead of legacy parse_table function.
        
        Args:
            record: CTRL_HEADER record containing table
            ole: OLE file object
            bin_data_map: BinData mapping information
            processed_images: Set of processed image paths
            
        Returns:
            Formatted table string (HTML or text)
        """
        try:
            # Set context for table extraction
            self.table_extractor.set_context(
                traverse_callback=self._traverse_tree,
                ole=ole,
                bin_data_map=bin_data_map,
                processed_images=processed_images
            )
            
            # Detect table region from this specific control record
            regions = self.table_extractor.detect_table_regions(record)
            
            if not regions:
                return None
            
            # Extract first table (should be only one for this control)
            region = regions[0]
            table_data = self.table_extractor.extract_table_from_region(record, region)
            
            if table_data is None:
                # Container table (1x1) - extract text directly
                return self._extract_container_table_text(record, ole, bin_data_map, processed_images)
            
            # Use table_processor to format output
            return self.table_processor.format_table_auto(table_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to process HWP 5.0 table: {e}")
            return "[Table Extraction Failed]"
    
    def _extract_container_table_text(self, record: 'HwpRecord', ole, bin_data_map, processed_images) -> str:
        """Extract text from 1x1 container table.
        
        Args:
            record: CTRL_HEADER record
            ole: OLE file object
            bin_data_map: BinData mapping
            processed_images: Processed images set
            
        Returns:
            Cell text content
        """
        from contextifier.core.processor.hwp5_helper.hwp5_constants import HWPTAG_LIST_HEADER
        
        # Find cell record
        cell_records = record.find_children_by_tag(HWPTAG_LIST_HEADER)
        if not cell_records:
            return ""
        
        cell_rec = cell_records[0]
        parts = []
        
        # Extract content using traverse callback
        if cell_rec.children:
            for child in cell_rec.children:
                text = self._traverse_tree(child, ole, bin_data_map, processed_images)
                parts.append(text)
        else:
            # Get paragraph count from payload
            import struct
            para_count = struct.unpack('<H', cell_rec.payload[0:2])[0] if len(cell_rec.payload) >= 2 else 0
            siblings = list(cell_rec.get_next_siblings(para_count))
            for sibling in siblings:
                text = self._traverse_tree(sibling, ole, bin_data_map, processed_images)
                parts.append(text)
        
        return "".join(parts).strip()

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

        image_processor = self.format_image_processor

        # Extract bindata index
        bindata_index = image_processor.extract_bindata_index(record.payload, len(bin_data_list))

        if bindata_index and 0 < bindata_index <= len(bin_data_list):
            storage_id, ext = bin_data_list[bindata_index - 1]
            if storage_id > 0:
                target_stream = image_processor.find_bindata_stream(ole, storage_id, ext)
                if target_stream:
                    return image_processor.extract_and_save_image(ole, target_stream, processed_images)

        if len(bin_data_list) == 1:
            storage_id, ext = bin_data_list[0]
            if storage_id > 0:
                target_stream = image_processor.find_bindata_stream(ole, storage_id, ext)
                if target_stream:
                    return image_processor.extract_and_save_image(ole, target_stream, processed_images)

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

            image_text = recover_images_from_raw(raw_data, image_processor=self.format_image_processor)
            if image_text:
                text_content.append(f"\n\n=== Recovered Images ===\n{image_text}")

        except Exception as e:
            self.logger.error(f"Forensic recovery failed: {e}")
            return f"Forensic recovery failed: {str(e)}"

        if not text_content:
            return "[Forensic Recovery: No text found]"

        return "\n".join(text_content)


__all__ = ['HWP5Handler']
