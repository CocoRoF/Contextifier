# libs/core/processor/hwpx_processor.py
"""
HWPX Handler - HWPX (ZIP/XML based) Document Processor

Class-based handler for HWPX files inheriting from BaseHandler.
"""
import io
import zipfile
import logging
from typing import Dict, Any, Set, TYPE_CHECKING

from libs.core.processor.base_handler import BaseHandler
from libs.core.processor.hwp_helper import MetadataHelper
from libs.core.processor.hwpx_helper import (
    extract_hwpx_metadata,
    parse_bin_item_map,
    parse_hwpx_section,
    process_hwpx_images,
    get_remaining_images,
    extract_charts_from_hwpx,
)

if TYPE_CHECKING:
    from libs.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


class HWPXHandler(BaseHandler):
    """HWPX (ZIP/XML based Korean document) Processing Handler Class"""
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from HWPX file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        text_content = []
        
        try:
            # Open ZIP from stream
            file_stream = self.get_file_stream(current_file)
            
            # Check if valid ZIP
            if not self._is_valid_zip(file_stream):
                self.logger.error(f"Not a valid Zip file: {file_path}")
                return ""
            
            # Reset stream position
            file_stream.seek(0)
            
            with zipfile.ZipFile(file_stream, 'r') as zf:
                if extract_metadata:
                    metadata = extract_hwpx_metadata(zf)
                    metadata_text = MetadataHelper.format_metadata(metadata)
                    if metadata_text:
                        text_content.append(metadata_text)
                        text_content.append("")
                
                bin_item_map = parse_bin_item_map(zf)
                
                section_files = [
                    f for f in zf.namelist() 
                    if f.startswith("Contents/section") and f.endswith(".xml")
                ]
                section_files.sort(key=lambda x: int(x.replace("Contents/section", "").replace(".xml", "")))
                
                processed_images: Set[str] = set()
                
                for sec_file in section_files:
                    with zf.open(sec_file) as f:
                        xml_content = f.read()
                        section_text = parse_hwpx_section(xml_content, zf, bin_item_map, processed_images, image_processor=self.image_processor)
                        text_content.append(section_text)
                
                remaining_images = get_remaining_images(zf, processed_images)
                if remaining_images:
                    image_text = process_hwpx_images(zf, remaining_images, image_processor=self.image_processor)
                    if image_text:
                        text_content.append("\n\n=== Extracted Images (Not Inline) ===\n")
                        text_content.append(image_text)
                
                chart_texts = extract_charts_from_hwpx(zf)
                if chart_texts:
                    text_content.extend(chart_texts)
        
        except Exception as e:
            self.logger.error(f"Error processing HWPX file: {e}")
            return f"Error processing HWPX file: {str(e)}"
        
        return "\n".join(text_content)
    
    def _is_valid_zip(self, file_stream: io.BytesIO) -> bool:
        """Check if stream is a valid ZIP file."""
        try:
            file_stream.seek(0)
            header = file_stream.read(4)
            file_stream.seek(0)
            return header == b'PK\x03\x04'
        except:
            return False
