# libs/core/processor/hwpx_processor.py
"""
HWPX Handler - HWPX (ZIP/XML based) Document Processor

Class-based handler for HWPX files inheriting from BaseHandler.
"""
import io
import zipfile
import logging
from typing import Dict, Any, Set, TYPE_CHECKING

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.chart_extractor import BaseChartExtractor
from contextifier.core.processor.hwp_helper import MetadataHelper
from contextifier.core.processor.hwpx_helper import (
    extract_hwpx_metadata,
    parse_bin_item_map,
    parse_hwpx_section,
    process_hwpx_images,
    get_remaining_images,
)
from contextifier.core.processor.hwpx_helper.hwpx_chart_extractor import HWPXChartExtractor

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile
    from contextifier.core.functions.chart_extractor import ChartData

logger = logging.getLogger("document-processor")


class HWPXHandler(BaseHandler):
    """HWPX (ZIP/XML based Korean document) Processing Handler Class"""
    
    def _create_chart_extractor(self) -> BaseChartExtractor:
        """Create HWPX-specific chart extractor."""
        return HWPXChartExtractor(self._chart_processor)
    
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
            
            # Pre-extract all charts using ChartExtractor
            chart_data_list = self.chart_extractor.extract_all_from_file(file_stream)
            chart_idx = [0]  # Mutable container for closure
            
            def get_next_chart() -> str:
                """Callback to get the next pre-extracted chart content."""
                if chart_idx[0] < len(chart_data_list):
                    chart_data = chart_data_list[chart_idx[0]]
                    chart_idx[0] += 1
                    return self._format_chart_data(chart_data)
                return ""
            
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
                
                # Add pre-extracted charts
                while chart_idx[0] < len(chart_data_list):
                    chart_text = get_next_chart()
                    if chart_text:
                        text_content.append(chart_text)
        
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
