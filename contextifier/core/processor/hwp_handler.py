# libs/core/processor/hwp_processor.py
"""
HWP Handler - HWP Legacy Format (2.0/3.0) File Processor

Class-based handler for HWP legacy format files (2.0/3.0).
For HWP 5.0 OLE format files, use HWP5Handler instead.

HWP legacy format (pre-5.0) uses a simple file structure:
- HWP 2.0: Signature 'HWP Document File V2.0'
- HWP 3.0: Signature 'HWP Document File V3.0'
"""
import io
import os
import logging
import traceback
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.chart_extractor import BaseChartExtractor, ChartData
from contextifier.core.processor.hwp_helper import check_file_signature

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor.HWP-Legacy")


class HWPHandler(BaseHandler):
    """HWP Legacy Format (2.0/3.0) File Processing Handler Class"""

    def _create_file_converter(self):
        """Create HWP legacy file converter (minimal)."""
        from contextifier.core.processor.base_handler import BaseFileConverter
        return BaseFileConverter()

    def _create_preprocessor(self):
        """Create HWP legacy preprocessor (minimal)."""
        from contextifier.core.processor.base_handler import BasePreprocessor
        return BasePreprocessor()

    def _create_chart_extractor(self) -> BaseChartExtractor:
        """Create HWP chart extractor (minimal - legacy doesn't have charts)."""
        class LegacyChartExtractor(BaseChartExtractor):
            def extract_all_from_file(self, file_stream) -> List[ChartData]:
                return []
            def extract_charts_from_element(self, element, zip_file=None) -> List[ChartData]:
                return []
        return LegacyChartExtractor(self._chart_processor)

    def _create_metadata_extractor(self):
        """Create HWP legacy metadata extractor (minimal)."""
        from contextifier.core.processor.base_handler import BaseMetadataExtractor
        return BaseMetadataExtractor()

    def _create_format_image_processor(self):
        """Create HWP legacy image processor (minimal)."""
        return self._image_processor

    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from HWP legacy file (2.0/3.0).

        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options

        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")

        # Check file signature to determine format
        file_type = check_file_signature(file_data)
        
        if file_type == "OLE":
            # HWP 5.0 OLE format - redirect to HWP5Handler
            self.logger.info(f"File {file_path} is HWP 5.0 OLE format. Redirecting to HWP5Handler.")
            from contextifier.core.processor.hwp5_handler import HWP5Handler
            hwp5_handler = HWP5Handler(
                config=self.config,
                image_processor=self.format_image_processor,
                page_tag_processor=self._page_tag_processor,
                chart_processor=self._chart_processor
            )
            return hwp5_handler.extract_text(current_file, extract_metadata=extract_metadata)
        
        if file_type == "ZIP":
            # HWPX format - redirect to HWPXHandler
            self.logger.info(f"File {file_path} is HWPX format. Redirecting to HWPXHandler.")
            from contextifier.core.processor.hwpx_handler import HWPXHandler
            hwpx_handler = HWPXHandler(
                config=self.config,
                image_processor=self.format_image_processor,
                page_tag_processor=self._page_tag_processor,
                chart_processor=self._chart_processor
            )
            return hwpx_handler.extract_text(current_file, extract_metadata=extract_metadata)
        
        # Process HWP legacy format (2.0/3.0)
        return self._process_hwp_legacy(current_file)

    def _process_hwp_legacy(self, current_file: "CurrentFile") -> str:
        """
        Process HWP legacy format file (HWP 2.0/3.0).
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            
        Returns:
            Extracted text or error message
        """
        from contextifier.core.processor.hwp_helper.hwp_legacy_parser import (
            HWP2Parser,
            HWP2Config,
        )
        
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"Processing HWP legacy format file: {file_path}")
        
        # Parse using HWP2Parser
        config = HWP2Config(
            min_korean_chars=3,
            skip_formatting_markers=True,
            clean_output=True,
        )
        parser = HWP2Parser(config)
        result = parser.parse(file_data)
        
        if not result.is_supported:
            return f"[HWP {result.version} Format - Not Supported]"
        
        if result.text and result.text.strip():
            self.logger.info(
                f"HWP {result.version}: Extracted {result.total_char_count} chars "
                f"({result.korean_char_count} Korean)"
            )
            return result.text
        
        # Fallback: return unsupported message
        return f"[HWP {result.version} Format - Text extraction failed]"


__all__ = ['HWPHandler']
