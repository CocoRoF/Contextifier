# libs/core/processor/hwpx_processor.py
"""
HWPX Handler - HWPX (ZIP/XML based) Document Processor

Class-based handler for HWPX files inheriting from BaseHandler.
"""
import io
import logging
from typing import Dict, Any, Set, TYPE_CHECKING

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.chart_extractor import BaseChartExtractor
from contextifier.core.functions.table_extractor import BaseTableExtractor
from contextifier.core.processor.hwpx_helper import (
    parse_bin_item_map,
    parse_hwpx_section,
)
from contextifier.core.processor.hwpx_helper.hwpx_chart_extractor import HWPXChartExtractor
from contextifier.core.processor.hwpx_helper.hwpx_metadata import HWPXMetadataExtractor
from contextifier.core.processor.hwpx_helper.hwpx_image_processor import HWPXImageProcessor
from contextifier.core.processor.hwpx_helper.hwpx_table_extractor import HWPXTableExtractor
from contextifier.core.processor.hwpx_helper.hwpx_table_processor import HWPXTableProcessor

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile
    from contextifier.core.functions.chart_extractor import ChartData

logger = logging.getLogger("document-processor")


class HWPXHandler(BaseHandler):
    """HWPX (ZIP/XML based Korean document) Processing Handler Class"""

    def __init__(self, *args, **kwargs):
        """Initialize HWPX handler."""
        super().__init__(*args, **kwargs)
        self._table_extractor = None
        self._table_processor = None

    # ========================================================================
    # Table Extractor and Processor Properties
    # ========================================================================

    @property
    def table_extractor(self) -> BaseTableExtractor:
        """Get or create HWPX table extractor.
        
        Returns:
            HWPXTableExtractor instance
        """
        if self._table_extractor is None:
            self._table_extractor = HWPXTableExtractor()
        return self._table_extractor

    @property
    def table_processor(self) -> HWPXTableProcessor:
        """Get or create HWPX table processor.
        
        Returns:
            HWPXTableProcessor instance
        """
        if self._table_processor is None:
            self._table_processor = HWPXTableProcessor()
        return self._table_processor

    # ========================================================================
    # Factory Methods
    # ========================================================================

    def _create_file_converter(self):
        """Create HWPX-specific file converter."""
        from contextifier.core.processor.hwpx_helper.hwpx_file_converter import HWPXFileConverter
        return HWPXFileConverter()

    def _create_preprocessor(self):
        """Create HWPX-specific preprocessor."""
        from contextifier.core.processor.hwpx_helper.hwpx_preprocessor import HWPXPreprocessor
        return HWPXPreprocessor()

    def _create_chart_extractor(self) -> BaseChartExtractor:
        """Create HWPX-specific chart extractor."""
        return HWPXChartExtractor(self._chart_processor)

    def _create_metadata_extractor(self):
        """Create HWPX-specific metadata extractor."""
        return HWPXMetadataExtractor()

    def _create_format_image_processor(self):
        """Create HWPX-specific image processor."""
        return HWPXImageProcessor(
            directory_path=self._image_processor.config.directory_path,
            tag_prefix=self._image_processor.config.tag_prefix,
            tag_suffix=self._image_processor.config.tag_suffix,
            storage_backend=self._image_processor.storage_backend,
        )

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
        file_data = current_file.get("file_data", b"")
        text_content = []

        # Check if it's a valid ZIP file using file_converter.validate()
        if not self.file_converter.validate(file_data):
            self.logger.error("Not a valid Zip file: %s", file_path)
            return ""

        try:
            # Get file stream
            file_stream = self.get_file_stream(current_file)

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

            # Step 1: Convert binary to ZipFile using file_converter
            zf = self.file_converter.convert(file_data, file_stream)

            # Step 2: Preprocess - clean_content is the TRUE SOURCE
            preprocessed = self.preprocess(zf)
            zf = preprocessed.clean_content  # TRUE SOURCE

            try:
                if extract_metadata:
                    metadata_text = self.extract_and_format_metadata(zf)
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
                        section_text = parse_hwpx_section(xml_content, zf, bin_item_map, processed_images, image_processor=self.format_image_processor)
                        text_content.append(section_text)

                # Use format_image_processor directly
                image_processor = self.format_image_processor
                if hasattr(image_processor, 'get_remaining_images'):
                    remaining_images = image_processor.get_remaining_images(zf, processed_images)
                    if remaining_images and hasattr(image_processor, 'process_images'):
                        image_text = image_processor.process_images(zf, remaining_images)
                        if image_text:
                            text_content.append("\n\n=== Extracted Images (Not Inline) ===\n")
                            text_content.append(image_text)

                # Add pre-extracted charts
                while chart_idx[0] < len(chart_data_list):
                    chart_text = get_next_chart()
                    if chart_text:
                        text_content.append(chart_text)
            finally:
                # Close ZipFile using file_converter
                self.file_converter.close(zf)

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error processing HWPX file: %s", e)
            return f"Error processing HWPX file: {str(e)}"

        return "\n".join(text_content)

    def _is_valid_zip(self, file_stream: io.BytesIO) -> bool:
        """Check if stream is a valid ZIP file."""
        try:
            file_stream.seek(0)
            header = file_stream.read(4)
            file_stream.seek(0)
            return header == b'PK\x03\x04'
        except Exception:  # noqa: BLE001
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
