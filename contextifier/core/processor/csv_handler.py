# libs/core/processor/csv_handler.py
"""
CSV Handler - CSV/TSV File Processor

Class-based handler for CSV/TSV files inheriting from BaseHandler.
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.chart_extractor import BaseChartExtractor, NullChartExtractor
from contextifier.core.functions.table_extractor import BaseTableExtractor
from contextifier.core.processor.csv_helper import (
    detect_bom,
    detect_delimiter,
    parse_csv_content,
    detect_header,
    convert_rows_to_table,
)
from contextifier.core.processor.csv_helper.csv_metadata import CSVMetadataExtractor, CSVSourceInfo
from contextifier.core.processor.csv_helper.csv_image_processor import CSVImageProcessor
from contextifier.core.processor.csv_helper.csv_table_extractor import CSVTableExtractor
from contextifier.core.processor.csv_helper.csv_table_processor import CSVTableProcessor
from contextifier.core.functions.img_processor import ImageProcessor

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")

# Encoding candidates for fallback
ENCODING_CANDIDATES = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'iso-8859-1', 'latin-1']


class CSVHandler(BaseHandler):
    """CSV/TSV File Processing Handler Class"""

    def __init__(self, *args, **kwargs):
        """Initialize CSV handler."""
        super().__init__(*args, **kwargs)
        self._table_extractor = None
        self._table_processor = None

    # ========================================================================
    # Table Extractor and Processor Properties
    # ========================================================================

    @property
    def table_extractor(self) -> BaseTableExtractor:
        """Get or create CSV table extractor.
        
        Returns:
            CSVTableExtractor instance
        """
        if self._table_extractor is None:
            self._table_extractor = CSVTableExtractor()
        return self._table_extractor

    @property
    def table_processor(self) -> CSVTableProcessor:
        """Get or create CSV table processor.
        
        Returns:
            CSVTableProcessor instance
        """
        if self._table_processor is None:
            self._table_processor = CSVTableProcessor()
        return self._table_processor

    # ========================================================================
    # Factory Methods
    # ========================================================================

    def _create_file_converter(self):
        """Create CSV-specific file converter."""
        from contextifier.core.processor.csv_helper.csv_file_converter import CSVFileConverter
        return CSVFileConverter()

    def _create_preprocessor(self):
        """Create CSV-specific preprocessor."""
        from contextifier.core.processor.csv_helper.csv_preprocessor import CSVPreprocessor
        return CSVPreprocessor()

    def _create_chart_extractor(self) -> BaseChartExtractor:
        """CSV files do not contain charts. Return NullChartExtractor."""
        return NullChartExtractor(self._chart_processor)

    def _create_metadata_extractor(self):
        """Create CSV-specific metadata extractor."""
        return CSVMetadataExtractor()

    def _create_format_image_processor(self) -> ImageProcessor:
        """Create CSV-specific image processor."""
        return CSVImageProcessor()

    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Extract text from CSV/TSV file.

        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            encoding: Encoding (None for auto-detect)
            delimiter: Delimiter (None for auto-detect)
            **kwargs: Additional options

        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        ext = current_file.get("file_extension", os.path.splitext(file_path)[1]).lower()
        self.logger.info(f"CSV processing: {file_path}, ext: {ext}")

        if ext == '.tsv' and delimiter is None:
            delimiter = '\t'

        try:
            result_parts = []

            # Step 1: Decode file_data using file_converter
            file_data = current_file.get("file_data", b"")
            content, detected_encoding = self.file_converter.convert(file_data, encoding=encoding)

            # Step 2: Preprocess - clean_content is the TRUE SOURCE
            preprocessed = self.preprocess(content)
            content = preprocessed.clean_content  # TRUE SOURCE

            if delimiter is None:
                delimiter = detect_delimiter(content)

            self.logger.info(f"CSV: encoding={detected_encoding}, delimiter={repr(delimiter)}")

            rows = parse_csv_content(content, delimiter)

            if not rows:
                return ""

            has_header = detect_header(rows)

            if extract_metadata:
                source_info = CSVSourceInfo(
                    file_path=file_path,
                    encoding=detected_encoding,
                    delimiter=delimiter,
                    rows=rows,
                    has_header=has_header
                )
                metadata_str = self.extract_and_format_metadata(source_info)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")

            table = convert_rows_to_table(rows, has_header)
            if table:
                result_parts.append(table)

            result = "".join(result_parts)
            self.logger.info(f"CSV processing completed: {len(rows)} rows")

            return result

        except Exception as e:
            self.logger.error(f"Error extracting text from CSV {file_path}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise
