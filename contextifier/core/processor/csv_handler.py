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
from contextifier.core.processor.csv_helper import (
    CSVMetadata,
    extract_csv_metadata,
    format_metadata,
    detect_bom,
    detect_delimiter,
    parse_csv_content,
    detect_header,
    convert_rows_to_table,
)

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")

# Encoding candidates for fallback
ENCODING_CANDIDATES = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'iso-8859-1', 'latin-1']


class CSVHandler(BaseHandler):
    """CSV/TSV File Processing Handler Class"""
    
    def _create_chart_extractor(self) -> BaseChartExtractor:
        """CSV files do not contain charts. Return NullChartExtractor."""
        return NullChartExtractor(self._chart_processor)
    
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
            
            # Decode file_data with encoding detection
            file_data = current_file.get("file_data", b"")
            content, detected_encoding = self._decode_with_encoding(file_data, encoding)
            
            if delimiter is None:
                delimiter = detect_delimiter(content)
            
            self.logger.info(f"CSV: encoding={detected_encoding}, delimiter={repr(delimiter)}")
            
            rows = parse_csv_content(content, delimiter)
            
            if not rows:
                return ""
            
            has_header = detect_header(rows)
            
            if extract_metadata:
                metadata = extract_csv_metadata(file_path, detected_encoding, delimiter, rows, has_header)
                metadata_str = format_metadata(metadata)
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
    
    def _decode_with_encoding(
        self,
        file_data: bytes,
        preferred_encoding: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Decode bytes with encoding detection.
        
        Args:
            file_data: Raw bytes data
            preferred_encoding: Preferred encoding (None for auto-detect)
            
        Returns:
            Tuple of (decoded content, detected encoding)
        """
        # BOM detection
        bom_encoding = detect_bom(file_data)
        if bom_encoding:
            try:
                return file_data.decode(bom_encoding), bom_encoding
            except UnicodeDecodeError:
                pass
        
        # Try preferred encoding
        if preferred_encoding:
            try:
                return file_data.decode(preferred_encoding), preferred_encoding
            except UnicodeDecodeError:
                self.logger.debug(f"Preferred encoding {preferred_encoding} failed")
        
        # Try chardet if available
        try:
            import chardet
            detected = chardet.detect(file_data)
            if detected and detected.get('encoding'):
                enc = detected['encoding']
                try:
                    return file_data.decode(enc), enc
                except UnicodeDecodeError:
                    pass
        except ImportError:
            pass
        
        # Try encoding candidates
        for enc in ENCODING_CANDIDATES:
            try:
                return file_data.decode(enc), enc
            except UnicodeDecodeError:
                continue
        
        # Fallback to latin-1 (always succeeds)
        return file_data.decode('latin-1'), 'latin-1'
