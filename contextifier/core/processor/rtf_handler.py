# contextifier/core/processor/rtf_handler.py
"""
RTF Handler

Class-based handler for RTF files.
Follows the correct architecture:
1. Converter: Pass through (RTF uses raw binary)
2. Preprocessor: Binary preprocessing (image extraction, \\bin removal)
3. Handler: Sequential processing (metadata → tables → content → result)
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from striprtf.striprtf import rtf_to_text

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.chart_extractor import BaseChartExtractor, NullChartExtractor

# Import from rtf_helper
from contextifier.core.processor.rtf_helper import (
    RTFFileConverter,
    RTFConvertedData,
    RTFMetadataExtractor,
    RTFSourceInfo,
    RTFPreprocessor,
    extract_tables_with_positions,
    extract_inline_content,
    extract_text_only,
    decode_content,
    detect_encoding,
)

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile

logger = logging.getLogger("contextify.rtf.handler")


class RTFHandler(BaseHandler):
    """
    RTF Document Processing Handler.
    
    Processing flow:
    1. file_converter.convert() → bytes (pass through)
    2. preprocessor.preprocess() → PreprocessedData (image extraction, binary cleanup)
    3. decode content → string
    4. metadata_extractor.extract() → DocumentMetadata
    5. extract_tables_with_positions() → List[RTFTable]
    6. extract_inline_content() → str
    7. Build result string
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize RTFHandler."""
        super().__init__(*args, **kwargs)
        self._preprocessor = RTFPreprocessor()
    
    def _create_file_converter(self) -> RTFFileConverter:
        """Create RTF-specific file converter."""
        return RTFFileConverter()
    
    def _create_chart_extractor(self) -> BaseChartExtractor:
        """RTF files do not contain charts. Return NullChartExtractor."""
        return NullChartExtractor(self._chart_processor)
    
    def _create_metadata_extractor(self) -> RTFMetadataExtractor:
        """Create RTF-specific metadata extractor."""
        return RTFMetadataExtractor()
    
    def _create_format_image_processor(self) -> ImageProcessor:
        """Create RTF-specific image processor (use base for now)."""
        return self._image_processor
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from RTF file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"RTF processing: {file_path}")
        
        if not file_data:
            self.logger.error(f"Empty file data: {file_path}")
            return f"[RTF file is empty: {file_path}]"
        
        # Validate RTF format
        if not file_data.strip().startswith(b'{\\rtf'):
            self.logger.warning(f"Invalid RTF format: {file_path}")
            return self._extract_fallback(file_data, extract_metadata)
        
        try:
            # Step 1: Converter - pass through (RTF uses raw binary)
            raw_data: bytes = self.file_converter.convert(file_data)
            
            # Step 2: Preprocessor - extract images, remove binary data
            output_dir = self._get_output_dir(file_path)
            doc_name = Path(file_path).stem if file_path != "unknown" else "document"
            
            preprocessed = self._preprocessor.preprocess(
                raw_data,
                output_dir=output_dir,
                doc_name=doc_name,
            )
            
            clean_content = preprocessed.clean_content
            image_tags = preprocessed.extracted_resources.get("image_tags", [])
            encoding = preprocessed.encoding or "cp949"
            
            # Step 3: Decode to string if still bytes
            if isinstance(clean_content, bytes):
                encoding = detect_encoding(clean_content) or encoding
                content = decode_content(clean_content, encoding)
            else:
                content = clean_content
            
            # Build RTFConvertedData for downstream processing
            converted = RTFConvertedData(
                content=content,
                encoding=encoding,
                image_tags=image_tags,
                original_size=len(file_data),
            )
            
            self.logger.debug(
                f"RTF preprocessed: encoding={encoding}, "
                f"images={len(image_tags)}, size={len(file_data)}"
            )
            
            # Step 4: Extract content
            return self._extract_from_converted(
                converted,
                current_file,
                extract_metadata,
            )
            
        except Exception as e:
            self.logger.error(f"Error in RTF processing: {e}", exc_info=True)
            return self._extract_fallback(file_data, extract_metadata)
    
    def _extract_from_converted(
        self,
        converted: RTFConvertedData,
        current_file: "CurrentFile",
        extract_metadata: bool,
    ) -> str:
        """
        Internal method to extract content from RTFConvertedData.
        
        Args:
            converted: RTFConvertedData object
            current_file: CurrentFile dict
            extract_metadata: Whether to extract metadata
            
        Returns:
            Extracted text
        """
        content = converted.content
        encoding = converted.encoding
        
        result_parts = []
        
        # Step 2: Extract metadata
        if extract_metadata:
            source = RTFSourceInfo(content=content, encoding=encoding)
            metadata = self.metadata_extractor.extract(source)
            metadata_str = self.metadata_extractor.format(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")
        
        # Add page tag
        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")
        
        # Step 3: Extract tables with positions
        tables, table_regions = extract_tables_with_positions(content, encoding)
        
        # Step 4: Extract inline content (preserves table positions)
        inline_content = extract_inline_content(content, table_regions, encoding)
        
        if inline_content:
            result_parts.append(inline_content)
        else:
            # Fallback: separate text and tables
            text_only = extract_text_only(content, encoding)
            if text_only:
                result_parts.append(text_only)
            
            for table in tables:
                if not table.rows:
                    continue
                if table.is_real_table():
                    result_parts.append("\n" + table.to_html() + "\n")
                else:
                    result_parts.append("\n" + table.to_text_list() + "\n")
        
        # Step 5: Add image tags
        if converted.image_tags:
            result_parts.append("\n")
            for tag in converted.image_tags:
                result_parts.append(tag + "\n")
        
        result = "\n".join(result_parts)
        
        # Clean up invalid image tags
        result = re.sub(r'\[image:[^\]]*uploads/\.[^\]]*\]', '', result)
        
        return result
    
    def _extract_fallback(
        self,
        file_data: bytes,
        extract_metadata: bool,
    ) -> str:
        """
        Fallback extraction using striprtf library.
        
        Args:
            file_data: Raw binary data
            extract_metadata: Whether to extract metadata
            
        Returns:
            Extracted text
        """
        # Try different encodings
        content = None
        for encoding in ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
            try:
                content = file_data.decode(encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            content = file_data.decode('cp1252', errors='replace')
        
        result_parts = []
        
        # Extract metadata from raw content
        if extract_metadata:
            source = RTFSourceInfo(content=content, encoding='cp1252')
            metadata = self.metadata_extractor.extract(source)
            metadata_str = self.extract_and_format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")
        
        # Add page tag
        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")
        
        # Extract text using striprtf
        try:
            text = rtf_to_text(content)
        except Exception:
            # Manual cleanup
            text = re.sub(r'\\[a-z]+\d*\s?', '', content)
            text = re.sub(r"\\'[0-9a-fA-F]{2}", '', text)
            text = re.sub(r'[{}]', '', text)
        
        if text:
            text = re.sub(r'\n{3,}', '\n\n', text)
            result_parts.append(text.strip())
        
        return "\n".join(result_parts)
    
    def _get_output_dir(self, file_path: str) -> Optional[Path]:
        """Get output directory for images."""
        if hasattr(self._image_processor, 'config'):
            dir_path = self._image_processor.config.directory_path
            if dir_path:
                return Path(dir_path)
        return None


__all__ = ['RTFHandler']
