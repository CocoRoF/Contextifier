# contextifier/core/processor/rtf_helper/rtf_file_converter.py
"""
RTF File Converter

RTF uses raw binary directly, so converter just passes through.
All actual processing is done by Preprocessor in Handler.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, BinaryIO, List, Optional

from contextifier.core.functions.file_converter import BaseFileConverter

logger = logging.getLogger("contextify.rtf.converter")


@dataclass
class RTFConvertedData:
    """
    RTF converted data container.
    
    Attributes:
        content: RTF content string (after preprocessing)
        encoding: Detected encoding
        image_tags: List of image tags from preprocessing
        original_size: Original binary data size
        has_images: Whether images were extracted
    """
    content: str
    encoding: str = "cp949"
    image_tags: List[str] = field(default_factory=list)
    original_size: int = 0
    has_images: bool = False
    
    def __post_init__(self):
        """Set has_images based on image_tags."""
        if self.image_tags:
            self.has_images = True


class RTFFileConverter(BaseFileConverter):
    """
    RTF file converter.
    
    RTF uses raw binary directly, so this converter just passes through.
    All actual processing (image extraction, binary removal, decoding)
    is done by RTFPreprocessor called from Handler.
    """
    
    def __init__(self):
        """Initialize RTFFileConverter."""
        self.logger = logger
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> bytes:
        """
        Pass through binary data.
        
        RTF processing uses raw binary, so just return as-is.
        
        Args:
            file_data: Raw binary RTF data
            file_stream: Optional file stream (not used)
            **kwargs: Not used
                
        Returns:
            Original bytes (pass through)
        """
        return file_data
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "RTF Document"
    
    def close(self, converted_object: Any) -> None:
        """Nothing to close."""
        pass


__all__ = [
    'RTFFileConverter',
    'RTFConvertedData',
]
