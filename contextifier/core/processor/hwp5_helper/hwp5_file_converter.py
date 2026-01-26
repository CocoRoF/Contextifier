# contextifier/core/processor/hwp5_helper/hwp5_file_converter.py
"""
HWP 5.0 File Converter

Converts binary HWP data to OLE file object.
"""
from io import BytesIO
from typing import Any, Optional, BinaryIO

from contextifier.core.functions.file_converter import BaseFileConverter


class HWP5FileConverter(BaseFileConverter):
    """
    HWP 5.0 file converter using olefile.
    
    Converts binary HWP (OLE format) data to OleFileIO object.
    """
    
    # OLE magic number
    OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
    
    def convert(
        self,
        file_data: bytes,
        file_stream: Optional[BinaryIO] = None,
        **kwargs
    ) -> Any:
        """
        Convert binary HWP data to OleFileIO object.
        
        Args:
            file_data: Raw binary HWP data
            file_stream: Optional file stream
            **kwargs: Additional options
            
        Returns:
            olefile.OleFileIO object
        """
        import olefile
        
        stream = file_stream if file_stream is not None else BytesIO(file_data)
        stream.seek(0)
        return olefile.OleFileIO(stream)
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "HWP 5.0 Document (OLE)"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is a valid OLE file."""
        if not file_data or len(file_data) < 8:
            return False
        return file_data[:8] == self.OLE_MAGIC
    
    def close(self, converted_object: Any) -> None:
        """Close the OLE file."""
        if converted_object is not None and hasattr(converted_object, 'close'):
            converted_object.close()


__all__ = ['HWP5FileConverter']
