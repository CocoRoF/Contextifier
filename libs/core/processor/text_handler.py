# libs/core/processor/text_handler.py
"""
Text Handler - Text File Processor

Class-based handler for text files inheriting from BaseHandler.
"""
import logging
from typing import List, Optional, TYPE_CHECKING

from libs.core.processor.base_handler import BaseHandler
from libs.core.functions.utils import clean_text, clean_code_text

if TYPE_CHECKING:
    from libs.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


DEFAULT_ENCODINGS = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1', 'ascii']


class TextHandler(BaseHandler):
    """Text File Processing Handler Class"""
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        file_type: Optional[str] = None,
        encodings: Optional[List[str]] = None,
        is_code: bool = False,
        **kwargs
    ) -> str:
        """
        Extract text from text file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata (ignored for text files)
            file_type: File type (extension)
            encodings: List of encodings to try
            is_code: Whether this is a code file
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        enc = encodings or DEFAULT_ENCODINGS
        
        for e in enc:
            try:
                text = file_data.decode(e)
                self.logger.info(f"Successfully decoded {file_path} with {e} encoding")
                return clean_code_text(text) if is_code else clean_text(text)
            except UnicodeDecodeError:
                self.logger.debug(f"Failed to decode {file_path} with {e}, trying next...")
                continue
            except Exception as ex:
                self.logger.error(f"Error decoding file {file_path} with {e}: {ex}")
                continue
        
        raise Exception(f"Could not decode file {file_path} with any supported encoding")
