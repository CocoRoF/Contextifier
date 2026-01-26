# contextifier/core/processor/hwp5_helper/hwp5_preprocessor.py
"""
HWP 5.0 Preprocessor

Process HWP OLE document after conversion.

Processing Pipeline Position:
    1. HWP5FileConverter.convert() → olefile.OleFileIO
    2. HWP5Preprocessor.preprocess() → PreprocessedData (THIS STEP)
    3. HWP5MetadataExtractor.extract() → DocumentMetadata
    4. Content extraction (body text, tables, images)
"""
import logging
from typing import Any, Dict

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)

logger = logging.getLogger("document-processor.HWP5")


class HWP5Preprocessor(BasePreprocessor):
    """
    HWP 5.0 OLE Document Preprocessor.
    
    Currently a pass-through implementation as HWP processing
    is handled during the content extraction phase using olefile.
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess the converted HWP OLE document.
        
        Args:
            converted_data: olefile.OleFileIO object
            **kwargs: Additional options
            
        Returns:
            PreprocessedData with the OLE object
        """
        metadata: Dict[str, Any] = {}

        if hasattr(converted_data, 'listdir'):
            try:
                streams = converted_data.listdir()
                metadata['stream_count'] = len(streams)
                
                # Check for common HWP streams
                has_body = any('BodyText' in '/'.join(s) for s in streams)
                has_docinfo = any('DocInfo' in '/'.join(s) for s in streams)
                has_bindata = any('BinData' in '/'.join(s) for s in streams)
                
                metadata['has_body_text'] = has_body
                metadata['has_doc_info'] = has_docinfo
                metadata['has_bin_data'] = has_bindata
            except Exception:
                pass

        logger.debug(f"HWP5 preprocessor: pass-through, metadata={metadata}")

        # clean_content is the TRUE SOURCE - contains the OLE object
        return PreprocessedData(
            raw_content=converted_data,
            clean_content=converted_data,  # TRUE SOURCE - olefile.OleFileIO
            encoding="utf-8",
            extracted_resources={},
            metadata=metadata,
        )

    def get_format_name(self) -> str:
        """Return format name."""
        return "HWP 5.0 Preprocessor"

    def validate(self, data: Any) -> bool:
        """Validate if data is an OLE file object."""
        return hasattr(data, 'listdir') and hasattr(data, 'openstream')


__all__ = ['HWP5Preprocessor']
