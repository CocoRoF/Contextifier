# libs/core/functions/preprocessor.py
"""
BasePreprocessor - Abstract base class for binary preprocessing

Defines the interface for preprocessing binary file data before conversion.
Used when raw binary data needs special handling before it can be parsed.

The preprocessor's job is to:
1. Clean raw binary data (remove problematic sections)
2. Extract embedded resources (images, etc.)
3. Detect encoding information
4. Return preprocessed data ready for further processing

This is an OPTIONAL step BEFORE FileConverter in the processing pipeline:
    Binary Data → Preprocessor (optional) → FileConverter → Handler Processing

Usage:
    class RTFPreprocessor(BasePreprocessor):
        def preprocess(self, file_data: bytes, **kwargs) -> PreprocessedData:
            # Clean binary data, extract images, detect encoding
            return PreprocessedData(
                clean_content=clean_bytes,
                encoding="cp949",
                extracted_resources={"images": image_tags}
            )
        
        def get_format_name(self) -> str:
            return "RTF Preprocessor"
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PreprocessedData:
    """
    Result of preprocessing operation.
    
    Contains cleaned content and any extracted resources.
    
    Attributes:
        raw_content: Original binary content (for reference)
        clean_content: Cleaned binary content ready for parsing
        encoding: Detected or default encoding
        extracted_resources: Dict of extracted resources (images, etc.)
        metadata: Any metadata discovered during preprocessing
    """
    raw_content: bytes = b""
    clean_content: bytes = b""
    encoding: str = "utf-8"
    extracted_resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePreprocessor(ABC):
    """
    Abstract base class for file preprocessors.
    
    Preprocesses raw binary file data before conversion.
    Used when binary data contains embedded resources or requires
    special handling before parsing.
    
    Subclasses must implement:
    - preprocess(): Process binary data and return PreprocessedData
    - get_format_name(): Return human-readable format name
    """
    
    @abstractmethod
    def preprocess(
        self,
        file_data: bytes,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess binary file data.
        
        Args:
            file_data: Raw binary file data
            **kwargs: Additional format-specific options
            
        Returns:
            PreprocessedData containing cleaned content and extracted resources
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """
        Return human-readable format name.
        
        Returns:
            Format name string (e.g., "RTF Preprocessor")
        """
        pass
    
    def validate(self, file_data: bytes) -> bool:
        """
        Validate if the file data can be preprocessed by this preprocessor.
        
        Override this method to add format-specific validation.
        Default implementation returns True.
        
        Args:
            file_data: Raw binary file data
            
        Returns:
            True if file can be preprocessed, False otherwise
        """
        return True


class NullPreprocessor(BasePreprocessor):
    """
    Null preprocessor that passes data through unchanged.
    
    Used as default when no preprocessing is needed.
    """
    
    def preprocess(
        self,
        file_data: bytes,
        **kwargs
    ) -> PreprocessedData:
        """Pass data through unchanged."""
        encoding = kwargs.get("encoding", "utf-8")
        return PreprocessedData(
            raw_content=file_data,
            clean_content=file_data,
            encoding=encoding,
        )
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "Null Preprocessor (pass-through)"


__all__ = [
    'BasePreprocessor',
    'NullPreprocessor', 
    'PreprocessedData',
]
