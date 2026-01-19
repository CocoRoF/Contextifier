# libs/core/processor/base_handler.py
"""
BaseHandler - Abstract base class for document processing handlers

Defines the base interface for all document handlers.
Manages config and ImageProcessor passed from DocumentProcessor
at instance level for reuse by internal methods.

Usage Example:
    class PDFHandler(BaseHandler):
        def extract_text(self, current_file: CurrentFile, extract_metadata: bool = True) -> str:
            # Access self.config, self.image_processor
            ...
"""
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

from libs.core.functions.img_processor import ImageProcessor

if TYPE_CHECKING:
    from libs.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


class BaseHandler(ABC):
    """
    Abstract base class for document handlers.
    
    All handlers inherit from this class.
    config and image_processor are passed at creation and stored as instance variables,
    accessible via self.config, self.image_processor from all internal methods.
    
    Attributes:
        config: Configuration dictionary passed from DocumentProcessor
        image_processor: ImageProcessor instance passed from DocumentProcessor
        logger: Logging instance
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        image_processor: Optional[ImageProcessor] = None
    ):
        """
        Initialize BaseHandler.
        
        Args:
            config: Configuration dictionary (passed from DocumentProcessor)
            image_processor: ImageProcessor instance (passed from DocumentProcessor)
        """
        self._config = config or {}
        self._image_processor = image_processor or ImageProcessor()
        self._logger = logging.getLogger(f"document-processor.{self.__class__.__name__}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Configuration dictionary."""
        return self._config
    
    @property
    def image_processor(self) -> ImageProcessor:
        """ImageProcessor instance."""
        return self._image_processor
    
    @property
    def logger(self) -> logging.Logger:
        """Logger instance."""
        return self._logger
    
    @abstractmethod
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        pass
    
    def get_file_stream(self, current_file: "CurrentFile") -> io.BytesIO:
        """
        Get a fresh BytesIO stream from current_file.
        
        Resets the stream position to the beginning for reuse.
        
        Args:
            current_file: CurrentFile dict
            
        Returns:
            BytesIO stream ready for reading
        """
        stream = current_file.get("file_stream")
        if stream is not None:
            stream.seek(0)
            return stream
        # Fallback: create new stream from file_data
        return io.BytesIO(current_file.get("file_data", b""))
    
    def save_image(self, image_data: bytes, processed_images: Optional[set] = None) -> Optional[str]:
        """
        Save image and return tag.
        
        Convenience method that wraps self.image_processor.save_image().
        
        Args:
            image_data: Image binary data
            processed_images: Set of processed image hashes (for deduplication)
            
        Returns:
            Image tag string or None
        """
        return self._image_processor.save_image(image_data, processed_images=processed_images)


__all__ = ["BaseHandler"]
