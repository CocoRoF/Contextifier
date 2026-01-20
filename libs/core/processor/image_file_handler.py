# libs/core/processor/image_file_handler.py
"""
Image File Handler - Image File Processor

Class-based handler for image files (jpg, jpeg, png, gif, bmp, webp).
Converts images to text using OCR engine when available.
If no OCR engine is provided, returns a placeholder or empty string.
"""
import logging
import os
from typing import Any, Optional, TYPE_CHECKING

from libs.core.processor.base_handler import BaseHandler

if TYPE_CHECKING:
    from libs.core.document_processor import CurrentFile
    from libs.ocr.base import BaseOCR

logger = logging.getLogger("document-processor")


# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = frozenset(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'])


class ImageFileHandler(BaseHandler):
    """
    Image File Processing Handler Class.
    
    Processes standalone image files by converting them to text using OCR.
    Requires an OCR engine to be provided for actual text extraction.
    
    Args:
        config: Configuration dictionary (passed from DocumentProcessor)
        image_processor: ImageProcessor instance (passed from DocumentProcessor)
        page_tag_processor: PageTagProcessor instance (passed from DocumentProcessor)
        ocr_engine: OCR engine instance (BaseOCR subclass) for image-to-text conversion
    
    Example:
        >>> from libs.ocr.ocr_engine import OpenAIOCR
        >>> ocr = OpenAIOCR(api_key="sk-...", model="gpt-4o")
        >>> handler = ImageFileHandler(ocr_engine=ocr)
        >>> text = handler.extract_text(current_file)
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        image_processor: Optional[Any] = None,
        page_tag_processor: Optional[Any] = None,
        ocr_engine: Optional["BaseOCR"] = None
    ):
        """
        Initialize ImageFileHandler.
        
        Args:
            config: Configuration dictionary (passed from DocumentProcessor)
            image_processor: ImageProcessor instance (passed from DocumentProcessor)
            page_tag_processor: PageTagProcessor instance (passed from DocumentProcessor)
            ocr_engine: OCR engine instance (BaseOCR subclass) for image-to-text conversion.
                       If None, images cannot be converted to text.
        """
        super().__init__(
            config=config,
            image_processor=image_processor,
            page_tag_processor=page_tag_processor
        )
        self._ocr_engine = ocr_engine
    
    @property
    def ocr_engine(self) -> Optional["BaseOCR"]:
        """Current OCR engine instance."""
        return self._ocr_engine
    
    @ocr_engine.setter
    def ocr_engine(self, engine: Optional["BaseOCR"]) -> None:
        """Set OCR engine instance."""
        self._ocr_engine = engine
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from image file using OCR.
        
        Converts the image file to text using the configured OCR engine.
        If no OCR engine is available, returns an error message.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata (not used for images)
            **kwargs: Additional options (not used)
            
        Returns:
            Extracted text from image, or error message if OCR is not available
            
        Raises:
            ValueError: If OCR engine is not configured
        """
        file_path = current_file.get("file_path", "unknown")
        file_name = current_file.get("file_name", "unknown")
        file_extension = current_file.get("file_extension", "").lower()
        
        self.logger.info(f"Processing image file: {file_name}")
        
        # Validate file extension
        if file_extension not in SUPPORTED_IMAGE_EXTENSIONS:
            self.logger.warning(f"Unsupported image extension: {file_extension}")
            return f"[Unsupported image format: {file_extension}]"
        
        # If OCR engine is not available, return image tag format
        # This allows the image to be processed later when OCR is available
        if self._ocr_engine is None:
            self.logger.debug(f"OCR engine not available, returning image tag: {file_name}")
            # Use ImageProcessor's tag format (e.g., [Image:path] or custom format)
            return self._build_image_tag(file_path)
        
        # Use OCR engine to convert image to text
        try:
            # Use the file path directly for OCR conversion
            result = self._ocr_engine.convert_image_to_text(file_path)
            
            if result is None:
                self.logger.error(f"OCR returned None for image: {file_name}")
                return f"[Image OCR failed: {file_name}]"
            
            if result.startswith("[Image conversion error:"):
                self.logger.error(f"OCR error for image {file_name}: {result}")
                return result
            
            self.logger.info(f"Successfully extracted text from image: {file_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image {file_name}: {e}")
            return f"[Image processing error: {str(e)}]"
    
    def is_supported(self, file_extension: str) -> bool:
        """
        Check if file extension is supported.
        
        Args:
            file_extension: File extension (with or without dot)
            
        Returns:
            True if extension is supported, False otherwise
        """
        ext = file_extension.lower().lstrip('.')
        return ext in SUPPORTED_IMAGE_EXTENSIONS

    def _build_image_tag(self, file_path: str) -> str:
        """
        Build image tag using ImageProcessor's tag format.
        
        Uses the configured tag_prefix and tag_suffix from ImageProcessor
        to create a consistent image tag format.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Image tag string (e.g., "[Image:path]" or custom format)
        """
        # Normalize path separators (Windows -> Unix style)
        path_str = file_path.replace("\\", "/")
        
        # Use ImageProcessor's tag format
        prefix = self.image_processor.config.tag_prefix
        suffix = self.image_processor.config.tag_suffix
        
        return f"{prefix}{path_str}{suffix}"


__all__ = ["ImageFileHandler", "SUPPORTED_IMAGE_EXTENSIONS"]
