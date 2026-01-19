# libs/core/document_processor.py
"""DocumentProcessor - Document Processing Class

Main document processing class for the Contextify library.
Provides a unified interface for extracting text from various document formats
(PDF, DOCX, PPT, Excel, HWP, etc.) and performing text chunking.

This class is the recommended entry point when using the library.

Usage Example:
    from libs.core.document_processor import DocumentProcessor
    from libs.ocr.ocr_engine import OpenAIOCR

    # Create instance (with optional OCR engine)
    ocr_engine = OpenAIOCR(api_key="sk-...", model="gpt-4o")
    processor = DocumentProcessor(ocr_engine=ocr_engine)

    # Extract text from file
    text = processor.extract_text(file_path, file_extension)

    # Extract text with OCR processing
    text = processor.extract_text(file_path, file_extension, ocr_processing=True)

    # Chunk text
    chunks = processor.chunk_text(text, chunk_size=1000)
"""

import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TypedDict

logger = logging.getLogger("contextify")


class CurrentFile(TypedDict, total=False):
    """
    TypedDict containing file information.
    
    Standard structure for reading files at binary level and passing to handlers.
    Resolves file system issues such as non-ASCII (Korean, etc.) paths.
    
    Attributes:
        file_path: Absolute path of the original file
        file_name: File name (including extension)
        file_extension: File extension (lowercase, without dot)
        file_data: Binary data of the file
        file_stream: BytesIO stream (reusable)
        file_size: File size in bytes
    """
    file_path: str
    file_name: str
    file_extension: str
    file_data: bytes
    file_stream: io.BytesIO
    file_size: int

class DocumentProcessor:
    """
    Contextify Main Document Processing Class

    A unified interface for processing various document formats and extracting text.

    Attributes:
        config: Configuration dictionary or ConfigComposer instance
        supported_extensions: List of supported file extensions

    Example:
        >>> processor = DocumentProcessor()
        >>> text = processor.extract_text("document.pdf", "pdf")
        >>> chunks = processor.chunk_text(text, chunk_size=1000)
    """

    # === Supported File Type Classifications ===
    DOCUMENT_TYPES = frozenset(['pdf', 'docx', 'doc', 'pptx', 'ppt', 'hwp', 'hwpx'])
    TEXT_TYPES = frozenset(['txt', 'md', 'markdown', 'rtf'])
    CODE_TYPES = frozenset([
        'py', 'js', 'ts', 'java', 'cpp', 'c', 'h', 'cs', 'go', 'rs',
        'php', 'rb', 'swift', 'kt', 'scala', 'dart', 'r', 'sql',
        'html', 'css', 'jsx', 'tsx', 'vue', 'svelte'
    ])
    CONFIG_TYPES = frozenset(['json', 'yaml', 'yml', 'xml', 'toml', 'ini', 'cfg', 'conf', 'properties', 'env'])
    DATA_TYPES = frozenset(['csv', 'tsv', 'xlsx', 'xls'])
    SCRIPT_TYPES = frozenset(['sh', 'bat', 'ps1', 'zsh', 'fish'])
    LOG_TYPES = frozenset(['log'])
    WEB_TYPES = frozenset(['htm', 'xhtml'])
    IMAGE_TYPES = frozenset(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'])

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        ocr_engine: Optional[Any] = None,
        *,
        image_directory: Optional[str] = None,
        image_tag_prefix: Optional[str] = None,
        image_tag_suffix: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DocumentProcessor.

        Args:
            config: Configuration dictionary or ConfigComposer instance
                   - Dict: Pass configuration dictionary directly
                   - ConfigComposer: Existing config_composer instance
                   - None: Use default settings
            ocr_engine: OCR engine instance (BaseOCR subclass)
                   - If provided, OCR processing can be enabled in extract_text
                   - Example: OpenAIOCR, AnthropicOCR, GeminiOCR, VllmOCR
            image_directory: Directory path for saving extracted images
                   - Default: "temp/images"
            image_tag_prefix: Prefix for image tags in extracted text
                   - Default: "[Image:"
                   - Example: "<img src='" for HTML format
            image_tag_suffix: Suffix for image tags in extracted text
                   - Default: "]"
                   - Example: "'/>" for HTML format
            **kwargs: Additional configuration options

        Example:
            >>> # Default image tags: [Image:path/to/image.png]
            >>> processor = DocumentProcessor()

            >>> # Custom HTML image tags: <img src='path/to/image.png'/>
            >>> processor = DocumentProcessor(
            ...     image_directory="output/images",
            ...     image_tag_prefix="<img src='",
            ...     image_tag_suffix="'/>"
            ... )

            >>> # Markdown image tags: ![image](path/to/image.png)
            >>> processor = DocumentProcessor(
            ...     image_tag_prefix="![image](",
            ...     image_tag_suffix=")"
            ... )
        """
        self._config = config or {}
        self._ocr_engine = ocr_engine
        self._kwargs = kwargs
        self._supported_extensions: Optional[List[str]] = None

        # Logger setup
        self._logger = logging.getLogger("contextify.processor")

        # Cache for library availability check results
        self._library_availability: Optional[Dict[str, bool]] = None

        # Handler registry
        self._handler_registry: Optional[Dict[str, Callable]] = None

        # Create instance-specific ImageProcessor
        self._image_processor = self._create_image_processor(
            directory=image_directory,
            tag_prefix=image_tag_prefix,
            tag_suffix=image_tag_suffix
        )

        # Add image_processor to config for handlers to access
        if isinstance(self._config, dict):
            self._config["image_processor"] = self._image_processor

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def supported_extensions(self) -> List[str]:
        """List of all supported file extensions."""
        if self._supported_extensions is None:
            self._supported_extensions = self._build_supported_extensions()
        return self._supported_extensions.copy()

    @property
    def config(self) -> Optional[Union[Dict[str, Any], Any]]:
        """Current configuration."""
        return self._config

    @property
    def image_config(self) -> Dict[str, Any]:
        """
        Current image processor configuration.

        Returns:
            Dictionary containing:
            - directory_path: Image save directory
            - tag_prefix: Image tag prefix
            - tag_suffix: Image tag suffix
            - naming_strategy: File naming strategy
        """
        return {
            "directory_path": self._image_processor.config.directory_path,
            "tag_prefix": self._image_processor.config.tag_prefix,
            "tag_suffix": self._image_processor.config.tag_suffix,
            "naming_strategy": self._image_processor.config.naming_strategy.value,
        }

    @property
    def image_processor(self) -> Any:
        """Current ImageProcessor instance for this DocumentProcessor."""
        return self._image_processor

    @property
    def ocr_engine(self) -> Optional[Any]:
        """Current OCR engine instance."""
        return self._ocr_engine

    @ocr_engine.setter
    def ocr_engine(self, engine: Optional[Any]) -> None:
        """Set OCR engine instance."""
        self._ocr_engine = engine

    # =========================================================================
    # Public Methods - Text Extraction
    # =========================================================================

    def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs
    ) -> str:
        """
        Extract text from a file.

        Args:
            file_path: File path
            file_extension: File extension (if None, auto-extracted from file_path)
            extract_metadata: Whether to extract metadata
            ocr_processing: Whether to perform OCR on image tags in extracted text
                           - If True and ocr_engine is set, processes [Image:...] tags
                           - If True but ocr_engine is None, skips OCR processing
            **kwargs: Additional handler-specific options

        Returns:
            Extracted text string

        Raises:
            FileNotFoundError: If file cannot be found
            ValueError: If file format is not supported
        """
        # Convert to string path
        file_path_str = str(file_path)

        # Check file existence
        if not os.path.exists(file_path_str):
            raise FileNotFoundError(f"File not found: {file_path_str}")

        # Extract extension if not provided
        if file_extension is None:
            file_extension = os.path.splitext(file_path_str)[1].lstrip('.')

        ext = file_extension.lower().lstrip('.')

        # Check if extension is supported
        if not self.is_supported(ext):
            raise ValueError(f"Unsupported file format: {ext}")

        self._logger.info(f"Extracting text from: {file_path_str} (ext={ext})")

        # Create current_file dict with binary data
        current_file = self._create_current_file(file_path_str, ext)

        # Get handler and extract text
        handler = self._get_handler(ext)
        text = self._invoke_handler(handler, current_file, ext, extract_metadata, **kwargs)

        # Apply OCR processing if enabled and ocr_engine is available
        if ocr_processing and self._ocr_engine is not None:
            self._logger.info(f"Applying OCR processing with {self._ocr_engine}")
            text = self._ocr_engine.process_text(text)
        elif ocr_processing and self._ocr_engine is None:
            self._logger.warning("OCR processing requested but no ocr_engine is configured. Skipping OCR.")

        return text

    # =========================================================================
    # Public Methods - Text Chunking
    # =========================================================================

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        file_extension: Optional[str] = None,
        preserve_tables: bool = True,
    ) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split
            chunk_size: Chunk size (character count)
            chunk_overlap: Overlap size between chunks
            file_extension: File extension (used for table-based file processing)
            preserve_tables: Whether to preserve table structure

        Returns:
            List of chunk strings
        """
        from libs.chunking.chunking import split_text_preserving_html_blocks

        if not text or not text.strip():
            return [""]

        # Use force_chunking to disable table protection if preserve_tables is False
        force_chunking = not preserve_tables

        chunks = split_text_preserving_html_blocks(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_extension=file_extension,
            force_chunking=force_chunking
        )

        return chunks

    def extract_chunks(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Extract text from a file and split into chunks in one step.

        This is a convenience method that combines extract_text() and chunk_text().

        Args:
            file_path: File path
            file_extension: File extension (if None, auto-extracted from file_path)
            extract_metadata: Whether to extract metadata
            ocr_processing: Whether to perform OCR on image tags in extracted text
            chunk_size: Chunk size (character count)
            chunk_overlap: Overlap size between chunks
            preserve_tables: Whether to preserve table structure
            **kwargs: Additional handler-specific options

        Returns:
            List of chunk strings

        Raises:
            FileNotFoundError: If file cannot be found
            ValueError: If file format is not supported

        Example:
            >>> processor = DocumentProcessor()
            >>> chunks = processor.extract_chunks("document.pdf", chunk_size=1000)
            >>> for i, chunk in enumerate(chunks):
            ...     print(f"Chunk {i+1}: {len(chunk)} chars")
        """
        # Extract text
        text = self.extract_text(
            file_path=file_path,
            file_extension=file_extension,
            extract_metadata=extract_metadata,
            ocr_processing=ocr_processing,
            **kwargs
        )

        # Determine file extension for chunking
        if file_extension is None:
            file_extension = os.path.splitext(str(file_path))[1].lstrip('.')

        # Chunk text
        chunks = self.chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_extension=file_extension,
            preserve_tables=preserve_tables
        )

        return chunks

    # =========================================================================
    # Public Methods - Utilities
    # =========================================================================

    def get_file_category(self, file_extension: str) -> str:
        """
        Return the category of a file extension.

        Args:
            file_extension: File extension

        Returns:
            Category string ('document', 'text', 'code', 'data', etc.)
        """
        ext = file_extension.lower().lstrip('.')

        if ext in self.DOCUMENT_TYPES:
            return 'document'
        if ext in self.TEXT_TYPES:
            return 'text'
        if ext in self.CODE_TYPES:
            return 'code'
        if ext in self.CONFIG_TYPES:
            return 'config'
        if ext in self.DATA_TYPES:
            return 'data'
        if ext in self.SCRIPT_TYPES:
            return 'script'
        if ext in self.LOG_TYPES:
            return 'log'
        if ext in self.WEB_TYPES:
            return 'web'
        if ext in self.IMAGE_TYPES:
            return 'image'

        return 'unknown'

    def is_supported(self, file_extension: str) -> bool:
        """
        Check if a file extension is supported.

        Args:
            file_extension: File extension

        Returns:
            Whether supported
        """
        ext = file_extension.lower().lstrip('.')
        return ext in self.supported_extensions

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        from libs.core.functions.utils import clean_text as _clean_text
        return _clean_text(text)

    @staticmethod
    def clean_code_text(text: str) -> str:
        """
        Clean code text.

        Args:
            text: Code text to clean

        Returns:
            Cleaned code text
        """
        from libs.core.functions.utils import clean_code_text as _clean_code_text
        return _clean_code_text(text)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _create_image_processor(
        self,
        directory: Optional[str] = None,
        tag_prefix: Optional[str] = None,
        tag_suffix: Optional[str] = None
    ) -> Any:
        """
        Create an ImageProcessor instance for this DocumentProcessor.

        This creates an instance-specific ImageProcessor that will be
        passed to handlers via config.

        Args:
            directory: Image save directory
            tag_prefix: Image tag prefix
            tag_suffix: Image tag suffix

        Returns:
            ImageProcessor instance
        """
        from libs.core.functions.img_processor import create_image_processor

        return create_image_processor(
            directory_path=directory,
            tag_prefix=tag_prefix,
            tag_suffix=tag_suffix
        )

    def _build_supported_extensions(self) -> List[str]:
        """Build list of supported extensions."""
        extensions = list(
            self.DOCUMENT_TYPES |
            self.TEXT_TYPES |
            self.CODE_TYPES |
            self.CONFIG_TYPES |
            self.DATA_TYPES |
            self.SCRIPT_TYPES |
            self.LOG_TYPES |
            self.WEB_TYPES |
            self.IMAGE_TYPES
        )

        return sorted(extensions)

    def _get_handler_registry(self) -> Dict[str, Callable]:
        """Build and cache handler registry.
        
        All handlers are class-based, inheriting from BaseHandler.
        """
        if self._handler_registry is not None:
            return self._handler_registry

        self._handler_registry = {}

        # PDF handler
        try:
            from libs.core.processor.pdf_handler import PDFHandler
            pdf_handler = PDFHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['pdf'] = pdf_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"PDF handler not available: {e}")

        # DOCX handler
        try:
            from libs.core.processor.docx_handler import DOCXHandler
            docx_handler = DOCXHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['docx'] = docx_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"DOCX handler not available: {e}")

        # DOC handler
        try:
            from libs.core.processor.doc_handler import DOCHandler
            doc_handler = DOCHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['doc'] = doc_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"DOC handler not available: {e}")

        # PPT/PPTX handler
        try:
            from libs.core.processor.ppt_handler import PPTHandler
            ppt_handler = PPTHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['ppt'] = ppt_handler.extract_text
            self._handler_registry['pptx'] = ppt_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"PPT handler not available: {e}")

        # Excel handler
        try:
            from libs.core.processor.excel_handler import ExcelHandler
            excel_handler = ExcelHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['xlsx'] = excel_handler.extract_text
            self._handler_registry['xls'] = excel_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"Excel handler not available: {e}")

        # CSV/TSV handler
        try:
            from libs.core.processor.csv_handler import CSVHandler
            csv_handler = CSVHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['csv'] = csv_handler.extract_text
            self._handler_registry['tsv'] = csv_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"CSV handler not available: {e}")

        # HWP handler
        try:
            from libs.core.processor.hwp_handler import HWPHandler
            hwp_handler = HWPHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['hwp'] = hwp_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"HWP handler not available: {e}")

        # HWPX handler
        try:
            from libs.core.processor.hwps_handler import HWPXHandler
            hwpx_handler = HWPXHandler(config=self._config, image_processor=self._image_processor)
            self._handler_registry['hwpx'] = hwpx_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"HWPX handler not available: {e}")

        # Text handler (for text, code, config, script, log, web types)
        try:
            from libs.core.processor.text_handler import TextHandler
            text_handler = TextHandler(config=self._config, image_processor=self._image_processor)
            text_extensions = (
                self.TEXT_TYPES |
                self.CODE_TYPES |
                self.CONFIG_TYPES |
                self.SCRIPT_TYPES |
                self.LOG_TYPES |
                self.WEB_TYPES
            )
            for ext in text_extensions:
                self._handler_registry[ext] = text_handler.extract_text
        except ImportError as e:
            self._logger.warning(f"Text handler not available: {e}")

        return self._handler_registry

    def _create_current_file(self, file_path: str, ext: str) -> CurrentFile:
        """
        Create a CurrentFile dict from a file path.
        
        Reads the file at binary level to avoid path encoding issues
        (e.g., Korean characters in Windows paths).
        
        Args:
            file_path: Absolute path to the file
            ext: File extension (lowercase, without dot)
            
        Returns:
            CurrentFile dict containing file info and binary data
            
        Raises:
            IOError: If file cannot be read
        """
        file_path = os.path.abspath(file_path)
        file_name = os.path.basename(file_path)
        
        # Read file as binary
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Create BytesIO stream for handlers that need seekable stream
        file_stream = io.BytesIO(file_data)
        
        # Return as plain dict (TypedDict is for type hints only)
        return {
            "file_path": file_path,
            "file_name": file_name,
            "file_extension": ext,
            "file_data": file_data,
            "file_stream": file_stream,
            "file_size": len(file_data)
        }

    def _get_handler(self, ext: str) -> Optional[Callable]:
        """Get handler for file extension."""
        registry = self._get_handler_registry()
        return registry.get(ext)

    def _invoke_handler(
        self,
        handler: Optional[Callable],
        current_file: CurrentFile,
        ext: str,
        extract_metadata: bool,
        **kwargs
    ) -> str:
        """
        Invoke the appropriate handler based on extension.

        All handlers are class-based and use the same signature:
        handler(current_file, extract_metadata=..., **kwargs)

        Args:
            handler: Handler method (bound method from Handler class)
            current_file: CurrentFile dict containing file info and binary data
            ext: File extension
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options

        Returns:
            Extracted text
        """
        if handler is None:
            raise ValueError(f"No handler available for extension: {ext}")

        # Determine if this is a code file
        is_code = ext in self.CODE_TYPES

        # Text-based files include file_type and is_code in kwargs
        text_extensions = (
            self.TEXT_TYPES |
            self.CODE_TYPES |
            self.CONFIG_TYPES |
            self.SCRIPT_TYPES |
            self.LOG_TYPES |
            self.WEB_TYPES
        )

        if ext in text_extensions:
            return handler(current_file, extract_metadata=extract_metadata, file_type=ext, is_code=is_code, **kwargs)

        # All other handlers use standard signature
        return handler(current_file, extract_metadata=extract_metadata, **kwargs)

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> "DocumentProcessor":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # Perform resource cleanup here if needed
        pass

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        return f"DocumentProcessor(supported_extensions={len(self.supported_extensions)})"

    def __str__(self) -> str:
        return f"Contextify DocumentProcessor ({len(self.supported_extensions)} supported formats)"


# === Module-level Convenience Functions ===

def create_processor(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    ocr_engine: Optional[Any] = None,
    *,
    image_directory: Optional[str] = None,
    image_tag_prefix: Optional[str] = None,
    image_tag_suffix: Optional[str] = None,
    **kwargs
) -> DocumentProcessor:
    """
    Create a DocumentProcessor instance.

    Args:
        config: Configuration dictionary or ConfigComposer instance
        ocr_engine: OCR engine instance (BaseOCR subclass)
        image_directory: Directory path for saving extracted images
        image_tag_prefix: Prefix for image tags (default: "[Image:")
        image_tag_suffix: Suffix for image tags (default: "]")
        **kwargs: Additional configuration options

    Returns:
        DocumentProcessor instance

    Example:
        >>> processor = create_processor()
        >>> processor = create_processor(config={"vision_model": "gpt-4-vision"})

        # With OCR engine
        >>> from libs.ocr.ocr_engine import OpenAIOCR
        >>> ocr = OpenAIOCR(api_key="sk-...", model="gpt-4o")
        >>> processor = create_processor(ocr_engine=ocr)

        # With custom image tags (HTML format)
        >>> processor = create_processor(
        ...     image_directory="output/images",
        ...     image_tag_prefix="<img src='",
        ...     image_tag_suffix="'/>"
        ... )
    """
    return DocumentProcessor(
        config=config,
        ocr_engine=ocr_engine,
        image_directory=image_directory,
        image_tag_prefix=image_tag_prefix,
        image_tag_suffix=image_tag_suffix,
        **kwargs
    )


__all__ = [
    "DocumentProcessor",
    "CurrentFile",
    "create_processor",
]
