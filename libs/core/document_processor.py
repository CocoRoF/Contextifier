# libs/core/document_processor.py
"""
DocumentProcessor - Document Processing Class

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
    text = await processor.extract_text(file_path, file_extension)

    # Extract text with OCR processing
    text = await processor.extract_text(file_path, file_extension, ocr_processing=True)

    # Chunk text
    chunks = processor.chunk_text(text, chunk_size=1000)
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("contextify")


class DocumentProcessor:
    """
    Contextify Main Document Processing Class

    A unified interface for processing various document formats and extracting text.

    Attributes:
        config: Configuration dictionary or ConfigComposer instance
        supported_extensions: List of supported file extensions

    Example:
        >>> processor = DocumentProcessor()
        >>> text = await processor.extract_text("document.pdf", "pdf")
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
            **kwargs: Additional configuration options
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

    async def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        process_type: str = "default",
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs
    ) -> str:
        """
        Extract text from a file.

        Args:
            file_path: File path
            file_extension: File extension (if None, auto-extracted from file_path)
            process_type: Processing type ('default', 'enhanced', 'enhanced_v4', 'enhanced_ocr', etc.)
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

        # Get handler and extract text
        handler = self._get_handler(ext)
        text = await self._invoke_handler(handler, file_path_str, ext, extract_metadata, **kwargs)

        # Apply OCR processing if enabled and ocr_engine is available
        if ocr_processing and self._ocr_engine is not None:
            self._logger.info(f"Applying OCR processing with {self._ocr_engine}")
            text = await self._ocr_engine.process_text(text)
        elif ocr_processing and self._ocr_engine is None:
            self._logger.warning("OCR processing requested but no ocr_engine is configured. Skipping OCR.")

        return text

    async def extract_text_batch(
        self,
        file_paths: List[Union[str, Path]],
        *,
        process_type: str = "default",
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract text from multiple files in batch.

        Args:
            file_paths: List of file paths
            process_type: Processing type
            extract_metadata: Whether to extract metadata
            ocr_processing: Whether to perform OCR on image tags
            max_concurrent: Maximum concurrent processing count
            **kwargs: Additional handler-specific options

        Returns:
            List of extraction result dictionaries
            [{"file_path": str, "text": str, "success": bool, "error": Optional[str]}, ...]
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def process_single(fp: Union[str, Path]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    text = await self.extract_text(
                        fp,
                        process_type=process_type,
                        extract_metadata=extract_metadata,
                        ocr_processing=ocr_processing,
                        **kwargs
                    )
                    return {
                        "file_path": str(fp),
                        "text": text,
                        "success": True,
                        "error": None
                    }
                except Exception as e:
                    self._logger.error(f"Error processing {fp}: {e}")
                    return {
                        "file_path": str(fp),
                        "text": "",
                        "success": False,
                        "error": str(e)
                    }

        tasks = [process_single(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)

        return list(results)

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
        """Build and cache handler registry."""
        if self._handler_registry is not None:
            return self._handler_registry

        self._handler_registry = {}

        # PDF handlers
        try:
            from libs.core.processor.pdf_handler import extract_text_from_pdf
            self._handler_registry['pdf'] = extract_text_from_pdf
        except ImportError as e:
            self._logger.warning(f"PDF handler not available: {e}")

        # DOCX handler
        try:
            from libs.core.processor.docx_handler import extract_text_from_docx
            self._handler_registry['docx'] = extract_text_from_docx
        except ImportError as e:
            self._logger.warning(f"DOCX handler not available: {e}")

        # DOC handler
        try:
            from libs.core.processor.doc_handler import extract_text_from_doc
            self._handler_registry['doc'] = extract_text_from_doc
        except ImportError as e:
            self._logger.warning(f"DOC handler not available: {e}")

        # PPT/PPTX handler
        try:
            from libs.core.processor.ppt_handler import extract_text_from_ppt
            self._handler_registry['ppt'] = extract_text_from_ppt
            self._handler_registry['pptx'] = extract_text_from_ppt
        except ImportError as e:
            self._logger.warning(f"PPT handler not available: {e}")

        # Excel handlers
        try:
            from libs.core.processor.excel_handler import extract_text_from_excel
            self._handler_registry['xlsx'] = extract_text_from_excel
            self._handler_registry['xls'] = extract_text_from_excel
        except ImportError as e:
            self._logger.warning(f"Excel handler not available: {e}")

        # CSV/TSV handler
        try:
            from libs.core.processor.csv_handler import extract_text_from_csv
            self._handler_registry['csv'] = extract_text_from_csv
            self._handler_registry['tsv'] = extract_text_from_csv
        except ImportError as e:
            self._logger.warning(f"CSV handler not available: {e}")

        # HWP handler
        try:
            from libs.core.processor.hwp_processor import extract_text_from_hwp
            self._handler_registry['hwp'] = extract_text_from_hwp
        except ImportError as e:
            self._logger.warning(f"HWP handler not available: {e}")

        # HWPX handler
        try:
            from libs.core.processor.hwpx_processor import extract_text_from_hwpx
            self._handler_registry['hwpx'] = extract_text_from_hwpx
        except ImportError as e:
            self._logger.warning(f"HWPX handler not available: {e}")

        # Text handler (for text, code, config, script, log, web types)
        try:
            from libs.core.processor.text_handler import extract_text_from_text_file
            text_extensions = (
                self.TEXT_TYPES |
                self.CODE_TYPES |
                self.CONFIG_TYPES |
                self.SCRIPT_TYPES |
                self.LOG_TYPES |
                self.WEB_TYPES
            )
            for ext in text_extensions:
                self._handler_registry[ext] = extract_text_from_text_file
        except ImportError as e:
            self._logger.warning(f"Text handler not available: {e}")

        return self._handler_registry

    def _get_handler(self, ext: str) -> Optional[Callable]:
        """Get handler for file extension."""
        registry = self._get_handler_registry()
        return registry.get(ext)

    async def _invoke_handler(
        self,
        handler: Optional[Callable],
        file_path: str,
        ext: str,
        extract_metadata: bool,
        **kwargs
    ) -> str:
        """
        Invoke the appropriate handler based on extension.

        Args:
            handler: Handler function
            file_path: File path
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

        # Text-based files use different signature
        text_extensions = (
            self.TEXT_TYPES |
            self.CODE_TYPES |
            self.CONFIG_TYPES |
            self.SCRIPT_TYPES |
            self.LOG_TYPES |
            self.WEB_TYPES
        )

        if ext in text_extensions:
            # text_handler signature: (file_path, file_type, encodings, is_code)
            return await handler(file_path, ext, is_code=is_code)

        # HWP/HWPX signature: (file_path, config, extract_default_metadata)
        if ext in ('hwp', 'hwpx'):
            return await handler(file_path, self._config, extract_default_metadata=extract_metadata)

        # Standard handler signature: (file_path, current_config, extract_default_metadata)
        return await handler(file_path, self._config, extract_default_metadata=extract_metadata)

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> "DocumentProcessor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Perform resource cleanup here if needed
        pass

    def __enter__(self) -> "DocumentProcessor":
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit."""
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
    **kwargs
) -> DocumentProcessor:
    """
    Create a DocumentProcessor instance.

    Args:
        config: Configuration dictionary or ConfigComposer instance
        ocr_engine: OCR engine instance (BaseOCR subclass)
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
    """
    return DocumentProcessor(config=config, ocr_engine=ocr_engine, **kwargs)


__all__ = [
    "DocumentProcessor",
    "create_processor",
]
