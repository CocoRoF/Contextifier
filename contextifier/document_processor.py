# contextifier/document_processor.py
"""
DocumentProcessor — Main Entry Point for Contextify v2

This is the single public-facing API for all document processing.
It replaces the v1 DocumentProcessor with a cleaner architecture:

Usage:
    from contextifier import DocumentProcessor

    # Simple extraction
    processor = DocumentProcessor()
    text = processor.extract_text("document.pdf")

    # With config
    from contextifier.config import ProcessingConfig
    config = ProcessingConfig(extract_metadata=True)
    processor = DocumentProcessor(config=config)
    text = processor.extract_text("document.pdf")

    # All-in-one extraction + chunking
    result = processor.extract_chunks("document.pdf", chunk_size=1000)
    for chunk in result.chunks:
        print(chunk)

    # Standalone chunking
    chunks = processor.chunk_text(text, chunk_size=1000)

Architecture:
    DocumentProcessor (facade)
      ├── HandlerRegistry  — maps extensions → handlers
      ├── TextChunker      — chunking subsystem
      ├── OCRProcessor     — optional OCR subsystem
      └── Services         — shared services (image, tag, chart, table, metadata)

Design improvements:
- No handler-specific logic in the processor (delegated to registry)
- No try/except ImportError blocks for handler registration
- OCR is optional and config-driven
- Chunking is encapsulated in TextChunker
- All services are created once and shared
- FileContext creation is standardised
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from contextifier.config import ProcessingConfig
from contextifier.types import (
    ExtractionResult,
    FileContext,
    Chunk,
    get_category,
)
from contextifier.errors import (
    FileNotFoundError as ContextifyFileNotFoundError,
    UnsupportedFormatError,
    HandlerNotFoundError,
    ContextifierError,
)
from contextifier.handlers.registry import HandlerRegistry
from contextifier.chunking.chunker import TextChunker

# Optional service imports
from contextifier.services.tag_service import TagService
from contextifier.services.image_service import ImageService
from contextifier.services.chart_service import ChartService
from contextifier.services.table_service import TableService
from contextifier.services.metadata_service import MetadataService
from contextifier.services.storage.local import LocalStorageBackend

logger = logging.getLogger("contextifier")


# ── ChunkResult ──────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    """
    Container for text chunks with optional metadata.

    Provides utility methods for accessing and saving chunked results.
    Supports both plain text chunks and Chunk objects with position metadata.
    """

    chunks: List[str] = field(default_factory=list)
    chunks_with_metadata: Optional[List[Chunk]] = None
    source_file: Optional[str] = None

    @property
    def has_metadata(self) -> bool:
        """Whether position metadata is available."""
        return self.chunks_with_metadata is not None and len(self.chunks_with_metadata) > 0

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> str:
        return self.chunks[index]

    def __iter__(self):
        return iter(self.chunks)

    def save_to_md(
        self,
        output_dir: Union[str, Path],
        *,
        filename_prefix: str = "chunk",
        separator: str = "---",
    ) -> List[str]:
        """
        Save each chunk as a separate markdown file.

        Args:
            output_dir: Directory to save files to.
            filename_prefix: Prefix for chunk filenames.
            separator: Separator between metadata and content.

        Returns:
            List of created file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        created_files: List[str] = []
        for i, chunk_text in enumerate(self.chunks, start=1):
            file_path = output_path / f"{filename_prefix}_{i:04d}.md"
            content_lines = []

            # Add metadata header if available
            if self.has_metadata and i <= len(self.chunks_with_metadata):  # type: ignore
                meta = self.chunks_with_metadata[i - 1].metadata  # type: ignore
                if meta:
                    content_lines.append(f"<!-- chunk_index: {meta.chunk_index} -->")
                    if meta.page_number is not None:
                        content_lines.append(f"<!-- page: {meta.page_number} -->")
                    content_lines.append(separator)
                    content_lines.append("")

            content_lines.append(chunk_text)

            file_path.write_text("\n".join(content_lines), encoding="utf-8")
            created_files.append(str(file_path))

        return created_files


# ── DocumentProcessor ────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Main entry point for document processing.

    Provides three core methods:
    1. extract_text() — extract text from any supported file
    2. chunk_text()   — split text into chunks
    3. extract_chunks() — extract + chunk in one step

    All format-specific logic is encapsulated in handlers via HandlerRegistry.
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        *,
        ocr_engine: Optional[Any] = None,
    ) -> None:
        """
        Initialize the document processor.

        Args:
            config: Processing configuration. Uses defaults if None.
            ocr_engine: Optional BaseOCREngine instance for OCR processing.
                        Configured via OCRConfig in the processing config.
        """
        self._config = config or ProcessingConfig()
        self._ocr_engine = ocr_engine
        self._logger = logger

        # Create shared services
        self._services = self._create_services()

        # Create handler registry and register all built-in handlers
        self._registry = HandlerRegistry(
            self._config,
            services=self._services,
        )
        self._registry.register_defaults()

        # Create chunker
        self._chunker = TextChunker(self._config)

        # Create OCR processor (optional)
        self._ocr_processor = None
        if self._ocr_engine is not None:
            from contextifier.ocr.processor import OCRProcessor
            self._ocr_processor = OCRProcessor(
                engine=self._ocr_engine,
                config=self._config,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Public API — Text Extraction
    # ═══════════════════════════════════════════════════════════════════════

    def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from a document file.

        Args:
            file_path: Path to the document file.
            file_extension: File extension override. If None, auto-detected.
            extract_metadata: Whether to include metadata in output.
            ocr_processing: Whether to perform OCR on image tags.
            **kwargs: Additional options passed to the handler pipeline.

        Returns:
            Extracted text as a string.

        Raises:
            ContextifyFileNotFoundError: If file does not exist.
            UnsupportedFormatError: If file extension is not supported.
            HandlerNotFoundError: If no handler found for extension.
        """
        file_path_str = str(file_path)

        # Validate file
        if not os.path.exists(file_path_str):
            raise ContextifyFileNotFoundError(
                f"File not found: {file_path_str}",
                context={"file_path": file_path_str},
            )

        # Determine extension
        ext = self._resolve_extension(file_path_str, file_extension)

        # Build FileContext
        file_context = self._create_file_context(file_path_str, ext)

        self._logger.info(f"Extracting text: {file_path_str} (ext={ext})")

        # Reset per-file image deduplication state
        self._services["image_service"].clear_state()

        # Get handler and extract
        handler = self._registry.get_handler(ext)
        text = handler.extract_text(
            file_context,
            include_metadata=extract_metadata,
            **kwargs,
        )

        # Optional OCR post-processing
        if ocr_processing and self._ocr_processor is not None:
            self._logger.info("Applying OCR processing")
            text = self._ocr_processor.process(text)
        elif ocr_processing and self._ocr_processor is None:
            self._logger.warning("OCR requested but no engine configured. Skipping.")

        return text

    def process(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs: Any,
    ) -> ExtractionResult:
        """
        Process a file and return the full ExtractionResult.

        Like extract_text() but returns structured result with metadata,
        tables, images, and charts instead of just text.

        Args:
            file_path: Path to the document file.
            file_extension: File extension override.
            extract_metadata: Whether to extract metadata.
            ocr_processing: Whether to apply OCR.
            **kwargs: Additional options.

        Returns:
            ExtractionResult with text, metadata, tables, charts, images.
        """
        file_path_str = str(file_path)

        if not os.path.exists(file_path_str):
            raise ContextifyFileNotFoundError(
                f"File not found: {file_path_str}",
                context={"file_path": file_path_str},
            )

        ext = self._resolve_extension(file_path_str, file_extension)
        file_context = self._create_file_context(file_path_str, ext)

        # Reset per-file image deduplication state
        self._services["image_service"].clear_state()

        handler = self._registry.get_handler(ext)
        result = handler.process(
            file_context,
            include_metadata=extract_metadata,
            **kwargs,
        )

        # Optional OCR
        if ocr_processing and self._ocr_processor is not None:
            result.text = self._ocr_processor.process(result.text)

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Public API — Chunking
    # ═══════════════════════════════════════════════════════════════════════

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        file_extension: str = "",
        preserve_tables: bool = True,
        include_position_metadata: bool = False,
    ) -> Union[List[str], List[Chunk]]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk.
            chunk_size: Maximum characters per chunk (overrides config).
            chunk_overlap: Character overlap between chunks (overrides config).
            file_extension: Source file extension (for strategy selection).
            preserve_tables: Whether to preserve table structures.
            include_position_metadata: Whether to include position metadata.

        Returns:
            List of plain text strings, or List of Chunk objects if
            include_position_metadata is True.
        """
        return self._chunker.chunk(
            text,
            file_extension=file_extension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_position_metadata=include_position_metadata,
            preserve_tables=preserve_tables,
        )

    def extract_chunks(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        preserve_tables: bool = True,
        include_position_metadata: bool = False,
        **kwargs: Any,
    ) -> ChunkResult:
        """
        Extract text from a file and chunk it in one step.

        Convenience method combining extract_text() and chunk_text().

        Args:
            file_path: Path to the document file.
            file_extension: File extension override.
            extract_metadata: Whether to include metadata in text.
            ocr_processing: Whether to apply OCR processing.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
            preserve_tables: Whether to preserve table structures.
            include_position_metadata: Whether to include position metadata.
            **kwargs: Additional handler options.

        Returns:
            ChunkResult with chunks and optional metadata.
        """
        # Extract text
        text = self.extract_text(
            file_path=file_path,
            file_extension=file_extension,
            extract_metadata=extract_metadata,
            ocr_processing=ocr_processing,
            **kwargs,
        )

        # Resolve extension for chunking strategy
        ext = self._resolve_extension(str(file_path), file_extension)

        # Chunk text
        raw_chunks = self.chunk_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_extension=ext,
            preserve_tables=preserve_tables,
            include_position_metadata=include_position_metadata,
        )

        # Build ChunkResult
        if include_position_metadata and raw_chunks and isinstance(raw_chunks[0], Chunk):
            return ChunkResult(
                chunks=[c.text for c in raw_chunks],  # type: ignore
                chunks_with_metadata=raw_chunks,  # type: ignore
                source_file=str(file_path),
            )
        else:
            return ChunkResult(
                chunks=raw_chunks,  # type: ignore
                source_file=str(file_path),
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Public API — Utility
    # ═══════════════════════════════════════════════════════════════════════

    def is_supported(self, extension: str) -> bool:
        """Check if a file extension is supported."""
        return self._registry.is_supported(extension)

    @property
    def supported_extensions(self) -> frozenset:
        """All supported file extensions."""
        return self._registry.supported_extensions

    @property
    def config(self) -> ProcessingConfig:
        """Current processing configuration."""
        return self._config

    @property
    def registry(self) -> HandlerRegistry:
        """Handler registry (for advanced users)."""
        return self._registry

    # ═══════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _create_services(self) -> Dict[str, Any]:
        """
        Create and return all shared services.

        Service dependency graph:
            TagService (standalone — no dependencies)
            ├── ImageService (depends on TagService + StorageBackend)
            ├── ChartService (depends on TagService)
            MetadataService (standalone)
            TableService (standalone)

        TagService is created first because ImageService and ChartService
        delegate tag creation to it for format consistency.
        """
        # TagService: standalone, no dependencies
        tag_service = TagService(self._config)

        # StorageBackend for images
        storage_backend = LocalStorageBackend(
            base_path=self._config.images.directory_path,
        )

        # ImageService: depends on TagService for tag creation
        image_service = ImageService(
            config=self._config,
            storage_backend=storage_backend,
            tag_service=tag_service,
        )

        # ChartService: depends on TagService for chart tag wrapping
        chart_service = ChartService(
            self._config,
            tag_service=tag_service,
        )

        # TableService, MetadataService: standalone
        table_service = TableService(self._config)
        metadata_service = MetadataService(self._config)

        return {
            "image_service": image_service,
            "tag_service": tag_service,
            "chart_service": chart_service,
            "table_service": table_service,
            "metadata_service": metadata_service,
        }

    @staticmethod
    def _resolve_extension(file_path: str, override: Optional[str]) -> str:
        """Resolve file extension (override > auto-detect)."""
        if override:
            return override.lower().lstrip(".")
        return os.path.splitext(file_path)[1].lower().lstrip(".")

    @staticmethod
    def _create_file_context(file_path: str, extension: str) -> FileContext:
        """Create a standardised FileContext dict from a file path."""
        file_data = Path(file_path).read_bytes()
        return FileContext(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            file_extension=extension,
            file_category=get_category(extension).value,
            file_data=file_data,
            file_stream=io.BytesIO(file_data),
            file_size=len(file_data),
        )

    def __repr__(self) -> str:
        n_ext = len(self._registry.supported_extensions)
        return f"DocumentProcessor(extensions={n_ext}, config={self._config!r})"


__all__ = ["DocumentProcessor", "ChunkResult"]
