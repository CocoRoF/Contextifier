# contextifier/__init__.py
"""
Contextifier v2 — Unified Document Processing Library

A complete rewrite with strict interface contracts, unified pipeline,
and consistent processing across all file formats.

Architecture:
    DocumentProcessor (entry point)
      └── Handler (per format: PDF, DOCX, PPT, Excel, ...)
            └── Pipeline stages (convert → preprocess → extract → postprocess)
                  ├── Converter: binary → format object
                  ├── Preprocessor: clean/transform
                  ├── ContentExtractor: text, images, tables, charts
                  ├── MetadataExtractor: document metadata
                  └── Postprocessor: final assembly & cleanup
      └── Services (shared, injected)
            ├── ImageService: image save/tag generation
            ├── TagService: page/slide/sheet/chart tags
            ├── TableService: table formatting (HTML/MD/Text)
            ├── MetadataService: metadata formatting
            └── StorageBackend: local/cloud file storage
      └── TextChunker (chunking subsystem)
      └── OCR (optional vision-based extraction)

Usage:
    from contextifier import DocumentProcessor, open_raw

    # AI-friendly view (lightweight, normalized)
    processor = DocumentProcessor()
    text = processor.extract_text("document.pdf")
    chunks = processor.extract_chunks("document.pdf", chunk_size=1000)

    # Raw view (lossless, addressable, WRITABLE — xlsx/docx/pptx)
    raw = open_raw("report.xlsx")
    raw.sheets["Sales"].set_cell("B3", 142)
    raw.save("report-edited.xlsx")     # untouched parts stay byte-identical
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("contextifier")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from contextifier.document_processor import DocumentProcessor, ChunkResult
from contextifier.config import ProcessingConfig, ChunkingConfig
from contextifier.types import ExtractionResult, FileContext, Chunk, ChunkMetadata
from contextifier.chunking.chunker import TextChunker
from contextifier.errors import ContextifierError, UnsupportedFormatError
from contextifier.raw import open_raw
from contextifier.async_processor import AsyncDocumentProcessor
from contextifier.cached_processor import CachedDocumentProcessor

__all__ = [
    "__version__",
    # Core
    "DocumentProcessor",
    "AsyncDocumentProcessor",
    "CachedDocumentProcessor",
    "ChunkResult",
    # Config
    "ProcessingConfig",
    "ChunkingConfig",
    # Types
    "ExtractionResult",
    "FileContext",
    "Chunk",
    "ChunkMetadata",
    # Chunking
    "TextChunker",
    # Raw (lossless, writable) access
    "open_raw",
    # Errors
    "ContextifierError",
    "UnsupportedFormatError",
]
