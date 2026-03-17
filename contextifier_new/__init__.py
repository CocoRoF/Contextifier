# contextifier_new/__init__.py
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
    from contextifier_new import DocumentProcessor

    processor = DocumentProcessor()
    text = processor.extract_text("document.pdf")
    chunks = processor.extract_chunks("document.pdf", chunk_size=1000)
"""

__version__ = "2.0.0-alpha"

from contextifier_new.document_processor import DocumentProcessor

__all__ = [
    "__version__",
    "DocumentProcessor",
]
