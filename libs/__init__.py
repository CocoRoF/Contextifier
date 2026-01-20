# libs/__init__.py
"""
Contextifier Library

A document processing and chunking library for AI applications.

Package Structure:
- core: Document processing core module
    - DocumentProcessor: Main document processing class
    - processor: Individual document type handlers (PDF, DOCX, PPT, Excel, HWP, etc.)
    - functions: Utility functions

- chunking: Text chunking module
    - Text splitting and chunking logic
    - Table-preserving chunking
    - Page-based chunking

Usage:
    from contextifier import DocumentProcessor
    
    processor = DocumentProcessor()
    text = processor.extract_text("document.pdf")
    result = processor.extract_chunks("document.pdf", chunk_size=1000)
"""

__version__ = "0.1.0"

# Expose core classes at top level
from libs.core import DocumentProcessor

# Explicit subpackages
from libs import core
from libs import chunking

__all__ = [
    "__version__",
    # Core classes
    "DocumentProcessor",
    # Subpackages
    "core",
    "chunking",
]
