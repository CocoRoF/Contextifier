# contextifier_new/chunking/__init__.py
"""
Chunking — Text Chunking Subsystem

Replaces the old monolithic create_chunks() function with a class-based
TextChunker that uses pluggable strategies.

Architecture:
    TextChunker (facade)
      └── ChunkingStrategy (abstract)
            ├── TableChunkingStrategy     — table-aware splitting  (priority 5)
            ├── PageChunkingStrategy      — split by page markers  (priority 10)
            ├── ProtectedChunkingStrategy — protected region splitting (priority 20)
            └── PlainChunkingStrategy     — recursive text splitting  (priority 100)

    Shared utilities:
        table_parser   — HTML & Markdown table structure analysis
        table_chunker  — Row-level splitting for large tables
        constants      — Patterns, thresholds, dataclasses

Design improvements over v1:
- Strategy pattern replaces mega-function branching
- TextChunker is injectable (services passed to constructor)
- Each strategy is independently testable
- Constants/patterns centralized and config-driven
"""

from contextifier_new.chunking.chunker import TextChunker
from contextifier_new.chunking.table_parser import (
    parse_html_table,
    parse_markdown_table,
    is_markdown_table,
)
from contextifier_new.chunking.table_chunker import (
    chunk_html_table,
    chunk_markdown_table,
    chunk_large_table,
)

__all__ = [
    "TextChunker",
    # Table utilities (useful for direct use and testing)
    "parse_html_table",
    "parse_markdown_table",
    "is_markdown_table",
    "chunk_html_table",
    "chunk_markdown_table",
    "chunk_large_table",
]
