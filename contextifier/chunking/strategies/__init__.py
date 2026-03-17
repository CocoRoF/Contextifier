# contextifier_new/chunking/strategies/__init__.py
"""
Chunking Strategies

Each strategy implements one approach to splitting text into chunks.
The TextChunker selects and applies the appropriate strategy based
on the content characteristics and configuration.
"""

from contextifier.chunking.strategies.base import BaseChunkingStrategy
from contextifier.chunking.strategies.page_strategy import PageChunkingStrategy
from contextifier.chunking.strategies.table_strategy import TableChunkingStrategy
from contextifier.chunking.strategies.protected_strategy import ProtectedChunkingStrategy
from contextifier.chunking.strategies.plain_strategy import PlainChunkingStrategy

__all__ = [
    "BaseChunkingStrategy",
    "PageChunkingStrategy",
    "TableChunkingStrategy",
    "ProtectedChunkingStrategy",
    "PlainChunkingStrategy",
]
