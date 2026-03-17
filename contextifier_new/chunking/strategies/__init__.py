# contextifier_new/chunking/strategies/__init__.py
"""
Chunking Strategies

Each strategy implements one approach to splitting text into chunks.
The TextChunker selects and applies the appropriate strategy based
on the content characteristics and configuration.
"""

from contextifier_new.chunking.strategies.base import BaseChunkingStrategy
from contextifier_new.chunking.strategies.page_strategy import PageChunkingStrategy
from contextifier_new.chunking.strategies.table_strategy import TableChunkingStrategy
from contextifier_new.chunking.strategies.protected_strategy import ProtectedChunkingStrategy
from contextifier_new.chunking.strategies.plain_strategy import PlainChunkingStrategy

__all__ = [
    "BaseChunkingStrategy",
    "PageChunkingStrategy",
    "TableChunkingStrategy",
    "ProtectedChunkingStrategy",
    "PlainChunkingStrategy",
]
