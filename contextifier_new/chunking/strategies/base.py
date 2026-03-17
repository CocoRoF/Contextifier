# contextifier_new/chunking/strategies/base.py
"""
BaseChunkingStrategy — Abstract Chunking Strategy

All chunking strategies implement this interface.
Each strategy knows how to split text into chunks using its
specific approach, while respecting protected regions and
structural elements.

Contract:
- can_handle(text, config) → bool : Does this strategy apply?
- chunk(text, config) → List[str]  : Perform the chunking
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from contextifier_new.config import ProcessingConfig, ChunkingConfig
from contextifier_new.types import Chunk, ChunkMetadata


class BaseChunkingStrategy(ABC):
    """
    Abstract base for chunking strategies.

    Subclasses implement:
    - can_handle(): Whether this strategy is appropriate for the content
    - chunk(): Perform the actual chunking
    - strategy_name: Human-readable name
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"contextifier.chunking.{self.__class__.__name__}"
        )

    @abstractmethod
    def can_handle(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        **context: Any,
    ) -> bool:
        """
        Determine if this strategy should handle the given text.

        Args:
            text: The text to potentially chunk.
            config: Processing configuration.
            file_extension: Source file extension.
            **context: Additional context (e.g., has_tables, has_pages).

        Returns:
            True if this strategy can and should handle the text.
        """
        ...

    @abstractmethod
    def chunk(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        include_position_metadata: bool = False,
        **context: Any,
    ) -> Union[List[str], List[Chunk]]:
        """
        Split text into chunks.

        Args:
            text: The text to split.
            config: Processing configuration (includes ChunkingConfig).
            file_extension: Source file extension.
            include_position_metadata: If True, return List[Chunk] with metadata.
                                       If False, return List[str].
            **context: Additional context.

        Returns:
            List of chunks (as strings or Chunk objects).
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    def priority(self) -> int:
        """
        Strategy priority (lower = higher priority).

        When multiple strategies can_handle, the one with lowest
        priority number is selected.

        Default: 100
        """
        return 100


__all__ = ["BaseChunkingStrategy"]
