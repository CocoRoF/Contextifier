# contextifier_new/chunking/chunker.py
"""
TextChunker — Main Chunking Facade

Replaces the old create_chunks() monolithic function with a class-based
approach using the Strategy pattern.

The TextChunker:
1. Receives text + config
2. Selects the best chunking strategy based on content analysis
3. Applies the strategy
4. Returns chunked results with optional position metadata

Strategy selection order (by priority):
1. TableChunkingStrategy (priority 5)  — CSV/TSV/XLSX/XLS files
2. PageChunkingStrategy (priority 10)  — documents with page markers
3. ProtectedChunkingStrategy (priority 20) — text with protected regions
4. PlainChunkingStrategy (priority 100) — fallback for everything else
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import Chunk, ChunkMetadata
from contextifier_new.errors import ChunkingError
from contextifier_new.chunking.strategies.base import BaseChunkingStrategy
from contextifier_new.chunking.strategies.page_strategy import PageChunkingStrategy
from contextifier_new.chunking.strategies.table_strategy import TableChunkingStrategy
from contextifier_new.chunking.strategies.protected_strategy import ProtectedChunkingStrategy
from contextifier_new.chunking.strategies.plain_strategy import PlainChunkingStrategy


class TextChunker:
    """
    Main chunking interface.

    Usage:
        chunker = TextChunker(config)
        chunks = chunker.chunk(text, file_extension="pdf")
        chunks = chunker.chunk(text, include_position_metadata=True)

    Strategies are prioritized and the first matching one is used.
    Custom strategies can be added via add_strategy().
    """

    def __init__(
        self,
        config: ProcessingConfig,
        *,
        custom_strategies: Optional[List[BaseChunkingStrategy]] = None,
    ) -> None:
        """
        Initialize chunker with config and strategies.

        Args:
            config: Processing configuration.
            custom_strategies: Additional strategies (prepended to built-ins).
        """
        self._config = config
        self._logger = logging.getLogger("contextifier.chunking")

        # Build strategy list (sorted by priority)
        self._strategies: List[BaseChunkingStrategy] = []

        if custom_strategies:
            self._strategies.extend(custom_strategies)

        # Built-in strategies
        self._strategies.extend([
            TableChunkingStrategy(),
            PageChunkingStrategy(),
            ProtectedChunkingStrategy(),
            PlainChunkingStrategy(),
        ])

        # Sort by priority (lower = higher priority)
        self._strategies.sort(key=lambda s: s.priority)

    def chunk(
        self,
        text: str,
        *,
        file_extension: str = "",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        include_position_metadata: Optional[bool] = None,
        preserve_tables: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[List[str], List[Chunk]]:
        """
        Split text into chunks using the best available strategy.

        Args:
            text: Text to chunk.
            file_extension: Source file extension (for strategy selection).
            chunk_size: Override config chunk_size.
            chunk_overlap: Override config chunk_overlap.
            include_position_metadata: Override config metadata flag.
            preserve_tables: Override config table preservation flag.
            **kwargs: Additional context for strategies.

        Returns:
            List of chunk strings or List of Chunk objects.

        Raises:
            ChunkingError: If all strategies fail.
        """
        if not text or not text.strip():
            return [""]

        # Build effective config with overrides
        effective_config = self._apply_overrides(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_position_metadata=include_position_metadata,
            preserve_tables=preserve_tables,
        )

        include_meta = (
            include_position_metadata
            if include_position_metadata is not None
            else self._config.chunking.include_position_metadata
        )

        ext = file_extension.lower().lstrip(".")

        # Select strategy
        strategy = self._select_strategy(text, effective_config, ext, **kwargs)
        self._logger.debug(f"Selected chunking strategy: {strategy.strategy_name}")

        # Execute
        try:
            result = strategy.chunk(
                text,
                effective_config,
                file_extension=ext,
                include_position_metadata=include_meta,
                **kwargs,
            )
            return result
        except NotImplementedError:
            # Strategy not yet implemented — try next
            return self._fallback_chunk(text, effective_config, ext, include_meta, **kwargs)
        except Exception as e:
            raise ChunkingError(
                f"Chunking failed with strategy '{strategy.strategy_name}': {e}",
                cause=e,
            )

    def add_strategy(self, strategy: BaseChunkingStrategy) -> None:
        """Add a custom strategy and re-sort by priority."""
        self._strategies.append(strategy)
        self._strategies.sort(key=lambda s: s.priority)

    @property
    def strategies(self) -> List[BaseChunkingStrategy]:
        """List of registered strategies (sorted by priority)."""
        return list(self._strategies)

    # ── Private ───────────────────────────────────────────────────────────

    def _select_strategy(
        self,
        text: str,
        config: ProcessingConfig,
        ext: str,
        **context: Any,
    ) -> BaseChunkingStrategy:
        """Select the highest-priority strategy that can handle the text."""
        for strategy in self._strategies:
            try:
                if strategy.can_handle(text, config, file_extension=ext, **context):
                    return strategy
            except Exception as e:
                self._logger.warning(
                    f"Strategy {strategy.strategy_name} check failed: {e}"
                )
                continue

        # Should never reach here since PlainChunkingStrategy always returns True
        return self._strategies[-1]

    def _fallback_chunk(
        self,
        text: str,
        config: ProcessingConfig,
        ext: str,
        include_meta: bool,
        **context: Any,
    ) -> Union[List[str], List[Chunk]]:
        """Try remaining strategies as fallback."""
        for strategy in reversed(self._strategies):
            try:
                if strategy.can_handle(text, config, file_extension=ext, **context):
                    return strategy.chunk(
                        text, config,
                        file_extension=ext,
                        include_position_metadata=include_meta,
                        **context,
                    )
            except NotImplementedError:
                continue
            except Exception:
                continue

        # Ultimate fallback: return text as single chunk
        self._logger.warning("All chunking strategies failed, returning text as single chunk")
        if include_meta:
            return [Chunk(text=text, metadata=ChunkMetadata(chunk_index=0))]
        return [text]

    def _apply_overrides(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        include_position_metadata: Optional[bool] = None,
        preserve_tables: Optional[bool] = None,
    ) -> ProcessingConfig:
        """Create config with overrides applied."""
        changes: Dict[str, Any] = {}
        if chunk_size is not None:
            changes["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            changes["chunk_overlap"] = chunk_overlap
        if include_position_metadata is not None:
            changes["include_position_metadata"] = include_position_metadata
        if preserve_tables is not None:
            changes["preserve_tables"] = preserve_tables

        if changes:
            return self._config.with_chunking(**changes)
        return self._config


__all__ = ["TextChunker"]
