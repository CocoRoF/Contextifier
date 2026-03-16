# contextifier_new/chunking/strategies/protected_strategy.py
"""
ProtectedChunkingStrategy — Region-aware chunking.

Handles text with protected regions (HTML tables, chart blocks,
image tags, metadata blocks) that must not be split.

Priority: 20 (after page strategy, before plain)
"""

from __future__ import annotations

from typing import Any, List, Union

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import Chunk
from contextifier_new.chunking.strategies.base import BaseChunkingStrategy


class ProtectedChunkingStrategy(BaseChunkingStrategy):
    """
    Split text while preserving protected regions.

    Protected region types:
    - HTML tables (<table>...</table>)
    - Chart blocks (config-driven, default: [chart]...[/chart])
    - Textbox blocks ([textbox]...[/textbox])
    - Image tags (config-driven, default: [Image:...])
    - Markdown tables (pipe-delimited)
    - Metadata blocks (config-driven, default: <Document-Metadata>...</Document-Metadata>)

    Algorithm:
    1. Detect all protected regions and their boundaries
    2. Advance through text in chunk_size steps
    3. If split point falls inside protected region → adjust to boundary
    4. Large tables (when force_chunking=True) → split by rows
    5. Overlap is NEVER applied across protected region boundaries
    """

    def can_handle(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        **context: Any,
    ) -> bool:
        """Handle if text contains protected regions."""
        tags = config.tags
        protected_markers = [
            "<table",
            tags.chart_prefix,
            "[textbox]",
            tags.image_prefix,
            tags.metadata_prefix,
        ]
        return any(marker in text for marker in protected_markers)

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
        Chunk text while respecting protected regions.

        Implementation will be provided in the concrete implementation phase.
        """
        # TODO: Implement protected region chunking logic
        # (port from old protected_regions.py with improved interface)
        raise NotImplementedError("Concrete implementation pending")

    @property
    def strategy_name(self) -> str:
        return "protected"

    @property
    def priority(self) -> int:
        return 20


__all__ = ["ProtectedChunkingStrategy"]
