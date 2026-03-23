# contextifier/pipeline/postprocessor.py
"""
BasePostprocessor — Stage 5: Final Assembly & Cleanup

Responsible for:
- Taking the ExtractionResult from Stage 4
- Applying final transformations (text cleanup, tag normalization)
- Assembling metadata block into the text
- Producing the final output string

This stage addresses a major inconsistency in the old code where
each handler had its own way of:
- Prepending metadata to text
- Cleaning up extra whitespace
- Normalizing tag formats
- Handling edge cases (empty text, error messages)

In v2, the Postprocessor is a standard stage that all handlers pass through,
ensuring consistent output formatting regardless of source format.

Contract:
- postprocess() is the ONLY abstract method
- DefaultPostprocessor provides the standard implementation
  that most handlers should use

Service dependencies:
- MetadataService: for formatting metadata block (injected, optional)
- TagService: for potential tag-based operations (injected, optional)
- ProcessingConfig: for format-specific postprocessing options

Constructor signature MUST match what handlers pass:
    DefaultPostprocessor(
        config,
        metadata_service=...,
        tag_service=...,
    )
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from contextifier.types import DocumentMetadata, ExtractionResult
from contextifier.errors import PostprocessingError

if TYPE_CHECKING:
    from contextifier.config import ProcessingConfig
    from contextifier.services.metadata_service import MetadataService
    from contextifier.services.tag_service import TagService


class BasePostprocessor(ABC):
    """
    Abstract base for all postprocessors.

    Constructor accepts config + services to match handler factory pattern.
    Subclasses that don't need all dependencies can ignore them.
    """

    def __init__(
        self,
        config: Optional["ProcessingConfig"] = None,
        *,
        metadata_service: Optional["MetadataService"] = None,
        tag_service: Optional["TagService"] = None,
    ) -> None:
        """
        Initialize postprocessor.

        Args:
            config: Processing configuration (positional).
            metadata_service: For formatting metadata blocks.
            tag_service: For tag-based operations.
        """
        self._config = config
        self._metadata_service = metadata_service
        self._tag_service = tag_service
        self._logger = logging.getLogger(
            f"contextifier.postprocessor.{self.__class__.__name__}"
        )

    @abstractmethod
    def postprocess(
        self,
        result: ExtractionResult,
        *,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Postprocess the extraction result into final output text.

        Args:
            result: The ExtractionResult from ContentExtractor.
            include_metadata: Whether to prepend metadata block.
            **kwargs: Additional options.

        Returns:
            Final processed text string.

        Raises:
            PostprocessingError: If postprocessing fails.
        """
        ...

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the canonical format name."""
        ...


class DefaultPostprocessor(BasePostprocessor):
    """
    Standard postprocessor used by most handlers.

    Applies:
    1. Metadata block prepending (if metadata exists and include_metadata=True)
    2. Text normalization (excess whitespace, trailing newlines)
    3. Warning comments (if any warnings from extraction)
    """

    def postprocess(
        self,
        result: ExtractionResult,
        *,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> str:
        """Standard postprocessing pipeline."""
        text = result.text or ""

        # 1. Format and prepend metadata
        if include_metadata and result.has_metadata and self._metadata_service:
            metadata_block = self._metadata_service.format_metadata(result.metadata)
            if metadata_block:
                text = metadata_block + "\n\n" + text

        # 2. Normalize whitespace
        text = self._normalize_text(text)

        return text

    def get_format_name(self) -> str:
        return "default"

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean up excessive whitespace while preserving structure."""
        # Collapse 3+ consecutive newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing whitespace on each line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # Strip leading/trailing whitespace from entire text
        text = text.strip()
        return text


class NullPostprocessor(BasePostprocessor):
    """Null postprocessor — returns extraction text unchanged."""

    def postprocess(
        self,
        result: ExtractionResult,
        *,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> str:
        return result.text or ""

    def get_format_name(self) -> str:
        return "null"


__all__ = ["BasePostprocessor", "DefaultPostprocessor", "NullPostprocessor"]
