# contextifier_new/pipeline/content_extractor.py
"""
BaseContentExtractor — Stage 4: Content Extraction

This is the main workhorse stage where format-specific text, tables,
images, and charts are extracted from the preprocessed document.

Responsible for:
- Extracting text content from the document
- Extracting tables (as TableData instances)
- Extracting images (saving via ImageService, embedding tags)
- Extracting charts (as ChartData instances)
- Returning an ExtractionResult with all content

CRITICAL DESIGN: This replaces the heterogeneous extract_text() methods
from the old code. The old handlers each had wildly different extraction
logic. In v2, the ContentExtractor has a UNIFORM interface where:

1. extract_text() → str                is ALWAYS available
2. extract_tables() → List[TableData]  is ALWAYS available (empty list if N/A)
3. extract_images() → List[str]        is ALWAYS available (empty list if N/A)
4. extract_charts() → List[ChartData]  is ALWAYS available (empty list if N/A)
5. extract_all() → ExtractionResult    orchestrates all four above

The handler calls extract_all() and gets everything in one shot.
Subclasses override the individual extract_*() methods.

Services (ImageService, TagService, etc.) are injected via constructor,
not accessed through handler properties — this enables testing and
ensures all extractors have the tools they need.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from contextifier_new.types import (
    ChartData,
    DocumentMetadata,
    ExtractionResult,
    PreprocessedData,
    TableData,
)
from contextifier_new.errors import ExtractionError

if TYPE_CHECKING:
    from contextifier_new.services.image_service import ImageService
    from contextifier_new.services.tag_service import TagService
    from contextifier_new.services.chart_service import ChartService
    from contextifier_new.services.table_service import TableService


class BaseContentExtractor(ABC):
    """
    Abstract base for all content extractors.

    Subclasses MUST implement:
    - extract_text() — the core text extraction
    - get_format_name() — format identifier

    Subclasses MAY override:
    - extract_tables() — table extraction (default: empty list)
    - extract_images() — image extraction (default: empty list)
    - extract_charts() — chart extraction (default: empty list)

    The extract_all() method orchestrates all extractions and assembles
    an ExtractionResult. It should NOT be overridden in most cases.
    """

    def __init__(
        self,
        image_service: Optional["ImageService"] = None,
        tag_service: Optional["TagService"] = None,
        chart_service: Optional["ChartService"] = None,
        table_service: Optional["TableService"] = None,
    ) -> None:
        """
        Initialize with injected services.

        All services are optional — extractors that don't need them
        can pass None. But services MUST be provided if the extractor
        uses the corresponding feature.

        Args:
            image_service: For saving images and generating image tags.
            tag_service: For generating page/slide/sheet tags.
            chart_service: For formatting chart data.
            table_service: For formatting table data.
        """
        self._image_service = image_service
        self._tag_service = tag_service
        self._chart_service = chart_service
        self._table_service = table_service
        self._logger = logging.getLogger(
            f"contextifier.extractor.{self.__class__.__name__}"
        )

    # ── Abstract methods (MUST implement) ─────────────────────────────────

    @abstractmethod
    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text content from the preprocessed data.

        This is the primary extraction method. The returned string
        should contain the full document text with structural tags
        (page markers, image tags, chart blocks) already embedded.

        Args:
            preprocessed: Output from the Preprocessor stage.
            **kwargs: Format-specific extraction options.

        Returns:
            The extracted text with embedded structural tags.

        Raises:
            ExtractionError: If text extraction fails.
        """
        ...

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the canonical format name."""
        ...

    # ── Optional methods (override if applicable) ─────────────────────────

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """
        Extract tables from the preprocessed data.

        Default: returns empty list (no tables).
        Override in formats that contain tables (PDF, DOCX, XLSX, etc.).

        Args:
            preprocessed: Output from the Preprocessor stage.
            **kwargs: Format-specific options.

        Returns:
            List of extracted tables as TableData instances.
        """
        return []

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        """
        Extract images from the preprocessed data.

        Default: returns empty list (no images).
        Override in formats that contain embedded images.
        Images should be saved via self._image_service and
        the returned list contains image tag strings.

        Args:
            preprocessed: Output from the Preprocessor stage.
            **kwargs: Format-specific options.

        Returns:
            List of image tag strings.
        """
        return []

    def extract_charts(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[ChartData]:
        """
        Extract charts from the preprocessed data.

        Default: returns empty list (no charts).
        Override in formats that contain charts (DOCX, PPTX, XLSX, HWP, HWPX).

        Args:
            preprocessed: Output from the Preprocessor stage.
            **kwargs: Format-specific options.

        Returns:
            List of ChartData instances.
        """
        return []

    # ── Orchestration (usually not overridden) ────────────────────────────

    def extract_all(
        self,
        preprocessed: PreprocessedData,
        *,
        extract_metadata_result: Optional[DocumentMetadata] = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """
        Run all extraction methods and assemble the result.

        This is the method called by the Handler. It orchestrates:
        1. extract_text()
        2. extract_tables()
        3. extract_images()
        4. extract_charts()

        And packages everything into an ExtractionResult.

        Args:
            preprocessed: Output from the Preprocessor stage.
            extract_metadata_result: Metadata from Stage 3 (passed through).
            **kwargs: Passed to all extract_*() methods.

        Returns:
            Complete ExtractionResult.
        """
        warnings: List[str] = []
        text = ""
        tables: List[TableData] = []
        images: List[str] = []
        charts: List[ChartData] = []

        # Extract text (required)
        try:
            text = self.extract_text(preprocessed, **kwargs)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                f"Text extraction failed: {e}",
                stage="extract_text",
                cause=e,
            )

        # Extract tables (optional)
        try:
            tables = self.extract_tables(preprocessed, **kwargs)
        except Exception as e:
            self._logger.warning(f"Table extraction failed: {e}")
            warnings.append(f"Table extraction failed: {e}")

        # Extract images (optional)
        try:
            images = self.extract_images(preprocessed, **kwargs)
        except Exception as e:
            self._logger.warning(f"Image extraction failed: {e}")
            warnings.append(f"Image extraction failed: {e}")

        # Extract charts (optional)
        try:
            charts = self.extract_charts(preprocessed, **kwargs)
        except Exception as e:
            self._logger.warning(f"Chart extraction failed: {e}")
            warnings.append(f"Chart extraction failed: {e}")

        return ExtractionResult(
            text=text,
            metadata=extract_metadata_result,
            tables=tables,
            charts=charts,
            images=images,
            warnings=warnings,
        )

    # ── Service accessors ─────────────────────────────────────────────────

    @property
    def image_service(self) -> Optional["ImageService"]:
        return self._image_service

    @property
    def tag_service(self) -> Optional["TagService"]:
        return self._tag_service

    @property
    def chart_service(self) -> Optional["ChartService"]:
        return self._chart_service

    @property
    def table_service(self) -> Optional["TableService"]:
        return self._table_service


class NullContentExtractor(BaseContentExtractor):
    """
    Null content extractor — returns empty content.

    Used as a safe default; should not typically be used in practice
    since every format should have a real content extractor.
    """

    def extract_text(self, preprocessed: PreprocessedData, **kwargs: Any) -> str:
        return ""

    def get_format_name(self) -> str:
        return "null"


__all__ = ["BaseContentExtractor", "NullContentExtractor"]
