# contextifier_new/handlers/base.py
"""
BaseHandler — Abstract Base Class for All Document Handlers

KEY DESIGN PRINCIPLES:

1. ONE EXTENSION PER HANDLER
   Every handler handles exactly ONE file extension. No more PPTHandler
   covering both .ppt and .pptx — each gets its own handler class in its
   own folder. Different binary formats MUST have different handlers even
   if they belong to the same "family" (e.g., .xls != .xlsx).

   Exception: category handlers (text, image) where the extension is a
   naming convention, not a format indicator (.txt, .md, .py are all the
   same "plain text" format; .jpg, .png are all "raster image").

2. ENFORCED PIPELINE
   The 5-stage pipeline is NOT overridable:

       process(file_context) -> ExtractionResult
           |- Stage 0: _check_delegation()  [optional delegation hook]
           |- Stage 1: converter.convert()
           |- Stage 2: preprocessor.preprocess()
           |- Stage 3: metadata_extractor.extract()
           |- Stage 4: content_extractor.extract_all()
           +- Stage 5: postprocessor.postprocess()

3. CONTROLLED DELEGATION
   Some extensions may internally contain a different format (e.g., a .doc
   file that is actually RTF). Handlers can override _check_delegation()
   to detect this and delegate to the correct handler via the registry.
   This is the ONLY legitimate cross-handler interaction.

Differences from old BaseHandler:
1. Pipeline is ENFORCED (not bypassable)
2. ONE extension per handler - no multi-extension handlers for document formats
3. No lazy initialization - components are created in __init__
4. Services (image, tag, chart, table, metadata) are injected
5. extract_text() is a thin wrapper around process()
6. _format_chart_data() is eliminated (moved to ChartService)
7. Delegation is EXPLICIT via _check_delegation() + _delegate_to(), not ad-hoc
8. Consistent constructor signature for ALL handlers
9. HandlerRegistry reference injected post-construction for delegation support
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, FrozenSet, Optional

from contextifier_new.config import ProcessingConfig
from contextifier_new.types import (
    ExtractionResult,
    FileContext,
    PipelineStage,
)
from contextifier_new.errors import (
    ConversionError,
    ExtractionError,
    HandlerExecutionError,
    PostprocessingError,
    PreprocessingError,
)
from contextifier_new.pipeline.converter import BaseConverter
from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.pipeline.postprocessor import BasePostprocessor

if TYPE_CHECKING:
    from contextifier_new.handlers.registry import HandlerRegistry
    from contextifier_new.services.image_service import ImageService
    from contextifier_new.services.tag_service import TagService
    from contextifier_new.services.chart_service import ChartService
    from contextifier_new.services.table_service import TableService
    from contextifier_new.services.metadata_service import MetadataService


class BaseHandler(ABC):
    """
    Abstract base class for all document handlers.

    Enforces a strict 5-stage processing pipeline via Template Method.

    Subclasses MUST implement:
    - create_converter() -> BaseConverter
    - create_preprocessor() -> BasePreprocessor
    - create_metadata_extractor() -> BaseMetadataExtractor
    - create_content_extractor() -> BaseContentExtractor
    - create_postprocessor() -> BasePostprocessor
    - supported_extensions -> FrozenSet[str]  (exactly ONE for document handlers)
    - handler_name -> str

    Subclasses MAY override:
    - _check_delegation() - to delegate to another handler based on content

    Subclasses MUST NOT override:
    - process() - the enforced pipeline
    - extract_text() - the user-facing convenience

    Lifecycle:
        handler = PDFHandler(config, services)
        result = handler.process(file_context)          # Full result
        text = handler.extract_text(file_context)       # Text only

    Delegation example (DOCHandler detecting RTF content):
        def _check_delegation(self, file_context, **kwargs):
            if self._is_rtf_content(file_context):
                return self._delegate_to("rtf", file_context, **kwargs)
            return None
    """

    def __init__(
        self,
        config: ProcessingConfig,
        *,
        image_service: Optional["ImageService"] = None,
        tag_service: Optional["TagService"] = None,
        chart_service: Optional["ChartService"] = None,
        table_service: Optional["TableService"] = None,
        metadata_service: Optional["MetadataService"] = None,
    ) -> None:
        """
        Initialize handler with config and services.

        UNIFORM constructor for ALL handlers. No handler adds extra
        parameters (resolves the ImageFileHandler/ocr_engine issue).
        Format-specific options use config.format_options.

        Args:
            config: Processing configuration (immutable).
            image_service: Shared image processing service.
            tag_service: Shared tag generation service.
            chart_service: Shared chart formatting service.
            table_service: Shared table formatting service.
            metadata_service: Shared metadata formatting service.
        """
        self._config = config
        self._image_service = image_service
        self._tag_service = tag_service
        self._chart_service = chart_service
        self._table_service = table_service
        self._metadata_service = metadata_service
        self._handler_registry: Optional["HandlerRegistry"] = None
        self._logger = logging.getLogger(
            f"contextifier.handler.{self.__class__.__name__}"
        )

        # Create pipeline components (eager, not lazy)
        self._converter = self.create_converter()
        self._preprocessor = self.create_preprocessor()
        self._metadata_extractor = self.create_metadata_extractor()
        self._content_extractor = self.create_content_extractor()
        self._postprocessor = self.create_postprocessor()

    # ═══════════════════════════════════════════════════════════════════════
    # Abstract Factory Methods (MUST implement in subclasses)
    # ═══════════════════════════════════════════════════════════════════════

    @abstractmethod
    def create_converter(self) -> BaseConverter:
        """
        Create the format-specific converter.

        Returns:
            A BaseConverter subclass instance for this format.
            Use NullConverter for formats that work with raw bytes.
        """
        ...

    @abstractmethod
    def create_preprocessor(self) -> BasePreprocessor:
        """
        Create the format-specific preprocessor.

        Returns:
            A BasePreprocessor subclass instance for this format.
            Use NullPreprocessor if no preprocessing is needed.
        """
        ...

    @abstractmethod
    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        """
        Create the format-specific metadata extractor.

        Returns:
            A BaseMetadataExtractor subclass instance for this format.
            Use NullMetadataExtractor for formats with no metadata.
        """
        ...

    @abstractmethod
    def create_content_extractor(self) -> BaseContentExtractor:
        """
        Create the format-specific content extractor.

        This is the main workload — subclasses provide format-specific
        implementations that extract text, tables, images, and charts.

        Returns:
            A BaseContentExtractor subclass instance for this format.
        """
        ...

    @abstractmethod
    def create_postprocessor(self) -> BasePostprocessor:
        """
        Create the postprocessor for final assembly.

        Most handlers should return DefaultPostprocessor.
        Override only if format requires special postprocessing.

        Returns:
            A BasePostprocessor subclass instance.
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> FrozenSet[str]:
        """
        Set of file extensions this handler supports.

        RULE: Document format handlers MUST return exactly ONE extension.
        Category handlers (text, image) may return multiple extensions
        since they handle a format category, not a specific binary format.

        Returns:
            Frozenset of lowercase extensions without dots.
            E.g., frozenset({"pdf"}) for PDFHandler
        """
        ...

    @property
    @abstractmethod
    def handler_name(self) -> str:
        """
        Human-readable handler name.

        Returns:
            E.g., "PDF Handler", "DOCX Handler"
        """
        ...

    # ═══════════════════════════════════════════════════════════════════════
    # Public API (FINAL — do not override)
    # ═══════════════════════════════════════════════════════════════════════

    def process(
        self,
        file_context: FileContext,
        *,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> ExtractionResult:
        """
        Execute the full 5-stage processing pipeline.

        ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Convert  │──▶│Preprocess│──▶│ Metadata │──▶│ Content  │──▶│Postproc. │
        └─────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘

        This method is NOT overridable. Handlers customize behaviour
        by providing different pipeline components via factory methods.

        Args:
            file_context: Standardized file input.
            include_metadata: Whether to extract & include metadata.
            **kwargs: Passed to all pipeline stages.

        Returns:
            ExtractionResult with text, metadata, tables, charts, images.

        Raises:
            HandlerExecutionError: Wraps any pipeline stage failure.
        """
        # Stage 0: Delegation check (e.g., .doc that is actually RTF)
        delegation_result = self._check_delegation(file_context, **kwargs)
        if delegation_result is not None:
            self._logger.info(
                f"{self.handler_name} delegated to another handler"
            )
            return delegation_result

        converted = None
        try:
            # Stage 1: Convert
            self._logger.debug(f"Stage 1/5: Converting ({self.handler_name})")
            if not self._converter.validate(file_context):
                raise ConversionError(
                    f"File validation failed for {self.handler_name}",
                    stage=PipelineStage.CONVERT.value,
                    handler=self.handler_name,
                )
            converted = self._converter.convert(file_context, **kwargs)

            # Stage 2: Preprocess
            self._logger.debug(f"Stage 2/5: Preprocessing ({self.handler_name})")
            preprocessed = self._preprocessor.preprocess(converted, **kwargs)

            # Stage 3: Metadata
            self._logger.debug(f"Stage 3/5: Extracting metadata ({self.handler_name})")
            metadata = None
            if include_metadata:
                try:
                    metadata = self._metadata_extractor.extract(
                        preprocessed.content
                    )
                except Exception as e:
                    self._logger.warning(f"Metadata extraction failed: {e}")
                    metadata = None

            # Stage 4: Content extraction
            self._logger.debug(f"Stage 4/5: Extracting content ({self.handler_name})")
            result = self._content_extractor.extract_all(
                preprocessed,
                extract_metadata_result=metadata,
                **kwargs,
            )

            # Stage 5: Postprocess
            self._logger.debug(f"Stage 5/5: Postprocessing ({self.handler_name})")
            final_text = self._postprocessor.postprocess(
                result,
                include_metadata=include_metadata,
                **kwargs,
            )

            # Update result with final text
            result.text = final_text
            return result

        except (ConversionError, PreprocessingError, ExtractionError, PostprocessingError):
            raise
        except Exception as e:
            raise HandlerExecutionError(
                f"Handler '{self.handler_name}' failed: {e}",
                context={"file": file_context.get("file_name", "unknown")},
                cause=e,
            )
        finally:
            # Cleanup converted object
            if converted is not None:
                try:
                    self._converter.close(converted)
                except Exception:
                    pass

    def extract_text(
        self,
        file_context: FileContext,
        *,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from a file (convenience wrapper around process()).

        This is the primary user-facing method. It runs the full pipeline
        and returns just the text string.

        Args:
            file_context: Standardized file input.
            include_metadata: Whether to include metadata block in output.
            **kwargs: Passed to the pipeline.

        Returns:
            Extracted text string.
        """
        result = self.process(file_context, include_metadata=include_metadata, **kwargs)
        return result.text

    # ═══════════════════════════════════════════════════════════════════════
    # Properties (read-only access to internals)
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def config(self) -> ProcessingConfig:
        """Processing configuration."""
        return self._config

    @property
    def converter(self) -> BaseConverter:
        """The converter component."""
        return self._converter

    @property
    def preprocessor(self) -> BasePreprocessor:
        """The preprocessor component."""
        return self._preprocessor

    @property
    def metadata_extractor(self) -> BaseMetadataExtractor:
        """The metadata extractor component."""
        return self._metadata_extractor

    @property
    def content_extractor(self) -> BaseContentExtractor:
        """The content extractor component."""
        return self._content_extractor

    @property
    def postprocessor(self) -> BasePostprocessor:
        """The postprocessor component."""
        return self._postprocessor

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

    @property
    def metadata_service(self) -> Optional["MetadataService"]:
        return self._metadata_service

    @property
    def handler_registry(self) -> Optional["HandlerRegistry"]:
        """Handler registry (set post-construction by HandlerRegistry)."""
        return self._handler_registry

    # ═══════════════════════════════════════════════════════════════════════
    # Delegation Support (override _check_delegation in subclasses)
    # ═══════════════════════════════════════════════════════════════════════

    def set_registry(self, registry: "HandlerRegistry") -> None:
        """
        Inject the handler registry (called by HandlerRegistry after construction).

        This enables cross-handler delegation for formats like .doc that
        may internally contain RTF or misnamed DOCX content.
        """
        self._handler_registry = registry

    def _check_delegation(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> Optional[ExtractionResult]:
        """
        Pre-pipeline hook: check if processing should be delegated.

        Override this in handlers where the file extension doesn't
        guarantee the actual format. For example, .doc files can be:
        - OLE/CFBF binary (real DOC)
        - RTF with .doc extension
        - Misnamed DOCX (ZIP/OOXML)
        - HTML saved as .doc

        Return None to proceed with normal pipeline.
        Return an ExtractionResult to skip the pipeline entirely.

        Args:
            file_context: The file to check.
            **kwargs: Passed through to delegated handler.

        Returns:
            None (proceed normally) or ExtractionResult (delegation result).
        """
        return None

    def _delegate_to(
        self,
        extension: str,
        file_context: FileContext,
        *,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> ExtractionResult:
        """
        Delegate processing to the handler registered for a specific extension.

        This is the ONLY way for one handler to invoke another.
        Requires that set_registry() was called (done automatically by
        HandlerRegistry.register()).

        Args:
            extension: Target extension to delegate to (e.g., "rtf").
            file_context: File context to pass to the delegate handler.
            include_metadata: Whether to include metadata.
            **kwargs: Additional arguments.

        Returns:
            ExtractionResult from the delegate handler.

        Raises:
            HandlerExecutionError: If no registry is available.
            HandlerNotFoundError: If no handler found for extension.
        """
        if self._handler_registry is None:
            raise HandlerExecutionError(
                f"{self.handler_name}: Cannot delegate — no registry available",
                context={"target_extension": extension},
            )
        delegate = self._handler_registry.get_handler(extension)
        self._logger.info(
            f"{self.handler_name} -> delegating to {delegate.handler_name}"
        )
        return delegate.process(
            file_context,
            include_metadata=include_metadata,
            **kwargs,
        )

    def __repr__(self) -> str:
        exts = ", ".join(sorted(self.supported_extensions))
        return f"{self.__class__.__name__}(extensions=[{exts}])"


__all__ = ["BaseHandler"]
