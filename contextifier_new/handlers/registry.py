# contextifier_new/handlers/registry.py
"""
HandlerRegistry — Centralized Handler Management

Responsible for:
- Registering handlers for file extensions
- Looking up the correct handler for a given extension
- Auto-discovering handler modules (optional)
- Ensuring one-to-one mapping from extension → handler

Design:
- Singleton-safe but not enforced (can create multiple registries)
- Extension matching is case-insensitive and dot-agnostic
- Supports lazy handler instantiation for memory efficiency
- Handlers can be registered explicitly or auto-discovered

Fixes from old code:
- Old code built handler registry inside DocumentProcessor._get_handler_registry()
  with lots of try/except ImportError blocks. Now handlers self-register.
- Old code stored bound methods (handler.extract_text) in registry.
  Now stores handler instances.
"""

from __future__ import annotations

import logging
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Type,
)

from contextifier_new.config import ProcessingConfig
from contextifier_new.errors import HandlerNotFoundError
from contextifier_new.handlers.base import BaseHandler

logger = logging.getLogger("contextifier.registry")


class HandlerRegistry:
    """
    Registry that maps file extensions to handler instances.

    Usage:
        registry = HandlerRegistry(config, services=...)

        # Register handler classes
        registry.register(PDFHandler)
        registry.register(DOCXHandler)

        # Or auto-discover all built-in handlers
        registry.register_defaults()

        # Lookup
        handler = registry.get_handler("pdf")
        handler = registry.get_handler(".pdf")  # dot is stripped

        # Check support
        registry.is_supported("pdf")  # True
        registry.supported_extensions  # frozenset({"pdf", "docx", ...})
    """

    def __init__(
        self,
        config: ProcessingConfig,
        *,
        services: Optional[Dict] = None,
    ) -> None:
        """
        Initialize registry.

        Args:
            config: Processing configuration shared by all handlers.
            services: Dictionary of service instances:
                {
                    "image_service": ImageService,
                    "tag_service": TagService,
                    "chart_service": ChartService,
                    "table_service": TableService,
                    "metadata_service": MetadataService,
                }
        """
        self._config = config
        self._services = services or {}
        self._handlers: Dict[str, BaseHandler] = {}
        self._handler_classes: List[Type[BaseHandler]] = []

    def register(self, handler_class: Type[BaseHandler]) -> None:
        """
        Register a handler class.

        Creates an instance and maps all its supported extensions.

        Args:
            handler_class: A BaseHandler subclass.

        Raises:
            TypeError: If handler_class is not a BaseHandler subclass.
        """
        if not (isinstance(handler_class, type) and issubclass(handler_class, BaseHandler)):
            raise TypeError(
                f"Expected BaseHandler subclass, got {handler_class}"
            )

        try:
            handler = handler_class(
                config=self._config,
                image_service=self._services.get("image_service"),
                tag_service=self._services.get("tag_service"),
                chart_service=self._services.get("chart_service"),
                table_service=self._services.get("table_service"),
                metadata_service=self._services.get("metadata_service"),
            )
        except Exception as e:
            logger.warning(
                f"Failed to instantiate {handler_class.__name__}: {e}"
            )
            return

        # Inject registry reference so handler can use delegation
        handler.set_registry(self)

        for ext in handler.supported_extensions:
            ext_lower = ext.lower().lstrip(".")
            if ext_lower in self._handlers:
                logger.warning(
                    f"Extension '{ext_lower}' already registered to "
                    f"{self._handlers[ext_lower].handler_name}, "
                    f"overriding with {handler.handler_name}"
                )
            self._handlers[ext_lower] = handler

        self._handler_classes.append(handler_class)
        logger.debug(
            f"Registered {handler.handler_name} for "
            f"extensions: {sorted(handler.supported_extensions)}"
        )

    def register_defaults(self) -> None:
        """
        Register all built-in handlers.

        Imports and registers each handler, silently skipping any
        whose dependencies are not installed.
        """
        default_handlers: List[tuple] = [
            # Document formats — one extension per handler
            ("contextifier_new.handlers.pdf.handler", "PDFHandler"),
            ("contextifier_new.handlers.docx.handler", "DOCXHandler"),
            ("contextifier_new.handlers.doc.handler", "DOCHandler"),
            ("contextifier_new.handlers.pptx.handler", "PPTXHandler"),
            ("contextifier_new.handlers.ppt.handler", "PPTHandler"),
            ("contextifier_new.handlers.xlsx.handler", "XLSXHandler"),
            ("contextifier_new.handlers.xls.handler", "XLSHandler"),
            ("contextifier_new.handlers.csv.handler", "CSVHandler"),
            ("contextifier_new.handlers.tsv.handler", "TSVHandler"),
            ("contextifier_new.handlers.hwp.handler", "HWPHandler"),
            ("contextifier_new.handlers.hwpx.handler", "HWPXHandler"),
            ("contextifier_new.handlers.rtf.handler", "RTFHandler"),
            # Category handlers — multiple extensions by design
            ("contextifier_new.handlers.text.handler", "TextHandler"),
            ("contextifier_new.handlers.image.handler", "ImageFileHandler"),
        ]

        for module_path, class_name in default_handlers:
            try:
                import importlib
                module = importlib.import_module(module_path)
                handler_class = getattr(module, class_name)
                self.register(handler_class)
            except (ImportError, AttributeError) as e:
                logger.info(
                    f"Handler {class_name} not available: {e}"
                )

    def get_handler(self, extension: str) -> BaseHandler:
        """
        Get the handler for a file extension.

        Args:
            extension: File extension (with or without dot, any case).

        Returns:
            The registered handler instance.

        Raises:
            HandlerNotFoundError: If no handler is registered.
        """
        ext = extension.lower().lstrip(".")
        handler = self._handlers.get(ext)
        if handler is None:
            raise HandlerNotFoundError(
                f"No handler registered for extension: '{ext}'",
                context={"extension": ext, "registered": sorted(self._handlers.keys())},
            )
        return handler

    def is_supported(self, extension: str) -> bool:
        """Check if an extension has a registered handler."""
        return extension.lower().lstrip(".") in self._handlers

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        """All currently registered extensions."""
        return frozenset(self._handlers.keys())

    @property
    def registered_handlers(self) -> List[BaseHandler]:
        """List of unique registered handler instances."""
        seen = set()
        result = []
        for handler in self._handlers.values():
            handler_id = id(handler)
            if handler_id not in seen:
                seen.add(handler_id)
                result.append(handler)
        return result

    def __repr__(self) -> str:
        n_ext = len(self._handlers)
        n_handlers = len(self.registered_handlers)
        return f"HandlerRegistry({n_ext} extensions, {n_handlers} handlers)"


__all__ = ["HandlerRegistry"]
