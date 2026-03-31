# contextifier/handlers/registry.py
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

from contextifier.config import ProcessingConfig
from contextifier.errors import HandlerNotFoundError
from contextifier.handlers.base import BaseHandler

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
        Register all built-in handlers, then discover third-party plugins.

        Built-in handlers are imported directly; missing optional
        dependencies are silently skipped.

        Third-party handlers are discovered via the
        ``contextifier.handlers`` entry-point group.  A package can
        advertise a handler by adding to its ``pyproject.toml``::

            [project.entry-points."contextifier.handlers"]
            my_format = "my_package.handler:MyHandler"
        """
        default_handlers: List[tuple] = [
            # Document formats — one extension per handler
            ("contextifier.handlers.pdf.handler", "PDFHandler"),
            ("contextifier.handlers.docx.handler", "DOCXHandler"),
            ("contextifier.handlers.doc.handler", "DOCHandler"),
            ("contextifier.handlers.pptx.handler", "PPTXHandler"),
            ("contextifier.handlers.ppt.handler", "PPTHandler"),
            ("contextifier.handlers.xlsx.handler", "XLSXHandler"),
            ("contextifier.handlers.xls.handler", "XLSHandler"),
            ("contextifier.handlers.csv.handler", "CSVHandler"),
            ("contextifier.handlers.tsv.handler", "TSVHandler"),
            ("contextifier.handlers.hwp.handler", "HWPHandler"),
            ("contextifier.handlers.hwpx.handler", "HWPXHandler"),
            ("contextifier.handlers.rtf.handler", "RTFHandler"),
            ("contextifier.handlers.html.handler", "HtmlHandler"),
            # Category handlers — multiple extensions by design
            ("contextifier.handlers.text.handler", "TextHandler"),
            ("contextifier.handlers.image.handler", "ImageFileHandler"),
        ]

        for module_path, class_name in default_handlers:
            try:
                import importlib
                module = importlib.import_module(module_path)
                handler_class = getattr(module, class_name)
                self.register(handler_class)
            except (ImportError, AttributeError) as e:
                logger.warning(
                    f"Handler {class_name} not available: {e}"
                )

        # Discover third-party handler plugins
        self._discover_plugins()

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

    def unregister(self, extension: str) -> bool:
        """Remove the handler registration for *extension*.

        Args:
            extension: File extension (with or without dot, any case).

        Returns:
            ``True`` if the extension was previously registered and
            has now been removed, ``False`` if it was not registered.
        """
        ext = extension.lower().lstrip(".")
        return self._handlers.pop(ext, None) is not None

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

    # ── Plugin discovery ──────────────────────────────────────────────────

    _ENTRY_POINT_GROUP = "contextifier.handlers"

    def _discover_plugins(self) -> None:
        """Load third-party handlers advertised via entry points."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return

        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups / dict
        group = (
            eps.select(group=self._ENTRY_POINT_GROUP)
            if hasattr(eps, "select")
            else eps.get(self._ENTRY_POINT_GROUP, [])
        )

        for ep in group:
            try:
                handler_class = ep.load()
                if isinstance(handler_class, type) and issubclass(handler_class, BaseHandler):
                    self.register(handler_class)
                    logger.info("Loaded plugin handler: %s (%s)", ep.name, handler_class.__name__)
                else:
                    logger.warning(
                        "Plugin '%s' does not provide a BaseHandler subclass", ep.name,
                    )
            except Exception as exc:
                logger.warning("Failed to load plugin '%s': %s", ep.name, exc)

    def __repr__(self) -> str:
        n_ext = len(self._handlers)
        n_handlers = len(self.registered_handlers)
        return f"HandlerRegistry({n_ext} extensions, {n_handlers} handlers)"


__all__ = ["HandlerRegistry"]
