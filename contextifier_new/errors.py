# contextifier_new/errors.py
"""
Unified exception hierarchy for Contextifier v2.

All exceptions inherit from ContextifierError, providing:
- Consistent error handling across the entire library
- Machine-readable error codes
- Rich context for debugging

Exception Hierarchy:
    ContextifierError (base)
    ├── ConfigurationError           — Invalid configuration
    ├── FileError                    — File I/O related
    │   ├── FileNotFoundError        — File does not exist
    │   ├── FileReadError            — Cannot read file
    │   └── UnsupportedFormatError   — Unknown/unsupported format
    ├── PipelineError                — Processing pipeline failures
    │   ├── ConversionError          — Binary → format conversion failure
    │   ├── PreprocessingError       — Preprocessing stage failure
    │   ├── ExtractionError          — Content/metadata extraction failure
    │   └── PostprocessingError      — Postprocessing stage failure
    ├── HandlerError                 — Handler-level failures
    │   ├── HandlerNotFoundError     — No handler for extension
    │   └── HandlerExecutionError    — Handler runtime error
    ├── ServiceError                 — Service-level failures
    │   ├── ImageServiceError        — Image processing failure
    │   ├── StorageError             — Storage backend failure
    │   └── OCRError                 — OCR processing failure
    └── ChunkingError                — Text chunking failures
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ContextifierError(Exception):
    """
    Base exception for all Contextifier errors.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code (e.g., "E_CONVERSION")
        context: Optional dictionary with additional debug information
        cause: Optional original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.message = message
        self.code = code or self._default_code()
        self.context = context or {}
        self.cause = cause
        super().__init__(self._format_message())

    def _default_code(self) -> str:
        """Derive error code from class name. E.g., ConversionError → E_CONVERSION."""
        name = type(self).__name__
        # Remove 'Error' suffix and convert CamelCase to UPPER_SNAKE
        name = name.replace("Error", "")
        result = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0:
                result.append("_")
            result.append(ch.upper())
        return f"E_{''.join(result)}"

    def _format_message(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.cause:
            parts.append(f"  Caused by: {type(self.cause).__name__}: {self.cause}")
        if self.context:
            for k, v in self.context.items():
                parts.append(f"  {k}: {v}")
        return "\n".join(parts)

    def with_context(self, **kwargs: Any) -> "ContextifierError":
        """Add context and return self (fluent API)."""
        self.context.update(kwargs)
        return self


# ─── Configuration Errors ─────────────────────────────────────────────────────

class ConfigurationError(ContextifierError):
    """Invalid or missing configuration."""
    pass


# ─── File Errors ──────────────────────────────────────────────────────────────

class FileError(ContextifierError):
    """Base class for file-related errors."""
    pass


class FileNotFoundError(FileError):
    """File does not exist at the specified path."""
    pass


class FileReadError(FileError):
    """Cannot read the file (permission, corruption, etc.)."""
    pass


class UnsupportedFormatError(FileError):
    """File extension is not supported by any registered handler."""
    pass


# ─── Pipeline Errors ─────────────────────────────────────────────────────────

class PipelineError(ContextifierError):
    """Base class for processing pipeline failures."""

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        handler: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {}) or {}
        if stage:
            context["stage"] = stage
        if handler:
            context["handler"] = handler
        super().__init__(message, context=context, **kwargs)


class ConversionError(PipelineError):
    """Failed to convert binary data to format-specific object."""
    pass


class PreprocessingError(PipelineError):
    """Failed during preprocessing stage."""
    pass


class ExtractionError(PipelineError):
    """Failed during content or metadata extraction."""
    pass


class PostprocessingError(PipelineError):
    """Failed during postprocessing / final assembly."""
    pass


# ─── Handler Errors ──────────────────────────────────────────────────────────

class HandlerError(ContextifierError):
    """Base class for handler-level failures."""
    pass


class HandlerNotFoundError(HandlerError):
    """No handler registered for the requested file extension."""
    pass


class HandlerExecutionError(HandlerError):
    """Handler encountered an error during execution."""
    pass


# ─── Service Errors ──────────────────────────────────────────────────────────

class ServiceError(ContextifierError):
    """Base class for service-level failures."""
    pass


class ImageServiceError(ServiceError):
    """Image processing or storage failure."""
    pass


class StorageError(ServiceError):
    """Storage backend operation failure."""
    pass


class OCRError(ServiceError):
    """OCR processing failure."""
    pass


# ─── Chunking Errors ─────────────────────────────────────────────────────────

class ChunkingError(ContextifierError):
    """Text chunking operation failure."""
    pass


__all__ = [
    "ContextifierError",
    "ConfigurationError",
    "FileError",
    "FileNotFoundError",
    "FileReadError",
    "UnsupportedFormatError",
    "PipelineError",
    "ConversionError",
    "PreprocessingError",
    "ExtractionError",
    "PostprocessingError",
    "HandlerError",
    "HandlerNotFoundError",
    "HandlerExecutionError",
    "ServiceError",
    "ImageServiceError",
    "StorageError",
    "OCRError",
    "ChunkingError",
]
