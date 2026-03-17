# contextifier_new/pipeline/converter.py
"""
BaseConverter — Stage 1: Binary → Format Object

Responsible for:
- Taking raw binary file data
- Producing a format-specific workable object
  (e.g., fitz.Document for PDF, python-docx Document for DOCX,
   ZipFile for HWPX, openpyxl Workbook for XLSX)
- Validating that the binary data is well-formed for this format

Contract:
- convert() is the ONLY abstract method
- validate() pre-checks data integrity (optional override)
- close() releases resources (optional override)
- get_format_name() identifies the format (abstract)

Every handler MUST have a converter. For formats that don't need
conversion (plain text), use NullConverter which passes data through.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Optional

from contextifier_new.types import FileContext
from contextifier_new.errors import ConversionError


class BaseConverter(ABC):
    """
    Abstract base for all format converters.

    Subclasses implement convert() and get_format_name().
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"contextifier.converter.{self.__class__.__name__}"
        )

    @abstractmethod
    def convert(self, file_context: FileContext, **kwargs: Any) -> Any:
        """
        Convert binary file data to a format-specific workable object.

        Args:
            file_context: Standardized file input containing binary data
                          and stream. Implementations should use file_context["file_data"]
                          or file_context["file_stream"].
            **kwargs: Format-specific conversion options.

        Returns:
            Format-specific object (Document, Workbook, ZipFile, OLE, etc.)

        Raises:
            ConversionError: If conversion fails.
        """
        ...

    @abstractmethod
    def get_format_name(self) -> str:
        """
        Return the canonical format name.

        Returns:
            Short lowercase name, e.g., "pdf", "docx", "xlsx"
        """
        ...

    def validate(self, file_context: FileContext) -> bool:
        """
        Pre-check whether the binary data is valid for this format.

        Override in subclasses for format-specific header/magic-byte checks.
        Default implementation always returns True.

        Args:
            file_context: File information to validate.

        Returns:
            True if data appears valid, False otherwise.
        """
        file_data = file_context.get("file_data", b"")
        return len(file_data) > 0

    def close(self, converted_object: Any) -> None:
        """
        Release resources held by the converted object.

        Override in subclasses if the format object needs explicit cleanup
        (e.g., fitz.Document.close(), ZipFile.close()).

        Args:
            converted_object: The object returned by convert().
        """
        if hasattr(converted_object, "close") and callable(converted_object.close):
            try:
                converted_object.close()
            except Exception:
                pass

    def _get_stream(self, file_context: FileContext) -> io.BytesIO:
        """Get a fresh seekable stream from file context."""
        stream = file_context.get("file_stream")
        if stream is not None:
            stream.seek(0)
            return stream
        return io.BytesIO(file_context.get("file_data", b""))


class NullConverter(BaseConverter):
    """
    Null converter — passes binary data through unchanged.

    Used for formats that work directly with raw bytes
    (e.g., text files, CSV files).
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> bytes:
        """Return raw file data unchanged."""
        return file_context.get("file_data", b"")

    def get_format_name(self) -> str:
        return "raw"


__all__ = ["BaseConverter", "NullConverter"]
