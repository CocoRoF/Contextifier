# contextifier/handlers/text/converter.py
"""
TextConverter — Stage 1: Binary bytes → Decoded string

Converts raw binary text data into a decoded Python string with
automatic encoding detection. Tries a configurable list of encodings
in order, falling back to UTF-8 with replacement characters.

The converter returns a TextConvertedData NamedTuple that carries:
- text: the decoded string
- encoding: the encoding that succeeded
- file_extension: from FileContext (for downstream stages)
- file_category: from FileContext (for code/text mode detection)

This extra metadata allows the preprocessor and content extractor to
make format-aware decisions without needing direct access to FileContext.

v1.0 Issues resolved:
- TextFileConverter was created but never called (decode was inline)
- Encoding detection was duplicated in handler and converter
- No fallback with errors='replace'
"""

from __future__ import annotations

import logging
from typing import Any, List, NamedTuple, Optional

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError


class TextConvertedData(NamedTuple):
    """
    Output of the TextConverter.

    Carries decoded text along with metadata that downstream
    pipeline stages need for format-aware processing.
    """
    text: str
    encoding: str
    file_extension: str
    file_category: str


# Default encoding priority order.
# utf-8-sig handles BOM-prefixed UTF-8 files.
# cp949/euc-kr cover Korean encodings.
# latin-1 is a lossless 8-bit fallback that never fails,
# but we try it late to prefer more specific encodings first.
DEFAULT_ENCODINGS: List[str] = [
    "utf-8",
    "utf-8-sig",
    "cp949",
    "euc-kr",
    "latin-1",
    "ascii",
]


class TextConverter(BaseConverter):
    """
    Converter for plain text and source code files.

    Decodes raw bytes into a string by trying a prioritised list
    of encodings. The first encoding that succeeds without
    UnicodeDecodeError wins.

    Encoding override:
        - Constructor: ``TextConverter(encodings=["utf-8", "shift_jis"])``
        - Per-call via kwargs: ``converter.convert(ctx, encodings=["shift_jis"])``
    """

    def __init__(self, encodings: Optional[List[str]] = None) -> None:
        super().__init__()
        self._encodings = encodings or list(DEFAULT_ENCODINGS)

    def convert(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> TextConvertedData:
        """
        Decode binary file data to a string.

        Encoding resolution priority:
        1. ``kwargs["encodings"]`` (per-call override, prepended)
        2. Constructor-level ``self._encodings``

        Falls back to ``utf-8`` with ``errors='replace'`` if all
        encodings fail, so this method never raises on encoding.

        Args:
            file_context: Standardized file input with binary data.
            **kwargs: Optional ``encodings`` list to prepend.

        Returns:
            TextConvertedData with decoded text and metadata.

        Raises:
            ConversionError: If file data is missing or empty.
        """
        file_data: bytes = file_context.get("file_data", b"")
        file_ext: str = file_context.get("file_extension", "")
        file_cat: str = file_context.get("file_category", "")

        if not file_data:
            raise ConversionError(
                "Empty file data — nothing to decode",
                stage="convert",
                handler="TextConverter",
            )

        # Merge per-call encodings (prepend) with instance defaults
        extra: List[str] = kwargs.get("encodings", [])
        all_encodings = self._deduplicate(extra + self._encodings)

        # Try each encoding in priority order
        for enc in all_encodings:
            try:
                text = file_data.decode(enc)
                self._logger.debug(
                    "Decoded %d bytes with %s encoding", len(file_data), enc
                )
                return TextConvertedData(
                    text=text,
                    encoding=enc,
                    file_extension=file_ext,
                    file_category=file_cat,
                )
            except (UnicodeDecodeError, LookupError):
                continue

        # Fallback: decode with replacement characters
        self._logger.warning(
            "All encodings failed for %s, falling back to utf-8 with errors='replace'",
            file_context.get("file_name", "unknown"),
        )
        text = file_data.decode("utf-8", errors="replace")
        return TextConvertedData(
            text=text,
            encoding="utf-8",
            file_extension=file_ext,
            file_category=file_cat,
        )

    def get_format_name(self) -> str:
        return "text"

    def validate(self, file_context: FileContext) -> bool:
        """Validate that file data exists and is non-empty."""
        file_data = file_context.get("file_data", b"")
        return len(file_data) > 0

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(encodings: List[str]) -> List[str]:
        """Remove duplicate encodings while preserving order."""
        seen: set[str] = set()
        result: List[str] = []
        for enc in encodings:
            key = enc.lower().replace("-", "").replace("_", "")
            if key not in seen:
                seen.add(key)
                result.append(enc)
        return result


__all__ = ["TextConverter", "TextConvertedData", "DEFAULT_ENCODINGS"]
