# contextifier_new/handlers/text/preprocessor.py
"""
TextPreprocessor — Stage 2: Normalize decoded text

Takes the TextConvertedData from the TextConverter and produces
a clean PreprocessedData ready for content extraction.

Responsibilities:
1. Strip BOM character (U+FEFF) if present
2. Normalize line endings (\\r\\n, \\r → \\n)
3. Compute text properties (char count, line count)
4. Carry file metadata (extension, category) into PreprocessedData.properties
   so the ContentExtractor can make format-aware decisions (e.g., code vs text mode)

v1.0 Issues resolved:
- TextPreprocessor was a pass-through that stored raw bytes unchanged
- No line ending normalization
- No BOM handling (relied on utf-8-sig encoding detection)
"""

from __future__ import annotations

import logging
from typing import Any, Union

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.handlers.text.converter import TextConvertedData


class TextPreprocessor(BasePreprocessor):
    """
    Preprocessor for plain text and source code files.

    Normalizes decoded text and stores properties that downstream
    stages need (especially file_category for code mode detection).
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs: Any,
    ) -> PreprocessedData:
        """
        Normalize text and compute properties.

        Accepts either:
        - TextConvertedData (from TextConverter) — preferred
        - Plain str — for testing or when converter is bypassed
        - bytes — emergency fallback, decoded with utf-8 + replace

        Args:
            converted_data: Output from TextConverter.
            **kwargs: Ignored (interface compatibility).

        Returns:
            PreprocessedData with normalized text in ``content``
            and metadata in ``properties``.
        """
        text, encoding, file_extension, file_category = self._unpack(converted_data)

        raw_text = text

        # 1. Strip BOM character (U+FEFF)
        #    utf-8-sig encoding strips it during decode, but if the file
        #    was decoded with plain utf-8, a leading BOM may remain.
        if text.startswith("\ufeff"):
            text = text[1:]

        # 2. Normalize line endings → LF only
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Compute text properties
        char_count = len(text)
        line_count = text.count("\n") + 1 if text else 0

        return PreprocessedData(
            content=text,
            raw_content=raw_text,
            encoding=encoding,
            resources={},
            properties={
                "char_count": char_count,
                "line_count": line_count,
                "file_extension": file_extension,
                "file_category": file_category,
            },
        )

    def get_format_name(self) -> str:
        return "text"

    def validate(self, data: Any) -> bool:
        """Accept TextConvertedData, str, or bytes."""
        if isinstance(data, (str, bytes)):
            return True
        if isinstance(data, tuple) and hasattr(data, "text"):
            return True
        return data is not None

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(
        converted_data: Any,
    ) -> tuple[str, str, str, str]:
        """
        Extract text + metadata from converter output.

        Returns (text, encoding, file_extension, file_category).
        """
        if isinstance(converted_data, TextConvertedData):
            return (
                converted_data.text,
                converted_data.encoding,
                converted_data.file_extension,
                converted_data.file_category,
            )

        if isinstance(converted_data, str):
            return (converted_data, "utf-8", "", "")

        if isinstance(converted_data, bytes):
            return (
                converted_data.decode("utf-8", errors="replace"),
                "utf-8",
                "",
                "",
            )

        # Last resort
        return (str(converted_data) if converted_data is not None else "", "utf-8", "", "")


__all__ = ["TextPreprocessor"]
