# contextifier_new/handlers/rtf/converter.py
"""
RtfConverter — Stage 1: Validate and wrap raw RTF bytes.

RTF files are binary blobs that interleave control codes, text,
and embedded image data. Unlike DOCX (which has a clear ZIP → XML
structure), RTF conversion is minimal at this stage:

1. Validate the ``{\\rtf`` magic signature.
2. Detect encoding from ``\\ansicpg`` header (quick ASCII scan).
3. Wrap in RtfConvertedData for downstream stages.

The heavy lifting (binary removal, decoding, image extraction)
is deferred to the Preprocessor stage where it naturally fits.

Returns:
    RtfConvertedData NamedTuple carrying raw bytes + encoding hint.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError
from contextifier.handlers.rtf._decoder import detect_encoding


_RTF_MAGIC = b"{\\rtf"


class RtfConvertedData(NamedTuple):
    """
    Output of the RtfConverter.

    Attributes:
        raw_bytes: Original RTF binary data (unmodified).
        encoding: Detected encoding from \\ansicpg (hint for preprocessor).
        file_extension: From FileContext ("rtf").
    """
    raw_bytes: bytes
    encoding: str
    file_extension: str


class RtfConverter(BaseConverter):
    """
    RTF file converter — validation and wrapping.

    This converter is intentionally thin. RTF's complexity is in
    its control code grammar, not in a container structure. The
    Preprocessor handles binary extraction and decoding.
    """

    def convert(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> RtfConvertedData:
        """
        Validate and wrap RTF file data.

        Args:
            file_context: Standardized file input with binary data.
            **kwargs: Ignored.

        Returns:
            RtfConvertedData with raw bytes and encoding hint.

        Raises:
            ConversionError: If data is empty or not RTF format.
        """
        file_data: bytes = file_context.get("file_data", b"")
        file_ext: str = file_context.get("file_extension", "rtf")

        if not file_data:
            raise ConversionError(
                "Empty file data — nothing to process",
                stage="convert",
                handler="RtfConverter",
            )

        # Validate RTF magic (allow leading whitespace)
        stripped = file_data.lstrip()
        if not stripped.startswith(_RTF_MAGIC):
            raise ConversionError(
                f"Not a valid RTF file (missing {{\\rtf header)",
                stage="convert",
                handler="RtfConverter",
            )

        # Detect encoding from header
        encoding = detect_encoding(file_data, default_encoding="cp949")

        self._logger.debug(
            "RTF validated: %d bytes, encoding hint=%s",
            len(file_data),
            encoding,
        )

        return RtfConvertedData(
            raw_bytes=file_data,
            encoding=encoding,
            file_extension=file_ext,
        )

    def validate(self, file_context: FileContext) -> bool:
        """Check if file data looks like RTF."""
        data = file_context.get("file_data", b"")
        if not data or len(data) < 5:
            return False
        return data.lstrip()[:5] == _RTF_MAGIC

    def get_format_name(self) -> str:
        return "rtf"

    def close(self, converted: Any) -> None:
        """Nothing to close for RTF."""
        pass


__all__ = ["RtfConverter", "RtfConvertedData"]
