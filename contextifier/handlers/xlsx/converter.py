"""
XlsxConverter — Stage 1: Raw bytes → openpyxl Workbook.

Validates the XLSX file (ZIP format with OOXML content types)
and opens it with ``openpyxl.load_workbook(data_only=True)``
to get calculated cell values instead of formulas.

Returns ``XlsxConvertedData`` containing both the Workbook
and the original file bytes (needed by the preprocessor to
re-open the file as a ZIP archive for chart/image extraction).
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Any, NamedTuple

import openpyxl

from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError

from contextifier.handlers.xlsx._constants import ZIP_MAGIC

logger = logging.getLogger(__name__)


class XlsxConvertedData(NamedTuple):
    """Result of the XLSX conversion stage."""
    workbook: openpyxl.Workbook   # Opened openpyxl Workbook
    file_data: bytes              # Original file bytes (for ZIP re-reading)


class XlsxConverter(BaseConverter):
    """
    Converter for XLSX files → openpyxl Workbook.

    Uses ``data_only=True`` so formulas return their cached
    calculated values rather than formula strings.
    """

    def convert(self, file_context: FileContext, **kwargs: Any) -> XlsxConvertedData:
        """
        Open XLSX file with openpyxl.

        Returns:
            An ``XlsxConvertedData`` with the Workbook and raw bytes.

        Raises:
            ConversionError: If the data is not a valid XLSX file.
        """
        file_data: bytes = file_context.get("file_data", b"")
        if not file_data:
            raise ConversionError(
                "Empty file data",
                stage="convert",
                handler="xlsx",
            )

        stream = io.BytesIO(file_data)

        try:
            wb = openpyxl.load_workbook(stream, data_only=True)
        except Exception as exc:
            raise ConversionError(
                f"Failed to open XLSX file: {exc}",
                stage="convert",
                handler="xlsx",
                cause=exc,
            ) from exc

        return XlsxConvertedData(workbook=wb, file_data=file_data)

    def get_format_name(self) -> str:
        return "xlsx"

    def validate(self, file_context: FileContext) -> bool:
        """
        Check that the file is a valid XLSX (ZIP with OOXML content).

        Validates:
        1. Starts with ZIP magic bytes (PK\\x03\\x04)
        2. Contains [Content_Types].xml (OOXML marker)
        """
        data: bytes = file_context.get("file_data", b"")
        if len(data) < len(ZIP_MAGIC):
            return False
        if data[:4] != ZIP_MAGIC:
            return False

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "[Content_Types].xml" not in zf.namelist():
                    return False
        except (zipfile.BadZipFile, Exception):
            return False

        return True

    def close(self, converted_object: Any) -> None:
        """Close the openpyxl Workbook."""
        wb = None
        if isinstance(converted_object, XlsxConvertedData):
            wb = converted_object.workbook
        elif isinstance(converted_object, openpyxl.Workbook):
            wb = converted_object

        if wb is not None:
            try:
                wb.close()
            except Exception:
                pass


__all__ = ["XlsxConverter", "XlsxConvertedData"]
