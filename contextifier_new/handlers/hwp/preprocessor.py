# contextifier_new/handlers/hwp/preprocessor.py
"""
HwpPreprocessor — Stage 2: OLE → PreprocessedData

Unwraps the ``HwpConvertedData`` from the converter and stores the
OLE object as ``content``.  Parses the ``DocInfo`` stream to extract
the BinData mapping and records section metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.types import PreprocessedData
from contextifier_new.errors import PreprocessingError

from contextifier_new.handlers.hwp.converter import HwpConvertedData
from contextifier_new.handlers.hwp._docinfo import parse_doc_info
from contextifier_new.handlers.hwp._decoder import is_compressed
from contextifier_new.handlers.hwp._constants import STREAM_BODY_TEXT

logger = logging.getLogger(__name__)


class HwpPreprocessor(BasePreprocessor):
    """
    Preprocess an HWP OLE compound file.

    Records:
    - ``resources["file_data"]``  — raw bytes for OLE metadata extraction
    - ``resources["bin_data_map"]`` — BinData mapping from DocInfo
    - ``properties["compressed"]``  — whether body streams are compressed
    - ``properties["section_count"]`` — number of BodyText sections
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        ole, file_data = self._unwrap(converted_data)
        if ole is None:
            raise PreprocessingError(
                "No OLE object to preprocess",
                stage="preprocess",
                handler="hwp",
            )

        compressed = is_compressed(ole)

        # Parse DocInfo for BinData mapping
        by_id, ordered = parse_doc_info(ole)
        bin_data_map = {"by_storage_id": by_id, "by_index": ordered}

        # Count sections
        sections = [
            e for e in ole.listdir()
            if len(e) >= 2 and e[0] == STREAM_BODY_TEXT and e[1].startswith("Section")
        ]

        return PreprocessedData(
            content=ole,
            raw_content=ole,
            encoding="utf-16le",
            resources={
                "file_data": file_data,
                "bin_data_map": bin_data_map,
            },
            properties={
                "compressed": compressed,
                "section_count": len(sections),
            },
        )

    def get_format_name(self) -> str:
        return "hwp"

    # ── internal ──────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(source: Any):
        """Return (ole, file_data) from either HwpConvertedData or bare OLE."""
        if isinstance(source, HwpConvertedData):
            return source.ole, source.file_data
        if hasattr(source, "listdir"):
            # bare olefile.OleFileIO
            return source, b""
        return None, b""


__all__ = ["HwpPreprocessor"]
