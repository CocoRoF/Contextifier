# contextifier_new/handlers/doc/preprocessor.py
"""
DocPreprocessor — Stage 2: OLE2 object → preprocessed streams

Reads the essential OLE2 streams out of the opened compound file and
packages them into a ``PreprocessedData`` instance for later stages:

- **WordDocument** stream  → the raw binary that contains text fragments
- **Table stream** (``1Table`` or ``0Table``) → formatting / piece table
- **OLE directory listing** → needed by image extractor

The FIB (File Information Block) header is validated here: we check the
Word magic number (``0xA5EC`` / ``0xA5DC``) so that downstream stages
can rely on having a well-formed stream.
"""

from __future__ import annotations

import struct
import logging
from typing import Any, Dict, List, NamedTuple, Optional

from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.types import PreprocessedData
from contextifier_new.errors import PreprocessingError

from contextifier_new.handlers.doc._constants import (
    WORD_DOCUMENT_STREAM,
    WORD_FIB_MAGIC,
    TABLE_STREAM_NAMES,
    IMAGE_STREAM_KEYWORDS,
)
from contextifier_new.handlers.doc.converter import DocConvertedData

logger = logging.getLogger(__name__)


class DocStreamData(NamedTuple):
    """Intermediate data extracted from OLE2 streams."""
    word_data: bytes                    # Raw WordDocument stream
    table_stream: Optional[bytes]       # 0Table or 1Table (if present)
    table_stream_name: Optional[str]    # Name of the table stream used
    fib_magic: int                      # FIB magic number (0xA5EC / 0xA5DC)
    image_streams: List[str]            # OLE entry paths that may contain images
    encoding: str                       # Inferred text encoding


class DocPreprocessor(BasePreprocessor):
    """
    Preprocessor for genuine OLE2 DOC files.

    Reads binary streams from the ``olefile.OleFileIO`` object produced
    by ``DocConverter`` and packages them for content extraction.
    """

    # ── BasePreprocessor abstract methods ─────────────────────────────────

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """
        Read essential streams from the OLE compound file.

        Args:
            converted_data: ``DocConvertedData`` from the Converter stage.

        Returns:
            PreprocessedData with ``content`` holding a ``DocStreamData``
            and ``resources["image_streams"]`` listing potential image
            stream paths.

        Raises:
            PreprocessingError: If the WordDocument stream is missing or
                                the FIB header is invalid.
        """
        if isinstance(converted_data, DocConvertedData):
            ole = converted_data.ole
        elif hasattr(converted_data, "ole"):
            ole = converted_data.ole
        else:
            # May receive raw olefile object in some delegation paths.
            ole = converted_data

        try:
            stream_data = self._read_streams(ole)
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to read OLE streams: {exc}",
                stage="preprocess",
                handler="doc",
                cause=exc,
            ) from exc

        return PreprocessedData(
            content=stream_data,
            raw_content=converted_data,
            encoding=stream_data.encoding,
            resources={
                "image_streams": stream_data.image_streams,
            },
            properties={
                "fib_magic": hex(stream_data.fib_magic),
                "table_stream": stream_data.table_stream_name,
                "file_extension": (
                    converted_data.file_extension
                    if isinstance(converted_data, DocConvertedData)
                    else "doc"
                ),
            },
        )

    def get_format_name(self) -> str:
        return "doc"

    def validate(self, data: Any) -> bool:
        """Check that we received a DocConvertedData or OLE object."""
        if isinstance(data, DocConvertedData):
            return True
        return hasattr(data, "exists") and callable(data.exists)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _read_streams(self, ole: Any) -> DocStreamData:
        """Read and validate the OLE2 streams we need."""

        # ─ WordDocument stream ─────────────────────────────────────────
        if not ole.exists(WORD_DOCUMENT_STREAM):
            raise PreprocessingError(
                f"'{WORD_DOCUMENT_STREAM}' stream not found in OLE2 file",
                stage="preprocess",
                handler="doc",
            )

        word_data: bytes = ole.openstream(WORD_DOCUMENT_STREAM).read()

        if len(word_data) < 12:
            raise PreprocessingError(
                "WordDocument stream is too short to contain a valid FIB",
                stage="preprocess",
                handler="doc",
            )

        # ─ FIB magic number ────────────────────────────────────────────
        fib_magic: int = struct.unpack("<H", word_data[0:2])[0]
        if fib_magic not in WORD_FIB_MAGIC:
            logger.warning(
                "Unexpected FIB magic 0x%04X (expected 0xA5EC or 0xA5DC). "
                "Proceeding with heuristic extraction.",
                fib_magic,
            )

        # ─ Table stream ───────────────────────────────────────────────
        table_stream: Optional[bytes] = None
        table_stream_name: Optional[str] = None
        for name in TABLE_STREAM_NAMES:
            if ole.exists(name):
                table_stream = ole.openstream(name).read()
                table_stream_name = name
                break

        # ─ Potential image streams ─────────────────────────────────────
        image_streams: List[str] = []
        try:
            for entry in ole.listdir():
                entry_path = "/".join(entry)
                if any(
                    kw in part.lower()
                    for part in entry
                    for kw in IMAGE_STREAM_KEYWORDS
                ):
                    image_streams.append(entry_path)
        except Exception as exc:
            logger.debug("Could not enumerate OLE directory: %s", exc)

        # ─ Encoding heuristic ─────────────────────────────────────────
        # Word 97-2003 stores text as UTF-16LE for Unicode runs.
        encoding = "utf-16-le"

        return DocStreamData(
            word_data=word_data,
            table_stream=table_stream,
            table_stream_name=table_stream_name,
            fib_magic=fib_magic,
            image_streams=image_streams,
            encoding=encoding,
        )


__all__ = ["DocPreprocessor", "DocStreamData"]
