# contextifier/handlers/doc/content_extractor.py
"""
DocContentExtractor — Stage 4: Extract text, tables, images from OLE2 DOC.

Text extraction strategy
========================
**Primary (v0.3.0)**: FIB + Piece Table based extraction.
Parses the File Information Block header in the WordDocument stream to
locate the Clx structure in the Table Stream, then reads piece descriptors
to reconstruct text in the correct document order.  This handles both
ANSI (cp1252) and Unicode (UTF-16LE) text runs accurately.

**Fallback**: Heuristic UTF-16LE byte scanning (v1.0 approach).
If FIB parsing fails (corrupted FIB, Word 95 documents, truncated
streams), we fall back to scanning the WordDocument stream byte-by-byte
for consecutive UTF-16LE code units in printable ranges.

Table extraction (v0.3.0)
=========================
In MS-DOC, table cells are delimited by ``\\x07`` (BEL character).
After extracting text via the piece table, cell markers are used to
detect and reconstruct table structures.

Image extraction
================
OLE2 streams whose paths contain keywords like ``Pictures``, ``Data``,
``Object`` are scanned for known image signatures (PNG, JPEG, GIF, BMP,
TIFF, EMF).  Detected images are saved via ``ImageService``.
"""

from __future__ import annotations

import re
import logging
from typing import Any, List, Optional, Set

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import PreprocessedData, TableData, TableCell

from contextifier.handlers.doc._constants import (
    IMAGE_SIGNATURES,
    MIN_TEXT_FRAGMENT_LENGTH,
    MIN_UNICODE_BYTES,
    CJK_HIGH_BYTE_RANGES,
)
from contextifier.handlers.doc._fib import parse_fib_text, detect_tables_from_text
from contextifier.handlers.doc.preprocessor import DocStreamData

logger = logging.getLogger(__name__)


class DocContentExtractor(BaseContentExtractor):
    """
    Content extractor for genuine OLE2 DOC files.

    Extracts text (heuristic UTF-16LE scanning) and images (OLE stream
    scanning) from the preprocessed ``DocStreamData``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Configurable threshold for heuristic text extraction
        self._min_text_fragment_length = MIN_TEXT_FRAGMENT_LENGTH
        self._min_unicode_bytes = MIN_UNICODE_BYTES
        if self._config is not None:
            frag_len = self._config.get_format_option(
                "doc",
                "min_text_fragment_length",
                self._min_text_fragment_length,
            )
            self._min_text_fragment_length = int(frag_len)
            self._min_unicode_bytes = self._min_text_fragment_length * 2

    # ── BaseContentExtractor abstract methods ─────────────────────────────

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from the WordDocument stream.

        Uses FIB + Piece Table parsing as the primary method.
        Falls back to heuristic UTF-16LE scanning if FIB parsing fails.

        Args:
            preprocessed: ``PreprocessedData`` whose ``content`` is a
                          ``DocStreamData`` instance.

        Returns:
            Extracted text (may be incomplete for complex documents).
        """
        stream_data = self._get_stream_data(preprocessed)
        if stream_data is None:
            return ""

        # Primary: FIB + Piece Table based extraction
        text = parse_fib_text(stream_data.word_data, stream_data.table_stream)

        if text is None:
            # Fallback: heuristic UTF-16LE byte scanning
            logger.debug("FIB parsing failed, falling back to heuristic extraction")
            text = self._extract_text_from_word_stream(stream_data.word_data)

        # Store raw FIB text for table detection in extract_tables()
        self._last_fib_text = text

        # Embed image tags if image service is available
        image_tags = self._save_images(preprocessed)
        if image_tags:
            text = text + "\n" + "\n".join(image_tags)

        # Remove cell markers from display text (they're used by extract_tables)
        text = text.replace("\x07", "")

        return text

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """
        Extract tables from DOC using cell markers in piece-table text.

        In MS-DOC format, table cells are delimited by ``\\x07`` (BEL).
        This method uses text extracted via the piece table (stored by
        ``extract_text()``) to detect table structures.

        Returns:
            List of ``TableData`` objects, or empty list if no tables found.
        """
        # Use raw FIB text (with cell markers) saved by extract_text()
        raw_text = getattr(self, "_last_fib_text", None)
        if not raw_text:
            return []

        raw_tables = detect_tables_from_text(raw_text)
        if not raw_tables:
            return []

        tables: List[TableData] = []
        for raw_table in raw_tables:
            rows: List[List[TableCell]] = []
            max_cols = 0
            for row_idx, row_cells in enumerate(raw_table):
                row: List[TableCell] = []
                for col_idx, cell_text in enumerate(row_cells):
                    row.append(
                        TableCell(
                            content=cell_text,
                            row_index=row_idx,
                            col_index=col_idx,
                        )
                    )
                rows.append(row)
                max_cols = max(max_cols, len(row_cells))

            tables.append(
                TableData(
                    rows=rows,
                    num_rows=len(rows),
                    num_cols=max_cols,
                )
            )

        return tables

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        """
        Extract and save images found in OLE streams.

        Returns:
            List of image tag strings.
        """
        return self._save_images(preprocessed)

    def get_format_name(self) -> str:
        return "doc"

    # ── Text extraction (heuristic) ───────────────────────────────────────

    def _extract_text_from_word_stream(self, data: bytes) -> str:
        """
        Scan the WordDocument stream for consecutive UTF-16LE text runs.

        This is a heuristic approach that works for the majority of
        Word 97-2003 documents.  It finds sequences of UTF-16LE code
        units where every pair is either:
        - Printable ASCII (0x0020-0x007E) with high byte 0x00
        - Whitespace (CR, LF, TAB) with high byte 0x00
        - Hangul Syllable (high byte 0xAC-0xD7)
        - CJK range (high byte 0x30-0x4E)

        Returns:
            Cleaned text with duplicate fragments removed.
        """
        text_parts: List[str] = []

        i = 0
        data_len = len(data)

        while i < data_len - 1:
            low_byte = data[i]
            high_byte = data[i + 1]

            # A new run can ONLY start with an ASCII printable character
            # (low byte 0x20-0x7E, high byte 0x00).  CJK/Hangul pairs
            # may continue an existing run but CANNOT start one.
            # This prevents false positives at padding-text boundaries.
            if self._is_run_start_pair(low_byte, high_byte):
                # Collect the entire run (CJK allowed as continuation)
                run_bytes: bytearray = bytearray()
                j = i
                while j < data_len - 1:
                    lo = data[j]
                    hi = data[j + 1]
                    if self._is_text_pair(lo, hi):
                        run_bytes.append(lo)
                        run_bytes.append(hi)
                        j += 2
                    else:
                        break

                if len(run_bytes) >= self._min_unicode_bytes:
                    try:
                        fragment = (
                            bytes(run_bytes)
                            .decode("utf-16-le", errors="ignore")
                            .strip()
                        )
                        if len(
                            fragment
                        ) >= self._min_text_fragment_length and not fragment.startswith(
                            "\\"
                        ):
                            # Clean control characters
                            fragment = fragment.replace("\r\n", "\n").replace(
                                "\r", "\n"
                            )
                            fragment = re.sub(
                                r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", fragment
                            )
                            if fragment:
                                text_parts.append(fragment)
                    except Exception:
                        pass

                i = j
            else:
                i += 1

        # Deduplicate while preserving order
        if not text_parts:
            return ""

        seen: Set[str] = set()
        unique: List[str] = []
        for part in text_parts:
            if part not in seen and len(part) > 3:
                seen.add(part)
                unique.append(part)

        result = "\n".join(unique)
        # Collapse excessive blank lines
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    @staticmethod
    def _is_run_start_pair(low: int, high: int) -> bool:
        """
        Check if a UTF-16LE byte pair can START a new text run.

        More restrictive than ``_is_text_pair`` to prevent false
        positives at null-padding → text boundaries.  Specifically,
        pairs with low=0x00 and a non-zero high byte are rejected
        because they are almost always misaligned boundary artefacts
        (e.g. a trailing 0x00 padding byte paired with the first byte
        of actual text).
        """
        # ASCII printable + common whitespace (high byte = 0x00)
        if high == 0x00 and (0x20 <= low <= 0x7E or low in (0x0D, 0x0A, 0x09)):
            return True

        # Hangul / CJK — allowed to start a run, but ONLY if low != 0x00
        # to guard against boundary false positives.
        if low != 0x00:
            if 0xAC <= high <= 0xD7:
                return True
            for range_lo, range_hi in CJK_HIGH_BYTE_RANGES:
                if range_lo <= high <= range_hi:
                    return True

        return False

    @staticmethod
    def _is_text_pair(low: int, high: int) -> bool:
        """Check if a UTF-16LE byte pair represents printable text."""
        # ASCII printable + common whitespace, high byte = 0x00
        if high == 0x00 and (0x20 <= low <= 0x7E or low in (0x0D, 0x0A, 0x09)):
            return True

        # Hangul Syllables (U+AC00 – U+D7AF)
        if 0xAC <= high <= 0xD7:
            return True

        # CJK Unified Ideographs and related blocks
        for range_lo, range_hi in CJK_HIGH_BYTE_RANGES:
            if range_lo <= high <= range_hi:
                return True

        return False

    # ── Image extraction ──────────────────────────────────────────────────

    def _save_images(self, preprocessed: PreprocessedData) -> List[str]:
        """
        Read OLE image streams, detect format, and save via ImageService.

        The preprocessor stored potential image stream paths in
        ``preprocessed.resources["image_streams"]``.

        Returns:
            List of image tag strings.
        """
        if self._image_service is None:
            return []

        image_streams: List[str] = preprocessed.resources.get("image_streams", [])
        if not image_streams:
            return []

        # We need the OLE object — which is in raw_content
        ole = self._get_ole(preprocessed)
        if ole is None:
            return []

        tags: List[str] = []
        processed_hashes: Set[str] = set()

        for stream_path in image_streams:
            try:
                entry = stream_path.split("/")
                stream = ole.openstream(entry)
                data = stream.read()
            except Exception:
                continue

            fmt = self._detect_image_format(data)
            if fmt is None:
                continue

            # Simple deduplication by content hash
            import hashlib

            content_hash = hashlib.md5(data).hexdigest()
            if content_hash in processed_hashes:
                continue
            processed_hashes.add(content_hash)

            try:
                tag = self._image_service.save_and_tag(
                    image_bytes=data,
                    custom_name=f"doc_ole_{content_hash[:12]}",
                )
                if tag:
                    tags.append(tag)
            except Exception as exc:
                logger.debug("Failed to save OLE image: %s", exc)

        return tags

    @staticmethod
    def _detect_image_format(data: bytes) -> Optional[str]:
        """
        Detect image format from binary data using header signatures.

        Returns:
            Format name (e.g. "png", "jpeg") or *None* if not an image.
        """
        if not data or len(data) < 2:
            return None

        for fmt_name, (signature, min_len) in IMAGE_SIGNATURES.items():
            if len(data) >= min_len and data[: len(signature)] == signature:
                # Normalise names that have variants
                if fmt_name.startswith("gif"):
                    return "gif"
                if fmt_name.startswith("tiff"):
                    return "tiff"
                return fmt_name
        return None

    @staticmethod
    def _get_stream_data(preprocessed: PreprocessedData) -> Optional[DocStreamData]:
        """Resolve ``DocStreamData`` from ``PreprocessedData.content``."""
        content = preprocessed.content
        if isinstance(content, DocStreamData):
            return content
        return None

    @staticmethod
    def _get_ole(preprocessed: PreprocessedData) -> Any:
        """
        Resolve the live ``olefile.OleFileIO`` from preprocessed data.

        The OLE object lives inside ``raw_content`` (a ``DocConvertedData``).
        """
        raw = preprocessed.raw_content
        # DocConvertedData
        if hasattr(raw, "ole"):
            return raw.ole
        # Direct OLE
        if hasattr(raw, "openstream"):
            return raw
        return None


__all__ = ["DocContentExtractor"]
