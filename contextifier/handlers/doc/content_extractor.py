# contextifier_new/handlers/doc/content_extractor.py
"""
DocContentExtractor — Stage 4: Extract text, images from OLE2 DOC.

Text extraction strategy
========================
Word 97-2003 stores text as either ANSI or Unicode (UTF-16LE) runs inside
the ``WordDocument`` stream.  Full FIB + piece-table parsing is extremely
complex (the Microsoft specification is ~400 pages).  We use the same
**heuristic** approach as v1.0:

1. Walk the WordDocument stream byte-by-byte looking for consecutive
   UTF-16LE code units that fall within printable ranges (ASCII, Hangul,
   CJK).
2. Collect runs of ≥ ``MIN_TEXT_FRAGMENT_LENGTH`` characters.
3. Deduplicate and join the fragments.

This is imperfect but handles the vast majority of real-world DOC files.

Image extraction
================
OLE2 streams whose paths contain keywords like ``Pictures``, ``Data``,
``Object`` are scanned for known image signatures (PNG, JPEG, GIF, BMP,
TIFF, EMF).  Detected images are saved via ``ImageService``.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Set

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import PreprocessedData, TableData

from contextifier.handlers.doc._constants import (
    IMAGE_SIGNATURES,
    MIN_TEXT_FRAGMENT_LENGTH,
    MIN_UNICODE_BYTES,
    CJK_HIGH_BYTE_RANGES,
)
from contextifier.handlers.doc.preprocessor import DocStreamData

logger = logging.getLogger(__name__)


class DocContentExtractor(BaseContentExtractor):
    """
    Content extractor for genuine OLE2 DOC files.

    Extracts text (heuristic UTF-16LE scanning) and images (OLE stream
    scanning) from the preprocessed ``DocStreamData``.
    """

    # ── BaseContentExtractor abstract methods ─────────────────────────────

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from the WordDocument stream using heuristic scanning.

        Args:
            preprocessed: ``PreprocessedData`` whose ``content`` is a
                          ``DocStreamData`` instance.

        Returns:
            Extracted text (may be incomplete for complex documents).
        """
        stream_data = self._get_stream_data(preprocessed)
        if stream_data is None:
            return ""

        text = self._extract_text_from_word_stream(stream_data.word_data)

        # Embed image tags if image service is available
        image_tags = self._save_images(preprocessed)
        if image_tags:
            text = text + "\n" + "\n".join(image_tags)

        return text

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """
        DOC table extraction is not supported (would require full FIB parsing).

        Returns:
            Empty list.
        """
        return []

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

                if len(run_bytes) >= MIN_UNICODE_BYTES:
                    try:
                        fragment = bytes(run_bytes).decode(
                            "utf-16-le", errors="ignore"
                        ).strip()
                        if (
                            len(fragment) >= MIN_TEXT_FRAGMENT_LENGTH
                            and not fragment.startswith("\\")
                        ):
                            # Clean control characters
                            fragment = fragment.replace("\r\n", "\n").replace("\r", "\n")
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
        if high == 0x00 and (
            0x20 <= low <= 0x7E or low in (0x0D, 0x0A, 0x09)
        ):
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
        if high == 0x00 and (
            0x20 <= low <= 0x7E or low in (0x0D, 0x0A, 0x09)
        ):
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
            if len(data) >= min_len and data[:len(signature)] == signature:
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
