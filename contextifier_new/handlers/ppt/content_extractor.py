"""
PptContentExtractor — Stage 4: Extract text from OLE2 PowerPoint.

Extracts text from the ``PowerPoint Document`` binary stream using
record-level parsing. The PPT binary format stores text in specific
record types within the UserEditAtom → PersistDirectory chain.

Key record types for text:
- 0x0FA0 (4000): TextBytesAtom (ANSI text)
- 0x0FA8 (4008): TextCharsAtom (Unicode text)
- 0x0FBA (4026): CString (Unicode string)
- 0x03F3 (1011): SlideListWithText container

This is a best-effort extraction — the PPT binary format is
extremely complex and full fidelity requires LibreOffice.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.types import (
    ChartData,
    PreprocessedData,
    TableData,
)

logger = logging.getLogger(__name__)

# PPT binary record types containing text
_TEXT_BYTES_ATOM = 0x0FA0    # TextBytesAtom: single-byte text (ANSI)
_TEXT_CHARS_ATOM = 0x0FA8    # TextCharsAtom: double-byte text (Unicode UTF-16LE)
_CSTRING_ATOM = 0x0FBA       # CString: Unicode string (used in headers etc.)
_SLIDE_ATOM = 0x03EF         # SlideAtom
_NOTES_CONTAINER = 0x0408    # NotesContainer


class PptContentExtractor(BaseContentExtractor):
    """
    Content extractor for genuine OLE2 PPT files.

    Parses the PowerPoint Document stream to extract text records.
    This is a heuristic approach that handles the majority of PPT files
    but may miss content in complex presentations.
    """

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract text from the PowerPoint Document stream.

        Returns text organized by slides with ``[Slide:N]`` tags.
        """
        pp_stream = preprocessed.resources.get("pp_stream")
        if not pp_stream:
            return ""

        # Parse all text records
        text_records = _parse_text_records(pp_stream)
        if not text_records:
            return ""

        # Group into slides
        slides = _group_into_slides(text_records)

        # Format output
        parts: List[str] = []
        for slide_idx, slide_texts in enumerate(slides):
            slide_tag = self._make_slide_tag(slide_idx + 1)
            if slide_tag:
                parts.append(slide_tag)

            clean_texts = [t.strip() for t in slide_texts if t.strip()]
            if clean_texts:
                parts.append("\n".join(clean_texts))
            else:
                parts.append("[Empty Slide]")

        if not parts:
            # No slides identified — dump all text
            all_texts = [t for _, t in text_records if t.strip()]
            if all_texts:
                tag = self._make_slide_tag(1)
                if tag:
                    parts.append(tag)
                parts.append("\n".join(all_texts))

        result = "\n\n".join(parts)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def extract_images(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[str]:
        """Extract images from the Pictures stream."""
        if self._image_service is None:
            return []

        image_streams: List[bytes] = preprocessed.resources.get("image_streams", [])
        tags: List[str] = []
        processed: Set[str] = set()

        for idx, img_data in enumerate(image_streams):
            if not img_data:
                continue

            import hashlib
            content_hash = hashlib.md5(img_data).hexdigest()[:16]
            if content_hash in processed:
                continue

            try:
                tag = self._image_service.save_and_tag(
                    image_bytes=img_data,
                    custom_name=f"ppt_image_{idx}",
                )
                if tag:
                    tags.append(tag)
                    processed.add(content_hash)
            except Exception as exc:
                logger.debug("Failed to save PPT image %d: %s", idx, exc)

        return tags

    def get_format_name(self) -> str:
        return "ppt"

    # ── Slide tags ────────────────────────────────────────────────────────

    def _make_slide_tag(self, slide_number: int) -> Optional[str]:
        """Generate a ``[Slide:N]`` tag using TagService."""
        if self._tag_service is not None:
            try:
                return self._tag_service.make_slide_tag(slide_number)
            except Exception:
                pass
        return f"[Slide:{slide_number}]"


# ═══════════════════════════════════════════════════════════════════════════════
# Binary stream parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_text_records(data: bytes) -> List[Tuple[int, str]]:
    """
    Parse text records from the PowerPoint Document stream.

    Returns a list of (offset, text) tuples in stream order.
    """
    records: List[Tuple[int, str]] = []
    offset = 0

    while offset + 8 <= len(data):
        # Record header: 8 bytes
        # [0-1]: recVer (4 bits) + recInstance (12 bits)
        # [2-3]: recType (16 bits LE)
        # [4-7]: recLen (32 bits LE)
        rec_type = int.from_bytes(data[offset + 2 : offset + 4], "little")
        rec_len = int.from_bytes(data[offset + 4 : offset + 8], "little")

        if rec_len < 0 or offset + 8 + rec_len > len(data):
            break

        if rec_type == _TEXT_CHARS_ATOM:
            # Unicode UTF-16LE text
            text_data = data[offset + 8 : offset + 8 + rec_len]
            try:
                text = text_data.decode("utf-16-le", errors="replace")
                text = _clean_text(text)
                if text:
                    records.append((offset, text))
            except Exception:
                pass

        elif rec_type == _TEXT_BYTES_ATOM:
            # ANSI single-byte text
            text_data = data[offset + 8 : offset + 8 + rec_len]
            try:
                text = text_data.decode("cp1252", errors="replace")
                text = _clean_text(text)
                if text:
                    records.append((offset, text))
            except Exception:
                pass

        elif rec_type == _CSTRING_ATOM:
            # Unicode string (headers, titles)
            text_data = data[offset + 8 : offset + 8 + rec_len]
            try:
                text = text_data.decode("utf-16-le", errors="replace")
                text = _clean_text(text)
                if text:
                    records.append((offset, text))
            except Exception:
                pass

        offset += 8 + rec_len

    return records


def _clean_text(text: str) -> str:
    """Clean extracted text by removing control characters."""
    # Remove null characters
    text = text.replace("\x00", "")
    # Replace common control chars
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    # Remove non-printable characters except common whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def _group_into_slides(records: List[Tuple[int, str]]) -> List[List[str]]:
    """
    Group text records into slides.

    This is a heuristic: in the PPT binary format, text records
    for each slide are grouped together sequentially. We use
    gaps between record offsets to detect slide boundaries.

    If we can't detect boundaries, all text goes into one slide.
    """
    if not records:
        return []

    if len(records) <= 1:
        return [[r[1] for r in records]]

    # Calculate gaps between consecutive records
    gaps: List[int] = []
    for i in range(1, len(records)):
        gap = records[i][0] - records[i - 1][0]
        gaps.append(gap)

    if not gaps:
        return [[r[1] for r in records]]

    # Use a threshold gap to split slides
    # Records within the same slide tend to be close together
    median_gap = sorted(gaps)[len(gaps) // 2]
    threshold = max(median_gap * 3, 1000)  # At least 1000 bytes gap

    slides: List[List[str]] = []
    current_slide: List[str] = [records[0][1]]

    for i in range(1, len(records)):
        gap = records[i][0] - records[i - 1][0]
        if gap > threshold:
            slides.append(current_slide)
            current_slide = []
        current_slide.append(records[i][1])

    if current_slide:
        slides.append(current_slide)

    return slides


__all__ = ["PptContentExtractor"]
