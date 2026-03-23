# contextifier/handlers/rtf/preprocessor.py
"""
RtfPreprocessor — Stage 2: Binary processing + image extraction + decoding.

Takes RtfConvertedData (raw RTF bytes) and performs:

1. Binary region scanning:
   - ``\\binN`` tags → skip N bytes of raw binary image data
   - ``\\pict`` groups → extract hex-encoded images
2. Image extraction:
   - Detect image format from magic bytes or RTF type keywords
   - Store extracted images as List[RtfImageData] in resources
3. Binary removal:
   - Replace \\bin and \\pict regions with empty content
4. Decoding:
   - Decode clean bytes → Python string

Returns PreprocessedData with:
- content: RtfParsedData (decoded text string + encoding)
- resources["images"]: List[RtfImageData] for ContentExtractor to save

Design note:
  The preprocessor extracts raw image bytes but does NOT save them.
  Image saving is deferred to the ContentExtractor which has access
  to ImageService. This keeps the preprocessor service-free.

Ported from v1.0 rtf_preprocessor.py with:
- Image extraction decoupled from ImageProcessor
- RtfParsedData NamedTuple for type-safe pipeline data
- Resources dict for image data passing
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData
from contextifier.handlers.rtf.converter import RtfConvertedData
from contextifier.handlers.rtf._decoder import (
    detect_encoding,
    decode_content,
)
from contextifier.handlers.rtf._constants import (
    IMAGE_SIGNATURES,
    RTF_IMAGE_TYPES,
    SUPPORTED_IMAGE_FORMATS,
)

_logger = logging.getLogger("contextifier.rtf.preprocessor")


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

class RtfImageData(NamedTuple):
    """
    Extracted image from RTF binary content.

    Stored in PreprocessedData.resources["images"].
    ContentExtractor saves these via ImageService.
    """
    image_format: str      # "jpeg", "png", "gif", "bmp", etc.
    image_bytes: bytes     # Raw image binary data
    position: int          # Position in original RTF stream
    content_hash: str      # MD5 hash for deduplication


class RtfParsedData(NamedTuple):
    """
    Structured output of the RtfPreprocessor.

    Stored in PreprocessedData.content.
    Consumed by RtfMetadataExtractor and RtfContentExtractor.
    """
    text: str              # Decoded RTF string (binary data removed)
    encoding: str          # Detected encoding
    image_count: int       # Number of images extracted


class _BinaryRegion(NamedTuple):
    """Internal: describes a binary data region in RTF bytes."""
    start_pos: int
    end_pos: int
    bin_type: str          # "bin" or "pict"
    image_format: str
    image_data: bytes


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessor
# ═══════════════════════════════════════════════════════════════════════════

class RtfPreprocessor(BasePreprocessor):
    """
    RTF-specific preprocessor.

    Handles:
    1. ``\\binN`` tag binary data removal
    2. ``\\pict`` hex image extraction
    3. Image deduplication
    4. Encoding detection + bytes → string decoding

    No external service dependencies (ImageService-free).
    """

    def preprocess(
        self,
        converted_data: Any,
        **kwargs: Any,
    ) -> PreprocessedData:
        """
        Preprocess RTF binary data.

        Args:
            converted_data: RtfConvertedData from RtfConverter,
                            or raw bytes for testing.
            **kwargs: Ignored.

        Returns:
            PreprocessedData with RtfParsedData in ``content``
            and image data in ``resources["images"]``.
        """
        raw_bytes, encoding_hint = self._unpack(converted_data)

        if not raw_bytes:
            return PreprocessedData(
                content=RtfParsedData(text="", encoding="utf-8", image_count=0),
                raw_content="",
                encoding="utf-8",
            )

        # Detect encoding (prefer header detection over converter hint)
        encoding = detect_encoding(raw_bytes, default_encoding=encoding_hint)

        # Extract images and clean binary data
        clean_bytes, images = self._process_binary(raw_bytes)

        # Decode clean bytes → string
        text = decode_content(clean_bytes, encoding)

        parsed = RtfParsedData(
            text=text,
            encoding=encoding,
            image_count=len(images),
        )

        return PreprocessedData(
            content=parsed,
            raw_content=text,
            encoding=encoding,
            resources={"images": images},
            properties={
                "file_extension": "rtf",
                "encoding": encoding,
                "image_count": len(images),
            },
        )

    def get_format_name(self) -> str:
        return "rtf"

    def validate(self, data: Any) -> bool:
        """Accept RtfConvertedData or bytes."""
        if isinstance(data, bytes):
            return data.lstrip()[:5] == b"{\\rtf"
        if isinstance(data, tuple) and hasattr(data, "raw_bytes"):
            return True
        return data is not None

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _unpack(converted_data: Any) -> Tuple[bytes, str]:
        """Extract raw bytes and encoding hint from converter output."""
        if isinstance(converted_data, RtfConvertedData):
            return converted_data.raw_bytes, converted_data.encoding
        if isinstance(converted_data, bytes):
            return converted_data, "cp949"
        return b"", "cp949"

    def _process_binary(
        self,
        content: bytes,
    ) -> Tuple[bytes, List[RtfImageData]]:
        """
        Extract images and remove binary data from RTF bytes.

        Returns:
            Tuple of (clean_bytes, list of RtfImageData).
        """
        seen_hashes: Set[str] = set()

        # Find all binary regions
        bin_regions = _find_bin_regions(content)
        pict_regions = _find_pict_regions(content, bin_regions)

        all_regions = sorted(
            bin_regions + pict_regions,
            key=lambda r: r.start_pos,
        )

        # Extract unique images
        images: List[RtfImageData] = []
        for region in all_regions:
            if not region.image_data:
                continue
            if region.image_format not in SUPPORTED_IMAGE_FORMATS:
                continue

            content_hash = hashlib.md5(region.image_data).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            images.append(RtfImageData(
                image_format=region.image_format,
                image_bytes=region.image_data,
                position=region.start_pos,
                content_hash=content_hash,
            ))

        # Remove binary regions from content
        clean_bytes = _remove_binary_regions(content, all_regions)

        return clean_bytes, images


# ═══════════════════════════════════════════════════════════════════════════
# Binary Region Discovery (module-level for testability)
# ═══════════════════════════════════════════════════════════════════════════

def _find_bin_regions(content: bytes) -> List[_BinaryRegion]:
    """
    Find ``\\binN`` tags that mark raw binary data embedded in RTF.

    RTF's ``\\binN`` directive skips the next N bytes of raw binary data.
    These regions typically contain embedded images.

    Args:
        content: RTF binary data.

    Returns:
        List of _BinaryRegion for \\bin tags.
    """
    regions: List[_BinaryRegion] = []
    pattern = rb"\\bin(\d+)"

    for match in re.finditer(pattern, content):
        try:
            bin_size = int(match.group(1))
            bin_tag_end = match.end()

            data_start = bin_tag_end
            # Skip optional space after \binN
            if data_start < len(content) and content[data_start : data_start + 1] == b" ":
                data_start += 1

            data_end = data_start + bin_size
            if data_end > len(content):
                continue

            binary_data = content[data_start:data_end]
            image_format = _detect_image_format(binary_data)

            # Expand region to include parent \shppict group if present
            group_start = match.start()
            group_end = data_end

            search_start = max(0, match.start() - 500)
            search_area = content[search_start : match.start()]

            shppict_pos = search_area.rfind(b"\\shppict")
            if shppict_pos != -1:
                abs_pos = search_start + shppict_pos
                brace_pos = abs_pos
                while brace_pos > 0 and content[brace_pos : brace_pos + 1] != b"{":
                    brace_pos -= 1
                group_start = brace_pos

                # Find matching closing brace
                depth = 1
                j = data_end
                while j < len(content) and depth > 0:
                    if content[j : j + 1] == b"{":
                        depth += 1
                    elif content[j : j + 1] == b"}":
                        depth -= 1
                    j += 1
                group_end = j

            regions.append(_BinaryRegion(
                start_pos=group_start,
                end_pos=group_end,
                bin_type="bin",
                image_format=image_format,
                image_data=binary_data,
            ))
        except (ValueError, IndexError):
            continue

    return regions


def _find_pict_regions(
    content: bytes,
    exclude_regions: List[_BinaryRegion],
) -> List[_BinaryRegion]:
    """
    Find ``\\pict`` groups containing hex-encoded image data.

    RTF images can be encoded as hex strings (pairs of hex chars)
    within ``\\pict`` groups. This function extracts those and
    decodes the hex to binary.

    Args:
        content: RTF binary data.
        exclude_regions: Already-found \\bin regions to skip.

    Returns:
        List of _BinaryRegion for hex-encoded images.
    """
    regions: List[_BinaryRegion] = []

    bin_positions = {r.start_pos for r in exclude_regions if r.bin_type == "bin"}
    excluded_ranges = [(r.start_pos, r.end_pos) for r in exclude_regions]

    def is_excluded(pos: int) -> bool:
        return any(start <= pos < end for start, end in excluded_ranges)

    def has_bin_nearby(pict_pos: int) -> bool:
        return any(pict_pos < bp < pict_pos + 200 for bp in bin_positions)

    try:
        # Decode as cp1252 for regex matching (lossless for control chars)
        text = content.decode("cp1252", errors="replace")
        pict_pattern = r"\\pict\s*((?:\\[a-zA-Z]+\d*\s*)*)"

        for match in re.finditer(pict_pattern, text):
            start_pos = match.start()

            if is_excluded(start_pos) or has_bin_nearby(start_pos):
                continue

            # Detect image type from attributes
            attrs = match.group(1)
            image_format = ""
            for rtf_type, fmt in RTF_IMAGE_TYPES.items():
                if rtf_type in attrs:
                    image_format = fmt
                    break

            # Extract hex data
            hex_start = match.end()
            hex_chars: List[str] = []
            i = hex_start

            while i < len(text):
                ch = text[i]
                if ch in "0123456789abcdefABCDEF":
                    hex_chars.append(ch)
                elif ch in " \t\r\n":
                    pass  # skip whitespace in hex data
                elif ch == "}":
                    break
                elif ch == "\\":
                    if text[i : i + 4] == "\\bin":
                        hex_chars = []
                        break
                    while i < len(text) and text[i] not in " \t\r\n}":
                        i += 1
                    continue
                else:
                    break
                i += 1

            hex_str = "".join(hex_chars)

            if len(hex_str) >= 32:
                try:
                    image_data = bytes.fromhex(hex_str)
                    if not image_format:
                        image_format = _detect_image_format(image_data)

                    if image_format:
                        regions.append(_BinaryRegion(
                            start_pos=start_pos,
                            end_pos=i,
                            bin_type="pict",
                            image_format=image_format,
                            image_data=image_data,
                        ))
                except ValueError:
                    continue
    except Exception as e:
        _logger.warning("Error scanning pict regions: %s", e)

    return regions


def _detect_image_format(data: bytes) -> str:
    """Detect image format from binary data's magic bytes."""
    if not data or len(data) < 4:
        return ""
    for signature, format_name in IMAGE_SIGNATURES.items():
        if data.startswith(signature):
            return format_name
    # Additional check: JPEG starts with FF D8
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return "jpeg"
    return ""


def _remove_binary_regions(
    content: bytes,
    regions: List[_BinaryRegion],
) -> bytes:
    """Remove binary data regions from RTF content."""
    if not regions:
        return content

    # Process in reverse order to preserve positions
    result = bytearray(content)
    for region in sorted(regions, key=lambda r: r.start_pos, reverse=True):
        result[region.start_pos : region.end_pos] = b""

    return bytes(result)


__all__ = [
    "RtfPreprocessor",
    "RtfParsedData",
    "RtfImageData",
]
