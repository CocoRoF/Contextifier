# contextifier_new/handlers/image/_constants.py
"""
Constants for image file processing.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# Magic Bytes (file signatures)
# ═══════════════════════════════════════════════════════════════════════════════

MAGIC_JPEG = b"\xff\xd8\xff"
MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
MAGIC_GIF87 = b"GIF87a"
MAGIC_GIF89 = b"GIF89a"
MAGIC_BMP = b"BM"
MAGIC_WEBP_RIFF = b"RIFF"
MAGIC_WEBP_TAG = b"WEBP"      # bytes 8..12
MAGIC_TIFF_LE = b"II\x2a\x00"
MAGIC_TIFF_BE = b"MM\x00\x2a"

# ═══════════════════════════════════════════════════════════════════════════════
# Format Detection Map  (ordered by specificity)
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: (offset, magic_bytes, format_name)
# Checked in order — first match wins.
MAGIC_TABLE: list[tuple[int, bytes, str]] = [
    (0, MAGIC_PNG, "png"),
    (0, MAGIC_GIF89, "gif"),
    (0, MAGIC_GIF87, "gif"),
    (0, MAGIC_JPEG, "jpeg"),
    (0, MAGIC_BMP, "bmp"),
    (0, MAGIC_TIFF_LE, "tiff"),
    (0, MAGIC_TIFF_BE, "tiff"),
    # WebP: RIFF at 0 + WEBP at 8
]

# ═══════════════════════════════════════════════════════════════════════════════
# Supported extensions (from handler.py skeleton)
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    "jpg", "jpeg", "png", "gif", "bmp", "webp",
    "tiff", "tif", "svg", "ico", "heic", "heif",
})

# Subset for which we can validate magic bytes
MAGIC_VALIDATED_EXTENSIONS: frozenset[str] = frozenset({
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif",
})


def detect_image_format(data: bytes) -> str | None:
    """
    Detect image format from binary magic bytes.

    Returns:
        Format string (``"jpeg"``, ``"png"``, ``"gif"``, ``"bmp"``,
        ``"tiff"``, ``"webp"``) or ``None`` if unrecognised.
    """
    if not data or len(data) < 8:
        return None

    for offset, magic, fmt in MAGIC_TABLE:
        end = offset + len(magic)
        if len(data) >= end and data[offset:end] == magic:
            return fmt

    # Special WebP check: RIFF at 0 + WEBP at 8
    if len(data) >= 12 and data[:4] == MAGIC_WEBP_RIFF and data[8:12] == MAGIC_WEBP_TAG:
        return "webp"

    return None


__all__ = [
    "IMAGE_EXTENSIONS",
    "MAGIC_VALIDATED_EXTENSIONS",
    "detect_image_format",
]
