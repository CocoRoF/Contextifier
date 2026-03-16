# contextifier_new/handlers/rtf/_constants.py
"""
RTF Constants — Shared constants for RTF parsing.

This module is INTERNAL (_-prefixed) and used by the RTF handler's
pipeline components. Not part of the public API.

Ported from v1.0 rtf_constants.py with:
- Type annotations added
- Unused constants removed
- Constants grouped by purpose
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List


# ═══════════════════════════════════════════════════════════════════════════
# Codepage → Python Encoding
# ═══════════════════════════════════════════════════════════════════════════

CODEPAGE_ENCODING_MAP: Dict[int, str] = {
    437: "cp437",
    850: "cp850",
    852: "cp852",
    855: "cp855",
    857: "cp857",
    860: "cp860",
    861: "cp861",
    863: "cp863",
    865: "cp865",
    866: "cp866",
    869: "cp869",
    874: "cp874",
    932: "cp932",      # Japanese Shift-JIS
    936: "gb2312",     # Simplified Chinese
    949: "cp949",      # Korean
    950: "big5",       # Traditional Chinese
    1250: "cp1250",    # Central European
    1251: "cp1251",    # Cyrillic
    1252: "cp1252",    # Western European
    1253: "cp1253",    # Greek
    1254: "cp1254",    # Turkish
    1255: "cp1255",    # Hebrew
    1256: "cp1256",    # Arabic
    1257: "cp1257",    # Baltic
    1258: "cp1258",    # Vietnamese
    10000: "mac_roman",
    65001: "utf-8",
}

# Default encoding fallback list
DEFAULT_ENCODINGS: List[str] = [
    "utf-8", "cp949", "euc-kr", "cp1252", "latin-1",
]


# ═══════════════════════════════════════════════════════════════════════════
# RTF Destination Groups (to skip during text extraction)
# ═══════════════════════════════════════════════════════════════════════════

SKIP_DESTINATIONS: FrozenSet[str] = frozenset({
    "fonttbl", "colortbl", "stylesheet", "listtable",
    "listoverridetable", "revtbl", "rsidtbl", "generator",
    "xmlnstbl", "mmathPr", "themedata", "colorschememapping",
    "datastore", "latentstyles", "pgptbl", "protusertbl",
    "bookmarkstart", "bookmarkend", "bkmkstart", "bkmkend",
    "fldinst", "fldrslt",
})

IMAGE_DESTINATIONS: FrozenSet[str] = frozenset({
    "pict", "shppict", "nonshppict", "blipuid",
})


# ═══════════════════════════════════════════════════════════════════════════
# Shape Property Names (removed during text cleaning)
# ═══════════════════════════════════════════════════════════════════════════

SHAPE_PROPERTY_NAMES: List[str] = [
    "shapeType", "fFlipH", "fFlipV", "rotation",
    "posh", "posrelh", "posv", "posrelv",
    "fLayoutInCell", "fAllowOverlap", "fBehindDocument",
    "fPseudoInline", "fLockAnchor", "fLockPosition",
    "fLockAspectRatio", "fLockRotation", "fLockAgainstSelect",
    "fLockCropping", "fLockVerticies", "fLockText",
    "fLockAdjustHandles", "fLockAgainstGrouping",
    "geoLeft", "geoTop", "geoRight", "geoBottom",
    "shapePath", "pWrapPolygonVertices", "dxWrapDistLeft",
    "dyWrapDistTop", "dxWrapDistRight", "dyWrapDistBottom",
    "fLine", "fFilled", "fillType", "fillColor",
    "fillOpacity", "fillBackColor", "fillBackOpacity",
    "lineColor", "lineOpacity", "lineWidth", "lineStyle",
    "lineDashing", "lineStartArrowhead", "lineStartArrowWidth",
    "lineStartArrowLength", "lineEndArrowhead", "lineEndArrowWidth",
    "lineEndArrowLength", "shadowType", "shadowColor",
    "shadowOpacity", "shadowOffsetX", "shadowOffsetY",
]


# ═══════════════════════════════════════════════════════════════════════════
# Image Format Signatures (for binary image detection)
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_SIGNATURES: Dict[bytes, str] = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"BM": "bmp",
    b"\xd7\xcd\xc6\x9a": "wmf",
    b"\x01\x00\x09\x00": "wmf",
    b"\x01\x00\x00\x00": "emf",
}

RTF_IMAGE_TYPES: Dict[str, str] = {
    "jpegblip": "jpeg",
    "pngblip": "png",
    "wmetafile": "wmf",
    "emfblip": "emf",
    "dibitmap": "bmp",
    "wbitmap": "bmp",
}

SUPPORTED_IMAGE_FORMATS: FrozenSet[str] = frozenset({
    "jpeg", "png", "gif", "bmp",
})
