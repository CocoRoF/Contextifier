# contextifier_new/handlers/hwpx/_constants.py
"""
Constants for HWPX (ZIP-based XML) document parsing.

HWPX is the open/XML variant of the Hangul Word Processor format.
It is a ZIP archive containing XML files structured under OPF conventions.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# ZIP Magic Bytes
# ═══════════════════════════════════════════════════════════════════════════════

ZIP_MAGIC = b"PK\x03\x04"  # Standard ZIP local file header

# ═══════════════════════════════════════════════════════════════════════════════
# XML Namespaces
# ═══════════════════════════════════════════════════════════════════════════════

HWPX_NAMESPACES: dict[str, str] = {
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "hc": "http://www.hancom.co.kr/hwpml/2011/core",
    "hh": "http://www.hancom.co.kr/hwpml/2011/head",
    "hs": "http://www.hancom.co.kr/hwpml/2011/section",
    "ha": "http://www.hancom.co.kr/hwpml/2011/app",
    "hl": "http://www.hancom.co.kr/hwpml/2011/layout",
}

OPF_NAMESPACES: dict[str, str] = {
    "opf": "http://www.idpf.org/2007/opf",
}

# Manifest namespace (OASIS)
MANIFEST_NAMESPACES: dict[str, str] = {
    "manifest": "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Standard Paths Inside the ZIP Archive
# ═══════════════════════════════════════════════════════════════════════════════

HPF_PATH = "Contents/content.hpf"           # OPF manifest — bin_item_id → file path
HEADER_PATH = "Contents/header.xml"         # Document metadata / docInfo
VERSION_PATH = "version.xml"                # Version information
MANIFEST_PATH = "META-INF/manifest.xml"     # MIME type information
MIMETYPE_PATH = "mimetype"                  # Plain text MIME type file

# Alternative header paths (for older/variant HWPX files)
HEADER_FILE_PATHS: list[str] = [
    HEADER_PATH,
    "Contents/Header.xml",
    "header.xml",
]

# Section XML pattern prefix
SECTION_PREFIX = "Contents/section"         # Sections are Contents/section0.xml, etc.

# BinData directory prefix
BINDATA_PREFIX = "BinData/"

# Chart directory prefixes (OOXML charts stored inside the ZIP)
CHART_PREFIXES: list[str] = [
    "Chart/",
    "Charts/",
    "Contents/Charts/",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Image Extensions
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp",
})

SKIP_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".wmf", ".emf",
})

# ═══════════════════════════════════════════════════════════════════════════════
# OOXML Chart Namespaces (for embedded charts)
# ═══════════════════════════════════════════════════════════════════════════════

OOXML_CHART_NS: dict[str, str] = {
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}

CHART_TYPE_MAP: dict[str, str] = {
    "barChart": "Bar Chart",
    "bar3DChart": "3D Bar Chart",
    "lineChart": "Line Chart",
    "line3DChart": "3D Line Chart",
    "pieChart": "Pie Chart",
    "pie3DChart": "3D Pie Chart",
    "doughnutChart": "Doughnut Chart",
    "areaChart": "Area Chart",
    "area3DChart": "3D Area Chart",
    "scatterChart": "Scatter Chart",
    "bubbleChart": "Bubble Chart",
    "radarChart": "Radar Chart",
}


__all__ = [
    "ZIP_MAGIC",
    "HWPX_NAMESPACES",
    "OPF_NAMESPACES",
    "MANIFEST_NAMESPACES",
    "HPF_PATH",
    "HEADER_PATH",
    "VERSION_PATH",
    "MANIFEST_PATH",
    "MIMETYPE_PATH",
    "HEADER_FILE_PATHS",
    "SECTION_PREFIX",
    "BINDATA_PREFIX",
    "CHART_PREFIXES",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SKIP_IMAGE_EXTENSIONS",
    "OOXML_CHART_NS",
    "CHART_TYPE_MAP",
]
