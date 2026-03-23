"""
XLSX handler constants.

Contains:
- ZIP magic bytes for XLSX validation
- OOXML namespace URIs
- Chart type mappings
- Supporting image extensions
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# Magic bytes
# ═══════════════════════════════════════════════════════════════════════════════

ZIP_MAGIC = b"PK\x03\x04"

# ═══════════════════════════════════════════════════════════════════════════════
# OOXML Namespaces
# ═══════════════════════════════════════════════════════════════════════════════

NS_CHART = "http://schemas.openxmlformats.org/drawingml/2006/chart"
NS_DRAWING_MAIN = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_SPREADSHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_SPREADSHEET_DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
NS_PACKAGE = "http://schemas.openxmlformats.org/package/2006/relationships"

OOXML_NS = {
    "c": NS_CHART,
    "a": NS_DRAWING_MAIN,
    "ss": NS_SPREADSHEET,
    "r": NS_RELATIONSHIPS,
    "xdr": NS_SPREADSHEET_DRAWING,
    "pkg": NS_PACKAGE,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Chart type mapping (OOXML chart element → display name)
# ═══════════════════════════════════════════════════════════════════════════════

CHART_TYPE_MAP: dict[str, str] = {
    "barChart": "Bar Chart",
    "bar3DChart": "3D Bar Chart",
    "lineChart": "Line Chart",
    "line3DChart": "3D Line Chart",
    "pieChart": "Pie Chart",
    "pie3DChart": "3D Pie Chart",
    "areaChart": "Area Chart",
    "area3DChart": "3D Area Chart",
    "scatterChart": "Scatter Chart",
    "bubbleChart": "Bubble Chart",
    "doughnutChart": "Doughnut Chart",
    "radarChart": "Radar Chart",
    "stockChart": "Stock Chart",
    "surfaceChart": "Surface Chart",
    "surface3DChart": "3D Surface Chart",
    "ofPieChart": "Pie of Pie Chart",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Image file extensions
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"})
UNSUPPORTED_IMAGE_EXTENSIONS = frozenset({".emf", ".wmf"})

# ═══════════════════════════════════════════════════════════════════════════════
# Layout detection limits
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SCAN_ROWS = 1000
MAX_SCAN_COLS = 100


__all__ = [
    "ZIP_MAGIC",
    "NS_CHART",
    "NS_DRAWING_MAIN",
    "NS_SPREADSHEET",
    "NS_RELATIONSHIPS",
    "NS_SPREADSHEET_DRAWING",
    "NS_PACKAGE",
    "OOXML_NS",
    "CHART_TYPE_MAP",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "UNSUPPORTED_IMAGE_EXTENSIONS",
    "MAX_SCAN_ROWS",
    "MAX_SCAN_COLS",
]
