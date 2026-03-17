"""
Constants for DOCX (Office Open XML) handler.

DOCX files are ZIP archives containing XML parts conforming to
ECMA-376 / ISO/IEC 29500 (Office Open XML).

Defines:
- OOXML namespaces used throughout DOCX parsing
- ElementType enum for classifying body elements
- Chart type mapping
- ZIP magic bytes for validation
"""

from __future__ import annotations

from enum import Enum, unique


# ── ZIP / OOXML magic ────────────────────────────────────────────────────

ZIP_MAGIC: bytes = b"PK\x03\x04"
CONTENT_TYPES_PATH: str = "[Content_Types].xml"

# ── OOXML namespaces ─────────────────────────────────────────────────────

NAMESPACES: dict[str, str] = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "v": "urn:schemas-microsoft-com:vml",
}


# ── Element type enum ────────────────────────────────────────────────────

@unique
class ElementType(str, Enum):
    """Classification of document body elements."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    PAGE_BREAK = "page_break"


# ── Chart type mapping ───────────────────────────────────────────────────

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
    "surfaceChart": "Surface Chart",
    "surface3DChart": "3D Surface Chart",
    "stockChart": "Stock Chart",
    "ofPieChart": "Split Pie Chart",
}


__all__ = [
    "ZIP_MAGIC",
    "CONTENT_TYPES_PATH",
    "NAMESPACES",
    "ElementType",
    "CHART_TYPE_MAP",
]
