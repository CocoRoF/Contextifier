"""
PPTX constants and shared types.

Contains:
- ElementType enum for slide content classification
- SlideElement dataclass for position-aware content
- Wingdings / Symbol font character mappings
- ZIP magic bytes for format validation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Format validation
# ═══════════════════════════════════════════════════════════════════════════════

ZIP_MAGIC = b"PK\x03\x04"


# ═══════════════════════════════════════════════════════════════════════════════
# Slide element types
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class ElementType(str, Enum):
    """Classification of content elements found on a slide."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"


@dataclass
class SlideElement:
    """
    A single content element on a slide, with position for ordering.

    Slide content is collected per-shape, then sorted by position
    (top first, then left) to reconstruct visual reading order.
    """
    element_type: ElementType
    content: str
    position: Tuple[int, int, int, int]  # (left, top, width, height) EMU
    shape_id: int = 0

    @property
    def sort_key(self) -> Tuple[int, int]:
        """Sort by (top, left) — top-to-bottom, then left-to-right."""
        return (self.position[1], self.position[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Special font character mappings
# ═══════════════════════════════════════════════════════════════════════════════

# Wingdings code-point → Unicode mapping
WINGDINGS_MAPPING: Dict[int, str] = {
    # Basic shapes
    0x6C: "●",  # filled circle
    0x6D: "○",  # empty circle
    0x6E: "■",  # filled square
    0x6F: "□",  # empty square
    0x70: "◆",  # filled diamond
    0x71: "◇",  # empty diamond
    0x75: "◆",  # diamond
    0x76: "❖",  # diamond variant
    # Check/X marks
    0xFC: "✓",
    0xFB: "✓",
    0xFD: "✗",
    0xFE: "✘",
    # Arrows
    0xD8: "➢",
    0xE0: "➢",
    0xE1: "⬅",
    0xE2: "⬆",
    0xE3: "⬇",
    0xE4: "⬌",
    0xE8: "➢",
    0xE9: "➣",
    0xEA: "➤",
    0xF0: "➢",
    0xD0: "➢",
    # Pointers
    0x46: "☞",
    0x47: "☜",
    # Stars
    0xAB: "★",
    0xAC: "☆",
    0xA7: "§",
    # Circled numbers
    0x31: "①",
    0x32: "②",
    0x33: "③",
    0x34: "④",
    0x35: "⑤",
    0x36: "⑥",
    0x37: "⑦",
    0x38: "⑧",
    0x39: "⑨",
    0x30: "⓪",
}

# Character-based Wingdings mapping (for string-level matching)
WINGDINGS_CHAR_MAPPING: Dict[str, str] = {
    "§": "■",
    "Ø": "➢",
    "ü": "✓",
    "u": "◆",
    "n": "■",
    "l": "●",
    "o": "□",
    "q": "◇",
    "v": "❖",
    "F": "☞",
    "ð": "➢",
    "Ð": "➢",
    "à": "➢",
    "è": "➢",
    "ê": "➤",
}

# Symbol font code-point → Unicode mapping
SYMBOL_MAPPING: Dict[int, str] = {
    0xB7: "•",   # Bullet
    0xD7: "×",   # Multiplication
    0xF7: "÷",   # Division
    0xA5: "∞",   # Infinity
    0xB1: "±",   # Plus-minus
}


__all__ = [
    "ZIP_MAGIC",
    "ElementType",
    "SlideElement",
    "WINGDINGS_MAPPING",
    "WINGDINGS_CHAR_MAPPING",
    "SYMBOL_MAPPING",
]
