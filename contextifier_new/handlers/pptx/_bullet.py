"""
Bullet and numbering extraction for PPTX text frames.

Handles:
- ``buChar``  → bullet characters (•, ○, ■, custom)
- ``buAutoNum`` → auto-numbered lists (1., A., i., (1), etc.)
- ``buNone``  → explicit "no bullet"
- Special font mapping (Wingdings, Symbol) → Unicode equivalents
- Nested indent levels via ``paragraph.level``
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from contextifier_new.handlers.pptx._constants import (
    WINGDINGS_MAPPING,
    WINGDINGS_CHAR_MAPPING,
    SYMBOL_MAPPING,
)

logger = logging.getLogger(__name__)

# DrawingML namespace
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS = {"a": _NS_A}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_with_bullets(text_frame: Any) -> str:
    """
    Extract text from a python-pptx TextFrame, preserving bullet
    and numbering formatting.

    Args:
        text_frame: ``pptx.text.text.TextFrame`` object.

    Returns:
        Multi-line string with bullet/number prefixes and indentation.
    """
    if not text_frame:
        return ""

    lines: list[str] = []
    numbering_state: Dict[int, int] = {}

    try:
        for paragraph in text_frame.paragraphs:
            para_text = paragraph.text.strip()
            level = paragraph.level if hasattr(paragraph, "level") else 0
            indent = "  " * level

            if not para_text:
                lines.append("")
                continue

            bullet_info = _extract_bullet_info(paragraph)

            if bullet_info["type"] == "numbered":
                num_format = bullet_info["format"]
                current_num = _get_or_increment_number(
                    numbering_state, level, bullet_info
                )
                formatted_num = _format_number(current_num, num_format)
                lines.append(f"{indent}{formatted_num} {para_text}")

            elif bullet_info["type"] == "bulleted":
                bullet_char = bullet_info["char"]
                lines.append(f"{indent}{bullet_char} {para_text}")

            else:
                # No bullet — reset numbering
                if numbering_state:
                    numbering_state.clear()
                if level > 0:
                    lines.append(f"{indent}{para_text}")
                else:
                    lines.append(para_text)

    except Exception as exc:
        logger.warning("Error extracting text with bullets: %s", exc)
        return text_frame.text.strip() if text_frame.text else ""

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Bullet detection
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_bullet_info(paragraph: Any) -> Dict[str, Any]:
    """
    Parse bullet/numbering XML from a paragraph element.

    Returns dict with keys:
    - ``type``: ``"bulleted"`` | ``"numbered"`` | ``"none"``
    - ``char``: bullet character (bulleted only)
    - ``format``: numbering format name (numbered only)
    - ``start_at``: starting number (numbered only)
    """
    result: Dict[str, Any] = {
        "type": "none",
        "char": None,
        "format": None,
        "start_at": 1,
    }

    try:
        pPr = paragraph._element.pPr
        if pPr is None:
            return result

        # Explicit no-bullet
        buNone = pPr.find(".//a:buNone", namespaces=_NS)
        if buNone is not None:
            return result

        # Bullet font (for special font mapping)
        buFont = pPr.find(".//a:buFont", namespaces=_NS)
        font_typeface = ""
        if buFont is not None:
            font_typeface = buFont.get("typeface", "").lower()

        # Character bullet
        buChar = pPr.find(".//a:buChar", namespaces=_NS)
        if buChar is not None:
            result["type"] = "bulleted"
            raw_char = buChar.get("char", "•")
            if font_typeface:
                result["char"] = _convert_special_font_char(raw_char, font_typeface)
            else:
                result["char"] = raw_char
            return result

        # Auto-numbering
        buAutoNum = pPr.find(".//a:buAutoNum", namespaces=_NS)
        if buAutoNum is not None:
            result["type"] = "numbered"
            result["format"] = buAutoNum.get("type", "arabicPeriod")
            result["start_at"] = int(buAutoNum.get("startAt", "1"))
            return result

        # Font present but no buChar/buAutoNum → default bullet
        if buFont is not None:
            result["type"] = "bulleted"
            result["char"] = "•"
            return result

    except Exception as exc:
        logger.debug("Error extracting bullet info: %s", exc)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Special font conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _convert_special_font_char(char: str, font_typeface: str) -> str:
    """Convert a Wingdings/Symbol character to a Unicode equivalent."""
    if not char:
        return "•"

    try:
        if "wingdings" in font_typeface:
            # Try character mapping first
            if char in WINGDINGS_CHAR_MAPPING:
                return WINGDINGS_CHAR_MAPPING[char]
            # Then code-point mapping
            char_code = ord(char[0]) if char else 0
            if char_code in WINGDINGS_MAPPING:
                return WINGDINGS_MAPPING[char_code]
            return "•"

        if "symbol" in font_typeface:
            char_code = ord(char[0]) if char else 0
            if char_code in SYMBOL_MAPPING:
                return SYMBOL_MAPPING[char_code]
            return char

        if "webdings" in font_typeface:
            return "•"

        return char

    except Exception:
        return "•"


# ═══════════════════════════════════════════════════════════════════════════════
# Numbering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_or_increment_number(
    numbering_state: Dict[int, int],
    level: int,
    bullet_info: Dict[str, Any],
) -> int:
    """Track and increment per-level numbering state."""
    if level not in numbering_state:
        numbering_state[level] = bullet_info["start_at"]
    else:
        numbering_state[level] += 1

    # Clear deeper levels (reset sub-numbering)
    for lv in list(numbering_state.keys()):
        if lv > level:
            del numbering_state[lv]

    return numbering_state[level]


def _format_number(num: int, format_type: str) -> str:
    """Format a number according to OOXML numbering type."""
    if "roman" in format_type.lower():
        num_str = _to_roman(num)
        if "Lc" in format_type:
            num_str = num_str.lower()
    elif "alpha" in format_type.lower():
        num_str = _to_alpha(num)
        if "Lc" in format_type:
            num_str = num_str.lower()
    else:
        num_str = str(num)

    if "Period" in format_type:
        return f"{num_str}."
    elif "ParenBoth" in format_type:
        return f"({num_str})"
    elif "ParenR" in format_type:
        return f"{num_str})"
    elif "ParenL" in format_type:
        return f"({num_str}"
    elif "Plain" in format_type:
        return num_str
    else:
        return f"{num_str}."


def _to_roman(num: int) -> str:
    """Convert integer to Roman numeral string."""
    val_map = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    parts: list[str] = []
    for value, letter in val_map:
        count, num = divmod(num, value)
        parts.append(letter * count)
    return "".join(parts)


def _to_alpha(num: int) -> str:
    """Convert integer to alphabetic label (1→A, 2→B, …)."""
    parts: list[str] = []
    while num > 0:
        num -= 1
        parts.append(chr(65 + (num % 26)))
        num //= 26
    return "".join(reversed(parts))


__all__ = ["extract_text_with_bullets"]
