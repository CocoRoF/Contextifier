# contextifier_new/handlers/rtf/_cleaner.py
"""
RTF Text Cleaner — Remove RTF control codes and extract pure text.

Internal module shared by RtfContentExtractor and RtfMetadataExtractor.

Functions:
- clean_rtf_text: Token-based parsing to remove RTF codes, preserve text
- remove_destination_groups: Remove {\\*\\destination ...} groups
- remove_shape_groups: Remove shape groups but preserve \\shptxt content
- remove_shape_property_groups: Remove {\\sp{\\sn xxx}{\\sv yyy}} blocks
- remove_shprslt_blocks: Remove backward-compatibility shape results
- find_excluded_regions: Find header/footer/footnote regions
- is_in_excluded_region: Position check against excluded regions

Ported from v1.0 rtf_text_cleaner.py + rtf_region_finder.py with:
- Consolidated into a single module (logically cohesive)
- Type annotations added throughout
- Image tag protection preserved
- Shape property cleanup preserved
"""

from __future__ import annotations

import re
from typing import List, Tuple

from contextifier.handlers.rtf._constants import (
    SHAPE_PROPERTY_NAMES,
    SKIP_DESTINATIONS,
    IMAGE_DESTINATIONS,
)
from contextifier.handlers.rtf._decoder import decode_bytes


# Precompile shape property name pattern for performance
_SHAPE_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(SHAPE_PROPERTY_NAMES) + r")\b"
)


# ═══════════════════════════════════════════════════════════════════════════
# Main Text Cleaner
# ═══════════════════════════════════════════════════════════════════════════

def clean_rtf_text(text: str, encoding: str = "cp949") -> str:
    """
    Remove RTF control codes and extract pure text.

    Uses token-based parsing (character-by-character scan) to
    accurately remove control words while preserving content.

    Handles:
    - Special escapes: ``\\\\``, ``\\{``, ``\\}``, ``\\~``, ``\\-``, ``\\_``
    - Hex escapes: ``\\'XX``
    - Unicode: ``\\uNNNN?``
    - Paragraph/line breaks: ``\\par``, ``\\line``
    - Tabs: ``\\tab``
    - Control words with numeric parameters
    - Braces (group delimiters)

    Protects ``[image:...]`` tags from being corrupted.

    Args:
        text: RTF text with control codes.
        encoding: Encoding for hex escape decoding.

    Returns:
        Cleaned plain text.
    """
    if not text:
        return ""

    # Protect image tags with temporary markers
    image_tags: List[str] = []

    def save_image_tag(m: re.Match) -> str:
        image_tags.append(m.group())
        return f"\x00IMG{len(image_tags) - 1}\x00"

    text = re.sub(r"\[image:[^\]]+\]", save_image_tag, text)

    # Remove shape property inline patterns
    text = re.sub(r"\{\\sp\{\\sn\s*\w+\}\{\\sv\s*[^}]*\}\}", "", text)
    text = re.sub(
        r"shapeType\d+[a-zA-Z0-9]+(?:posrelh\d+posrelv\d+)?", "", text
    )
    text = re.sub(
        r"\\shp(?:inst|txt|left|right|top|bottom|bx\w+|by\w+|wr\d+|fblwtxt\d+|z\d+|lid\d+)\b\d*",
        "",
        text,
    )

    # Token-based parsing
    result: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Restore image tag markers
        if ch == "\x00" and i + 3 < n and text[i + 1 : i + 4] == "IMG":
            end_idx = text.find("\x00", i + 4)
            if end_idx != -1:
                try:
                    tag_idx = int(text[i + 4 : end_idx])
                    result.append(image_tags[tag_idx])
                    i = end_idx + 1
                    continue
                except (ValueError, IndexError):
                    pass

        if ch == "\\":
            if i + 1 < n:
                next_ch = text[i + 1]

                # Special character escapes
                if next_ch == "\\":
                    result.append("\\")
                    i += 2
                    continue
                elif next_ch == "{":
                    result.append("{")
                    i += 2
                    continue
                elif next_ch == "}":
                    result.append("}")
                    i += 2
                    continue
                elif next_ch == "~":
                    result.append("\u00A0")  # non-breaking space
                    i += 2
                    continue
                elif next_ch == "-":
                    result.append("\u00AD")  # soft hyphen
                    i += 2
                    continue
                elif next_ch == "_":
                    result.append("\u2011")  # non-breaking hyphen
                    i += 2
                    continue
                elif next_ch == "'":
                    # Hex escape \'XX
                    if i + 3 < n:
                        try:
                            hex_val = text[i + 2 : i + 4]
                            byte_val = int(hex_val, 16)
                            try:
                                result.append(bytes([byte_val]).decode(encoding))
                            except Exception:
                                try:
                                    result.append(bytes([byte_val]).decode("cp1252"))
                                except Exception:
                                    pass
                            i += 4
                            continue
                        except (ValueError, IndexError):
                            pass
                    i += 1
                    continue
                elif next_ch == "*":
                    # \* destination marker — skip
                    i += 2
                    continue
                elif next_ch.isalpha():
                    # Control word: \word[N][delimiter]
                    j = i + 1
                    while j < n and text[j].isalpha():
                        j += 1

                    control_word = text[i + 1 : j]

                    # Skip numeric parameter
                    while j < n and (text[j].isdigit() or text[j] == "-"):
                        j += 1

                    # Delimiter space is part of control word
                    if j < n and text[j] == " ":
                        j += 1

                    # Special control words that produce output
                    if control_word in ("par", "line"):
                        result.append("\n")
                    elif control_word == "tab":
                        result.append("\t")
                    elif control_word == "u":
                        # Unicode: \uN?
                        um = re.match(r"\\u(-?\d+)\??", text[i:])
                        if um:
                            try:
                                code = int(um.group(1))
                                if code < 0:
                                    code += 65536
                                result.append(chr(code))
                            except Exception:
                                pass
                            j = i + um.end()

                    i = j
                    continue

            i += 1

        elif ch in ("{", "}"):
            # Group delimiters — skip
            i += 1

        elif ch in ("\r", "\n"):
            # RTF line breaks are formatting, not content
            i += 1

        else:
            result.append(ch)
            i += 1

    text_result = "".join(result)

    # Remove shape property names that leaked through
    text_result = _SHAPE_NAME_PATTERN.sub("", text_result)

    # Remove garbage negative numbers from shape properties
    text_result = re.sub(r"\s*-\d+\s*", " ", text_result)

    # Remove long hex data strings outside image tags
    text_result = _remove_hex_outside_image_tags(text_result)

    # Normalize whitespace
    text_result = re.sub(r"[ \t]+", " ", text_result)

    return text_result.strip()


def _remove_hex_outside_image_tags(text: str) -> str:
    """Remove long hex strings (32+ chars) that aren't inside image tags."""
    protected_ranges: List[Tuple[int, int]] = []
    for m in re.finditer(r"\[image:[^\]]+\]", text):
        protected_ranges.append((m.start(), m.end()))

    if not protected_ranges:
        return re.sub(r"(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])", "", text)

    result: List[str] = []
    last_end = 0
    for start, end in protected_ranges:
        before = text[last_end:start]
        before = re.sub(
            r"(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])", "", before
        )
        result.append(before)
        result.append(text[start:end])
        last_end = end

    after = text[last_end:]
    after = re.sub(r"(?<![a-zA-Z])[0-9a-fA-F]{32,}(?![a-zA-Z])", "", after)
    result.append(after)
    return "".join(result)


# ═══════════════════════════════════════════════════════════════════════════
# Destination Group Removal
# ═══════════════════════════════════════════════════════════════════════════

def remove_destination_groups(content: str) -> str:
    """
    Remove RTF destination groups.

    Handles both common forms:
    - ``{\\*\\destination ...}``  (with optional-destination marker)
    - ``{\\destination ...}``     (without marker)

    Destinations like fonttbl, colortbl, stylesheet, etc. contain
    metadata that should not appear in extracted text.

    Image destinations (pict, shppict) are handled specially —
    any ``[image:...]`` tags within them are preserved.

    Args:
        content: RTF content string.

    Returns:
        Content with destination groups removed.
    """
    result: List[str] = []
    i = 0
    n = len(content)

    while i < n:
        # Match opening brace followed by a control word
        if content[i] == "{" and i + 1 < n and content[i + 1] == "\\":
            # Check for {\*\word or {\word
            j = i + 2
            has_star = False
            if j < n and content[j] == "*":
                has_star = True
                j += 1
                # Skip whitespace and the next backslash
                while j < n and content[j] in " \t\r\n":
                    j += 1
                if j < n and content[j] == "\\":
                    j += 1
                else:
                    # Not a destination, keep as-is
                    result.append(content[i])
                    i += 1
                    continue
            # Now j points to the start of the control word name
            k = j
            while k < n and content[k].isalpha():
                k += 1
            ctrl_word = content[j:k]

            if ctrl_word in SKIP_DESTINATIONS:
                # Remove entire group (brace-balanced)
                depth = 1
                p = i + 1
                while p < n and depth > 0:
                    if content[p] == "{":
                        depth += 1
                    elif content[p] == "}":
                        depth -= 1
                    p += 1
                i = p
                continue

            if ctrl_word in IMAGE_DESTINATIONS:
                # Remove group but preserve [image:...] tags
                depth = 1
                group_start = i
                p = i + 1
                while p < n and depth > 0:
                    if content[p] == "{":
                        depth += 1
                    elif content[p] == "}":
                        depth -= 1
                    p += 1

                group_content = content[group_start:p]
                tag_match = re.search(r"\[image:[^\]]+\]", group_content)
                if tag_match:
                    tag = tag_match.group()
                    if "/uploads/." not in tag and "uploads/." not in tag:
                        result.append(tag)
                i = p
                continue

        result.append(content[i])
        i += 1

    return "".join(result)


def remove_shape_groups(content: str) -> str:
    """
    Remove shape groups but preserve text in ``\\shptxt``.

    RTF shapes are drawing objects that contain rendering instructions
    along with optional text content in ``{\\shptxt ...}`` blocks.
    We remove the shape scaffold but keep the text.

    Args:
        content: RTF content string.

    Returns:
        Content with shape groups cleaned, shptxt text preserved.
    """
    result: List[str] = []
    i = 0

    while i < len(content):
        if content[i : i + 5] == "{\\shp" or content[i : i + 10] == "{\\*\\shpinst":
            depth = 1
            i += 1
            shptxt_content: List[str] = []
            in_shptxt = False
            shptxt_depth = 0

            while i < len(content) and depth > 0:
                if content[i] == "{":
                    if content[i : i + 8] == "{\\shptxt":
                        in_shptxt = True
                        shptxt_depth = depth + 1
                        i += 8
                        continue
                    depth += 1
                elif content[i] == "}":
                    if in_shptxt and depth == shptxt_depth:
                        in_shptxt = False
                    depth -= 1
                elif in_shptxt:
                    shptxt_content.append(content[i])
                i += 1

            if shptxt_content:
                result.append("".join(shptxt_content))
        else:
            result.append(content[i])
            i += 1

    return "".join(result)


def remove_shape_property_groups(content: str) -> str:
    """
    Remove shape property groups ``{\\sp{\\sn xxx}{\\sv yyy}}``.

    Args:
        content: RTF content string.

    Returns:
        Content with shape properties removed.
    """
    content = re.sub(r"\{\\sp\{\\sn\s*[^}]*\}\{\\sv\s*[^}]*\}\}", "", content)
    content = re.sub(r"\{\\sp\s*[^}]*\}", "", content)
    content = re.sub(r"\{\\sn\s*[^}]*\}", "", content)
    content = re.sub(r"\{\\sv\s*[^}]*\}", "", content)
    return content


def remove_shprslt_blocks(content: str) -> str:
    """
    Remove ``\\shprslt{...}`` blocks.

    Word saves shapes in ``\\shp`` blocks and duplicates content in
    ``\\shprslt`` for backward compatibility with older viewers.
    Removing ``\\shprslt`` prevents text duplication.

    Args:
        content: RTF content string.

    Returns:
        Content with ``\\shprslt`` blocks removed.
    """
    result: List[str] = []
    i = 0
    pattern = "\\shprslt"

    while i < len(content):
        idx = content.find(pattern, i)
        if idx == -1:
            result.append(content[i:])
            break

        result.append(content[i:idx])

        brace_start = content.find("{", idx)
        if brace_start == -1:
            i = idx + len(pattern)
            continue

        depth = 1
        j = brace_start + 1
        while j < len(content) and depth > 0:
            if content[j] == "{":
                depth += 1
            elif content[j] == "}":
                depth -= 1
            j += 1

        i = j

    return "".join(result)


# ═══════════════════════════════════════════════════════════════════════════
# Region Finder (merged from rtf_region_finder.py)
# ═══════════════════════════════════════════════════════════════════════════

def find_excluded_regions(content: str) -> List[Tuple[int, int]]:
    """
    Find document regions to exclude from content extraction.

    Identifies header, footer, footnote, and annotation regions
    by scanning for their RTF control words and matching brace depth.

    Args:
        content: RTF content string.

    Returns:
        Sorted, merged list of ``(start, end)`` position tuples.
    """
    regions: List[Tuple[int, int]] = []

    # Header/footer/footnote start patterns
    start_patterns = [
        r"\\header[lrf]?\b",   # Headers (left/right/first)
        r"\\footer[lrf]?\b",   # Footers
        r"\\footnote\b",       # Footnotes
        r"\\annotation\b",     # Annotations/comments
        r"\{\\headerf",        # First-page header
        r"\{\\footerf",        # First-page footer
    ]

    for pattern in start_patterns:
        for match in re.finditer(pattern, content):
            start_pos = match.start()

            # Find the matching closing brace
            depth = 0
            i = start_pos
            found_start = False

            while i < len(content):
                if content[i] == "{":
                    if not found_start:
                        found_start = True
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if found_start and depth == 0:
                        regions.append((start_pos, i + 1))
                        break
                i += 1

    # Merge overlapping regions
    if regions:
        regions.sort(key=lambda x: x[0])
        merged = [regions[0]]
        for start, end in regions[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    return regions


def is_in_excluded_region(
    position: int,
    regions: List[Tuple[int, int]],
) -> bool:
    """
    Check if a position falls within an excluded region.

    Args:
        position: Character position to check.
        regions: List of ``(start, end)`` excluded region tuples.

    Returns:
        True if the position is inside an excluded region.
    """
    for start, end in regions:
        if start <= position < end:
            return True
    return False


__all__ = [
    "clean_rtf_text",
    "remove_destination_groups",
    "remove_shape_groups",
    "remove_shape_property_groups",
    "remove_shprslt_blocks",
    "find_excluded_regions",
    "is_in_excluded_region",
]
