# contextifier/handlers/text/content_extractor.py
"""
TextContentExtractor — Stage 4: Extract text content

Takes the PreprocessedData (normalized text) and applies final cleaning
based on whether the file is code or plain text.

Code mode:
  - Preserves indentation and structure
  - Replaces tabs with 4 spaces
  - Strips only trailing whitespace

Text mode:
  - Collapses 3+ consecutive blank lines into 2
  - Strips leading and trailing whitespace

Code mode is auto-detected from file_category (stored in
PreprocessedData.properties by TextPreprocessor), or can be
overridden explicitly via ``is_code=True`` kwarg.

Text files have no tables, images, or charts, so the base class
default empty lists are used for those.

v1.0 Issues resolved:
- Text cleaning was inline in handler's extract_text(), not in a pipeline stage
- is_code was a handler parameter, not derived from file metadata
- No formal ContentExtractor existed for text files
"""

from __future__ import annotations

import re
from typing import Any, FrozenSet

from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import PreprocessedData


# File categories where content should be treated as code
# (preserve indentation, replace tabs, don't collapse blank lines)
_CODE_CATEGORIES: FrozenSet[str] = frozenset({
    "code",     # .py, .js, .ts, .java, .cpp, ...
    "config",   # .json, .yaml, .xml, .toml, .ini, ...
    "script",   # .sh, .bat, .ps1, ...
    "web",      # .htm, .xhtml
})


class TextContentExtractor(BaseContentExtractor):
    """
    Content extractor for plain text and source code files.

    No tables, images, or charts — only text extraction with
    code-aware cleaning.
    """

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Extract and clean text content.

        Code mode resolution (in priority order):
        1. Explicit ``is_code`` kwarg (True/False)
        2. Auto-detection from ``file_category`` in preprocessed.properties

        Args:
            preprocessed: Output from TextPreprocessor.
            **kwargs: Optional ``is_code`` boolean.

        Returns:
            Cleaned text string.
        """
        text: str = preprocessed.content or ""

        if not text:
            return ""

        # JSON payloads (API responses, exports) render structure-aware:
        # one line per leaf path so downstream chunking respects object
        # boundaries instead of splitting mid-structure. Non-JSON content
        # in a .json file falls through to plain handling.
        if str(preprocessed.properties.get("file_extension", "")).lower() == "json":
            rendered = _render_json_context(text)
            if rendered is not None:
                return rendered

        # Determine code mode
        is_code = kwargs.get("is_code", None)
        if is_code is None:
            file_category = preprocessed.properties.get("file_category", "")
            is_code = file_category in _CODE_CATEGORIES

        if is_code:
            return _clean_code_text(text)
        return _clean_text(text)

    def get_format_name(self) -> str:
        return "text"


# ── Text cleaning functions ───────────────────────────────────────────────
# Extracted as module-level functions for testability and reuse.


def _clean_text(text: str) -> str:
    """
    Clean plain text content.

    - Collapses 3+ consecutive blank lines into 2 (one empty line)
    - Strips leading and trailing whitespace from the entire text

    Matches v1.0 ``clean_text()`` from ``utils.py``.
    """
    if not text:
        return ""
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _clean_code_text(text: str) -> str:
    """
    Clean source code content.

    - Preserves indentation and structure
    - Replaces tabs with 4 spaces (standard Python convention)
    - Strips only trailing whitespace from the text

    Matches v1.0 ``clean_code_text()`` from ``utils.py``.
    """
    if not text:
        return ""
    text = text.rstrip().replace("\t", "    ")
    return text


__all__ = ["TextContentExtractor"]

# ── JSON context rendering ────────────────────────────────────────────

_JSON_MAX_ITEMS = 20_000  # runaway-payload backstop


def _render_json_context(text: str) -> "str | None":
    """Render a JSON document as structure-aware context text.

    Objects/arrays become ``dotted.path[i]: value`` lines grouped in
    document order; arrays of objects get a blank-line boundary per
    element so chunkers keep records intact. Returns ``None`` when the
    content is not valid JSON (caller falls back to plain text).
    """
    import json as _json

    try:
        data = _json.loads(text)
    except (ValueError, TypeError):
        return None

    lines: list = []
    count = 0

    def _walk(node, path):
        nonlocal count
        if count >= _JSON_MAX_ITEMS:
            return
        if isinstance(node, dict):
            if not node:
                lines.append(f"{path or '$'}: {{}}")
                count += 1
                return
            for key, value in node.items():
                _walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            if not node:
                lines.append(f"{path or '$'}: []")
                count += 1
                return
            for i, value in enumerate(node):
                if isinstance(value, (dict, list)) and lines:
                    lines.append("")  # record boundary for chunkers
                _walk(value, f"{path}[{i}]")
        else:
            rendered = _json.dumps(node, ensure_ascii=False)
            lines.append(f"{path or '$'}: {rendered}")
            count += 1

    _walk(data, "")
    if count >= _JSON_MAX_ITEMS:
        lines.append(f"[JSON truncated at {_JSON_MAX_ITEMS} values]")
    return "\n".join(lines).strip() + "\n"
