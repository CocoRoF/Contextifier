# contextifier_new/chunking/strategies/plain_strategy.py
"""
PlainChunkingStrategy — Simple recursive text splitting.

Fallback strategy for text without structural markers.
Uses RecursiveCharacterTextSplitter with standard separators.

Priority: 100 (lowest — fallback)

Algorithm:
1. Detect code files → use language-specific splitter
2. Separate ```html code fences as atomic blocks
3. Split remaining text recursively by paragraph → newline → space → char
4. Clean out empty / marker-only chunks
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter

from contextifier.config import ProcessingConfig
from contextifier.types import Chunk, ChunkMetadata
from contextifier.chunking.constants import CODE_LANGUAGE_MAP
from contextifier.chunking.strategies.base import BaseChunkingStrategy


# ── Pre-compiled patterns ─────────────────────────────────────────────────────
_HTML_CODE_FENCE_RE = re.compile(r"```html\s*(.*?)\s*```", re.DOTALL)


class PlainChunkingStrategy(BaseChunkingStrategy):
    """
    Simple text splitting without structural awareness.

    Separators (in order of preference):
    1. Double newline (paragraph breaks)
    2. Single newline
    3. Space
    4. Empty string (character-level)

    Always applies as the fallback when no other strategy matches.
    """

    # ── Strategy contract ─────────────────────────────────────────────────

    def can_handle(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        **context: Any,
    ) -> bool:
        """Always returns True — this is the fallback strategy."""
        return True

    def chunk(
        self,
        text: str,
        config: ProcessingConfig,
        *,
        file_extension: str = "",
        include_position_metadata: bool = False,
        **context: Any,
    ) -> Union[List[str], List[Chunk]]:
        """
        Split *text* using recursive character splitting.

        - Code files (py, js, ts, …) use language-aware splitters.
        - Other text uses paragraph → newline → space → char hierarchy.
        - HTML fenced code blocks (triple-backtick html) are preserved as atomic chunks.
        """
        chunk_size = config.chunking.chunk_size
        chunk_overlap = config.chunking.chunk_overlap
        ext = file_extension.lower().lstrip(".")

        # ── Code-language splitting ───────────────────────────────────────
        lang = CODE_LANGUAGE_MAP.get(ext)
        if lang:
            raw = self._split_code(text, lang, chunk_size, chunk_overlap)
        else:
            raw = self._split_plain(text, chunk_size, chunk_overlap)

        # Clean out empty chunks and page-marker-only chunks
        cleaned = self._clean_chunks(raw, config)

        if not cleaned:
            cleaned = [text] if text.strip() else [""]

        if not include_position_metadata:
            return cleaned

        return self._attach_metadata(cleaned)

    @property
    def strategy_name(self) -> str:
        return "plain"

    @property
    def priority(self) -> int:
        return 100

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _split_code(
        text: str, lang: str, chunk_size: int, chunk_overlap: int,
    ) -> List[str]:
        """Use langchain's language-aware splitter."""
        try:
            from langchain_text_splitters import Language
            lang_enum = getattr(Language, lang, None)
            if lang_enum is not None:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                return splitter.split_text(text)
        except Exception:
            pass  # Fall through to plain splitting

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    @staticmethod
    def _split_plain(
        text: str, chunk_size: int, chunk_overlap: int,
    ) -> List[str]:
        """
        Paragraph-aware recursive splitting.

        HTML fenced code blocks are kept intact per the old behaviour.
        """
        matches = list(_HTML_CODE_FENCE_RE.finditer(text))

        if not matches:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
            return splitter.split_text(text)

        # Separate HTML code fences from surrounding text
        segments: List[tuple[str, str]] = []  # (kind, content)
        pos = 0
        for m in matches:
            s, e = m.span()
            before = text[pos:s].strip()
            if before:
                segments.append(("text", before))
            segments.append(("html", text[s:e]))
            pos = e
        tail = text[pos:].strip()
        if tail:
            segments.append(("text", tail))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        result: List[str] = []
        for kind, content in segments:
            if kind == "html":
                result.append(content)
            else:
                result.extend(splitter.split_text(content))

        return result

    @staticmethod
    def _clean_chunks(chunks: List[str], config: ProcessingConfig) -> List[str]:
        """Remove empty chunks and page-marker-only chunks."""
        tags = config.tags
        page_pat = re.compile(
            rf"{re.escape(tags.page_prefix)}\d+(?:\s*\(OCR(?:\+Ref)?\))?{re.escape(tags.page_suffix)}"
        )
        slide_pat = re.compile(
            rf"{re.escape(tags.slide_prefix)}\d+(?:\s*\(OCR\))?{re.escape(tags.slide_suffix)}"
        )

        result: List[str] = []
        for chunk in chunks:
            stripped = chunk.strip()
            if not stripped:
                continue
            if page_pat.fullmatch(stripped) or slide_pat.fullmatch(stripped):
                continue
            result.append(chunk)
        return result

    @staticmethod
    def _attach_metadata(chunks: List[str]) -> List[Chunk]:
        """Wrap plain strings into Chunk objects with positional metadata."""
        offset = 0
        result: List[Chunk] = []
        for idx, text in enumerate(chunks):
            meta = ChunkMetadata(
                chunk_index=idx,
                global_start=offset,
                global_end=offset + len(text),
            )
            result.append(Chunk(text=text, metadata=meta))
            offset += len(text)
        return result


__all__ = ["PlainChunkingStrategy"]
