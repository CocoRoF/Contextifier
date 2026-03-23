# contextifier/chunking/strategies/page_strategy.py
"""
PageChunkingStrategy — Split by page/slide markers.

Handles documents with ``[Page Number: N]`` or ``[Slide Number: N]``
tags.  Merges pages greedily until ``chunk_size`` is reached, then
starts a new chunk.

Priority: 10 (high — page boundaries are the best split points for
formatted documents)

Algorithm:
1. Split text at page/slide markers into page segments.
2. Greedily merge adjacent pages until ``chunk_size`` is exceeded.
3. Allow up to ``1.5 × chunk_size`` if a page boundary falls in that
   range (prefer keeping pages together).
4. Very large single pages are sub-split using
   ``ProtectedChunkingStrategy`` to respect embedded tables/charts.
5. Overlap content is carried from the end of the previous chunk and
   prepended to the next chunk's first page.
"""

from __future__ import annotations

import re
import logging
from typing import Any, List, Optional, Tuple, Union

from contextifier.config import ProcessingConfig
from contextifier.types import Chunk, ChunkMetadata
from contextifier.chunking.strategies.base import BaseChunkingStrategy

logger = logging.getLogger("contextifier.chunking.page_strategy")

# Pattern to detect OCR-tagged page/slide markers with a capture group for the number
_MARKER_NUMBER_RE = r"\d+(?:\s*\(OCR(?:\+Ref)?\))?"

# Maximum page-merge budget factor
_MAX_SIZE_FACTOR: float = 1.5


class PageChunkingStrategy(BaseChunkingStrategy):
    """
    Split text at page/slide boundaries.

    Merge Strategy:
    - ``size ≤ chunk_size``  → keep merging.
    - ``chunk_size < size ≤ 1.5 × chunk_size`` → accept and finalize.
    - ``size > 1.5 × chunk_size`` → finalize **previous** pages and
      start a new chunk with the current page.
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
        """Return True if *text* contains page or slide markers."""
        return config.tags.page_prefix in text or config.tags.slide_prefix in text

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
        Chunk *text* by page boundaries with greedy merging.
        """
        chunk_size = config.chunking.chunk_size
        chunk_overlap = config.chunking.chunk_overlap

        # Build page/slide marker patterns from config
        marker_patterns = self._build_marker_patterns(config)

        # 1. Split into pages using the first matching pattern
        pages = self._split_into_pages(text, marker_patterns)

        if not pages:
            # No page markers found → fall back to plain splitting
            from contextifier.chunking.strategies.plain_strategy import PlainChunkingStrategy
            return PlainChunkingStrategy().chunk(
                text, config,
                file_extension=file_extension,
                include_position_metadata=include_position_metadata,
                **context,
            )

        self._logger.debug(f"Split into {len(pages)} pages")

        # 2. Greedy page merging
        max_size = int(chunk_size * _MAX_SIZE_FACTOR)
        raw = self._merge_pages(
            pages, chunk_size, max_size, chunk_overlap, config,
        )

        if not raw:
            raw = [text] if text.strip() else [""]

        if not include_position_metadata:
            return raw
        return self._attach_metadata(raw)

    @property
    def strategy_name(self) -> str:
        return "page"

    @property
    def priority(self) -> int:
        return 10

    # ══════════════════════════════════════════════════════════════════════
    # Page splitting
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_marker_patterns(config: ProcessingConfig) -> List[re.Pattern]:
        """Build regex patterns for page/slide markers from config."""
        tags = config.tags
        patterns: List[re.Pattern] = []

        # Page pattern: "[Page Number: 1]" (with optional OCR suffix)
        page_pat = re.compile(
            rf"{re.escape(tags.page_prefix)}({_MARKER_NUMBER_RE}){re.escape(tags.page_suffix)}"
        )
        patterns.append(page_pat)

        # Slide pattern: "[Slide Number: 1]"
        if tags.slide_prefix != tags.page_prefix:
            slide_pat = re.compile(
                rf"{re.escape(tags.slide_prefix)}({_MARKER_NUMBER_RE}){re.escape(tags.slide_suffix)}"
            )
            patterns.append(slide_pat)

        return patterns

    @staticmethod
    def _split_into_pages(
        text: str, patterns: List[re.Pattern],
    ) -> List[Tuple[int, str]]:
        """
        Split *text* by the first matching marker pattern.

        Returns:
            ``[(page_number, page_content), ...]``
            where *page_content* includes the marker line.
        """
        for pattern in patterns:
            markers = list(pattern.finditer(text))
            if not markers:
                continue

            pages: List[Tuple[int, str]] = []

            # Content before first marker
            if markers[0].start() > 0:
                before = text[: markers[0].start()].strip()
                if before:
                    pages.append((0, before))

            for i, m in enumerate(markers):
                raw_num = m.group(1)
                # Extract leading digits from e.g. "3 (OCR)"
                num_match = re.match(r"\d+", raw_num)
                page_num = int(num_match.group()) if num_match else i + 1

                start = m.start()
                end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
                page_content = text[start:end].strip()

                if not page_content:
                    continue

                # Skip marker-only pages
                content_sans_marker = pattern.sub("", page_content).strip()
                if content_sans_marker:
                    pages.append((page_num, page_content))

            if pages:
                return pages

        return []

    # ══════════════════════════════════════════════════════════════════════
    # Greedy merging
    # ══════════════════════════════════════════════════════════════════════

    def _merge_pages(
        self,
        pages: List[Tuple[int, str]],
        chunk_size: int,
        max_size: int,
        chunk_overlap: int,
        config: ProcessingConfig,
    ) -> List[str]:
        """Merge pages greedily into chunks."""
        chunks: List[str] = []
        current_pages: List[Tuple[int, str]] = []
        current_size = 0
        pending_overlap = ""

        for page_num, page_content in pages:
            # Prepend pending overlap
            if pending_overlap:
                page_content = pending_overlap + "\n\n" + page_content
                pending_overlap = ""

            page_size = len(page_content)

            if not current_pages:
                current_pages.append((page_num, page_content))
                current_size = page_size
                continue

            potential = current_size + 4 + page_size  # +4 for "\n\n"

            if potential <= chunk_size:
                # Within budget — keep merging
                current_pages.append((page_num, page_content))
                current_size = potential

            elif potential <= max_size:
                # Within 1.5× — merge and finalize
                current_pages.append((page_num, page_content))
                chunk_text = self._merge_page_texts(current_pages)
                chunks.append(chunk_text)

                pending_overlap = self._get_overlap(current_pages, chunk_overlap)
                current_pages = []
                current_size = 0

            else:
                # Over 1.5× — finalize current, start new
                if current_pages:
                    chunk_text = self._merge_page_texts(current_pages)
                    chunks.append(chunk_text)

                current_pages = [(page_num, page_content)]
                current_size = page_size

        # Flush remaining
        if current_pages:
            chunk_text = self._merge_page_texts(current_pages)
            chunks.append(chunk_text)

        # Sub-split very large chunks
        final: List[str] = []
        extreme = int(max_size * 1.5)
        for chunk in chunks:
            if len(chunk) > extreme:
                sub = self._sub_split_large_chunk(chunk, chunk_size, chunk_overlap, config)
                final.extend(sub)
            else:
                final.append(chunk)

        return final

    @staticmethod
    def _merge_page_texts(pages: List[Tuple[int, str]]) -> str:
        return "\n\n".join(content for _, content in pages)

    @staticmethod
    def _get_overlap(pages: List[Tuple[int, str]], overlap_size: int) -> str:
        """Extract overlap content from the tail of the last page."""
        if not pages or overlap_size <= 0:
            return ""
        _, last = pages[-1]
        if len(last) <= overlap_size:
            return last
        return last[-overlap_size:]

    @staticmethod
    def _sub_split_large_chunk(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        config: ProcessingConfig,
    ) -> List[str]:
        """
        Re-split an oversized merged chunk using ProtectedChunkingStrategy
        so embedded tables/charts remain intact.
        """
        from contextifier.chunking.strategies.protected_strategy import ProtectedChunkingStrategy
        strategy = ProtectedChunkingStrategy()
        if strategy.can_handle(text, config):
            result = strategy.chunk(text, config, include_position_metadata=False)
            if isinstance(result, list) and result and isinstance(result[0], str):
                return result  # type: ignore[return-value]
        # Fallback to plain recursive split
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    # ── Metadata ──────────────────────────────────────────────────────────

    @staticmethod
    def _attach_metadata(chunks: List[str]) -> List[Chunk]:
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


__all__ = ["PageChunkingStrategy"]
