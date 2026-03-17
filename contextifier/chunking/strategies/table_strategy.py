# contextifier_new/chunking/strategies/table_strategy.py
"""
TableChunkingStrategy — Table-aware chunking for spreadsheet-type content.

Handles CSV, TSV, XLSX, XLS files where content is primarily tabular.
Each table is chunked independently with header restoration.

Priority: 5 (highest — table files always use this strategy)

Algorithm:
1. Extract metadata block (if present) as a shared context prefix.
2. If sheets exist → process each sheet independently (multi-sheet path).
3. Otherwise treat the whole text as a single table (single-table path).
4. Tables are split by rows via ``table_chunker`` — NO overlap.
5. Non-table segments (chart, textbox, image, plain-text) within a sheet
   are kept intact or recursively split as needed.
"""

from __future__ import annotations

import re
from typing import Any, FrozenSet, List, Optional, Tuple, Union

from contextifier.config import ProcessingConfig
from contextifier.types import Chunk, ChunkMetadata
from contextifier.chunking.constants import (
    HTML_TABLE_PATTERN,
    MARKDOWN_TABLE_PATTERN,
    TEXTBOX_BLOCK_PATTERN,
    TABLE_EXTENSIONS,
)
from contextifier.chunking.table_chunker import chunk_large_table
from contextifier.chunking.strategies.base import BaseChunkingStrategy

from langchain_text_splitters import RecursiveCharacterTextSplitter


class TableChunkingStrategy(BaseChunkingStrategy):
    """
    Split table-based content preserving table structure.

    Key invariant: Tables NEVER have overlap between chunks.
    This prevents data duplication in search/retrieval systems.

    For each table:
    1. Parse into header rows + data rows
    2. Calculate available space per chunk (chunk_size − header_size)
    3. Accumulate data rows until space exceeded
    4. Each chunk gets header rows prepended
    5. ``[Table Chunk N/M]`` annotation added to each chunk
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
        """Handle table-based file types (CSV, TSV, XLSX, XLS)."""
        return file_extension.lower().lstrip(".") in TABLE_EXTENSIONS

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
        Chunk table-based content.

        Delegates to multi-sheet or single-table path depending on the
        presence of ``[Sheet: …]`` markers.
        """
        chunk_size = config.chunking.chunk_size
        chunk_overlap = config.chunking.chunk_overlap
        tags = config.tags

        # ── 1. Extract metadata block ─────────────────────────────────────
        metadata_block, body = self._extract_metadata_block(text, tags)

        # ── 2. Detect sheets ──────────────────────────────────────────────
        sheet_pattern = re.compile(
            rf"{re.escape(tags.sheet_prefix)}\s*([^\]]+){re.escape(tags.sheet_suffix)}"
        )
        sheets = self._extract_sheets(body, sheet_pattern, tags)

        # Build context prefix (metadata shared across all chunks)
        context_prefix = metadata_block.strip() if metadata_block else ""

        if sheets:
            raw = self._chunk_multi_sheet(
                sheets, context_prefix, chunk_size, chunk_overlap, config,
            )
        else:
            raw = self._chunk_single_table(
                body, context_prefix, chunk_size, chunk_overlap, config,
            )

        if not raw:
            raw = [text] if text.strip() else [""]

        if not include_position_metadata:
            return raw

        return self._attach_metadata(raw)

    @property
    def strategy_name(self) -> str:
        return "table"

    @property
    def priority(self) -> int:
        return 5

    # ── Multi-sheet path ──────────────────────────────────────────────────

    def _chunk_multi_sheet(
        self,
        sheets: List[Tuple[str, str]],
        common_prefix: str,
        chunk_size: int,
        chunk_overlap: int,
        config: ProcessingConfig,
    ) -> List[str]:
        """Process each sheet independently, returning all chunks."""
        all_chunks: List[str] = []

        for sheet_name, sheet_content in sheets:
            # Build per-sheet prefix (common metadata + sheet marker)
            tags = config.tags
            sheet_marker = f"{tags.sheet_prefix}{sheet_name}{tags.sheet_suffix}"
            prefix_parts = [p for p in (common_prefix, sheet_marker) if p]
            sheet_prefix = "\n\n".join(prefix_parts) if prefix_parts else ""

            # Remove the sheet marker from content body
            pattern = re.compile(
                rf"{re.escape(tags.sheet_prefix)}\s*{re.escape(sheet_name)}\s*{re.escape(tags.sheet_suffix)}"
            )
            body = pattern.sub("", sheet_content, count=1).strip()

            # Split into segments: table, textbox, chart, image, text
            segments = self._extract_segments(body, config)

            for seg_type, seg_content in segments:
                if not seg_content.strip():
                    continue

                if seg_type == "table":
                    if len(sheet_prefix) + len(seg_content) <= chunk_size:
                        all_chunks.append(f"{sheet_prefix}\n{seg_content}".strip())
                    else:
                        table_chunks = chunk_large_table(
                            seg_content, chunk_size, sheet_prefix,
                        )
                        all_chunks.extend(table_chunks)

                elif seg_type in ("textbox", "chart", "image"):
                    # Protected: never split
                    all_chunks.append(f"{sheet_prefix}\n{seg_content}".strip())

                else:
                    # Plain text
                    if len(sheet_prefix) + len(seg_content) <= chunk_size:
                        all_chunks.append(f"{sheet_prefix}\n{seg_content}".strip())
                    else:
                        text_chunks = self._split_plain(seg_content, chunk_size, chunk_overlap)
                        for tc in text_chunks:
                            all_chunks.append(f"{sheet_prefix}\n{tc}".strip())

        return all_chunks

    # ── Single-table path ─────────────────────────────────────────────────

    def _chunk_single_table(
        self,
        text: str,
        context_prefix: str,
        chunk_size: int,
        chunk_overlap: int,
        config: ProcessingConfig,
    ) -> List[str]:
        """Handle content without sheet markers (single table/CSV)."""
        # Find tables (HTML & Markdown)
        tables = self._find_tables(text)

        if not tables:
            # No tables found → plain split with context
            full = f"{context_prefix}\n\n{text}".strip() if context_prefix else text
            return self._split_plain(full, chunk_size, chunk_overlap)

        all_chunks: List[str] = []
        for _, _, table_content in tables:
            total = len(context_prefix) + len(table_content)
            if total <= chunk_size:
                chunk = f"{context_prefix}\n\n{table_content}".strip() if context_prefix else table_content
                all_chunks.append(chunk)
            else:
                table_chunks = chunk_large_table(
                    table_content, chunk_size, context_prefix,
                )
                all_chunks.extend(table_chunks)

        return all_chunks

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_metadata_block(
        text: str, tags: Any,
    ) -> Tuple[Optional[str], str]:
        """Extract ``<Document-Metadata>…</Document-Metadata>`` block."""
        pattern = re.compile(
            rf"{re.escape(tags.metadata_prefix)}.*?{re.escape(tags.metadata_suffix)}\s*",
            re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            return m.group(0).strip(), (text[: m.start()] + text[m.end() :]).strip()
        return None, text

    @staticmethod
    def _extract_sheets(
        text: str,
        pattern: re.Pattern,
        tags: Any,
    ) -> List[Tuple[str, str]]:
        """Split text by ``[Sheet: …]`` markers."""
        matches = list(pattern.finditer(text))
        if not matches:
            return []

        sheets: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            name = m.group(1).strip()
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                sheets.append((name, content))
        return sheets

    @staticmethod
    def _extract_segments(
        content: str, config: ProcessingConfig,
    ) -> List[Tuple[str, str]]:
        """
        Extract typed segments (table / textbox / chart / image / text)
        from a sheet body.
        """
        tags = config.tags
        img_pattern = re.compile(
            rf"{re.escape(tags.image_prefix)}.+?{re.escape(tags.image_suffix)}"
        )
        chart_pattern = re.compile(
            rf"{re.escape(tags.chart_prefix)}.*?{re.escape(tags.chart_suffix)}",
            re.DOTALL | re.IGNORECASE,
        )

        patterns: List[Tuple[str, re.Pattern]] = [
            ("table", re.compile(
                r"(?:\[Table\s*\d+\]\s*)?<table[^>]*>.*?</table>",
                re.DOTALL | re.IGNORECASE,
            )),
            ("table", MARKDOWN_TABLE_PATTERN),
            ("textbox", TEXTBOX_BLOCK_PATTERN),
            ("chart", chart_pattern),
            ("image", img_pattern),
        ]

        all_matches: List[Tuple[int, int, str, str]] = []
        for seg_type, pat in patterns:
            for m in pat.finditer(content):
                matched = m.group(0).strip()
                if matched:
                    all_matches.append((m.start(), m.end(), seg_type, matched))

        all_matches.sort(key=lambda x: x[0])

        # De-overlap (first match wins)
        filtered: List[Tuple[int, int, str, str]] = []
        last_end = 0
        for s, e, t, c in all_matches:
            if s >= last_end:
                filtered.append((s, e, t, c))
                last_end = e

        segments: List[Tuple[str, str]] = []
        pos = 0
        for s, e, t, c in filtered:
            if s > pos:
                between = content[pos:s].strip()
                if between:
                    segments.append(("text", between))
            segments.append((t, c))
            pos = e

        if pos < len(content):
            tail = content[pos:].strip()
            if tail:
                segments.append(("text", tail))

        return segments

    @staticmethod
    def _find_tables(text: str) -> List[Tuple[int, int, str]]:
        """Find non-overlapping HTML + Markdown tables."""
        all_matches: List[Tuple[int, int, str]] = []

        for m in HTML_TABLE_PATTERN.finditer(text):
            all_matches.append((m.start(), m.end(), m.group(0)))

        for m in MARKDOWN_TABLE_PATTERN.finditer(text):
            s = m.start()
            if m.group(0).startswith("\n"):
                s += 1
            all_matches.append((s, m.end(), m.group(0).strip()))

        all_matches.sort(key=lambda x: x[0])

        filtered: List[Tuple[int, int, str]] = []
        last_end = 0
        for s, e, c in all_matches:
            if s >= last_end:
                filtered.append((s, e, c))
                last_end = e
        return filtered

    @staticmethod
    def _split_plain(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursive character splitting for non-table text."""
        if not text or not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    @staticmethod
    def _attach_metadata(chunks: List[str]) -> List[Chunk]:
        """Wrap plain strings into Chunk objects with positional metadata."""
        result: List[Chunk] = []
        offset = 0
        for idx, text in enumerate(chunks):
            meta = ChunkMetadata(
                chunk_index=idx,
                global_start=offset,
                global_end=offset + len(text),
            )
            result.append(Chunk(text=text, metadata=meta))
            offset += len(text)
        return result


__all__ = ["TableChunkingStrategy"]
