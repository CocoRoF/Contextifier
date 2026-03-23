# contextifier_new/chunking/strategies/protected_strategy.py
"""
ProtectedChunkingStrategy — Region-aware chunking.

Handles text with protected regions (HTML tables, chart blocks,
image tags, metadata blocks) that must not be split.

Priority: 20 (after page strategy, before plain)

Algorithm overview:
1. Detect all protected regions and record their type + boundaries.
2. Walk forward through the text in ``chunk_size`` steps.
3. If a split point falls inside a protected region → shift to the
   region boundary (before-start or after-end, whichever keeps the
   chunk within budget).
4. Large tables that exceed ``chunk_size`` **and** ``force_chunking``
   is enabled → delegate to ``table_chunker`` for row-level splitting.
5. Overlap is applied ONLY to plain-text boundaries.
   Protected regions (image tags, page/slide tags, chart blocks,
   metadata blocks, tables) NEVER participate in overlap.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from contextifier.config import ProcessingConfig
from contextifier.types import Chunk, ChunkMetadata
from contextifier.chunking.constants import (
    HTML_TABLE_PATTERN,
    MARKDOWN_TABLE_PATTERN,
    TEXTBOX_BLOCK_PATTERN,
    build_protected_patterns,
    TABLE_SIZE_THRESHOLD_MULTIPLIER,
)
from contextifier.chunking.table_chunker import chunk_large_table
from contextifier.chunking.table_parser import is_markdown_table
from contextifier.chunking.strategies.base import BaseChunkingStrategy

logger = logging.getLogger("contextifier.chunking.protected_strategy")


# ═══════════════════════════════════════════════════════════════════════════════
# Internal types
# ═══════════════════════════════════════════════════════════════════════════════

# (start, end, region_type)
_Region = Tuple[int, int, str]


class ProtectedChunkingStrategy(BaseChunkingStrategy):
    """
    Split text while preserving protected regions.

    Protected region types (from highest to lowest precedence):
    - Chart blocks       ``[chart]…[/chart]``
    - Textbox blocks     ``[textbox]…[/textbox]``
    - HTML tables        ``<table>…</table>``
    - Metadata blocks    ``[Document-Metadata]…[/Document-Metadata]``
    - Image tags         ``[Image:…]``
    - Page / Slide / Sheet tags
    - Markdown tables    ``|…|``
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
        """Return True if *text* contains any protected region marker."""
        tags = config.tags
        markers = [
            "<table",
            tags.chart_prefix,
            "[textbox]",
            tags.image_prefix,
            tags.metadata_prefix,
        ]
        # Quick linear scan — no regex needed for detection
        return any(marker in text for marker in markers)

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
        Chunk *text* while respecting protected region boundaries.
        """
        chunk_size = config.chunking.chunk_size
        chunk_overlap = config.chunking.chunk_overlap
        preserve_tables = config.chunking.preserve_tables

        # force_chunking = NOT preserve_tables (when tables may be split by rows)
        force_chunking = not preserve_tables

        # 1. Detect regions
        regions = self._find_protected_regions(text, config, force_chunking)

        # 2. Extract position lists for special handling
        no_overlap_positions = self._extract_no_overlap_positions(text, config)
        image_positions = self._extract_image_positions(text, config)

        # 3. Detect table regions for force_chunking path
        html_table_positions: List[_Region] = []
        md_table_positions: List[_Region] = []
        if force_chunking:
            html_table_positions = self._find_html_tables(text, regions)
            md_table_positions = self._find_markdown_tables(text, regions)

        # Build lookup structures
        block_regions = [(s, e) for s, e, _ in regions]
        region_type_map: Dict[Tuple[int, int], str] = {(s, e): t for s, e, t in regions}

        # Merge force-chunking table regions with block regions
        all_typed: List[_Region] = list(regions)
        all_typed.extend(html_table_positions)
        all_typed.extend(md_table_positions)
        all_typed.sort(key=lambda x: x[0])

        # De-overlap
        merged_typed: List[_Region] = []
        for s, e, t in all_typed:
            if merged_typed and s < merged_typed[-1][1]:
                ps, pe, pt = merged_typed[-1]
                merged_typed[-1] = (ps, max(pe, e), f"{pt}+{t}")
            else:
                merged_typed.append((s, e, t))

        all_block = [(s, e) for s, e, _ in merged_typed]
        all_type_map: Dict[Tuple[int, int], str] = {(s, e): t for s, e, t in merged_typed}

        # 4. Walk and split
        raw = self._walk_and_split(
            text=text,
            block_regions=all_block,
            type_map=all_type_map,
            image_positions=image_positions,
            no_overlap_positions=no_overlap_positions,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_chunking=force_chunking,
        )

        if not raw:
            raw = [text] if text.strip() else [""]

        if not include_position_metadata:
            return raw
        return self._attach_metadata(raw)

    @property
    def strategy_name(self) -> str:
        return "protected"

    @property
    def priority(self) -> int:
        return 20

    # ══════════════════════════════════════════════════════════════════════
    # Region detection
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_protected_regions(
        text: str,
        config: ProcessingConfig,
        force_chunking: bool,
    ) -> List[_Region]:
        """
        Detect all protected regions in *text*.

        When *force_chunking* is True, HTML/Markdown tables are **not**
        registered as protected (they will be handled via row-level split
        later). Charts and other blocks remain fully protected regardless.
        """
        tags = config.tags
        regions: List[_Region] = []

        # ── Tables (only when fully protected) ────────────────────────────
        if not force_chunking:
            for m in HTML_TABLE_PATTERN.finditer(text):
                regions.append((m.start(), m.end(), "html_table"))
            for m in MARKDOWN_TABLE_PATTERN.finditer(text):
                s = m.start() + (1 if m.group(0).startswith("\n") else 0)
                regions.append((s, m.end(), "markdown_table"))

        # ── Always-protected blocks ───────────────────────────────────────
        chart_pat = re.compile(
            rf"{re.escape(tags.chart_prefix)}.*?{re.escape(tags.chart_suffix)}",
            re.DOTALL | re.IGNORECASE,
        )
        for m in chart_pat.finditer(text):
            regions.append((m.start(), m.end(), "chart"))

        for m in TEXTBOX_BLOCK_PATTERN.finditer(text):
            regions.append((m.start(), m.end(), "textbox"))

        img_pat = re.compile(
            rf"{re.escape(tags.image_prefix)}.+?{re.escape(tags.image_suffix)}"
        )
        for m in img_pat.finditer(text):
            regions.append((m.start(), m.end(), "image_tag"))

        # ── Page / Slide / Sheet tags ─────────────────────────────────────
        page_pat = re.compile(
            rf"{re.escape(tags.page_prefix)}\d+(?:\s*\(OCR(?:\+Ref)?\))?{re.escape(tags.page_suffix)}"
        )
        for m in page_pat.finditer(text):
            regions.append((m.start(), m.end(), "page_tag"))

        slide_pat = re.compile(
            rf"{re.escape(tags.slide_prefix)}\d+(?:\s*\(OCR\))?{re.escape(tags.slide_suffix)}"
        )
        for m in slide_pat.finditer(text):
            regions.append((m.start(), m.end(), "slide_tag"))

        sheet_pat = re.compile(
            rf"{re.escape(tags.sheet_prefix)}.+?{re.escape(tags.sheet_suffix)}"
        )
        for m in sheet_pat.finditer(text):
            regions.append((m.start(), m.end(), "sheet_tag"))

        # ── Metadata blocks ───────────────────────────────────────────────
        meta_pat = re.compile(
            rf"{re.escape(tags.metadata_prefix)}.*?{re.escape(tags.metadata_suffix)}",
            re.DOTALL,
        )
        for m in meta_pat.finditer(text):
            regions.append((m.start(), m.end(), "metadata"))

        # Sort + merge overlapping
        regions.sort(key=lambda x: x[0])
        merged: List[_Region] = []
        for s, e, t in regions:
            if merged and s < merged[-1][1]:
                ps, pe, pt = merged[-1]
                merged[-1] = (ps, max(pe, e), f"{pt}+{t}")
            else:
                merged.append((s, e, t))

        return merged

    # ── Auxiliary region finders ───────────────────────────────────────────

    @staticmethod
    def _find_html_tables(
        text: str, existing_regions: List[_Region],
    ) -> List[_Region]:
        """Find HTML tables not already covered by *existing_regions*."""
        existing_set = {(s, e) for s, e, _ in existing_regions}
        results: List[_Region] = []
        for m in HTML_TABLE_PATTERN.finditer(text):
            s, e = m.start(), m.end()
            if not any(es <= s and ee >= e for es, ee in existing_set):
                results.append((s, e, "html"))
        return results

    @staticmethod
    def _find_markdown_tables(
        text: str, existing_regions: List[_Region],
    ) -> List[_Region]:
        """Find Markdown tables not already covered by *existing_regions*."""
        existing_set = {(s, e) for s, e, _ in existing_regions}
        results: List[_Region] = []
        for m in MARKDOWN_TABLE_PATTERN.finditer(text):
            s = m.start() + (1 if m.group(0).startswith("\n") else 0)
            e = m.end()
            if not any(es <= s and ee >= e for es, ee in existing_set):
                results.append((s, e, "markdown"))
        return results

    @staticmethod
    def _extract_no_overlap_positions(
        text: str, config: ProcessingConfig,
    ) -> List[_Region]:
        """Page/slide/sheet/chart/metadata tag positions — never overlap."""
        tags = config.tags
        positions: List[_Region] = []

        for pat_str, rtype in [
            (rf"{re.escape(tags.page_prefix)}\d+(?:\s*\(OCR(?:\+Ref)?\))?{re.escape(tags.page_suffix)}", "page_tag"),
            (rf"{re.escape(tags.slide_prefix)}\d+(?:\s*\(OCR\))?{re.escape(tags.slide_suffix)}", "slide_tag"),
            (rf"{re.escape(tags.sheet_prefix)}.+?{re.escape(tags.sheet_suffix)}", "sheet_tag"),
        ]:
            for m in re.finditer(pat_str, text):
                positions.append((m.start(), m.end(), rtype))

        chart_pat = re.compile(
            rf"{re.escape(tags.chart_prefix)}.*?{re.escape(tags.chart_suffix)}",
            re.DOTALL,
        )
        for m in chart_pat.finditer(text):
            positions.append((m.start(), m.end(), "chart"))

        meta_pat = re.compile(
            rf"{re.escape(tags.metadata_prefix)}.*?{re.escape(tags.metadata_suffix)}",
            re.DOTALL,
        )
        for m in meta_pat.finditer(text):
            positions.append((m.start(), m.end(), "metadata"))

        return positions

    @staticmethod
    def _extract_image_positions(
        text: str, config: ProcessingConfig,
    ) -> List[Tuple[int, int]]:
        """Image tag positions — never split mid-tag, never overlap."""
        tags = config.tags
        pat = re.compile(
            rf"{re.escape(tags.image_prefix)}.+?{re.escape(tags.image_suffix)}"
        )
        return [(m.start(), m.end()) for m in pat.finditer(text)]

    # ══════════════════════════════════════════════════════════════════════
    # Core walking algorithm
    # ══════════════════════════════════════════════════════════════════════

    def _walk_and_split(
        self,
        text: str,
        block_regions: List[Tuple[int, int]],
        type_map: Dict[Tuple[int, int], str],
        image_positions: List[Tuple[int, int]],
        no_overlap_positions: List[_Region],
        chunk_size: int,
        chunk_overlap: int,
        force_chunking: bool,
    ) -> List[str]:
        """
        Walk through *text*, emitting chunks while respecting regions.

        Core invariants:
        - Protected regions are never cut.
        - Large tables with *force_chunking* are split by rows.
        - Overlap is applied ONLY at plain-text boundaries.
        """
        chunks: List[str] = []
        pos = 0
        text_len = len(text)

        while pos < text_len:
            remaining = text_len - pos
            if remaining <= chunk_size:
                chunk = text[pos:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            tentative_end = pos + chunk_size

            # ── Check for block region in range ───────────────────────────
            block = self._find_block_in_range(pos, tentative_end, block_regions)

            if block is not None:
                b_start, b_end = block
                b_type = type_map.get(block, "block")
                block_size = b_end - b_start

                if b_start <= pos:
                    # We're at / inside the block
                    if block_size > chunk_size:
                        block_content = text[b_start:b_end].strip()
                        if force_chunking and self._is_table_type(b_type, block_content):
                            table_chunks = chunk_large_table(block_content, chunk_size)
                            chunks.extend(table_chunks)
                        else:
                            if block_content:
                                chunks.append(block_content)
                        pos = b_end  # No overlap for blocks
                    else:
                        # Block fits → try to extend past it
                        end_pos = min(b_end + (chunk_size - block_size), text_len)
                        end_pos = self._clamp_to_next_block(end_pos, b_end, block_regions)
                        end_pos = self._adjust_image_boundary(end_pos, image_positions, text_len)
                        chunk = text[pos:end_pos].strip()
                        if chunk:
                            chunks.append(chunk)
                        if self._ends_with_no_overlap(end_pos, no_overlap_positions, image_positions):
                            pos = end_pos
                        else:
                            pos = max(b_end, end_pos - chunk_overlap)
                else:
                    # Block is ahead
                    space_before = b_start - pos
                    space_with = b_end - pos

                    if space_with <= chunk_size:
                        # Include entire block in this chunk
                        end_pos = b_end
                        extra = chunk_size - space_with
                        if extra > 0:
                            end_pos = min(b_end + extra, text_len)
                            end_pos = self._clamp_to_next_block(end_pos, b_end, block_regions)
                        end_pos = self._adjust_image_boundary(end_pos, image_positions, text_len)
                        chunk = text[pos:end_pos].strip()
                        if chunk:
                            chunks.append(chunk)
                        if self._ends_with_no_overlap(end_pos, no_overlap_positions, image_positions):
                            pos = end_pos
                        else:
                            pos = max(b_end, end_pos - chunk_overlap)
                    else:
                        # Block doesn't fit
                        if space_before > chunk_overlap:
                            # Cut before the block
                            end_pos = b_start
                            end_pos = self._adjust_image_boundary(end_pos, image_positions, text_len)
                            chunk = text[pos:end_pos].strip()
                            if chunk:
                                chunks.append(chunk)
                            if self._ends_with_no_overlap(end_pos, no_overlap_positions, image_positions):
                                pos = end_pos
                            else:
                                pos = max(pos + 1, b_start - chunk_overlap)
                        else:
                            # Block is too close → emit block directly
                            block_content = text[b_start:b_end].strip()
                            if block_size > chunk_size and force_chunking and self._is_table_type(b_type, block_content):
                                table_chunks = chunk_large_table(block_content, chunk_size)
                                chunks.extend(table_chunks)
                            else:
                                if block_content:
                                    chunks.append(block_content)
                            pos = b_end  # No overlap for blocks
            else:
                # ── No block in range → find best natural split ───────────
                best_split = self._find_natural_split(text, pos, tentative_end, chunk_size)
                best_split = self._adjust_image_boundary(best_split, image_positions, text_len)

                chunk = text[pos:best_split].strip()
                if chunk:
                    chunks.append(chunk)

                if self._ends_with_no_overlap(best_split, no_overlap_positions, image_positions):
                    pos = best_split
                else:
                    pos = max(pos + 1, best_split - chunk_overlap)

        return chunks

    # ══════════════════════════════════════════════════════════════════════
    # Walking helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _find_block_in_range(
        pos: int, end: int, blocks: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Return the first block region overlapping ``[pos, end)``."""
        for b_start, b_end in blocks:
            if b_start < end and b_end > pos:
                return (b_start, b_end)
        return None

    @staticmethod
    def _clamp_to_next_block(
        end_pos: int, after: int, blocks: List[Tuple[int, int]],
    ) -> int:
        """Ensure *end_pos* doesn't reach into the next block."""
        for b_start, _ in blocks:
            if b_start > after and b_start < end_pos:
                return b_start
        return end_pos

    @staticmethod
    def _adjust_image_boundary(
        pos: int,
        image_positions: List[Tuple[int, int]],
        text_len: int,
    ) -> int:
        """If *pos* falls inside an image tag, extend to its end."""
        for img_s, img_e in image_positions:
            if img_s < pos < img_e:
                return min(img_e, text_len)
        return pos

    @staticmethod
    def _ends_with_no_overlap(
        end_pos: int,
        no_overlap: List[_Region],
        image_positions: List[Tuple[int, int]],
        tolerance: int = 5,
    ) -> bool:
        """Check if *end_pos* sits at or just past a no-overlap region."""
        for _, re_, _ in no_overlap:
            if re_ <= end_pos <= re_ + tolerance:
                return True
        for _, img_e in image_positions:
            if img_e <= end_pos <= img_e + tolerance:
                return True
        return False

    @staticmethod
    def _is_table_type(region_type: str, content: str) -> bool:
        """Is this region an HTML or Markdown table?"""
        if "html" in region_type or "markdown" in region_type:
            return True
        if content.lstrip().startswith("<table"):
            return True
        return is_markdown_table(content)

    @staticmethod
    def _find_natural_split(
        text: str, start: int, end: int, chunk_size: int,
    ) -> int:
        """Find the best natural split point in ``[start, end)``."""
        # Prefer paragraph break
        search_start = max(start, end - 200)
        para_match = None
        for m in re.finditer(r"\n\s*\n", text[search_start:end]):
            para_match = m
        if para_match:
            return search_start + para_match.end()

        # Then newline
        nl_pos = text.rfind("\n", start, end)
        if nl_pos > start + chunk_size // 2:
            return nl_pos + 1

        # Then space
        sp_pos = text.rfind(" ", start, end)
        if sp_pos > start + chunk_size // 2:
            return sp_pos + 1

        return end

    # ── Metadata attachment ───────────────────────────────────────────────

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


__all__ = ["ProtectedChunkingStrategy"]
