"""Post-chunking metadata enrichment — populate structural context per chunk.

The chunking strategies attach only index/offset metadata; the structural
markers Contextifier injects into the extracted text (``[Page Number: N]``,
``[Slide Number: N]``, ``[Sheet: name]``) and markdown headings were never
promoted into :class:`ChunkMetadata` — RAG pipelines had to re-parse chunk
text to know where a chunk came from.

This enricher runs ONCE over the ordered chunk list, tracking running
document state (current page/slide/sheet + heading stack). Each chunk gets:

* ``page_number`` — the page/slide in effect at the chunk's start (a marker
  at the very head of a chunk counts as that chunk's page);
* ``sheet_name`` — the spreadsheet sheet in effect, when present;
* ``heading_path`` — ``"H1 > H2 > H3"`` breadcrumb of markdown headings in
  effect at the chunk's start (a heading opening the chunk is included).

Purely additive: strategies stay untouched, plain-string chunking is
unaffected, and enrichment never raises (best-effort per chunk).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from contextifier.types import Chunk

_PAGE_RE = re.compile(r"\[(?:Page|Slide) Number:\s*(\d+)\]")
_SHEET_RE = re.compile(r"\[Sheet:\s*([^\]]+)\]")
# Markdown ATX headings — anchored per line; table/code content rarely
# collides because extraction renders headings with markdown hashes.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _heading_breadcrumb(stack: Dict[int, str]) -> Optional[str]:
    if not stack:
        return None
    return " > ".join(stack[level] for level in sorted(stack))


def enrich_chunk_metadata(chunks: List[Chunk]) -> List[Chunk]:
    """Populate ``page_number`` / ``sheet_name`` / ``heading_path`` on each
    chunk's metadata in document order. Mutates and returns *chunks*."""
    page: Optional[int] = None
    sheet: Optional[str] = None
    headings: Dict[int, str] = {}

    for chunk in chunks:
        text = chunk.text or ""
        meta = chunk.metadata
        if meta is None:
            continue  # plain-string mode requested no metadata

        try:
            # Page/slide: a marker counts as THIS chunk's page only when it
            # opens the chunk (nothing but whitespace before it) — content
            # preceding a mid-chunk marker still belongs to the prior page.
            head_page = _PAGE_RE.search(text)
            if head_page is not None and not text[: head_page.start()].strip():
                chunk_page: Optional[int] = int(head_page.group(1))
            else:
                chunk_page = page
            head_sheet = _SHEET_RE.search(text)
            if head_sheet is not None and not text[: head_sheet.start()].strip():
                chunk_sheet: Optional[str] = head_sheet.group(1).strip()
            else:
                chunk_sheet = sheet

            # Advance running page/sheet state with the whole chunk.
            for match in _PAGE_RE.finditer(text):
                page = int(match.group(1))
            for match in _SHEET_RE.finditer(text):
                sheet = match.group(1).strip()

            # Heading path: the breadcrumb where this chunk's content ENDS —
            # headings inside the chunk describe its content, so they join
            # the path (and it carries forward to heading-less chunks).
            for match in _HEADING_RE.finditer(text):
                level = len(match.group(1))
                headings[level] = match.group(2)
                for deeper in [k for k in headings if k > level]:
                    del headings[deeper]
            chunk_path = _heading_breadcrumb(headings)

            if meta.page_number is None and chunk_page is not None:
                meta.page_number = chunk_page
            if getattr(meta, "sheet_name", None) is None and chunk_sheet:
                meta.sheet_name = chunk_sheet
            if getattr(meta, "heading_path", None) is None and chunk_path:
                meta.heading_path = chunk_path
        except Exception:  # noqa: BLE001 — enrichment is best-effort
            continue

    return chunks


__all__ = ["enrich_chunk_metadata"]
