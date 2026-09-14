"""Post-chunking metadata enrichment — populate structural context per chunk.

The chunking strategies attach only index/offset metadata; the structural
markers Contextifier injects into the extracted text (``[Page Number: N]``,
``[Slide Number: N]``, ``[Sheet: name]``) and markdown headings were never
promoted into :class:`ChunkMetadata` — RAG pipelines had to re-parse chunk
text to know where a chunk came from.

This enricher runs ONCE over the ordered chunk list, tracking running
document state (current page/slide/sheet + heading stack). Each chunk gets:

* ``page_number`` — the page/slide in effect at the chunk's start (a marker
  at the head of a chunk counts as that chunk's page; the injected
  document-metadata block does not count as content, so an opening chunk that
  carries front matter still belongs to the first page it declares);
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


_DEFAULT_METADATA_BLOCK_RE = re.compile(
    r"\[Document-Metadata\].*?\[/Document-Metadata\]", re.DOTALL
)


def _metadata_block_pattern(config: Optional[object]) -> re.Pattern:
    """Regex for the injected document-metadata block, honouring custom tags."""
    tags = getattr(config, "tags", None)
    prefix = getattr(tags, "metadata_prefix", None)
    suffix = getattr(tags, "metadata_suffix", None)
    if not prefix or not suffix:
        return _DEFAULT_METADATA_BLOCK_RE
    return re.compile(re.escape(prefix) + r".*?" + re.escape(suffix), re.DOTALL)


def _lead_is_content(lead: str, metadata_pattern: re.Pattern) -> bool:
    """Does the text before a marker count as page content?

    A marker "opens" a chunk when nothing meaningful precedes it. The metadata
    block the extractor injects at the top of a document is front matter, not
    page content, so an opening chunk that carries it still belongs to the
    first page it declares. Real text before a marker, on the other hand, comes
    from the *previous* page and must keep that attribution.
    """
    return bool(metadata_pattern.sub("", lead).strip())


def _heading_breadcrumb(stack: Dict[int, str]) -> Optional[str]:
    if not stack:
        return None
    return " > ".join(stack[level] for level in sorted(stack))


def enrich_chunk_metadata(
    chunks: List[Chunk],
    config: Optional[object] = None,
) -> List[Chunk]:
    """Populate ``page_number`` / ``sheet_name`` / ``heading_path`` on each
    chunk's metadata in document order. Mutates and returns *chunks*.

    Args:
        chunks: Ordered chunks to enrich.
        config: Optional :class:`ProcessingConfig`, used only to recognise a
            custom document-metadata block delimiter.
    """
    page: Optional[int] = None
    sheet: Optional[str] = None
    headings: Dict[int, str] = {}
    metadata_pattern = _metadata_block_pattern(config)

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
            if head_page is not None and not _lead_is_content(
                text[: head_page.start()], metadata_pattern
            ):
                chunk_page: Optional[int] = int(head_page.group(1))
            else:
                chunk_page = page
            head_sheet = _SHEET_RE.search(text)
            if head_sheet is not None and not _lead_is_content(
                text[: head_sheet.start()], metadata_pattern
            ):
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
