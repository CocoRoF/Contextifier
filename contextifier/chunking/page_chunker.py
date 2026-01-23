# chunking_helper/page_chunker.py
"""
Page Chunker - Page-based chunking

Main Features:
- Split text by pages
- Page merging and chunking
- Overlap handling
- Table protection (HTML and Markdown) with NO overlap for tables
"""
import logging
import re
from typing import List, Optional, Tuple

from contextifier.chunking.protected_regions import (
    find_protected_regions, get_protected_region_positions,
    ensure_protected_region_integrity, split_large_chunk_with_protected_regions
)

logger = logging.getLogger("document-processor")


def split_into_pages(text: str, page_marker_pattern: str) -> List[Tuple[int, str]]:
    """
    Split text by pages.
    Exclude empty pages (pages with only page marker).

    Returns:
        [(page_num, page_content), ...] list
    """
    pages = []

    # Find page marker positions
    markers = list(re.finditer(page_marker_pattern, text))

    if not markers:
        return []

    for i, match in enumerate(markers):
        page_num = int(match.group(1))
        start = match.start()

        # Until next page marker or end of text
        if i + 1 < len(markers):
            end = markers[i + 1].start()
        else:
            end = len(text)

        # Page content (including marker)
        page_content = text[start:end].strip()

        # Empty page check: only page marker exists
        if page_content:
            content_without_marker = re.sub(page_marker_pattern, '', page_content).strip()

            if content_without_marker:
                # Add only pages with actual content
                pages.append((page_num, page_content))
            else:
                # Skip empty pages
                logger.debug(f"Skipping empty page {page_num}")

    # Add content before first page marker if exists
    if markers and markers[0].start() > 0:
        before_content = text[:markers[0].start()].strip()
        if before_content:
            pages.insert(0, (0, before_content))

    return pages


def merge_pages(pages: List[Tuple[int, str]]) -> str:
    """
    Merge pages into a single string.
    """
    return '\n\n'.join(content for _, content in pages)


def get_overlap_content(pages: List[Tuple[int, str]], overlap_size: int) -> str:
    """
    Extract overlap-size content from the last page.
    """
    if not pages:
        return ""

    _, last_content = pages[-1]
    if len(last_content) <= overlap_size:
        return last_content

    return last_content[-overlap_size:]


def chunk_by_pages(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    is_table_based: bool = False,
    force_chunking: bool = False,
    page_tag_processor = None,
    image_pattern: Optional[str] = None,
    chart_pattern: Optional[str] = None,
    metadata_pattern: Optional[str] = None
) -> List[str]:
    """
    Page-based text chunking.

    Algorithm:
    1. Split text by pages
    2. Try to merge pages sequentially
    3. If merged size <= chunk_size, continue merging
    4. If exceeds chunk_size:
       - Allow up to 1.5x
       - If exceeds 1.5x, finalize previous as chunk
    5. If protected regions (tables, charts, Markdown tables) span page boundaries, keep together
       (force_chunking only protects rows for tables, charts are always protected)
    6. Protected regions (image, page, slide, chart, metadata tags) NEVER overlap
    
    Args:
        text: Original text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks (NOT applied to protected regions)
        is_table_based: Whether the file is table-based
        force_chunking: Force chunking (disable table protection)
        page_tag_processor: PageTagProcessor instance for custom patterns
        image_pattern: Custom regex pattern for image tags
        chart_pattern: Custom regex pattern for chart blocks
        metadata_pattern: Custom regex pattern for metadata blocks
    """
    # Build page marker patterns from PageTagProcessor or use defaults
    if page_tag_processor is not None:
        page_marker_patterns = [
            page_tag_processor.get_pattern_string(),  # Page pattern
        ]
        config = page_tag_processor.config
        if config.slide_prefix != config.tag_prefix:
            from contextifier.core.functions.page_tag_processor import PageTagType
            page_marker_patterns.append(page_tag_processor.get_pattern_string(PageTagType.SLIDE))
    else:
        page_marker_patterns = [
            r'\[Page Number:\s*(\d+)\]',  # Default page format
            r'\[Slide Number:\s*(\d+)\]',  # Default slide format
        ]
    
    # Find first matching pattern
    pages = []
    for page_marker_pattern in page_marker_patterns:
        pages = split_into_pages(text, page_marker_pattern)
        if pages:
            break

    if not pages:
        # Page split failed, fall back to plain text chunking
        from .text_chunker import chunk_plain_text
        return chunk_plain_text(text, chunk_size, chunk_overlap)

    logger.debug(f"Split into {len(pages)} pages")

    # Identify protected region positions (HTML tables, chart blocks, Markdown tables, image tags)
    # force_chunking disables table protection (charts are always protected)
    protected_regions = find_protected_regions(
        text, is_table_based, force_chunking, image_pattern,
        chart_pattern, page_tag_processor, metadata_pattern
    )
    protected_positions = get_protected_region_positions(protected_regions)

    # Merge pages to create chunks
    chunks = []
    max_size = int(chunk_size * 1.5)  # Allow up to 1.5x

    current_chunk_pages = []  # Pages included in current chunk
    current_size = 0
    pending_overlap = ""  # Overlap content to prepend to next chunk

    for page_idx, (page_num, page_content) in enumerate(pages):
        page_size = len(page_content)

        # Apply pending overlap to page content
        if pending_overlap:
            page_content = pending_overlap + "\n\n" + page_content
            page_size = len(page_content)
            pending_overlap = ""

        if not current_chunk_pages:
            # First page
            current_chunk_pages.append((page_num, page_content))
            current_size = page_size
            continue

        # Try to merge
        # Add \n\n between pages (4 chars)
        potential_size = current_size + 4 + page_size

        if potential_size <= chunk_size:
            # Within chunk_size: merge
            current_chunk_pages.append((page_num, page_content))
            current_size = potential_size
        elif potential_size <= max_size:
            # Exceeds chunk_size but within 1.5x: allow merge
            current_chunk_pages.append((page_num, page_content))
            current_size = potential_size

            # Finalize this chunk (no more additions)
            chunk_content = merge_pages(current_chunk_pages)

            # Verify protected region integrity: warn if chunk ends mid-region
            chunk_content = ensure_protected_region_integrity(chunk_content)

            chunks.append(chunk_content)

            # Overlap handling: include part of last page in next chunk
            overlap_content = get_overlap_content(current_chunk_pages, chunk_overlap)
            current_chunk_pages = []
            current_size = 0

            if overlap_content:
                # Store overlap to prepend to next chunk's first page
                pending_overlap = overlap_content
        else:
            # Exceeds 1.5x: finalize current chunk, new page goes to next chunk
            if current_chunk_pages:
                chunk_content = merge_pages(current_chunk_pages)
                chunk_content = ensure_protected_region_integrity(chunk_content)
                chunks.append(chunk_content)

            # Start new chunk
            current_chunk_pages = [(page_num, page_content)]
            current_size = page_size

    # Process remaining pages
    if current_chunk_pages:
        chunk_content = merge_pages(current_chunk_pages)
        chunk_content = ensure_protected_region_integrity(chunk_content)
        chunks.append(chunk_content)

    # Split very large chunks (protect protected regions)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_size * 1.5:
            # Very large chunk: split while protecting regions
            sub_chunks = split_large_chunk_with_protected_regions(
                chunk, chunk_size, chunk_overlap, is_table_based, force_chunking,
                image_pattern, chart_pattern, page_tag_processor, metadata_pattern
            )
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks
