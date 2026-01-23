# chunking_helper/text_chunker.py
"""
Text Chunker - Text chunking functionality

Main Features:
- Plain text chunking
- Table-free text chunking
- Row-preserving chunking (for tables)
- Code text chunking
- Markdown table support with NO overlap
"""
import logging
import re
from typing import Any, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from contextifier.chunking.constants import (
    LANGCHAIN_CODE_LANGUAGE_MAP, HTML_TABLE_PATTERN, MARKDOWN_TABLE_PATTERN
)

logger = logging.getLogger("document-processor")


def chunk_plain_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk plain text using RecursiveCharacterTextSplitter.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_text(text)


def chunk_text_without_tables(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[str],
    prepend_metadata_func,
    page_tag_processor: Optional[Any] = None
) -> List[str]:
    """
    Chunk text that does not contain tables.

    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks
        metadata: Metadata to prepend to chunks
        prepend_metadata_func: Function to prepend metadata
        page_tag_processor: PageTagProcessor instance (for custom tag patterns)

    Returns:
        List of chunks
    """
    if not text or not text.strip():
        return []

    # Handle HTML code blocks (```html ... ```) separately
    html_code_pattern = r'```html\s*(.*?)\s*```'

    html_chunks = []
    matches = list(re.finditer(html_code_pattern, text, re.DOTALL))

    if matches:
        current_pos = 0
        for m in matches:
            s, e = m.span()
            before = text[current_pos:s].strip()
            if before:
                html_chunks.append(('text', before))
            html_chunks.append(('html', text[s:e]))
            current_pos = e
        after = text[current_pos:].strip()
        if after:
            html_chunks.append(('text', after))
    else:
        html_chunks = [('text', text)]

    final_chunks: List[str] = []

    for kind, content in html_chunks:
        if kind == 'html':
            # Keep HTML code blocks as-is
            final_chunks.append(content)
            continue

        # Plain text uses RecursiveCharacterTextSplitter for chunking
        text_chunks = chunk_plain_text(content, chunk_size, chunk_overlap)
        final_chunks.extend(text_chunks)

    cleaned_chunks = clean_chunks(final_chunks, page_tag_processor)
    cleaned_chunks = prepend_metadata_func(cleaned_chunks, metadata)

    return cleaned_chunks


def _is_markdown_table(text: str) -> bool:
    """
    Check if text is a Markdown table.
    """
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    has_pipe_rows = any(line.strip().startswith('|') for line in lines)
    has_separator = any('---' in line and '|' in line for line in lines)
    return has_pipe_rows and has_separator


def chunk_with_row_protection(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    split_with_protected_regions_func,
    chunk_large_table_func
) -> List[str]:
    """
    Chunk with row-level protection when table protection is disabled.

    HTML tables are processed with chunk_large_table_func to maintain structure.
    Markdown tables are processed with chunk_large_markdown_table for proper row-level splitting.
    Both table types have NO overlap applied.

    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks (NOT applied to tables)
        split_with_protected_regions_func: Protected region splitting function
        chunk_large_table_func: Large table chunking function (for HTML)

    Returns:
        List of chunks
    """
    if not text or not text.strip():
        return []

    # === Extract both HTML and Markdown tables for separate processing ===
    segments: List[Tuple[str, str]] = []  # [(type, content), ...]
    
    # Find all HTML tables
    html_matches = list(re.finditer(HTML_TABLE_PATTERN, text, re.DOTALL | re.IGNORECASE))
    
    # Find all Markdown tables
    markdown_matches = list(re.finditer(MARKDOWN_TABLE_PATTERN, text, re.MULTILINE))
    
    # Combine and sort by start position
    all_matches = []
    for match in html_matches:
        all_matches.append((match.start(), match.end(), 'html_table', match.group(0)))
    for match in markdown_matches:
        start = match.start()
        if match.group(0).startswith('\n'):
            start += 1
        all_matches.append((start, match.end(), 'markdown_table', match.group(0).strip()))
    
    # Sort by start position
    all_matches.sort(key=lambda x: x[0])
    
    # Remove overlapping matches (first non-overlapping match by position wins)
    filtered_matches = []
    last_end = 0
    for start, end, ttype, content in all_matches:
        if start >= last_end:
            filtered_matches.append((start, end, ttype, content))
            last_end = end
    
    # Build segments
    last_end = 0
    for start, end, ttype, content in filtered_matches:
        # Text before table
        if start > last_end:
            before_text = text[last_end:start].strip()
            if before_text:
                segments.append(('text', before_text))
        
        # Table
        segments.append((ttype, content))
        last_end = end
    
    # Text after last table
    if last_end < len(text):
        after_text = text[last_end:].strip()
        if after_text:
            segments.append(('text', after_text))

    # If no tables, use simple row protection
    if not any(seg_type in ('html_table', 'markdown_table') for seg_type, _ in segments):
        return chunk_with_row_protection_simple(
            text, chunk_size, chunk_overlap, split_with_protected_regions_func
        )

    # === Process each segment ===
    all_chunks: List[str] = []

    for seg_type, content in segments:
        if seg_type == 'html_table':
            # HTML table -> split efficiently by rows with NO overlap
            table_chunks = chunk_large_table_func(content, chunk_size, 0, "")
            all_chunks.extend(table_chunks)
        elif seg_type == 'markdown_table':
            # Markdown table -> split efficiently by rows with NO overlap
            from .table_chunker import chunk_large_markdown_table
            table_chunks = chunk_large_markdown_table(content, chunk_size, 0, "")
            all_chunks.extend(table_chunks)
        else:
            # Plain text -> chunk with Markdown row protection
            text_chunks = chunk_with_row_protection_simple(
                content, chunk_size, chunk_overlap, split_with_protected_regions_func
            )
            all_chunks.extend(text_chunks)

    return all_chunks


def chunk_with_row_protection_simple(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    split_with_protected_regions_func
) -> List[str]:
    """
    Chunk while protecting Markdown table rows from being split mid-row.
    Assumes HTML tables have already been separated.

    NOTE: If a complete Markdown table is found, it will be chunked with NO overlap
    using chunk_large_markdown_table. Only individual rows (not part of a complete table)
    are protected as regions.

    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks (NOT applied to Markdown tables)
        split_with_protected_regions_func: Protected region splitting function

    Returns:
        List of chunks
    """
    if not text or not text.strip():
        return []

    # Check if text contains a complete Markdown table
    if _is_markdown_table(text):
        # Process as a complete Markdown table with NO overlap
        from .table_chunker import chunk_large_markdown_table
        return chunk_large_markdown_table(text, chunk_size, 0, "")

    # Protect individual Markdown table rows (for mixed content)
    row_patterns = [
        r'\|[^\n]+\|',  # Markdown table row (headers, data, separators)
    ]

    # Find all row positions
    row_positions: List[Tuple[int, int]] = []
    for pattern in row_patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            row_positions.append((match.start(), match.end()))

    # Sort by position
    row_positions.sort(key=lambda x: x[0])

    # Merge overlapping regions
    merged_rows: List[Tuple[int, int]] = []
    for start, end in row_positions:
        if merged_rows and start < merged_rows[-1][1]:
            # Overlap -> merge
            prev_start, prev_end = merged_rows[-1]
            merged_rows[-1] = (prev_start, max(prev_end, end))
        else:
            merged_rows.append((start, end))

    if not merged_rows:
        # No rows to protect -> use plain chunking
        return chunk_plain_text(text, chunk_size, chunk_overlap)

    # Chunk while protecting rows
    return split_with_protected_regions_func(text, merged_rows, chunk_size, chunk_overlap)


def clean_chunks(
    chunks: List[str],
    page_tag_processor: Optional[Any] = None
) -> List[str]:
    """
    Clean chunks: remove empty chunks and chunks with only page markers.

    Args:
        chunks: List of chunks
        page_tag_processor: PageTagProcessor instance (for custom tag patterns)

    Returns:
        Cleaned list of chunks
    """
    cleaned_chunks = []

    # Build patterns from PageTagProcessor or use defaults
    if page_tag_processor is not None:
        config = page_tag_processor.config
        # Page pattern with optional OCR suffix
        page_prefix = re.escape(config.tag_prefix)
        page_suffix = re.escape(config.tag_suffix)
        slide_prefix = re.escape(config.slide_prefix)
        slide_suffix = re.escape(config.slide_suffix)

        page_marker_patterns = [
            f"{page_prefix}\\d+(\\s*\\(OCR[+Ref]*\\))?{page_suffix}",
            f"{slide_prefix}\\d+(\\s*\\(OCR\\))?{slide_suffix}",
        ]
    else:
        # Default patterns
        page_marker_patterns = [
            r"\[Page Number:\s*\d+(\s*\(OCR[+Ref]*\))?\]",
            r"\[Slide Number:\s*\d+(\s*\(OCR\))?\]",
        ]

    for chunk in chunks:
        if not chunk.strip():
            continue

        # Check if chunk contains only page marker
        is_page_marker_only = False
        for pattern in page_marker_patterns:
            if re.fullmatch(pattern, chunk.strip()):
                is_page_marker_only = True
                break

        if not is_page_marker_only:
            cleaned_chunks.append(chunk)

    return cleaned_chunks


def chunk_code_text(
    text: str,
    file_type: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 300
) -> List[str]:
    """
    Chunk code text using language-specific splitter.

    Args:
        text: Code text
        file_type: File extension (e.g., 'py', 'js')
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks

    Returns:
        List of chunks
    """
    if not text or not text.strip():
        return [""]

    lang = LANGCHAIN_CODE_LANGUAGE_MAP.get(file_type.lower())

    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len, separators=["\n\n", "\n", " ", ""]
        )

    chunks = splitter.split_text(text)
    logger.info(f"Code text split into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")

    return chunks


def reconstruct_text_from_chunks(chunks: List[str], chunk_overlap: int) -> str:
    """
    Reconstruct original text from chunks.
    Removes overlap portions to avoid duplication.

    Args:
        chunks: List of chunks
        chunk_overlap: Overlap size between chunks

    Returns:
        Reconstructed text
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    out = chunks[0]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        cur = chunks[i]
        ov = find_overlap_length(prev, cur, chunk_overlap)
        out += cur[ov:] if ov > 0 else cur

    return out


def find_overlap_length(c1: str, c2: str, max_overlap: int) -> int:
    """
    Find the actual overlap length between two chunks.

    Args:
        c1: Previous chunk
        c2: Current chunk
        max_overlap: Maximum overlap size

    Returns:
        Actual overlap length
    """
    max_check = min(len(c1), len(c2), max_overlap)
    for ov in range(max_check, 0, -1):
        if c1[-ov:] == c2[:ov]:
            return ov
    return 0


def estimate_chunks_count(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
    """
    Estimate the number of chunks when text is chunked.

    Args:
        text: Text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks

    Returns:
        Estimated chunk count
    """
    if not text:
        return 0
    if len(text) <= chunk_size:
        return 1

    eff = chunk_size - chunk_overlap
    return max(1, (len(text) - chunk_overlap) // eff + 1)
