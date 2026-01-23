# chunking_helper/sheet_processor.py
"""
Sheet Processor - Sheet and metadata processing

Main Features:
- Document metadata extraction
- Sheet section extraction
- Multi-sheet content chunking
- Single table content chunking
- NO overlap for table chunks (intentional for search quality)
"""
import logging
import re
from typing import List, Optional, Tuple

from contextifier.chunking.constants import (
    HTML_TABLE_PATTERN,
    MARKDOWN_TABLE_PATTERN,
    IMAGE_TAG_PATTERN,
    CHART_BLOCK_PATTERN,
    TEXTBOX_BLOCK_PATTERN
)

logger = logging.getLogger("document-processor")


def extract_document_metadata(
    text: str,
    metadata_pattern: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Extract Document-Metadata block from text.

    Args:
        text: Original text
        metadata_pattern: Custom metadata pattern (if None, uses default)

    Returns:
        (metadata_block, remaining_text) tuple
    """
    # Use custom pattern or default
    pattern = metadata_pattern if metadata_pattern is not None else r'<Document-Metadata>.*?</Document-Metadata>\s*'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        metadata_block = match.group(0).strip()
        remaining_text = text[:match.start()] + text[match.end():]
        return metadata_block, remaining_text.strip()

    return None, text


def prepend_metadata_to_chunks(chunks: List[str], metadata: Optional[str]) -> List[str]:
    """
    Prepend metadata to each chunk.

    Args:
        chunks: List of chunks
        metadata: Metadata block

    Returns:
        Chunks with metadata prepended
    """
    if not metadata:
        return chunks
    return [f"{metadata}\n\n{chunk}" for chunk in chunks]


def extract_sheet_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract Excel sheet sections.

    Args:
        text: Full text

    Returns:
        [(sheet_name, sheet_content), ...] list
    """
    # Sheet marker pattern - only standard format from PageTagProcessor
    sheet_pattern = r'\[Sheet:\s*([^\]]+)\]'
    marker_template = '[Sheet: {name}]'

    matches = list(re.finditer(sheet_pattern, text))

    if not matches:
        return []

    sheets = []

    for i, match in enumerate(matches):
        sheet_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start:end].strip()
        if content:
            # Include sheet marker in content
            sheet_marker = marker_template.format(name=sheet_name)
            full_content = f"{sheet_marker}\n{content}"
            sheets.append((sheet_name, full_content))

    return sheets


def extract_content_segments(
    content: str,
    image_pattern: Optional[str] = None,
    chart_pattern: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Extract various types of segments from content.

    Segment Types:
    - table: HTML table or Markdown table (including table markers)
    - textbox: [textbox]...[/textbox] block
    - chart: [chart]...[/chart] block
    - image: [image:...] tag
    - text: Plain text

    Args:
        content: Content to parse
        image_pattern: Custom image tag pattern (if None, uses default IMAGE_TAG_PATTERN)
        chart_pattern: Custom chart block pattern (if None, uses default CHART_BLOCK_PATTERN)

    Returns:
        [(segment_type, segment_content), ...] list
    """
    segments: List[Tuple[str, str]] = []

    # Use custom patterns or defaults
    img_pat = image_pattern if image_pattern is not None else IMAGE_TAG_PATTERN
    chart_pat = chart_pattern if chart_pattern is not None else CHART_BLOCK_PATTERN

    # Define special block patterns
    # Recognize [Table N] marker together with table as a single block
    patterns = [
        # [Table N] + HTML table
        ('table', r'(?:\[Table\s*\d+\]\s*)?<table\s+border=["\']1["\']>.*?</table>'),
        # [Table N] + Markdown table (multiple lines starting with |, last row matches even without newline)
        ('table', r'\[Table\s*\d+\]\s*\n(?:\|[^\n]*\|(?:\s*\n|$))+'),
        # Standalone Markdown table (starts with | and has --- separator, last row matches even without newline)
        ('table', r'(?:^|\n)(\|[^\n]*\|\s*\n\|[\s\-:]*\|[^\n]*(?:\n\|[^\n]*\|)*)'),
        ('textbox', TEXTBOX_BLOCK_PATTERN),
        ('chart', chart_pat),
        ('image', img_pat),
    ]

    # Find all special block positions
    all_matches: List[Tuple[int, int, str, str]] = []  # (start, end, type, content)

    for segment_type, pattern in patterns:
        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE | re.MULTILINE):
            matched_content = match.group(0).strip()
            # Ignore empty matches
            if not matched_content:
                continue
            all_matches.append((match.start(), match.end(), segment_type, matched_content))

    # Sort by start position
    all_matches.sort(key=lambda x: x[0])

    # Remove overlapping matches (longer match wins)
    filtered_matches: List[Tuple[int, int, str, str]] = []
    last_end = 0
    for start, end, segment_type, segment_content in all_matches:
        if start >= last_end:
            filtered_matches.append((start, end, segment_type, segment_content))
            last_end = end

    # Build segments (special blocks + plain text between them)
    current_pos = 0
    for start, end, segment_type, segment_content in filtered_matches:
        # Plain text before special block
        if start > current_pos:
            text_between = content[current_pos:start].strip()
            # Skip text that only contains [Table N] marker (will be combined with next table)
            if text_between and not re.match(r'^\[Table\s*\d+\]\s*$', text_between):
                segments.append(('text', text_between))

        # Special block
        segments.append((segment_type, segment_content))
        current_pos = end

    # Plain text after last special block
    if current_pos < len(content):
        remaining_text = content[current_pos:].strip()
        # Ignore text that only contains [Table N] marker
        if remaining_text and not re.match(r'^\[Table\s*\d+\]\s*$', remaining_text):
            segments.append(('text', remaining_text))

    return segments


def chunk_multi_sheet_content(
    sheets: List[Tuple[str, str]],
    metadata_block: Optional[str],
    analysis_block: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_plain_text_func,
    chunk_large_table_func,
    image_pattern: Optional[str] = None,
    chart_pattern: Optional[str] = None,
    metadata_pattern: Optional[str] = None
) -> List[str]:
    """
    Chunk multi-sheet content.

    Each sheet is processed independently and split if necessary.
    All chunks include metadata and sheet information.
    Handles not only tables but also additional content before/after tables (textbox, chart, image, etc.).

    Args:
        sheets: [(sheet_name, sheet_content), ...] list
        metadata_block: Metadata block
        analysis_block: Analysis block
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        chunk_plain_text_func: Plain text chunking function
        chunk_large_table_func: Large table chunking function
        image_pattern: Custom image tag pattern (if None, uses default)
        chart_pattern: Custom chart block pattern (if None, uses default)
        metadata_pattern: Custom metadata block pattern (if None, uses default)

    Returns:
        List of chunks
    """
    all_chunks: List[str] = []

    # Build common metadata (included in all chunks)
    common_metadata_parts = []
    if metadata_block:
        common_metadata_parts.append(metadata_block)
    if analysis_block:
        common_metadata_parts.append(analysis_block)
    common_metadata = "\n\n".join(common_metadata_parts) if common_metadata_parts else ""

    for sheet_idx, (sheet_name, sheet_content) in enumerate(sheets):
        # Extract sheet marker - only standard format
        sheet_marker_match = re.match(r'(\[Sheet:\s*[^\]]+\])', sheet_content)
        sheet_marker = sheet_marker_match.group(1) if sheet_marker_match else f"[Sheet: {sheet_name}]"

        # Build context for this sheet (metadata + sheet info)
        context_parts = []
        if common_metadata:
            context_parts.append(common_metadata)
        context_parts.append(sheet_marker)
        context_prefix = "\n\n".join(context_parts) if context_parts else ""

        # Remove sheet marker from content
        content_after_marker = sheet_content
        if sheet_marker_match:
            content_after_marker = sheet_content[sheet_marker_match.end():].strip()

        # === Split sheet content into segments ===
        # Segments: tables, textbox, chart, image blocks and plain text
        segments = extract_content_segments(
            content_after_marker,
            image_pattern=image_pattern,
            chart_pattern=chart_pattern
        )

        if not segments:
            # Skip empty sheets
            continue

        # Process each segment
        for segment_type, segment_content in segments:
            if not segment_content.strip():
                continue

            segment_size = len(segment_content)

            if segment_type == 'table':
                # Table processing - NO overlap for tables
                if segment_size + len(context_prefix) <= chunk_size:
                    all_chunks.append(f"{context_prefix}\n{segment_content}")
                else:
                    # Large table: split with NO overlap (0 is passed, not chunk_overlap)
                    table_chunks = chunk_large_table_func(
                        segment_content, chunk_size, 0,  # NO overlap for tables
                        context_prefix=context_prefix
                    )
                    all_chunks.extend(table_chunks)

            elif segment_type in ('textbox', 'chart', 'image'):
                # Protected blocks: never split, keep as single chunk
                if len(context_prefix) + segment_size > chunk_size:
                    # Exceeds chunk size but keep intact (protected block)
                    logger.warning(f"{segment_type} block exceeds chunk_size, but keeping it intact")
                all_chunks.append(f"{context_prefix}\n{segment_content}")

            else:
                # Plain text
                if len(context_prefix) + segment_size <= chunk_size:
                    all_chunks.append(f"{context_prefix}\n{segment_content}")
                else:
                    # Split long plain text
                    text_chunks = chunk_plain_text_func(segment_content, chunk_size, chunk_overlap)
                    for chunk in text_chunks:
                        all_chunks.append(f"{context_prefix}\n{chunk}")

    logger.info(f"Multi-sheet content split into {len(all_chunks)} chunks")

    return all_chunks


def chunk_single_table_content(
    text: str,
    metadata_block: Optional[str],
    analysis_block: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_plain_text_func,
    chunk_large_table_func,
    image_pattern: Optional[str] = None,
    chart_pattern: Optional[str] = None,
    metadata_pattern: Optional[str] = None
) -> List[str]:
    """
    Chunk single table content.
    Include metadata in all chunks.

    NOTE: Table chunks have NO overlap to prevent data duplication.

    Args:
        text: Text containing table
        metadata_block: Metadata block
        analysis_block: Analysis block
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap (NOT applied to tables)
        chunk_plain_text_func: Plain text chunking function
        chunk_large_table_func: Large table chunking function
        image_pattern: Custom image tag pattern (if None, uses default)
        chart_pattern: Custom chart block pattern (if None, uses default)
        metadata_pattern: Custom metadata block pattern (if None, uses default)

    Returns:
        List of chunks
    """
    # Build context (included in all chunks)
    context_parts = []
    if metadata_block:
        context_parts.append(metadata_block)
    if analysis_block:
        context_parts.append(analysis_block)
    context_prefix = "\n\n".join(context_parts) if context_parts else ""

    # Extract tables (HTML or Markdown)
    html_table_matches = list(re.finditer(HTML_TABLE_PATTERN, text, re.DOTALL | re.IGNORECASE))
    markdown_table_matches = list(re.finditer(MARKDOWN_TABLE_PATTERN, text, re.MULTILINE))
    
    # Combine all table matches with type info
    all_table_matches: List[Tuple[int, int, str, str]] = []  # (start, end, type, content)
    
    for match in html_table_matches:
        all_table_matches.append((match.start(), match.end(), 'html', match.group(0)))
    
    for match in markdown_table_matches:
        table_start = match.start()
        if match.group(0).startswith('\n'):
            table_start += 1
        all_table_matches.append((table_start, match.end(), 'markdown', match.group(0).strip()))
    
    # Sort by start position and remove overlaps
    all_table_matches.sort(key=lambda x: x[0])
    filtered_matches: List[Tuple[int, int, str, str]] = []
    last_end = 0
    for start, end, ttype, content in all_table_matches:
        if start >= last_end:
            filtered_matches.append((start, end, ttype, content))
            last_end = end

    if not filtered_matches:
        # No tables found - use plain text chunking
        full_text = text
        if context_prefix:
            full_text = f"{context_prefix}\n\n{full_text}"
        return chunk_plain_text_func(full_text, chunk_size, chunk_overlap)

    # Result chunks
    all_chunks: List[str] = []

    # Process each table
    for start, end, table_type, table_content in filtered_matches:
        table_size = len(table_content)

        logger.debug(f"Processing {table_type} table: {table_size} chars")

        if table_size + len(context_prefix) <= chunk_size:
            # Small table: include with context
            if context_prefix:
                all_chunks.append(f"{context_prefix}\n\n{table_content}")
            else:
                all_chunks.append(table_content)
        else:
            # Large table: split with NO overlap (context included in all chunks)
            table_chunks = chunk_large_table_func(
                table_content, chunk_size, 0,  # NO overlap for tables
                context_prefix=context_prefix
            )
            all_chunks.extend(table_chunks)

    logger.info(f"Single table content split into {len(all_chunks)} chunks")

    return all_chunks
