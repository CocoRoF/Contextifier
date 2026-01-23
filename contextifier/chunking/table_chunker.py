# chunking_helper/table_chunker.py
"""
Table Chunker - Core table chunking logic

Main Features:
- Split large HTML tables to fit chunk_size
- Split large Markdown tables to fit chunk_size  
- Preserve and restore table structure (headers)
- rowspan/colspan aware splitting for HTML
- rowspan adjustment
- NO OVERLAP for table chunks (intentional to prevent data duplication)
"""
import logging
import re
from typing import Dict, List, Optional

from contextifier.chunking.constants import (
    ParsedTable, TableRow, ParsedMarkdownTable,
    TABLE_WRAPPER_OVERHEAD, CHUNK_INDEX_OVERHEAD,
    MARKDOWN_TABLE_SEPARATOR_PATTERN
)
from contextifier.chunking.table_parser import (
    parse_html_table, extract_cell_spans_with_positions, has_complex_spans
)

logger = logging.getLogger("document-processor")


def calculate_available_space(
    chunk_size: int,
    header_size: int,
    chunk_index: int = 0,
    total_chunks: int = 1
) -> int:
    """
    Calculate available space for data rows in a chunk.

    Args:
        chunk_size: Total chunk size
        header_size: Header size
        chunk_index: Current chunk index (0-based)
        total_chunks: Expected total number of chunks

    Returns:
        Number of characters available for data rows
    """
    # Fixed overhead
    overhead = TABLE_WRAPPER_OVERHEAD

    # Chunk index metadata overhead (only when total chunks > 1)
    if total_chunks > 1:
        overhead += CHUNK_INDEX_OVERHEAD

    # Header overhead (include header even for non-first chunks)
    overhead += header_size

    available = chunk_size - overhead

    return max(available, 100)  # Guarantee at least 100 characters


def adjust_rowspan_in_chunk(rows_html: List[str], total_rows_in_chunk: int) -> List[str]:
    """
    Readjust rowspan values for rows in a chunk.

    Adjusts rowspan values to match the number of rows included in the chunk
    so that the table renders correctly.

    Args:
        rows_html: List of HTML row strings included in the chunk
        total_rows_in_chunk: Total number of rows in the chunk

    Returns:
        List of HTML row strings with adjusted rowspan values
    """
    if not rows_html:
        return rows_html

    adjusted_rows = []

    for row_idx, row_html in enumerate(rows_html):
        remaining_rows = total_rows_in_chunk - row_idx

        def adjust_cell_rowspan(match):
            """Callback function to adjust cell rowspan"""
            tag = match.group(1)  # td or th
            attrs = match.group(2)
            content = match.group(3)

            # Extract current rowspan
            rowspan_match = re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
            if rowspan_match:
                original_rowspan = int(rowspan_match.group(1))

                # Adjust if greater than remaining rows
                adjusted_rowspan = min(original_rowspan, remaining_rows)

                if adjusted_rowspan <= 1:
                    # Remove attribute if rowspan=1
                    new_attrs = re.sub(r'\s*rowspan=["\']?\d+["\']?', '', attrs, flags=re.IGNORECASE)
                else:
                    # Adjust rowspan value
                    new_attrs = re.sub(
                        r'rowspan=["\']?\d+["\']?',
                        f"rowspan='{adjusted_rowspan}'",
                        attrs,
                        flags=re.IGNORECASE
                    )

                return f'<{tag}{new_attrs}>{content}</{tag}>'

            return match.group(0)

        # Cell pattern: <td ...>...</td> or <th ...>...</th>
        cell_pattern = r'<(td|th)([^>]*)>(.*?)</\1>'
        adjusted_row = re.sub(cell_pattern, adjust_cell_rowspan, row_html, flags=re.DOTALL | re.IGNORECASE)

        adjusted_rows.append(adjusted_row)

    return adjusted_rows


def build_table_chunk(
    header_html: str,
    data_rows: List[TableRow],
    chunk_index: int = 0,
    total_chunks: int = 1,
    context_prefix: str = ""
) -> str:
    """
    Build a complete table HTML for a chunk.

    Automatically adjusts rowspan if it exceeds the chunk boundary.

    Args:
        header_html: HTML of header rows
        data_rows: Data rows
        chunk_index: Current chunk index (0-based)
        total_chunks: Total number of chunks
        context_prefix: Context info (metadata, sheet info, etc.) - included in all chunks

    Returns:
        Complete table HTML
    """
    parts = []

    # Context info (metadata, sheet info, etc.) - included in all chunks
    if context_prefix:
        parts.append(context_prefix)

    # Chunk index metadata (only when more than 1 chunk)
    if total_chunks > 1:
        parts.append(f"[Table Chunk {chunk_index + 1}/{total_chunks}]")

    # Table start
    parts.append("<table border='1'>")

    # Header (if exists)
    if header_html:
        parts.append(header_html)

    # Extract HTML for data rows
    rows_html = [row.html for row in data_rows]

    # Adjust rowspan
    adjusted_rows = adjust_rowspan_in_chunk(rows_html, len(data_rows))

    # Add adjusted rows
    for row_html in adjusted_rows:
        parts.append(row_html)

    # Table end
    parts.append("</table>")

    return "\n".join(parts)


def update_chunk_metadata(chunks: List[str], total_chunks: int) -> List[str]:
    """
    Update chunk metadata (total chunk count).
    """
    updated_chunks = []

    for idx, chunk in enumerate(chunks):
        # Existing metadata pattern
        old_pattern = r'\[Table Chunk \d+/\d+\]'
        new_metadata = f"[Table Chunk {idx + 1}/{total_chunks}]"

        if re.search(old_pattern, chunk):
            updated_chunk = re.sub(old_pattern, new_metadata, chunk)
        else:
            # Add metadata if not present
            updated_chunk = f"{new_metadata}\n{chunk}"

        updated_chunks.append(updated_chunk)

    return updated_chunks


def split_table_into_chunks(
    parsed_table: ParsedTable,
    chunk_size: int,
    chunk_overlap: int = 0,
    context_prefix: str = ""
) -> List[str]:
    """
    Split a parsed table to fit chunk_size.
    Each chunk has a complete table structure (including headers).

    NOTE: Table chunking does NOT apply overlap.
    Data duplication degrades search quality, so overlap is intentionally excluded.

    Row splitting rules:
    - Minimum 1 row per chunk (rows are NEVER split)
    - Chunks can expand up to 1.5x of chunk_size to include more rows
    - Only exceeds chunk_size when necessary to maintain row integrity

    Args:
        parsed_table: Parsed table information
        chunk_size: Maximum chunk size
        chunk_overlap: Not used (kept for compatibility)
        context_prefix: Context info (metadata, sheet info, etc.) - included in all chunks

    Returns:
        List of split table HTML chunks
    """
    data_rows = parsed_table.data_rows
    header_html = parsed_table.header_html
    header_size = parsed_table.header_size

    # Calculate context size
    context_size = len(context_prefix) + 2 if context_prefix else 0  # Including newline

    if not data_rows:
        # Return original if no data rows
        return [parsed_table.original_html]

    # Calculate estimated chunk count (approximate)
    total_data_size = sum(row.char_length for row in data_rows)
    available_per_chunk = calculate_available_space(chunk_size, header_size + context_size, 0, 1)
    estimated_chunks = max(1, (total_data_size + available_per_chunk - 1) // available_per_chunk)

    # Recalculate with actual chunk count
    available_per_chunk = calculate_available_space(chunk_size, header_size + context_size, 0, estimated_chunks)

    # Maximum allowed chunk size (1.5x of chunk_size)
    max_chunk_data_size = int(chunk_size * 1.5) - header_size - context_size - CHUNK_INDEX_OVERHEAD

    chunks: List[str] = []
    current_rows: List[TableRow] = []
    current_size = 0
    # Table chunking does not apply overlap (prevent data duplication)

    for row_idx, row in enumerate(data_rows):
        row_size = row.char_length + 1  # Including newline

        # Check if adding this row exceeds available space
        if current_rows and (current_size + row_size > available_per_chunk):
            # Check if we can still fit within 1.5x limit
            if current_size + row_size <= max_chunk_data_size:
                # Still within 1.5x limit - add row to current chunk
                current_rows.append(row)
                current_size += row_size
            else:
                # Exceeds 1.5x limit - flush current chunk and start new one
                chunk_html = build_table_chunk(
                    header_html,
                    current_rows,
                    chunk_index=len(chunks),
                    total_chunks=estimated_chunks,
                    context_prefix=context_prefix
                )
                chunks.append(chunk_html)

                # Start new chunk with this row (minimum 1 row guaranteed)
                current_rows = [row]
                current_size = row_size
        else:
            # Row fits - add to current chunk
            current_rows.append(row)
            current_size += row_size

    # Process last chunk
    if current_rows:
        chunk_html = build_table_chunk(
            header_html,
            current_rows,
            chunk_index=len(chunks),
            total_chunks=max(len(chunks) + 1, estimated_chunks),
            context_prefix=context_prefix
        )
        chunks.append(chunk_html)

    # Update metadata with actual total chunk count
    if len(chunks) != estimated_chunks and len(chunks) > 1:
        chunks = update_chunk_metadata(chunks, len(chunks))

    logger.info(f"Table split into {len(chunks)} chunks (original: {len(parsed_table.original_html)} chars)")

    return chunks


def split_table_preserving_rowspan(
    parsed_table: ParsedTable,
    chunk_size: int,
    chunk_overlap: int,
    context_prefix: str = ""
) -> List[str]:
    """
    Split a table considering rowspan.

    Rows connected by rowspan are kept together as semantic blocks.

    NOTE: Table chunking does NOT apply overlap.
    Data duplication degrades search quality, so overlap is intentionally excluded.

    Algorithm:
    1. Track active rowspan for each row (by column position, considering colspan)
    2. If all rowspans from previous row end and new rowspan starts, create new block
    3. Combine blocks to fit chunk_size

    Args:
        parsed_table: Parsed table
        chunk_size: Chunk size
        chunk_overlap: Not used (kept for compatibility)
        context_prefix: Context info (metadata, sheet info, etc.)

    Returns:
        List of split table chunks
    """
    data_rows = parsed_table.data_rows
    header_html = parsed_table.header_html
    header_size = parsed_table.header_size

    # Calculate context size
    context_size = len(context_prefix) + 2 if context_prefix else 0

    if not data_rows:
        if context_prefix:
            return [f"{context_prefix}\n{parsed_table.original_html}"]
        return [parsed_table.original_html]

    # === Identify rowspan blocks ===
    # Block = group of consecutive rows connected by rowspan
    active_rowspans: Dict[int, int] = {}  # column_position -> remaining_rows (including current row)
    row_block_ids: List[int] = []  # Block ID for each row
    current_block_id = -1

    for row_idx, row in enumerate(data_rows):
        # 1. Decrease remaining rowspan from previous row (except first row)
        if row_idx > 0:
            finished_cols = []
            for col in list(active_rowspans.keys()):
                active_rowspans[col] -= 1
                if active_rowspans[col] <= 0:
                    finished_cols.append(col)
            for col in finished_cols:
                del active_rowspans[col]

        # State after decrease (before adding new spans)
        had_active_before_new = len(active_rowspans) > 0

        # 2. Add new rowspans starting from current row
        new_spans = extract_cell_spans_with_positions(row.html)
        for col, span in new_spans.items():
            # Update if larger than existing rowspan (longer span takes priority)
            if col not in active_rowspans or span > active_rowspans[col]:
                active_rowspans[col] = span

        has_active_now = len(active_rowspans) > 0
        has_new_span = len(new_spans) > 0

        # Block determination logic:
        # - No active rowspan -> independent block
        # - No active after previous row processing but new span starts -> new block
        # - Otherwise maintain existing block
        if not has_active_now:
            # No rowspan - independent row
            current_block_id += 1
            row_block_ids.append(current_block_id)
        elif not had_active_before_new and has_new_span:
            # All previous rowspans ended and new rowspan starts - new block
            current_block_id += 1
            row_block_ids.append(current_block_id)
        else:
            # Maintain existing block
            row_block_ids.append(current_block_id)

    # Group rows by block
    block_groups: Dict[int, List[int]] = {}
    for row_idx, block_id in enumerate(row_block_ids):
        if block_id not in block_groups:
            block_groups[block_id] = []
        block_groups[block_id].append(row_idx)

    # Create row_groups in sorted block order
    row_groups: List[List[int]] = [
        block_groups[block_id]
        for block_id in sorted(block_groups.keys())
    ]

    # === Combine groups into chunks ===
    chunks: List[str] = []
    current_rows: List[TableRow] = []
    current_size = 0

    available_space = calculate_available_space(chunk_size, header_size + context_size, 0, 1)
    # Maximum allowed chunk size (1.5x of chunk_size)
    max_chunk_data_size = int(chunk_size * 1.5) - header_size - context_size - CHUNK_INDEX_OVERHEAD

    for group in row_groups:
        group_rows = [data_rows[idx] for idx in group]
        group_size = sum(row.char_length + 1 for row in group_rows)

        if current_rows and current_size + group_size > available_space:
            # Check if we can still fit within 1.5x limit
            if current_size + group_size <= max_chunk_data_size:
                # Still within 1.5x limit - add group to current chunk
                current_rows.extend(group_rows)
                current_size += group_size
            else:
                # Exceeds 1.5x limit - flush current chunk and start new one
                chunks.append(build_table_chunk(
                    header_html, current_rows, len(chunks), len(chunks) + 2,
                    context_prefix=context_prefix
                ))
                current_rows = group_rows[:]
                current_size = group_size
        else:
            current_rows.extend(group_rows)
            current_size += group_size

    # Last chunk
    if current_rows:
        chunks.append(build_table_chunk(
            header_html, current_rows, len(chunks), len(chunks) + 1,
            context_prefix=context_prefix
        ))

    # Update chunk count
    if len(chunks) > 1:
        chunks = update_chunk_metadata(chunks, len(chunks))

    return chunks


def chunk_large_table(
    table_html: str,
    chunk_size: int,
    chunk_overlap: int,
    context_prefix: str = ""
) -> List[str]:
    """
    Split large HTML table to fit chunk_size.
    Restores table structure (headers) in each chunk.

    Also handles complex tables with rowspan.

    NOTE: Table chunking does NOT apply overlap.
    Data duplication degrades search quality, so overlap is intentionally excluded.

    Args:
        table_html: HTML table string
        chunk_size: Maximum chunk size
        chunk_overlap: Not used (kept for compatibility)
        context_prefix: Context info (metadata, sheet info, etc.) - included in all chunks

    Returns:
        List of split table HTML chunks
    """
    # Parse table
    parsed = parse_html_table(table_html)

    if not parsed:
        logger.warning("Failed to parse table, returning original")
        if context_prefix:
            return [f"{context_prefix}\n{table_html}"]
        return [table_html]

    # No need to split if table fits in chunk_size
    if len(table_html) + len(context_prefix) <= chunk_size:
        if context_prefix:
            return [f"{context_prefix}\n{table_html}"]
        return [table_html]

    # No need to split if no data rows
    if not parsed.data_rows:
        if context_prefix:
            return [f"{context_prefix}\n{table_html}"]
        return [table_html]

    # Check for complex spans (rowspan)
    if has_complex_spans(table_html):
        logger.info("Complex table with rowspan detected, using span-aware splitting")
        return split_table_preserving_rowspan(parsed, chunk_size, chunk_overlap, context_prefix)

    # Standard table splitting
    chunks = split_table_into_chunks(parsed, chunk_size, chunk_overlap, context_prefix)

    return chunks


# ============================================================================
# Markdown Table Chunking Functions
# ============================================================================

def parse_markdown_table(table_text: str) -> Optional[ParsedMarkdownTable]:
    """
    Parse a Markdown table and extract structural information.

    A Markdown table has:
    - Header row: | col1 | col2 | col3 |
    - Separator row: |---|---|---| or |:---:|:---|---:|
    - Data rows: | data1 | data2 | data3 |

    Args:
        table_text: Markdown table text

    Returns:
        ParsedMarkdownTable object or None if parsing fails
    """
    try:
        # Split into lines and filter empty lines
        lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]

        if len(lines) < 2:
            logger.debug("Not enough lines for a valid Markdown table")
            return None

        # Find header and separator rows
        header_row = None
        separator_row = None
        separator_idx = -1

        for idx, line in enumerate(lines):
            # Check if this line is a separator (contains only |, -, :, and spaces)
            if re.match(MARKDOWN_TABLE_SEPARATOR_PATTERN, line):
                separator_row = line
                separator_idx = idx
                # Header is the line before separator
                if idx > 0:
                    header_row = lines[idx - 1]
                break

        if not separator_row or not header_row:
            # Try simpler detection: first row is header, second row is separator
            if len(lines) >= 2 and lines[0].startswith('|') and '---' in lines[1]:
                header_row = lines[0]
                separator_row = lines[1]
                separator_idx = 1
            else:
                logger.debug("Could not identify header/separator in Markdown table")
                return None

        # Count columns from separator
        total_cols = separator_row.count('|') - 1  # -1 because |---|---| has n+1 pipes for n columns

        # Data rows are all rows after separator
        data_rows = lines[separator_idx + 1:]

        # Construct header text (header + separator) for restoration in each chunk
        header_text = f"{header_row}\n{separator_row}"
        header_size = len(header_text) + 1  # +1 for newline

        return ParsedMarkdownTable(
            header_row=header_row,
            separator_row=separator_row,
            data_rows=data_rows,
            total_cols=total_cols,
            original_text=table_text,
            header_text=header_text,
            header_size=header_size
        )

    except Exception as e:
        logger.warning(f"Failed to parse Markdown table: {e}")
        return None


def build_markdown_table_chunk(
    header_text: str,
    data_rows: List[str],
    chunk_index: int = 0,
    total_chunks: int = 1,
    context_prefix: str = ""
) -> str:
    """
    Build a complete Markdown table chunk with header restored.

    Args:
        header_text: Header row + separator row
        data_rows: List of data row strings
        chunk_index: Current chunk index (0-based)
        total_chunks: Total number of chunks
        context_prefix: Context info (metadata, sheet info, etc.) - included in all chunks

    Returns:
        Complete Markdown table chunk
    """
    parts = []

    # Add context prefix if provided
    if context_prefix:
        parts.append(context_prefix)

    # Add chunk index metadata (only if more than 1 chunk)
    if total_chunks > 1:
        parts.append(f"[Table Chunk {chunk_index + 1}/{total_chunks}]")

    # Add header (header row + separator row)
    parts.append(header_text)

    # Add data rows
    for row in data_rows:
        parts.append(row)

    return "\n".join(parts)


def update_markdown_chunk_metadata(chunks: List[str], total_chunks: int) -> List[str]:
    """
    Update chunk metadata (total chunk count) in Markdown table chunks.

    Args:
        chunks: List of chunks
        total_chunks: Actual total number of chunks

    Returns:
        Updated chunks with correct metadata
    """
    updated_chunks = []

    for idx, chunk in enumerate(chunks):
        # Pattern for existing metadata
        old_pattern = r'\[Table Chunk \d+/\d+\]'
        new_metadata = f"[Table Chunk {idx + 1}/{total_chunks}]"

        if re.search(old_pattern, chunk):
            updated_chunk = re.sub(old_pattern, new_metadata, chunk)
        else:
            # No metadata found - add it
            updated_chunk = f"{new_metadata}\n{chunk}"

        updated_chunks.append(updated_chunk)

    return updated_chunks


def split_markdown_table_into_chunks(
    parsed_table: ParsedMarkdownTable,
    chunk_size: int,
    chunk_overlap: int = 0,
    context_prefix: str = ""
) -> List[str]:
    """
    Split a parsed Markdown table into chunks that fit chunk_size.
    Each chunk is a complete Markdown table with headers restored.

    NOTE: Table chunking does NOT apply overlap.
    Data duplication degrades search quality, so overlap is intentionally excluded.

    Args:
        parsed_table: Parsed Markdown table information
        chunk_size: Maximum chunk size
        chunk_overlap: Not used (kept for compatibility)
        context_prefix: Context info (metadata, sheet info, etc.) - included in all chunks

    Returns:
        List of Markdown table chunk strings
    """
    data_rows = parsed_table.data_rows
    header_text = parsed_table.header_text
    header_size = parsed_table.header_size

    # Calculate context size
    context_size = len(context_prefix) + 2 if context_prefix else 0  # +2 for newline

    if not data_rows:
        # No data rows - return original
        if context_prefix:
            return [f"{context_prefix}\n{parsed_table.original_text}"]
        return [parsed_table.original_text]

    # Calculate available space per chunk
    # Overhead: chunk index metadata (~25 chars) + header + context
    estimated_chunks = 1
    total_data_size = sum(len(row) + 1 for row in data_rows)  # +1 for newline
    available_per_chunk = chunk_size - header_size - context_size - CHUNK_INDEX_OVERHEAD

    if available_per_chunk > 0:
        estimated_chunks = max(1, (total_data_size + available_per_chunk - 1) // available_per_chunk)

    # Recalculate with estimated chunks
    if estimated_chunks > 1:
        available_per_chunk = chunk_size - header_size - context_size - CHUNK_INDEX_OVERHEAD
    else:
        available_per_chunk = chunk_size - header_size - context_size

    # Maximum allowed chunk size (1.5x of chunk_size)
    max_chunk_data_size = int(chunk_size * 1.5) - header_size - context_size - CHUNK_INDEX_OVERHEAD

    chunks: List[str] = []
    current_rows: List[str] = []
    current_size = 0

    for row in data_rows:
        row_size = len(row) + 1  # +1 for newline

        # Check if adding this row exceeds available space
        if current_rows and (current_size + row_size > available_per_chunk):
            # Check if we can still fit within 1.5x limit
            if current_size + row_size <= max_chunk_data_size:
                # Still within 1.5x limit - add row to current chunk
                current_rows.append(row)
                current_size += row_size
            else:
                # Exceeds 1.5x limit - flush current chunk and start new one
                chunk_text = build_markdown_table_chunk(
                    header_text,
                    current_rows,
                    chunk_index=len(chunks),
                    total_chunks=estimated_chunks,
                    context_prefix=context_prefix
                )
                chunks.append(chunk_text)

                # Start new chunk with this row (minimum 1 row guaranteed)
                current_rows = [row]
                current_size = row_size
        else:
            # Row fits - add to current chunk
            current_rows.append(row)
            current_size += row_size

    # Handle last chunk
    if current_rows:
        chunk_text = build_markdown_table_chunk(
            header_text,
            current_rows,
            chunk_index=len(chunks),
            total_chunks=max(len(chunks) + 1, estimated_chunks),
            context_prefix=context_prefix
        )
        chunks.append(chunk_text)

    # Update total chunk count in metadata if different from estimate
    if len(chunks) != estimated_chunks and len(chunks) > 1:
        chunks = update_markdown_chunk_metadata(chunks, len(chunks))

    logger.info(f"Markdown table split into {len(chunks)} chunks (original: {len(parsed_table.original_text)} chars)")

    return chunks


def chunk_large_markdown_table(
    table_text: str,
    chunk_size: int,
    chunk_overlap: int,
    context_prefix: str = ""
) -> List[str]:
    """
    Split a large Markdown table to fit chunk_size.
    Restores table structure (header + separator) in each chunk.

    NOTE: Table chunking does NOT apply overlap.
    Data duplication degrades search quality, so overlap is intentionally excluded.

    Args:
        table_text: Markdown table text
        chunk_size: Maximum chunk size
        chunk_overlap: Not used (kept for compatibility)
        context_prefix: Context info (metadata, sheet info, etc.) - included in all chunks

    Returns:
        List of split Markdown table chunks
    """
    # Parse table
    parsed = parse_markdown_table(table_text)

    if not parsed:
        logger.warning("Failed to parse Markdown table, returning original")
        if context_prefix:
            return [f"{context_prefix}\n{table_text}"]
        return [table_text]

    # No need to split if table fits in chunk_size
    if len(table_text) + len(context_prefix) <= chunk_size:
        if context_prefix:
            return [f"{context_prefix}\n{table_text}"]
        return [table_text]

    # No need to split if no data rows
    if not parsed.data_rows:
        if context_prefix:
            return [f"{context_prefix}\n{table_text}"]
        return [table_text]

    # Split table into chunks
    chunks = split_markdown_table_into_chunks(parsed, chunk_size, chunk_overlap, context_prefix)

    return chunks


def is_markdown_table(text: str) -> bool:
    """
    Check if text is a Markdown table.

    A Markdown table has:
    - Lines starting with |
    - A separator line with |---|

    Args:
        text: Text to check

    Returns:
        True if text is a Markdown table
    """
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False

    # Check for | at start of lines and separator pattern
    has_pipe_rows = any(line.strip().startswith('|') for line in lines)
    has_separator = any('---' in line and '|' in line for line in lines)

    return has_pipe_rows and has_separator


# Note: detect_table_type and chunk_large_table_unified were removed because they
# were not referenced anywhere in the codebase and duplicated logic handled elsewhere
# (e.g., via _chunk_table_unified in chunking.py). Keeping a single authoritative
# implementation reduces the risk of divergent behavior.
