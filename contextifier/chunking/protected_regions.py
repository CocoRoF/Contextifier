# chunking_helper/protected_regions.py
"""
Protected Regions - Protected region detection and processing

Main Features:
- Detect protected regions that should not be split during chunking
- Split text while protecting protected regions
- Efficient handling of large tables (HTML and Markdown)
- Row-level chunking for tables with NO overlap
- Support for dynamic tag patterns from processors (Image, Chart, Page, Slide, Metadata)
- Protected regions NEVER overlap when splitting chunks
"""
import logging
import re
from typing import Any, List, Optional, Tuple

from contextifier.chunking.constants import (
    HTML_TABLE_PATTERN, CHART_BLOCK_PATTERN, TEXTBOX_BLOCK_PATTERN,
    IMAGE_TAG_PATTERN, MARKDOWN_TABLE_PATTERN,
    PAGE_TAG_PATTERN, SLIDE_TAG_PATTERN, SHEET_TAG_PATTERN,
    PAGE_TAG_OCR_PATTERN, SLIDE_TAG_OCR_PATTERN,
    METADATA_BLOCK_PATTERN, DATA_ANALYSIS_PATTERN
)

logger = logging.getLogger("document-processor")


def find_protected_regions(
    text: str,
    is_table_based: bool = False,
    force_chunking: bool = False,
    image_pattern: Optional[str] = None,
    chart_pattern: Optional[str] = None,
    page_tag_processor: Optional[Any] = None,
    metadata_pattern: Optional[str] = None
) -> List[Tuple[int, int, str]]:
    """
    Find protected regions that should not be split during chunking.

    Protected Regions (NEVER split or overlap):
    1. HTML tables: <table>...</table> (row-level only when force_chunking/table-based)
    2. Chart blocks: [chart]...[/chart] or custom - always protected (never split)
    3. Textbox blocks: [textbox]...[/textbox] - always protected (never split)
    4. Image tags: [image:...] or custom - always protected (never split, no overlap)
    5. Markdown tables: |...|\\n|---|...| (row-level only when force_chunking/table-based)
    6. Page/Slide/Sheet tags: [Page Number: n], [Slide Number: n], [Sheet: name] - always protected (no overlap)
    7. Metadata blocks: <Document-Metadata>...</Document-Metadata> or custom - always protected (no overlap)
    8. Data analysis blocks: [Data Analysis]...[/Data Analysis] - always protected

    Args:
        text: Text to search
        is_table_based: Whether file is table-based (if True, row-level protection only for tables)
        force_chunking: Force chunking mode (if True, same as table-based for row-level protection)
        image_pattern: Image tag pattern (if None, uses default IMAGE_TAG_PATTERN)
        chart_pattern: Chart block pattern (if None, uses default CHART_BLOCK_PATTERN)
        page_tag_processor: PageTagProcessor instance for custom page/slide/sheet patterns
        metadata_pattern: Metadata block pattern (if None, uses default METADATA_BLOCK_PATTERN)

    Returns:
        [(start, end, type), ...] - Sorted list of protected regions
    """
    regions: List[Tuple[int, int, str]] = []

    # Table protection disabled when is_table_based or force_chunking is True
    disable_table_protection = is_table_based or force_chunking

    # 1. HTML tables (row-level only when table protection disabled)
    if not disable_table_protection:
        for match in re.finditer(HTML_TABLE_PATTERN, text, re.DOTALL | re.IGNORECASE):
            regions.append((match.start(), match.end(), 'html_table'))
    # else: HTML tables allow row-level chunking (handled by chunk_large_table)

    # 2. Chart blocks - always protected (never split under any condition)
    chart_pat = chart_pattern if chart_pattern is not None else CHART_BLOCK_PATTERN
    for match in re.finditer(chart_pat, text, re.DOTALL):
        regions.append((match.start(), match.end(), 'chart'))

    # 3. Textbox blocks - always protected (never split under any condition)
    for match in re.finditer(TEXTBOX_BLOCK_PATTERN, text, re.DOTALL):
        regions.append((match.start(), match.end(), 'textbox'))

    # 4. Image tags - always protected (never split under any condition, no overlap)
    img_pattern = image_pattern if image_pattern is not None else IMAGE_TAG_PATTERN
    for match in re.finditer(img_pattern, text):
        regions.append((match.start(), match.end(), 'image_tag'))

    # 5. Markdown tables (row-level only when table protection disabled)
    if not disable_table_protection:
        for match in re.finditer(MARKDOWN_TABLE_PATTERN, text, re.MULTILINE):
            table_start = match.start()
            if match.group(0).startswith('\n'):
                table_start += 1
            table_end = match.end()
            regions.append((table_start, table_end, 'markdown_table'))
    # else: Markdown tables allow row-level chunking (handled by chunk_large_markdown_table)

    # 6. Page/Slide/Sheet tags - always protected (no overlap)
    # Use dynamic patterns from PageTagProcessor if provided
    if page_tag_processor is not None:
        try:
            from contextifier.core.functions.page_tag_processor import PageTagType
            # Page tags
            page_pattern = page_tag_processor.get_pattern_string(PageTagType.PAGE)
            for match in re.finditer(page_pattern, text, re.IGNORECASE):
                regions.append((match.start(), match.end(), 'page_tag'))
            # OCR page tag variants (use stable default pattern)
            for match in re.finditer(PAGE_TAG_OCR_PATTERN, text, re.IGNORECASE):
                regions.append((match.start(), match.end(), 'page_tag'))
            # Slide tags
            slide_pattern = page_tag_processor.get_pattern_string(PageTagType.SLIDE)
            for match in re.finditer(slide_pattern, text, re.IGNORECASE):
                regions.append((match.start(), match.end(), 'slide_tag'))
            # OCR slide tag variants (use stable default pattern)
            for match in re.finditer(SLIDE_TAG_OCR_PATTERN, text, re.IGNORECASE):
                regions.append((match.start(), match.end(), 'slide_tag'))
            # Sheet tags
            sheet_pattern = page_tag_processor.get_pattern_string(PageTagType.SHEET)
            for match in re.finditer(sheet_pattern, text, re.IGNORECASE):
                regions.append((match.start(), match.end(), 'sheet_tag'))
        except Exception as e:
            logger.warning(f"Error getting patterns from page_tag_processor: {e}, using defaults")
            _add_default_page_tag_regions(text, regions)
    else:
        _add_default_page_tag_regions(text, regions)

    # 7. Metadata blocks - always protected (no overlap)
    meta_pattern = metadata_pattern if metadata_pattern is not None else METADATA_BLOCK_PATTERN
    for match in re.finditer(meta_pattern, text, re.DOTALL):
        regions.append((match.start(), match.end(), 'metadata'))

    # 8. Data analysis blocks - always protected
    for match in re.finditer(DATA_ANALYSIS_PATTERN, text, re.DOTALL):
        regions.append((match.start(), match.end(), 'data_analysis'))

    # Sort by start position
    regions.sort(key=lambda x: x[0])

    # Merge overlapping regions
    merged_regions: List[Tuple[int, int, str]] = []
    for start, end, region_type in regions:
        if merged_regions and start < merged_regions[-1][1]:
            # Overlap with previous region -> merge
            prev_start, prev_end, prev_type = merged_regions[-1]
            merged_regions[-1] = (prev_start, max(prev_end, end), f"{prev_type}+{region_type}")
        else:
            merged_regions.append((start, end, region_type))

    return merged_regions


def _add_default_page_tag_regions(text: str, regions: List[Tuple[int, int, str]]) -> None:
    """
    Add default page/slide/sheet tag regions using default patterns.
    
    Args:
        text: Text to search
        regions: List to append found regions to
    """
    # Page tags (including OCR variants)
    for match in re.finditer(PAGE_TAG_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'page_tag'))
    for match in re.finditer(PAGE_TAG_OCR_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'page_tag'))
    
    # Slide tags (including OCR variants)
    for match in re.finditer(SLIDE_TAG_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'slide_tag'))
    for match in re.finditer(SLIDE_TAG_OCR_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'slide_tag'))
    
    # Sheet tags
    for match in re.finditer(SHEET_TAG_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'sheet_tag'))


def _add_no_overlap_tag_regions_default(text: str, regions: List[Tuple[int, int, str]]) -> None:
    """
    Add default no-overlap tag regions (page/slide/sheet) using default patterns.
    
    Args:
        text: Text to search
        regions: List to append found regions to
    """
    # Page tags (including OCR variants)
    for match in re.finditer(PAGE_TAG_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'page_tag'))
    for match in re.finditer(PAGE_TAG_OCR_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'page_tag'))
    
    # Slide tags (including OCR variants)
    for match in re.finditer(SLIDE_TAG_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'slide_tag'))
    for match in re.finditer(SLIDE_TAG_OCR_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'slide_tag'))
    
    # Sheet tags
    for match in re.finditer(SHEET_TAG_PATTERN, text, re.IGNORECASE):
        regions.append((match.start(), match.end(), 'sheet_tag'))


def get_protected_region_positions(regions: List[Tuple[int, int, str]]) -> List[Tuple[int, int]]:
    """
    Extract (start, end) tuples from protected regions.
    """
    return [(start, end) for start, end, _ in regions]

def ensure_protected_region_integrity(content: str) -> str:
    """
    Verify that protected regions (HTML tables, chart blocks, Markdown tables) in chunk are complete.
    Log warning if incomplete protected region found (content is preserved).
    """
    # HTML table integrity check
    open_tables = len(re.findall(r'<table[^>]*>', content, re.IGNORECASE))
    close_tables = len(re.findall(r'</table>', content, re.IGNORECASE))
    if open_tables != close_tables:
        logger.warning(f"Incomplete HTML table detected in chunk: {open_tables} open, {close_tables} close tags")

    # Chart block integrity check
    open_charts = len(re.findall(r'\[chart\]', content))
    close_charts = len(re.findall(r'\[/chart\]', content))
    if open_charts != close_charts:
        logger.warning(f"Incomplete chart block detected in chunk: {open_charts} open, {close_charts} close tags")

    return content


def _is_markdown_table(text: str) -> bool:
    """
    Check if text contains a Markdown table pattern.
    """
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    has_pipe_rows = any(line.strip().startswith('|') for line in lines)
    has_separator = any('---' in line and '|' in line for line in lines)
    return has_pipe_rows and has_separator


def split_with_protected_regions(
    text: str,
    protected_regions: List[Tuple[int, int]],
    chunk_size: int,
    chunk_overlap: int,
    force_chunking: bool = False,
    image_pattern: Optional[str] = None,
    chart_pattern: Optional[str] = None,
    page_tag_processor: Optional[Any] = None,
    metadata_pattern: Optional[str] = None
) -> List[str]:
    """
    Split text into chunks while protecting regions (HTML tables, charts, Markdown tables, tags).

    Algorithm:
    1. Move forward by chunk_size from current position
    2. If that point is inside a protected region -> cut before region start or include until region end
    3. If protected region is larger than chunk_size:
       - HTML table -> split efficiently with chunk_large_table (row-level, NO overlap)
       - Markdown table -> split efficiently with chunk_large_markdown_table (row-level, NO overlap)
       - Other (charts, metadata, page tags, etc.) -> single chunk for protected region
    4. Apply overlap for next chunk start ONLY for plain text
       - Tables, images, charts, page/slide tags, metadata blocks: NO overlap

    Protected regions that NEVER overlap:
    - Image tags: [Image:...] or custom pattern
    - Page/Slide/Sheet tags: [Page Number: n], etc.
    - Chart blocks: [chart]...[/chart] or custom
    - Metadata blocks: <Document-Metadata>...</Document-Metadata> or custom
    - Tables: Split by rows, each chunk has NO overlap

    force_chunking handling:
    - When force_chunking=True, even if tables are not in protected_regions
    - Directly scan for HTML/Markdown tables to avoid cutting in the middle
    - Large tables are split by chunk_large_table/chunk_large_markdown_table with NO overlap

    Args:
        text: Text to split
        protected_regions: List of (start, end) tuples for protected regions
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size (NOT applied to protected regions)
        force_chunking: Force chunking mode
        image_pattern: Custom image tag pattern
        chart_pattern: Custom chart block pattern
        page_tag_processor: PageTagProcessor instance for custom patterns
        metadata_pattern: Custom metadata block pattern

    Returns:
        List of chunks
    """
    # Get image pattern (custom or default)
    img_pattern = image_pattern if image_pattern is not None else IMAGE_TAG_PATTERN
    
    # Extract image tag positions separately (to prevent mid-split and no overlap)
    image_regions = []
    for match in re.finditer(img_pattern, text):
        image_regions.append((match.start(), match.end()))
    
    # Extract all "no-overlap" tag regions (page, slide, sheet, chart, metadata)
    no_overlap_regions: List[Tuple[int, int, str]] = []
    
    # Page/Slide/Sheet tags
    if page_tag_processor is not None:
        try:
            from contextifier.core.functions.page_tag_processor import PageTagType
            for match in re.finditer(page_tag_processor.get_pattern_string(PageTagType.PAGE), text, re.IGNORECASE):
                no_overlap_regions.append((match.start(), match.end(), 'page_tag'))
            for match in re.finditer(page_tag_processor.get_pattern_string(PageTagType.SLIDE), text, re.IGNORECASE):
                no_overlap_regions.append((match.start(), match.end(), 'slide_tag'))
            for match in re.finditer(page_tag_processor.get_pattern_string(PageTagType.SHEET), text, re.IGNORECASE):
                no_overlap_regions.append((match.start(), match.end(), 'sheet_tag'))
        except Exception:
            _add_no_overlap_tag_regions_default(text, no_overlap_regions)
    else:
        _add_no_overlap_tag_regions_default(text, no_overlap_regions)
    
    # Chart blocks
    chart_pat = chart_pattern if chart_pattern is not None else CHART_BLOCK_PATTERN
    for match in re.finditer(chart_pat, text, re.DOTALL):
        no_overlap_regions.append((match.start(), match.end(), 'chart'))
    
    # Metadata blocks
    meta_pat = metadata_pattern if metadata_pattern is not None else METADATA_BLOCK_PATTERN
    for match in re.finditer(meta_pat, text, re.DOTALL):
        no_overlap_regions.append((match.start(), match.end(), 'metadata'))
    
    # Data analysis blocks
    for match in re.finditer(DATA_ANALYSIS_PATTERN, text, re.DOTALL):
        no_overlap_regions.append((match.start(), match.end(), 'data_analysis'))

    # Block protected regions (excluding images - handled separately)
    block_regions = []
    for t_start, t_end in protected_regions:
        is_image = False
        for img_start, img_end in image_regions:
            if t_start == img_start and t_end == img_end:
                is_image = True
                break
        if not is_image:
            block_regions.append((t_start, t_end))

    # When force_chunking, directly scan for HTML tables
    # (to handle tables not registered in protected_regions)
    html_table_regions = []
    markdown_table_regions = []
    
    if force_chunking:
        # Scan for HTML tables
        for match in re.finditer(HTML_TABLE_PATTERN, text, re.DOTALL | re.IGNORECASE):
            t_start, t_end = match.start(), match.end()
            # Check if already in block_regions
            already_in_block = any(
                bs <= t_start and be >= t_end
                for bs, be in block_regions
            )
            if not already_in_block:
                html_table_regions.append((t_start, t_end, 'html'))
        
        # Scan for Markdown tables
        for match in re.finditer(MARKDOWN_TABLE_PATTERN, text, re.MULTILINE):
            table_start = match.start()
            if match.group(0).startswith('\n'):
                table_start += 1
            t_start, t_end = table_start, match.end()
            # Check if already in block_regions
            already_in_block = any(
                bs <= t_start and be >= t_end
                for bs, be in block_regions
            )
            if not already_in_block:
                markdown_table_regions.append((t_start, t_end, 'markdown'))

    # Combine all block regions with type info
    # Convert existing block_regions to include type
    all_block_regions_with_type = [(s, e, 'block') for s, e in block_regions]
    all_block_regions_with_type.extend(html_table_regions)
    all_block_regions_with_type.extend(markdown_table_regions)
    
    # Sort by start position
    all_block_regions_with_type.sort(key=lambda x: x[0])
    
    # Extract just positions for compatibility
    all_block_regions = [(s, e) for s, e, _ in all_block_regions_with_type]
    
    # Create mapping from position to type
    region_type_map = {(s, e): t for s, e, t in all_block_regions_with_type}

    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        # If remaining text is <= chunk_size, it's the last chunk
        remaining = text_len - current_pos
        if remaining <= chunk_size:
            chunk = text[current_pos:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Calculate chunk_size endpoint
        tentative_end = current_pos + chunk_size

        # Check if there's a block protected region in this range
        block_in_range = None
        block_type = None
        for t_start, t_end in all_block_regions:
            if t_start < tentative_end and t_end > current_pos:
                block_in_range = (t_start, t_end)
                block_type = region_type_map.get((t_start, t_end), 'block')
                break

        if block_in_range:
            t_start, t_end = block_in_range
            table_size = t_end - t_start

            if t_start <= current_pos:
                # Current position is inside or at start of table/block
                if table_size > chunk_size:
                    # Table/block is larger than chunk_size
                    table_content = text[t_start:t_end].strip()

                    # Check type and split efficiently
                    if block_type == 'html' or table_content.startswith('<table'):
                        # HTML table - split by rows with NO overlap
                        from .table_chunker import chunk_large_table
                        table_chunks = chunk_large_table(table_content, chunk_size, 0, "")
                        chunks.extend(table_chunks)
                    elif block_type == 'markdown' or _is_markdown_table(table_content):
                        # Markdown table - split by rows with NO overlap
                        from .table_chunker import chunk_large_markdown_table
                        table_chunks = chunk_large_markdown_table(table_content, chunk_size, 0, "")
                        chunks.extend(table_chunks)
                    else:
                        # Charts, textboxes, etc. -> single chunk (never split)
                        if table_content:
                            chunks.append(table_content)

                    # Protected blocks have NO overlap - move to end
                    current_pos = t_end
                else:
                    # Table fits in chunk_size -> try to include table + text after
                    end_pos = min(t_end + (chunk_size - table_size), text_len)

                    # Check for collision with next block region (excluding images)
                    for next_t_start, next_t_end in all_block_regions:
                        if next_t_start > t_end and next_t_start < end_pos:
                            end_pos = next_t_start
                            break

                    # Adjust if end_pos is in the middle of an image or protected tag
                    end_pos, ends_with_image = _adjust_for_image_boundary(end_pos, image_regions, text_len)
                    ends_with_no_overlap = _check_ends_with_no_overlap_region(end_pos, no_overlap_regions)

                    chunk = text[current_pos:end_pos].strip()
                    if chunk:
                        chunks.append(chunk)

                    # Determine if this chunk contains a table (for overlap decision)
                    chunk_has_table = (block_type in ('html', 'markdown') or 
                                      text[t_start:t_end].strip().startswith('<table') or
                                      _is_markdown_table(text[t_start:t_end]))
                    
                    # NO overlap for: tables, images, page/slide tags, charts, metadata
                    if ends_with_image or ends_with_no_overlap or chunk_has_table:
                        current_pos = end_pos
                    else:
                        current_pos = max(t_end, end_pos - chunk_overlap)
            else:
                # Table is in the middle of potential chunk
                space_before_table = t_start - current_pos
                space_with_table = t_end - current_pos

                if space_with_table <= chunk_size:
                    # Can include entire table -> include up to table end
                    end_pos = t_end

                    # Check if we can add text after table with remaining space
                    remaining_space = chunk_size - space_with_table
                    if remaining_space > 0:
                        potential_end = min(t_end + remaining_space, text_len)

                        # Check for collision with next block region (excluding images)
                        for next_t_start, next_t_end in all_block_regions:
                            if next_t_start > t_end and next_t_start < potential_end:
                                potential_end = next_t_start
                                break

                        end_pos = potential_end

                    # Adjust if end_pos is in the middle of an image or protected tag
                    end_pos, ends_with_image = _adjust_for_image_boundary(end_pos, image_regions, text_len)
                    ends_with_no_overlap = _check_ends_with_no_overlap_region(end_pos, no_overlap_regions)

                    chunk = text[current_pos:end_pos].strip()
                    if chunk:
                        chunks.append(chunk)

                    # Determine if this chunk ends with a table
                    chunk_ends_with_table = (end_pos == t_end or 
                                            (block_type in ('html', 'markdown')))
                    
                    # NO overlap for: tables, images, page/slide tags, charts, metadata
                    if ends_with_image or ends_with_no_overlap or chunk_ends_with_table:
                        current_pos = end_pos
                    else:
                        current_pos = max(t_end, end_pos - chunk_overlap)
                else:
                    # Cannot include entire table
                    if space_before_table > chunk_overlap:
                        # Split text before table first
                        end_pos = t_start
                        # Adjust if end_pos is in the middle of an image or protected tag
                        end_pos, ends_with_image = _adjust_for_image_boundary(end_pos, image_regions, text_len)
                        ends_with_no_overlap = _check_ends_with_no_overlap_region(end_pos, no_overlap_regions)

                        chunk = text[current_pos:end_pos].strip()
                        if chunk:
                            chunks.append(chunk)

                        # NO overlap for: images, page/slide tags, charts, metadata
                        if ends_with_image or ends_with_no_overlap:
                            current_pos = end_pos
                        else:
                            current_pos = max(current_pos + 1, t_start - chunk_overlap)
                    else:
                        # Space before table too small -> handle table
                        table_content = text[t_start:t_end].strip()

                        # Split table if larger than chunk_size
                        if table_size > chunk_size:
                            if block_type == 'html' or table_content.startswith('<table'):
                                # HTML table - split by rows with NO overlap
                                from .table_chunker import chunk_large_table
                                table_chunks = chunk_large_table(table_content, chunk_size, 0, "")
                                chunks.extend(table_chunks)
                            elif block_type == 'markdown' or _is_markdown_table(table_content):
                                # Markdown table - split by rows with NO overlap
                                from .table_chunker import chunk_large_markdown_table
                                table_chunks = chunk_large_markdown_table(table_content, chunk_size, 0, "")
                                chunks.extend(table_chunks)
                            else:
                                # Charts, textboxes, etc. -> single chunk
                                if table_content:
                                    chunks.append(table_content)
                        else:
                            # Table or chart as single chunk
                            if table_content:
                                chunks.append(table_content)
                        # Tables have NO overlap
                        current_pos = t_end
        else:
            # No block protected region -> find best split point
            best_split = tentative_end

            # Look for paragraph separator
            search_start = max(current_pos, tentative_end - 200)
            para_match = None
            for m in re.finditer(r'\n\s*\n', text[search_start:tentative_end]):
                para_match = m

            if para_match:
                best_split = search_start + para_match.end()
            else:
                # Look for newline
                newline_pos = text.rfind('\n', current_pos, tentative_end)
                if newline_pos > current_pos + chunk_size // 2:
                    best_split = newline_pos + 1
                else:
                    # Look for space
                    space_pos = text.rfind(' ', current_pos, tentative_end)
                    if space_pos > current_pos + chunk_size // 2:
                        best_split = space_pos + 1

            # Adjust if best_split is in the middle of an image or protected tag
            best_split, ends_with_image = _adjust_for_image_boundary(best_split, image_regions, text_len)
            ends_with_no_overlap = _check_ends_with_no_overlap_region(best_split, no_overlap_regions)

            chunk = text[current_pos:best_split].strip()
            if chunk:
                chunks.append(chunk)

            # NO overlap for: images, page/slide tags, charts, metadata
            if ends_with_image or ends_with_no_overlap:
                current_pos = best_split
            else:
                current_pos = best_split - chunk_overlap
                if current_pos < 0:
                    current_pos = best_split

    return chunks


def _adjust_for_image_boundary(
    pos: int,
    image_regions: List[Tuple[int, int]],
    text_len: int
) -> Tuple[int, bool]:
    """
    Check if position is in the middle of an image tag and adjust to image end if so.

    Args:
        pos: Current split position
        image_regions: Image tag position list [(start, end), ...]
        text_len: Total text length

    Returns:
        (adjusted_pos, ends_with_image): Adjusted position and whether it ends with image
    """
    for img_start, img_end in image_regions:
        # If split position is in the middle of an image tag
        if img_start < pos < img_end:
            # Extend to image end
            return min(img_end, text_len), True
        # If split position is right after an image tag (including space/newline)
        if img_end <= pos <= img_end + 5:
            return pos, True
    return pos, False


def _check_ends_with_no_overlap_region(
    end_pos: int,
    no_overlap_regions: List[Tuple[int, int, str]],
    tolerance: int = 5
) -> bool:
    """
    Check if position ends with or is right after a no-overlap region.

    Args:
        end_pos: End position of chunk
        no_overlap_regions: List of (start, end, type) for no-overlap regions
        tolerance: Number of characters after region end to still consider it as ending with region

    Returns:
        True if position ends with a no-overlap region
    """
    for region_start, region_end, _ in no_overlap_regions:
        # If end_pos is exactly at or just after the region end (within tolerance)
        if region_end <= end_pos <= region_end + tolerance:
            return True
    return False


def split_large_chunk_with_protected_regions(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    is_table_based: bool = False,
    force_chunking: bool = False,
    image_pattern: Optional[str] = None,
    chart_pattern: Optional[str] = None,
    page_tag_processor: Optional[Any] = None,
    metadata_pattern: Optional[str] = None
) -> List[str]:
    """
    Split large chunk while protecting regions (HTML tables, charts, Markdown tables, tags).
    When force_chunking, table protection is disabled (charts always protected, rows protected).

    Protected regions that NEVER overlap:
    - Image tags, Page/Slide/Sheet tags, Chart blocks, Metadata blocks
    - Tables split by rows with NO overlap

    When force_chunking=True:
    - Tables are not registered as protected regions in find_protected_regions
    - But split_with_protected_regions directly scans for tables and handles them
    - Tables are split by rows with NO overlap

    Args:
        text: Text to split
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks (NOT applied to protected regions)
        is_table_based: Whether file is table-based
        force_chunking: Force chunking mode
        image_pattern: Custom image tag pattern
        chart_pattern: Custom chart block pattern
        page_tag_processor: PageTagProcessor instance for custom page/slide/sheet patterns
        metadata_pattern: Custom metadata block pattern

    Returns:
        List of chunks
    """
    protected_regions = find_protected_regions(
        text, is_table_based, force_chunking, image_pattern,
        chart_pattern, page_tag_processor, metadata_pattern
    )
    protected_positions = get_protected_region_positions(protected_regions)

    # split_with_protected_regions handles tables even with force_chunking
    # (it directly scans for tables when force_chunking=True)
    return split_with_protected_regions(
        text, protected_positions, chunk_size, chunk_overlap, force_chunking,
        image_pattern, chart_pattern, page_tag_processor, metadata_pattern
    )


# Backward compatibility aliases
def ensure_table_integrity(content: str, table_pattern: str) -> str:
    """Deprecated: Use ensure_protected_region_integrity instead."""
    return ensure_protected_region_integrity(content)


def split_large_chunk_with_table_protection(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """Deprecated: Use split_large_chunk_with_protected_regions instead."""
    return split_large_chunk_with_protected_regions(text, chunk_size, chunk_overlap, False)
