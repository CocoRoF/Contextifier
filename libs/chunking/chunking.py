# libs/chunking/chunking.py
"""
Document Chunking Module - Advanced Text Chunking System

Main Features:
- HTML table-preserving chunking
- Intelligent splitting for large table data (CSV/Excel)
- Table structure restoration (header preservation)
- Page-based chunking
- Language-specific code file chunking

Key Improvements (Table Chunking Enhancement):
- Split large tables by rows to fit chunk_size
- Automatically restore table headers in each chunk
- Ensure table structure integrity
- Add chunk indexing metadata

Refactoring:
- Core logic is separated into chunking_helper submodules
- This file maintains only the public API and integration logic
"""
import bisect
import logging
import re
from typing import Any, Dict, List, Optional, Union

# Import from individual modules
from libs.chunking.constants import (
    LANGCHAIN_CODE_LANGUAGE_MAP,
    HTML_TABLE_PATTERN,
    CHART_BLOCK_PATTERN,
    TEXTBOX_BLOCK_PATTERN,
    IMAGE_TAG_PATTERN,
    MARKDOWN_TABLE_PATTERN,
    TABLE_WRAPPER_OVERHEAD,
    CHUNK_INDEX_OVERHEAD,
    TABLE_SIZE_THRESHOLD_MULTIPLIER,
    TABLE_BASED_FILE_TYPES,
    TableRow,
    ParsedTable,
)

from libs.chunking.table_parser import (
    parse_html_table as _parse_html_table,
    extract_cell_spans as _extract_cell_spans,
    extract_cell_spans_with_positions as _extract_cell_spans_with_positions,
    has_complex_spans as _has_complex_spans,
)

from libs.chunking.table_chunker import (
    calculate_available_space as _calculate_available_space,
    adjust_rowspan_in_chunk as _adjust_rowspan_in_chunk,
    build_table_chunk as _build_table_chunk,
    update_chunk_metadata as _update_chunk_metadata,
    split_table_into_chunks as _split_table_into_chunks,
    split_table_preserving_rowspan as _split_table_preserving_rowspan,
    chunk_large_table as _chunk_large_table,
)

from libs.chunking.protected_regions import (
    find_protected_regions as _find_protected_regions,
    get_protected_region_positions as _get_protected_region_positions,
    ensure_protected_region_integrity as _ensure_protected_region_integrity,
    split_with_protected_regions as _split_with_protected_regions,
    split_large_chunk_with_protected_regions as _split_large_chunk_with_protected_regions,
    ensure_table_integrity as _ensure_table_integrity,
    split_large_chunk_with_table_protection as _split_large_chunk_with_table_protection,
)

from libs.chunking.page_chunker import (
    split_into_pages as _split_into_pages,
    merge_pages as _merge_pages,
    get_overlap_content as _get_overlap_content,
    chunk_by_pages as _chunk_by_pages,
)

from libs.chunking.text_chunker import (
    chunk_plain_text as _chunk_plain_text,
    chunk_text_without_tables,
    chunk_with_row_protection,
    chunk_with_row_protection_simple,
    clean_chunks as _clean_chunks,
    chunk_code_text,
    reconstruct_text_from_chunks,
    find_overlap_length,
    estimate_chunks_count,
)

from libs.chunking.sheet_processor import (
    extract_document_metadata as _extract_document_metadata,
    prepend_metadata_to_chunks as _prepend_metadata_to_chunks,
    extract_sheet_sections as _extract_sheet_sections,
    extract_content_segments as _extract_content_segments,
    chunk_multi_sheet_content,
    chunk_single_table_content,
)

logger = logging.getLogger("document-processor")


# ============================================================================
# Helper Functions for PageTagProcessor integration
# ============================================================================

def _get_page_marker_patterns(page_tag_processor: Optional[Any] = None) -> List[str]:
    """
    Get page marker regex patterns from PageTagProcessor or use defaults.
    
    Args:
        page_tag_processor: PageTagProcessor instance (optional)
        
    Returns:
        List of regex patterns for page/slide markers
    """
    if page_tag_processor is not None:
        # Build patterns from processor's config
        config = page_tag_processor.config
        patterns = [
            page_tag_processor.get_pattern_string(),  # Page pattern
        ]
        # Add slide pattern if different prefix
        if config.slide_prefix != config.tag_prefix:
            from libs.core.functions.page_tag_processor import PageTagType
            patterns.append(page_tag_processor.get_pattern_string(PageTagType.SLIDE))
        return patterns
    else:
        # Default patterns
        return [
            r'\[Page Number:\s*(\d+)\]',
            r'\[Slide Number:\s*(\d+)\]',
        ]


def _get_sheet_marker_pattern(page_tag_processor: Optional[Any] = None) -> str:
    """
    Get sheet marker regex pattern from PageTagProcessor or use default.
    
    Args:
        page_tag_processor: PageTagProcessor instance (optional)
        
    Returns:
        Regex pattern for sheet markers
    """
    if page_tag_processor is not None:
        from libs.core.functions.page_tag_processor import PageTagType
        return page_tag_processor.get_pattern_string(PageTagType.SHEET)
    else:
        return r'\[Sheet:\s*([^\]]+)\]'


# ============================================================================
# Public API - Single entry point for external use
# ============================================================================

def create_chunks(
    text: str,
    file_extension: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_chunking: bool = False,
    include_position_metadata: bool = True,
    chunking_strategy: str = "recursive",
    page_tag_processor: Optional[Any] = None,
    stride: Optional[int] = None,
    parent_chunk_size: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    **kwargs
) -> Union[List[str], List[Dict[str, Any]]]:
    """
    Split text into chunks. (Single public API)

    Args:
        text: Original text
        file_extension: File extension
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks
        force_chunking: Force chunking (disable table protection)
        include_position_metadata: Whether to include position metadata
            - True: Include metadata like page_number, line_start, line_end (List[Dict])
            - False: Return only chunk text (List[str])
        chunking_strategy: Chunking strategy (recursive, sliding, hierarchical) - future implementation
        page_tag_processor: PageTagProcessor instance for custom tag patterns
            - If None, uses default patterns [Page Number: n], [Slide Number: n], [Sheet: name]
            - If provided, uses the processor's configured patterns
        stride: Stride for sliding window strategy - future implementation
        parent_chunk_size: Parent chunk size for hierarchical strategy - future implementation
        child_chunk_size: Child chunk size for hierarchical strategy - future implementation

    Returns:
        When include_position_metadata=True:
            List of chunks with metadata [{"text", "page_number", "line_start", ...}, ...]
        When include_position_metadata=False:
            List of chunk texts ["chunk1", "chunk2", ...]
    """
    # TODO: Implement various chunking strategies based on chunking_strategy
    if chunking_strategy != "recursive":
        logger.warning(
            f"Chunking strategy '{chunking_strategy}' is not yet implemented, "
            "falling back to 'recursive'"
        )

    # Split text into chunks
    chunks = _split_text(
        text, chunk_size, chunk_overlap,
        file_extension=file_extension,
        force_chunking=force_chunking,
        page_tag_processor=page_tag_processor
    )

    # Return chunks without metadata
    if not include_position_metadata:
        return chunks

    # Reconstruct text and build line offset table
    reconstructed = reconstruct_text_from_chunks(chunks, chunk_overlap)
    line_table = _build_line_offset_table(reconstructed, file_extension, page_tag_processor)

    # Add metadata to each chunk
    result: List[Dict[str, Any]] = []
    current_pos = 0

    for idx, chunk in enumerate(chunks):
        start = current_pos
        end = current_pos + len(chunk) - 1

        start_line_idx = _find_line_index_by_pos(start, line_table)
        end_line_idx = _find_line_index_by_pos(end, line_table)

        line_start = line_table[start_line_idx]["line_num"]
        line_end = line_table[end_line_idx]["line_num"]
        page_number = line_table[start_line_idx].get("page", 1)

        result.append({
            "text": chunk,
            "page_number": page_number,
            "line_start": line_start,
            "line_end": line_end,
            "global_start": start,
            "global_end": end,
            "chunk_index": idx
        })

        current_pos += len(chunk)
        if idx < len(chunks) - 1:
            overlap_len = find_overlap_length(chunk, chunks[idx + 1], chunk_overlap)
            current_pos -= overlap_len

    logger.info(f"Created {len(result)} chunks with position metadata")
    return result


# ============================================================================
# Internal Functions - Table-based content processing
# ============================================================================

def _split_table_based_content(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Chunk table-based content (CSV/Excel).

    Split large tables to fit chunk_size and restore table structure
    in each chunk.

    For multi-sheet Excel files, process each sheet separately.

    Args:
        text: Full text (metadata + table)
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks

    Returns:
        List of chunks
    """
    if not text or not text.strip():
        return [""]

    # Extract metadata
    metadata_block, text_without_metadata = _extract_document_metadata(text)

    # Extract data analysis block
    analysis_pattern = r'(\[Data Analysis\].*?\[/Data Analysis\]|\[데이터 분석\].*?\[/데이터 분석\])\s*'
    analysis_match = re.search(analysis_pattern, text_without_metadata, re.DOTALL)
    analysis_block = ""

    if analysis_match:
        analysis_block = analysis_match.group(1)
        text_without_analysis = (
            text_without_metadata[:analysis_match.start()] +
            text_without_metadata[analysis_match.end():]
        ).strip()
    else:
        text_without_analysis = text_without_metadata

    # Check for multi-sheet (Excel)
    sheets = _extract_sheet_sections(text_without_analysis)

    if sheets:
        logger.info(f"Multi-sheet Excel detected: {len(sheets)} sheets")
        return chunk_multi_sheet_content(
            sheets, metadata_block, analysis_block, chunk_size, chunk_overlap,
            _chunk_plain_text, _chunk_large_table
        )

    # Single table/sheet processing
    return chunk_single_table_content(
        text_without_analysis, metadata_block, analysis_block, chunk_size, chunk_overlap,
        _chunk_plain_text, _chunk_large_table
    )


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    file_extension: Optional[str] = None,
    force_chunking: Optional[bool] = False,
    page_tag_processor: Optional[Any] = None
) -> List[str]:
    """
    Split text into chunks. (Internal use)

    Preserves HTML tables and considers page boundaries for chunking.

    Core Strategy:
    1. Apply table-based chunking if file_extension is CSV/Excel
    2. Apply page-based chunking first if page markers exist
    3. Merge pages based on chunk_size (allow up to 1.5x)
    4. Never cut in the middle of a table
    5. Apply overlap normally

    Args:
        text: Original text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks
        file_extension: File extension (csv, xlsx, pdf, etc.) - used for table-based processing
        force_chunking: Force chunking (disable table protection except for table-based files)
        page_tag_processor: PageTagProcessor instance for custom tag patterns

    Returns:
        List of chunks
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return [""]

    # === Check for table-based content (CSV/Excel files only) ===
    # Explicitly determine based on file_extension (no text content guessing)
    is_table_based = file_extension and file_extension.lower() in TABLE_BASED_FILE_TYPES

    # Disable table protection if is_table_based or force_chunking is True
    disable_table_protection = is_table_based or force_chunking

    if is_table_based:
        # Check for large tables
        table_pattern = r'<table\s+border=["\']1["\']>.*?</table>'
        table_matches = list(re.finditer(table_pattern, text, re.DOTALL | re.IGNORECASE))

        # Need to split if table is larger than chunk_size
        has_large_table = any(
            (m.end() - m.start()) > chunk_size * TABLE_SIZE_THRESHOLD_MULTIPLIER
            for m in table_matches
        )

        if has_large_table:
            logger.info(f"Large table detected in {file_extension} file, using table-based chunking")
            return _split_table_based_content(text, chunk_size, chunk_overlap)

    # Extract metadata
    metadata_block, text_without_metadata = _extract_document_metadata(text)
    text = text_without_metadata

    # === Check for page markers ===
    # Build patterns from PageTagProcessor or use defaults
    page_marker_patterns = _get_page_marker_patterns(page_tag_processor)
    has_page_markers = any(re.search(pattern, text) for pattern in page_marker_patterns)

    if has_page_markers:
        # Page-based chunking
        logger.debug("Page markers found, using page-based chunking")
        chunks = _chunk_by_pages(text, chunk_size, chunk_overlap, is_table_based, force_chunking, page_tag_processor)
    else:
        # Find protected regions (HTML tables, chart blocks, Markdown tables)
        # Disable table protection on force_chunking (charts are always protected)
        protected_regions = _find_protected_regions(text, is_table_based, force_chunking)
        protected_positions = _get_protected_region_positions(protected_regions)

        if protected_positions:
            region_types = set(r[2] for r in protected_regions)
            logger.info(f"Found {len(protected_positions)} protected regions in document: {region_types}")
            chunks = _split_with_protected_regions(
                text, protected_positions, chunk_size, chunk_overlap, force_chunking
            )
        else:
            # No protected regions: apply row-level chunking if force_chunking
            if disable_table_protection:
                logger.debug("Force chunking enabled, using row-preserving chunking")
                chunks = _chunk_with_row_protection(text, chunk_size, chunk_overlap, force_chunking)
            else:
                logger.debug("No protected blocks found, using standard chunking")
                return _chunk_text_without_tables(text, chunk_size, chunk_overlap, metadata_block)

    # Clean chunks
    cleaned_chunks = _clean_chunks(chunks)

    # Add metadata
    cleaned_chunks = _prepend_metadata_to_chunks(cleaned_chunks, metadata_block)

    logger.info(f"Final text split into {len(cleaned_chunks)} chunks")

    return cleaned_chunks


def _is_table_based_file_type(file_extension: Optional[str]) -> bool:
    """
    Check if the file extension is a table-based file type. (Internal use)

    Args:
        file_extension: File extension

    Returns:
        True if table-based file type
    """
    if not file_extension:
        return False
    return file_extension.lower() in TABLE_BASED_FILE_TYPES


# ============================================================================
# Internal Wrapper Functions
# ============================================================================

def _chunk_text_without_tables(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[str]
) -> List[str]:
    """
    Chunking logic for text without tables.
    Wrapper function for chunk_text_without_tables.
    """
    return chunk_text_without_tables(
        text, chunk_size, chunk_overlap, metadata,
        _prepend_metadata_to_chunks
    )


def _chunk_with_row_protection(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    force_chunking: bool = False
) -> List[str]:
    """
    Chunk while protecting row boundaries when table protection is disabled.
    Wrapper function for chunk_with_row_protection.
    """
    # Wrapper function to pass force_chunking
    def split_with_protected_regions_wrapper(text, regions, chunk_size, chunk_overlap):
        return _split_with_protected_regions(text, regions, chunk_size, chunk_overlap, force_chunking)

    return chunk_with_row_protection(
        text, chunk_size, chunk_overlap,
        split_with_protected_regions_wrapper, _chunk_large_table
    )


# ============================================================================
# Internal Functions - Page/Line Mapping
# ============================================================================

def _extract_page_mapping(
    text: str,
    file_extension: str,
    page_tag_processor: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Extract page/slide mapping information from text.

    Recognizes page markers for various file formats:
    - PDF/PPT/DOCX: Page/slide markers
    - Excel: Sheet markers
    - Others: Line-based estimation

    Args:
        text: Original text
        file_extension: File extension
        page_tag_processor: PageTagProcessor instance for custom patterns

    Returns:
        Page mapping list [{"page_num": int, "start_pos": int, "end_pos": int, ...}, ...]
    """
    try:
        page_mapping: List[Dict[str, Any]] = []
        ext_lower = file_extension.lower() if file_extension else ""

        if ext_lower in ['pdf', 'ppt', 'pptx', 'doc', 'docx']:
            # Build patterns from PageTagProcessor or use defaults
            patterns = _get_page_marker_patterns(page_tag_processor)
            # Add OCR variants
            ocr_patterns = []
            for p in patterns:
                # Add (OCR) and (OCR+Ref) variants
                base_pattern = p.rstrip(']').rstrip(')')
                if base_pattern.endswith('\\d+'):
                    ocr_patterns.append(p[:-1] + r'\s*\(OCR\)\]')
                    ocr_patterns.append(p[:-1] + r'\s*\(OCR\+Ref\)\]')
            patterns.extend(ocr_patterns)

            for pattern in patterns:
                matches = list(re.finditer(pattern, text))
                if matches:
                    for i, match in enumerate(matches):
                        page_num = int(match.group(1))
                        start = match.end()
                        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                        page_mapping.append({
                            "page_num": page_num,
                            "start_pos": start,
                            "end_pos": end
                        })
                    page_mapping.sort(key=lambda x: x["page_num"])
                    break

            # Estimate pages for doc/docx if no markers found
            if not page_mapping and ext_lower in ['doc', 'docx']:
                chars_per_page = 1500
                text_len = len(text)
                if text_len > chars_per_page:
                    estimated_pages = (text_len + chars_per_page - 1) // chars_per_page
                    for page_num in range(1, estimated_pages + 1):
                        start = (page_num - 1) * chars_per_page
                        end = min(page_num * chars_per_page, text_len)
                        page_mapping.append({
                            "page_num": page_num,
                            "start_pos": start,
                            "end_pos": end
                        })

            if not page_mapping:
                page_mapping = [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]

        elif ext_lower in ['xlsx', 'xls']:
            # Build sheet pattern from PageTagProcessor or use default
            sheet_pattern = _get_sheet_marker_pattern(page_tag_processor)
            matches = list(re.finditer(sheet_pattern, text))

            if matches:
                for i, match in enumerate(matches):
                    start = match.end()
                    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    page_mapping.append({
                        "page_num": i + 1,
                        "start_pos": start,
                        "end_pos": end,
                        "sheet_name": match.group(1).strip()
                    })
            else:
                page_mapping = [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]

        else:
            # Line-based estimation for other file types
            lines = text.split('\n')
            lines_per_page = 1000

            if len(lines) > lines_per_page:
                page_count = (len(lines) + lines_per_page - 1) // lines_per_page
                current_pos = 0

                for page_num in range(1, page_count + 1):
                    start_line = (page_num - 1) * lines_per_page
                    end_line = min(page_num * lines_per_page, len(lines))
                    page_text = '\n'.join(lines[start_line:end_line])
                    start = current_pos
                    end = current_pos + len(page_text)
                    page_mapping.append({
                        "page_num": page_num,
                        "start_pos": start,
                        "end_pos": end
                    })
                    current_pos = end + 1
            else:
                page_mapping = [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]

        return page_mapping

    except Exception:
        return [{"page_num": 1, "start_pos": 0, "end_pos": len(text)}]


def _find_line_index_by_pos(pos: int, line_table: List[Dict[str, int]]) -> int:
    """
    Find the line index corresponding to the given position.

    Args:
        pos: Position in text
        line_table: Line offset table

    Returns:
        Line index (0-based)
    """
    try:
        if not line_table:
            return 0
        starts = [line["start"] for line in line_table]
        idx = bisect.bisect_right(starts, pos) - 1
        return 0 if idx < 0 else min(idx, len(line_table) - 1)
    except Exception:
        return 0


def _build_line_offset_table(
    text: str,
    file_extension: str,
    page_tag_processor: Optional[Any] = None
) -> List[Dict[str, int]]:
    """
    Build an offset table for each line in the text.

    Args:
        text: Original text
        file_extension: File extension
        page_tag_processor: PageTagProcessor instance for custom patterns

    Returns:
        Line offset table [{"line_num": int, "start": int, "end": int, "page": int}, ...]
    """
    try:
        lines = text.split('\n')
        table: List[Dict[str, int]] = []
        pos = 0
        page_mapping = _extract_page_mapping(text, file_extension, page_tag_processor)

        def _page_for_pos(p: int) -> int:
            for info in page_mapping:
                if info["start_pos"] <= p < info["end_pos"]:
                    return info["page_num"]
            return 1

        for i, line in enumerate(lines):
            start = pos
            end = pos + len(line)
            mid = start + max(0, (end - start) // 2)
            page = _page_for_pos(mid)
            table.append({
                "line_num": i + 1,
                "start": start,
                "end": end,
                "page": page
            })
            pos = end + 1

        return table

    except Exception:
        return [{"line_num": 1, "start": 0, "end": len(text), "page": 1}]


# ============================================================================
# Backward Compatibility - Deprecated public functions
# ============================================================================

def split_table_based_content(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Chunk table-based content (CSV/Excel).

    .. deprecated::
        Use `create_chunks()` instead. This function is kept for backward compatibility.

    Args:
        text: Full text (metadata + table)
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks

    Returns:
        List of chunks
    """
    logger.warning(
        "split_table_based_content() is deprecated. "
        "Use create_chunks() with appropriate file_extension instead."
    )
    return _split_table_based_content(text, chunk_size, chunk_overlap)


def split_text_preserving_html_blocks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    file_extension: Optional[str] = None,
    force_chunking: Optional[bool] = False
) -> List[str]:
    """
    Chunk text while preserving HTML tables and considering page boundaries.

    .. deprecated::
        Use `create_chunks()` instead. This function is kept for backward compatibility.

    Args:
        text: Original text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks
        file_extension: File extension (csv, xlsx, pdf, etc.)
        force_chunking: Force chunking (disable table protection)

    Returns:
        List of chunks
    """
    logger.warning(
        "split_text_preserving_html_blocks() is deprecated. "
        "Use create_chunks() instead."
    )
    return _split_text(
        text, chunk_size, chunk_overlap,
        file_extension=file_extension,
        force_chunking=force_chunking
    )


def is_table_based_file_type(file_extension: Optional[str]) -> bool:
    """
    Check if the file extension is a table-based file type.

    .. deprecated::
        Use `_is_table_based_file_type()` instead (internal use only).

    Args:
        file_extension: File extension

    Returns:
        True if table-based file type
    """
    return _is_table_based_file_type(file_extension)
