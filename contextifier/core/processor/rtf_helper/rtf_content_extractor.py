# contextifier/core/processor/rtf_helper/rtf_content_extractor.py
"""
RTF Content Extractor

Extracts inline content (text + tables) from RTF documents.
"""
import logging
import re
from typing import List, Tuple

from contextifier.core.processor.rtf_helper.rtf_decoder import (
    decode_hex_escapes,
)
from contextifier.core.processor.rtf_helper.rtf_text_cleaner import (
    clean_rtf_text,
    remove_destination_groups,
    remove_shape_groups,
    remove_shape_property_groups,
)
from contextifier.core.processor.rtf_helper.rtf_region_finder import (
    find_excluded_regions,
)
from contextifier.core.processor.rtf_helper.rtf_table_extractor import (
    RTFTable,
)

logger = logging.getLogger("contextify.rtf.content")


def extract_inline_content(
    content: str,
    table_regions: List[Tuple[int, int, RTFTable]],
    encoding: str = "cp949"
) -> str:
    """
    Extract inline content from RTF with tables in original positions.
    
    Args:
        content: RTF string content
        table_regions: Table region list [(start, end, table), ...]
        encoding: Encoding to use
        
    Returns:
        Content string with tables inline
    """
    # Find header end (before first \pard)
    header_end = 0
    pard_match = re.search(r'\\pard\b', content)
    if pard_match:
        header_end = pard_match.start()
    
    # Find excluded regions (header, footer, footnote, etc.)
    excluded_regions = find_excluded_regions(content)
    
    def clean_segment(segment: str, start_pos: int) -> str:
        """Clean a segment while respecting excluded regions."""
        if not excluded_regions:
            segment = remove_destination_groups(segment)
            decoded = decode_hex_escapes(segment, encoding)
            return clean_rtf_text(decoded, encoding)
        
        result_parts = []
        seg_pos = 0
        
        for excl_start, excl_end in excluded_regions:
            rel_start = excl_start - start_pos
            rel_end = excl_end - start_pos
            
            if rel_end <= 0 or rel_start >= len(segment):
                continue
            
            rel_start = max(0, rel_start)
            rel_end = min(len(segment), rel_end)
            
            if rel_start > seg_pos:
                part = segment[seg_pos:rel_start]
                part = remove_destination_groups(part)
                decoded = decode_hex_escapes(part, encoding)
                clean = clean_rtf_text(decoded, encoding)
                if clean.strip():
                    result_parts.append(clean)
            
            seg_pos = rel_end
        
        if seg_pos < len(segment):
            part = segment[seg_pos:]
            part = remove_destination_groups(part)
            decoded = decode_hex_escapes(part, encoding)
            clean = clean_rtf_text(decoded, encoding)
            if clean.strip():
                result_parts.append(clean)
        
        return ' '.join(result_parts)
    
    result_parts = []
    
    # No tables - just extract text
    if not table_regions:
        clean = clean_segment(content[header_end:], header_end)
        if clean.strip():
            result_parts.append(clean)
        return '\n\n'.join(result_parts)
    
    # Adjust regions for header offset
    adjusted_regions = []
    for start_pos, end_pos, table in table_regions:
        if end_pos > header_end:
            adj_start = max(start_pos, header_end)
            adjusted_regions.append((adj_start, end_pos, table))
    
    # Build content parts
    last_end = header_end
    
    for start_pos, end_pos, table in adjusted_regions:
        # Text before table
        if start_pos > last_end:
            segment = content[last_end:start_pos]
            clean = clean_segment(segment, last_end)
            if clean.strip():
                result_parts.append(clean)
        
        # Table
        if table.is_real_table():
            result_parts.append(table.to_html())
        else:
            text_list = table.to_text_list()
            if text_list:
                result_parts.append(text_list)
        
        last_end = end_pos
    
    # Text after last table
    if last_end < len(content):
        segment = content[last_end:]
        clean = clean_segment(segment, last_end)
        if clean.strip():
            result_parts.append(clean)
    
    return '\n\n'.join(result_parts)


def extract_text_only(content: str, encoding: str = "cp949") -> str:
    """
    Extract only text from RTF (exclude tables).
    
    Legacy compatibility function.
    
    Args:
        content: RTF string content
        encoding: Encoding to use
        
    Returns:
        Extracted text
    """
    # Remove header (fonttbl, colortbl, stylesheet, etc.)
    pard_match = re.search(r'\\pard\b', content)
    if pard_match:
        content = content[pard_match.start():]
    
    # Remove destination groups
    content = remove_destination_groups(content)
    
    # Handle shape groups (preserve shptxt content)
    content = remove_shape_groups(content)
    
    # Remove shape property groups
    content = remove_shape_property_groups(content)
    
    # Find table regions
    table_regions = []
    for match in re.finditer(r'\\trowd.*?\\row', content, re.DOTALL):
        table_regions.append((match.start(), match.end()))
    
    # Merge adjacent tables
    merged_regions = []
    for start, end in table_regions:
        if merged_regions and start - merged_regions[-1][1] < 100:
            merged_regions[-1] = (merged_regions[-1][0], end)
        else:
            merged_regions.append((start, end))
    
    # Extract text excluding table regions
    text_parts = []
    last_end = 0
    
    for start, end in merged_regions:
        if start > last_end:
            segment = content[last_end:start]
            decoded = decode_hex_escapes(segment, encoding)
            clean = clean_rtf_text(decoded, encoding)
            if clean:
                text_parts.append(clean)
        last_end = end
    
    if last_end < len(content):
        segment = content[last_end:]
        decoded = decode_hex_escapes(segment, encoding)
        clean = clean_rtf_text(decoded, encoding)
        if clean:
            text_parts.append(clean)
    
    text = '\n'.join(text_parts)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


__all__ = [
    'extract_inline_content',
    'extract_text_only',
]
