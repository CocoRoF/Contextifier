# contextifier/core/processor/rtf_helper/rtf_region_finder.py
"""
RTF Region Finder

Functions for finding excluded regions (header, footer, footnote, etc.) in RTF.
"""
import re
from typing import List, Tuple


def find_excluded_regions(content: str) -> List[Tuple[int, int]]:
    """
    Find regions to exclude from content extraction.
    
    Finds header, footer, footnote, and other special regions
    that should not be part of main content.
    
    Args:
        content: RTF content string
        
    Returns:
        List of (start, end) position tuples
    """
    regions = []
    
    # Header/footer patterns
    patterns = [
        (r'\\header[lrf]?\b', r'\\par\s*\}'),      # Headers
        (r'\\footer[lrf]?\b', r'\\par\s*\}'),      # Footers
        (r'\\footnote\b', r'\}'),                   # Footnotes
        (r'\\annotation\b', r'\}'),                 # Annotations
        (r'\{\\headerf', r'\}'),                    # First page header
        (r'\{\\footerf', r'\}'),                    # First page footer
    ]
    
    for start_pattern, end_pattern in patterns:
        for match in re.finditer(start_pattern, content):
            start_pos = match.start()
            
            # Find matching closing brace
            depth = 0
            i = start_pos
            found_start = False
            
            while i < len(content):
                if content[i] == '{':
                    if not found_start:
                        found_start = True
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if found_start and depth == 0:
                        regions.append((start_pos, i + 1))
                        break
                i += 1
    
    # Merge overlapping regions
    if regions:
        regions.sort(key=lambda x: x[0])
        merged = [regions[0]]
        for start, end in regions[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged
    
    return regions


def is_in_excluded_region(position: int, regions: List[Tuple[int, int]]) -> bool:
    """
    Check if a position is within an excluded region.
    
    Args:
        position: Position to check
        regions: List of (start, end) tuples
        
    Returns:
        True if position is in an excluded region
    """
    for start, end in regions:
        if start <= position < end:
            return True
    return False


__all__ = [
    'find_excluded_regions',
    'is_in_excluded_region',
]
