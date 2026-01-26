# contextifier/core/processor/hwp5_helper/hwp5_record.py
"""
HWP 5.0 Record Parser

Parses HWP 5.0 binary records from BodyText/Section streams.

Record Structure:
- Header (4 bytes): TagID (10 bits) | Level (10 bits) | Size (12 bits)
- Extended Size (4 bytes): Only if Size field == 0xFFF
- Payload: Variable length data

Records form a tree structure based on Level values.
Level 0 = root children, Level N+1 = children of Level N record.
"""
import struct
import logging
from itertools import islice
from typing import Optional, List

from contextifier.core.processor.hwp5_helper.hwp5_constants import HWPTAG_PARA_TEXT

logger = logging.getLogger("document-processor.HWP5")


class HwpRecord:
    """
    HWP 5.0 Binary Record.
    
    Represents a single record in the HWP binary stream.
    Records form a tree structure based on their level values.
    
    Attributes:
        tag_id: Record type identifier (10 bits, 0-1023)
        payload: Record data (variable length)
        parent: Parent record (or None for root children)
        children: Child records
    
    Usage:
        # Build tree from decompressed section data
        root = HwpRecord.build_tree(section_data)
        
        # Traverse tree
        for child in root.children:
            if child.tag_id == HWPTAG_PARA_HEADER:
                text = child.get_text()
    """
    
    def __init__(self, tag_id: int, payload: bytes, parent: 'HwpRecord' = None):
        """
        Initialize HwpRecord.
        
        Args:
            tag_id: Record type identifier
            payload: Record binary data
            parent: Parent record
        """
        self.tag_id = tag_id
        self.payload = payload
        self.parent = parent
        self.children: List['HwpRecord'] = []

    def get_next_siblings(self, count=None):
        """
        Get subsequent sibling records.
        
        Used for table cell processing where cells reference
        subsequent paragraph records.
        
        Args:
            count: Maximum number of siblings to return (None for all)
            
        Returns:
            Iterator of sibling records
        """
        if not self.parent:
            return []
        try:
            start_idx = self.parent.children.index(self) + 1
            if count is None:
                end_idx = None
            else:
                end_idx = start_idx + count
            return islice(self.parent.children, start_idx, end_idx)
        except ValueError:
            return []

    def get_text(self) -> str:
        """
        Extract text from HWPTAG_PARA_TEXT payload.
        
        HWP text is encoded as UTF-16LE with control characters:
        - Code >= 32: Normal character
        - Code < 32: Control character with extended data
        
        Control character handling:
        - 9: Tab
        - 10: Line break
        - 13: Paragraph break
        - 11: Extended control marker (for tables, GSO, etc.)
        - Others: Variable length extended data
        
        Returns:
            Extracted text with control markers
        """
        if self.tag_id != HWPTAG_PARA_TEXT:
            return ""

        # HWP text is UTF-16LE
        text = ''
        payload = self.payload
        cursor = 0

        while cursor < len(payload):
            if cursor + 1 >= len(payload):
                break

            code = struct.unpack('<H', payload[cursor:cursor+2])[0]

            if code >= 32:
                # Normal character
                text += chr(code)
                cursor += 2
            else:
                # Control character handling
                if code == 13:  # Paragraph break
                    text += '\n'
                    cursor += 2
                elif code == 10:  # Line break
                    text += '\n'
                    cursor += 2
                elif code == 9:  # Tab
                    text += '\t'
                    cursor += 2
                else:
                    # Extended control chars have extra data
                    size = 1
                    if code in [4, 5, 6, 7, 8, 9, 19, 20]:  # Inline controls
                        size = 8
                    elif code in [1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23]:  # Extended
                        size = 8
                        # Code 11 is the standard "Extended Control" marker
                        # Used for Tables, GSO (images), etc.
                        if code == 11:
                            text += '\x0b'

                    cursor += size * 2

        return text

    def find_children_by_tag(self, tag_id: int) -> List['HwpRecord']:
        """
        Find all direct children with specified tag ID.
        
        Args:
            tag_id: Tag ID to search for
            
        Returns:
            List of matching child records
        """
        return [c for c in self.children if c.tag_id == tag_id]
    
    def find_first_child_by_tag(self, tag_id: int) -> Optional['HwpRecord']:
        """
        Find first direct child with specified tag ID.
        
        Args:
            tag_id: Tag ID to search for
            
        Returns:
            First matching child record or None
        """
        for c in self.children:
            if c.tag_id == tag_id:
                return c
        return None
    
    def find_descendants_by_tag(self, tag_id: int) -> List['HwpRecord']:
        """
        Find all descendants (recursive) with specified tag ID.
        
        Args:
            tag_id: Tag ID to search for
            
        Returns:
            List of all matching descendant records
        """
        results = []
        if self.tag_id == tag_id:
            results.append(self)
        for child in self.children:
            results.extend(child.find_descendants_by_tag(tag_id))
        return results

    @staticmethod
    def build_tree(data: bytes) -> 'HwpRecord':
        """
        Build record tree from decompressed section data.
        
        Parses binary stream and builds hierarchical tree structure
        based on record level values.
        
        Args:
            data: Decompressed BodyText/Section stream data
            
        Returns:
            Root record containing all parsed records as children
        """
        root = HwpRecord(0, b'')
        pos = 0
        size = len(data)

        # Stack to keep track of parents based on level
        # Level 0 records are children of root
        stack = {0: root}

        while pos < size:
            try:
                if pos + 4 > size:
                    break
                    
                # Parse 4-byte header
                header = struct.unpack('<I', data[pos:pos+4])[0]
                pos += 4

                # Extract fields from header
                tag_id = header & 0x3FF           # bits 0-9: Tag ID
                level = (header >> 10) & 0x3FF   # bits 10-19: Level
                rec_len = (header >> 20) & 0xFFF # bits 20-31: Size

                # Extended size: if rec_len == 0xFFF, next 4 bytes contain actual size
                if rec_len == 0xFFF:
                    if pos + 4 > size:
                        break
                    rec_len = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4

                # Validate record bounds
                if pos + rec_len > size:
                    # Truncated record, stop parsing
                    logger.debug(f"Truncated record at pos {pos}, expected {rec_len} bytes")
                    break

                payload = data[pos:pos+rec_len]
                pos += rec_len

                # Determine parent based on level
                parent = stack.get(level - 1, root)
                if level == 0:
                    parent = root

                # If parent is not in stack (gap in levels), fallback to nearest
                if parent is None:
                    for l in range(level - 1, -1, -1):
                        if l in stack:
                            parent = stack[l]
                            break
                    if parent is None:
                        parent = root

                record = HwpRecord(tag_id, payload, parent)
                parent.children.append(record)

                # Update stack for this level
                stack[level] = record

                # Clear deeper levels from stack
                keys_to_remove = [k for k in stack.keys() if k > level]
                for k in keys_to_remove:
                    del stack[k]
                    
            except Exception as e:
                logger.debug(f"Error parsing HWP record at pos {pos}: {e}")
                break

        return root
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"HwpRecord(tag_id={self.tag_id}, payload_size={len(self.payload)}, children={len(self.children)})"


__all__ = ['HwpRecord']
