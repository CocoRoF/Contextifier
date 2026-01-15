"""
HWP Record 파싱 클래스
"""
import struct
import logging
from itertools import islice
from typing import Optional

from .hwp_constants import HWPTAG_PARA_TEXT

logger = logging.getLogger("document-processor")


class HwpRecord:
    def __init__(self, tag_id: int, payload: bytes, parent: 'HwpRecord' = None):
        self.tag_id = tag_id
        self.payload = payload
        self.parent = parent
        self.children = []

    def get_next_siblings(self, count=None):
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
        Extract text from HWPTAG_PARA_TEXT payload, handling control characters.
        Returns text with \\x0b markers for extended controls.
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
                # Normal char
                text += chr(code)
                cursor += 2
            else:
                # Control char handling
                if code == 13: # Para break
                    text += '\n'
                    cursor += 2
                elif code == 10: # Line break
                    text += '\n'
                    cursor += 2
                elif code == 9: # Tab
                    text += '\t'
                    cursor += 2
                else:
                    # Extended control chars have extra data
                    # Simplified logic based on known HWP structure
                    size = 1
                    if code in [4, 5, 6, 7, 8, 9, 19, 20]: # Inline
                        size = 8
                    elif code in [1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23]: # Extended
                        size = 8
                        # Code 11 is the standard "Extended Control" marker (for Tables, GSO, etc.)
                        if code == 11:
                            text += '\x0b'
                            # logger.debug(f"Found Extended Control Marker (11) at cursor {cursor}")

                    cursor += size * 2

        return text

    @staticmethod
    def build_tree(data: bytes) -> 'HwpRecord':
        root = HwpRecord(0, b'')
        pos = 0
        size = len(data)

        # Stack to keep track of parents based on level
        # Level 0 is root children
        # stack[0] = root
        stack = {0: root}

        while pos < size:
            try:
                if pos + 4 > size:
                    break
                header = struct.unpack('<I', data[pos:pos+4])[0]
                pos += 4

                tag_id = header & 0x3FF
                level = (header >> 10) & 0x3FF
                rec_len = (header >> 20) & 0xFFF

                if rec_len == 0xFFF:
                    if pos + 4 > size:
                        break
                    rec_len = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4

                if pos + rec_len > size:
                    # Truncated record, stop parsing
                    break

                payload = data[pos:pos+rec_len]
                pos += rec_len

                # Determine parent
                parent = stack.get(level - 1, root)
                if level == 0:
                    parent = root

                # If parent is not in stack (gap in levels), fallback to root or nearest
                if parent is None:
                    # Find nearest lower level
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

                # Clear deeper levels from stack as we moved to a new node at this level
                keys_to_remove = [k for k in stack.keys() if k > level]
                for k in keys_to_remove:
                    del stack[k]
            except Exception as e:
                logger.debug(f"Error parsing HWP record at pos {pos}: {e}")
                break

        return root
