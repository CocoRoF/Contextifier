# contextifier/handlers/hwp/_record.py
"""
HWP 5.0 binary record parser.

HWP body streams consist of a sequence of *records*.  Each record
has a 4-byte header encoded as a little-endian uint32:

    bits  0–9  : tag_id   (0-1023)
    bits 10–19 : level    (0-1023, nesting depth)
    bits 20–31 : rec_size (0-4095; if 0xFFF → next 4 bytes hold the
                           actual 32-bit size)

Records form a tree: a record at *level N* is the child of the
nearest preceding record at *level N−1*.

Public API:
    HwpRecord          — one record node in the tree
    HwpRecord.build_tree(data) → virtual root whose children are
                                 top-level records
"""

from __future__ import annotations

import struct
from typing import List, Optional

from contextifier.handlers.hwp._constants import (
    CTRL_CHAR_DRAWING_TABLE_OBJECT,
    CTRL_CHAR_LINE_BREAK,
    CTRL_CHAR_PARA_BREAK,
    CTRL_CHAR_TAB,
    EXTENDED_CHAR_UNITS,
)


class HwpRecord:
    """A single HWP binary record, with optional children."""

    __slots__ = ("tag_id", "level", "payload", "children", "parent")

    def __init__(
        self,
        tag_id: int = 0,
        level: int = 0,
        payload: bytes = b"",
        parent: Optional["HwpRecord"] = None,
    ) -> None:
        self.tag_id = tag_id
        self.level = level
        self.payload = payload
        self.children: List["HwpRecord"] = []
        self.parent = parent

    # ── Tree construction ─────────────────────────────────────────────

    @classmethod
    def build_tree(cls, data: bytes) -> "HwpRecord":
        """
        Parse a flat byte stream of HWP records into a tree.

        Returns a virtual *root* record (tag_id=0, level=-1) whose
        ``children`` are the top-level records.
        """
        root = cls(tag_id=0, level=-1, payload=b"")
        records = cls._parse_flat(data)
        if not records:
            return root

        # Build parent↔child relationships via level
        stack: List[HwpRecord] = [root]
        for rec in records:
            # Pop stack until we find the parent level
            while len(stack) > 1 and stack[-1].level >= rec.level:
                stack.pop()
            rec.parent = stack[-1]
            stack[-1].children.append(rec)
            stack.append(rec)

        return root

    @classmethod
    def _parse_flat(cls, data: bytes) -> List["HwpRecord"]:
        """Parse all records sequentially (flat list)."""
        records: List[HwpRecord] = []
        pos = 0
        data_len = len(data)

        while pos + 4 <= data_len:
            header = struct.unpack_from("<I", data, pos)[0]
            pos += 4

            tag_id = header & 0x3FF
            level = (header >> 10) & 0x3FF
            rec_size = (header >> 20) & 0xFFF

            # Extended size
            if rec_size == 0xFFF:
                if pos + 4 > data_len:
                    break
                rec_size = struct.unpack_from("<I", data, pos)[0]
                pos += 4

            end = pos + rec_size
            if end > data_len:
                # Truncated record — take what we can
                payload = data[pos:data_len]
                records.append(cls(tag_id=tag_id, level=level, payload=payload))
                break

            payload = data[pos:end]
            records.append(cls(tag_id=tag_id, level=level, payload=payload))
            pos = end

        return records

    # ── Text extraction ───────────────────────────────────────────────

    def get_text(self) -> str:
        """
        Decode PARA_TEXT payload to a Python string.

        HWP stores paragraph text as UTF-16LE with embedded control
        characters.  Characters < 32 are treated as controls:

        - 0x0D → ``\\n``  (paragraph break)
        - 0x0A → ``\\n``  (line break)
        - 0x09 → ``\\t``  (tab)
        - 0x0B → ``\\x0b`` (drawing/table object marker — kept so the
          caller can split around it to insert table/image content)
        - Other codes < 32 → *extended character* occupying 8 code
          units (16 bytes) total.  We skip these.
        """
        payload = self.payload
        result: List[str] = []
        i = 0
        length = len(payload)

        while i + 1 < length:
            code = struct.unpack_from("<H", payload, i)[0]

            if code == CTRL_CHAR_PARA_BREAK:
                result.append("\n")
                i += 2
            elif code == CTRL_CHAR_LINE_BREAK:
                result.append("\n")
                i += 2
            elif code == CTRL_CHAR_TAB:
                result.append("\t")
                i += 2
            elif code == CTRL_CHAR_DRAWING_TABLE_OBJECT:
                result.append("\x0b")
                i += 2
            elif code < 32:
                # Extended control char — occupies 8 code units
                i += EXTENDED_CHAR_UNITS * 2
            else:
                result.append(chr(code))
                i += 2

        return "".join(result)

    # ── Sibling iteration (for table cells) ───────────────────────────

    def get_next_siblings(self, count: int) -> List["HwpRecord"]:
        """
        Return the next *count* siblings after this record.

        Used by the table parser: a LIST_HEADER cell may reference
        subsequent paragraph records that are siblings rather than
        children.
        """
        if self.parent is None:
            return []

        siblings = self.parent.children
        try:
            idx = siblings.index(self)
        except ValueError:
            return []

        return siblings[idx + 1: idx + 1 + count]


__all__ = ["HwpRecord"]
