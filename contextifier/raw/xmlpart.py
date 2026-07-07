# contextifier/raw/xmlpart.py
"""
Lazy lxml view over an :class:`~contextifier.raw.opc.OpcPart`.

Format models hold ``XmlPart`` facades for the parts they understand.
The tree is parsed on first access and serialized back into the package
only when the model mutated it (``mark_dirty()`` + ``flush()``), so
opening a document and reading a few values never rewrites anything —
the byte-preservation contract stays intact by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from lxml.etree import _Element

    from contextifier.raw.opc import OpcPart

__all__ = ["NS", "qn", "XmlPart"]

# The OOXML namespace registry shared by the raw layer. Prefixes follow
# the conventional short names used across ECMA-376 documentation.
NS: dict[str, str] = {
    # wordprocessingml
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    # drawingml
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "cx": "http://schemas.microsoft.com/office/drawing/2014/chartex",
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    # presentationml
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    # spreadsheetml
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    # shared
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}


def qn(tag: str) -> str:
    """``"w:p"`` → ``"{http://…/wordprocessingml/2006/main}p"``."""
    prefix, _, local = tag.partition(":")
    if not local:
        return tag
    try:
        return f"{{{NS[prefix]}}}{local}"
    except KeyError:
        raise KeyError(f"Unknown XML namespace prefix: {prefix!r}") from None


class XmlPart:
    """A parsed XML part with explicit dirty tracking.

    ``root`` parses lazily; call :meth:`mark_dirty` after mutating the
    tree and :meth:`flush` (usually via the owning document's ``save``)
    to serialize back into the package part.
    """

    __slots__ = ("part", "_root", "_dirty")

    def __init__(self, part: "OpcPart"):
        self.part = part
        self._root: "_Element | None" = None
        self._dirty = False

    @property
    def name(self) -> str:
        return self.part.name

    @property
    def root(self) -> "_Element":
        if self._root is None:
            from lxml import etree

            self._root = etree.fromstring(self.part.read())
        return self._root

    @property
    def loaded(self) -> bool:
        return self._root is not None

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    def flush(self) -> None:
        """Serialize the tree into the part iff this facade dirtied it."""
        if self._dirty and self._root is not None:
            from lxml import etree

            self.part.write(
                etree.tostring(
                    self._root, xml_declaration=True, encoding="UTF-8", standalone=True
                )
            )
            self._dirty = False

    # -- conveniences -----------------------------------------------------------

    def find(self, path: str):
        """`find` with the shared namespace map (``"s:sheetData/s:row"``)."""
        return self.root.find(path, NS)

    def findall(self, path: str):
        return self.root.findall(path, NS)

    def iter(self, tag: str):
        return self.root.iter(qn(tag))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        state = "dirty" if self._dirty else ("loaded" if self.loaded else "lazy")
        return f"<XmlPart {self.name!r} [{state}]>"
