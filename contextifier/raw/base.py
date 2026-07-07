# contextifier/raw/base.py
"""
Shared base for format raw-document models (xlsx / docx / pptx).

A format model owns a set of :class:`~contextifier.raw.xmlpart.XmlPart`
facades over the parts it understands. ``save()`` flushes every dirty
facade into the package, then serializes the package under the
byte-preservation contract. Parts the model does NOT understand are
never touched at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from contextifier.raw.opc import OpcPackage, OpcPart
from contextifier.raw.xmlpart import XmlPart

__all__ = ["RawDocumentBase"]


class RawDocumentBase:
    """Common plumbing: part registry, flush-on-save, byte export."""

    #: subclasses set this ("xlsx" / "docx" / "pptx")
    format: str = ""

    def __init__(self, package: OpcPackage):
        self.package = package
        self._xml_parts: dict[str, XmlPart] = {}

    # -- part facades ------------------------------------------------------------

    def xml_part(self, name: str) -> XmlPart:
        """The (cached) XmlPart facade for a package part."""
        if name not in self._xml_parts:
            self._xml_parts[name] = XmlPart(self.package.get_part(name))
        return self._xml_parts[name]

    def raw_part(self, name: str) -> OpcPart:
        """Direct part access — the escape hatch for anything the model
        doesn't cover."""
        return self.package.get_part(name)

    # -- persistence ----------------------------------------------------------

    def flush(self) -> None:
        """Serialize every dirty XML facade into its package part."""
        for xp in self._xml_parts.values():
            xp.flush()

    def to_bytes(self) -> bytes:
        self.flush()
        return self.package.to_bytes()

    def save(self, target: str | Path | BinaryIO | None = None) -> bytes:
        """Write the package; untouched parts stay byte-identical."""
        self.flush()
        return self.package.save(target)

    def close(self) -> None:
        self.package.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
