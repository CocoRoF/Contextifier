# contextifier/raw/opc.py
"""
OPC (Open Packaging Conventions) container with a byte-preservation
write contract.

An OOXML file is a ZIP of *parts* plus ``[Content_Types].xml`` and
``_rels/*.rels`` relationship graphs. The higher-level Office libraries
(openpyxl & friends) parse the whole package into their own object model
and re-serialize everything on save — silently destroying whatever they
don't model. This container does the opposite:

**Byte-preservation contract** — on :meth:`OpcPackage.save`, every part
that was not explicitly written through :meth:`OpcPart.write` (or
removed/added) is emitted with its original decompressed bytes,
unchanged. Only dirty parts are re-serialized. The ZIP container itself
may be re-encoded (compression is not part of the contract); part
*content* is.

The container knows nothing about spreadsheets or slides — format
models (:mod:`contextifier.raw.xlsx` etc.) sit on top.
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterator

from contextifier.errors import ContextifierError

__all__ = ["OpcPackage", "OpcPart", "Relationships", "RawUnsupportedError"]

_CONTENT_TYPES = "[Content_Types].xml"

# Mirrors the pipeline's zip-bomb guard (handlers validate the same cap).
_MAX_UNCOMPRESSED = 1 << 30  # 1 GiB


class RawUnsupportedError(ContextifierError):
    """The requested raw capability does not exist for this format."""


class OpcPart:
    """One part (file entry) inside the package.

    Parts are lazy: bytes are read from the source ZIP on first access.
    Writing replaces the content and marks the part dirty; clean parts
    round-trip byte-identically.
    """

    __slots__ = ("name", "_package", "_data", "dirty", "is_new")

    def __init__(self, name: str, package: "OpcPackage", *, is_new: bool = False):
        self.name = name
        self._package = package
        self._data: bytes | None = None
        self.dirty = is_new
        self.is_new = is_new

    def read(self) -> bytes:
        if self._data is None:
            self._data = self._package._read_source(self.name)
        return self._data

    def write(self, data: bytes) -> None:
        self._data = bytes(data)
        self.dirty = True

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        state = "new" if self.is_new else ("dirty" if self.dirty else "clean")
        return f"<OpcPart {self.name!r} [{state}]>"


class Relationships:
    """A part's ``_rels`` graph (``_rels/<basename>.rels``).

    Minimal, allocation-light XML handling: relationships are parsed with
    lxml on demand and re-serialized only when mutated.
    """

    NS = "http://schemas.openxmlformats.org/package/2006/relationships"

    def __init__(self, part: OpcPart):
        from lxml import etree

        self._part = part
        self._root = etree.fromstring(part.read())

    def __iter__(self) -> Iterator[dict]:
        for rel in self._root:
            if rel.tag == f"{{{self.NS}}}Relationship":
                yield {
                    "id": rel.get("Id"),
                    "type": rel.get("Type"),
                    "target": rel.get("Target"),
                    "mode": rel.get("TargetMode", "Internal"),
                }

    def target_of(self, rel_id: str) -> str | None:
        for rel in self:
            if rel["id"] == rel_id:
                return rel["target"]
        return None

    def by_type(self, type_suffix: str) -> list[dict]:
        """Relationships whose Type ends with *type_suffix* (e.g. ``/chart``)."""
        return [rel for rel in self if rel["type"].endswith(type_suffix)]

    def resolve(self, base_part: str, target: str) -> str:
        """Resolve a relationship target to an absolute part name."""
        if target.startswith("/"):
            return target[1:]
        base_dir = posixpath.dirname(base_part)
        return posixpath.normpath(posixpath.join(base_dir, target))

    def add(
        self, rel_id: str, rel_type: str, target: str, *, external: bool = False
    ) -> None:
        from lxml import etree

        rel = etree.SubElement(self._root, f"{{{self.NS}}}Relationship")
        rel.set("Id", rel_id)
        rel.set("Type", rel_type)
        rel.set("Target", target)
        if external:
            rel.set("TargetMode", "External")
        self.flush()

    def remove(self, rel_id: str) -> bool:
        for rel in list(self._root):
            if rel.get("Id") == rel_id:
                self._root.remove(rel)
                self.flush()
                return True
        return False

    def next_id(self) -> str:
        used = {rel["id"] for rel in self}
        n = len(used) + 1
        while f"rId{n}" in used:
            n += 1
        return f"rId{n}"

    def flush(self) -> None:
        from lxml import etree

        self._part.write(
            etree.tostring(
                self._root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
        )


class OpcPackage:
    """The package: parts + content types + relationship graphs.

    Open with :meth:`open`, mutate parts, then :meth:`save` /
    :meth:`to_bytes`. Untouched parts keep their exact original bytes.
    """

    def __init__(self, source_bytes: bytes):
        self._source = source_bytes
        self._zip = zipfile.ZipFile(io.BytesIO(source_bytes))
        total = sum(i.file_size for i in self._zip.infolist())
        if total > _MAX_UNCOMPRESSED:
            raise ContextifierError(
                f"Package inflates to {total} bytes (> {_MAX_UNCOMPRESSED}); refusing"
            )
        self._parts: dict[str, OpcPart] = {
            info.filename: OpcPart(info.filename, self)
            for info in self._zip.infolist()
            if not info.is_dir()
        }
        self._removed: set[str] = set()
        self._rels_cache: dict[str, Relationships] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def open(cls, source: str | Path | bytes | BinaryIO) -> "OpcPackage":
        if isinstance(source, (str, Path)):
            data = Path(source).read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = source.read()
        if data[:4] != b"PK\x03\x04":
            raise ContextifierError("Not a ZIP/OPC package (bad magic)")
        return cls(data)

    def sniff_format(self) -> str | None:
        """Best-effort format detection from the package layout."""
        if self.has_part("xl/workbook.xml"):
            return "xlsx"
        if self.has_part("word/document.xml"):
            return "docx"
        if self.has_part("ppt/presentation.xml"):
            return "pptx"
        return None

    # -- part access ----------------------------------------------------------

    def _read_source(self, name: str) -> bytes:
        return self._zip.read(name)

    @property
    def part_names(self) -> list[str]:
        return sorted(self._parts)

    def has_part(self, name: str) -> bool:
        return name in self._parts

    def get_part(self, name: str) -> OpcPart:
        try:
            return self._parts[name]
        except KeyError:
            raise KeyError(f"No such part: {name!r}") from None

    __getitem__ = get_part

    def add_part(self, name: str, data: bytes) -> OpcPart:
        part = OpcPart(name, self, is_new=True)
        part.write(data)
        self._parts[name] = part
        self._removed.discard(name)
        return part

    def remove_part(self, name: str) -> None:
        self._parts.pop(name, None)
        self._rels_cache.pop(self._rels_name_for(name), None)
        self._removed.add(name)

    # -- relationships & content types ---------------------------------------

    @staticmethod
    def _rels_name_for(part_name: str) -> str:
        if part_name == "":  # package-level rels
            return "_rels/.rels"
        directory = posixpath.dirname(part_name)
        base = posixpath.basename(part_name)
        return (
            posixpath.join(directory, "_rels", f"{base}.rels")
            if directory
            else f"_rels/{base}.rels"
        )

    def rels_for(self, part_name: str) -> Relationships | None:
        """The relationships of *part_name* ('' = package root), or None."""
        rels_name = self._rels_name_for(part_name)
        if rels_name in self._rels_cache:
            return self._rels_cache[rels_name]
        if not self.has_part(rels_name):
            return None
        rels = Relationships(self.get_part(rels_name))
        self._rels_cache[rels_name] = rels
        return rels

    def content_type_of(self, part_name: str) -> str | None:
        from lxml import etree

        root = etree.fromstring(self.get_part(_CONTENT_TYPES).read())
        ns = "http://schemas.openxmlformats.org/package/2006/content-types"
        for el in root:
            if el.tag == f"{{{ns}}}Override" and el.get("PartName") == f"/{part_name}":
                return el.get("ContentType")
        ext = part_name.rsplit(".", 1)[-1].lower()
        for el in root:
            if (
                el.tag == f"{{{ns}}}Default"
                and (el.get("Extension") or "").lower() == ext
            ):
                return el.get("ContentType")
        return None

    def set_content_type_override(self, part_name: str, content_type: str) -> None:
        from lxml import etree

        ct = self.get_part(_CONTENT_TYPES)
        root = etree.fromstring(ct.read())
        ns = "http://schemas.openxmlformats.org/package/2006/content-types"
        for el in root:
            if el.tag == f"{{{ns}}}Override" and el.get("PartName") == f"/{part_name}":
                el.set("ContentType", content_type)
                break
        else:
            override = etree.SubElement(root, f"{{{ns}}}Override")
            override.set("PartName", f"/{part_name}")
            override.set("ContentType", content_type)
        ct.write(
            etree.tostring(
                root, xml_declaration=True, encoding="UTF-8", standalone=True
            )
        )

    # -- saving ----------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize the package. Clean parts are byte-identical."""
        buf = io.BytesIO()
        original_infos = {i.filename: i for i in self._zip.infolist()}
        with zipfile.ZipFile(buf, "w") as out:
            # Original entry order first (Office is order-tolerant, but
            # keeping it minimizes diffs), then new parts.
            for name in [
                *(n for n in original_infos if n in self._parts),
                *(n for n in self._parts if n not in original_infos),
            ]:
                part = self._parts[name]
                src_info = original_infos.get(name)
                if part.dirty or src_info is None:
                    data = part.read()
                    compress = (
                        src_info.compress_type
                        if src_info is not None
                        else zipfile.ZIP_DEFLATED
                    )
                    out.writestr(
                        zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)),
                        data,
                        compress_type=compress,
                    )
                else:
                    # Clean part: original decompressed bytes, original
                    # compression method, original timestamp.
                    out.writestr(
                        src_info,
                        self._zip.read(name),
                        compress_type=src_info.compress_type,
                    )
        return buf.getvalue()

    def save(self, target: str | Path | BinaryIO | None = None) -> bytes:
        data = self.to_bytes()
        if isinstance(target, (str, Path)):
            Path(target).write_bytes(data)
        elif target is not None:
            target.write(data)
        return data

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "OpcPackage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
