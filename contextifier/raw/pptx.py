# contextifier/raw/pptx.py
"""
PptxRawDocument — the raw (lossless, writable) model for .pptx decks.

Slides are thin views over their XML parts: reading never dirties, and
edits are surgical, so untouched parts round-trip byte-identical (the
OPC byte-preservation contract). Two operations exist specifically to
fix what template-driven pipelines (edit2docs-style) get wrong today:

* :meth:`RawSlide.replace_content` — swap a slide's XML for freshly
  generated markup while *pulling the original native objects along*:
  chart / table / diagram graphicFrames (and optionally pictures) are
  lifted out of the old tree and re-appended into the new one, ids
  renumbered, relationships untouched. Native charts stay native
  instead of being rasterized or dropped.
* :meth:`PptxRawDocument.remove_slide` — deletes the slide *and*
  reference-counts every part it pulled in (charts, embedded workbooks,
  images, notes), removing the ones no remaining slide uses. No more
  orphan-part bloat in the package.
"""

from __future__ import annotations

import copy
import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from contextifier.raw.base import RawDocumentBase
from contextifier.raw.chart import ChartModel, find_chart_parts
from contextifier.raw.opc import OpcPackage
from contextifier.raw.xmlpart import NS, qn

if TYPE_CHECKING:  # pragma: no cover
    from lxml.etree import _Element

__all__ = [
    "PptxRawDocument",
    "RawSlide",
    "RawShapeInfo",
    "RawTable",
    "RawTableCell",
]

_PRESENTATION = "ppt/presentation.xml"

#: graphicData/@uri tail → shape kind
_GRAPHIC_KIND = {
    "table": "table",
    "chart": "chart",
    "chartex": "chart",
    "diagram": "diagram",
}

#: uri tails of graphicFrames replace_content must always carry over
_NATIVE_FRAME_TAILS = ("table", "chart", "chartex", "diagram")

#: nv*Pr wrappers whose first p:cNvPr identifies the shape
_NV_PR_TAGS = frozenset(
    qn(t)
    for t in (
        "p:nvSpPr",
        "p:nvPicPr",
        "p:nvGraphicFramePr",
        "p:nvGrpSpPr",
        "p:nvCxnSpPr",
    )
)


@dataclass
class RawShapeInfo:
    """Inventory entry for one shape on a slide."""

    id: int
    name: str
    kind: str  # "text" | "picture" | "table" | "chart" | "group" | "diagram" | "other"
    text: str | None


# -- XML helpers ---------------------------------------------------------------


def _para_text(para: "_Element") -> str:
    return "".join(t.text or "" for t in para.iter(qn("a:t")))


def _body_text(tx_body: "_Element") -> str:
    return "\n".join(_para_text(p) for p in tx_body.findall(qn("a:p")))


def _replace_para_text(para: "_Element", new_text: str) -> None:
    """Set a paragraph's text under the format-preserving contract.

    The first ``a:r`` keeps its ``a:rPr`` (formatting) and receives the
    new text; every *other* plain-text run is dropped; anything that is
    not a plain run — ``a:fld``, ``a:br``, ``a:pPr``, ``a:endParaRPr`` —
    stays exactly where it was.
    """
    from lxml import etree

    a_r, a_t = qn("a:r"), qn("a:t")
    runs = [child for child in para if child.tag == a_r]
    if runs:
        first = runs[0]
        t = first.find(a_t)
        if t is None:
            t = etree.SubElement(first, a_t)
        t.text = new_text
        for extra in runs[1:]:
            para.remove(extra)
        return
    run = etree.SubElement(para, a_r)
    etree.SubElement(run, a_t).text = new_text
    end = para.find(qn("a:endParaRPr"))
    if end is not None:
        end.addprevious(run)


def _shape_cnvpr(shape_el: "_Element") -> "_Element | None":
    """The shape's own ``p:cNvPr`` (never a nested child's)."""
    for child in shape_el:
        if child.tag in _NV_PR_TAGS:
            return child.find(qn("p:cNvPr"))
    return None


def _graphic_kind(frame_el: "_Element") -> str | None:
    data = frame_el.find("a:graphic/a:graphicData", NS)
    if data is None:
        return None
    uri = data.get("uri") or ""
    return _GRAPHIC_KIND.get(uri.rsplit("/", 1)[-1])


def _walk_shapes(container: "_Element") -> Iterator["_Element"]:
    """Shape elements in document order, descending into p:grpSp."""
    shape_tags = (
        qn("p:sp"),
        qn("p:pic"),
        qn("p:graphicFrame"),
        qn("p:grpSp"),
        qn("p:cxnSp"),
    )
    for child in container:
        if child.tag in shape_tags:
            yield child
            if child.tag == qn("p:grpSp"):
                yield from _walk_shapes(child)


def _first_rid(el: "_Element", attr: str) -> str | None:
    """First ``attr`` (a qualified name) found on *el* or a descendant."""
    for node in el.iter():
        rid = node.get(attr)
        if rid:
            return rid
    return None


# -- tables --------------------------------------------------------------------


class RawTableCell:
    """One ``a:tc`` — read/format-preserving-write of its text."""

    def __init__(self, table: "RawTable", tc_el: "_Element"):
        self._table = table
        self._tc = tc_el

    @property
    def text(self) -> str:
        tx = self._tc.find(qn("a:txBody"))
        return _body_text(tx) if tx is not None else ""

    def set_text(self, text: str) -> None:
        """Replace the first paragraph's text (first run's ``a:rPr`` and
        any non-text elements are preserved — same rules as
        :meth:`RawSlide.set_text`)."""
        from lxml import etree

        tx = self._tc.find(qn("a:txBody"))
        if tx is None:
            tx = etree.SubElement(self._tc, qn("a:txBody"))
            etree.SubElement(tx, qn("a:bodyPr"))
            etree.SubElement(tx, qn("a:lstStyle"))
        para = tx.find(qn("a:p"))
        if para is None:
            para = etree.SubElement(tx, qn("a:p"))
        _replace_para_text(para, text)
        self._table._slide._mark_dirty()


class RawTable:
    """A native DrawingML table (``a:tbl``) hosted in a graphicFrame.

    v0.4 scope: cell text + row insert/delete. Column operations would
    require ``a:gridCol`` surgery and are deliberately out of scope.
    """

    def __init__(self, slide: "RawSlide", frame_el: "_Element"):
        self._slide = slide
        self._frame = frame_el
        cnvpr = _shape_cnvpr(frame_el)
        self.shape_id: int = int(cnvpr.get("id")) if cnvpr is not None else -1
        tbl = frame_el.find("a:graphic/a:graphicData/a:tbl", NS)
        if tbl is None:  # pragma: no cover - guarded by caller's uri check
            raise ValueError("graphicFrame does not contain an a:tbl")
        self._tbl = tbl

    # -- geometry ---------------------------------------------------------------

    @property
    def _rows(self) -> list["_Element"]:
        return self._tbl.findall(qn("a:tr"))

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    @property
    def n_cols(self) -> int:
        return len(self._tbl.findall("a:tblGrid/a:gridCol", NS))

    # -- cells ------------------------------------------------------------------

    def cell(self, r: int, c: int) -> RawTableCell:
        rows = self._rows
        if not 0 <= r < len(rows):
            raise IndexError(f"row {r} out of range (table has {len(rows)} rows)")
        cells = rows[r].findall(qn("a:tc"))
        if not 0 <= c < len(cells):
            raise IndexError(f"col {c} out of range (row has {len(cells)} cells)")
        return RawTableCell(self, cells[c])

    # -- rows -------------------------------------------------------------------

    def insert_row(self, idx: int) -> None:
        """Insert an empty row at *idx*, cloning the row above (or the
        first row) as the style template. Column count stays consistent
        with ``a:tblGrid`` because the template row already matches it."""
        from lxml import etree

        rows = self._rows
        if not rows:
            raise ValueError("cannot insert into a table with no template row")
        if not 0 <= idx <= len(rows):
            raise IndexError(f"insert index {idx} out of range (0..{len(rows)})")
        template = rows[idx - 1] if idx > 0 else rows[0]
        new_row = copy.deepcopy(template)
        for tc in new_row.findall(qn("a:tc")):
            tx = tc.find(qn("a:txBody"))
            if tx is not None:
                for para in tx.findall(qn("a:p")):
                    tx.remove(para)
                etree.SubElement(tx, qn("a:p"))
        if idx == len(rows):
            rows[-1].addnext(new_row)
        else:
            rows[idx].addprevious(new_row)
        self._slide._mark_dirty()

    def delete_row(self, idx: int) -> None:
        rows = self._rows
        if not 0 <= idx < len(rows):
            raise IndexError(f"row {idx} out of range (table has {len(rows)} rows)")
        self._tbl.remove(rows[idx])
        self._slide._mark_dirty()


# -- slides --------------------------------------------------------------------


class RawSlide:
    """One slide part, addressed by shape id."""

    def __init__(self, doc: "PptxRawDocument", part_name: str, index: int):
        self._doc = doc
        self.part_name = part_name
        self.index = index

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<RawSlide #{self.index} {self.part_name!r}>"

    @property
    def _xp(self):
        return self._doc.xml_part(self.part_name)

    def _mark_dirty(self) -> None:
        self._xp.mark_dirty()

    @property
    def _sp_tree(self) -> "_Element":
        tree = self._xp.find("p:cSld/p:spTree")
        if tree is None:
            raise ValueError(f"{self.part_name} has no p:cSld/p:spTree")
        return tree

    # -- inventory ---------------------------------------------------------------

    @property
    def shapes(self) -> list[RawShapeInfo]:
        """All shapes (document order, groups flattened in place)."""
        out: list[RawShapeInfo] = []
        for el in _walk_shapes(self._sp_tree):
            cnvpr = _shape_cnvpr(el)
            shape_id = int(cnvpr.get("id")) if cnvpr is not None else -1
            name = (cnvpr.get("name") or "") if cnvpr is not None else ""
            kind, text = "other", None
            if el.tag == qn("p:sp"):
                tx = el.find(qn("p:txBody"))
                if tx is not None:
                    kind, text = "text", _body_text(tx)
            elif el.tag == qn("p:pic"):
                kind = "picture"
            elif el.tag == qn("p:grpSp"):
                kind = "group"
            elif el.tag == qn("p:graphicFrame"):
                kind = _graphic_kind(el) or "other"
            out.append(RawShapeInfo(id=shape_id, name=name, kind=kind, text=text))
        return out

    def _find_shape(self, shape_id: int) -> "_Element":
        for el in _walk_shapes(self._sp_tree):
            cnvpr = _shape_cnvpr(el)
            if cnvpr is not None and cnvpr.get("id") == str(shape_id):
                return el
        raise KeyError(f"No shape with id={shape_id} on {self.part_name}")

    # -- text --------------------------------------------------------------------

    def get_text(self, shape_id: int) -> str:
        el = self._find_shape(shape_id)
        tx = el.find(qn("p:txBody"))
        if tx is None:
            raise ValueError(f"Shape id={shape_id} has no text body")
        return _body_text(tx)

    def set_text(self, shape_id: int, new_text: str, para: int = 0) -> None:
        """Replace paragraph *para*'s text, preserving the first run's
        formatting (``a:rPr``) and any non-text elements (``a:fld``,
        ``a:br``); other plain-text runs in the paragraph are removed."""
        el = self._find_shape(shape_id)
        tx = el.find(qn("p:txBody"))
        if tx is None:
            raise ValueError(f"Shape id={shape_id} has no text body")
        paras = tx.findall(qn("a:p"))
        if not 0 <= para < len(paras):
            raise IndexError(
                f"paragraph {para} out of range (shape has {len(paras)} paragraphs)"
            )
        _replace_para_text(paras[para], new_text)
        self._mark_dirty()

    # -- tables / charts / notes ---------------------------------------------------

    @property
    def tables(self) -> list[RawTable]:
        return [
            RawTable(self, el)
            for el in _walk_shapes(self._sp_tree)
            if el.tag == qn("p:graphicFrame") and _graphic_kind(el) == "table"
        ]

    @property
    def chart_part_names(self) -> list[str]:
        """Chart parts referenced from this slide's relationships."""
        return find_chart_parts(self._doc.package, self.part_name)

    @property
    def charts(self) -> list[ChartModel]:
        """Lazy :class:`ChartModel` views (implemented in milestone C3)."""
        return [
            ChartModel(self._doc.xml_part(name), self._doc.package)
            for name in self.chart_part_names
        ]

    @property
    def notes_text(self) -> str | None:
        """Text of the notes slide's body placeholder, or ``None``."""
        rels = self._doc.package.rels_for(self.part_name)
        if rels is None:
            return None
        notes_rels = rels.by_type("/notesSlide")
        if not notes_rels:
            return None
        notes_part = rels.resolve(self.part_name, notes_rels[0]["target"])
        xp = self._doc.xml_part(notes_part)
        for sp in xp.findall(".//p:sp"):
            ph = sp.find("p:nvSpPr/p:nvPr/p:ph", NS)
            tx = sp.find(qn("p:txBody"))
            if tx is not None and ph is not None and ph.get("type") == "body":
                return _body_text(tx)
        texts = [_body_text(tx) for tx in xp.root.iter(qn("p:txBody"))]
        return "\n".join(t for t in texts if t)

    # -- content replacement --------------------------------------------------------

    def replace_content(
        self,
        new_slide_xml: bytes,
        *,
        preserve_native: bool = True,
        preserve_pictures: bool = True,
    ) -> list[str]:
        """Replace this slide's XML with *new_slide_xml*, carrying the
        original native objects over.

        With ``preserve_native=True`` every chart / table / diagram
        graphicFrame of the ORIGINAL slide (and, when
        ``preserve_pictures``, every ``p:pic`` that references embedded
        media) is deep-copied into the new tree's ``p:spTree``, with
        their ``p:cNvPr/@id`` renumbered past any id used by the new
        XML. The slide keeps its part name, so its relationships part is
        untouched and every carried-over ``r:id`` / ``r:embed`` keeps
        resolving.

        *new_slide_xml* must be a complete ``<p:sld>`` document and may
        only reference relationship ids that already exist in this
        slide's rels (or none at all) — this method never edits the rels
        part, so unknown ``r:id`` / ``r:embed`` values in the new XML
        would dangle.

        Returns descriptions of the preserved elements, e.g.
        ``["table", "chart:chart1.xml", "picture:image1.png"]``.
        """
        from lxml import etree

        new_root = etree.fromstring(new_slide_xml)
        if new_root.tag != qn("p:sld"):
            raise ValueError("new_slide_xml must be a complete <p:sld> document")
        new_tree = new_root.find("p:cSld/p:spTree", NS)
        if new_tree is None:
            raise ValueError("new_slide_xml has no p:cSld/p:spTree")

        preserved: list[str] = []
        if preserve_native:
            rels = self._doc.package.rels_for(self.part_name)
            keep: list[tuple["_Element", str]] = []
            for el in _walk_shapes(self._sp_tree):
                if el.tag == qn("p:graphicFrame"):
                    data = el.find("a:graphic/a:graphicData", NS)
                    uri_tail = (
                        (data.get("uri") or "").rsplit("/", 1)[-1]
                        if data is not None
                        else ""
                    )
                    if uri_tail not in _NATIVE_FRAME_TAILS:
                        continue
                    desc = _GRAPHIC_KIND[uri_tail]
                    if desc == "chart":
                        rid = _first_rid(data, qn("r:id"))
                        target = rels.target_of(rid) if (rels and rid) else None
                        if target:
                            desc = f"chart:{posixpath.basename(target)}"
                    keep.append((el, desc))
                elif el.tag == qn("p:pic") and preserve_pictures:
                    blip = el.find("p:blipFill/a:blip", NS)
                    rid = blip.get(qn("r:embed")) if blip is not None else None
                    if not rid:
                        continue
                    target = rels.target_of(rid) if rels else None
                    desc = (
                        f"picture:{posixpath.basename(target)}" if target else "picture"
                    )
                    keep.append((el, desc))

            used_ids = {
                int(c.get("id"))
                for c in new_root.iter(qn("p:cNvPr"))
                if (c.get("id") or "").isdigit()
            }
            next_id = max(used_ids, default=1) + 1
            for el, desc in keep:
                clone = copy.deepcopy(el)
                for cnvpr in clone.iter(qn("p:cNvPr")):
                    cnvpr.set("id", str(next_id))
                    next_id += 1
                new_tree.append(clone)
                preserved.append(desc)

        xp = self._xp
        xp._root = new_root  # swap the facade's tree in place
        xp.mark_dirty()
        return preserved


# -- document ------------------------------------------------------------------


class PptxRawDocument(RawDocumentBase):
    """Raw model for a .pptx package."""

    format = "pptx"

    @property
    def slides(self) -> list[RawSlide]:
        """Slides in presentation order (``p:sldIdLst``)."""
        pres = self.xml_part(_PRESENTATION)
        rels = self.package.rels_for(_PRESENTATION)
        sld_id_lst = pres.find("p:sldIdLst")
        if sld_id_lst is None or rels is None:
            return []
        out: list[RawSlide] = []
        for sld_id in sld_id_lst.findall(qn("p:sldId")):
            rid = sld_id.get(qn("r:id"))
            target = rels.target_of(rid) if rid else None
            if target is None:
                continue
            out.append(RawSlide(self, rels.resolve(_PRESENTATION, target), len(out)))
        return out

    # -- slide removal ------------------------------------------------------------

    def remove_slide(self, index: int) -> None:
        """Remove the slide at *index* and everything only it used.

        Beyond dropping the ``p:sldId`` entry and the slide part itself
        (plus its rels and notes slide), this reference-counts every
        part transitively reachable from the removed slide — charts,
        embedded chart workbooks, images, chart colors/style parts —
        against the relationships of everything still in the package,
        and deletes the now-orphaned ones, including their content-type
        overrides. Parts shared with surviving slides (or anchored by
        the presentation / masters, like layouts and the notes master)
        are untouched, and surviving slide parts stay byte-identical.
        """
        slides = self.slides
        if not 0 <= index < len(slides):
            raise IndexError(f"slide index {index} out of range (0..{len(slides) - 1})")
        part_name = slides[index].part_name

        pres = self.xml_part(_PRESENTATION)
        pres_rels = self.package.rels_for(_PRESENTATION)
        if pres_rels is None:  # pragma: no cover - malformed package
            raise ValueError("presentation has no relationships part")
        rid = next(
            (
                rel["id"]
                for rel in pres_rels.by_type("/slide")
                if pres_rels.resolve(_PRESENTATION, rel["target"]) == part_name
            ),
            None,
        )
        sld_id_lst = pres.find("p:sldIdLst")
        if sld_id_lst is not None and rid is not None:
            for sld_id in list(sld_id_lst):
                if sld_id.get(qn("r:id")) == rid:
                    sld_id_lst.remove(sld_id)
            pres.mark_dirty()
        if rid is not None:
            pres_rels.remove(rid)

        # The slide dies; so does its notes slide.
        doomed = [part_name]
        slide_rels = self.package.rels_for(part_name)
        if slide_rels is not None:
            doomed += [
                slide_rels.resolve(part_name, rel["target"])
                for rel in slide_rels.by_type("/notesSlide")
            ]

        # Everything transitively reachable from the doomed parts is an
        # orphan *candidate* — collected before any rels disappear.
        candidates: set[str] = set()
        stack, visited = list(doomed), set(doomed)
        while stack:
            src = stack.pop()
            rels = self.package.rels_for(src)
            if rels is None:
                continue
            for rel in rels:
                if rel["mode"] == "External":
                    continue
                target = rels.resolve(src, rel["target"])
                if target not in visited:
                    visited.add(target)
                    candidates.add(target)
                    stack.append(target)

        removed: list[str] = []
        for name in doomed:
            removed += self._delete_part(name)

        # Sweep candidates no remaining part references, to a fixpoint
        # (deleting a chart un-anchors its embedded workbook, etc.).
        while True:
            referenced = self._referenced_parts()
            orphans = [
                c
                for c in sorted(candidates)
                if self.package.has_part(c) and c not in referenced
            ]
            if not orphans:
                break
            for name in orphans:
                removed += self._delete_part(name)
                candidates.discard(name)

        self._drop_content_type_overrides(removed)

    # -- internals ------------------------------------------------------------------

    def _delete_part(self, name: str) -> list[str]:
        """Remove *name* and its rels part; returns what was removed."""
        removed = []
        if self.package.has_part(name):
            self.package.remove_part(name)
            removed.append(name)
        rels_name = OpcPackage._rels_name_for(name)
        if self.package.has_part(rels_name):
            self.package.remove_part(rels_name)
            removed.append(rels_name)
        self._xml_parts.pop(name, None)
        self._xml_parts.pop(rels_name, None)
        return removed

    def _referenced_parts(self) -> set[str]:
        """Internal targets of every relationships part still present."""
        referenced: set[str] = set()
        for rels_name in list(self.package.part_names):
            if not rels_name.endswith(".rels"):
                continue
            directory, base = posixpath.split(rels_name)
            owner_dir = posixpath.dirname(directory)
            owner_base = base[: -len(".rels")]
            owner = posixpath.join(owner_dir, owner_base) if owner_base else ""
            rels = self.package.rels_for(owner)
            if rels is None:
                continue
            for rel in rels:
                if rel["mode"] == "External":
                    continue
                referenced.add(rels.resolve(owner, rel["target"]))
        return referenced

    def _drop_content_type_overrides(self, part_names: list[str]) -> None:
        from lxml import etree

        ct_part = self.package.get_part("[Content_Types].xml")
        root = etree.fromstring(ct_part.read())
        doomed = {f"/{name}" for name in part_names}
        changed = False
        for el in list(root):
            if el.tag == qn("ct:Override") and el.get("PartName") in doomed:
                root.remove(el)
                changed = True
        if changed:
            ct_part.write(
                etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8", standalone=True
                )
            )
