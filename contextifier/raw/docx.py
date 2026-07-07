# contextifier/raw/docx.py
"""
DocxRawDocument — the raw (lossless, addressable, writable) model for
WordprocessingML documents.

Addressing invariant
--------------------
``paragraphs[i]`` is the *i*-th direct ``w:p`` child of ``w:body`` and
``tables[t]`` is the *t*-th direct ``w:tbl`` child — exactly the
elements python-docx exposes as ``document.paragraphs`` /
``document.tables``. Table cells use **grid addressing** with the same
semantics as python-docx ``row.cells``: a horizontally merged cell
(``w:gridSpan``) occupies every grid column it spans, and a vertically
merged continuation (``w:vMerge``) resolves to its start cell. This
keeps raw addresses interchangeable with edit2docs' existing addresses.

Run-preserving text replacement (the P0-3 fix)
----------------------------------------------
``set_paragraph_text`` never rebuilds a paragraph. Within the
paragraph, runs are classified:

* **text runs** — ``w:r`` containing ``w:t`` and *no* protected content
  (``w:drawing`` / ``w:pict`` / ``w:object`` / ``mc:AlternateContent``);
* **protected elements** — runs holding drawings/pictures/OLE, plus
  every non-run child (bookmarks, math, comment ranges, revision
  containers, ...). These are left in place, in order, untouched.

The new text goes into the **first direct text run** (its ``w:rPr`` is
kept; ``w:t`` gets ``xml:space="preserve"`` when the text has leading
or trailing whitespace); the other direct pure-text runs are deleted.

Hyperlink policy: ``w:hyperlink`` elements are never deleted by a text
replace. If the paragraph has no direct text run, the new text goes
into the *first hyperlink's first text run* (the hyperlink itself is
preserved). All other pure-text runs inside hyperlinks are emptied, but
the (now empty) hyperlink elements remain — callers that want them gone
use :meth:`DocxRawDocument.strip_empty_hyperlinks`, which also drops
the hyperlink relationship when nothing else references it.

``RawCell.set_text`` applies the same rules per paragraph: the first
paragraph that owns a text run becomes the carrier, paragraphs with
protected content (inline images, ...) are left completely untouched,
and the remaining pure-text paragraphs are emptied — never removed, so
cell layout, nested tables and images survive (the edit2docs
cell-image-destruction fix).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from contextifier.raw.base import RawDocumentBase
from contextifier.raw.chart import ChartModel, find_chart_parts
from contextifier.raw.xmlpart import NS, qn

if TYPE_CHECKING:  # pragma: no cover
    from lxml.etree import _Element

    from contextifier.raw.opc import OpcPackage

__all__ = ["DocxRawDocument", "RawParagraph", "RawTable", "RawCell"]

_DOCUMENT_PART = "word/document.xml"

#: Content that makes a run (or a paragraph) non-replaceable.
_PROTECTED_TAGS = (
    qn("w:drawing"),
    qn("w:pict"),
    qn("w:object"),
    qn("mc:AlternateContent"),
)
_TEXTUAL_TAGS = (qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr"))


# -- element-level helpers ---------------------------------------------------


def _contains_protected(el: "_Element") -> bool:
    return next(el.iter(*_PROTECTED_TAGS), None) is not None


def _is_text_run(el: "_Element") -> bool:
    """A pure-text run: ``w:r`` with a ``w:t`` and no protected content."""
    return (
        el.tag == qn("w:r")
        and next(el.iter(qn("w:t")), None) is not None
        and not _contains_protected(el)
    )


def _has_del_ancestor(node: "_Element", stop: "_Element") -> bool:
    parent = node.getparent()
    while parent is not None and parent is not stop:
        if parent.tag == qn("w:del"):
            return True
        parent = parent.getparent()
    return False


def _element_text(el: "_Element") -> str:
    """Visible text of *el* (python-docx ``.text`` semantics).

    Concatenates ``w:t`` anywhere below *el* — including inside
    ``w:hyperlink`` and ``w:ins`` — excluding deleted (``w:del``)
    content; run-level ``w:tab``/``w:br``/``w:cr`` render as
    ``"\\t"``/``"\\n"``.
    """
    parts: list[str] = []
    for node in el.iter(*_TEXTUAL_TAGS):
        if _has_del_ancestor(node, el):
            continue
        if node.tag == qn("w:t"):
            parts.append(node.text or "")
        elif node.getparent() is not None and node.getparent().tag == qn("w:r"):
            parts.append("\t" if node.tag == qn("w:tab") else "\n")
    return "".join(parts)


def _set_run_text(run: "_Element", text: str) -> None:
    """Replace the run's content with a single ``w:t``, keeping ``w:rPr``."""
    from lxml import etree

    for child in list(run):
        if child.tag != qn("w:rPr"):
            run.remove(child)
    t = etree.SubElement(run, qn("w:t"))
    t.text = text
    if text != text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def _clear_run_text(run: "_Element") -> None:
    """Empty a pure-text run (drop everything but ``w:rPr``)."""
    for child in list(run):
        if child.tag != qn("w:rPr"):
            run.remove(child)


def _paragraph_text_runs(p_el: "_Element") -> tuple[list, list[tuple]]:
    """(direct pure-text runs, [(hyperlink, its pure-text runs), ...])."""
    direct = [r for r in p_el.findall(qn("w:r")) if _is_text_run(r)]
    in_links = [
        (h, [r for r in h.findall(qn("w:r")) if _is_text_run(r)])
        for h in p_el.findall(qn("w:hyperlink"))
    ]
    return direct, in_links


def _has_text_run(p_el: "_Element") -> bool:
    direct, in_links = _paragraph_text_runs(p_el)
    return bool(direct) or any(runs for _, runs in in_links)


def _replace_paragraph_text(p_el: "_Element", text: str) -> None:
    """The run-preserving replace documented in the module docstring."""
    from lxml import etree

    direct, in_links = _paragraph_text_runs(p_el)

    carrier = direct[0] if direct else None
    if carrier is None:
        for _, runs in in_links:
            if runs:
                carrier = runs[0]
                break
    if carrier is None:  # no text run anywhere — append a fresh one
        carrier = etree.SubElement(p_el, qn("w:r"))

    _set_run_text(carrier, text)
    for run in direct:
        if run is not carrier:
            p_el.remove(run)
    for _, runs in in_links:
        for run in runs:
            if run is not carrier:
                _clear_run_text(run)


def _clear_paragraph_text(p_el: "_Element") -> None:
    """Empty a paragraph's text: delete direct pure-text runs, empty the
    text runs inside hyperlinks (hyperlink elements stay)."""
    direct, in_links = _paragraph_text_runs(p_el)
    for run in direct:
        p_el.remove(run)
    for _, runs in in_links:
        for run in runs:
            _clear_run_text(run)


def _clear_row_copy(tr_el: "_Element") -> None:
    """Blank a deep-copied row template: clear every paragraph's text and
    drop the (copied) hyperlink elements. Cell count, ``w:trPr``,
    ``w:tcPr`` (shading, spans, merges) all stay."""
    for p_el in tr_el.iter(qn("w:p")):
        _clear_paragraph_text(p_el)
        for h in list(p_el.findall(qn("w:hyperlink"))):
            p_el.remove(h)


# -- model classes -------------------------------------------------------------


class RawParagraph:
    """A body-level paragraph (positional snapshot — re-fetch
    ``document.paragraphs`` after structural edits)."""

    __slots__ = ("element", "index")

    def __init__(self, element: "_Element", index: int):
        self.element = element
        self.index = index

    @property
    def text(self) -> str:
        return _element_text(self.element)

    @property
    def style(self) -> str:
        """The ``w:pStyle`` style *id* (e.g. ``"Heading1"``), or ``"Normal"``."""
        style = self.element.find(f"{qn('w:pPr')}/{qn('w:pStyle')}")
        return style.get(qn("w:val")) if style is not None else "Normal"

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<RawParagraph {self.index} {self.text[:40]!r}>"


class RawCell:
    """One (grid-resolved) table cell."""

    __slots__ = ("element", "_doc")

    def __init__(self, element: "_Element", doc: "DocxRawDocument"):
        self.element = element
        self._doc = doc

    def _paragraph_els(self) -> list:
        return self.element.findall(qn("w:p"))

    @property
    def text(self) -> str:
        """python-docx parity: direct paragraphs joined with newlines
        (nested-table text is *not* included)."""
        return "\n".join(_element_text(p) for p in self._paragraph_els())

    @property
    def paragraph_count(self) -> int:
        return len(self._paragraph_els())

    def set_text(self, text: str) -> None:
        """Run- and layout-preserving text replace (see module docstring).

        The carrier is the first paragraph that owns a text run (else
        the first paragraph without protected content, else a new
        trailing paragraph). Paragraphs containing drawings / pictures /
        OLE stay byte-for-byte untouched; other pure-text paragraphs are
        emptied but never removed, so images and nested tables keep
        their positions.
        """
        from lxml import etree

        paras = self._paragraph_els()
        target = next((p for p in paras if _has_text_run(p)), None)
        if target is None:
            target = next((p for p in paras if not _contains_protected(p)), None)
        if target is None:
            target = etree.SubElement(self.element, qn("w:p"))
        _replace_paragraph_text(target, text)
        for p in paras:
            if p is not target and not _contains_protected(p):
                _clear_paragraph_text(p)
        self._doc._mark_document_dirty()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<RawCell {self.text[:40]!r}>"


class RawTable:
    """A table (body-level or nested) with grid-resolved cell access."""

    __slots__ = ("element", "index", "_doc")

    def __init__(self, element: "_Element", index: int, doc: "DocxRawDocument"):
        self.element = element
        self.index = index
        self._doc = doc

    def _row_els(self) -> list:
        return self.element.findall(qn("w:tr"))

    @property
    def n_rows(self) -> int:
        return len(self._row_els())

    @property
    def n_cols(self) -> int:
        grid = self.element.find(qn("w:tblGrid"))
        return len(grid.findall(qn("w:gridCol"))) if grid is not None else 0

    # -- grid resolution -----------------------------------------------------

    def _grid(self) -> list[list]:
        """``matrix[r][c]`` → the ``w:tc`` owning grid position (r, c).

        gridSpan repeats a cell across the columns it spans; a vMerge
        continuation resolves to the start cell of the merge — the same
        answers python-docx gives for ``table.rows[r].cells[c]``.
        """
        rows = self._row_els()
        n_cols = self.n_cols
        matrix: list[list] = [[None] * n_cols for _ in rows]
        for ri, tr in enumerate(rows):
            ci = 0
            before = tr.find(f"{qn('w:trPr')}/{qn('w:gridBefore')}")
            if before is not None:
                ci = int(before.get(qn("w:val"), 0))
            for tc in tr.findall(qn("w:tc")):
                tc_pr = tc.find(qn("w:tcPr"))
                span, resolved = 1, tc
                if tc_pr is not None:
                    grid_span = tc_pr.find(qn("w:gridSpan"))
                    if grid_span is not None:
                        span = int(grid_span.get(qn("w:val"), 1))
                    v_merge = tc_pr.find(qn("w:vMerge"))
                    if (
                        v_merge is not None
                        and v_merge.get(qn("w:val"), "continue") == "continue"
                        and ri > 0
                        and ci < n_cols
                        and matrix[ri - 1][ci] is not None
                    ):
                        resolved = matrix[ri - 1][ci]
                for k in range(span):
                    if ci + k < n_cols:
                        matrix[ri][ci + k] = resolved
                ci += span
        return matrix

    def _tc_at(self, r: int, c: int) -> "_Element":
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            raise IndexError(
                f"cell ({r}, {c}) out of range for {self.n_rows}x{self.n_cols} table"
            )
        tc = self._grid()[r][c]
        if tc is None:
            raise IndexError(f"grid position ({r}, {c}) has no cell")
        return tc

    def cell(self, r: int, c: int) -> RawCell:
        return RawCell(self._tc_at(r, c), self._doc)

    def nested_tables(self, r: int, c: int) -> list["RawTable"]:
        """Tables directly inside cell (r, c), addressable like any table."""
        return [
            RawTable(el, i, self._doc)
            for i, el in enumerate(self._tc_at(r, c).findall(qn("w:tbl")))
        ]

    # -- row editing --------------------------------------------------------

    def insert_row(self, idx: int) -> None:
        """Insert a blank row at *idx* (0 ≤ idx ≤ n_rows), deep-copying the
        row above (row 0 for idx=0) as the template: cell count and all
        row/cell properties carry over, text is cleared."""
        rows = self._row_els()
        if not rows:
            raise ValueError("cannot insert into a table with no rows")
        if not 0 <= idx <= len(rows):
            raise IndexError(f"row index {idx} out of range 0..{len(rows)}")
        template = rows[idx - 1] if idx > 0 else rows[0]
        new_tr = copy.deepcopy(template)
        _clear_row_copy(new_tr)
        if idx == len(rows):
            rows[-1].addnext(new_tr)
        else:
            rows[idx].addprevious(new_tr)
        self._doc._mark_document_dirty()

    def delete_row(self, idx: int) -> None:
        rows = self._row_els()
        if not 0 <= idx < len(rows):
            raise IndexError(f"row index {idx} out of range 0..{len(rows) - 1}")
        self.element.remove(rows[idx])
        self._doc._mark_document_dirty()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<RawTable {self.index} {self.n_rows}x{self.n_cols}>"


class DocxRawDocument(RawDocumentBase):
    """Raw model over ``word/document.xml`` (plus header/footer/chart
    parts). Everything the model doesn't touch round-trips
    byte-identically via the OPC container."""

    format = "docx"

    def __init__(self, package: "OpcPackage"):
        super().__init__(package)
        if not package.has_part(_DOCUMENT_PART):
            from contextifier.raw.opc import RawUnsupportedError

            raise RawUnsupportedError("Package has no word/document.xml — not a docx")
        self._document = self.xml_part(_DOCUMENT_PART)
        self._charts: list[ChartModel] | None = None

    # -- internals -----------------------------------------------------------

    @property
    def _body(self) -> "_Element":
        return self._document.find("w:body")

    def _mark_document_dirty(self) -> None:
        self._document.mark_dirty()

    def _paragraph_els(self) -> list:
        return self._body.findall(qn("w:p"))

    def _paragraph_el(self, index: int) -> "_Element":
        els = self._paragraph_els()
        if not 0 <= index < len(els):
            raise IndexError(f"paragraph index {index} out of range 0..{len(els) - 1}")
        return els[index]

    # -- paragraphs -----------------------------------------------------------

    @property
    def paragraphs(self) -> list[RawParagraph]:
        """Body-level paragraphs, indexed like python-docx
        ``document.paragraphs``."""
        return [RawParagraph(el, i) for i, el in enumerate(self._paragraph_els())]

    def set_paragraph_text(self, index: int, text: str) -> None:
        """Run-preserving text replace (see module docstring): first text
        run carries the new text with its formatting, protected runs
        (images, OLE, bookmarks, math, ...) stay in place, hyperlink
        elements survive (possibly with empty text)."""
        _replace_paragraph_text(self._paragraph_el(index), text)
        self._mark_document_dirty()

    def strip_empty_hyperlinks(self, index: int) -> int:
        """Remove the paragraph's ``w:hyperlink`` children whose text is
        empty (e.g. after :meth:`set_paragraph_text`). Relationships no
        longer referenced anywhere in the document part are dropped too.
        Returns the number of hyperlinks removed."""
        p_el = self._paragraph_el(index)
        removed_ids: list[str] = []
        removed = 0
        for h in list(p_el.findall(qn("w:hyperlink"))):
            if _element_text(h) == "":
                rid = h.get(qn("r:id"))
                p_el.remove(h)
                removed += 1
                if rid:
                    removed_ids.append(rid)
        if removed:
            self._mark_document_dirty()
        if removed_ids:
            rels = self.package.rels_for(_DOCUMENT_PART)
            r_ns = "{%s}" % NS["r"]
            for rid in removed_ids:
                still_used = any(
                    value == rid
                    for el in self._document.root.iter()
                    for key, value in el.attrib.items()
                    if key.startswith(r_ns)
                )
                if rels is not None and not still_used:
                    rels.remove(rid)
        return removed

    def insert_paragraph_after(
        self, index: int, text: str, style: str | None = None
    ) -> RawParagraph:
        """Insert a new paragraph after ``paragraphs[index]``; ``index=-1``
        inserts at the very start of the body. Safe with a trailing body
        ``w:sectPr`` (insertion is always anchored to an existing
        paragraph, never appended past it)."""
        from lxml import etree

        new_p = etree.Element(qn("w:p"))
        if style:
            p_pr = etree.SubElement(new_p, qn("w:pPr"))
            p_style = etree.SubElement(p_pr, qn("w:pStyle"))
            p_style.set(qn("w:val"), style)
        run = etree.SubElement(new_p, qn("w:r"))
        _set_run_text(run, text)

        if index == -1:
            self._body.insert(0, new_p)
            new_index = 0
        else:
            self._paragraph_el(index).addnext(new_p)
            new_index = index + 1
        self._mark_document_dirty()
        return RawParagraph(new_p, new_index)

    def delete_paragraph(self, index: int) -> None:
        el = self._paragraph_el(index)
        el.getparent().remove(el)
        self._mark_document_dirty()

    # -- tables ---------------------------------------------------------------

    @property
    def tables(self) -> list[RawTable]:
        """Body-level tables, indexed like python-docx ``document.tables``
        (nested tables are reached via :meth:`RawTable.nested_tables`)."""
        return [
            RawTable(el, i, self)
            for i, el in enumerate(self._body.findall(qn("w:tbl")))
        ]

    # -- charts ---------------------------------------------------------------

    @property
    def chart_part_names(self) -> list[str]:
        """Chart parts referenced from the document part's rels."""
        return find_chart_parts(self.package, _DOCUMENT_PART)

    @property
    def charts(self) -> list[ChartModel]:
        """Lazy :class:`ChartModel` per chart part (shared C3 contract)."""
        if self._charts is None:
            self._charts = [
                ChartModel(self.xml_part(name), self.package)
                for name in self.chart_part_names
            ]
        return self._charts

    # -- headers / footers ------------------------------------------------------

    def _header_footer_text(self, type_suffix: str) -> dict[str, str]:
        rels = self.package.rels_for(_DOCUMENT_PART)
        if rels is None:
            return {}
        out: dict[str, str] = {}
        for rel in rels.by_type(type_suffix):
            name = rels.resolve(_DOCUMENT_PART, rel["target"])
            root = self.xml_part(name).root
            out[name] = "\n".join(_element_text(p) for p in root.iter(qn("w:p")))
        return out

    @property
    def headers(self) -> dict[str, str]:
        """Read-only ``{part name: text}`` for the header parts."""
        return self._header_footer_text("/header")

    @property
    def footers(self) -> dict[str, str]:
        """Read-only ``{part name: text}`` for the footer parts."""
        return self._header_footer_text("/footer")

    # -- structure ---------------------------------------------------------------

    def body_order(self) -> list[tuple[str, int]]:
        """Document order of body children as ``("p"|"tbl", index)`` pairs —
        the skeleton outline builders walk."""
        out: list[tuple[str, int]] = []
        p_i = t_i = 0
        for child in self._body:
            if child.tag == qn("w:p"):
                out.append(("p", p_i))
                p_i += 1
            elif child.tag == qn("w:tbl"):
                out.append(("tbl", t_i))
                t_i += 1
        return out
