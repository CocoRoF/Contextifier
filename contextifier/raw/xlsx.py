# contextifier/raw/xlsx.py
"""
XLSX raw document model — SpreadsheetML semantics on the OPC container.

:class:`XlsxRawDocument` gives addressable, *surgical* read/write access
to a workbook. Only the worksheet XML you actually touch (plus, when
needed, ``xl/workbook.xml`` for ``calcPr``) is ever re-serialized —
every other part rides the byte-preservation contract untouched, so
charts, chart styles, pivot tables, sparkline ``extLst`` blocks, custom
XML, themes and styles all survive a save (openpyxl round-trips destroy
all of these).

Design decisions:

* **String writes use inline strings** (``t="inlineStr"``), never the
  shared-string table — so ``xl/sharedStrings.xml`` is never rewritten
  (and never *created*; a workbook without one stays without one).
* **Overwriting a formula cell removes the formula.** ``set_cell`` is
  "the user supplies the value now": the ``<f>`` element is dropped and
  the literal is written. Use :meth:`RawSheet.get_formula` to inspect
  formulas before overwriting.
* **Stale caches recalculate on open.** Whenever a ``set_cell`` /
  ``append_rows`` overwrites a formula cell — or any formula exists
  anywhere in the workbook (its cached result may now be stale) — the
  workbook's ``<calcPr>`` gains ``fullCalcOnLoad="1"`` (created after
  the last of sheets/definedNames if absent) so Excel recalculates.
* **Values pass through as stored.** No date-system handling (1900 vs
  1904 is ignored); date cells read back as their serial numbers.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Iterator

from contextifier.raw.base import RawDocumentBase
from contextifier.raw.xmlpart import XmlPart, qn

if TYPE_CHECKING:  # pragma: no cover
    from lxml.etree import _Element

    from contextifier.raw.chart import ChartModel
    from contextifier.raw.opc import OpcPackage

__all__ = ["XlsxRawDocument", "RawSheet", "SheetCollection"]

#: attribute key for ``xml:space``
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

_REF_RE = re.compile(r"^\$?([A-Za-z]{1,3})\$?([1-9][0-9]*)$")
_RANGE_RE = re.compile(
    r"^\$?([A-Za-z]{1,3})\$?([1-9][0-9]*)(?::\$?([A-Za-z]{1,3})\$?([1-9][0-9]*))?$"
)

#: workbook children that must precede <calcPr> (CT_Workbook sequence)
_PRE_CALC_PR = (
    "fileVersion",
    "fileSharing",
    "workbookPr",
    "workbookProtection",
    "bookViews",
    "sheets",
    "functionGroups",
    "externalReferences",
    "definedNames",
)


# -- A1-reference helpers ----------------------------------------------------


def col_letters_to_index(letters: str) -> int:
    """``"A"`` → 1, ``"B"`` → 2, ..., ``"AA"`` → 27."""
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - 64)
    return idx


def col_index_to_letters(idx: int) -> str:
    """1 → ``"A"``, 27 → ``"AA"``."""
    if idx < 1:
        raise ValueError(f"Column index must be >= 1, got {idx}")
    out: list[str] = []
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out.append(chr(65 + rem))
    return "".join(reversed(out))


def parse_ref(ref: str) -> tuple[int, int]:
    """``"B3"`` → ``(3, 2)`` (row, column), 1-based. ``$`` anchors OK."""
    m = _REF_RE.match(ref.strip())
    if m is None:
        raise ValueError(f"Not a cell reference: {ref!r}")
    return int(m.group(2)), col_letters_to_index(m.group(1))


def _row_number(row_el: "_Element", prev: int) -> int:
    r = row_el.get("r")
    return int(r) if r else prev + 1


def _cell_column(c_el: "_Element", prev: int) -> int:
    r = c_el.get("r")
    if r:
        m = _REF_RE.match(r)
        if m is not None:
            return col_letters_to_index(m.group(1))
    return prev + 1


class RawSheet:
    """One worksheet: addressable cells over the live lxml tree.

    Reads never dirty the part; every mutation marks it dirty so
    ``save()`` re-serializes exactly this worksheet and nothing else.
    """

    def __init__(self, doc: "XlsxRawDocument", name: str, part_name: str):
        self._doc = doc
        self.name = name
        self.part_name = part_name

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<RawSheet {self.name!r} ({self.part_name})>"

    @property
    def _xp(self) -> XmlPart:
        return self._doc.xml_part(self.part_name)

    def _sheet_data(self) -> "_Element":
        sd = self._xp.find("s:sheetData")
        if sd is None:  # degenerate but legal: create the container
            sd = self._xp.root.makeelement(qn("s:sheetData"), {})
            self._xp.root.append(sd)
        return sd

    # -- reading ---------------------------------------------------------------

    def get_cell(self, ref: str) -> object | None:
        """The cell's value (formula cells: the *cached* value), or None.

        Types follow the cell's ``t``: shared/inline strings → str,
        ``b`` → bool, ``str``/``e`` → str, numeric → int when integral
        else float. Missing row/cell (or a value-less cell) → None.
        """
        row_idx, col_idx = parse_ref(ref)
        row = self._find_row(row_idx)
        if row is None:
            return None
        cell = self._find_cell(row, col_idx)
        if cell is None:
            return None
        return self._read_value(cell)

    def get_formula(self, ref: str) -> str | None:
        """The cell's formula text (``<f>``, without the leading ``=``),
        or None if the cell has no formula."""
        row_idx, col_idx = parse_ref(ref)
        row = self._find_row(row_idx)
        if row is None:
            return None
        cell = self._find_cell(row, col_idx)
        if cell is None:
            return None
        f = cell.find(qn("s:f"))
        return None if f is None else (f.text or "")

    @property
    def dimensions(self) -> tuple[int, int]:
        """``(max_row, max_col)`` actually present in sheetData; (0, 0)
        for an empty sheet."""
        max_row = max_col = 0
        prev_row = 0
        for row_el in self._sheet_data():
            if row_el.tag != qn("s:row"):
                continue
            prev_row = _row_number(row_el, prev_row)
            max_row = max(max_row, prev_row)
            prev_col = 0
            for c_el in row_el:
                if c_el.tag != qn("s:c"):
                    continue
                prev_col = _cell_column(c_el, prev_col)
                max_col = max(max_col, prev_col)
        return (max_row, max_col)

    @property
    def merged_ranges(self) -> list[str]:
        """The sheet's merged ranges, e.g. ``["A5:B5"]``."""
        return [
            mc.get("ref")
            for mc in self._xp.findall("s:mergeCells/s:mergeCell")
            if mc.get("ref")
        ]

    def iter_rows(
        self, min_row: int = 1, max_row: int | None = None
    ) -> Iterator[tuple[str, object | None]]:
        """Yield ``(ref, value)`` for every stored cell in row order."""
        prev_row = 0
        for row_el in self._sheet_data():
            if row_el.tag != qn("s:row"):
                continue
            prev_row = _row_number(row_el, prev_row)
            if prev_row < min_row:
                continue
            if max_row is not None and prev_row > max_row:
                break  # rows are kept sorted by @r
            prev_col = 0
            for c_el in row_el:
                if c_el.tag != qn("s:c"):
                    continue
                prev_col = _cell_column(c_el, prev_col)
                ref = c_el.get("r") or f"{col_index_to_letters(prev_col)}{prev_row}"
                yield ref, self._read_value(c_el)

    # -- writing ---------------------------------------------------------------

    def set_cell(self, ref: str, value: object) -> None:
        """Surgically write one cell, preserving everything else.

        * str → ``t="inlineStr"`` (sharedStrings is never touched)
        * bool → ``t="b"``, int/float → plain numeric, None → value
          removed (the cell and its style stay)
        * The cell's style attribute ``s`` is preserved.
        * An existing ``<f>`` formula is **removed** — set_cell means
          "this literal is the value now". ``fullCalcOnLoad`` is then
          ensured so Excel recalculates any dependents on open.
        """
        row_idx, col_idx = parse_ref(ref)
        row = self._find_row(row_idx)
        if row is None:
            row = self._insert_row(row_idx)
        cell = self._find_cell(row, col_idx)
        if cell is None:
            cell = self._insert_cell(row, row_idx, col_idx)
        had_formula = cell.find(qn("s:f")) is not None
        self._write_value(cell, value)
        self._xp.mark_dirty()
        self._extend_dimension(row_idx, col_idx)
        if had_formula or self._doc._any_formula_exists():
            self._doc._ensure_full_calc_on_load()

    def append_rows(self, rows: list[list]) -> None:
        """Append *rows* after the last existing row. ``None`` entries
        leave their cell unstored (sparse), matching Excel semantics."""
        sd = self._sheet_data()
        last = self.dimensions[0]
        for offset, values in enumerate(rows, start=1):
            row_idx = last + offset
            row_el = sd.makeelement(qn("s:row"), {"r": str(row_idx)})
            sd.append(row_el)
            for col_idx, value in enumerate(values, start=1):
                if value is None:
                    continue
                ref = f"{col_index_to_letters(col_idx)}{row_idx}"
                c_el = row_el.makeelement(qn("s:c"), {"r": ref})
                row_el.append(c_el)
                self._write_value(c_el, value)
            if values:
                self._extend_dimension(row_idx, len(values))
            else:
                self._extend_dimension(row_idx, 1)
        if rows:
            self._xp.mark_dirty()
            if self._doc._any_formula_exists():
                self._doc._ensure_full_calc_on_load()

    # -- internals ---------------------------------------------------------------

    def _find_row(self, row_idx: int) -> "_Element | None":
        prev = 0
        for row_el in self._sheet_data():
            if row_el.tag != qn("s:row"):
                continue
            prev = _row_number(row_el, prev)
            if prev == row_idx:
                return row_el
            if prev > row_idx:
                return None  # rows sorted by @r
        return None

    def _find_cell(self, row_el: "_Element", col_idx: int) -> "_Element | None":
        prev = 0
        for c_el in row_el:
            if c_el.tag != qn("s:c"):
                continue
            prev = _cell_column(c_el, prev)
            if prev == col_idx:
                return c_el
            if prev > col_idx:
                return None  # cells sorted by column
        return None

    def _insert_row(self, row_idx: int) -> "_Element":
        """Create ``<row r="...">`` keeping sheetData sorted by @r
        (Excel requires ascending rows)."""
        sd = self._sheet_data()
        new = sd.makeelement(qn("s:row"), {"r": str(row_idx)})
        prev = 0
        for row_el in sd:
            if row_el.tag != qn("s:row"):
                continue
            prev = _row_number(row_el, prev)
            if prev > row_idx:
                row_el.addprevious(new)
                return new
        sd.append(new)
        return new

    def _insert_cell(
        self, row_el: "_Element", row_idx: int, col_idx: int
    ) -> "_Element":
        """Create ``<c r="...">`` keeping the row's cells column-sorted."""
        ref = f"{col_index_to_letters(col_idx)}{row_idx}"
        new = row_el.makeelement(qn("s:c"), {"r": ref})
        prev = 0
        for c_el in row_el:
            if c_el.tag != qn("s:c"):
                continue
            prev = _cell_column(c_el, prev)
            if prev > col_idx:
                c_el.addprevious(new)
                return new
        row_el.append(new)
        return new

    def _read_value(self, c_el: "_Element") -> object | None:
        t = c_el.get("t", "n")
        if t == "inlineStr":
            is_el = c_el.find(qn("s:is"))
            if is_el is None:
                return None
            return "".join(tel.text or "" for tel in is_el.iter(qn("s:t")))
        v = c_el.find(qn("s:v"))
        if v is None or v.text is None:
            return None
        text = v.text
        if t == "s":
            strings = self._doc._shared_strings()
            try:
                return strings[int(text)]
            except (ValueError, IndexError):
                return None
        if t == "b":
            return text.strip() in ("1", "true", "TRUE")
        if t in ("str", "e"):
            return text
        # default: numeric
        try:
            return int(text)
        except ValueError:
            f = float(text)
            return int(f) if f.is_integer() else f

    def _write_value(self, c_el: "_Element", value: object) -> None:
        """Replace the cell's content, preserving its style (@s).

        Any existing formula is removed — overwriting means the caller
        supplies the value from now on."""
        for tag in ("s:f", "s:v", "s:is"):
            el = c_el.find(qn(tag))
            if el is not None:
                c_el.remove(el)
        if value is None:
            c_el.attrib.pop("t", None)
            return
        if isinstance(value, bool):  # before int: bool is an int subclass
            c_el.set("t", "b")
            v = c_el.makeelement(qn("s:v"), {})
            v.text = "1" if value else "0"
            c_el.append(v)
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Cannot store non-finite float {value!r}")
            c_el.attrib.pop("t", None)  # default numeric type
            v = c_el.makeelement(qn("s:v"), {})
            v.text = repr(value) if isinstance(value, float) else str(value)
            c_el.append(v)
        elif isinstance(value, str):
            c_el.set("t", "inlineStr")
            is_el = c_el.makeelement(qn("s:is"), {})
            t_el = is_el.makeelement(qn("s:t"), {_XML_SPACE: "preserve"})
            t_el.text = value
            is_el.append(t_el)
            c_el.append(is_el)
        else:
            raise TypeError(
                f"Unsupported cell value type {type(value).__name__!r} "
                "(str, bool, int, float or None)"
            )

    def _extend_dimension(self, row_idx: int, col_idx: int) -> None:
        """Grow the sheet's ``<dimension ref>`` to include the cell."""
        dim = self._xp.find("s:dimension")
        if dim is None:
            return
        m = _RANGE_RE.match(dim.get("ref") or "")
        if m is None:
            return
        min_c, min_r = col_letters_to_index(m.group(1)), int(m.group(2))
        if m.group(3):
            max_c, max_r = col_letters_to_index(m.group(3)), int(m.group(4))
        else:
            max_c, max_r = min_c, min_r
        new_min_c, new_min_r = min(min_c, col_idx), min(min_r, row_idx)
        new_max_c, new_max_r = max(max_c, col_idx), max(max_r, row_idx)
        if (new_min_c, new_min_r, new_max_c, new_max_r) != (
            min_c,
            min_r,
            max_c,
            max_r,
        ):
            dim.set(
                "ref",
                f"{col_index_to_letters(new_min_c)}{new_min_r}"
                f":{col_index_to_letters(new_max_c)}{new_max_r}",
            )
            self._xp.mark_dirty()


class SheetCollection:
    """Mapping-like sheet accessor: by name (``raw.sheets["Sales"]``) or
    by position (``raw.sheets[0]``)."""

    def __init__(self, doc: "XlsxRawDocument"):
        self._doc = doc

    def __getitem__(self, key: int | str) -> RawSheet:
        entries = self._doc._sheet_entries()
        if isinstance(key, int):
            name, part = entries[key]  # IndexError speaks for itself
        else:
            for name, part in entries:
                if name == key:
                    break
            else:
                raise KeyError(f"No sheet named {key!r}")
        return self._doc._sheet(name, part)

    def __len__(self) -> int:
        return len(self._doc._sheet_entries())

    def __iter__(self) -> Iterator[RawSheet]:
        for name, part in self._doc._sheet_entries():
            yield self._doc._sheet(name, part)

    def __contains__(self, name: object) -> bool:
        return any(n == name for n, _ in self._doc._sheet_entries())


class XlsxRawDocument(RawDocumentBase):
    """Lossless, writable view of an .xlsx workbook."""

    format = "xlsx"

    def __init__(self, package: "OpcPackage"):
        super().__init__(package)
        self._workbook_name = self._locate_workbook()
        self._sheet_cache: dict[str, RawSheet] = {}
        self._shared_cache: list[str] | None = None
        self._formulas_exist: bool | None = None

    def _locate_workbook(self) -> str:
        rels = self.package.rels_for("")
        if rels is not None:
            for rel in rels.by_type("/officeDocument"):
                return rels.resolve("", rel["target"])
        return "xl/workbook.xml"

    @property
    def workbook(self) -> XmlPart:
        """The ``xl/workbook.xml`` facade."""
        return self.xml_part(self._workbook_name)

    # -- sheets ---------------------------------------------------------------

    @property
    def sheet_names(self) -> list[str]:
        """Sheet names in workbook (tab) order."""
        return [s.get("name") or "" for s in self.workbook.findall("s:sheets/s:sheet")]

    @property
    def sheets(self) -> SheetCollection:
        return SheetCollection(self)

    def _sheet_entries(self) -> list[tuple[str, str]]:
        """``(name, worksheet part name)`` in workbook order."""
        rels = self.package.rels_for(self._workbook_name)
        entries: list[tuple[str, str]] = []
        for sheet in self.workbook.findall("s:sheets/s:sheet"):
            rid = sheet.get(qn("r:id"))
            target = rels.target_of(rid) if (rels is not None and rid) else None
            if target is None:
                continue
            entries.append(
                (sheet.get("name") or "", rels.resolve(self._workbook_name, target))
            )
        return entries

    def _sheet(self, name: str, part_name: str) -> RawSheet:
        if part_name not in self._sheet_cache:
            self._sheet_cache[part_name] = RawSheet(self, name, part_name)
        return self._sheet_cache[part_name]

    # -- workbook-level reads ----------------------------------------------------

    @property
    def defined_names(self) -> dict[str, str]:
        """``definedName`` → refers-to text (read-only view)."""
        out: dict[str, str] = {}
        for dn in self.workbook.findall("s:definedNames/s:definedName"):
            name = dn.get("name")
            if name:
                out[name] = dn.text or ""
        return out

    # -- charts ---------------------------------------------------------------

    @property
    def chart_part_names(self) -> list[str]:
        """Chart part names reachable from the sheets' drawings, in
        sheet order (duplicates removed)."""
        from contextifier.raw.chart import find_chart_parts

        names: list[str] = []
        seen: set[str] = set()
        for _, ws_name in self._sheet_entries():
            if not self.package.has_part(ws_name):
                continue
            drawing = self.xml_part(ws_name).find("s:drawing")
            if drawing is None:
                continue
            rid = drawing.get(qn("r:id"))
            rels = self.package.rels_for(ws_name)
            target = rels.target_of(rid) if (rels is not None and rid) else None
            if target is None:
                continue
            drawing_part = rels.resolve(ws_name, target)
            for chart_name in find_chart_parts(self.package, drawing_part):
                if chart_name not in seen:
                    seen.add(chart_name)
                    names.append(chart_name)
        return names

    @property
    def charts(self) -> list["ChartModel"]:
        """ChartModel per chart part (raises NotImplementedError until
        the C3 chart milestone lands; use :attr:`chart_part_names` for
        discovery in the meantime)."""
        from contextifier.raw.chart import ChartModel

        models: list["ChartModel"] = []
        for name in self.chart_part_names:
            try:
                models.append(ChartModel(self.xml_part(name), self.package))
            except NotImplementedError:
                raise NotImplementedError(
                    "ChartModel is not implemented yet (milestone C3); "
                    "chart_part_names still lists the chart parts"
                ) from None
        return models

    # -- shared strings ------------------------------------------------------------

    def _shared_strings(self) -> list[str]:
        """The shared-string table (read-only, cached). A workbook with
        no sharedStrings part simply yields an empty table — writes use
        inline strings, so the part is never created."""
        if self._shared_cache is None:
            name = "xl/sharedStrings.xml"
            rels = self.package.rels_for(self._workbook_name)
            if rels is not None:
                for rel in rels.by_type("/sharedStrings"):
                    name = rels.resolve(self._workbook_name, rel["target"])
                    break
            if not self.package.has_part(name):
                self._shared_cache = []
            else:
                root = self.xml_part(name).root
                self._shared_cache = [
                    "".join(t.text or "" for t in si.iter(qn("s:t")))
                    for si in root.findall(qn("s:si"))
                ]
        return self._shared_cache

    # -- formula staleness ---------------------------------------------------------

    def _any_formula_exists(self) -> bool:
        """Whether any worksheet contains a formula (cached; used to
        decide if edits make cached results stale)."""
        if self._formulas_exist is None:
            self._formulas_exist = any(
                self._part_has_formula(part) for _, part in self._sheet_entries()
            )
        return self._formulas_exist

    def _part_has_formula(self, part_name: str) -> bool:
        if not self.package.has_part(part_name):
            return False
        xp = self._xml_parts.get(part_name)
        if xp is not None and xp.loaded:
            return next(xp.root.iter(qn("s:f")), None) is not None
        # Unparsed part: cheap byte scan (an <f> element in the default
        # spreadsheetml namespace; false positives merely add calcPr).
        data = self.package.get_part(part_name).read()
        return b"<f>" in data or b"<f " in data or b"<f/" in data

    def _ensure_full_calc_on_load(self) -> None:
        """Guarantee ``<calcPr fullCalcOnLoad="1"/>`` in workbook.xml so
        Excel recalculates stale formula caches on open. No-op (and no
        rewrite) when the flag is already set."""
        wb = self.workbook
        calc = wb.find("s:calcPr")
        if calc is None:
            calc = wb.root.makeelement(qn("s:calcPr"), {"fullCalcOnLoad": "1"})
            preceding = {qn(f"s:{tag}") for tag in _PRE_CALC_PR}
            pos = 0
            for i, child in enumerate(wb.root):
                if child.tag in preceding:
                    pos = i + 1
            wb.root.insert(pos, calc)
            wb.mark_dirty()
        elif calc.get("fullCalcOnLoad") != "1":
            calc.set("fullCalcOnLoad", "1")
            wb.mark_dirty()
