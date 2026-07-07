# contextifier/raw/chart.py
"""
ChartModel — read & write DrawingML charts, shared across all three
OOXML formats.

A chart is a ``chartN.xml`` part (classic ``c:`` namespace, or ``cx:``
chartEx for the 2016+ types) referenced from a sheet drawing (xlsx), a
document drawing (docx) or a slide graphicFrame (pptx), usually paired
with an embedded ``.xlsx`` workbook holding the source table. This
module is format-agnostic: it only needs the chart part and the
package, so the three format models can share one implementation.

Reading::

    chart = raw.charts[0]
    chart.kind          # "bar" | "line" | "pie" | ... | "chartex:<type>"
    chart.title         # str | None
    chart.series        # [ChartSeriesData(name, categories, values), ...]

Writing::

    chart.set_title("Q3 Sales")
    chart.set_data(
        categories=["Q1", "Q2", "Q3"],
        series=[("Sales", [120, 135, 150]), ("Cost", [80, 90, 95])],
    )
    raw.save("out.xlsx")

``set_data`` must rewrite BOTH the caches inside the chart XML
(``c:cat/c:strRef/c:strCache``, ``c:val/c:numRef/c:numCache`` per
series — creating/removing ``c:ser`` elements as the series count
changes) AND the embedded workbook part (if present) so that
double-click-edit in Office shows the same numbers. Formula references
(``c:f``) should be regenerated against the embedded workbook's sheet
("Sheet1!$B$2:$B$4" style). For xlsx-hosted charts whose series
reference the HOST workbook's own cells, ``set_data`` rewrites caches
and leaves the ``c:f`` references pointing at the host sheet (values
there are the caller's responsibility — typically edited through
``sheet.set_cell`` alongside).

Embedded workbook regeneration
------------------------------
When the chart has an embedded ``.../embeddings/*.xlsx`` workbook (the
usual case for pptx/docx-hosted charts), ``set_data`` REGENERATES that
part from scratch with openpyxl: a single sheet holding the plain data
table (header row ``[None, name1, name2, ...]`` followed by one
``[category, v1, v2, ...]`` row per category). The embedded workbook is
chart-internal source data, not user content, so replacing it wholesale
is safe — it is exactly what keeps Office's "Edit Data" view in sync
with the rewritten caches.

chartEx (cx:) limitations in this milestone
-------------------------------------------
chartEx charts are fully **readable** (``kind`` / ``title`` /
``series``), but ``set_title`` and ``set_data`` raise
:class:`~contextifier.raw.opc.RawUnsupportedError` for them — the cx:
write path is deferred (documented limitation, see the class contract).
"""

from __future__ import annotations

import copy
import io
import posixpath
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from lxml import etree

from contextifier.raw.opc import RawUnsupportedError
from contextifier.raw.xmlpart import NS, XmlPart, qn

if TYPE_CHECKING:  # pragma: no cover
    from lxml.etree import _Element

    from contextifier.raw.opc import OpcPackage, OpcPart

__all__ = ["ChartSeriesData", "ChartModel", "find_chart_parts", "load_chart"]


@dataclass
class ChartSeriesData:
    """One series as read from (or written to) the chart caches."""

    name: str | None
    categories: list[str] = field(default_factory=list)
    values: list[float | None] = field(default_factory=list)


# Classic plot-type element (local name) → kind. bar/bar3D are handled
# separately because they split on c:barDir ("bar" vs "column").
_PLOT_KIND: dict[str, str] = {
    "lineChart": "line",
    "line3DChart": "line",
    "pieChart": "pie",
    "pie3DChart": "pie",
    "areaChart": "area",
    "area3DChart": "area",
    "scatterChart": "scatter",
    "doughnutChart": "doughnut",
    "radarChart": "radar",
    "bubbleChart": "bubble",
    "ofPieChart": "of_pie",
    "surfaceChart": "surface",
    "surface3DChart": "surface",
    "stockChart": "stock",
}

# Plot types whose series carry c:xVal/c:yVal instead of c:cat/c:val.
_XY_PLOTS = {"scatterChart", "bubbleChart"}

# c:ser children that must stay AFTER c:cat/c:val (schema tail).
_SER_TAIL_TAGS = ("c:smooth", "c:shape", "c:bubbleSize", "c:bubble3D", "c:extLst")


def _local(el: "_Element") -> str:
    """Local (namespace-stripped) tag name; '' for comments/PIs."""
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _col_letter(n: int) -> str:
    """1-based column index → spreadsheet letters (1→A, 27→AA)."""
    letters = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _num_text(value: float) -> str:
    """Float/int → the shortest cache text Excel reads back exactly."""
    f = float(value)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def _unquote_sheet(ref_prefix: str) -> str:
    """``"'My Sheet'"`` → ``"My Sheet"`` (c:f quoting → real sheet name)."""
    if ref_prefix.startswith("'") and ref_prefix.endswith("'"):
        return ref_prefix[1:-1].replace("''", "'")
    return ref_prefix


class ChartModel:
    """Read/write view over one chart part.

    Contract (implemented in this module; consumed by the xlsx/docx/pptx
    models — do not change signatures without updating all three):

    * ``ChartModel(xml_part, package)`` — *xml_part* is the chart part
      facade; *package* is the owning :class:`OpcPackage` (used to reach
      the embedded workbook through the chart part's rels).
    * ``kind: str`` (property) — first plot type found: ``"bar"``,
      ``"line"``, ``"pie"``, ``"area"``, ``"scatter"``, ``"doughnut"``,
      ``"radar"``, ``"bubble"``, ... or ``"chartex:funnel"`` etc. for
      cx: charts.
    * ``title: str | None`` (property) — concatenated ``a:t`` runs of the
      chart title, if any.
    * ``series: list[ChartSeriesData]`` (property) — parsed from
      str/num caches (classic) or ``cx:strDim``/``cx:numDim`` (chartEx).
      Missing cache points yield ``None`` values.
    * ``set_title(text: str) -> None`` — replace/insert the title text,
      preserving existing run formatting where present.
    * ``set_data(categories, series) -> None`` — see module docstring.
      ``series`` accepts ``[(name, values), ...]`` or
      ``[ChartSeriesData, ...]``. Raises ``ValueError`` on ragged input
      (series length != len(categories)). chartEx write support may
      raise ``RawUnsupportedError`` (documented limitation) in v0.4.
    * ``embedded_workbook_part() -> OpcPart | None`` — the
      ``.../embeddings/*.xlsx`` part referenced by this chart's rels.

    Implementation notes (behavior, not signatures):

    * Parsing is lazy — ``__init__`` stores the facade; the tree is
      first parsed on the first property access.
    * Mutators (``set_title`` / ``set_data``) mark the XML facade dirty
      AND flush it immediately, so ``package.to_bytes()`` alone reflects
      the edit; format-model ``save()`` flushing stays a no-op for it.
    * chartEx: read-only in this milestone — ``set_title`` and
      ``set_data`` raise :class:`RawUnsupportedError` for cx: charts.
    * Charts without value caches (e.g. openpyxl-authored charts, which
      write only ``c:f`` references into the host workbook) read as
      series with ``name=None``, empty categories and ``values`` of
      ``None`` per cached ``ptCount`` (or empty when no cache exists).
    """

    def __init__(self, xml_part: "XmlPart", package: "OpcPackage"):
        self.xml = xml_part
        self.package = package

    # -- detection ---------------------------------------------------------------

    @property
    def _is_chartex(self) -> bool:
        return self.xml.root.tag == qn("cx:chartSpace")

    def _plot_elements(self) -> list["_Element"]:
        """Plot-type children of c:plotArea (barChart, lineChart, ...)."""
        plot_area = self.xml.root.find("c:chart/c:plotArea", NS)
        if plot_area is None:
            return []
        return [el for el in plot_area if _local(el).endswith("Chart")]

    # -- reading -----------------------------------------------------------------

    @property
    def kind(self) -> str:
        if self._is_chartex:
            ser = self.xml.root.find(
                "cx:chart/cx:plotArea/cx:plotAreaRegion/cx:series", NS
            )
            if ser is None:
                ser = next(self.xml.root.iter(qn("cx:series")), None)
            layout = ser.get("layoutId") if ser is not None else None
            return f"chartex:{layout or 'unknown'}"
        for plot in self._plot_elements():
            name = _local(plot)
            if name in ("barChart", "bar3DChart"):
                bar_dir = plot.find("c:barDir", NS)
                val = bar_dir.get("val", "col") if bar_dir is not None else "col"
                return "bar" if val == "bar" else "column"
            return _PLOT_KIND.get(name, name)
        return "unknown"

    @property
    def title(self) -> str | None:
        root = self.xml.root
        if self._is_chartex:
            title_el = root.find("cx:chart/cx:title", NS)
        else:
            chart = root.find("c:chart", NS)
            if chart is None:
                return None
            atd = chart.find("c:autoTitleDeleted", NS)
            if atd is not None and atd.get("val", "1") in ("1", "true"):
                return None
            title_el = chart.find("c:title", NS)
        if title_el is None:
            return None
        text = "".join(t.text for t in title_el.iter(qn("a:t")) if t.text)
        return text or None

    @property
    def series(self) -> list[ChartSeriesData]:
        if self._is_chartex:
            return self._series_chartex()
        out: list[ChartSeriesData] = []
        for plot in self._plot_elements():
            uses_xy = _local(plot) in _XY_PLOTS
            for ser in plot.findall("c:ser", NS):
                out.append(self._read_ser(ser, uses_xy))
        return out

    def _read_ser(self, ser: "_Element", uses_xy: bool) -> ChartSeriesData:
        name = None
        name_el = ser.find("c:tx/c:strRef/c:strCache/c:pt/c:v", NS)
        if name_el is None:
            name_el = ser.find("c:tx/c:v", NS)
        if name_el is not None and name_el.text:
            name = name_el.text
        cat_holder = ser.find("c:xVal" if uses_xy else "c:cat", NS)
        val_holder = ser.find("c:yVal" if uses_xy else "c:val", NS)
        return ChartSeriesData(
            name=name,
            categories=self._cache_texts(cat_holder),
            values=self._cache_values(val_holder),
        )

    @staticmethod
    def _find_cache(holder: "_Element | None") -> "_Element | None":
        """The cache element inside a c:cat/c:val/c:xVal/c:yVal holder."""
        if holder is None:
            return None
        for path in (
            "c:strRef/c:strCache",
            "c:numRef/c:numCache",
            "c:strLit",
            "c:numLit",
            # multi-level categories: first c:lvl = leaf labels
            "c:multiLvlStrRef/c:multiLvlStrCache/c:lvl",
        ):
            cache = holder.find(path, NS)
            if cache is not None:
                return cache
        return None

    @classmethod
    def _cache_texts(cls, holder: "_Element | None") -> list[str]:
        """Cache points as strings in idx order; gaps are skipped."""
        cache = cls._find_cache(holder)
        if cache is None:
            return []
        pts: list[tuple[int, str]] = []
        for pt in cache.findall("c:pt", NS):
            v = pt.find("c:v", NS)
            if v is None:
                continue
            try:
                idx = int(pt.get("idx", str(len(pts))))
            except ValueError:
                idx = len(pts)
            pts.append((idx, v.text or ""))
        return [text for _, text in sorted(pts, key=lambda p: p[0])]

    @classmethod
    def _cache_values(cls, holder: "_Element | None") -> list[float | None]:
        """Numeric cache; length = ptCount, gaps/unparseable → None."""
        cache = cls._find_cache(holder)
        if cache is None:
            return []
        by_idx: dict[int, float | None] = {}
        for pt in cache.findall("c:pt", NS):
            v = pt.find("c:v", NS)
            try:
                idx = int(pt.get("idx", "0"))
            except ValueError:
                continue
            if v is None or v.text is None:
                by_idx[idx] = None
                continue
            try:
                by_idx[idx] = float(v.text)
            except ValueError:
                by_idx[idx] = None
        pt_count_el = cache.find("c:ptCount", NS)
        try:
            count = int(pt_count_el.get("val", "")) if pt_count_el is not None else -1
        except ValueError:
            count = -1
        if count < 0:
            count = (max(by_idx) + 1) if by_idx else 0
        return [by_idx.get(i) for i in range(count)]

    def _series_chartex(self) -> list[ChartSeriesData]:
        root = self.xml.root
        data_by_id = {
            data.get("id"): data for data in root.findall("cx:chartData/cx:data", NS)
        }
        series_els = root.findall(
            "cx:chart/cx:plotArea/cx:plotAreaRegion/cx:series", NS
        ) or list(root.iter(qn("cx:series")))
        out: list[ChartSeriesData] = []
        if not series_els:
            # No series elements: expose each data block as an unnamed series.
            for data in data_by_id.values():
                cats, vals = self._chartex_dims(data)
                out.append(ChartSeriesData(name=None, categories=cats, values=vals))
            return out
        for ser in series_els:
            name_el = ser.find("cx:tx/cx:txData/cx:v", NS)
            name = name_el.text if name_el is not None and name_el.text else None
            data_id_el = ser.find("cx:dataId", NS)
            data = (
                data_by_id.get(data_id_el.get("val"))
                if data_id_el is not None
                else None
            )
            cats: list[str] = []
            vals: list[float | None] = []
            if data is not None:
                cats, vals = self._chartex_dims(data)
            out.append(ChartSeriesData(name=name, categories=cats, values=vals))
        return out

    @staticmethod
    def _chartex_dims(data: "_Element") -> tuple[list[str], list[float | None]]:
        """(categories, values) from a cx:data block's dims."""

        def pick(dims: list["_Element"], wanted_type: str) -> "_Element | None":
            for dim in dims:
                if dim.get("type") == wanted_type:
                    return dim
            return dims[0] if dims else None

        cats: list[str] = []
        vals: list[float | None] = []
        str_dim = pick(data.findall("cx:strDim", NS), "cat")
        if str_dim is not None:
            lvl = str_dim.find("cx:lvl", NS)
            if lvl is not None:
                pts = []
                for pt in lvl.findall("cx:pt", NS):
                    try:
                        idx = int(pt.get("idx", str(len(pts))))
                    except ValueError:
                        idx = len(pts)
                    pts.append((idx, pt.text or ""))
                cats = [text for _, text in sorted(pts, key=lambda p: p[0])]
        num_dim = pick(data.findall("cx:numDim", NS), "val")
        if num_dim is not None:
            lvl = num_dim.find("cx:lvl", NS)
            if lvl is not None:
                by_idx: dict[int, float | None] = {}
                for pt in lvl.findall("cx:pt", NS):
                    try:
                        idx = int(pt.get("idx", "0"))
                    except ValueError:
                        continue
                    try:
                        by_idx[idx] = float(pt.text) if pt.text else None
                    except ValueError:
                        by_idx[idx] = None
                try:
                    count = int(lvl.get("ptCount", ""))
                except ValueError:
                    count = (max(by_idx) + 1) if by_idx else 0
                vals = [by_idx.get(i) for i in range(count)]
        return cats, vals

    # -- writing: title ----------------------------------------------------------

    def set_title(self, text: str) -> None:
        """Replace (or insert) the chart title text.

        Keeps the first existing run's ``a:rPr`` formatting; extra runs
        and extra paragraphs of the old title are removed so the title
        reads exactly *text*. Removes ``c:autoTitleDeleted val="1"`` so
        the explicit title is shown. chartEx charts are read-only in
        this milestone → :class:`RawUnsupportedError`.
        """
        if self._is_chartex:
            raise RawUnsupportedError(
                "set_title is not supported for chartEx (cx:) charts yet; "
                "classic c: charts only"
            )
        chart = self.xml.root.find("c:chart", NS)
        if chart is None:
            raise RawUnsupportedError("chart part has no c:chart element")

        atd = chart.find("c:autoTitleDeleted", NS)
        if atd is not None and atd.get("val", "1") in ("1", "true"):
            chart.remove(atd)

        title_el = chart.find("c:title", NS)
        if title_el is None:
            title_el = etree.Element(qn("c:title"))
            chart.insert(0, title_el)  # c:title is the first child per schema
        tx = title_el.find("c:tx", NS)
        if tx is None:
            tx = etree.Element(qn("c:tx"))
            title_el.insert(0, tx)  # c:tx is the first child of c:title
        rich = tx.find("c:rich", NS)
        if rich is None:
            # Replace whatever c:tx held (e.g. a c:strRef) with rich text.
            for child in list(tx):
                tx.remove(child)
            rich = etree.SubElement(tx, qn("c:rich"))
            etree.SubElement(rich, qn("a:bodyPr"))
            etree.SubElement(rich, qn("a:lstStyle"))

        paragraphs = rich.findall("a:p", NS)
        if paragraphs:
            para = paragraphs[0]
            for extra in paragraphs[1:]:
                rich.remove(extra)
        else:
            para = etree.SubElement(rich, qn("a:p"))

        run_props = None
        first_run = para.find("a:r", NS)
        if first_run is not None:
            run_props = first_run.find("a:rPr", NS)
        for el in list(para):
            if el.tag in (qn("a:r"), qn("a:fld"), qn("a:br")):
                para.remove(el)
        run = etree.Element(qn("a:r"))
        if run_props is not None:
            run.append(copy.deepcopy(run_props))
        t = etree.SubElement(run, qn("a:t"))
        t.text = text
        end_props = para.find("a:endParaRPr", NS)
        if end_props is not None:
            end_props.addprevious(run)
        else:
            para.append(run)
        self._commit()

    # -- writing: data -----------------------------------------------------------

    def set_data(
        self,
        categories: Sequence[object],
        series: Sequence["tuple[str | None, Sequence[float | None]] | ChartSeriesData"],
    ) -> None:
        """Rewrite the chart's cached data (and its embedded workbook).

        *series* accepts ``[(name, values), ...]`` or
        ``[ChartSeriesData, ...]`` (whose ``.categories`` are ignored —
        the *categories* argument is canonical for every series).

        Raises ``ValueError`` on ragged input and
        :class:`RawUnsupportedError` for chartEx (cx:) charts.
        """
        if self._is_chartex:
            raise RawUnsupportedError(
                "set_data is not supported for chartEx (cx:) charts in this "
                "version; classic c: charts only"
            )
        cats = list(categories)
        normalized: list[tuple[str | None, list[float | None]]] = []
        for item in series:
            if isinstance(item, ChartSeriesData):
                normalized.append((item.name, list(item.values)))
            else:
                name, values = item
                normalized.append((name, list(values)))
        if not normalized:
            raise ValueError("set_data requires at least one series")
        for name, values in normalized:
            if len(values) != len(cats):
                raise ValueError(
                    f"series {name!r} has {len(values)} values for "
                    f"{len(cats)} categories (lengths must match)"
                )

        sers: list[tuple["_Element", "_Element"]] = [
            (plot, ser)
            for plot in self._plot_elements()
            for ser in plot.findall("c:ser", NS)
        ]
        if not sers:
            raise RawUnsupportedError(
                "chart has no existing c:ser element to rewrite or clone from"
            )

        sheet_ref = self._sheet_ref()

        # Shrink: drop trailing series.
        while len(sers) > len(normalized):
            plot, ser = sers.pop()
            plot.remove(ser)
        # Grow: clone the LAST series (deep copy keeps its styling).
        if len(sers) < len(normalized):
            used_indices = []
            for _, ser in sers:
                for tag in ("c:idx", "c:order"):
                    el = ser.find(tag, NS)
                    if el is not None:
                        try:
                            used_indices.append(int(el.get("val", "")))
                        except ValueError:
                            pass
            next_index = (max(used_indices) + 1) if used_indices else len(sers)
            plot, last = sers[-1]
            while len(sers) < len(normalized):
                clone = copy.deepcopy(last)
                self._set_ser_index(clone, next_index)
                last.addnext(clone)
                sers.append((plot, clone))
                last = clone
                next_index += 1

        n = len(cats)
        for i, ((plot, ser), (name, values)) in enumerate(zip(sers, normalized)):
            col = _col_letter(i + 2)  # data columns start at B
            self._rewrite_ser(
                ser,
                name,
                cats,
                values,
                uses_xy=_local(plot) in _XY_PLOTS,
                cat_ref=f"{sheet_ref}!$A$2:$A${n + 1}",
                val_ref=f"{sheet_ref}!${col}$2:${col}${n + 1}",
                name_ref=f"{sheet_ref}!${col}$1",
            )

        workbook_part = self.embedded_workbook_part()
        if workbook_part is not None:
            workbook_part.write(
                self._workbook_bytes(_unquote_sheet(sheet_ref), cats, normalized)
            )
        self._commit()

    def _sheet_ref(self) -> str:
        """Sheet prefix (verbatim, quotes kept) of the first c:f found."""
        for f in self.xml.root.iter(qn("c:f")):
            text = f.text or ""
            if "!" in text:
                return text.rsplit("!", 1)[0]
        return "Sheet1"

    @staticmethod
    def _set_ser_index(ser: "_Element", index: int) -> None:
        for tag in ("c:idx", "c:order"):
            el = ser.find(tag, NS)
            if el is None:
                el = etree.Element(qn(tag))
                ser.insert(0 if tag == "c:idx" else 1, el)
            el.set("val", str(index))

    def _rewrite_ser(
        self,
        ser: "_Element",
        name: str | None,
        cats: list[object],
        values: list[float | None],
        *,
        uses_xy: bool,
        cat_ref: str,
        val_ref: str,
        name_ref: str,
    ) -> None:
        if name is not None:
            self._rewrite_ser_name(ser, name, name_ref)
        cat_tag = "c:xVal" if uses_xy else "c:cat"
        val_tag = "c:yVal" if uses_xy else "c:val"
        numeric_cats = all(
            isinstance(c, (int, float)) and not isinstance(c, bool) for c in cats
        )
        if uses_xy and numeric_cats:
            new_cat = self._build_num_holder(cat_tag, cat_ref, cats)  # type: ignore[arg-type]
        else:
            new_cat = self._build_str_holder(cat_tag, cat_ref, cats)
        new_val = self._build_num_holder(val_tag, val_ref, values)
        self._replace_ser_child(ser, cat_tag, new_cat, also_before=(val_tag,))
        self._replace_ser_child(ser, val_tag, new_val)

    def _rewrite_ser_name(self, ser: "_Element", name: str, name_ref: str) -> None:
        tx = ser.find("c:tx", NS)
        if tx is None:
            tx = etree.Element(qn("c:tx"))
            anchor = ser.find("c:order", NS)
            if anchor is None:
                anchor = ser.find("c:idx", NS)
            if anchor is not None:
                anchor.addnext(tx)
            else:
                ser.insert(0, tx)
        literal = tx.find("c:v", NS)
        if literal is not None and tx.find("c:strRef", NS) is None:
            literal.text = name
            return
        for child in list(tx):
            tx.remove(child)
        str_ref = etree.SubElement(tx, qn("c:strRef"))
        f = etree.SubElement(str_ref, qn("c:f"))
        f.text = name_ref
        cache = etree.SubElement(str_ref, qn("c:strCache"))
        pt_count = etree.SubElement(cache, qn("c:ptCount"))
        pt_count.set("val", "1")
        pt = etree.SubElement(cache, qn("c:pt"))
        pt.set("idx", "0")
        v = etree.SubElement(pt, qn("c:v"))
        v.text = name

    @staticmethod
    def _build_str_holder(tag: str, ref: str, items: list[object]) -> "_Element":
        holder = etree.Element(qn(tag))
        str_ref = etree.SubElement(holder, qn("c:strRef"))
        f = etree.SubElement(str_ref, qn("c:f"))
        f.text = ref
        cache = etree.SubElement(str_ref, qn("c:strCache"))
        pt_count = etree.SubElement(cache, qn("c:ptCount"))
        pt_count.set("val", str(len(items)))
        for idx, item in enumerate(items):
            if item is None:
                continue  # gap
            pt = etree.SubElement(cache, qn("c:pt"))
            pt.set("idx", str(idx))
            v = etree.SubElement(pt, qn("c:v"))
            v.text = str(item)
        return holder

    @staticmethod
    def _build_num_holder(tag: str, ref: str, values: list[float | None]) -> "_Element":
        holder = etree.Element(qn(tag))
        num_ref = etree.SubElement(holder, qn("c:numRef"))
        f = etree.SubElement(num_ref, qn("c:f"))
        f.text = ref
        cache = etree.SubElement(num_ref, qn("c:numCache"))
        format_code = etree.SubElement(cache, qn("c:formatCode"))
        format_code.text = "General"
        pt_count = etree.SubElement(cache, qn("c:ptCount"))
        pt_count.set("val", str(len(values)))
        for idx, value in enumerate(values):
            if value is None:
                continue  # gap
            pt = etree.SubElement(cache, qn("c:pt"))
            pt.set("idx", str(idx))
            v = etree.SubElement(pt, qn("c:v"))
            v.text = _num_text(value)
        return holder

    @staticmethod
    def _replace_ser_child(
        ser: "_Element",
        tag: str,
        new_el: "_Element",
        also_before: tuple[str, ...] = (),
    ) -> None:
        """Swap the *tag* child in place, or insert it at a schema-valid
        position (before the c:ser tail elements)."""
        existing = ser.find(tag, NS)
        if existing is not None:
            existing.addprevious(new_el)
            ser.remove(existing)
            return
        stop_tags = {qn(t) for t in (*_SER_TAIL_TAGS, *also_before)}
        for child in ser:
            if child.tag in stop_tags:
                child.addprevious(new_el)
                return
        ser.append(new_el)

    # -- embedded workbook ---------------------------------------------------------

    def embedded_workbook_part(self) -> "OpcPart | None":
        """The ``.../embeddings/*.xlsx`` part behind this chart, if any."""
        rels = self.package.rels_for(self.xml.name)
        if rels is None:
            return None
        for rel in rels:
            if rel["mode"] == "External":
                continue
            rel_type = rel["type"] or ""
            if not (rel_type.endswith("/package") or rel_type.endswith("/oleObject")):
                continue
            target = rel["target"] or ""
            if posixpath.splitext(target)[1].lower() != ".xlsx":
                continue
            name = rels.resolve(self.xml.name, target)
            if self.package.has_part(name):
                return self.package.get_part(name)
        return None

    @staticmethod
    def _workbook_bytes(
        sheet_name: str,
        cats: list[object],
        normalized: list[tuple[str | None, list[float | None]]],
    ) -> bytes:
        """A fresh data-only workbook mirroring the rewritten caches."""
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        ws.append([None, *(name for name, _ in normalized)])
        for i, cat in enumerate(cats):
            ws.append([cat, *(values[i] for _, values in normalized)])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    # -- persistence -----------------------------------------------------------

    def _commit(self) -> None:
        """Mark the facade dirty and serialize into the package part."""
        self.xml.mark_dirty()
        self.xml.flush()


def load_chart(package: "OpcPackage", chart_part_name: str) -> ChartModel:
    """Convenience: chart part name → :class:`ChartModel` facade."""
    return ChartModel(XmlPart(package.get_part(chart_part_name)), package)


def find_chart_parts(package: "OpcPackage", from_part: str) -> list[str]:
    """Chart part names referenced (directly) by *from_part*'s rels.

    Format models use this from a drawing/slide part:
    ``rels.by_type("/chart")`` → resolve targets → chart part names.
    """
    rels = package.rels_for(from_part)
    if rels is None:
        return []
    return [rels.resolve(from_part, rel["target"]) for rel in rels.by_type("/chart")]
