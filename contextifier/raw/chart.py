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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from contextifier.raw.opc import OpcPackage
    from contextifier.raw.xmlpart import XmlPart

__all__ = ["ChartSeriesData", "ChartModel", "find_chart_parts"]


@dataclass
class ChartSeriesData:
    """One series as read from (or written to) the chart caches."""

    name: str | None
    categories: list[str] = field(default_factory=list)
    values: list[float | None] = field(default_factory=list)


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
    """

    def __init__(self, xml_part: "XmlPart", package: "OpcPackage"):
        self.xml = xml_part
        self.package = package
        raise NotImplementedError("ChartModel is implemented in milestone C3")

    # NOTE: property/method signatures above are the frozen contract.
    # The C3 implementation replaces this stub in-place.


def find_chart_parts(package: "OpcPackage", from_part: str) -> list[str]:
    """Chart part names referenced (directly) by *from_part*'s rels.

    Format models use this from a drawing/slide part:
    ``rels.by_type("/chart")`` → resolve targets → chart part names.
    """
    rels = package.rels_for(from_part)
    if rels is None:
        return []
    return [rels.resolve(from_part, rel["target"]) for rel in rels.by_type("/chart")]
