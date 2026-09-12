# contextifier/handlers/html/_tables.py
"""
Repair for tables that a producer split into a header table and a body table.

Report exporters — the same ones that serve HTML under a ``.xls`` name — often
emit the column headings as one ``<table>`` and the data as a second
``<table>`` immediately after it, so that the heading row can be frozen while
the body scrolls. Read literally that is two tables, and the data table has no
column names at all: chunk it and every value loses its label.

Merging them is only safe when the split is unambiguous, so all of these must
hold: the first table contains header cells and nothing else, the second
contains data cells and nothing else, they are adjacent siblings, and they
agree on how many columns they have.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _column_count(table: Any) -> int:
    """Widest row in the table, counting ``colspan``."""
    widths: List[int] = []
    for tr in table.find_all("tr"):
        width = 0
        for cell in tr.find_all(["td", "th"]):
            try:
                width += max(1, int(cell.get("colspan", 1) or 1))
            except (TypeError, ValueError):
                width += 1
        if width:
            widths.append(width)
    return max(widths) if widths else 0


def _cell_kinds(table: Any) -> set:
    """The set of cell tag names used anywhere in the table."""
    return {cell.name for cell in table.find_all(["td", "th"])}


def _next_table_sibling(table: Any) -> Optional[Any]:
    """The next sibling element, when it is a table with nothing in between."""
    from bs4 import NavigableString, Tag

    for sibling in table.next_siblings:
        if isinstance(sibling, NavigableString):
            if sibling.strip():
                return None  # real text between the two tables
            continue
        if isinstance(sibling, Tag):
            return sibling if sibling.name == "table" else None
    return None


def merge_split_tables(soup: Any) -> int:
    """
    Merge each header-only table into the data table that follows it.

    Mutates *soup* in place.

    Args:
        soup: Parsed document (BeautifulSoup object or subtree).

    Returns:
        How many pairs were merged.
    """
    merged = 0

    for table in list(soup.find_all("table")):
        if table.decomposed:
            continue
        if _cell_kinds(table) != {"th"}:
            continue

        partner = _next_table_sibling(table)
        if partner is None or partner.decomposed:
            continue
        if _cell_kinds(partner) != {"td"}:
            continue

        header_cols = _column_count(table)
        body_cols = _column_count(partner)
        if header_cols == 0 or header_cols != body_cols:
            continue

        rows = partner.find_all("tr")
        if not rows:
            continue

        # Keep the header table's own structure and append the body rows to
        # whichever container holds them, so <thead>/<tbody> stays valid.
        body_container = table.find("tbody") or table
        for row in rows:
            body_container.append(row.extract())

        partner.decompose()
        merged += 1
        logger.debug("Merged a header-only table with the data table below it")

    return merged


__all__ = ["merge_split_tables"]
