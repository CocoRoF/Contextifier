# tests/unit/chunking/test_table_rowspan_carry.py
"""
Splitting a table across chunks must not lose the merged cells that span the cut.

When a table is split at a row boundary, a cell with ``rowspan`` that started
in an earlier chunk is simply absent from the continuation chunk's markup —
its content is gone and every row below is one cell short. For the usual
shape (a left-hand category column spanning its group of rows) that means the
category name, the single most useful piece of context in the table, survives
only in the first chunk.

Clamping the rowspan value, which is all the splitter used to do, keeps the
markup well formed but does nothing about the missing cell.
"""

from __future__ import annotations

import re

from contextifier.chunking.table_chunker import chunk_html_table
from contextifier.chunking.table_parser import compute_carried_cells, parse_html_table


def _rows(html: str) -> list:
    return re.findall(r"<tr[^>]*>.*?</tr>", html, re.DOTALL | re.IGNORECASE)


def _cells(row_html: str) -> list:
    return re.findall(r"<(?:td|th)[^>]*>.*?</(?:td|th)>", row_html, re.DOTALL | re.IGNORECASE)


class TestComputeCarriedCells:
    def test_left_column_span(self) -> None:
        table = (
            "<table>"
            '<tr><td rowspan="3">CAT</td><td>r1</td></tr>'
            "<tr><td>r2</td></tr>"
            "<tr><td>r3</td></tr>"
            "</table>"
        )
        carried = compute_carried_cells(parse_html_table(table).data_rows)

        assert carried[0] == {}
        assert set(carried[1]) == {0}
        assert carried[1][0][0].content == "CAT"
        assert carried[1][0][1] == 2, "two rows of the span remain, including this one"
        assert carried[2][0][1] == 1

    def test_span_in_a_middle_column(self) -> None:
        """A row under a span omits the covered cell, shifting the ones written."""
        table = (
            "<table>"
            '<tr><td>a1</td><td rowspan="2">MID</td><td>c1</td></tr>'
            "<tr><td>a2</td><td>c2</td></tr>"
            "</table>"
        )
        carried = compute_carried_cells(parse_html_table(table).data_rows)

        assert set(carried[1]) == {1}, "the carried cell sits at visual column 1"
        assert carried[1][1][0].content == "MID"

    def test_colspan_shifts_later_columns(self) -> None:
        table = (
            "<table>"
            '<tr><td colspan="2">wide</td><td rowspan="2">RIGHT</td></tr>'
            "<tr><td>x</td><td>y</td></tr>"
            "</table>"
        )
        carried = compute_carried_cells(parse_html_table(table).data_rows)

        assert set(carried[1]) == {2}, "RIGHT starts after a colspan=2 cell"

    def test_span_registered_under_another_span(self) -> None:
        table = (
            "<table>"
            '<tr><td rowspan="3">A</td><td>b1</td></tr>'
            '<tr><td rowspan="2">B</td></tr>'
            "<tr><td>c</td></tr>"
            "</table>"
        )
        carried = compute_carried_cells(parse_html_table(table).data_rows)

        # Row 2 is covered by A only; row 3 by both A and B.
        assert set(carried[1]) == {0}
        assert set(carried[2]) == {0, 1}
        assert carried[2][1][0].content == "B"


class TestCarryDuringChunking:
    TABLE = (
        "<table>"
        "<tr><th>Category</th><th>Item</th><th>Note</th></tr>"
        + "".join(
            f'<tr>{"<td rowspan=\'4\'>GROUP-" + str(g) + "</td>" if i == 0 else ""}'
            f"<td>item {g}-{i} with enough text to take space</td>"
            f"<td>note {g}-{i} padding padding padding</td></tr>"
            for g in range(3)
            for i in range(4)
        )
        + "</table>"
    )

    def test_split_actually_happens(self) -> None:
        chunks = chunk_html_table(self.TABLE, chunk_size=400)
        assert len(chunks) > 1

    def test_every_chunk_keeps_its_group_label(self) -> None:
        chunks = chunk_html_table(self.TABLE, chunk_size=400)

        for chunk in chunks:
            body = chunk[chunk.index("<table>") :]
            data_rows = [r for r in _rows(body) if "<th>" not in r]
            if not data_rows:
                continue
            assert re.search(r"GROUP-\d", data_rows[0]), (
                "the continuation chunk lost the merged category cell:\n" + chunk
            )

    def test_rows_are_rectangular_within_a_chunk(self) -> None:
        chunks = chunk_html_table(self.TABLE, chunk_size=400)

        for chunk in chunks:
            body = chunk[chunk.index("<table>") :]
            data_rows = [r for r in _rows(body) if "<th>" not in r]
            if len(data_rows) < 2:
                continue
            first = len(_cells(data_rows[0]))
            for row in data_rows[1:]:
                span_covered = first - len(_cells(row))
                assert span_covered in (0, 1), (
                    f"row width jumped from {first} to {len(_cells(row))}"
                )

    def test_reissued_rowspan_is_clamped_to_the_chunk(self) -> None:
        chunks = chunk_html_table(self.TABLE, chunk_size=400)

        for chunk in chunks:
            body = chunk[chunk.index("<table>") :]
            rows_in_chunk = len([r for r in _rows(body) if "<th>" not in r])
            for value in re.findall(r'rowspan=["\']?(\d+)', body):
                assert int(value) <= rows_in_chunk, (
                    f"rowspan={value} exceeds the {rows_in_chunk} rows in the chunk"
                )

    def test_no_content_is_lost(self) -> None:
        chunks = chunk_html_table(self.TABLE, chunk_size=400)
        joined = "".join(chunks)
        for g in range(3):
            for i in range(4):
                assert f"item {g}-{i}" in joined

    def test_table_without_spans_is_untouched(self) -> None:
        table = "<table>" + "".join(
            f"<tr><td>plain row {i} with padding text</td></tr>" for i in range(30)
        ) + "</table>"
        chunks = chunk_html_table(table, chunk_size=300)
        assert len(chunks) > 1
        assert not any("rowspan" in c for c in chunks)
