# tests/unit/chunking/test_nested_html_tables.py
"""
Nested HTML tables must be treated as one region.

Extracted documents nest tables routinely: a merged cell holding a sub-table
is how HWP, DOCX and PDF layouts express a form. ``<table[^>]*>.*?</table>``
is non-greedy, so it stops at the FIRST ``</table>`` — the inner one — and
reports a region that ends in the middle of the outer table. The chunker then
believes the rest of the outer table is ordinary prose and is free to split
it anywhere, producing chunks with unbalanced table markup.
"""

from __future__ import annotations

from contextifier.chunking.constants import find_html_tables

SIMPLE = "<table><tr><td>a</td></tr></table>"

NESTED = (
    "<table>"
    "<tr><td>outer-left</td>"
    "<td><table><tr><td>inner-1</td></tr><tr><td>inner-2</td></tr></table></td></tr>"
    "<tr><td>outer-tail</td></tr>"
    "</table>"
)


class TestFindHtmlTables:
    def test_simple_table(self) -> None:
        assert find_html_tables(SIMPLE) == [(0, len(SIMPLE))]

    def test_nested_table_is_one_region(self) -> None:
        regions = find_html_tables(NESTED)
        assert regions == [(0, len(NESTED))], (
            "the inner </table> must not terminate the outer region"
        )

    def test_triple_nesting(self) -> None:
        text = "<table><tr><td><table><tr><td><table><tr><td>x</td></tr></table></td></tr></table></td></tr></table>"
        assert find_html_tables(text) == [(0, len(text))]

    def test_sibling_tables_are_separate_regions(self) -> None:
        text = SIMPLE + "\n\n" + SIMPLE
        regions = find_html_tables(text)
        assert len(regions) == 2
        assert regions[0] == (0, len(SIMPLE))
        assert regions[1] == (len(SIMPLE) + 2, len(text))

    def test_attributes_on_the_open_tag(self) -> None:
        text = "<table border='1' class=\"x\"><tr><td>a</td></tr></table>"
        assert find_html_tables(text) == [(0, len(text))]

    def test_case_insensitive(self) -> None:
        text = "<TABLE><TR><TD>a</TD></TR></TABLE>"
        assert find_html_tables(text) == [(0, len(text))]

    def test_unclosed_table_is_ignored(self) -> None:
        """Malformed markup must not swallow the rest of the document."""
        text = "before <table><tr><td>a</td></tr> after"
        assert find_html_tables(text) == []

    def test_stray_close_tag_is_ignored(self) -> None:
        text = "</table>" + SIMPLE
        regions = find_html_tables(text)
        assert regions == [(len("</table>"), len(text))]

    def test_text_between_tables_is_not_included(self) -> None:
        text = f"lead {SIMPLE} middle {SIMPLE} tail"
        regions = find_html_tables(text)
        assert len(regions) == 2
        assert text[regions[0][0] : regions[0][1]] == SIMPLE
        assert text[regions[1][0] : regions[1][1]] == SIMPLE


class TestProtectedRegionIntegration:
    def test_nested_table_is_never_split(self) -> None:
        from contextifier.chunking.chunker import TextChunker
        from contextifier.config import ProcessingConfig

        # A nested table larger than the chunk budget: the protected strategy
        # must still emit it whole rather than cutting it mid-markup.
        inner_rows = "".join(f"<tr><td>inner cell {i}</td></tr>" for i in range(40))
        table = (
            "<table><tr><td>outer</td>"
            f"<td><table>{inner_rows}</table></td></tr>"
            "<tr><td>outer-tail-marker</td></tr></table>"
        )
        text = "Intro paragraph.\n\n" + table + "\n\nClosing paragraph."

        chunks = TextChunker(ProcessingConfig()).chunk(text, chunk_size=400)

        joined = "".join(chunks)
        assert joined.count("<table>") == joined.count("</table>")
        holders = [c for c in chunks if "outer-tail-marker" in c]
        assert holders, "outer table tail disappeared"
        assert all(c.count("<table>") == c.count("</table>") for c in chunks), (
            "a chunk ended with unbalanced table markup"
        )
