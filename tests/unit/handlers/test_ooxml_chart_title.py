# tests/unit/handlers/test_ooxml_chart_title.py
"""
Chart titles are stored as a sequence of text runs.

A run boundary is a formatting change, so a title with one bold word is split
mid-phrase. Reading only the first run truncates the title; joining the runs
with spaces inserts them where none belong.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from contextifier.handlers._ooxml import NS_A, NS_C, extract_chart_title


def _chart(inner: str) -> ET.Element:
    return ET.fromstring(
        f'<c:chart xmlns:c="{NS_C}" xmlns:a="{NS_A}">{inner}</c:chart>'.encode()
    )


def _rich(*runs: str, paragraphs: bool = True) -> str:
    body = "".join(f"<a:r><a:t>{r}</a:t></a:r>" for r in runs)
    inner = f"<a:p>{body}</a:p>" if paragraphs else body
    return f"<c:title><c:tx><c:rich>{inner}</c:rich></c:tx></c:title>"


class TestRichTitles:
    def test_single_run(self) -> None:
        assert extract_chart_title(_chart(_rich("Quarterly Sales"))) == "Quarterly Sales"

    def test_runs_are_joined_without_inserted_spaces(self) -> None:
        title = extract_chart_title(_chart(_rich("2026", "년 ", "매출 실적")))
        assert title == "2026년 매출 실적"

    def test_a_wrapped_title_reads_as_one_line(self) -> None:
        inner = (
            "<c:title><c:tx><c:rich>"
            "<a:p><a:r><a:t>Revenue by</a:t></a:r></a:p>"
            "<a:p><a:r><a:t>Region</a:t></a:r></a:p>"
            "</c:rich></c:tx></c:title>"
        )
        assert extract_chart_title(_chart(inner)) == "Revenue by Region"

    def test_runs_without_a_paragraph_wrapper(self) -> None:
        title = extract_chart_title(_chart(_rich("Part", "One", paragraphs=False)))
        assert title == "PartOne"


class TestStringReference:
    def test_title_pointing_at_a_cell(self) -> None:
        inner = (
            "<c:title><c:tx><c:strRef>"
            "<c:strCache><c:pt idx='0'><c:v>From A1</c:v></c:pt></c:strCache>"
            "</c:strRef></c:tx></c:title>"
        )
        assert extract_chart_title(_chart(inner)) == "From A1"


class TestAbsentTitle:
    @pytest.mark.parametrize(
        "inner",
        ["", "<c:title/>", "<c:title><c:tx><c:rich><a:p/></c:rich></c:tx></c:title>"],
    )
    def test_no_title(self, inner: str) -> None:
        assert extract_chart_title(_chart(inner)) is None
