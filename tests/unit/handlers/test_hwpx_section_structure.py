# tests/unit/handlers/test_hwpx_section_structure.py
"""
Structural traversal of an HWPX section.

A section was walked with ``root.findall(".//hp:p")`` — every paragraph
*anywhere* in the subtree. Table cells, shape text boxes and header/footer
parts all store their content as ``hp:p`` too, so each one was emitted a
second time as a loose body paragraph: table content appeared twice (once
inside the rendered HTML, once after it), and shape and header text landed at
the end of the section instead of where they belong.

These tests pin the containment rules: a paragraph belongs to exactly one
renderer, and every renderer is reached.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from contextifier.handlers.hwpx._section import HwpxSupplementary, parse_hwpx_section

NS = (
    'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph" '
    'xmlns:hc="http://www.hancom.co.kr/hwpml/2011/core" '
    'xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"'
)


def _p(text: str) -> str:
    return f"<hp:p><hp:run><hp:t>{text}</hp:t></hp:run></hp:p>"


def _tc(row: int, col: int, text: str) -> str:
    return (
        f'<hp:tc><hp:cellAddr colAddr="{col}" rowAddr="{row}"/>'
        f'<hp:cellSpan colSpan="1" rowSpan="1"/>'
        f"<hp:subList>{_p(text)}</hp:subList></hp:tc>"
    )


_TABLE = (
    '<hp:tbl rowCnt="2" colCnt="2">'
    f'<hp:tr>{_tc(0, 0, "CELL-A1")}{_tc(0, 1, "CELL-B1")}</hp:tr>'
    f'<hp:tr>{_tc(1, 0, "CELL-A2")}{_tc(1, 1, "CELL-B2")}</hp:tr>'
    "</hp:tbl>"
)


def _section(body: str) -> bytes:
    return f"<hs:sec {NS}>{body}</hs:sec>".encode()


@pytest.fixture()
def empty_zip() -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("placeholder", b"")
    buf.seek(0)
    return zipfile.ZipFile(buf)


def _parse(xml: bytes, zf: zipfile.ZipFile, supplementary=None) -> str:
    return parse_hwpx_section(xml, zf, {}, supplementary=supplementary)


class TestTableContainment:
    def test_cell_text_appears_once(self, empty_zip) -> None:
        out = _parse(_section(_p("BODY") + f"<hp:p><hp:run>{_TABLE}</hp:run></hp:p>"), empty_zip)

        for cell in ("CELL-A1", "CELL-B1", "CELL-A2", "CELL-B2"):
            assert out.count(cell) == 1, f"{cell} emitted {out.count(cell)}×:\n{out}"

    def test_table_is_still_rendered_as_html(self, empty_zip) -> None:
        out = _parse(_section(f"<hp:p><hp:run>{_TABLE}</hp:run></hp:p>"), empty_zip)

        assert "<table>" in out
        assert "<td>CELL-A1</td>" in out
        assert "<td>CELL-B2</td>" in out

    def test_body_text_keeps_its_position(self, empty_zip) -> None:
        out = _parse(
            _section(_p("BEFORE") + f"<hp:p><hp:run>{_TABLE}</hp:run></hp:p>" + _p("AFTER")),
            empty_zip,
        )
        assert out.index("BEFORE") < out.index("<table>") < out.index("AFTER")


class TestShapeText:
    @pytest.mark.parametrize(
        "shape", ["rect", "ellipse", "polygon", "arc", "curve", "container", "line"]
    )
    def test_draw_text_is_extracted(self, empty_zip, shape: str) -> None:
        body = (
            f"<hp:p><hp:run><hp:{shape}>"
            f"<hp:drawText><hp:subList>{_p('SHAPE TEXT')}</hp:subList></hp:drawText>"
            f"</hp:{shape}></hp:run></hp:p>"
        )
        out = _parse(_section(body), empty_zip)
        assert out.count("SHAPE TEXT") == 1

    def test_shape_text_keeps_document_position(self, empty_zip) -> None:
        body = (
            _p("BEFORE")
            + "<hp:p><hp:run><hp:rect><hp:drawText><hp:subList>"
            + _p("SHAPE TEXT")
            + "</hp:subList></hp:drawText></hp:rect></hp:run></hp:p>"
            + _p("AFTER")
        )
        out = _parse(_section(body), empty_zip)
        assert out.index("BEFORE") < out.index("SHAPE TEXT") < out.index("AFTER")

    def test_nested_container_shapes(self, empty_zip) -> None:
        body = (
            "<hp:p><hp:run><hp:container>"
            "<hp:rect><hp:drawText><hp:subList>" + _p("INNER ONE") + "</hp:subList></hp:drawText></hp:rect>"
            "<hp:rect><hp:drawText><hp:subList>" + _p("INNER TWO") + "</hp:subList></hp:drawText></hp:rect>"
            "</hp:container></hp:run></hp:p>"
        )
        out = _parse(_section(body), empty_zip)
        assert out.count("INNER ONE") == 1
        assert out.count("INNER TWO") == 1

    def test_table_inside_a_shape_is_rendered_once(self, empty_zip) -> None:
        body = (
            "<hp:p><hp:run><hp:rect><hp:drawText><hp:subList>"
            f"<hp:p><hp:run>{_TABLE}</hp:run></hp:p>"
            "</hp:subList></hp:drawText></hp:rect></hp:run></hp:p>"
        )
        out = _parse(_section(body), empty_zip)
        assert out.count("CELL-A1") == 1
        assert "<table>" in out


class TestControlParts:
    def test_header_and_footer_are_collected_separately(self, empty_zip) -> None:
        body = (
            _p("BODY")
            + "<hp:p><hp:run><hp:ctrl><hp:header><hp:subList>"
            + _p("HEADER TEXT")
            + "</hp:subList></hp:header></hp:ctrl></hp:run></hp:p>"
            + "<hp:p><hp:run><hp:ctrl><hp:footer><hp:subList>"
            + _p("FOOTER TEXT")
            + "</hp:subList></hp:footer></hp:ctrl></hp:run></hp:p>"
        )
        supp = HwpxSupplementary()
        out = _parse(_section(body), empty_zip, supp)

        assert "HEADER TEXT" not in out, "header must not be spliced into body text"
        assert "FOOTER TEXT" not in out
        assert supp.headers == ["HEADER TEXT"]
        assert supp.footers == ["FOOTER TEXT"]

    def test_footnote_is_collected(self, empty_zip) -> None:
        body = (
            _p("BODY")
            + "<hp:p><hp:run><hp:ctrl><hp:footNote><hp:subList>"
            + _p("FOOTNOTE TEXT")
            + "</hp:subList></hp:footNote></hp:ctrl></hp:run></hp:p>"
        )
        supp = HwpxSupplementary()
        _parse(_section(body), empty_zip, supp)
        assert supp.notes == ["FOOTNOTE TEXT"]

    def test_without_an_accumulator_nothing_is_lost(self, empty_zip) -> None:
        """Standalone callers must still see the content, just inline."""
        body = (
            "<hp:p><hp:run><hp:ctrl><hp:header><hp:subList>"
            + _p("HEADER TEXT")
            + "</hp:subList></hp:header></hp:ctrl></hp:run></hp:p>"
        )
        out = _parse(_section(body), empty_zip)
        assert "HEADER TEXT" in out

    def test_table_inside_a_control_is_rendered(self, empty_zip) -> None:
        body = f"<hp:p><hp:run><hp:ctrl>{_TABLE}</hp:ctrl></hp:run></hp:p>"
        out = _parse(_section(body), empty_zip)
        assert out.count("CELL-A1") == 1
        assert "<table>" in out
