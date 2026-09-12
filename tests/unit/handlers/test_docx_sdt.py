# tests/unit/handlers/test_docx_sdt.py
"""
Structured Document Tags (``<w:sdt>``) in DOCX bodies.

A content control wraps its real content in ``<w:sdtContent>``. Tables of
contents, bibliographies, form fields, cover-page fields and anything a user
inserted as a content control live there. A body walk that only knows ``w:p``
and ``w:tbl`` skips the wrapper and therefore the content inside it.
"""

from __future__ import annotations

from lxml import etree

from contextifier.handlers.docx._constants import NAMESPACES
from contextifier.handlers.docx.content_extractor import iter_block_elements

NS = " ".join(f'xmlns:{k}="{v}"' for k, v in NAMESPACES.items())


def _body(inner: str):
    return etree.fromstring(f"<w:body {NS}>{inner}</w:body>".encode())


def _tags(elements) -> list:
    return [etree.QName(e).localname for e in elements]


class TestSdtFlattening:
    def test_sdt_content_is_surfaced(self) -> None:
        body = _body(
            """
            <w:p><w:r><w:t>before</w:t></w:r></w:p>
            <w:sdt>
              <w:sdtPr/>
              <w:sdtContent>
                <w:p><w:r><w:t>inside control</w:t></w:r></w:p>
                <w:tbl/>
              </w:sdtContent>
            </w:sdt>
            <w:p><w:r><w:t>after</w:t></w:r></w:p>
            """
        )
        assert _tags(iter_block_elements(body)) == ["p", "p", "tbl", "p"]

    def test_nested_sdt_is_flattened(self) -> None:
        body = _body(
            """
            <w:sdt><w:sdtContent>
              <w:sdt><w:sdtContent>
                <w:p><w:r><w:t>deep</w:t></w:r></w:p>
              </w:sdtContent></w:sdt>
            </w:sdtContent></w:sdt>
            """
        )
        assert _tags(iter_block_elements(body)) == ["p"]

    def test_document_order_is_preserved(self) -> None:
        body = _body(
            """
            <w:tbl/>
            <w:sdt><w:sdtContent><w:p/></w:sdtContent></w:sdt>
            <w:tbl/>
            """
        )
        assert _tags(iter_block_elements(body)) == ["tbl", "p", "tbl"]

    def test_sdt_without_content_is_skipped(self) -> None:
        body = _body("<w:sdt><w:sdtPr/></w:sdt><w:p/>")
        assert _tags(iter_block_elements(body)) == ["p"]

    def test_sectpr_still_reaches_the_caller(self) -> None:
        """The body loop branches on sectPr; flattening must not swallow it."""
        body = _body("<w:p/><w:sectPr/>")
        assert _tags(iter_block_elements(body)) == ["p", "sectPr"]


class TestRunLevelSdt:
    def test_inline_content_control_text_is_extracted(self) -> None:
        from contextifier.handlers.docx._paragraph import process_paragraph

        para = etree.fromstring(
            f"""<w:p {NS}>
              <w:r><w:t>Name: </w:t></w:r>
              <w:sdt><w:sdtPr/><w:sdtContent>
                <w:r><w:t>Alice</w:t></w:r>
              </w:sdtContent></w:sdt>
            </w:p>""".encode()
        )
        text, _, _, _ = process_paragraph(para)
        assert text.strip() == "Name: Alice"
