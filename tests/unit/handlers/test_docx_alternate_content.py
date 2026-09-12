# tests/unit/handlers/test_docx_alternate_content.py
"""
``mc:AlternateContent`` handling in DOCX paragraphs.

Word stores every modern shape twice: a DrawingML version under ``mc:Choice``
and a legacy VML version under ``mc:Fallback``. Both carry the same text, so
a naive recursive search over ``.//w:r`` yields it twice, while a walk that
only knows ``w:r`` and ``w:hyperlink`` yields it zero times — which is what
Contextifier did, losing every text box, shape caption and grouped-shape label.

A resolver must pick exactly one branch and descend into it.
"""

from __future__ import annotations

from lxml import etree

from contextifier.handlers.docx._paragraph import process_paragraph
from contextifier.handlers.docx._constants import NAMESPACES

NS = " ".join(f'xmlns:{k}="{v}"' for k, v in NAMESPACES.items())


def _p(inner: str):
    return etree.fromstring(f"<w:p {NS}>{inner}</w:p>".encode())


_TEXTBOX_RUN = """
  <w:r>
    <mc:AlternateContent>
      <mc:Choice Requires="wps">
        <w:drawing>
          <wp:inline>
            <a:graphic>
              <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
                <wps:wsp>
                  <wps:txbx>
                    <w:txbxContent>
                      <w:p><w:r><w:t>BOXED LINE ONE</w:t></w:r></w:p>
                      <w:p><w:r><w:t>BOXED LINE TWO</w:t></w:r></w:p>
                    </w:txbxContent>
                  </wps:txbx>
                </wps:wsp>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </mc:Choice>
      <mc:Fallback>
        <w:pict>
          <v:shape>
            <v:textbox>
              <w:txbxContent>
                <w:p><w:r><w:t>BOXED LINE ONE</w:t></w:r></w:p>
                <w:p><w:r><w:t>BOXED LINE TWO</w:t></w:r></w:p>
              </w:txbxContent>
            </v:textbox>
          </v:shape>
        </w:pict>
      </mc:Fallback>
    </mc:AlternateContent>
  </w:r>
"""


class TestAlternateContentText:
    def test_textbox_text_is_extracted_once(self) -> None:
        text, _, _, _ = process_paragraph(_p(_TEXTBOX_RUN))

        assert "BOXED LINE ONE" in text
        assert "BOXED LINE TWO" in text
        assert text.count("BOXED LINE ONE") == 1, (
            "Choice and Fallback both carry the text; only one branch may be read"
        )

    def test_surrounding_run_text_is_preserved(self) -> None:
        text, _, _, _ = process_paragraph(
            _p("<w:r><w:t>before </w:t></w:r>" + _TEXTBOX_RUN + "<w:r><w:t> after</w:t></w:r>")
        )

        assert text.startswith("before ")
        assert text.rstrip().endswith("after")
        assert "BOXED LINE ONE" in text

    def test_fallback_is_used_when_choice_is_absent(self) -> None:
        """Word 2007-era files ship only the VML branch."""
        inner = """
          <w:r>
            <mc:AlternateContent>
              <mc:Fallback>
                <w:pict><v:shape><v:textbox><w:txbxContent>
                  <w:p><w:r><w:t>LEGACY BOX</w:t></w:r></w:p>
                </w:txbxContent></v:textbox></v:shape></w:pict>
              </mc:Fallback>
            </mc:AlternateContent>
          </w:r>
        """
        text, _, _, _ = process_paragraph(_p(inner))
        assert "LEGACY BOX" in text


class TestAlternateContentDrawings:
    def test_picture_inside_choice_is_still_detected(self) -> None:
        """Resolving the branch must not drop the image it contains."""
        inner = """
          <w:r>
            <mc:AlternateContent>
              <mc:Choice Requires="wpg">
                <w:drawing>
                  <wp:inline>
                    <a:graphic>
                      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                        <pic:pic><pic:blipFill><a:blip r:embed="rId99"/></pic:blipFill></pic:pic>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing>
              </mc:Choice>
              <mc:Fallback>
                <w:pict><v:shape><v:imagedata r:id="rId99"/></v:shape></w:pict>
              </mc:Fallback>
            </mc:AlternateContent>
          </w:r>
        """
        _, drawings, picts, _ = process_paragraph(_p(inner))

        rel_ids = [d.rel_id for d in drawings] + [p.rel_id for p in picts]
        assert rel_ids.count("rId99") == 1, (
            f"image must be reported exactly once, got {rel_ids}"
        )


class TestPlainDrawingTextbox:
    def test_textbox_outside_alternate_content(self) -> None:
        """A shape can sit in a bare ``w:drawing`` with no compatibility wrapper."""
        inner = """
          <w:r>
            <w:drawing><wp:inline><a:graphic>
              <a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">
                <wps:wsp><wps:txbx><w:txbxContent>
                  <w:p><w:r><w:t>BARE SHAPE TEXT</w:t></w:r></w:p>
                </w:txbxContent></wps:txbx></wps:wsp>
              </a:graphicData>
            </a:graphic></wp:inline></w:drawing>
          </w:r>
        """
        text, _, _, _ = process_paragraph(_p(inner))
        assert "BARE SHAPE TEXT" in text
