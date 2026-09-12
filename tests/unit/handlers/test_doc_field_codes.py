# tests/unit/handlers/test_doc_field_codes.py
"""
Word field codes in legacy .doc text.

A field is stored as ``\\x13 instruction \\x14 result \\x15``. The instruction
is markup Word never displays. Removing only the control characters leaves it
in the body, so every document with page numbers, a table of contents,
hyperlinks or cross-references carries strings like
``PAGE \\* MERGEFORMAT`` and ``HYPERLINK "https://…"`` through into its chunks.
"""

from __future__ import annotations

from contextifier.handlers.doc._fib import _clean_doc_text, strip_field_codes

BEGIN, SEP, END = "\x13", "\x14", "\x15"


def _field(instruction: str, result: str) -> str:
    return f"{BEGIN}{instruction}{SEP}{result}{END}"


class TestStripFieldCodes:
    def test_instruction_is_dropped_and_result_kept(self) -> None:
        text = "Page " + _field(" PAGE  \\* MERGEFORMAT ", "12") + " of 30"
        assert strip_field_codes(text) == "Page 12 of 30"

    def test_hyperlink_instruction_is_dropped(self) -> None:
        text = "See " + _field('HYPERLINK "https://example.com"', "our site") + "."
        assert strip_field_codes(text) == "See our site."

    def test_field_without_a_result_disappears(self) -> None:
        assert strip_field_codes(f"a{BEGIN} XE \"index entry\" {END}b") == "ab"

    def test_nested_field_inside_a_result(self) -> None:
        inner = _field("PAGE", "7")
        text = "Total: " + _field("IF x", f"page {inner}") + "!"
        assert strip_field_codes(text) == "Total: page 7!"

    def test_nested_field_inside_an_instruction_is_dropped(self) -> None:
        inner = _field("PAGE", "7")
        text = "x" + _field(f"REF {inner}", "Section 3") + "y"
        assert strip_field_codes(text) == "xSection 3y"

    def test_text_without_fields_is_untouched(self) -> None:
        assert strip_field_codes("plain text") == "plain text"

    def test_unterminated_field_does_not_swallow_the_document(self) -> None:
        text = "before " + BEGIN + " PAGE " + SEP + "12 and then the rest of the page"
        out = strip_field_codes(text)

        assert out == "before  PAGE 12 and then the rest of the page", (
            "the remainder must be kept once, not dropped and not duplicated"
        )
        assert BEGIN not in out and SEP not in out

    def test_stray_end_marker_leaves_text_intact(self) -> None:
        """Nothing to strip without an opener; the marker itself is a control
        character that `_clean_doc_text` removes."""
        assert strip_field_codes(f"a{END}b") == f"a{END}b"
        assert _clean_doc_text(f"a{END}b") == "ab"


class TestCleanDocText:
    def test_clean_removes_instructions_end_to_end(self) -> None:
        raw = (
            "Report of " + _field(" PAGE  \\* MERGEFORMAT ", "12") + " pages.\r"
            "See " + _field('HYPERLINK "https://example.com"', "our site") + "."
        )
        cleaned = _clean_doc_text(raw)

        assert "MERGEFORMAT" not in cleaned
        assert "HYPERLINK" not in cleaned
        assert "Report of 12 pages." in cleaned
        assert "See our site." in cleaned
