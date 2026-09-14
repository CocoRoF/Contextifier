# tests/unit/ocr/test_ocr_deduplication.py
"""
The same image referenced twice is one conversion, not two.

A logo in a repeated header or a diagram cited from two sections produces
several identical tags. Each one used to become its own model call — paid for,
waited on, and discarded in favour of a result already in hand.
"""

from __future__ import annotations

from typing import List

from contextifier.config import ProcessingConfig
from contextifier.ocr.processor import OCRProcessor


class CountingEngine:
    """Records every path it is asked to convert."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def convert_image_to_text(self, path: str) -> str:
        self.calls.append(path)
        return f"<text of {path.split('/')[-1]}>"


def _processor(engine: CountingEngine, tmp_path) -> OCRProcessor:
    return OCRProcessor(engine=engine, config=ProcessingConfig())


def _make_images(tmp_path, *names: str) -> List[str]:
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        paths.append(str(p))
    return paths


def test_repeated_tag_is_converted_once(tmp_path) -> None:
    (a,) = _make_images(tmp_path, "logo.png")
    engine = CountingEngine()

    text = f"Header [Image:{a}]\n\nBody\n\nFooter [Image:{a}]"
    out = _processor(engine, tmp_path).process(text)

    assert engine.calls == [a], f"expected one conversion, got {engine.calls}"
    assert out.count("<text of logo.png>") == 2, (
        "every occurrence must still be replaced"
    )
    assert "[Image:" not in out


def test_distinct_images_are_each_converted(tmp_path) -> None:
    a, b = _make_images(tmp_path, "one.png", "two.png")
    engine = CountingEngine()

    text = f"[Image:{a}] and [Image:{b}] and [Image:{a}]"
    out = _processor(engine, tmp_path).process(text)

    assert sorted(engine.calls) == sorted([a, b])
    assert "<text of one.png>" in out and "<text of two.png>" in out


class TableStructureEngine:
    """Answers the way a vision model actually does: with markup."""

    def convert_image_to_text(self, path: str) -> str:
        return (
            "```html\n<table><tr><th>Item</th><th>Qty</th></tr>"
            "<tr><td>Bolt</td><td>12</td></tr></table>\n```"
        )


def test_output_inside_a_cell_is_flattened(tmp_path) -> None:
    """Model output is markup; a `<table>` inside a `<td>` breaks the parser."""
    (img,) = _make_images(tmp_path, "cell.png")
    text = (
        "<table>\n<tr><th>Figure</th><th>Content</th></tr>\n"
        f"<tr><td>Fig 1</td><td>[Image:{img}]</td></tr>\n</table>"
    )
    out = OCRProcessor(engine=TableStructureEngine(), config=ProcessingConfig()).process(
        text
    )

    assert out.count("<table>") == 1, "nested table markup leaked into a cell:\n" + out
    assert out.count("</table>") == 1
    assert "Bolt" in out and "12" in out
    assert "&lt;" in out or "Item" in out


def test_output_outside_a_cell_is_inserted_verbatim(tmp_path) -> None:
    (img,) = _make_images(tmp_path, "body.png")
    text = f"Before\n\n[Image:{img}]\n\nAfter"
    out = OCRProcessor(engine=TableStructureEngine(), config=ProcessingConfig()).process(
        text
    )

    assert "<table>" in out, "structure outside a cell should be preserved"


def test_failed_conversion_keeps_the_original_tag(tmp_path) -> None:
    class FailingEngine:
        def convert_image_to_text(self, path: str) -> str:
            return "[Image conversion error: unreachable]"

    (img,) = _make_images(tmp_path, "broken.png")
    text = f"a [Image:{img}] b"
    out = OCRProcessor(engine=FailingEngine(), config=ProcessingConfig()).process(text)

    assert f"[Image:{img}]" in out, "a failed conversion must leave the tag to retry"
