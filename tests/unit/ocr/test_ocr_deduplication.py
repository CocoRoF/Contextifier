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
