# tests/unit/chunking/test_table_segment_packing.py
"""
Packing for spreadsheet-style content.

A sheet is split into segments — tables, charts, prose — and each chunk is
prefixed with the document metadata and the sheet marker so it can stand on
its own. Emitting one chunk per segment makes that prefix the majority of
every chunk: a sheet of six small tables produced eight chunks of under a
hundred characters each against a thousand-character budget, most of it the
same boilerplate repeated.

Segments that fit are packed together instead. The budget is measured on
segment bodies, because the prefix is written once per chunk rather than once
per segment.
"""

from __future__ import annotations

from contextifier.chunking.chunker import TextChunker
from contextifier.config import ProcessingConfig

_PREFIX = "[Document-Metadata]\n  title: Book\n[/Document-Metadata]\n\n"


def _chunk(text: str, size: int = 1000):
    return TextChunker(ProcessingConfig()).chunk(
        text, chunk_size=size, file_extension="xlsx"
    )


def _small_tables(count: int, sheet: str = "S") -> str:
    body = "\n\n".join(
        f"| a{i} | b{i} |\n| --- | --- |\n| c{i} | d{i} |" for i in range(count)
    )
    return f"{_PREFIX}[Sheet: {sheet}]\n{body}"


class TestPacking:
    def test_small_segments_share_a_chunk(self) -> None:
        chunks = _chunk(_small_tables(6))
        assert len(chunks) == 1, (
            f"six small tables should fit one 1000-char chunk, got "
            f"{[len(c) for c in chunks]}"
        )

    def test_budget_is_respected(self) -> None:
        chunks = _chunk(_small_tables(40), size=600)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 600, f"chunk of {len(chunk)} exceeds the budget"

    def test_every_chunk_keeps_its_context(self) -> None:
        chunks = _chunk(_small_tables(40), size=600)
        for chunk in chunks:
            assert chunk.startswith("[Document-Metadata]")
            assert "[Sheet: S]" in chunk

    def test_no_content_is_lost(self) -> None:
        chunks = _chunk(_small_tables(12), size=400)
        joined = "".join(chunks)
        for i in range(12):
            assert f"c{i}" in joined and f"d{i}" in joined

    def test_sheets_are_never_merged_into_one_chunk(self) -> None:
        text = (
            f"{_PREFIX}[Sheet: First]\n| a | b |\n| --- | --- |\n| c | d |\n\n"
            "[Sheet: Second]\n| e | f |\n| --- | --- |\n| g | h |"
        )
        chunks = _chunk(text)

        for chunk in chunks:
            assert not ("[Sheet: First]" in chunk and "[Sheet: Second]" in chunk), (
                "a chunk must not span two sheets:\n" + chunk
            )

    def test_oversized_table_still_splits(self) -> None:
        rows = "\n".join(f"| value {i} padding padding | second column |" for i in range(60))
        text = f"{_PREFIX}[Sheet: S]\n| h1 | h2 |\n| --- | --- |\n{rows}"
        chunks = _chunk(text, size=500)
        assert len(chunks) > 1
        joined = "".join(chunks)
        assert "value 59" in joined


class TestImageAndTextboxSegments:
    def test_image_tag_does_not_claim_its_own_chunk(self) -> None:
        text = (
            f"{_PREFIX}[Sheet: S]\n"
            "Some leading prose about the sheet.\n\n"
            "[Image:temp/images/diagram.png]\n\n"
            "Some trailing prose about the sheet."
        )
        chunks = _chunk(text)

        assert len(chunks) == 1, (
            "an image tag between two short paragraphs should not force three "
            f"chunks, got {[len(c) for c in chunks]}"
        )
        assert "[Image:temp/images/diagram.png]" in chunks[0]

    def test_image_tag_is_never_split(self) -> None:
        text = (
            f"{_PREFIX}[Sheet: S]\n"
            + ("filler text. " * 40)
            + "\n\n[Image:temp/images/a-rather-long-image-path-name.png]\n\n"
            + ("more filler. " * 40)
        )
        chunks = _chunk(text, size=400)
        tag = "[Image:temp/images/a-rather-long-image-path-name.png]"
        assert sum(c.count(tag) for c in chunks) == 1
