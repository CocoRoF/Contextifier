"""Chunk metadata enrichment + JSON context rendering (0.3.x).

RAG pipelines need per-chunk provenance: the page/slide/sheet markers and
markdown headings Contextifier injects into extracted text are now
promoted into ChunkMetadata, and JSON payloads render as path-per-line
context that chunkers can split safely.
"""

from __future__ import annotations

from contextifier import DocumentProcessor
from contextifier.chunking.metadata_enricher import enrich_chunk_metadata
from contextifier.handlers.text.content_extractor import _render_json_context
from contextifier.types import Chunk, ChunkMetadata


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(text=text, metadata=ChunkMetadata(chunk_index=idx))


# ── enricher unit behavior ───────────────────────────────────────────


def test_page_marker_carries_forward():
    chunks = [
        _chunk("[Page Number: 1]\n서문 내용", 0),
        _chunk("이어지는 본문 (마커 없음)", 1),
        _chunk("[Page Number: 2]\n2페이지 시작", 2),
    ]
    enrich_chunk_metadata(chunks)
    assert chunks[0].metadata.page_number == 1
    assert chunks[1].metadata.page_number == 1  # inherited
    assert chunks[2].metadata.page_number == 2  # marker at chunk head


def test_mid_chunk_marker_applies_to_next_chunk():
    chunks = [
        _chunk("1페이지 끝부분\n[Page Number: 2]\n2페이지 도입", 0),
        _chunk("2페이지 계속", 1),
    ]
    enrich_chunk_metadata(chunks)
    assert chunks[0].metadata.page_number is None  # started before any marker
    assert chunks[1].metadata.page_number == 2


def test_heading_path_breadcrumb():
    chunks = [
        _chunk("# 설치\n## 요구사항\n파이썬 3.12", 0),
        _chunk("추가 요구사항 설명", 1),
        _chunk("## 빠른 시작\n명령어", 2),
    ]
    enrich_chunk_metadata(chunks)
    assert chunks[0].metadata.heading_path == "설치 > 요구사항"
    assert chunks[1].metadata.heading_path == "설치 > 요구사항"
    assert chunks[2].metadata.heading_path == "설치 > 빠른 시작"  # sibling replaced


def test_sheet_marker():
    chunks = [_chunk("[Sheet: 매출]\n1월: 100", 0), _chunk("2월: 200", 1)]
    enrich_chunk_metadata(chunks)
    assert chunks[0].metadata.sheet_name == "매출"
    assert chunks[1].metadata.sheet_name == "매출"


def test_plain_string_chunks_untouched():
    chunks = [Chunk(text="no meta", metadata=None)]
    enrich_chunk_metadata(chunks)  # must not raise
    assert chunks[0].metadata is None


# ── end-to-end through TextChunker ───────────────────────────────────


def test_chunker_populates_page_metadata_end_to_end():
    text = "\n\n".join(
        f"[Page Number: {p}]\n" + (f"{p}페이지 본문 문단. " * 40)
        for p in range(1, 4)
    )
    chunks = DocumentProcessor().chunk_text(
        text, file_extension="pdf", chunk_size=400, chunk_overlap=50,
        include_position_metadata=True,
    )
    assert chunks and not isinstance(chunks[0], str)
    pages = {c.metadata.page_number for c in chunks}
    assert pages >= {1, 2, 3}
    assert all(c.metadata.page_number is not None for c in chunks)


# ── JSON context rendering ───────────────────────────────────────────


def test_json_renders_leaf_paths():
    out = _render_json_context(
        '{"user": {"name": "하렴", "roles": ["boss", "dev"]}, "count": 2}'
    )
    assert "user.name: \"하렴\"" in out
    assert "user.roles[0]: \"boss\"" in out
    assert "count: 2" in out


def test_json_array_of_objects_gets_record_boundaries():
    out = _render_json_context(
        '{"items": [{"id": 1, "t": "a"}, {"id": 2, "t": "b"}]}'
    )
    blocks = out.split("\n\n")
    assert len(blocks) >= 2  # record boundary between array elements
    assert "items[1].id: 2" in out


def test_invalid_json_falls_back():
    assert _render_json_context("not json at all {") is None
