# tests/unit/chunking/test_chunker.py
"""Unit tests for TextChunker."""

from __future__ import annotations

import pytest

from contextifier.config import ProcessingConfig
from contextifier.chunking.chunker import TextChunker


@pytest.fixture()
def chunker() -> TextChunker:
    return TextChunker(ProcessingConfig())


class TestChunkerInit:
    def test_strategies_ordered_by_priority(self, chunker: TextChunker) -> None:
        priorities = [s.priority for s in chunker.strategies]
        assert priorities == sorted(priorities)

    def test_has_builtin_strategies(self, chunker: TextChunker) -> None:
        names = [s.strategy_name for s in chunker.strategies]
        assert len(names) >= 2  # at least plain + one more

    def test_add_custom_strategy(self, chunker: TextChunker) -> None:
        from contextifier.chunking.strategies.base import BaseChunkingStrategy
        from unittest.mock import MagicMock
        custom = MagicMock(spec=BaseChunkingStrategy)
        custom.priority = 1
        custom.strategy_name = "custom"
        chunker.add_strategy(custom)
        assert any(s.strategy_name == "custom" for s in chunker.strategies)


class TestChunkEmptyInput:
    def test_empty_string_returns_empty_list(self, chunker: TextChunker) -> None:
        result = chunker.chunk("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self, chunker: TextChunker) -> None:
        result = chunker.chunk("   \n  ")
        assert result == []


class TestChunkPlainText:
    def test_short_text_single_chunk(self) -> None:
        config = ProcessingConfig()
        chunker = TextChunker(config)
        text = "Hello world"
        result = chunker.chunk(text)
        assert len(result) >= 1
        assert "Hello world" in result[0]

    def test_long_text_multiple_chunks(self) -> None:
        config = ProcessingConfig()
        config = config.with_chunking(chunk_size=100, chunk_overlap=10)
        chunker = TextChunker(config)
        text = "word " * 200  # ~1000 chars
        result = chunker.chunk(text)
        assert len(result) > 1

    def test_chunk_size_override(self) -> None:
        config = ProcessingConfig()
        config = config.with_chunking(chunk_overlap=10)
        chunker = TextChunker(config)
        text = "word " * 200
        result = chunker.chunk(text, chunk_size=50)
        # With chunk_size=50, more chunks expected
        assert len(result) > 1


class TestChunkWithExtension:
    def test_csv_extension_hint(self, chunker: TextChunker) -> None:
        csv_text = "a,b,c\n1,2,3\n4,5,6"
        result = chunker.chunk(csv_text, file_extension="csv")
        assert len(result) >= 1
