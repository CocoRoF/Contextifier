# tests/unit/test_config.py
"""Unit tests for contextifier.config — configuration validation."""

from __future__ import annotations

import pytest

from contextifier.config import (
    ChunkingConfig,
    ProcessingConfig,
    TagConfig,
    ImageConfig,
)
from contextifier.errors import ConfigurationError


class TestChunkingConfig:
    """Tests for ChunkingConfig, especially strategy validation."""

    def test_default_strategy(self) -> None:
        cfg = ChunkingConfig()
        assert cfg.strategy == "recursive"

    def test_valid_strategies(self) -> None:
        for s in ("recursive", "sliding", "hierarchical"):
            cfg = ChunkingConfig(strategy=s)
            assert cfg.strategy == s

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid chunking strategy"):
            ChunkingConfig(strategy="unknown")

    def test_typo_strategy_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid chunking strategy"):
            ChunkingConfig(strategy="recusive")  # typo

    def test_frozen(self) -> None:
        cfg = ChunkingConfig()
        with pytest.raises(AttributeError):
            cfg.chunk_size = 9999  # type: ignore[misc]


class TestProcessingConfig:
    """Tests for root ProcessingConfig."""

    def test_defaults(self) -> None:
        cfg = ProcessingConfig()
        assert cfg.chunking.chunk_size == 1000
        assert cfg.tags.page_prefix is not None

    def test_to_dict_roundtrip(self) -> None:
        cfg = ProcessingConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "chunking" in d or "tags" in d  # has sub-configs

    def test_format_options_frozen(self) -> None:
        cfg = ProcessingConfig(format_options={"pdf": {"mode": "fast"}})
        # format_options should be a MappingProxyType (immutable)
        with pytest.raises(TypeError):
            cfg.format_options["pdf"] = {}  # type: ignore[index]

    def test_with_chunking(self) -> None:
        cfg = ProcessingConfig()
        cfg2 = cfg.with_chunking(chunk_size=500)
        assert cfg2.chunking.chunk_size == 500
        assert cfg.chunking.chunk_size != 500  # original unchanged
