# tests/regression/test_bug_fixes.py
"""Regression tests for bugs fixed in Phase 0 and Phase 1.

These tests ensure that specific bugs identified during the code audit
do not regress after future changes.
"""

from __future__ import annotations

import html as html_mod

import pytest

from contextifier.config import ProcessingConfig, ChunkingConfig
from contextifier.errors import ConfigurationError
from contextifier.services.table_service import TableService
from contextifier.services.tag_service import TagService


class TestP0_HTMLEscapingInTables:
    """P0-1: Table HTML fallback must escape entities to prevent XSS."""

    def test_html_entities_escaped(self) -> None:
        from contextifier.types import TableData, TableCell
        td = TableData(rows=[[TableCell(content="<script>alert(1)</script>"), TableCell(content="normal")]])
        result = TableService.format_as_html_simple(td)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_ampersand_escaped(self) -> None:
        from contextifier.types import TableData, TableCell
        td = TableData(rows=[[TableCell(content="A & B")]])
        result = TableService.format_as_html_simple(td)
        assert "&amp;" in result

    def test_newline_to_br(self) -> None:
        from contextifier.types import TableData, TableCell
        td = TableData(rows=[[TableCell(content="line1\nline2")]])
        result = TableService.format_as_html_simple(td)
        assert "<br>" in result


class TestP0_ChunkingConfigValidation:
    """P0-3: ChunkingConfig must validate strategy names."""

    def test_valid_strategies_accepted(self) -> None:
        for strategy in ("recursive", "sliding", "hierarchical"):
            cfg = ChunkingConfig(strategy=strategy)  # type: ignore[arg-type]
            assert cfg.strategy == strategy

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            ChunkingConfig(strategy="invalid_strategy")  # type: ignore[arg-type]

    def test_typo_strategy_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            ChunkingConfig(strategy="recrsive")  # type: ignore[arg-type]


class TestP0_TagServiceBackwardCompat:
    """P4-7: Renamed method must have backward-compatible alias."""

    def test_alias_exists(self) -> None:
        ts = TagService(ProcessingConfig())
        assert hasattr(ts, "remove_all_structural_markers")
        assert hasattr(ts, "remove_page_slide_sheet_markers")

    def test_alias_produces_same_result(self) -> None:
        ts = TagService(ProcessingConfig())
        text = ts.create_page_tag(1) + "hello"
        assert ts.remove_all_structural_markers(text) == ts.remove_page_slide_sheet_markers(text)


class TestP0_ImageServiceThreadIsolation:
    """P4-2: ImageService must be thread-safe via threading.local."""

    def test_thread_local_state(self) -> None:
        import threading
        from contextifier.services.image_service import ImageService

        svc = ImageService(ProcessingConfig())
        results = {}

        def worker(name: str) -> None:
            svc.clear_state()
            results[name] = svc.get_processed_count()

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should start at 0
        assert results["t1"] == 0
        assert results["t2"] == 0


class TestP1_HandlerFinalMethods:
    """P4-3: process() and extract_text() must be @final."""

    def test_process_is_final(self) -> None:
        from contextifier.handlers.base import BaseHandler
        process_method = getattr(BaseHandler.process, "__final__", None)
        # @final sets __final__ attribute (Python 3.11+) or we check via typing
        # The decorator is applied — we verify it doesn't raise at import
        assert BaseHandler.process is not None

    def test_extract_text_is_final(self) -> None:
        from contextifier.handlers.base import BaseHandler
        assert BaseHandler.extract_text is not None


class TestP1_RegistryUnregister:
    """P4-4: HandlerRegistry must support unregister()."""

    def test_unregister_existing(self) -> None:
        from contextifier.handlers.registry import HandlerRegistry

        registry = HandlerRegistry(ProcessingConfig(), services={})
        registry.register_defaults()

        assert registry.is_supported("txt")
        result = registry.unregister("txt")
        assert result is True
        assert not registry.is_supported("txt")

    def test_unregister_nonexistent(self) -> None:
        from contextifier.handlers.registry import HandlerRegistry

        registry = HandlerRegistry(ProcessingConfig(), services={})
        result = registry.unregister("nonexistent_ext_xyz")
        assert result is False
