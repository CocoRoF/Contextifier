# tests/unit/handlers/test_delegation.py
"""P1-6 + P3-6: Tests for delegation depth limit and handler delegation paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from contextifier.config import ProcessingConfig
from contextifier.handlers.base import BaseHandler
from contextifier.errors import HandlerExecutionError
from contextifier.types import ExtractionResult, FileContext


class _StubHandler(BaseHandler):
    """Minimal concrete handler for testing."""

    @property
    def supported_extensions(self):
        return frozenset({"stub"})

    @property
    def handler_name(self):
        return "StubHandler"

    def create_converter(self):
        return MagicMock()

    def create_preprocessor(self):
        return MagicMock()

    def create_metadata_extractor(self):
        return MagicMock()

    def create_content_extractor(self):
        return MagicMock()

    def create_postprocessor(self):
        return MagicMock()


@pytest.fixture()
def handler() -> _StubHandler:
    h = _StubHandler(ProcessingConfig())
    registry = MagicMock()
    h.set_registry(registry)
    return h


class TestDelegationDepthLimit:
    def test_single_delegation_succeeds(self, handler: _StubHandler) -> None:
        """One level of delegation should work fine."""
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="delegated")
        handler._handler_registry.get_handler.return_value = target

        result = handler._delegate_to("rtf", {"file_data": b"test"})
        assert result.text == "delegated"

    def test_depth_limit_raises(self, handler: _StubHandler) -> None:
        """Exceeding _MAX_DELEGATION_DEPTH should raise HandlerExecutionError."""
        # Simulate already being at max depth
        BaseHandler._delegation_state.depth = BaseHandler._MAX_DELEGATION_DEPTH

        try:
            with pytest.raises(HandlerExecutionError, match="Delegation depth limit"):
                handler._delegate_to(
                    "rtf",
                    {"file_data": b"test"},
                    fallback_to_self=False,
                )
        finally:
            BaseHandler._delegation_state.depth = 0

    def test_depth_restores_after_delegation(self, handler: _StubHandler) -> None:
        """Depth counter should be restored after delegation completes."""
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="ok")
        handler._handler_registry.get_handler.return_value = target

        BaseHandler._delegation_state.depth = 0
        handler._delegate_to("rtf", {"file_data": b"test"})
        assert getattr(BaseHandler._delegation_state, "depth", 0) == 0

    def test_depth_restores_on_exception(self, handler: _StubHandler) -> None:
        """Depth counter should restore even when delegation raises."""
        target = MagicMock()
        target.process.side_effect = RuntimeError("boom")
        handler._handler_registry.get_handler.return_value = target

        BaseHandler._delegation_state.depth = 1
        try:
            # fallback_to_self=True → should catch and fallback
            handler._delegate_to("rtf", {"file_data": b"test"}, fallback_to_self=True)
        except Exception:
            pass
        assert BaseHandler._delegation_state.depth == 1


# ═══════════════════════════════════════════════════════════════════════════════
# P3-6: Handler delegation path tests — magic byte detection triggers
# ═══════════════════════════════════════════════════════════════════════════════

class TestDOCDelegation:
    """DOCHandler delegates to DOCX, RTF, or HTML based on magic bytes."""

    def _make_doc_handler(self):
        from contextifier.handlers.doc.handler import DOCHandler
        h = DOCHandler(ProcessingConfig())
        registry = MagicMock()
        h.set_registry(registry)
        return h, registry

    def test_zip_magic_delegates_to_docx(self) -> None:
        h, reg = self._make_doc_handler()
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="docx content")
        reg.get_handler.return_value = target

        ctx = FileContext(file_data=b"PK\x03\x04" + b"\x00" * 100, extension="doc")
        result = h._check_delegation(ctx)
        assert result is not None
        assert result.text == "docx content"
        reg.get_handler.assert_called_with("docx")

    def test_rtf_magic_delegates_to_rtf(self) -> None:
        h, reg = self._make_doc_handler()
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="rtf content")
        reg.get_handler.return_value = target

        ctx = FileContext(file_data=b"{\\rtf1" + b"\x00" * 100, extension="doc")
        result = h._check_delegation(ctx)
        assert result is not None
        reg.get_handler.assert_called_with("rtf")

    def test_html_magic_delegates_to_html(self) -> None:
        h, reg = self._make_doc_handler()
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="html content")
        reg.get_handler.return_value = target

        ctx = FileContext(
            file_data=b"<html><body>test</body></html>" + b"\x00" * 100,
            extension="doc",
        )
        result = h._check_delegation(ctx)
        assert result is not None
        reg.get_handler.assert_called_with("html")

    def test_ole2_magic_no_delegation(self) -> None:
        h, reg = self._make_doc_handler()
        # OLE2 Compound Document magic bytes
        ole2_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        ctx = FileContext(file_data=ole2_magic + b"\x00" * 100, extension="doc")
        result = h._check_delegation(ctx)
        assert result is None  # No delegation; handle as OLE2 DOC


class TestPPTDelegation:
    """PPTHandler delegates to PPTX when ZIP magic is detected."""

    def _make_ppt_handler(self):
        from contextifier.handlers.ppt.handler import PPTHandler
        h = PPTHandler(ProcessingConfig())
        registry = MagicMock()
        h.set_registry(registry)
        return h, registry

    def test_zip_magic_delegates_to_pptx(self) -> None:
        h, reg = self._make_ppt_handler()
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="pptx content")
        reg.get_handler.return_value = target

        ctx = FileContext(file_data=b"PK\x03\x04" + b"\x00" * 100, extension="ppt")
        result = h._check_delegation(ctx)
        assert result is not None
        reg.get_handler.assert_called_with("pptx")

    def test_ole2_no_delegation(self) -> None:
        h, reg = self._make_ppt_handler()
        ole2_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        ctx = FileContext(file_data=ole2_magic + b"\x00" * 100, extension="ppt")
        result = h._check_delegation(ctx)
        assert result is None


class TestXLSDelegation:
    """XLSHandler delegates to XLSX when ZIP magic is detected."""

    def _make_xls_handler(self):
        from contextifier.handlers.xls.handler import XLSHandler
        h = XLSHandler(ProcessingConfig())
        registry = MagicMock()
        h.set_registry(registry)
        return h, registry

    def test_zip_magic_delegates_to_xlsx(self) -> None:
        h, reg = self._make_xls_handler()
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="xlsx content")
        reg.get_handler.return_value = target

        ctx = FileContext(file_data=b"PK\x03\x04" + b"\x00" * 100, extension="xls")
        result = h._check_delegation(ctx)
        assert result is not None
        reg.get_handler.assert_called_with("xlsx")

    def test_biff_no_delegation(self) -> None:
        h, reg = self._make_xls_handler()
        ole2_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        ctx = FileContext(file_data=ole2_magic + b"\x00" * 100, extension="xls")
        result = h._check_delegation(ctx)
        assert result is None


class TestHWPDelegation:
    """HWPHandler delegates to HWPX when ZIP magic is detected."""

    def _make_hwp_handler(self):
        from contextifier.handlers.hwp.handler import HWPHandler
        h = HWPHandler(ProcessingConfig())
        registry = MagicMock()
        h.set_registry(registry)
        return h, registry

    def test_zip_magic_delegates_to_hwpx(self) -> None:
        h, reg = self._make_hwp_handler()
        target = MagicMock()
        target.process.return_value = ExtractionResult(text="hwpx content")
        reg.get_handler.return_value = target

        ctx = FileContext(file_data=b"PK\x03\x04" + b"\x00" * 100, extension="hwp")
        result = h._check_delegation(ctx)
        assert result is not None
        reg.get_handler.assert_called_with("hwpx")

    def test_ole2_no_delegation(self) -> None:
        h, reg = self._make_hwp_handler()
        ole2_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        ctx = FileContext(file_data=ole2_magic + b"\x00" * 100, extension="hwp")
        result = h._check_delegation(ctx)
        assert result is None
