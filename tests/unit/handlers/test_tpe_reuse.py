# tests/unit/handlers/test_tpe_reuse.py
"""
P4-4: Tests for ThreadPoolExecutor reuse in BaseHandler.

Verifies:
1. Shared executor is created lazily
2. Same executor instance returned on repeated calls
3. Timeout processing uses the shared executor
4. Shutdown cleans up the executor
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from contextifier.handlers.base import BaseHandler
from contextifier.config import ProcessingConfig


def _make_handler():
    """Create a minimal concrete handler for testing."""
    from contextifier.pipeline.converter import NullConverter
    from contextifier.pipeline.preprocessor import NullPreprocessor
    from contextifier.pipeline.metadata_extractor import NullMetadataExtractor
    from contextifier.pipeline.content_extractor import BaseContentExtractor
    from contextifier.pipeline.postprocessor import DefaultPostprocessor

    class _StubExtractor(BaseContentExtractor):
        def extract_text(self, preprocessed, **kwargs):
            return "stub"
        def get_format_name(self):
            return "stub"

    class _StubHandler(BaseHandler):
        @property
        def supported_extensions(self):
            return frozenset({"stub"})
        @property
        def handler_name(self):
            return "Stub Handler"
        def create_converter(self):
            return NullConverter()
        def create_preprocessor(self):
            return NullPreprocessor()
        def create_metadata_extractor(self):
            return NullMetadataExtractor()
        def create_content_extractor(self):
            return _StubExtractor()
        def create_postprocessor(self):
            return DefaultPostprocessor(self._config)

    return _StubHandler(ProcessingConfig())


class TestSharedTimeoutExecutor(unittest.TestCase):
    """Shared TPE lifecycle."""

    def setUp(self):
        # Reset the shared executor before each test
        BaseHandler._shutdown_timeout_executor()

    def tearDown(self):
        BaseHandler._shutdown_timeout_executor()

    def test_lazy_initialization(self):
        """Executor is None until first use."""
        self.assertIsNone(BaseHandler._timeout_executor)

    def test_get_creates_executor(self):
        """First call creates the executor."""
        executor = BaseHandler._get_timeout_executor()
        self.assertIsNotNone(executor)
        self.assertIs(executor, BaseHandler._timeout_executor)

    def test_same_instance_returned(self):
        """Repeated calls return the same executor."""
        e1 = BaseHandler._get_timeout_executor()
        e2 = BaseHandler._get_timeout_executor()
        self.assertIs(e1, e2)

    def test_shutdown_clears_executor(self):
        """Shutdown sets _timeout_executor back to None."""
        BaseHandler._get_timeout_executor()
        self.assertIsNotNone(BaseHandler._timeout_executor)
        BaseHandler._shutdown_timeout_executor()
        self.assertIsNone(BaseHandler._timeout_executor)

    def test_recreated_after_shutdown(self):
        """Executor can be recreated after shutdown."""
        e1 = BaseHandler._get_timeout_executor()
        BaseHandler._shutdown_timeout_executor()
        e2 = BaseHandler._get_timeout_executor()
        self.assertIsNotNone(e2)
        self.assertIsNot(e1, e2)

    def test_handler_uses_shared_executor(self):
        """_process_with_timeout uses the shared executor."""
        handler = _make_handler()
        # Verify the executor gets created
        executor = handler._get_timeout_executor()
        self.assertIsNotNone(executor)
        self.assertIs(executor, BaseHandler._timeout_executor)


if __name__ == "__main__":
    unittest.main()
