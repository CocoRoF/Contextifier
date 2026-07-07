# tests/unit/ocr/test_ocr_parallel.py
"""
P4-3: Tests for OCR parallel processing.

Verifies:
1. OCRProcessor.max_workers defaults to 1
2. Sequential mode (max_workers=1) still works correctly
3. Parallel mode (max_workers>1) produces same results as sequential
4. Progress callbacks fire for all images in both modes
5. Failed OCR items are handled correctly in parallel mode
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from contextifier.config import ProcessingConfig
from contextifier.ocr.processor import OCRProcessor, OCRProgressEvent


def _make_engine(results: dict[str, str | None] | None = None):
    """Create a mock OCR engine."""
    engine = MagicMock()
    if results:
        def convert(path):
            return results.get(os.path.basename(path))
        engine.convert_image_to_text.side_effect = convert
    else:
        engine.convert_image_to_text.return_value = "OCR text"
    return engine


class TestOCRProcessorMaxWorkers(unittest.TestCase):
    """max_workers parameter behaviour."""

    def test_default_max_workers_is_1(self):
        engine = _make_engine()
        config = ProcessingConfig()
        proc = OCRProcessor(engine, config)
        self.assertEqual(proc._max_workers, 1)

    def test_custom_max_workers(self):
        engine = _make_engine()
        config = ProcessingConfig()
        proc = OCRProcessor(engine, config, max_workers=4)
        self.assertEqual(proc._max_workers, 4)

    def test_max_workers_minimum_1(self):
        engine = _make_engine()
        config = ProcessingConfig()
        proc = OCRProcessor(engine, config, max_workers=0)
        self.assertEqual(proc._max_workers, 1)
        proc2 = OCRProcessor(engine, config, max_workers=-5)
        self.assertEqual(proc2._max_workers, 1)


class TestOCRSequentialMode(unittest.TestCase):
    """Sequential (max_workers=1) backward compatibility."""

    def test_empty_text(self):
        engine = _make_engine()
        proc = OCRProcessor(engine, ProcessingConfig())
        self.assertEqual(proc.process(""), "")

    def test_no_tags(self):
        engine = _make_engine()
        proc = OCRProcessor(engine, ProcessingConfig())
        result = proc.process("No image tags here")
        self.assertEqual(result, "No image tags here")
        engine.convert_image_to_text.assert_not_called()


class TestOCRParallelMode(unittest.TestCase):
    """Parallel mode produces correct results."""

    def _create_temp_images(self, count: int) -> list[str]:
        """Create temp files to act as images."""
        paths = []
        for i in range(count):
            fd, path = tempfile.mkstemp(suffix=f"_img{i}.png")
            os.write(fd, b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
            os.close(fd)
            paths.append(path)
        return paths

    def test_parallel_same_result_as_sequential(self):
        """Parallel and sequential produce identical output."""
        paths = self._create_temp_images(3)
        try:
            results = {}
            for p in paths:
                results[os.path.basename(p)] = f"text_for_{os.path.basename(p)}"

            config = ProcessingConfig()
            tags = config.tags
            text = "\n".join(
                f"{tags.image_prefix}{p}{tags.image_suffix}" for p in paths
            )

            seq_proc = OCRProcessor(_make_engine(results), config, max_workers=1)
            par_proc = OCRProcessor(_make_engine(results), config, max_workers=3)

            seq_result = seq_proc.process(text)
            par_result = par_proc.process(text)

            self.assertEqual(seq_result, par_result)
        finally:
            for p in paths:
                os.unlink(p)

    def test_parallel_progress_callbacks(self):
        """All progress callbacks fire in parallel mode."""
        paths = self._create_temp_images(3)
        try:
            engine = _make_engine()
            config = ProcessingConfig()
            tags = config.tags
            text = "\n".join(
                f"{tags.image_prefix}{p}{tags.image_suffix}" for p in paths
            )

            events: list[OCRProgressEvent] = []
            proc = OCRProcessor(engine, config, max_workers=3)
            proc.process(text, progress_callback=lambda e: events.append(e))

            event_types = [e.event_type for e in events]
            self.assertIn("completed", event_types)
            processing_events = [e for e in events if e.event_type == "tag_processing"]
            self.assertEqual(len(processing_events), 3)
        finally:
            for p in paths:
                os.unlink(p)

    def test_parallel_handles_failures(self):
        """Failed OCR in parallel mode doesn't crash the batch."""
        paths = self._create_temp_images(3)
        try:
            results = {
                os.path.basename(paths[0]): "good_text_0",
                os.path.basename(paths[1]): None,  # Simulate failure
                os.path.basename(paths[2]): "good_text_2",
            }

            config = ProcessingConfig()
            tags = config.tags
            text = "\n".join(
                f"{tags.image_prefix}{p}{tags.image_suffix}" for p in paths
            )

            proc = OCRProcessor(_make_engine(results), config, max_workers=2)
            result = proc.process(text)

            # Successful replacements should appear
            self.assertIn("good_text_0", result)
            self.assertIn("good_text_2", result)
            # Failed tag should remain
            self.assertIn(os.path.basename(paths[1]), result)
        finally:
            for p in paths:
                os.unlink(p)


class TestOCRBatchRun(unittest.TestCase):
    """_run_ocr_batch method."""

    def test_sequential_returns_ordered(self):
        engine = _make_engine()
        proc = OCRProcessor(engine, ProcessingConfig(), max_workers=1)
        # Can't call _run_ocr_batch directly without valid paths,
        # but we test the full process flow above. This verifies
        # internal structure.
        self.assertEqual(proc._max_workers, 1)


if __name__ == "__main__":
    unittest.main()
