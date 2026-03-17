# contextifier_new/ocr/processor.py
"""
OCRProcessor — Orchestrator for replacing image tags with OCR results.

Responsibilities:
- Scan text for image tags (configurable pattern)
- Resolve image paths to local files
- Invoke the engine for each image
- Replace tags with OCR output
- Report progress via callback protocol

This replaces BOTH the old process_text_with_ocr() function AND
BaseOCR.process_text() method — single source of truth.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Pattern, Protocol, Union

from contextifier.config import ProcessingConfig
from contextifier.ocr.base import BaseOCREngine

logger = logging.getLogger("contextifier.ocr")


# ── Progress Protocol ─────────────────────────────────────────────────────

class OCRProgressCallback(Protocol):
    """Protocol for OCR progress reporting."""

    def __call__(self, event: OCRProgressEvent) -> None:
        ...


@dataclass(frozen=True)
class OCRProgressEvent:
    """Structured progress event (replaces raw dict from old code)."""

    event_type: str          # 'tag_processing' | 'tag_processed' | 'completed'
    current_index: int       # 0-based index of current image
    total_count: int         # Total number of images
    image_path: str = ""     # Path of image being processed
    status: str = ""         # 'success' | 'failed' | ''
    error: str = ""          # Error message if failed


# ── OCRProcessor ──────────────────────────────────────────────────────────

class OCRProcessor:
    """
    Orchestrates OCR replacement of image tags in text.

    Usage:
        engine = OpenAIOCREngine(llm_client)
        processor = OCRProcessor(engine, config)
        result = processor.process(text)

    The processor does NOT own the engine; it receives it at construction.
    """

    def __init__(
        self,
        engine: BaseOCREngine,
        config: ProcessingConfig,
        *,
        image_pattern: Optional[Pattern[str]] = None,
    ) -> None:
        """
        Args:
            engine: OCR engine to use for image-to-text conversion.
            config: Processing configuration.
            image_pattern: Custom regex pattern for image tags.
                           Must have exactly one capture group for the path.
        """
        self._engine = engine
        self._config = config
        if image_pattern is not None:
            self._pattern = image_pattern
        else:
            # Build from config.tags to stay in sync with TagConfig
            tags = config.tags
            self._pattern = re.compile(
                rf"{re.escape(tags.image_prefix)}\s*(.+?)\s*{re.escape(tags.image_suffix)}"
            )

    @property
    def engine(self) -> BaseOCREngine:
        return self._engine

    def process(
        self,
        text: str,
        *,
        progress_callback: Optional[Callable[[OCRProgressEvent], Any]] = None,
    ) -> str:
        """
        Replace all image tags in text with OCR results.

        Args:
            text: Text containing image tags.
            progress_callback: Optional progress reporting function.

        Returns:
            Text with image tags replaced by OCR output.
        """
        if not text:
            return text

        image_paths = self._extract_image_paths(text)
        if not image_paths:
            logger.debug("No image tags found in text")
            return text

        total = len(image_paths)
        logger.info(f"Detected {total} image tag(s) for OCR processing")

        result = text
        success_count = 0

        for idx, img_path in enumerate(image_paths):
            # Notify: processing started
            if progress_callback:
                progress_callback(OCRProgressEvent(
                    event_type="tag_processing",
                    current_index=idx,
                    total_count=total,
                    image_path=img_path,
                ))

            # Resolve and convert
            local_path = self._resolve_image_path(img_path)
            if local_path is None:
                logger.warning(f"Image not found, keeping original tag: {img_path}")
                self._notify_failed(progress_callback, idx, total, img_path, "File not found")
                continue

            ocr_text = self._engine.convert_image_to_text(local_path)
            if ocr_text is None or ocr_text.startswith("[Image conversion error:"):
                logger.warning(f"OCR failed, keeping original tag: {img_path}")
                self._notify_failed(
                    progress_callback, idx, total, img_path,
                    ocr_text or "OCR returned None",
                )
                continue

            # Replace the tag
            result = self._replace_tag(result, img_path, ocr_text)
            success_count += 1

            if progress_callback:
                progress_callback(OCRProgressEvent(
                    event_type="tag_processed",
                    current_index=idx,
                    total_count=total,
                    image_path=img_path,
                    status="success",
                ))

        # Final notification
        if progress_callback:
            progress_callback(OCRProgressEvent(
                event_type="completed",
                current_index=total,
                total_count=total,
                status=f"{success_count}/{total} succeeded",
            ))

        return result

    # ── Private helpers ───────────────────────────────────────────────────

    def _extract_image_paths(self, text: str) -> List[str]:
        """Extract all image paths from text using the configured pattern."""
        return self._pattern.findall(text)

    def _resolve_image_path(self, image_path: str) -> Optional[str]:
        """Validate that the image path exists and is non-empty."""
        try:
            path = image_path.strip()
            if not os.path.isabs(path):
                path = os.path.abspath(path)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
            return None
        except Exception:
            return None

    def _replace_tag(self, text: str, img_path: str, replacement: str) -> str:
        """Replace the image tag for the given path with OCR result."""
        escaped = re.escape(img_path)
        pattern_str = self._pattern.pattern
        # Replace the capture group with the escaped literal path
        tag_pattern_str = re.sub(r"\([^)]+\)", escaped, pattern_str, count=1)
        tag_re = re.compile(tag_pattern_str)
        return tag_re.sub(replacement, text)

    @staticmethod
    def _notify_failed(
        callback: Optional[Callable[[OCRProgressEvent], Any]],
        idx: int,
        total: int,
        path: str,
        error: str,
    ) -> None:
        if callback:
            callback(OCRProgressEvent(
                event_type="tag_processed",
                current_index=idx,
                total_count=total,
                image_path=path,
                status="failed",
                error=error,
            ))


__all__ = [
    "OCRProcessor",
    "OCRProgressEvent",
    "OCRProgressCallback",
    "DEFAULT_IMAGE_TAG_PATTERN",
]
