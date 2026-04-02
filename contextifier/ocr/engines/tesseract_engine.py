# contextifier/ocr/engines/tesseract_engine.py
"""
Tesseract OCR engine — local OCR without LLM dependency.

Unlike the LLM-based engines (OpenAI, Anthropic, etc.), Tesseract
runs entirely locally via ``pytesseract``.  Since there is no LLM
client involved, ``build_message_content()`` is a no-op and
``convert_image_to_text()`` is overridden to call pytesseract directly.

Requirements:
    - pip install pytesseract Pillow
    - Tesseract binary installed and on PATH (or configured via
      ``pytesseract.pytesseract.tesseract_cmd``)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from contextifier.ocr.base import BaseOCREngine

logger = logging.getLogger("contextifier.ocr.tesseract")

_DEFAULT_LANG = "eng"


class TesseractOCREngine(BaseOCREngine):
    """
    OCR engine using Tesseract via pytesseract.

    Usage::

        engine = TesseractOCREngine(lang="kor+eng")
        text = engine.convert_image_to_text("/path/to/image.png")
    """

    def __init__(
        self,
        *,
        lang: str = _DEFAULT_LANG,
        tesseract_cmd: Optional[str] = None,
        config: str = "",
        prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            lang: Tesseract language(s), e.g. ``"eng"``, ``"kor+eng"``.
            tesseract_cmd: Path to tesseract binary (if not on PATH).
            config: Extra Tesseract CLI config string (e.g. ``"--psm 6"``).
            prompt: Unused for Tesseract but kept for API compatibility.
        """
        # Pass None as llm_client since Tesseract is local
        super().__init__(llm_client=None, prompt=prompt)
        self._lang = lang
        self._tesseract_config = config

        if tesseract_cmd:
            import pytesseract as _pt

            _pt.pytesseract.tesseract_cmd = tesseract_cmd

    # ── Abstract interface implementation ─────────────────────────────────

    @property
    def provider(self) -> str:
        return "tesseract"

    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """No-op — Tesseract does not use LLM message payloads."""
        return []

    # ── Override: direct Tesseract call ───────────────────────────────────

    def convert_image_to_text(self, image_path: str) -> Optional[str]:
        """
        Convert image to text using Tesseract OCR.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Extracted text wrapped in ``[Figure:...]`` format,
            or error string on failure.
        """
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)
            result = pytesseract.image_to_string(
                img,
                lang=self._lang,
                config=self._tesseract_config,
            ).strip()

            if not result:
                logger.info(
                    "[TESSERACT] No text detected: %s",
                    os.path.basename(image_path),
                )
                return "[Figure: (no text detected)]"

            logger.info(
                "[TESSERACT] OCR completed: %s (%d chars)",
                os.path.basename(image_path),
                len(result),
            )
            return f"[Figure:{result}]"

        except ImportError:
            logger.error(
                "[TESSERACT] pytesseract or Pillow not installed. "
                "Install with: pip install pytesseract Pillow"
            )
            return "[Image conversion error: pytesseract not installed]"
        except Exception as e:
            logger.error("[TESSERACT] OCR failed: %s — %s", image_path, e)
            return f"[Image conversion error: {e!s}]"

    def __repr__(self) -> str:
        return f"TesseractOCREngine(lang='{self._lang}')"


__all__ = ["TesseractOCREngine"]
