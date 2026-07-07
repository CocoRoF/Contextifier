# tests/unit/ocr/test_ocr_prompt_language.py
"""
P2-7: OCR prompt language configurability tests.

Verifies that:
1. get_ocr_prompt() returns language-specific prompts
2. OCRConfig.prompt_language field exists and defaults to "ko"
3. Unknown languages fall back to English
"""

from __future__ import annotations


from contextifier.config import OCRConfig
from contextifier.ocr.base import (
    DEFAULT_OCR_PROMPT,
    get_ocr_prompt,
    _OCR_PROMPT_TEMPLATE,
)


class TestOCRPromptLanguage:

    def test_default_prompt_is_korean(self):
        assert "Output in Korean" in DEFAULT_OCR_PROMPT

    def test_get_ocr_prompt_korean(self):
        prompt = get_ocr_prompt("ko")
        assert "Output in Korean" in prompt

    def test_get_ocr_prompt_english(self):
        prompt = get_ocr_prompt("en")
        assert "Output in English" in prompt
        assert "Output in Korean" not in prompt

    def test_get_ocr_prompt_unknown_falls_back_to_english(self):
        prompt = get_ocr_prompt("fr")
        assert "Output in English" in prompt

    def test_prompt_template_has_ko_and_en(self):
        assert "ko" in _OCR_PROMPT_TEMPLATE
        assert "en" in _OCR_PROMPT_TEMPLATE

    def test_ocr_config_prompt_language_default(self):
        config = OCRConfig()
        assert config.prompt_language == "ko"

    def test_ocr_config_prompt_language_custom(self):
        config = OCRConfig(prompt_language="en")
        assert config.prompt_language == "en"

    def test_ocr_config_serialization(self):
        """prompt_language survives to_dict/from_dict round-trip."""
        from contextifier.config import ProcessingConfig

        cfg = ProcessingConfig(ocr=OCRConfig(prompt_language="en"))
        d = cfg.to_dict()
        restored = ProcessingConfig.from_dict(d)
        assert restored.ocr.prompt_language == "en"
