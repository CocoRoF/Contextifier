# contextifier_new/ocr/engines/anthropic_engine.py
"""Anthropic Claude Vision OCR engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier_new.ocr.base import BaseOCREngine

logger = logging.getLogger("contextifier.ocr.anthropic")

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


class AnthropicOCREngine(BaseOCREngine):
    """OCR engine using Anthropic Claude Vision API.

    Two ways to create:
        engine = AnthropicOCREngine(llm_client)
        engine = AnthropicOCREngine.from_api_key("sk-ant-...")
    """

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> "AnthropicOCREngine":
        """Create engine from an Anthropic API key."""
        from langchain_anthropic import ChatAnthropic

        client = ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return cls(client, prompt=prompt)

    @property
    def provider(self) -> str:
        return "anthropic"

    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_image,
                },
            },
            {"type": "text", "text": prompt},
        ]
