# contextifier_new/ocr/engines/openai_engine.py
"""OpenAI Vision OCR engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier.ocr.base import BaseOCREngine

logger = logging.getLogger("contextifier.ocr.openai")

DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAIOCREngine(BaseOCREngine):
    """OCR engine using OpenAI Vision API (GPT-4V / GPT-4o).

    Two ways to create:
        # 1. Direct injection (preferred for advanced use)
        engine = OpenAIOCREngine(llm_client)

        # 2. Factory from API key (convenience)
        engine = OpenAIOCREngine.from_api_key("sk-...")
        engine = OpenAIOCREngine.from_api_key("sk-...", model="gpt-4o", temperature=0.0)
    """

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> "OpenAIOCREngine":
        """Create engine from an OpenAI API key.

        Args:
            api_key: OpenAI API key.
            model: Model name (default: gpt-4o).
            prompt: Custom OCR prompt.
            temperature: LLM temperature.
            max_tokens: Maximum output tokens.
            base_url: Custom API base URL.
        """
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if base_url is not None:
            kwargs["base_url"] = base_url

        return cls(ChatOpenAI(**kwargs), prompt=prompt)

    @property
    def provider(self) -> str:
        return "openai"

    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
        ]
