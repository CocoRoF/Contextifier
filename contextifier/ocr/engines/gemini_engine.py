# contextifier/ocr/engines/gemini_engine.py
"""Google Gemini Vision OCR engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier.ocr.base import BaseOCREngine

logger = logging.getLogger("contextifier.ocr.gemini")

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiOCREngine(BaseOCREngine):
    """OCR engine using Google Gemini Vision API.

    Two ways to create:
        engine = GeminiOCREngine(llm_client)
        engine = GeminiOCREngine.from_api_key("...")
    """

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> "GeminiOCREngine":
        """Create engine from a Google API key."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: Dict[str, Any] = {
            "model": model,
            "google_api_key": api_key,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        return cls(ChatGoogleGenerativeAI(**kwargs), prompt=prompt)

    @property
    def provider(self) -> str:
        return "gemini"

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
