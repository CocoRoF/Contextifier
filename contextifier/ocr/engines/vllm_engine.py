# contextifier/ocr/engines/vllm_engine.py
"""Self-hosted VLLM Vision OCR engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier.ocr.base import BaseOCREngine, SIMPLE_OCR_PROMPT

logger = logging.getLogger("contextifier.ocr.vllm")

DEFAULT_VLLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


class VLLMOCREngine(BaseOCREngine):
    """OCR engine using a self-hosted VLLM Vision model.

    Two ways to create:
        engine = VLLMOCREngine(llm_client)
        engine = VLLMOCREngine.from_endpoint("http://localhost:8000/v1")
    """

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt or SIMPLE_OCR_PROMPT)

    @classmethod
    def from_endpoint(
        cls,
        base_url: str,
        *,
        model: str = DEFAULT_VLLM_MODEL,
        api_key: str = "EMPTY",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> "VLLMOCREngine":
        """Create engine from a VLLM endpoint URL."""
        from langchain_openai import ChatOpenAI

        kwargs: Dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        return cls(ChatOpenAI(**kwargs), prompt=prompt)

    @property
    def provider(self) -> str:
        return "vllm"

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
