# contextifier/ocr/engines/deepseek_engine.py
"""DeepSeek-OCR engine, served through a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier.ocr.base import BaseOCREngine

logger = logging.getLogger("contextifier.ocr.deepseek")

DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-OCR-2"

# The recipe's own instruction. DeepSeek-OCR is trained on a fixed prompt
# rather than on arbitrary instructions, so the descriptive prompt the other
# engines use returns little or nothing.
DEEPSEEK_PROMPT = "Free OCR."


class DeepSeekOCREngine(BaseOCREngine):
    """
    OCR engine for DeepSeek-OCR / DeepSeek-OCR-2.

    It speaks the same OpenAI-compatible protocol as the generic vLLM engine
    but is not interchangeable with it: the model's chat template puts the
    image before the instruction, and it expects its own prompt. Sent the
    generic engine's message shape, it answers with nothing.

    The server has to be started for it — the recipe asks for the n-gram
    logits processor, prefix caching off and the multimodal processor cache
    disabled — which is outside this class's reach:

    .. code-block:: shell

        vllm serve deepseek-ai/DeepSeek-OCR-2 \\
          --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \\
          --no-enable-prefix-caching --mm-processor-cache-gb 0

    Usage::

        engine = DeepSeekOCREngine.from_endpoint("http://localhost:8000/v1")
    """

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt or DEEPSEEK_PROMPT)

    @classmethod
    def from_endpoint(
        cls,
        base_url: str,
        *,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        api_key: str = "EMPTY",
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> "DeepSeekOCREngine":
        """Create the engine from a vLLM endpoint URL."""
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
        return "deepseek"

    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """Image first, then the instruction — the order the template requires."""
        return [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
            {"type": "text", "text": prompt},
        ]


__all__ = ["DeepSeekOCREngine", "DEEPSEEK_PROMPT", "DEFAULT_DEEPSEEK_MODEL"]
