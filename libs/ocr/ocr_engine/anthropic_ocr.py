# libs/ocr/ocr_engine/anthropic_ocr.py
# OCR class using Anthropic Claude Vision model
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-anthropic")

# Default model
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


class AnthropicOCR(BaseOCR):
    """
    OCR processing class using Anthropic Claude Vision model.

    Supported models: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-sonnet-4, etc.

    Example:
        ```python
        from libs.ocr.ocr_engine import AnthropicOCR

        # Method 1: Initialize with api_key and model
        ocr = AnthropicOCR(api_key="sk-ant-...", model="claude-sonnet-4-20250514")

        # Method 2: Use existing LLM client
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, api_key="sk-ant-...")
        ocr = AnthropicOCR(llm_client=llm)

        # Single image conversion
        result = await ocr.convert_image_to_text("/path/to/image.png")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        """
        Initialize Anthropic OCR.

        Args:
            api_key: Anthropic API key (required if llm_client is not provided)
            model: Model name to use (default: claude-sonnet-4-20250514)
            llm_client: Existing LangChain Anthropic client (if provided, api_key and model are ignored)
            prompt: Custom prompt (if None, default prompt is used)
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (default: 4096)
        """
        if llm_client is None:
            if api_key is None:
                raise ValueError("Either api_key or llm_client is required.")

            from langchain_anthropic import ChatAnthropic

            llm_client = ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(f"[Anthropic OCR] Client created: model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        logger.info("[Anthropic OCR] Initialization completed")

    @property
    def provider(self) -> str:
        return "anthropic"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_image
                }
            },
            {"type": "text", "text": self.prompt}
        ]
