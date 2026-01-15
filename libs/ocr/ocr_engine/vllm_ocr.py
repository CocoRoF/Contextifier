# libs/ocr/ocr_engine/vllm_ocr.py
# OCR class using vLLM-based Vision model
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-vllm")

# Default model (varies by user environment)
DEFAULT_VLLM_MODEL = "Qwen/Qwen2-VL-7B-Instruct"


class VllmOCR(BaseOCR):
    """
    OCR processing class using vLLM-based Vision model.

    Uses OpenAI-compatible API provided by vLLM server.

    Example:
        ```python
        from libs.ocr.ocr_engine import VllmOCR

        # Method 1: Initialize with base_url and model
        ocr = VllmOCR(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2-VL-7B-Instruct"
        )

        # Method 2: When api_key is required
        ocr = VllmOCR(
            base_url="http://your-vllm-server:8000/v1",
            api_key="your-api-key",
            model="Qwen/Qwen2-VL-7B-Instruct"
        )

        # Method 3: Use existing LLM client
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="Qwen/Qwen2-VL-7B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY"
        )
        ocr = VllmOCR(llm_client=llm)

        # Single image conversion
        result = await ocr.convert_image_to_text("/path/to/image.png")
        ```
    """

    # vLLM uses simple prompt
    DEFAULT_PROMPT = "Describe the contents of this image."

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = "EMPTY",
        model: str = DEFAULT_VLLM_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize vLLM OCR.

        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8000/v1")
            api_key: API key (default: "EMPTY", vLLM default setting)
            model: Model name to use (default: Qwen/Qwen2-VL-7B-Instruct)
            llm_client: Existing LangChain client (if provided, base_url, api_key, and model are ignored)
            prompt: Custom prompt (if None, SIMPLE_PROMPT is used)
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (if None, model default is used)
        """
        # vLLM uses simple prompt by default
        if prompt is None:
            prompt = self.DEFAULT_PROMPT

        if llm_client is None:
            if base_url is None:
                raise ValueError("Either base_url or llm_client is required.")

            from langchain_openai import ChatOpenAI

            client_kwargs = {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "temperature": temperature,
            }

            if max_tokens is not None:
                client_kwargs["max_tokens"] = max_tokens

            llm_client = ChatOpenAI(**client_kwargs)
            logger.info(f"[vLLM OCR] Client created: base_url={base_url}, model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        self.base_url = base_url
        logger.info("[vLLM OCR] Initialization completed")

    @property
    def provider(self) -> str:
        return "vllm"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {"type": "text", "text": self.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            }
        ]
