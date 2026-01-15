# libs/ocr/ocr_engine/openai_ocr.py
# OCR class using OpenAI Vision model
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-openai")

# Default model
DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAIOCR(BaseOCR):
    """
    OCR processing class using OpenAI Vision model.

    Supported models: gpt-4-vision-preview, gpt-4o, gpt-4o-mini, etc.

    Example:
        ```python
        from libs.ocr.ocr_engine import OpenAIOCR

        # Method 1: Initialize with api_key and model
        ocr = OpenAIOCR(api_key="sk-...", model="gpt-4o")

        # Method 2: Use existing LLM client
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="sk-...")
        ocr = OpenAIOCR(llm_client=llm)

        # Single image conversion
        result = await ocr.convert_image_to_text("/path/to/image.png")

        # Process image tags in text
        text = "Document content [Image:/path/to/image.png] continues..."
        processed = await ocr.process_text(text)
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_OPENAI_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI OCR.

        Args:
            api_key: OpenAI API key (required if llm_client is not provided)
            model: Model name to use (default: gpt-4o)
            llm_client: Existing LangChain OpenAI client (if provided, api_key and model are ignored)
            prompt: Custom prompt (if None, default prompt is used)
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (if None, model default is used)
            base_url: OpenAI API base URL (for Azure, etc.)
        """
        if llm_client is None:
            if api_key is None:
                raise ValueError("Either api_key or llm_client is required.")

            from langchain_openai import ChatOpenAI

            client_kwargs = {
                "model": model,
                "api_key": api_key,
                "temperature": temperature,
            }

            if max_tokens is not None:
                client_kwargs["max_tokens"] = max_tokens

            if base_url is not None:
                client_kwargs["base_url"] = base_url

            llm_client = ChatOpenAI(**client_kwargs)
            logger.info(f"[OpenAI OCR] Client created: model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        logger.info("[OpenAI OCR] Initialization completed")

    @property
    def provider(self) -> str:
        return "openai"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {"type": "text", "text": self.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            }
        ]
