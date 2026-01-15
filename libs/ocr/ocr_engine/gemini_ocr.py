# libs/ocr/ocr_engine/gemini_ocr.py
# OCR class using Google Gemini Vision model
import logging
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-gemini")

# Default model
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiOCR(BaseOCR):
    """
    OCR processing class using Google Gemini Vision model.

    Supported models: gemini-pro-vision, gemini-1.5-pro, gemini-2.0-flash, etc.

    Example:
        ```python
        from libs.ocr.ocr_engine import GeminiOCR

        # Method 1: Initialize with api_key and model
        ocr = GeminiOCR(api_key="...", model="gemini-2.0-flash")

        # Method 2: Use existing LLM client
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="...")
        ocr = GeminiOCR(llm_client=llm)

        # Single image conversion
        result = await ocr.convert_image_to_text("/path/to/image.png")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize Gemini OCR.

        Args:
            api_key: Google API key (required if llm_client is not provided)
            model: Model name to use (default: gemini-2.0-flash)
            llm_client: Existing LangChain Gemini client (if provided, api_key and model are ignored)
            prompt: Custom prompt (if None, default prompt is used)
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (if None, model default is used)
        """
        if llm_client is None:
            if api_key is None:
                raise ValueError("Either api_key or llm_client is required.")

            from langchain_google_genai import ChatGoogleGenerativeAI

            client_kwargs = {
                "model": model,
                "google_api_key": api_key,
                "temperature": temperature,
            }

            if max_tokens is not None:
                client_kwargs["max_output_tokens"] = max_tokens

            llm_client = ChatGoogleGenerativeAI(**client_kwargs)
            logger.info(f"[Gemini OCR] Client created: model={model}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        logger.info("[Gemini OCR] Initialization completed")

    @property
    def provider(self) -> str:
        return "gemini"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        return [
            {"type": "text", "text": self.prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            }
        ]
