# contextifier_new/ocr/engines/bedrock_engine.py
"""AWS Bedrock (Claude) Vision OCR engine."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from contextifier_new.ocr.base import BaseOCREngine

logger = logging.getLogger("contextifier.ocr.bedrock")

DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"


class BedrockOCREngine(BaseOCREngine):
    """OCR engine using AWS Bedrock Claude Vision API.

    Two ways to create:
        engine = BedrockOCREngine(llm_client)
        engine = BedrockOCREngine.from_aws_credentials(
            aws_access_key_id="...", aws_secret_access_key="...",
            aws_region="us-east-1",
        )
    """

    def __init__(self, llm_client: Any, *, prompt: Optional[str] = None) -> None:
        super().__init__(llm_client, prompt=prompt)

    @classmethod
    def from_aws_credentials(
        cls,
        *,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model: str = DEFAULT_BEDROCK_MODEL,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        connect_timeout: int = 60,
        read_timeout: int = 120,
        max_retries: int = 10,
    ) -> "BedrockOCREngine":
        """Create engine from AWS credentials."""
        from langchain_aws import ChatBedrockConverse
        from botocore.config import Config as BotocoreConfig

        # Set env vars if provided
        if aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        if aws_session_token:
            os.environ["AWS_SESSION_TOKEN"] = aws_session_token
        if aws_region:
            os.environ["AWS_DEFAULT_REGION"] = aws_region

        boto_config = BotocoreConfig(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"max_attempts": max_retries, "mode": "adaptive"},
        )

        client_kwargs: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "config": boto_config,
        }
        if aws_region:
            client_kwargs["region_name"] = aws_region
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        return cls(ChatBedrockConverse(**client_kwargs), prompt=prompt)

    @property
    def provider(self) -> str:
        return "aws_bedrock"

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
