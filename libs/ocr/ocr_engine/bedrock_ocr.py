# libs/ocr/ocr_engine/bedrock_ocr.py
# OCR class using AWS Bedrock Vision model
import logging
import os
from typing import Any, Optional

from libs.ocr.base import BaseOCR

logger = logging.getLogger("ocr-bedrock")

# Default model
DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"


class BedrockOCR(BaseOCR):
    """
    OCR processing class using AWS Bedrock Vision model.

    Supports Claude and other vision-capable models available on AWS Bedrock.

    Example:
        ```python
        from libs.ocr.ocr_engine import BedrockOCR

        # Method 1: Initialize with AWS credentials
        ocr = BedrockOCR(
            aws_access_key_id="AKIA...",
            aws_secret_access_key="...",
            aws_region="us-east-1",
            model="anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

        # Method 2: Use existing AWS credentials from environment
        # (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        ocr = BedrockOCR(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

        # Method 3: Use with session token (temporary credentials)
        ocr = BedrockOCR(
            aws_access_key_id="ASIA...",
            aws_secret_access_key="...",
            aws_session_token="...",
            aws_region="ap-northeast-2"
        )

        # Method 4: Use existing LLM client
        from langchain_aws import ChatBedrockConverse
        llm = ChatBedrockConverse(model="anthropic.claude-3-5-sonnet-20241022-v2:0")
        ocr = BedrockOCR(llm_client=llm)

        # Single image conversion
        result = ocr.convert_image_to_text("/path/to/image.png")
        ```
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model: str = DEFAULT_BEDROCK_MODEL,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        connect_timeout: int = 60,
        read_timeout: int = 120,
        max_retries: int = 10,
    ):
        """
        Initialize AWS Bedrock OCR.

        Args:
            aws_access_key_id: AWS access key ID (if not provided, uses environment variable)
            aws_secret_access_key: AWS secret access key (if not provided, uses environment variable)
            aws_session_token: AWS session token for temporary credentials (optional)
            aws_region: AWS region (default: from environment or "ap-northeast-2")
            endpoint_url: Custom endpoint URL (for VPC endpoints, etc.)
            model: Model ID to use (default: anthropic.claude-3-5-sonnet-20241022-v2:0)
            llm_client: Existing LangChain Bedrock client (if provided, other params are ignored)
            prompt: Custom prompt (if None, default prompt is used)
            temperature: Generation temperature (default: 0.0)
            max_tokens: Maximum number of tokens (default: 4096)
            connect_timeout: Connection timeout in seconds (default: 60)
            read_timeout: Read timeout in seconds (default: 120)
            max_retries: Maximum retry attempts (default: 10)
        """
        if llm_client is None:
            from langchain_aws import ChatBedrockConverse
            from botocore.config import Config as BotocoreConfig

            # Set environment variables for boto3 auto-discovery
            if aws_access_key_id:
                os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
            if aws_secret_access_key:
                os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
            if aws_session_token:
                os.environ["AWS_SESSION_TOKEN"] = aws_session_token

            # Determine region
            if not aws_region:
                aws_region = os.environ.get(
                    "AWS_REGION",
                    os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
                )

            logger.info(f"[Bedrock OCR] Using: model={model}, region={aws_region}")

            # Configure botocore with retry settings
            bedrock_config = BotocoreConfig(
                retries={
                    "max_attempts": max_retries,
                    "mode": "adaptive",
                },
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )

            # Build kwargs for ChatBedrockConverse
            llm_kwargs = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "disable_streaming": False,
                "config": bedrock_config,
            }

            if aws_region:
                llm_kwargs["region_name"] = aws_region
            if aws_access_key_id:
                llm_kwargs["aws_access_key_id"] = aws_access_key_id
            if aws_secret_access_key:
                llm_kwargs["aws_secret_access_key"] = aws_secret_access_key
            if aws_session_token:
                llm_kwargs["aws_session_token"] = aws_session_token
            if endpoint_url:
                llm_kwargs["endpoint_url"] = endpoint_url

            llm_client = ChatBedrockConverse(**llm_kwargs)
            logger.info(f"[Bedrock OCR] Client created: model={model}, region={aws_region}")

        super().__init__(llm_client=llm_client, prompt=prompt)
        self.model = model
        self.aws_region = aws_region
        logger.info("[Bedrock OCR] Initialization completed")

    @property
    def provider(self) -> str:
        return "aws_bedrock"

    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        """
        Build message content for AWS Bedrock.

        AWS Bedrock uses the same format as Anthropic Claude models.
        """
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


__all__ = ["BedrockOCR", "DEFAULT_BEDROCK_MODEL"]
