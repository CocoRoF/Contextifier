# contextifier_new/ocr/engines/__init__.py
"""
OCR Engine Implementations

Each engine implements BaseOCREngine.build_message_content()
for its specific provider's message format.
"""

from contextifier.ocr.engines.openai_engine import OpenAIOCREngine
from contextifier.ocr.engines.anthropic_engine import AnthropicOCREngine
from contextifier.ocr.engines.gemini_engine import GeminiOCREngine
from contextifier.ocr.engines.bedrock_engine import BedrockOCREngine
from contextifier.ocr.engines.vllm_engine import VLLMOCREngine

__all__ = [
    "OpenAIOCREngine",
    "AnthropicOCREngine",
    "GeminiOCREngine",
    "BedrockOCREngine",
    "VLLMOCREngine",
]
