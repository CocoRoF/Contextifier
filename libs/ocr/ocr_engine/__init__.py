# libs/ocr/ocr_engine/__init__.py
# OCR engine module initialization
"""
OCR Engine Module

Provides OCR engine classes for each LLM provider.
"""

from libs.ocr.ocr_engine.openai_ocr import OpenAIOCR
from libs.ocr.ocr_engine.anthropic_ocr import AnthropicOCR
from libs.ocr.ocr_engine.gemini_ocr import GeminiOCR
from libs.ocr.ocr_engine.vllm_ocr import VllmOCR
from libs.ocr.ocr_engine.bedrock_ocr import BedrockOCR

__all__ = [
    "OpenAIOCR",
    "AnthropicOCR",
    "GeminiOCR",
    "VllmOCR",
    "BedrockOCR",
]
