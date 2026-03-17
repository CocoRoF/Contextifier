# contextifier_new/ocr/__init__.py
"""
OCR — Optical Character Recognition Subsystem

Provides a unified interface for converting images embedded in extracted
text into textual descriptions using Vision Language Models.

Architecture:
    BaseOCREngine (ABC)       — abstract engine interface
      ├── OpenAIOCREngine      — OpenAI Vision API
      ├── AnthropicOCREngine   — Anthropic Claude Vision
      ├── GeminiOCREngine      — Google Gemini Vision
      ├── BedrockOCREngine     — AWS Bedrock (Claude)
      └── VLLMOCREngine        — Self-hosted VLLM

    OCRProcessor               — orchestrator (finds image tags, invokes engine)

Design improvements over old code:
- Legacy function API (process_text_with_ocr) eliminated
- No duplicated prompt constants between base.py and ocr_processor.py
- Prompt is config-driven through OCRConfig
- Engine is just message formatting; orchestration lives in OCRProcessor
- Progress callback uses a proper Protocol, not raw Dict
"""

from contextifier_new.ocr.base import BaseOCREngine
from contextifier_new.ocr.processor import OCRProcessor

__all__ = [
    "BaseOCREngine",
    "OCRProcessor",
]
