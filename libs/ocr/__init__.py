# libs/ocr/__init__.py
# OCR module package initialization
"""
OCR Processing Module

This module provides OCR functionality to extract text from images
using various LLM Vision models.

Usage Examples:
    ```python
    from libs.ocr.ocr_engine import OpenAIOCR, AnthropicOCR, GeminiOCR, VllmOCR

    # OCR processing with OpenAI Vision model
    ocr = OpenAIOCR(api_key="sk-...", model="gpt-4o")
    result = await ocr.convert_image_to_text("/path/to/image.png")

    # OCR processing with Anthropic Claude Vision model
    ocr = AnthropicOCR(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    result = await ocr.convert_image_to_text("/path/to/image.png")

    # OCR processing with Google Gemini Vision model
    ocr = GeminiOCR(api_key="...", model="gemini-2.0-flash")
    result = await ocr.convert_image_to_text("/path/to/image.png")

    # OCR processing with vLLM-based Vision model
    ocr = VllmOCR(base_url="http://localhost:8000/v1", model="Qwen/Qwen2-VL-7B-Instruct")
    result = await ocr.convert_image_to_text("/path/to/image.png")
    ```

Classes:
    - BaseOCR: Abstract base class for OCR processing
    - OpenAIOCR: OpenAI Vision model based OCR (ocr_engine module)
    - AnthropicOCR: Anthropic Claude Vision model based OCR (ocr_engine module)
    - GeminiOCR: Google Gemini Vision model based OCR (ocr_engine module)
    - VllmOCR: vLLM-based Vision model OCR (ocr_engine module)
"""

from libs.ocr.base import BaseOCR
from libs.ocr.ocr_engine import OpenAIOCR, AnthropicOCR, GeminiOCR, VllmOCR
from libs.ocr.ocr_processor import (
    IMAGE_TAG_PATTERN,
    extract_image_tags,
    load_image_from_path,
    convert_image_to_text_with_llm,
    process_text_with_ocr,
    process_text_with_ocr_progress,
    process_batch_texts_with_ocr,
    _b64_from_file,
    _get_mime_type,
)

__all__ = [
    # Base Class
    "BaseOCR",
    # OCR Engines
    "OpenAIOCR",
    "AnthropicOCR",
    "GeminiOCR",
    "VllmOCR",
    # Functions
    "IMAGE_TAG_PATTERN",
    "extract_image_tags",
    "load_image_from_path",
    "convert_image_to_text_with_llm",
    "process_text_with_ocr",
    "process_text_with_ocr_progress",
    "process_batch_texts_with_ocr",
]
