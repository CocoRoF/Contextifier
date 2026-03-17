# contextifier_new/ocr/base.py
"""
BaseOCREngine — Abstract interface for all OCR engine implementations.

Each engine is responsible ONLY for formatting the LLM message payload
specific to its provider. The actual orchestration (reading images,
replacing tags, progress tracking) is handled by OCRProcessor.

Old problems fixed:
- BaseOCR had convert_image_to_text() AND process_text() — mixing concerns
- process_text() duplicated between BaseOCR and ocr_processor module
- Prompt was hardcoded in 2 places
"""

from __future__ import annotations

import base64
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("contextifier.ocr")


# ── MIME type mapping ─────────────────────────────────────────────────────

_MIME_MAP: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".svg": "image/svg+xml",
}


def get_mime_type(file_path: str) -> str:
    """Return MIME type for an image file path."""
    ext = os.path.splitext(file_path)[1].lower()
    return _MIME_MAP.get(ext, "image/jpeg")


def encode_image_base64(file_path: str) -> str:
    """Read file and return Base64-encoded string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Default prompts ───────────────────────────────────────────────────────

DEFAULT_OCR_PROMPT: str = (
    "Extract meaningful information from this image.\n\n"
    "**If the image contains a TABLE:**\n"
    "- Convert to HTML table format (<table>, <tr>, <td>, <th>)\n"
    "- Use 'rowspan' and 'colspan' attributes for merged cells\n"
    "- Preserve all cell content exactly as shown\n\n"
    "**If the image contains TEXT (non-table):**\n"
    "- Extract all text exactly as shown\n"
    "- Keep layout, hierarchy, and structure\n\n"
    "**If the image contains DATA (charts, graphs, diagrams):**\n"
    "- Extract the data and its meaning\n"
    "- Describe trends, relationships, or key insights\n\n"
    "**If the image is decorative or has no semantic meaning:**\n"
    "- Simply state what it is in one short sentence\n\n"
    "**Rules:**\n"
    "- Output in Korean (except HTML tags)\n"
    "- Tables MUST use HTML format with proper rowspan/colspan\n"
    "- Be concise - only include what is semantically meaningful\n"
    "- No filler words or unnecessary descriptions"
)

SIMPLE_OCR_PROMPT: str = "Describe the contents of this image."


# ── Abstract Base ─────────────────────────────────────────────────────────

class BaseOCREngine(ABC):
    """
    Abstract base class for OCR engine implementations.

    Subclasses MUST implement:
        - provider (property) — engine name string
        - build_message_content() — format LLM message payload

    The engine does NOT own the LLM client lifecycle; it receives
    the client at construction time.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            llm_client: LangChain-compatible LLM client with vision support.
            prompt: Custom OCR prompt. Defaults to DEFAULT_OCR_PROMPT.
        """
        self._llm_client = llm_client
        self._prompt = prompt or DEFAULT_OCR_PROMPT

    # ── Abstract interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return engine provider name (e.g., 'openai', 'anthropic')."""
        ...

    @abstractmethod
    def build_message_content(
        self,
        b64_image: str,
        mime_type: str,
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Build the message content payload for the LLM.

        Each provider formats the image+prompt differently. This method
        returns the ``content`` list suitable for a LangChain HumanMessage.

        Args:
            b64_image: Base64-encoded image data.
            mime_type: Image MIME type (e.g., 'image/png').
            prompt: The OCR prompt text to use.

        Returns:
            Content list for LangChain HumanMessage.
        """
        ...

    # ── Concrete interface ────────────────────────────────────────────────

    @property
    def prompt(self) -> str:
        """Current OCR prompt."""
        return self._prompt

    @prompt.setter
    def prompt(self, value: str) -> None:
        self._prompt = value

    @property
    def llm_client(self) -> Any:
        """LLM client instance."""
        return self._llm_client

    def convert_image_to_text(self, image_path: str) -> Optional[str]:
        """
        Convert a single image file to text using the LLM.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Extracted text wrapped in [Figure:...] format, or error string.
        """
        try:
            b64 = encode_image_base64(image_path)
            mime = get_mime_type(image_path)

            content = self.build_message_content(b64, mime, self._prompt)

            # Import lazily to avoid hard dependency
            from langchain_core.messages import HumanMessage

            message = HumanMessage(content=content)
            response = self._llm_client.invoke([message])
            result = response.content.strip()

            logger.info(f"[{self.provider.upper()}] OCR completed: {os.path.basename(image_path)}")
            return f"[Figure:{result}]"

        except Exception as e:
            logger.error(f"[{self.provider.upper()}] OCR failed: {image_path} — {e}")
            return f"[Image conversion error: {e!s}]"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.provider}')"


__all__ = [
    "BaseOCREngine",
    "DEFAULT_OCR_PROMPT",
    "SIMPLE_OCR_PROMPT",
    "get_mime_type",
    "encode_image_base64",
]
