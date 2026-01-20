# libs/ocr/base.py
# Abstract base class for OCR models
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Pattern

logger = logging.getLogger("ocr-base")


class BaseOCR(ABC):
    """
    Abstract base class for OCR processing.

    All OCR model implementations must inherit from this class.
    """

    # Default prompt (can be overridden in subclasses)
    DEFAULT_PROMPT = (
        "Extract meaningful information from this image.\n\n"
        "**If the image contains a TABLE:**\n"
        "- Convert to HTML table format (<table>, <tr>, <td>, <th>)\n"
        "- Use 'rowspan' and 'colspan' attributes for merged cells\n"
        "- Preserve all cell content exactly as shown\n"
        "- Example:\n"
        "  <table>\n"
        "    <tr><th colspan=\"2\">Header</th></tr>\n"
        "    <tr><td rowspan=\"2\">Merged</td><td>A</td></tr>\n"
        "    <tr><td>B</td></tr>\n"
        "  </table>\n\n"
        "**If the image contains TEXT (non-table):**\n"
        "- Extract all text exactly as shown\n"
        "- Keep layout, hierarchy, and structure\n\n"
        "**If the image contains DATA (charts, graphs, diagrams):**\n"
        "- Extract the data and its meaning\n"
        "- Describe trends, relationships, or key insights\n\n"
        "**If the image is decorative or has no semantic meaning:**\n"
        "- Simply state what it is in one short sentence\n"
        "- Example: 'A decorative geometric shape' or 'Company logo'\n"
        "- Do NOT over-analyze decorative elements\n\n"
        "**Rules:**\n"
        "- Output in Korean (except HTML tags)\n"
        "- Tables MUST use HTML format with proper rowspan/colspan\n"
        "- Be concise - only include what is semantically meaningful\n"
        "- No filler words or unnecessary descriptions"
    )

    # Simple prompt (used for vllm, etc.)
    SIMPLE_PROMPT = "Describe the contents of this image."

    def __init__(self, llm_client: Any, prompt: Optional[str] = None):
        """
        Initialize OCR model.

        Args:
            llm_client: LangChain LLM client (must support Vision models)
            prompt: Custom prompt (uses default prompt if None)
        """
        self.llm_client = llm_client
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        self._image_pattern: Optional[Pattern[str]] = None

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return OCR provider name (e.g., 'openai', 'anthropic')"""
        pass

    @abstractmethod
    def build_message_content(self, b64_image: str, mime_type: str) -> list:
        """
        Build message content for LLM.

        Args:
            b64_image: Base64 encoded image
            mime_type: Image MIME type

        Returns:
            Content list for LangChain HumanMessage
        """
        pass

    def convert_image_to_text(self, image_path: str) -> Optional[str]:
        """
        Convert image to text.

        Args:
            image_path: Local image file path

        Returns:
            Extracted text from image or None (on failure)
        """
        from libs.ocr.ocr_processor import (
            _b64_from_file,
            _get_mime_type,
        )
        from langchain_core.messages import HumanMessage

        try:
            b64_image = _b64_from_file(image_path)
            mime_type = _get_mime_type(image_path)

            content = self.build_message_content(b64_image, mime_type)
            message = HumanMessage(content=content)

            response = self.llm_client.invoke([message])
            result = response.content.strip()

            # Wrap result in [Figure:...] format
            result = f"[Figure:{result}]"

            logger.info(f"[{self.provider.upper()}] Image to text conversion completed")
            return result

        except Exception as e:
            logger.error(f"[{self.provider.upper()}] Image to text conversion failed: {e}")
            return f"[Image conversion error: {str(e)}]"

    def set_image_pattern(self, pattern: Optional[Pattern[str]] = None) -> None:
        """
        Set custom image pattern for tag detection.

        Args:
            pattern: Compiled regex pattern with capture group for image path.
                     If None, uses default [Image:{path}] pattern.

        Examples:
            >>> import re
            >>> ocr.set_image_pattern(re.compile(r"<img src='([^']+)'/>"))
        """
        self._image_pattern = pattern

    def set_image_pattern_from_string(self, pattern_string: str) -> None:
        """
        Set custom image pattern from pattern string.

        Args:
            pattern_string: Regex pattern string with capture group for image path.

        Examples:
            >>> ocr.set_image_pattern_from_string(r"<img src='([^']+)'/>")
        """
        self._image_pattern = re.compile(pattern_string)

    def process_text(self, text: str, image_pattern: Optional[Pattern[str]] = None) -> str:
        """
        Detect image tags in text and replace with OCR results.

        Args:
            text: Text containing image tags
            image_pattern: Custom regex pattern for image tags.
                           If None, uses instance pattern or default [Image:{path}] pattern.

        Returns:
            Text with image tags replaced by OCR results
        """
        from libs.ocr.ocr_processor import (
            extract_image_tags,
            load_image_from_path,
            DEFAULT_IMAGE_TAG_PATTERN,
        )

        if not self.llm_client:
            logger.warning(f"[{self.provider.upper()}] Skipping OCR processing: no LLM client")
            return text

        # Determine which pattern to use: parameter > instance > default
        pattern = image_pattern or self._image_pattern or DEFAULT_IMAGE_TAG_PATTERN

        image_paths = extract_image_tags(text, pattern)

        if not image_paths:
            logger.debug(f"[{self.provider.upper()}] No image tags found in text")
            return text

        logger.info(f"[{self.provider.upper()}] Detected {len(image_paths)} image tags")

        result_text = text

        for img_path in image_paths:
            # Build replacement pattern using the same pattern structure
            # Escape the path and create a pattern that matches the full tag
            escaped_path = re.escape(img_path)
            # Get the pattern string and replace capture group with escaped path
            pattern_str = pattern.pattern
            # Replace the capture group (.*), ([^...]+), etc. with the escaped path
            tag_pattern_str = re.sub(r'\([^)]+\)', escaped_path, pattern_str, count=1)
            tag_pattern = re.compile(tag_pattern_str)

            local_path = load_image_from_path(img_path)

            if local_path is None:
                logger.warning(f"[{self.provider.upper()}] Image load failed, keeping original tag: {img_path}")
                continue

            ocr_result = self.convert_image_to_text(local_path)

            if ocr_result is None or ocr_result.startswith("[Image conversion error:"):
                logger.warning(f"[{self.provider.upper()}] Image conversion failed, keeping original tag: {img_path}")
                continue

            result_text = tag_pattern.sub(ocr_result, result_text)
            logger.info(f"[{self.provider.upper()}] Tag replacement completed: {img_path[:50]}...")

        return result_text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider='{self.provider}')"
