# libs/ocr/ocr_processor.py
# Module for loading image files and processing OCR.
import re
import base64
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Pattern

logger = logging.getLogger("ocr-processor")

# Default image tag pattern: [Image:{path}] or [image:{path}] (case-insensitive)
DEFAULT_IMAGE_TAG_PATTERN = re.compile(r'\[[Ii]mage:([^\]]+)\]')

# Keep for backward compatibility
IMAGE_TAG_PATTERN = DEFAULT_IMAGE_TAG_PATTERN


def _b64_from_file(path: str) -> str:
    """Encode file to Base64"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime_type(file_path: str) -> str:
    """Return MIME type based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
    }
    return mime_map.get(ext, "image/jpeg")


def extract_image_tags(
    text: str,
    pattern: Optional[Pattern[str]] = None
) -> List[str]:
    """
    Extract image tags from text.

    Args:
        text: Text containing image tags
        pattern: Custom regex pattern with capture group for path.
                 If None, uses default [Image:{path}] pattern.

    Returns:
        List of extracted image_path values
    """
    if pattern is None:
        pattern = DEFAULT_IMAGE_TAG_PATTERN
    matches = pattern.findall(text)
    return matches


def load_image_from_path(image_path: str) -> Optional[str]:
    """
    Validate and return local image file path.

    Args:
        image_path: Image file path

    Returns:
        Valid local file path or None
    """
    try:
        # Convert to absolute path
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)

        # Check file existence
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            logger.info(f"[OCR] Image loaded: {image_path}")
            return image_path

        logger.warning(f"[OCR] Image file not found: {image_path}")
        return None

    except Exception as e:
        logger.error(f"[OCR] Image load failed: {image_path}, error: {e}")
        return None


def convert_image_to_text_with_llm(
    image_path: str,
    llm_client: Any,
    provider: str
) -> str:
    """
    Convert image to text using VL model.

    Args:
        image_path: Local image file path
        llm_client: LangChain LLM client
        provider: LLM provider (openai, anthropic, gemini, vllm)

    Returns:
        Text extracted from image
    """
    try:
        from langchain_core.messages import HumanMessage

        b64_image = _b64_from_file(image_path)
        mime_type = _get_mime_type(image_path)

        # vllm uses simple prompt
        if provider == "vllm":
            prompt = "Describe the contents of this image."
        else:
            prompt = (
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

        # Build message by provider
        if provider in ("openai", "vllm"):
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                }
            ]
            message = HumanMessage(content=content)

        elif provider == "anthropic":
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_image
                    }
                },
                {"type": "text", "text": prompt}
            ]
            message = HumanMessage(content=content)

        elif provider == "gemini":
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                }
            ]
            message = HumanMessage(content=content)

        elif provider == "aws_bedrock":
            # AWS Bedrock (Claude via Bedrock) - Uses same format as Anthropic
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_image
                    }
                },
                {"type": "text", "text": prompt}
            ]
            message = HumanMessage(content=content)

        else:
            return None  # Unsupported provider

        response = llm_client.invoke([message])
        result = response.content.strip()

        # Wrap result in [Figure:...] format
        result = f"[Figure:{result}]"

        logger.info(f"[OCR] Image to text conversion completed: {os.path.basename(image_path)}")
        return result

    except Exception as e:
        logger.error(f"[OCR] Image to text conversion failed: {image_path}, error: {e}")
        return f"[Image conversion error: {str(e)}]"


def process_text_with_ocr(
    text: str,
    llm_client: Any,
    provider: str
) -> str:
    """
    Detect image tags in text and replace with OCR results.

    Args:
        text: Text containing [Image:{path}] tags
        llm_client: LangChain LLM client
        provider: LLM provider

    Returns:
        Text with image tags replaced by OCR results
    """
    if not llm_client:
        logger.warning("[OCR] Skipping OCR processing: no LLM client")
        return text

    # Extract image tags
    image_paths = extract_image_tags(text)

    if not image_paths:
        logger.debug("[OCR] No image tags found in text")
        return text

    logger.info(f"[OCR] Detected {len(image_paths)} image tags")

    result_text = text

    for img_path in image_paths:
        # Case-insensitive tag matching
        tag_pattern = re.compile(r'\[[Ii]mage:' + re.escape(img_path) + r'\]')

        # Load image from local path
        local_path = load_image_from_path(img_path)

        if local_path is None:
            # Keep original tag on load failure
            logger.warning(f"[OCR] Image load failed, keeping original tag: {img_path}")
            continue

        # Convert image to text using VL model
        ocr_result = convert_image_to_text_with_llm(
            image_path=local_path,
            llm_client=llm_client,
            provider=provider
        )

        # Keep original tag on OCR failure (None or error message)
        if ocr_result is None or ocr_result.startswith("[Image conversion error:"):
            logger.warning(f"[OCR] Image conversion failed, keeping original tag: {img_path}")
            continue

        # Replace tag with OCR result
        result_text = tag_pattern.sub(ocr_result, result_text)
        logger.info(f"[OCR] Tag replacement completed: {img_path[:50]}...")

    return result_text


def process_text_with_ocr_progress(
    text: str,
    llm_client: Any,
    provider: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
) -> str:
    """
    Detect image tags in text and replace with OCR results (with progress callback support).

    Args:
        text: Text containing [Image:{path}] tags
        llm_client: LangChain LLM client
        provider: LLM provider
        progress_callback: Progress callback function

    Returns:
        Text with image tags replaced by OCR results
    """
    if not llm_client:
        logger.warning("[OCR] Skipping OCR processing: no LLM client")
        return text

    # Extract image tags
    image_paths = extract_image_tags(text)

    if not image_paths:
        logger.debug("[OCR] No image tags found in text")
        return text

    total_chunks = len(image_paths)
    logger.info(f"[OCR] Detected {total_chunks} image tags")

    result_text = text
    success_count = 0
    failed_count = 0

    for idx, img_path in enumerate(image_paths):
        # Progress callback - processing started
        if progress_callback:
            progress_callback({
                'event': 'ocr_tag_processing',
                'chunk_index': idx,
                'total_chunks': total_chunks,
                'image_path': img_path
            })

        # Case-insensitive tag matching
        tag_pattern = re.compile(r'\[[Ii]mage:' + re.escape(img_path) + r'\]')

        # Load image from local path
        local_path = load_image_from_path(img_path)

        if local_path is None:
            # Keep original tag on load failure
            logger.warning(f"[OCR] Image load failed, keeping original tag: {img_path}")
            failed_count += 1
            if progress_callback:
                progress_callback({
                    'event': 'ocr_chunk_processed',
                    'chunk_index': idx,
                    'total_chunks': total_chunks,
                    'status': 'failed',
                    'error': f'Load failed: {img_path}'
                })
            continue

        try:
            # Convert image to text using VL model
            ocr_result = convert_image_to_text_with_llm(
                image_path=local_path,
                llm_client=llm_client,
                provider=provider
            )

            # Keep original tag on OCR failure (None or error message)
            if ocr_result is None or ocr_result.startswith("[Image conversion error:"):
                logger.warning(f"[OCR] Image conversion failed, keeping original tag: {img_path}")
                failed_count += 1
                if progress_callback:
                    progress_callback({
                        'event': 'ocr_chunk_processed',
                        'chunk_index': idx,
                        'total_chunks': total_chunks,
                        'status': 'failed',
                        'error': ocr_result or 'OCR returned None'
                    })
                continue

            # Replace tag with OCR result
            result_text = tag_pattern.sub(ocr_result, result_text)
            success_count += 1
            logger.info(f"[OCR] Tag replacement completed: {img_path[:50]}...")

            if progress_callback:
                progress_callback({
                    'event': 'ocr_chunk_processed',
                    'chunk_index': idx,
                    'total_chunks': total_chunks,
                    'status': 'success'
                })

        except Exception as e:
            logger.error(f"[OCR] Image processing error: {img_path}, error: {e}")
            failed_count += 1
            if progress_callback:
                progress_callback({
                    'event': 'ocr_chunk_processed',
                    'chunk_index': idx,
                    'total_chunks': total_chunks,
                    'status': 'failed',
                    'error': str(e)
                })

    return result_text
