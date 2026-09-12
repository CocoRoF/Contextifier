# tests/unit/ocr/test_engines.py
"""
Engine message shapes and response handling.

An OCR engine is thin — it decides how to say "read this image" to one
provider — but the two things it gets wrong are expensive: sending a message
shape the model does not answer, and mistaking a working answer for a failure.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from contextifier.ocr.base import normalize_response_content
from contextifier.ocr.engines import DeepSeekOCREngine, VLLMOCREngine


class Response:
    def __init__(self, content: Any) -> None:
        self.content = content


class RecordingClient:
    """Captures the message it was asked to send."""

    def __init__(self, content: Any = "recognised text") -> None:
        self.content = content
        self.messages: List[Any] = []

    def invoke(self, messages: List[Any]) -> Response:
        self.messages = messages
        return Response(self.content)


@pytest.fixture()
def image(tmp_path):
    path = tmp_path / "page.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    return str(path)


class TestNormalizeResponseContent:
    def test_plain_string(self) -> None:
        assert normalize_response_content("  text  ") == "text"

    def test_list_of_blocks(self) -> None:
        """OpenAI-compatible servers answer in blocks; vLLM does it for one part."""
        content = [{"type": "text", "text": "first "}, {"type": "text", "text": "second"}]
        assert normalize_response_content(content) == "first second"

    def test_list_of_strings(self) -> None:
        assert normalize_response_content(["a", "b"]) == "ab"

    def test_none_and_empty(self) -> None:
        assert normalize_response_content(None) == ""
        assert normalize_response_content([]) == ""


class TestDeepSeekMessageShape:
    def test_image_comes_before_the_instruction(self, image: str) -> None:
        """DeepSeek-OCR's chat template requires image-first; text-first returns nothing."""
        client = RecordingClient()
        DeepSeekOCREngine(client).convert_image_to_text(image)

        parts = client.messages[0].content
        assert parts[0]["type"] == "image_url"
        assert parts[1]["type"] == "text"

    def test_uses_the_models_own_prompt(self, image: str) -> None:
        client = RecordingClient()
        DeepSeekOCREngine(client).convert_image_to_text(image)

        assert client.messages[0].content[1]["text"] == "Free OCR."

    def test_generic_vllm_engine_keeps_text_first(self, image: str) -> None:
        client = RecordingClient()
        VLLMOCREngine(client).convert_image_to_text(image)

        assert client.messages[0].content[0]["type"] == "text"


class TestResponseHandling:
    def test_block_shaped_answer_is_accepted(self, image: str) -> None:
        client = RecordingClient([{"type": "text", "text": "table of results"}])
        out = DeepSeekOCREngine(client).convert_image_to_text(image)

        assert out == "[Figure:table of results]"

    def test_empty_answer_is_a_failure_not_an_empty_figure(self, image: str) -> None:
        """An empty placeholder would replace the tag and lose the image."""
        out = DeepSeekOCREngine(RecordingClient("   ")).convert_image_to_text(image)

        assert out.startswith("[Image conversion error:")
        assert out != "[Figure:]"
