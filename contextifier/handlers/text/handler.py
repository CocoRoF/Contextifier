# contextifier/handlers/text/handler.py
"""
TextHandler — Unified handler for plain text files.

Pipeline:
    Stage 1 (Convert):     Raw bytes → decoded string (encoding detection)
    Stage 2 (Preprocess):  BOM strip, line ending normalization → PreprocessedData
    Stage 3 (Metadata):    No embedded metadata → empty DocumentMetadata
    Stage 4 (Content):     Code/text mode cleaning → ExtractionResult
    Stage 5 (Postprocess): Standard whitespace normalization → final string

Supported file categories:
    text (.txt, .md, .rst), code (.py, .js, .java, .cpp, ...),
    config (.json, .yaml, .xml, .ini, ...), script (.sh, .bat, .ps1),
    log (.log), web (.html, .htm, .css, ...)

Code mode detection:
    Auto-detected from file_category (code/config/script/web → code mode).
    Can be overridden explicitly via ``is_code=True`` kwarg.

v1.0 Issues resolved:
- TextHandler skipped convert() entirely — now uses TextConverter
- TextFileConverter was created but never called — decode was inline
- extract_metadata flag was ignored — now respected via pipeline
- Text cleaning was inline in handler, not in a pipeline stage
- is_code was a handler parameter, now auto-detected from file_category
"""

from __future__ import annotations

from typing import FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import (
    BaseMetadataExtractor,
    NullMetadataExtractor,
)
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.text.converter import TextConverter
from contextifier.handlers.text.preprocessor import TextPreprocessor
from contextifier.handlers.text.content_extractor import TextContentExtractor


# All text-based extensions supported by this handler.
# Covers: plain text, markup, source code, config, scripts, stylesheets.
_TEXT_EXTENSIONS = frozenset({
    # Plain text & markup
    "txt", "md", "markdown", "rst", "log",
    # Config & data formats
    "cfg", "ini", "conf", "yaml", "yml", "toml", "json", "xml", "svg",
    "properties", "env",
    # Source code
    "py", "js", "ts", "jsx", "tsx", "java", "cpp", "c", "h", "hpp",
    "cs", "go", "rs", "php", "rb", "swift", "kt", "scala", "dart",
    "r", "m", "lua", "pl", "pm",
    # Scripts
    "sh", "bash", "zsh", "bat", "ps1", "cmd", "fish",
    # SQL
    "sql",
    # Web
    "html", "htm", "xhtml", "css", "scss", "less", "sass",
    # Vue / Svelte (single-file components)
    "vue", "svelte",
    # Dotfiles
    "gitignore", "dockerignore", "editorconfig",
})


class TextHandler(BaseHandler):
    """
    Handler for plain text and source code files.

    This is a CATEGORY handler — it supports many extensions that
    all share the same "read bytes, decode, clean" processing model.
    Unlike document format handlers (PDF, DOCX) that each get one
    extension, the TextHandler covers all text-based formats.
    """

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return _TEXT_EXTENSIONS

    @property
    def handler_name(self) -> str:
        return "Text Handler"

    def create_converter(self) -> BaseConverter:
        """
        Create TextConverter for encoding detection and decoding.

        Encoding list can be customized via config.format_options:
            config = ProcessingConfig(
                format_options={"text": {"encodings": ["shift_jis", "utf-8"]}}
            )
        """
        text_opts = self._config.format_options.get("text", {})
        encodings = text_opts.get("encodings", None)
        return TextConverter(encodings=encodings)

    def create_preprocessor(self) -> BasePreprocessor:
        """Create TextPreprocessor for BOM strip and line ending normalization."""
        return TextPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        """
        Text files have no embedded metadata.

        Returns NullMetadataExtractor which produces empty DocumentMetadata.
        """
        return NullMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        """
        Create TextContentExtractor for code/text mode cleaning.

        Code mode is auto-detected from file_category or can be
        overridden via ``is_code=True`` kwarg at extraction time.
        """
        return TextContentExtractor()

    def create_postprocessor(self) -> BasePostprocessor:
        """Create standard postprocessor for metadata block and whitespace cleanup."""
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["TextHandler"]
