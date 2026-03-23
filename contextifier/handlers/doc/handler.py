# contextifier/handlers/doc/handler.py
"""
DOCHandler — Handler for legacy Microsoft Word DOC documents (.doc ONLY).

CRITICAL: .doc is a notoriously polymorphic extension. Files with .doc
extension can actually be:

1. Genuine OLE2/CFBF DOC (binary Word format) — most common
2. RTF (Rich Text Format) saved as .doc
3. Misnamed DOCX (ZIP/OOXML) saved with .doc extension
4. HTML saved as .doc (Word's "Web Page" format)

This handler uses the DELEGATION pattern to detect the actual format
in _check_delegation() and route to the correct handler:

    .doc file arrives
        ├── Starts with PK (ZIP magic) → delegate to DOCX handler
        ├── Starts with {\\rtf          → delegate to RTF handler
        ├── Starts with <html / <!DOCTYPE → delegate to HTML handler (future)
        └── OLE2 signature (D0CF11E0)   → process as genuine DOC

Pipeline (for genuine OLE2 DOC):
    Convert:  Raw bytes → intermediate format (antiword / LibreOffice)
    Preprocess: Clean converted output
    Metadata: Extract available metadata from OLE compound document
    Content:  Text, tables (limited), images (limited)
    Postprocess: Assemble with page tags and metadata block

Old issues resolved:
- DOC handler delegated to DOCX handler unconditionally — now detects actual format
- Delegation is explicit and uses the registry (not ad-hoc handler references)
- Each format has its own self-contained handler
"""

from __future__ import annotations

from typing import Any, FrozenSet, Optional

from contextifier.handlers.base import BaseHandler
from contextifier.types import ExtractionResult, FileContext
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.doc.converter import DocConverter
from contextifier.handlers.doc.preprocessor import DocPreprocessor
from contextifier.handlers.doc.metadata_extractor import DocMetadataExtractor
from contextifier.handlers.doc.content_extractor import DocContentExtractor

# Magic bytes for format detection
_ZIP_MAGIC = b"PK"                              # ZIP/OOXML (DOCX)
_RTF_MAGIC = b"{\\rtf"                           # RTF
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE2/CFBF (genuine DOC)
_HTML_MARKERS = (b"<html", b"<!doctype", b"<HTML", b"<!DOCTYPE")


class DOCHandler(BaseHandler):
    """
    Handler for DOC files (.doc only).

    Uses format detection + delegation for polymorphic .doc files.
    Only processes genuine OLE2 DOC files through its own pipeline;
    RTF, DOCX, and HTML disguised as .doc are delegated to their
    respective handlers.
    """

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"doc"})

    @property
    def handler_name(self) -> str:
        return "DOC Handler"

    # ── Delegation: detect actual format ──────────────────────────────────

    def _check_delegation(
        self,
        file_context: FileContext,
        **kwargs: Any,
    ) -> Optional[ExtractionResult]:
        """
        Detect the actual format of a .doc file and delegate if needed.

        Checks magic bytes to determine if the file is:
        - ZIP/OOXML → delegate to 'docx' handler
        - RTF       → delegate to 'rtf' handler
        - HTML      → delegate to 'html' handler (future)
        - OLE2      → return None (process as genuine DOC)
        """
        data = file_context.get("file_data", b"")
        if not data or len(data) < 8:
            return None  # Too small to detect, try normal pipeline

        # Check for ZIP/OOXML (misnamed DOCX)
        if data[:2] == _ZIP_MAGIC:
            self._logger.info("DOC file is actually DOCX (ZIP magic detected)")
            return self._delegate_to(
                "docx", file_context,
                include_metadata=kwargs.get("include_metadata", True),
                **{k: v for k, v in kwargs.items() if k != "include_metadata"},
            )

        # Check for RTF
        if data[:5] == _RTF_MAGIC:
            self._logger.info("DOC file is actually RTF (RTF magic detected)")
            return self._delegate_to(
                "rtf", file_context,
                include_metadata=kwargs.get("include_metadata", True),
                **{k: v for k, v in kwargs.items() if k != "include_metadata"},
            )

        # Check for HTML
        header = data[:256].lstrip()
        if any(header.startswith(marker) for marker in _HTML_MARKERS):
            self._logger.info("DOC file is actually HTML (HTML markers detected)")
            # TODO: delegate to 'html' handler once implemented
            # For now, fall through to DOC pipeline
            return None

        # OLE2 or unknown — proceed with DOC pipeline
        return None

    # ── Pipeline factories ────────────────────────────────────────────────

    def create_converter(self) -> BaseConverter:
        return DocConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return DocPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return DocMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return DocContentExtractor(
            image_service=self._image_service,
            tag_service=self._tag_service,
            chart_service=self._chart_service,
            table_service=self._table_service,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )


__all__ = ["DOCHandler"]
