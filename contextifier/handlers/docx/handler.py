# contextifier/handlers/docx/handler.py
"""
DOCXHandler — Handler for Microsoft Word DOCX documents (.docx ONLY).

Pipeline:
    Convert:  Raw bytes → python-docx Document (ZIP + OOXML validation)
    Preprocess: Wrap Document, compute stats, pre-extract charts from ZIP
    Metadata: core_properties → DocumentMetadata (title, author, dates)
    Content:  Body traversal → paragraphs, tables (vMerge/gridSpan),
              images (relationship-based), charts (OOXML DrawingML),
              diagrams, page breaks
    Postprocess: Assemble with page tags and metadata block

Old issues resolved:
- Chart formatting duplicated — now uses ChartService
- Image processor created without standard config args — fixed
- Dual metadata approach eliminated
"""

from __future__ import annotations

from typing import Any, FrozenSet

from contextifier.handlers.base import BaseHandler
from contextifier.handlers.docx._constants import NAMESPACES
from contextifier.types import FileContext
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from contextifier.handlers.docx.converter import DocxConverter
from contextifier.handlers.docx.preprocessor import DocxPreprocessor
from contextifier.handlers.docx.metadata_extractor import DocxMetadataExtractor
from contextifier.handlers.docx.content_extractor import DocxContentExtractor


class DOCXHandler(BaseHandler):
    """Handler for DOCX files (.docx)."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        return frozenset({"docx"})

    @property
    def handler_name(self) -> str:
        return "DOCX Handler"

    # Parts of the package that hold readable text. Headers and footers are
    # included because a document number or a name is as likely to sit there
    # as in the body.
    _TEXT_PARTS = (
        "word/document.xml",
        "word/footnotes.xml",
        "word/endnotes.xml",
        "word/comments.xml",
    )
    _TEXT_PART_PREFIXES = ("word/header", "word/footer")

    def extract_text_fast(self, file_context: FileContext, **kwargs: Any) -> str:
        """
        Every text node in the package, read straight from the XML.

        python-docx would be the obvious tool and is the wrong one here: its
        object model rebuilds rows and cells on access, which costs more than
        the full pipeline's raw traversal — measured at roughly half the speed
        on a real document. Walking ``w:t`` nodes skips the model entirely and
        picks up body, tables, text boxes, headers, footers and notes without
        having to know which is which.

        Text inside a shape appears twice, once per compatibility branch. That
        is deliberate: resolving the branches costs a second walk, and a scan
        asks whether a term is present, not how often.
        """
        import io
        import zipfile

        from lxml import etree

        try:
            archive = zipfile.ZipFile(io.BytesIO(file_context.get("file_data", b"")))
        except Exception as exc:
            self._logger.debug("Fast DOCX path unavailable (%s); using pipeline", exc)
            return super().extract_text_fast(file_context, **kwargs)

        qn_p = f"{{{NAMESPACES['w']}}}p"
        qn_t = f"{{{NAMESPACES['w']}}}t"
        lines: list[str] = []

        with archive:
            names = [
                name
                for name in archive.namelist()
                if name in self._TEXT_PARTS or name.startswith(self._TEXT_PART_PREFIXES)
            ]
            for name in names:
                try:
                    root = etree.fromstring(archive.read(name))
                except Exception as exc:
                    self._logger.debug("Skipping unreadable part %s: %s", name, exc)
                    continue
                # Join per paragraph, not per node: Word splits a phrase across
                # runs at every formatting change, so one line per `w:t` would
                # cut "경희대학교 경영연구원" into pieces and a search for it
                # would find nothing.
                for paragraph in root.iter(qn_p):
                    text = "".join(
                        node.text or "" for node in paragraph.iter(qn_t)
                    ).strip()
                    if text:
                        lines.append(text)

        return "\n".join(lines)

    def create_converter(self) -> BaseConverter:
        return DocxConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return DocxPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return DocxMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return DocxContentExtractor(
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


__all__ = ["DOCXHandler"]
