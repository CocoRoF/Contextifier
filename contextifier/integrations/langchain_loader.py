# contextifier/integrations/langchain_loader.py
"""
LangChain Document Loader integration for Contextifier.

Provides a standard LangChain ``BaseLoader`` that wraps
``DocumentProcessor`` to produce ``Document`` objects with
rich metadata.

Usage::

    from contextifier.integrations.langchain_loader import ContextifierLoader

    loader = ContextifierLoader("report.pdf")
    docs = loader.load()

    # With config
    from contextifier.config import ProcessingConfig
    config = ProcessingConfig().with_chunking(chunk_size=2000)
    loader = ContextifierLoader("report.pdf", config=config, chunk=True)
    docs = loader.load()

    # With OCR
    from contextifier.ocr.engines import OpenAIOCREngine
    ocr = OpenAIOCREngine.from_api_key("sk-...")
    loader = ContextifierLoader("scan.pdf", ocr_engine=ocr, ocr_processing=True)
    docs = loader.load()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from contextifier.config import ProcessingConfig
from contextifier.document_processor import DocumentProcessor


class ContextifierLoader(BaseLoader):
    """LangChain document loader backed by Contextifier.

    Supports two modes:

    * **Single document** (``chunk=False``, default): returns one
      ``Document`` whose ``page_content`` is the full extracted text.
    * **Chunked** (``chunk=True``): returns multiple ``Document`` objects,
      one per chunk, with position metadata.

    Args:
        file_path: Path to the document file.
        config: Optional ``ProcessingConfig``.
        ocr_engine: Optional OCR engine instance.
        ocr_processing: Whether to apply OCR post-processing.
        chunk: If ``True``, return chunked documents.
        chunk_size: Override chunk size (characters).
        chunk_overlap: Override chunk overlap (characters).
        extract_metadata: Whether to include document metadata in text.
        extra_metadata: Additional metadata dict merged into every
            returned ``Document.metadata``.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        config: Optional[ProcessingConfig] = None,
        ocr_engine: Optional[Any] = None,
        ocr_processing: bool = False,
        chunk: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        extract_metadata: bool = True,
        extra_metadata: Optional[dict] = None,
    ) -> None:
        self.file_path = str(file_path)
        self.config = config
        self.ocr_engine = ocr_engine
        self.ocr_processing = ocr_processing
        self.chunk = chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_metadata = extract_metadata
        self.extra_metadata = extra_metadata or {}

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the file.

        Yields:
            ``Document`` objects with extracted text and metadata.
        """
        processor = DocumentProcessor(
            config=self.config,
            ocr_engine=self.ocr_engine,
        )

        base_meta = {
            "source": self.file_path,
            "file_name": os.path.basename(self.file_path),
            "file_extension": os.path.splitext(self.file_path)[1].lstrip(".").lower(),
            **self.extra_metadata,
        }

        if self.chunk:
            result = processor.extract_chunks(
                self.file_path,
                extract_metadata=self.extract_metadata,
                ocr_processing=self.ocr_processing,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                include_position_metadata=True,
            )
            for i, chunk_text in enumerate(result.chunks):
                chunk_meta = {**base_meta, "chunk_index": i}
                if result.has_metadata and i < len(result.chunks_with_metadata):
                    cm = result.chunks_with_metadata[i].metadata
                    if cm and cm.page_number is not None:
                        chunk_meta["page_number"] = cm.page_number
                yield Document(page_content=chunk_text, metadata=chunk_meta)
        else:
            text = processor.extract_text(
                self.file_path,
                extract_metadata=self.extract_metadata,
                ocr_processing=self.ocr_processing,
            )
            yield Document(page_content=text, metadata=base_meta)
