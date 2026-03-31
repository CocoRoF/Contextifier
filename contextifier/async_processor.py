# contextifier/async_processor.py
"""
AsyncDocumentProcessor — Async wrapper around DocumentProcessor.

Provides ``async`` versions of all public extraction methods by
delegating to ``asyncio.to_thread()`` so that the synchronous
pipeline runs in a thread-pool without blocking the event loop.

Usage::

    import asyncio
    from contextifier.async_processor import AsyncDocumentProcessor

    async def main():
        processor = AsyncDocumentProcessor()
        text = await processor.extract_text("doc.pdf")
        chunks = await processor.extract_chunks("doc.pdf", chunk_size=1000)

    asyncio.run(main())

    # Batch processing with concurrency control
    async def batch():
        processor = AsyncDocumentProcessor()
        results = await processor.extract_batch(
            ["a.pdf", "b.docx", "c.xlsx"],
            max_concurrent=4,
        )

Design: thin ``async`` veneer — *all* real work is still synchronous
inside ``DocumentProcessor``.  This avoids duplicating business logic
while providing a first-class async API.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from contextifier.config import ProcessingConfig
from contextifier.document_processor import ChunkResult, DocumentProcessor
from contextifier.types import ExtractionResult


class AsyncDocumentProcessor:
    """Async façade over :class:`DocumentProcessor`."""

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        *,
        ocr_engine: Optional[Any] = None,
    ) -> None:
        self._sync = DocumentProcessor(config=config, ocr_engine=ocr_engine)

    # ── Text extraction ───────────────────────────────────────────────────

    async def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs: Any,
    ) -> str:
        """Async version of :meth:`DocumentProcessor.extract_text`."""
        return await asyncio.to_thread(
            self._sync.extract_text,
            file_path,
            file_extension,
            extract_metadata=extract_metadata,
            ocr_processing=ocr_processing,
            **kwargs,
        )

    async def process(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Async version of :meth:`DocumentProcessor.process`."""
        return await asyncio.to_thread(
            self._sync.process,
            file_path,
            file_extension,
            extract_metadata=extract_metadata,
            ocr_processing=ocr_processing,
            **kwargs,
        )

    # ── Chunking ──────────────────────────────────────────────────────────

    async def extract_chunks(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        preserve_tables: bool = True,
        include_position_metadata: bool = False,
        **kwargs: Any,
    ) -> ChunkResult:
        """Async version of :meth:`DocumentProcessor.extract_chunks`."""
        return await asyncio.to_thread(
            self._sync.extract_chunks,
            file_path,
            file_extension,
            extract_metadata=extract_metadata,
            ocr_processing=ocr_processing,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_tables=preserve_tables,
            include_position_metadata=include_position_metadata,
            **kwargs,
        )

    # ── Batch processing ──────────────────────────────────────────────────

    async def extract_batch(
        self,
        file_paths: List[Union[str, Path]],
        *,
        max_concurrent: int = 4,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Union[str, Exception]]:
        """Extract text from multiple files concurrently.

        Args:
            file_paths: List of file paths to process.
            max_concurrent: Maximum number of concurrent extractions.
            extract_metadata: Whether to include metadata.
            ocr_processing: Whether to apply OCR.
            **kwargs: Extra options passed to each extraction.

        Returns:
            Dict mapping each file path (str) to either the extracted
            text or the ``Exception`` that occurred during processing.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: Dict[str, Union[str, Exception]] = {}

        async def _process_one(fp: Union[str, Path]) -> None:
            key = str(fp)
            async with semaphore:
                try:
                    text = await self.extract_text(
                        fp,
                        extract_metadata=extract_metadata,
                        ocr_processing=ocr_processing,
                        **kwargs,
                    )
                    results[key] = text
                except Exception as exc:
                    results[key] = exc

        await asyncio.gather(*[_process_one(fp) for fp in file_paths])
        return results

    # ── Utility ───────────────────────────────────────────────────────────

    def is_supported(self, extension: str) -> bool:
        return self._sync.is_supported(extension)

    @property
    def supported_extensions(self) -> frozenset:
        return self._sync.supported_extensions

    @property
    def config(self) -> ProcessingConfig:
        return self._sync.config

    def __repr__(self) -> str:
        return f"Async{self._sync!r}"


__all__ = ["AsyncDocumentProcessor"]
