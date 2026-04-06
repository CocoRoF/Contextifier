# contextifier/cached_processor.py
"""
CachedDocumentProcessor — Content-hash caching layer.

Wraps :class:`DocumentProcessor` and caches extraction results keyed
by ``(file_content_hash, config_hash)`` so that identical files
processed with the same configuration are never parsed twice.

Backends:
- In-memory ``dict`` (default) — fast, process-lifetime only.
- Disk-backed (JSON files in a cache directory) — survives restarts.
- Pluggable via ``cache_backend`` constructor argument.

Usage::

    from contextifier.cached_processor import CachedDocumentProcessor

    proc = CachedDocumentProcessor()          # in-memory cache
    text1 = proc.extract_text("report.pdf")   # parsed
    text2 = proc.extract_text("report.pdf")   # cache hit ─ instant
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from contextifier.config import ProcessingConfig
from contextifier.document_processor import DocumentProcessor, ChunkResult
from contextifier.types import (
    ChartData,
    ChartSeries,
    Chunk,
    ChunkMetadata,
    DocumentMetadata,
    ExtractionResult,
    TableCell,
    TableData,
)

logger = logging.getLogger("contextifier.cache")


# ── Cache backend protocol ────────────────────────────────────────────────

class CacheBackend(Protocol):
    """Minimal interface for a cache backend."""

    def get(self, key: str) -> Optional[str]:
        """Return cached text or ``None``."""
        ...

    def put(self, key: str, value: str) -> None:
        """Store *value* under *key*."""
        ...


# ── Built-in backends ────────────────────────────────────────────────────

class MemoryCacheBackend:
    """
    In-memory LRU cache using OrderedDict.

    On each ``get()`` hit, the entry is moved to the end (most recent).
    When the cache is full, the **least recently used** entry (front)
    is evicted on ``put()``.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._store: collections.OrderedDict[str, str] = collections.OrderedDict()

    def get(self, key: str) -> Optional[str]:
        value = self._store.get(key)
        if value is not None:
            # Move to end (most recently used)
            self._store.move_to_end(key)
        return value

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
        else:
            if len(self._store) >= self._max_size:
                # Evict least recently used (first item)
                self._store.popitem(last=False)
            self._store[key] = value


class DiskCacheBackend:
    """JSON-file-per-entry disk cache."""

    def __init__(self, cache_dir: Union[str, Path] = ".contextifier_cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        p = self._path(key)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return data.get("text")
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def put(self, key: str, value: str) -> None:
        p = self._path(key)
        try:
            p.write_text(
                json.dumps({"text": value}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Cache write failed for %s: %s", key, exc)


# ── Cached processor ─────────────────────────────────────────────────────

class CachedDocumentProcessor:
    """Document processor with transparent result caching."""

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        *,
        ocr_engine: Optional[Any] = None,
        cache_backend: Optional[CacheBackend] = None,
    ) -> None:
        self._sync = DocumentProcessor(config=config, ocr_engine=ocr_engine)
        self._cache: CacheBackend = cache_backend or MemoryCacheBackend()
        self._config_hash = self._hash_config(self._sync.config)

    def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs: Any,
    ) -> str:
        """Extract text with cache lookup."""
        cache_key = self._make_key(str(file_path), extract_metadata, ocr_processing)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s", file_path)
            return cached

        text = self._sync.extract_text(
            file_path,
            file_extension,
            extract_metadata=extract_metadata,
            ocr_processing=ocr_processing,
            **kwargs,
        )
        self._cache.put(cache_key, text)
        return text

    def process(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Process a file with cache lookup for the full ExtractionResult."""
        cache_key = self._make_key(
            str(file_path), extract_metadata, ocr_processing, suffix="process",
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit (process) for %s", file_path)
            return _deserialize_extraction_result(cached)

        result = self._sync.process(
            file_path,
            file_extension,
            extract_metadata=extract_metadata,
            ocr_processing=ocr_processing,
            **kwargs,
        )
        self._cache.put(cache_key, _serialize_extraction_result(result))
        return result

    def extract_chunks(
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
        """Extract and chunk text with cache lookup."""
        cache_key = self._make_key(
            str(file_path), extract_metadata, ocr_processing,
            suffix=f"chunks|cs={chunk_size}|co={chunk_overlap}"
                   f"|pt={preserve_tables}|pm={include_position_metadata}",
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit (chunks) for %s", file_path)
            return _deserialize_chunk_result(cached)

        result = self._sync.extract_chunks(
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
        self._cache.put(cache_key, _serialize_chunk_result(result))
        return result

    # Delegate non-cached methods directly
    def is_supported(self, extension: str) -> bool:
        return self._sync.is_supported(extension)

    @property
    def supported_extensions(self) -> frozenset:
        return self._sync.supported_extensions

    @property
    def config(self) -> ProcessingConfig:
        return self._sync.config

    # ── Private ───────────────────────────────────────────────────────────

    def _make_key(
        self,
        file_path: str,
        extract_metadata: bool,
        ocr: bool,
        suffix: str = "text",
    ) -> str:
        """Compute cache key from file content hash + config hash + options."""
        path = Path(file_path)
        if not path.is_file():
            return ""  # will miss — extraction will raise its own error

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        parts = (
            f"{content_hash}|{self._config_hash}"
            f"|meta={extract_metadata}|ocr={ocr}|{suffix}"
        )
        return hashlib.sha256(parts.encode()).hexdigest()

    @staticmethod
    def _hash_config(config: ProcessingConfig) -> str:
        config_str = str(config.to_dict())
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        return f"Cached{self._sync!r}"


__all__ = [
    "CachedDocumentProcessor",
    "CacheBackend",
    "MemoryCacheBackend",
    "DiskCacheBackend",
]


# ── Serialization helpers ─────────────────────────────────────────────────

def _serialize_extraction_result(result: ExtractionResult) -> str:
    """Serialize an ExtractionResult to a JSON string."""
    data: Dict[str, Any] = {
        "text": result.text,
        "metadata": result.metadata.to_dict() if result.metadata else None,
        "images": result.images,
        "page_count": result.page_count,
        "warnings": result.warnings,
        "tables": [_table_to_dict(t) for t in result.tables],
        "charts": [_chart_to_dict(c) for c in result.charts],
    }
    return json.dumps(data, ensure_ascii=False)


def _deserialize_extraction_result(s: str) -> ExtractionResult:
    """Deserialize an ExtractionResult from a JSON string."""
    data = json.loads(s)
    return ExtractionResult(
        text=data.get("text", ""),
        metadata=DocumentMetadata.from_dict(data["metadata"]) if data.get("metadata") else None,
        images=data.get("images", []),
        page_count=data.get("page_count", 0),
        warnings=data.get("warnings", []),
        tables=[_dict_to_table(t) for t in data.get("tables", [])],
        charts=[_dict_to_chart(c) for c in data.get("charts", [])],
    )


def _serialize_chunk_result(result: ChunkResult) -> str:
    """Serialize a ChunkResult to a JSON string."""
    chunks_meta = None
    if result.chunks_with_metadata:
        chunks_meta = [
            {
                "text": c.text,
                "metadata": {
                    "chunk_index": c.metadata.chunk_index,
                    "page_number": c.metadata.page_number,
                    "line_start": c.metadata.line_start,
                    "line_end": c.metadata.line_end,
                    "global_start": c.metadata.global_start,
                    "global_end": c.metadata.global_end,
                } if c.metadata else None,
            }
            for c in result.chunks_with_metadata
        ]
    data = {
        "chunks": result.chunks,
        "chunks_with_metadata": chunks_meta,
        "source_file": result.source_file,
    }
    return json.dumps(data, ensure_ascii=False)


def _deserialize_chunk_result(s: str) -> ChunkResult:
    """Deserialize a ChunkResult from a JSON string."""
    data = json.loads(s)
    chunks_meta = None
    if data.get("chunks_with_metadata"):
        chunks_meta = []
        for item in data["chunks_with_metadata"]:
            meta = None
            if item.get("metadata"):
                m = item["metadata"]
                meta = ChunkMetadata(
                    chunk_index=m.get("chunk_index", 0),
                    page_number=m.get("page_number"),
                    line_start=m.get("line_start", 0),
                    line_end=m.get("line_end", 0),
                    global_start=m.get("global_start", 0),
                    global_end=m.get("global_end", 0),
                )
            chunks_meta.append(Chunk(text=item["text"], metadata=meta))
    return ChunkResult(
        chunks=data.get("chunks", []),
        chunks_with_metadata=chunks_meta,
        source_file=data.get("source_file"),
    )


def _table_to_dict(table: TableData) -> Dict[str, Any]:
    rows = []
    for row in table.rows:
        rows.append([
            {
                "content": c.content,
                "row_span": c.row_span,
                "col_span": c.col_span,
                "is_header": c.is_header,
                "row_index": c.row_index,
                "col_index": c.col_index,
            }
            for c in row
        ])
    return {
        "rows": rows,
        "num_rows": table.num_rows,
        "num_cols": table.num_cols,
        "has_header": table.has_header,
        "caption": table.caption,
    }


def _dict_to_table(d: Dict[str, Any]) -> TableData:
    rows = []
    for row_data in d.get("rows", []):
        rows.append([
            TableCell(
                content=c["content"],
                row_span=c.get("row_span", 1),
                col_span=c.get("col_span", 1),
                is_header=c.get("is_header", False),
                row_index=c.get("row_index", 0),
                col_index=c.get("col_index", 0),
            )
            for c in row_data
        ])
    return TableData(
        rows=rows,
        num_rows=d.get("num_rows", 0),
        num_cols=d.get("num_cols", 0),
        has_header=d.get("has_header", False),
        caption=d.get("caption"),
    )


def _chart_to_dict(chart: ChartData) -> Dict[str, Any]:
    return {
        "chart_type": chart.chart_type,
        "title": chart.title,
        "categories": chart.categories,
        "series": [
            {"name": s.name, "values": s.values}
            for s in chart.series
        ],
    }


def _dict_to_chart(d: Dict[str, Any]) -> ChartData:
    return ChartData(
        chart_type=d.get("chart_type"),
        title=d.get("title"),
        categories=d.get("categories", []),
        series=[
            ChartSeries(name=s.get("name"), values=s.get("values", []))
            for s in d.get("series", [])
        ],
    )
