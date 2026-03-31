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

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

from contextifier.config import ProcessingConfig
from contextifier.document_processor import DocumentProcessor

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
    """Simple in-memory dict cache."""

    def __init__(self, max_size: int = 256) -> None:
        self._max_size = max_size
        self._store: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        if len(self._store) >= self._max_size:
            # evict oldest entry (FIFO)
            oldest = next(iter(self._store))
            del self._store[oldest]
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

    def _make_key(self, file_path: str, extract_metadata: bool, ocr: bool) -> str:
        """Compute cache key from file content hash + config hash + options."""
        path = Path(file_path)
        if not path.is_file():
            return ""  # will miss — extraction will raise its own error

        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        parts = f"{content_hash}|{self._config_hash}|meta={extract_metadata}|ocr={ocr}"
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
