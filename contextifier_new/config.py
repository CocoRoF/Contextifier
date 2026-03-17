# contextifier_new/config.py
"""
Unified configuration system for Contextifier v2.

Single source of truth for all configurable behaviour.
Uses frozen dataclasses for immutability after creation,
with a builder pattern for convenience.

Design goals:
- ONE config object flows through the entire system
- Format-specific options are namespaced, not ad-hoc kwargs
- Sensible defaults for everything → zero-config usage works
- Config is serializable to/from dict for persistence

Configuration Hierarchy:
    ProcessingConfig (root)
    ├── TagConfig       — all tag prefix/suffix settings
    ├── ImageConfig     — image saving & naming settings
    ├── ChartConfig     — chart formatting settings
    ├── MetadataConfig  — metadata formatting settings
    ├── TableConfig     — table output format settings
    ├── ChunkingConfig  — chunking parameters
    └── OCRConfig       — OCR engine settings
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Optional

from contextifier_new.types import (
    NamingStrategy,
    OutputFormat,
    StorageType,
)


# ─── Tag Configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TagConfig:
    """
    Configuration for all structural tags in extracted text.

    Controls the prefix/suffix wrapping for page numbers, slide numbers,
    sheet names, image references, and chart blocks.
    """
    # Page tags: e.g., "[Page Number: 1]"
    page_prefix: str = "[Page Number: "
    page_suffix: str = "]"

    # Slide tags: e.g., "[Slide Number: 1]"
    slide_prefix: str = "[Slide Number: "
    slide_suffix: str = "]"

    # Sheet tags: e.g., "[Sheet: Sheet1]"
    sheet_prefix: str = "[Sheet: "
    sheet_suffix: str = "]"

    # Image tags: e.g., "[Image: path/to/image.png]"
    image_prefix: str = "[Image:"
    image_suffix: str = "]"

    # Chart tags: e.g., "[chart]...[/chart]"
    chart_prefix: str = "[chart]"
    chart_suffix: str = "[/chart]"

    # Metadata tags: e.g., "<Document-Metadata>...</Document-Metadata>"
    metadata_prefix: str = "<Document-Metadata>"
    metadata_suffix: str = "</Document-Metadata>"


# ─── Image Configuration ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImageConfig:
    """
    Configuration for image extraction and storage.
    """
    directory_path: str = "temp/images"
    naming_strategy: NamingStrategy = NamingStrategy.HASH
    default_format: str = "png"
    quality: int = 95
    skip_duplicate: bool = True
    storage_type: StorageType = StorageType.LOCAL


# ─── Chart Configuration ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChartConfig:
    """
    Configuration for chart data formatting.
    """
    use_html_table: bool = True
    include_chart_type: bool = True
    include_chart_title: bool = True


# ─── Metadata Configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class MetadataConfig:
    """
    Configuration for metadata formatting.
    """
    language: str = "ko"              # "ko" or "en"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    indent: str = "  "


# ─── Table Configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class TableConfig:
    """
    Configuration for table output formatting.
    """
    output_format: OutputFormat = OutputFormat.HTML
    clean_whitespace: bool = True
    preserve_merged_cells: bool = True


# ─── Chunking Configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkingConfig:
    """
    Configuration for text chunking.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_tables: bool = True
    include_position_metadata: bool = False
    strategy: str = "recursive"       # "recursive", "sliding", "hierarchical"


# ─── OCR Configuration ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class OCRConfig:
    """
    Configuration for OCR processing.
    """
    enabled: bool = False
    provider: Optional[str] = None    # "openai", "anthropic", "gemini", "vllm", "bedrock"
    prompt: Optional[str] = None      # Custom OCR prompt (None = use default)


# ─── Root Configuration ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingConfig:
    """
    Root configuration object for the entire processing pipeline.

    Immutable after creation. Use `with_*()` methods or `replace()`
    to create modified copies.

    Example:
        # Zero-config (all defaults)
        config = ProcessingConfig()

        # Customized
        config = ProcessingConfig(
            tags=TagConfig(page_prefix="<page>", page_suffix="</page>"),
            images=ImageConfig(directory_path="output/images"),
            chunking=ChunkingConfig(chunk_size=2000),
        )

        # Modify existing config
        config2 = config.with_tags(page_prefix="<!-- Page ", page_suffix=" -->")
    """
    tags: TagConfig = field(default_factory=TagConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    charts: ChartConfig = field(default_factory=ChartConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)

    # Format-specific overrides (handler implementations access these)
    format_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass doesn't allow assignment, but we need to
        # convert mutable default dict. Use object.__setattr__ for frozen.
        if not isinstance(self.format_options, dict):
            object.__setattr__(self, "format_options", dict(self.format_options))

    # ── Fluent modification methods ───────────────────────────────────────

    def with_tags(self, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with modified tag settings."""
        return replace(self, tags=replace(self.tags, **kwargs))

    def with_images(self, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with modified image settings."""
        return replace(self, images=replace(self.images, **kwargs))

    def with_charts(self, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with modified chart settings."""
        return replace(self, charts=replace(self.charts, **kwargs))

    def with_metadata(self, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with modified metadata settings."""
        return replace(self, metadata=replace(self.metadata, **kwargs))

    def with_tables(self, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with modified table settings."""
        return replace(self, tables=replace(self.tables, **kwargs))

    def with_chunking(self, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with modified chunking settings."""
        return replace(self, chunking=replace(self.chunking, **kwargs))

    def with_ocr(self, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with modified OCR settings."""
        return replace(self, ocr=replace(self.ocr, **kwargs))

    def with_format_option(self, format_name: str, **kwargs: Any) -> "ProcessingConfig":
        """Return new config with added format-specific options."""
        new_opts = dict(self.format_options)
        if format_name in new_opts:
            new_opts[format_name] = {**new_opts[format_name], **kwargs}
        else:
            new_opts[format_name] = kwargs
        return replace(self, format_options=new_opts)

    def get_format_option(self, format_name: str, key: str, default: Any = None) -> Any:
        """Get a format-specific option value."""
        return self.format_options.get(format_name, {}).get(key, default)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """Deserialize config from dictionary."""
        tags = TagConfig(**data.get("tags", {})) if "tags" in data else TagConfig()
        images_data = data.get("images", {})
        if "naming_strategy" in images_data and isinstance(images_data["naming_strategy"], str):
            images_data["naming_strategy"] = NamingStrategy(images_data["naming_strategy"])
        if "storage_type" in images_data and isinstance(images_data["storage_type"], str):
            images_data["storage_type"] = StorageType(images_data["storage_type"])
        images = ImageConfig(**images_data) if "images" in data else ImageConfig()
        charts = ChartConfig(**data.get("charts", {})) if "charts" in data else ChartConfig()
        meta = MetadataConfig(**data.get("metadata", {})) if "metadata" in data else MetadataConfig()
        tables_data = data.get("tables", {})
        if "output_format" in tables_data and isinstance(tables_data["output_format"], str):
            tables_data["output_format"] = OutputFormat(tables_data["output_format"])
        tables = TableConfig(**tables_data) if "tables" in data else TableConfig()
        chunking = ChunkingConfig(**data.get("chunking", {})) if "chunking" in data else ChunkingConfig()
        ocr = OCRConfig(**data.get("ocr", {})) if "ocr" in data else OCRConfig()
        fmt_opts = data.get("format_options", {})

        return cls(
            tags=tags,
            images=images,
            charts=charts,
            metadata=meta,
            tables=tables,
            chunking=chunking,
            ocr=ocr,
            format_options=fmt_opts,
        )


__all__ = [
    "TagConfig",
    "ImageConfig",
    "ChartConfig",
    "MetadataConfig",
    "TableConfig",
    "ChunkingConfig",
    "OCRConfig",
    "ProcessingConfig",
]
