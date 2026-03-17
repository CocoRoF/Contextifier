# contextifier_new/types.py
"""
Core type definitions for Contextifier v2.

Centralizes all shared types, enums, protocols, and data structures
so that every module in the library speaks the same language.

Design principles:
- Immutable data structures via frozen dataclasses
- Exhaustive enums for all classification values
- TypedDict for dictionary-shaped data passed across boundaries
- Protocol classes for structural typing where ABC is too rigid
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, unique
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    TypedDict,
    Union,
)


# ═══════════════════════════════════════════════════════════════════════════════
# File & Format Types
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class FileCategory(str, Enum):
    """Classification of file types by their nature."""
    DOCUMENT = "document"       # PDF, DOCX, DOC, RTF, HWP, HWPX
    PRESENTATION = "presentation"  # PPT, PPTX
    SPREADSHEET = "spreadsheet"    # XLSX, XLS
    TEXT = "text"               # TXT, MD
    CODE = "code"               # PY, JS, TS, Java, ...
    CONFIG = "config"           # JSON, YAML, XML, TOML, ...
    DATA = "data"               # CSV, TSV
    SCRIPT = "script"           # SH, BAT, PS1, ...
    LOG = "log"                 # LOG
    WEB = "web"                 # HTM, XHTML
    IMAGE = "image"             # JPG, PNG, GIF, ...
    UNKNOWN = "unknown"


@unique
class OutputFormat(str, Enum):
    """Output format for table rendering."""
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


@unique
class ImageFormat(str, Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    GIF = "gif"
    BMP = "bmp"
    WEBP = "webp"
    TIFF = "tiff"
    UNKNOWN = "unknown"


@unique
class NamingStrategy(str, Enum):
    """Strategy for naming saved files."""
    HASH = "hash"           # Content-based hash (deduplication)
    UUID = "uuid"           # Random UUID
    SEQUENTIAL = "sequential"  # Counter-based
    TIMESTAMP = "timestamp"    # Time-based


@unique
class StorageType(str, Enum):
    """Storage backend types."""
    LOCAL = "local"
    MINIO = "minio"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


@unique
class TagType(str, Enum):
    """Types of structural tags in extracted text."""
    PAGE = "page"
    SLIDE = "slide"
    SHEET = "sheet"


@unique
class PipelineStage(str, Enum):
    """Named stages in the processing pipeline."""
    CONVERT = "convert"
    PREPROCESS = "preprocess"
    EXTRACT_METADATA = "extract_metadata"
    EXTRACT_CONTENT = "extract_content"
    POSTPROCESS = "postprocess"


# ═══════════════════════════════════════════════════════════════════════════════
# File Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class FileContext(TypedDict):
    """
    Standardized file input for all handlers.

    Created once by DocumentProcessor from the file path,
    then passed through the entire pipeline unchanged.
    Binary-level reading resolves encoding/path issues upfront.
    """
    file_path: str          # Absolute path to the original file
    file_name: str          # Filename with extension
    file_extension: str     # Lowercase extension without dot
    file_category: str      # FileCategory value
    file_data: bytes        # Raw binary content
    file_stream: io.BytesIO # Seekable binary stream (reusable)
    file_size: int          # Size in bytes


# ═══════════════════════════════════════════════════════════════════════════════
# Document Metadata
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class MetadataField(str, Enum):
    """Standard metadata field names."""
    TITLE = "title"
    SUBJECT = "subject"
    AUTHOR = "author"
    KEYWORDS = "keywords"
    COMMENTS = "comments"
    LAST_SAVED_BY = "last_saved_by"
    CREATE_TIME = "create_time"
    LAST_SAVED_TIME = "last_saved_time"
    PAGE_COUNT = "page_count"
    WORD_COUNT = "word_count"
    CATEGORY = "category"
    REVISION = "revision"


@dataclass
class DocumentMetadata:
    """
    Standardized metadata container used across all formats.

    Every handler's MetadataExtractor produces this same structure,
    ensuring uniform metadata handling regardless of source format.
    """
    title: Optional[str] = None
    subject: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[str] = None
    comments: Optional[str] = None
    last_saved_by: Optional[str] = None
    create_time: Optional[datetime] = None
    last_saved_time: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    category: Optional[str] = None
    revision: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        result: Dict[str, Any] = {}
        for f in [
            "title", "subject", "author", "keywords", "comments",
            "last_saved_by", "create_time", "last_saved_time",
            "page_count", "word_count", "category", "revision",
        ]:
            val = getattr(self, f)
            if val is not None:
                if isinstance(val, datetime):
                    result[f] = val.isoformat()
                else:
                    result[f] = val
        if self.custom:
            result["custom"] = self.custom
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create from dictionary."""
        kwargs: Dict[str, Any] = {}
        for f in [
            "title", "subject", "author", "keywords", "comments",
            "last_saved_by", "page_count", "word_count", "category", "revision",
        ]:
            if f in data:
                kwargs[f] = data[f]
        for time_field in ("create_time", "last_saved_time"):
            if time_field in data:
                val = data[time_field]
                if isinstance(val, str):
                    try:
                        kwargs[time_field] = datetime.fromisoformat(val)
                    except ValueError:
                        kwargs[time_field] = None
                elif isinstance(val, datetime):
                    kwargs[time_field] = val
        if "custom" in data:
            kwargs["custom"] = data["custom"]
        return cls(**kwargs)

    def is_empty(self) -> bool:
        """Check if all fields are empty/None."""
        for f in [
            "title", "subject", "author", "keywords", "comments",
            "last_saved_by", "create_time", "last_saved_time",
            "page_count", "word_count", "category", "revision",
        ]:
            if getattr(self, f) is not None:
                return False
        return not self.custom


# ═══════════════════════════════════════════════════════════════════════════════
# Table Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TableCell:
    """Single cell in a table."""
    content: str = ""
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    row_index: int = 0
    col_index: int = 0
    nested_table: Optional["TableData"] = None


@dataclass
class TableData:
    """
    Complete parsed table, uniform across all formats.

    Every format's ContentExtractor produces TableData instances
    using the same structure — PDF tables, DOCX tables, Excel sheets,
    CSV data all normalize to this representation.
    """
    rows: List[List[TableCell]] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0
    has_header: bool = False
    col_widths_percent: Optional[List[float]] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Chart Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChartSeries:
    """One data series in a chart."""
    name: Optional[str] = None
    values: List[Any] = field(default_factory=list)


@dataclass
class ChartData:
    """
    Standardized chart representation across all formats.

    Whether from DOCX, PPTX, XLSX, or HWP — chart data normalizes here.
    """
    chart_type: Optional[str] = None
    title: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    series: List[ChartSeries] = field(default_factory=list)
    raw_content: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionResult:
    """
    Unified result from the processing pipeline.

    Every handler returns this same structure, making downstream
    consumers (chunking, OCR, user code) format-agnostic.
    """
    text: str = ""
    metadata: Optional[DocumentMetadata] = None
    tables: List[TableData] = field(default_factory=list)
    charts: List[ChartData] = field(default_factory=list)
    images: List[str] = field(default_factory=list)  # Image tag strings or paths
    page_count: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def has_metadata(self) -> bool:
        return self.metadata is not None and not self.metadata.is_empty()

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def has_charts(self) -> bool:
        return len(self.charts) > 0

    @property
    def has_images(self) -> bool:
        return len(self.images) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Chunk Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkMetadata:
    """Position metadata for a single chunk."""
    chunk_index: int = 0
    page_number: Optional[int] = None
    line_start: int = 0
    line_end: int = 0
    global_start: int = 0
    global_end: int = 0


@dataclass
class Chunk:
    """A single text chunk with optional position metadata."""
    text: str = ""
    metadata: Optional[ChunkMetadata] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessed Data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessedData:
    """
    Output of the preprocessing stage.

    The preprocessor transforms the converted format object
    into cleaned data ready for content extraction.
    """
    content: Any = None                      # Primary processed content
    raw_content: Any = None                  # Original input (for reference)
    encoding: str = "utf-8"                  # Detected encoding
    resources: Dict[str, Any] = field(default_factory=dict)   # Extracted resources (images, etc.)
    properties: Dict[str, Any] = field(default_factory=dict)  # Discovered properties during preprocessing


# ═══════════════════════════════════════════════════════════════════════════════
# File Extension Registry
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical extension → category mapping
EXTENSION_CATEGORIES: Dict[str, FileCategory] = {}

_CATEGORY_EXTENSIONS: Dict[FileCategory, FrozenSet[str]] = {
    FileCategory.DOCUMENT: frozenset(["pdf", "docx", "doc", "rtf", "hwp", "hwpx"]),
    FileCategory.PRESENTATION: frozenset(["ppt", "pptx"]),
    FileCategory.SPREADSHEET: frozenset(["xlsx", "xls"]),
    FileCategory.TEXT: frozenset(["txt", "md", "markdown"]),
    FileCategory.CODE: frozenset([
        "py", "js", "ts", "java", "cpp", "c", "h", "cs", "go", "rs",
        "php", "rb", "swift", "kt", "scala", "dart", "r", "sql",
        "html", "css", "jsx", "tsx", "vue", "svelte",
    ]),
    FileCategory.CONFIG: frozenset([
        "json", "yaml", "yml", "xml", "toml", "ini", "cfg", "conf", "properties", "env",
    ]),
    FileCategory.DATA: frozenset(["csv", "tsv"]),
    FileCategory.SCRIPT: frozenset(["sh", "bat", "ps1", "zsh", "fish"]),
    FileCategory.LOG: frozenset(["log"]),
    FileCategory.WEB: frozenset(["htm", "xhtml"]),
    FileCategory.IMAGE: frozenset(["jpg", "jpeg", "png", "gif", "bmp", "webp"]),
}

# Build reverse lookup
for _cat, _exts in _CATEGORY_EXTENSIONS.items():
    for _ext in _exts:
        EXTENSION_CATEGORIES[_ext] = _cat

ALL_SUPPORTED_EXTENSIONS: FrozenSet[str] = frozenset(EXTENSION_CATEGORIES.keys())


def get_category(extension: str) -> FileCategory:
    """Get the category for a file extension."""
    return EXTENSION_CATEGORIES.get(extension.lower().lstrip("."), FileCategory.UNKNOWN)


def get_extensions(category: FileCategory) -> FrozenSet[str]:
    """Get all extensions for a category."""
    return _CATEGORY_EXTENSIONS.get(category, frozenset())


__all__ = [
    # Enums
    "FileCategory",
    "OutputFormat",
    "ImageFormat",
    "NamingStrategy",
    "StorageType",
    "TagType",
    "PipelineStage",
    "MetadataField",
    # Data structures
    "FileContext",
    "DocumentMetadata",
    "TableCell",
    "TableData",
    "ChartSeries",
    "ChartData",
    "ExtractionResult",
    "ChunkMetadata",
    "Chunk",
    "PreprocessedData",
    # Extension registry
    "EXTENSION_CATEGORIES",
    "ALL_SUPPORTED_EXTENSIONS",
    "get_category",
    "get_extensions",
]
