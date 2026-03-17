# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0-alpha] — 2025-07-15

### Breaking Changes — Full Architecture Redesign

v2 is a **complete rewrite** that is **not backwards-compatible** with v1. The package has moved from `contextifier` to `contextifier_new`.

### Added

- **Enforced 5-stage pipeline**: Convert → Preprocess → Metadata → Content → Postprocess
  - `BaseHandler` enforces execution order — all handlers follow the same structure
  - Each stage is defined as an ABC: `Converter`, `Preprocessor`, `MetadataExtractor`, `ContentExtractor`, `Postprocessor`
- **14 format handlers**: PDF, PDF-Plus, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV/TSV, HWP, HWPX, RTF, Text, Image
- **HandlerRegistry**: Automatic extension → handler mapping via `register_defaults()`
- **Immutable config system**: Frozen dataclass-based `ProcessingConfig`
  - `TagConfig`, `ImageConfig`, `ChartConfig`, `MetadataConfig`, `TableConfig`, `ChunkingConfig`, `OCRConfig`
  - Fluent builder: `config.with_tags()`, `config.with_chunking()`, ...
  - Serialization: `to_dict()` / `from_dict()`
  - Format-specific options: `config.with_format_option("pdf", ...)`
- **4 chunking strategies with automatic selection**:
  - `TableChunkingStrategy` (priority 5) — spreadsheet-specific
  - `PageChunkingStrategy` (priority 10) — page boundary-based
  - `ProtectedChunkingStrategy` (priority 20) — HTML table / protected region preservation
  - `PlainChunkingStrategy` (priority 100) — recursive splitting fallback
- **5 OCR engines**: OpenAI, Anthropic, Google Gemini, AWS Bedrock, vLLM
  - Convenience constructors: `from_api_key()` for each engine
  - Direct LangChain client passthrough
  - Custom prompt support
- **5 shared services** (DI pattern):
  - `TagService` — page / slide / sheet tag generation
  - `ImageService` — image saving / tagging / deduplication / storage backends
  - `ChartService` — chart data formatting
  - `TableService` — table HTML / MD / Text rendering
  - `MetadataService` — metadata formatting (Korean / English)
- **Unified type system** (`types.py`):
  - `FileContext` TypedDict — standard input for all handlers
  - `ExtractionResult` — unified text / metadata / table / image / chart output
  - `DocumentMetadata`, `TableData`, `TableCell`, `ChartData` shared dataclasses
  - `FileCategory`, `OutputFormat`, `NamingStrategy`, `StorageType` enums
- **Unified exception hierarchy** (`errors.py`):
  - `ContextifierError` base exception tree
  - `FileNotFoundError`, `UnsupportedFormatError`, `HandlerNotFoundError`, etc.
- **ChunkResult**: `save_to_md()`, `__len__`, `__iter__`, `__getitem__` support
- **DOC handler auto-detection**: OLE, HTML, DOCX, RTF internal format auto-detection

### Removed

- All legacy v1 code in the `contextifier` package
  - `core/document_processor.py` (monolithic single file)
  - `core/functions/` (utils.py, individual processor modules)
  - `core/processor/` (per-handler files without unified structure)
  - `chunking/` (single chunking.py with all logic)
  - `ocr/ocr_engine/` (per-engine files without consistency)

### Architecture Changes

- **Facade pattern**: `DocumentProcessor` is the sole public entry point
- **Strategy pattern**: Automatic chunking strategy selection
- **Template Method pattern**: `BaseHandler.process()` enforces the 5-stage order
- **Dependency Injection**: Services are created once and shared across handlers
- **Registry pattern**: Automatic extension → handler mapping

---

## [0.1.2] — 2025-05-14

### Fixed
- `requirements.txt` path resolution for packaged installs.

---

## [0.1.0] — 2025-05-14

### Added
- Initial release of Contextifier v1.
- Support for PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV, TSV, HWP, HWPX, RTF, TXT, Image.
- OCR integration via OpenAI, Anthropic, Gemini, Bedrock.
- Basic text chunking with page/table awareness.
- Metadata extraction for common document formats.
