# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2025-07-19

### Security (Phase 0)

- **HTML Escape**: `TableService.format_as_html()` now escapes cell content (`html.escape()`) to prevent XSS
- **Base64 Image Size Limit**: HTML handler limits base64 image decode to 50 MB (`MAX_IMAGE_DECODE_SIZE`)
- **Path Traversal Defense**: `LocalStorageBackend` and `ImageService` validate output paths stay within base directory
- **ZIP Bomb Defense**: DOCX/PPTX/XLSX/HWPX converters check decompressed size against 1 GB threshold

### Fixed (Phase 1)

- **PDF Scan OCR**: PDF default handler now renders scanned pages as images and inserts `[Image: ...]` tags for OCR pipeline
- **DOC Native Parsing**: FIB + Piece Table based text extraction, Table Stream parsing for `extract_tables()`, OLE stream image extraction
- **PPT Table/Chart Extraction**: OLE2 record-based table detection and chart extraction from PowerPoint binary format
- **XLS Image/Chart Extraction**: OLE stream image signature scanning, BIFF chart record parsing
- **Removed libreoffice.py**: Eliminated external tool dependency (133 LOC); all formats use native binary parsing
- **Delegation Depth Limit**: `_delegate_to()` enforces max 3-level delegation depth to prevent infinite loops

### Improved (Phase 2)

- **RTF Merged Cell Verification**: `\clmgf`, `\clmrg`, `\clvmgf`, `\clvmrg` support verified and tested
- **HWPX Chart Extraction**: OOXML chart XML parsed inline during `extract_text()`
- **Image Handler OCR**: Integrated with OCR pipeline via `[Image: ...]` tag insertion
- **Tesseract OCR Engine**: `TesseractOCREngine` implemented (`pytesseract` wrapper) — local OCR without LLM
- **PPTX Group Shape Charts**: Recursive shape traversal includes chart extraction within group shapes
- **XLSX Chart Extraction**: openpyxl chart API integration for `extract_charts()`
- **OCR Prompt Language**: `OCRConfig.prompt_language` setting with `"ko"`/`"en"` prompt templates
- **Configurable Thresholds**: `format_options` for `pdf.table_size`, `pptx.max_group_depth`, `csv.delimiter_candidates`, `doc.min_text_fragment_length`

### Added (Phase 3 — Test Infrastructure)

- **423 unit tests** covering all core components:
  - `ImageService` (24 tests), `ChartService` (12 tests), `MetadataService` (10 tests)
  - `CachedDocumentProcessor` (28 tests), `AsyncDocumentProcessor` (12 tests)
  - Delegation path tests (14 tests), Security tests (16 tests)
  - Integration test framework with `conftest.py`

### Performance (Phase 4)

- **CSV Streaming**: `max_rows` parameter limits in-memory row count; `truncated` flag in output
- **XLSX Read-Only Mode**: `format_options["xlsx"]["read_only"]` for memory-efficient large file processing
- **OCR Parallel Processing**: `OCRProcessor.max_workers` parameter enables `ThreadPoolExecutor`-based parallel OCR
- **Shared ThreadPoolExecutor**: `BaseHandler._timeout_executor` is a class-level shared executor (lazy init, atexit cleanup)
- **CachedProcessor Extension**: `process()` and `extract_chunks()` now cacheable with JSON serialization
- **LRU Cache**: `MemoryCacheBackend` uses `OrderedDict`-based LRU eviction instead of FIFO
- **Image Size Limit**: `ImageConfig.max_file_size_mb` skips oversized images with warning

### Documentation (Phase 5)

- **Handler Comparison Table**: `docs/handler_comparison.md` — feature matrix for all 15 handlers
- **Configuration Reference**: `docs/configuration.md` — all config classes, options, defaults, examples
- **Error Codes Reference**: `docs/error_codes.md` — exception hierarchy, error codes, troubleshooting
- **OCR Setup Guide**: `docs/ocr_guide.md` — 6 engine setup, prompt customization, parallel processing
- **Plugin Development Guide**: `docs/plugin_development.md` — BaseHandler extension, 5-stage pipeline, testing
- **CHANGELOG v0.3.0**: This changelog entry

### Ecosystem (Phase 6)

- **LangChain Integration**: `ContextifierLoader(BaseLoader)` in `contextifier.integrations.langchain_loader`
  - Single document / chunked mode, OCR support, lazy_load
- **CI/CD Pipeline**: `.github/workflows/ci.yml` — lint, test matrix (Python 3.12/3.13), type-check, PyPI publish
- **Docker Support**: Multi-stage `Dockerfile` with Tesseract OCR and Poppler
- **Password-Protected Files**: `crypto_service.decrypt_if_encrypted()` via msoffcrypto-tool
  - `extract_text(password=)`, `process(password=)`, `extract_chunks(password=)` API
- **License Review**: PyMuPDF (AGPL-3.0) moved to optional `[pdf]` extra; guarded imports
- **CSV Delimiter Confidence**: `_detect_delimiter()` returns `(delimiter, confidence)` tuple
  - `CsvParsedData.delimiter_confidence` field, exposed in `PreprocessedData.properties`
- **EncodingConfig**: New config class with `fallback_encodings`, `force_encoding`, `min_confidence`
  - `ProcessingConfig.with_encoding()` fluent API; wired through CSV/TSV/Text converters

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
