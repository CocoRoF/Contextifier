# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.9.0] — 2026-09-12

A pass over extraction correctness, driven by comparing this codebase against
a fork that stayed on the pre-2.0 architecture and kept fixing real documents.
Each item below was reproduced here before being changed, and the fork's own
fix was reviewed rather than copied — several were rejected outright (see
*Considered and declined*).

### Fixed — content that never reached the output

- **Handler↔service contracts.** Thirteen call sites addressed an API that does
  not exist: `save_and_tag(image_bytes=…)` against a parameter named
  `image_data`, `ImageService.save_image()` which is not a method, and
  `TagService.make_page_tag` / `page_tag` against `create_page_tag`. Every one
  sat inside an `except Exception`, so image extraction was dead in seven
  handlers and page/slide tagging in five — with a green test suite, because
  the service doubles were plain `MagicMock`s written against the wrong names.
  Tagging drove more than itself: `PageChunkingStrategy` never matched, so
  every chunk came back with `page_number=None`.
- **DOCX `mc:AlternateContent`.** Word stores every modern shape twice; the
  paragraph walker knew neither branch, so text boxes, shape captions and
  grouped-shape labels produced nothing. Shape text (`w:txbxContent`) is read
  for both renderings, once.
- **DOCX `w:sdt`.** A content control is a wrapper — tables of contents,
  bibliographies, cover-page fields and user-inserted controls keep their
  content in `w:sdtContent`, and a walk matching only `w:p`/`w:tbl` dropped it.
- **DOCX headers and footers** were read through `part.paragraphs`, which sees
  paragraphs only; a header table (document number, revision, classification)
  was discarded.
- **HWPX traversal** collected every `hp:p` in the subtree, so table cells and
  header parts were emitted a second time as loose body text while shape text
  landed at the end of the section. The walk now follows containment, which is
  also what makes shape text, control-anchored tables and header/footer parts
  reachable at all.
- **Table cells** were read for text runs only: a pasted screenshot left the
  cell empty *and* produced no image tag anywhere, so OCR never saw it; a
  nested table was dropped; a content control was skipped.
- **Word field codes** in legacy `.doc` left their instruction text —
  `PAGE \* MERGEFORMAT`, `HYPERLINK "…"` — in the body.
- **Chart titles** were truncated to their first text run in DOCX and HWPX, and
  space-separated in XLSX, so `2026년 매출 실적` arrived as `2026` or
  `2026 년 매출 실적`.
- **Spreadsheets** were scanned within a fixed 1000×100 cell window, silently
  truncating anything larger; a merged category column broke its own table into
  "unrelated" blocks and was lost from all but the first; a merge anchored
  outside a region lost its label and left the row short.
- **HTML served as `.xls`/`.xlsx`** (report portals do this routinely) failed to
  convert at all. Both handlers delegate to the HTML handler, and a table split
  into a header table plus a body table is rejoined.
- **Nested HTML tables** were cut at the inner `</table>`, so the chunker split
  the remainder of the outer table wherever it liked.
- **Merged cells spanning a chunk boundary** were dropped from the continuation
  chunk — on a three-group table split six ways, three chunks lost their
  category label entirely.
- **PDF text quality** was assessed only by whether structured extraction
  returned anything, so a page with a broken font mapping — plenty of
  characters, all wrong — never reached the fallback.

### Added

- `DocumentProcessor.extract_text_fast()` — plain text for scanning ("does this
  file contain a forbidden word?"), skipping table reconstruction, images, OCR,
  charts and layout analysis. Measured: PDF 197×, PPTX 7.0×, XLSX 2.9×,
  DOCX 3.1×.
- `FragmentedTextReconstructor` — rebuilds lines from glyph positions when a
  producer exported each character as its own text object.
- CJK Compatibility density as a garbled-text signal (counted, never
  substituted — see below).
- `DeepSeekOCREngine`, whose message shape and prompt the generic vLLM engine
  cannot stand in for.
- `find_html_tables()`, `compute_carried_cells()` / `reissue_carried_cells()`,
  `strip_field_codes()`, `looks_like_html()`, `iter_block_elements()`,
  `normalize_response_content()`.
- `tests/unit/services/test_service_contracts.py` — an AST check that every
  service call site matches the real signature, so the failure mode that
  started this cannot return silently.

### Changed

- Spreadsheet chunking packs segments that fit instead of emitting one chunk
  per segment; six small tables against a 1000-character budget went from eight
  chunks averaging 97 characters to one.
- Image tags and textbox blocks are no longer chunk boundaries of their own.
- Regions smaller than 2×2 render as plain text rather than as a "table".
- `TableCell.nested_table` is rendered (HTML nests it; Markdown and text
  flatten it) — the field existed and no renderer read it.
- OCR converts each distinct image once, and output landing inside a `<td>` is
  flattened so the table stays parseable.
- An empty OCR answer is a failure rather than an empty `[Figure:]`, so the
  image tag survives for a retry.
- Test doubles for the services are `create_autospec` of the real classes.
- pytest `asyncio_mode = "auto"`; the async tests errored on a fresh checkout.

### Considered and declined

- **A CJK substitution table.** Six of its nine mappings target CJK Extension A
  — ordinary Hanja — so rewriting U+3711 (㜑) as an arrow corrupts any document
  that genuinely uses the character. Density drives OCR instead, and a page
  using ㎏/㎞/㎡ normally is left alone.
- **"Average line under fifteen characters" as fragmentation.** A bulleted
  list, a table of contents and a column of figures all fail that test while
  being perfectly intact.
- **Merging page-marker-only chunks forward.** It attributes the next page's
  content to the empty page; attribution here is already correct.
- **Unconditional small-chunk merging.** 0.9% of chunks across the repo's real
  documents are that small, and the version reviewed has no size check, so
  chained merges can exceed the budget several times over.
- **Relaxed PDF table validation** (paragraph ratio 0.25 → 0.60, two checks
  commented out) — it accepts prose as tables.
- **`[Page Number: 1]` for DOC and RTF**, whose pagination is never computed.
- **`is_linked_to_previous` special-casing** for a first-section header:
  writing to a header clears the flag, so when it is set there is no own header
  part to read.

### Not ported — unverifiable in this environment

Legacy binary parsing that could not be exercised: `.doc` table reconstruction
from `sprmTDefTable`/TC80, `.doc` text boxes and the brute-force piece-table
search, and `.xls` BIFF text boxes and per-sheet Escher images. No `.doc`
samples exist on this machine and LibreOffice cannot convert here, so roughly
1,700 lines of struct parsing would have been merged without ever being run.
The heuristics currently in place (`\x07` cell markers for `.doc` tables, OLE
signature scanning for `.xls` images) remain.

---

## [0.8.0] — 2026-08-06

### Added — native slide authoring, notes, links, bullets, z-order, theme, chart depth

- **`PptxRawDocument.layouts`** + **`add_slide(layout_index, *, at=…)`** — insert a
  NEW native slide from a slide layout, with empty placeholders that inherit the
  layout's geometry (content the caller writes FILLS the layout). Returns the
  `RawSlide`. New parts only; existing parts byte-identical.
- **`RawSlide.set_paragraphs(id, lines)`** — one paragraph per line (fill a
  content placeholder with bullets).
- **`RawSlide.set_notes(text)`** — write speaker notes, creating the notesSlide
  part (+ content-type + rels to the slide and notes master) if absent.
- **`RawSlide.set_paragraph_bullet` / `set_z_order` / `set_hyperlink`** — bullets
  (char/number/none), bring-to-front/send-to-back, run- or shape-level links.
- **`PptxRawDocument.set_theme_color` / `set_theme_font`** — deck-wide theme
  color (accent1..6, dk/lt, hlink) and major/minor font swap.
- **`ChartModel.set_legend(position)` / `set_series_color(idx, color)`** — legend
  show/position and per-series fill.

---

## [0.7.0] — 2026-08-06

### Added — PPTX authoring primitives (shape lifecycle, rich runs, table columns)

- **`RawSlide.add_textbox(text, *, left, top, width, height, color?, size_pt?,
  bold?, italic?)`** — append a real `p:sp` text box; returns its id.
- **`RawSlide.delete_shape(id)`** — remove a shape (+ orphan-part sweep).
- **`RawSlide.duplicate_shape(id, *, left?, top?)`** — deep-copy a shape with a
  fresh `cNvPr` id, optionally repositioned.
- **`RawSlide.set_runs(id, para, [{text, bold?, italic?, color?, size_pt?}…])`**
  — replace a paragraph with independently-styled runs (bold one word).
- **`RawTable.insert_column(idx)` / `delete_column(idx)`** — grid + per-row `a:tc`
  surgery (was out of scope in v0.4).
- **`RawTable.merge_cells(r1, c1, r2, c2)`** — gridSpan/rowSpan + hMerge/vMerge.
- **`RawSlide.paragraphs_by_shape()`** — all text paragraphs in one walk (avoids
  an O(shapes²) outline build).

### Fixed

- `ChartModel.set_data` refuses **bubble charts** (`RawUnsupportedError`) instead
  of silently corrupting the stale `c:bubbleSize` cache.
- `RawSlide.set_text` materializes `a:p` on an empty text placeholder (was
  raising for a `para 0` the outline advertised).

## [0.6.1] — 2026-08-06

### Fixed — PPTX styling edge cases (from an adversarial review)

- `RawTableCell.set_style(fill=…)` now inserts the cell fill AFTER the border
  lines / cell3D in `a:tcPr` (was inserted first → schema-invalid ordering when
  a cell already had explicit borders).
- `set_shape_fill` / `set_shape_position` create a missing `p:spPr` in the
  correct position (after `nvSpPr`, before `txBody`) instead of appending it;
  `set_shape_fill` now rejects a graphicFrame (table/chart) with a clear error
  instead of injecting an invalid `p:spPr`.
- `set_shape_position` handles graphicFrames (edits their `p:xfrm`) and groups
  (`p:grpSpPr/a:xfrm`), not just `p:sp`.

### Added

- `RawSlide.get_paragraphs(shape_id)` — per-`a:p` text, so a caller addresses
  paragraphs by the exact index `set_text` mutates even when a run's text holds
  a literal newline.

## [0.6.0] — 2026-08-06

### Added — in-place PPTX shape styling & geometry (raw layer)

Surgical, byte-preserving mutators so a slide's shapes, text and table cells
can be restyled/moved without redrawing the slide (only the touched slide part
is re-serialized; everything else round-trips byte-identical).

- **`RawShapeInfo.left/top/width/height`** — the shape's own transform in EMU
  (read from `a:xfrm`; `None` when placement is inherited from a placeholder).
- **`RawSlide.set_shape_font(shape_id, *, color, size_pt, bold, italic, para)`**
  — restyle a text shape's runs in place (only the given attributes; runs keep
  their text; `a:endParaRPr` kept in sync).
- **`RawSlide.set_shape_fill(shape_id, color)`** — set a shape's solid fill.
- **`RawSlide.set_shape_position(shape_id, *, left, top, width, height)`** —
  move/resize via `a:xfrm` (created if the shape inherited its placement).
- **`RawTableCell.set_style(*, fill, color, size_pt, bold, italic)`** — restyle
  a table cell's background + text; cell text untouched.

## [0.5.0] — 2026-08-06

### Added — structural operations on the raw layer

Whole slides and sheets become movable, copyable objects — deterministic,
keyless, and byte-preserving (untouched parts round-trip byte-identical).

- **`OpcPackage.clone_part_graph(src, *, rename, share_types, stop_types)`**
  — deep-copies a part and the subtree it references under fresh names,
  duplicating each part's content-type `Override` and rebuilding its
  `.rels` with the same ids but retargeted internal targets. A share/stop
  policy chooses whether a referenced part is copied, shared (referenced),
  or dropped; diamonds and back-references resolve to a single clone.
  Helper `make_part_renamer(package)` allocates fresh in-directory names.
- **`OpcPackage.content_type_override_of(part)`** — the explicit
  `<Override>` content type (no by-extension `<Default>` fallback).
- **PPTX** (`raw/pptx.py`): **`duplicate_slide(index, *, at=None)`** (an
  independent copy — charts + embedded workbooks + notes are cloned,
  images/layouts/notes-master shared, notes back-reference retargeted) and
  **`move_slide(index, to)`** (a pure `p:sldIdLst` reorder — only
  `presentation.xml` changes). `remove_slide` unchanged.
- **XLSX** (`raw/xlsx.py`): **`add_sheet`**, **`copy_sheet`** (clones the
  worksheet + drawing/chart/table subtree, shares images), **`move_sheet`**,
  **`rename_sheet`** (tab only; does not rewrite formula/name references),
  **`delete_sheet`** (orphan-swept; refuses the last sheet).
- The delete/orphan-sweep helpers (`_delete_part`, `_referenced_parts`,
  `_drop_content_type_overrides`, `_sweep_orphans`) moved to
  `RawDocumentBase` so both formats share one tested implementation.

25 new raw tests (byte-preservation, copy independence, orphan sweep,
Office reopen); 623 pass.

## [0.4.0] — 2026-07-07

### Added — the raw layer (`contextifier.raw`)

Contextifier now offers **two views of every document**: the existing
AI-friendly extraction pipeline, and `open_raw()` — a lossless,
addressable, **writable** view of OOXML files (`.xlsx` / `.docx` /
`.pptx`).

- **`open_raw(source)`** (top-level export) and
  `DocumentProcessor.open_raw()` — dispatch to `XlsxRawDocument` /
  `DocxRawDocument` / `PptxRawDocument` by extension or package sniffing.
- **Byte-preservation contract** (`raw/opc.py`): saving writes untouched
  parts back byte-identically (BOM, whitespace, extensions and all).
  Charts, pivot tables, sparklines, custom XML, styles — everything the
  usual Office libraries destroy on a load→save round-trip — survives.
  Contract-tested against synthetic and real files, including a contrast
  test showing openpyxl 3.1.5 dropping sparklines/customXml/chart styles
  that the raw layer preserves.
- **XLSX** (`raw/xlsx.py`): surgical `set_cell` (style attribute
  preserved, strings as `inlineStr`, rows/cells kept sorted),
  `append_rows`, cached-formula reads + `get_formula`, merged ranges,
  defined names, `charts`; overwriting a formula sets
  `calcPr fullCalcOnLoad` so Excel recalculates on open.
- **DOCX** (`raw/docx.py`): python-docx-compatible paragraph/table
  addressing (incl. gridSpan/vMerge grid semantics and nested tables),
  **run-preserving** `set_paragraph_text` and `cell.set_text` (inline
  images, drawings, bookmarks and hyperlink elements survive; hyperlinks
  are emptied, never deleted — `strip_empty_hyperlinks()` when you want
  them gone), `insert_paragraph_after`/`delete_paragraph`,
  `insert_row`/`delete_row`, headers/footers, `body_order()`.
- **PPTX** (`raw/pptx.py`): slide/shape inventory (tables, charts,
  diagrams, groups), format-preserving `set_text`, table cell edits +
  row insert/delete, `notes_text`, **`replace_content(new_xml,
  preserve_native=True)`** — swap a slide's XML while carrying over its
  native charts/tables/pictures (id-renumbered), and `remove_slide()`
  with transitive orphan-part sweeping (chart XML, embedded workbooks,
  media) that provably spares shared layouts/masters.
- **Charts** (`raw/chart.py`): one `ChartModel` for all three formats —
  reads kind/title/series from classic `c:` charts (incl. scatter/bubble
  x/y, literals, multi-level caches) and chartEx; **writes** titles and
  full data (`set_data`) with schema-valid cache/ref rewriting, series
  grow/shrink keeping styling, and embedded-workbook regeneration —
  verified by reopening with python-pptx/openpyxl.
- `lxml` promoted to an explicit dependency.

99 new tests (598 total).

---

## [0.3.0] — 2025-07-19 (released 2026-07-07)

### Added (RAG provenance & payload ingestion — 2026-07-07)

- **Chunk metadata enrichment** (`chunking/metadata_enricher.py`): with
  `include_position_metadata=True`, every `Chunk` now carries real
  structural context — `page_number` (promoted from `[Page Number: N]` /
  `[Slide Number: N]` markers, inherited across marker-less chunks),
  `sheet_name` (`[Sheet: …]`), and a new `heading_path` breadcrumb
  ("H1 > H2 > H3") tracking markdown headings. RAG pipelines no longer
  re-parse chunk text for provenance. `ChunkMetadata` gains
  `heading_path` / `sheet_name` fields (additive).
- **Structure-aware JSON rendering**: `.json` inputs (API payloads,
  exports) extract as one `dotted.path[i]: value` line per leaf with
  blank-line record boundaries between array elements, so chunking
  respects object boundaries. Invalid JSON falls back to plain text.

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
