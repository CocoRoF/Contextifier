# Contextifier v2 — Processing Flow

This document describes the complete document processing flow in Contextifier v2 with diagrams.

---

## 1. Overall Flow

```
User Code
    │
    ▼
DocumentProcessor
    │
    ├─ extract_text(file_path) ──────────────────┐
    │                                             │
    ├─ process(file_path) ───────────────────────┤
    │                                             │
    ├─ extract_chunks(file_path) ────────────────┤
    │       │                                     │
    │       ├── calls extract_text()              │
    │       └── calls chunk_text()                │
    │                                             │
    └─ chunk_text(text) ─── TextChunker           │
                                                  ▼
                                    HandlerRegistry
                                         │
                                         ▼
                                Extension → Handler lookup
                                         │
                                         ▼
                               BaseHandler.process()
                                         │
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                   ▼
            _check_delegation()    5-Stage Pipeline     ExtractionResult
            (delegate to another         │                  returned
             handler if needed)          ▼
                    ┌─────────────────────────────────────┐
                    │  Stage 1: Converter.convert()       │
                    │  Stage 2: Preprocessor.preprocess()  │
                    │  Stage 3: MetadataExtractor.extract() │
                    │  Stage 4: ContentExtractor.extract_all() │
                    │  Stage 5: Postprocessor.postprocess()│
                    └─────────────────────────────────────┘
                                        │
                                        ▼
                              OCRProcessor (optional)
                              Image tags → text conversion
```

---

## 2. 5-Stage Pipeline Detail

Every handler executes the same 5 stages, enforced by `BaseHandler.process()`.
Handlers implement **only the stage components** — execution order cannot be overridden.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Stage 1     │     │  Stage 2     │     │  Stage 3     │     │  Stage 4     │     │  Stage 5     │
│  CONVERT     │────▶│  PREPROCESS  │────▶│  METADATA    │────▶│  CONTENT     │────▶│  POSTPROCESS │
│              │     │              │     │              │     │              │     │              │
│  Binary →    │     │  Normalize,  │     │  Title /     │     │  Text /      │     │  Metadata    │
│  Format Obj  │     │  Encoding,   │     │  Author /    │     │  Table /     │     │  tag insert, │
│              │     │  Preprocess  │     │  Date / Page │     │  Image /     │     │  Final       │
│              │     │              │     │  count       │     │  Chart       │     │  assembly    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Stage 1: Convert

Transforms raw binary file data into a format-specific object.

| Handler | Conversion |
|---------|-----------|
| PDF | `file_data` → `pdfplumber.PDF` object |
| DOCX | `file_data` → `docx.Document` object |
| DOC | `file_data` → LibreOffice → DOCX/HTML |
| PPTX | `file_stream` → `pptx.Presentation` object |
| PPT | `file_data` → LibreOffice → PPTX |
| XLSX | `file_data` → `openpyxl.Workbook` object |
| XLS | `file_data` → `xlrd.Book` object |
| CSV/TSV | `file_data` → auto encoding detection → string |
| HWP | `file_data` → OLE compound file parsing |
| HWPX | `file_data` → ZIP extraction → XML DOM |
| RTF | `file_data` → LibreOffice → HTML |
| Text | `file_data` → auto encoding detection → string |
| Image | `file_data` → temp file save (for OCR) |

### Stage 2: Preprocess

Normalization and preprocessing on the converted object.

- Encoding unification
- Removal of unnecessary metadata streams
- Page/slide order validation
- Temp file preparation

### Stage 3: Metadata Extract

Produces a `DocumentMetadata` structure.

```python
@dataclass
class DocumentMetadata:
    title: str | None
    subject: str | None
    author: str | None
    keywords: str | None
    comments: str | None
    last_saved_by: str | None
    create_time: datetime | None
    last_saved_time: datetime | None
    page_count: int | None
    word_count: int | None
    category: str | None
    revision: str | None
    custom: dict
```

### Stage 4: Content Extract

Extracts text, tables, images, and charts into `ExtractionResult`.

```python
@dataclass
class ExtractionResult:
    text: str                          # Full extracted text
    metadata: DocumentMetadata | None  # Stage 3 result
    tables: list[TableData]            # Extracted tables
    images: list[str]                  # Saved image paths
    charts: list[ChartData]            # Extracted charts
```

This stage leverages shared services:
- **TagService**: Generates page / slide / sheet tags
- **ImageService**: Saves images + generates tags + deduplication
- **ChartService**: Formats chart data → HTML table + tag wrapping
- **TableService**: Renders `TableData` → HTML / Markdown / Text

### Stage 5: Postprocess

Assembles the final text:

1. Format metadata via `MetadataService` → wrap with tags
2. Combine per-page text + table + image tags + charts
3. Clean up consecutive blank lines
4. Return final text string

---

## 3. Delegation Flow

Some files have an internal format that differs from their extension.
In these cases, the handler automatically **delegates** to the correct handler.

```
DOC Handler
    │
    ├─ _check_delegation(file_context)
    │     │
    │     ├─ OLE signature detected → DOC Handler continues
    │     ├─ HTML marker detected  → Delegate to HTML Reprocessor
    │     ├─ DOCX signature detected → Delegate to DOCX Handler
    │     └─ RTF signature detected  → Delegate to RTF Handler
    │
    ▼
Normal DOC processing OR delegated handler result returned
```

When delegation occurs, `BaseHandler._delegate_to(extension, file_context)` looks up the appropriate handler via `HandlerRegistry`. The original handler's pipeline does **not** execute.

---

## 4. Chunking Flow

```
extract_chunks() or chunk_text()
    │
    ▼
TextChunker
    │
    ├─ Registered strategies (sorted by priority)
    │     │
    │     ├─ TableChunkingStrategy     (priority 5)
    │     ├─ PageChunkingStrategy      (priority 10)
    │     ├─ ProtectedChunkingStrategy (priority 20)
    │     └─ PlainChunkingStrategy     (priority 100)
    │
    ├─ Query each strategy: can_handle(text, extension)
    │     │
    │     ├─ Spreadsheet (xlsx, xls, csv, tsv) → TableChunkingStrategy ✓
    │     ├─ Page tags present                 → PageChunkingStrategy ✓
    │     ├─ HTML tables present               → ProtectedChunkingStrategy ✓
    │     └─ Default                           → PlainChunkingStrategy ✓ (always True)
    │
    ▼
Selected strategy executes chunk(text, chunk_size, chunk_overlap)
    │
    ▼
Returns List[str] or List[Chunk]
```

### Strategy Details

#### TableChunkingStrategy (Spreadsheets)
1. Split on `[Sheet: ...]` tags to isolate sheets
2. Separate tables within each sheet into individual chunks
3. Split oversized tables by row

#### PageChunkingStrategy (Page-based)
1. Split on `[Page Number: ...]` tags to isolate pages
2. If a single page exceeds `chunk_size`, apply recursive splitting
3. Preserve tables within pages

#### ProtectedChunkingStrategy (Protected Regions)
1. Identify **protected regions** (HTML tables, metadata blocks, etc.)
2. Replace protected regions with placeholders
3. Recursively split the remaining text
4. Restore placeholders with original content

#### PlainChunkingStrategy (Default Fallback)
1. Recursive character splitting (similar to LangChain `RecursiveCharacterTextSplitter`)
2. Split hierarchy: `\n\n` → `\n` → `. ` → ` ` → character
3. Maintain `chunk_overlap` between chunks

---

## 5. OCR Processing Flow

```
extract_text(..., ocr_processing=True)
    │
    ├─ Handler pipeline runs → produces text
    │     (Image positions have [Image: path/to/img.png] tags)
    │
    ▼
OCRProcessor.process(text)
    │
    ├─ Regex scan for [Image: ...] tags
    │
    ├─ For each tag:
    │     │
    │     ├─ Extract image file path
    │     ├─ BaseOCREngine.convert_image_to_text(path)
    │     │     │
    │     │     ├─ Image → Base64 encoding
    │     │     ├─ build_message_content(b64, mime, prompt)
    │     │     │     (engine-specific LLM message payload)
    │     │     ├─ Create LangChain HumanMessage
    │     │     ├─ Call LLM (Vision API)
    │     │     └─ Return [Figure: result_text]
    │     │
    │     └─ Replace [Image: ...] tag with [Figure: ...] text
    │
    ▼
Return OCR-processed final text
```

### Supported OCR Engines

| Engine | Provider | Default Model |
|--------|----------|---------------|
| `OpenAIOCREngine` | OpenAI | gpt-4o |
| `AnthropicOCREngine` | Anthropic | claude-sonnet-4-20250514 |
| `GeminiOCREngine` | Google | gemini-2.0-flash |
| `BedrockOCREngine` | AWS Bedrock | anthropic.claude-3-5-sonnet |
| `VLLMOCREngine` | Self-hosted | (user-specified) |

---

## 6. Service Dependency Graph

```
DocumentProcessor
    │
    ├─ TagService (standalone — no dependencies)
    │     ├─ Generates page / slide / sheet tags
    │     └─ Customizable via TagConfig prefix/suffix
    │
    ├─ ImageService (depends on TagService + StorageBackend)
    │     ├─ Saves images (Local / MinIO / S3 / Azure / GCS)
    │     ├─ Generates image tags (delegates to TagService)
    │     └─ Hash-based deduplication
    │
    ├─ ChartService (depends on TagService)
    │     ├─ ChartData → HTML table conversion
    │     └─ Wraps with [chart]...[/chart] tags
    │
    ├─ TableService (standalone)
    │     └─ TableData → HTML / Markdown / Text rendering
    │
    └─ MetadataService (standalone)
          ├─ DocumentMetadata → formatted text
          └─ Korean / English label support
```

Services are created **once** by `DocumentProcessor` at initialization and shared across all handlers.

---

## 7. Handler Summary

| Handler | Package | Extension(s) | Conversion Method | Notes |
|---------|---------|-------------|-------------------|-------|
| **PDFHandler** | `handlers/pdf/` | `.pdf` | pdfplumber | Default PDF processing |
| **PDFPlusHandler** | `handlers/pdf_plus/` | `.pdf` | pdfplumber + advanced analysis | Table detection, text quality analysis, complex layouts |
| **DOCXHandler** | `handlers/docx/` | `.docx` | python-docx | Direct table / chart / image extraction |
| **DOCHandler** | `handlers/doc/` | `.doc` | LibreOffice conversion | Auto-detects OLE / HTML / DOCX / RTF + delegation |
| **PPTXHandler** | `handlers/pptx/` | `.pptx` | python-pptx | Slide / notes / chart extraction |
| **PPTHandler** | `handlers/ppt/` | `.ppt` | LibreOffice → PPTX | Delegates to PPTX Handler |
| **XLSXHandler** | `handlers/xlsx/` | `.xlsx` | openpyxl | Multi-sheet, charts, formulas |
| **XLSHandler** | `handlers/xls/` | `.xls` | xlrd | Multi-sheet, charts |
| **CSVHandler** | `handlers/csv/` | `.csv`, `.tsv` | Custom parser | Auto encoding / delimiter detection |
| **HWPHandler** | `handlers/hwp/` | `.hwp` | OLE parsing | HWP 5.0 binary; 3.0 not supported |
| **HWPXHandler** | `handlers/hwpx/` | `.hwpx` | ZIP + XML | OWPML format |
| **RTFHandler** | `handlers/rtf/` | `.rtf` | LibreOffice → HTML | Uses HTML Reprocessor |
| **TextHandler** | `handlers/text/` | `.txt`, `.md`, `.py`, `.json`, ... | Auto encoding detection | 80+ extension category handler |
| **ImageHandler** | `handlers/image/` | `.jpg`, `.png`, `.gif`, ... | Temp file save | Requires OCR engine |

---

## 8. Configuration Flow

```
ProcessingConfig (immutable, frozen dataclass)
    │
    ├─ Received by DocumentProcessor.__init__()
    │
    ├─ Passed to services at creation
    │     ├─ TagService(config)
    │     ├─ ImageService(config, storage, tag_service)
    │     ├─ ChartService(config, tag_service)
    │     ├─ TableService(config)
    │     └─ MetadataService(config)
    │
    ├─ HandlerRegistry(config, services)
    │     └─ Passes config + services to each handler at creation
    │
    └─ TextChunker(config)
          └─ Reads ChunkingConfig
```

Configuration is **immutable after creation**.
If you need different settings, create a new `ProcessingConfig` and a new `DocumentProcessor`.

```python
config1 = ProcessingConfig(chunking=ChunkingConfig(chunk_size=1000))
config2 = config1.with_chunking(chunk_size=2000)  # New instance

proc1 = DocumentProcessor(config=config1)  # chunk_size=1000
proc2 = DocumentProcessor(config=config2)  # chunk_size=2000
```
