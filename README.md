# Contextifier v2

**Contextifier** is a Python document processing library that converts documents of various formats into structured, AI-ready text. It applies a **uniform 5-stage pipeline** to every document format, ensuring consistent and predictable output.

## Key Features

- **Broad Format Support**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HWP, HWPX, RTF, CSV, TSV, TXT, MD, HTML, images, code files, and 80+ extensions
- **Two Views of Every Document**: the AI-friendly pipeline (lightweight, normalized text for LLMs) *and* `open_raw()` — a lossless, addressable, **writable** view of OOXML files where saving keeps untouched parts byte-identical
- **Intelligent Text Extraction**: Preserves document structure (headings, tables, image positions) with automatic metadata extraction
- **Table Processing**: Converts tables to HTML/Markdown/Text with `rowspan`/`colspan` support for merged cells
- **OCR Integration**: 5 Vision LLM engines — OpenAI, Anthropic, Google Gemini, AWS Bedrock, vLLM
- **Smart Chunking**: 4 strategies with automatic selection — table-aware, page-boundary, protected-region, and recursive splitting
- **Immutable Config System**: Frozen dataclass-based `ProcessingConfig` controls all behavior

## Installation

```bash
pip install contextifier
```

or

```bash
uv add contextifier
```

## Quick Start

### 1. Basic Text Extraction

```python
from contextifier import DocumentProcessor

processor = DocumentProcessor()
text = processor.extract_text("document.pdf")
print(text)
```

### 2. Raw Access — Read *and Write* OOXML Losslessly

The extraction pipeline renders an AI-friendly view and discards the
rest. `open_raw()` is its lossless twin: the full package stays
available, edits are surgical, and **untouched parts round-trip
byte-identically** — charts, pivot tables, sparklines, styles and
custom XML all survive (unlike a load→save round-trip through the
usual Office libraries).

```python
from contextifier import open_raw

raw = open_raw("report.xlsx")              # XlsxRawDocument
raw.sheets["Sales"].set_cell("B3", 142)    # surgical edit
raw.charts[0].set_data(                    # real chart-data editing
    categories=["Q1", "Q2", "Q3"],
    series=[("Sales", [120, 135, 150])],
)
raw.save("report-edited.xlsx")

raw = open_raw("paper.docx")               # DocxRawDocument
raw.set_paragraph_text(3, "Revised text")  # runs & inline images preserved
raw.tables[0].insert_row(2)

raw = open_raw("deck.pptx")                # PptxRawDocument
raw.slides[0].set_text(shape_id=2, new_text="New title")
raw.save("deck2.pptx")
```

Supported raw formats: `.xlsx` / `.docx` / `.pptx` (the OOXML trio).
Every model also exposes `.package` for part-level OPC access.

### 3. Extract + Chunk in One Step

```python
from contextifier import DocumentProcessor

processor = DocumentProcessor()
result = processor.extract_chunks("document.pdf")

for i, chunk in enumerate(result.chunks, 1):
    print(f"Chunk {i}: {chunk[:100]}...")

# Save as Markdown files
result.save_to_md("output/chunks")
```

### 4. Custom Configuration

```python
from contextifier import DocumentProcessor
from contextifier.config import ProcessingConfig, ChunkingConfig, TagConfig

config = ProcessingConfig(
    tags=TagConfig(page_prefix="<page>", page_suffix="</page>"),
    chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=300),
)

processor = DocumentProcessor(config=config)
text = processor.extract_text("report.xlsx")
```

### 5. OCR Integration

```python
from contextifier import DocumentProcessor
from contextifier.ocr.engines import OpenAIOCREngine

ocr = OpenAIOCREngine.from_api_key("sk-...", model="gpt-4o")
processor = DocumentProcessor(ocr_engine=ocr)

text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

## Supported Formats

| Category | Extensions | Notes |
|----------|-----------|-------|
| **Documents** | `.pdf`, `.docx`, `.doc`, `.hwp`, `.hwpx`, `.rtf` | HWP 5.0+, HWPX supported |
| **Presentations** | `.pptx`, `.ppt` | Slides, notes, and charts extracted |
| **Spreadsheets** | `.xlsx`, `.xls`, `.csv`, `.tsv` | Multi-sheet, formulas, charts |
| **Text** | `.txt`, `.md`, `.log`, `.rst` | Auto encoding detection |
| **Web** | `.html`, `.htm`, `.xhtml` | Table/structure preservation |
| **Code** | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs`, etc. (20+) | Language-aware highlighting |
| **Config** | `.json`, `.yaml`, `.toml`, `.ini`, `.xml`, `.env` | Structure preservation |
| **Images** | `.jpg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff` | Requires OCR engine |

## Architecture

```
contextifier/
├── document_processor.py     # Facade: single public entry point
├── config.py                 # Immutable config system (ProcessingConfig)
├── types.py                  # Shared types / Enums / TypedDicts
├── errors.py                 # Unified exception hierarchy
│
├── handlers/                 # 14 format-specific handlers
│   ├── base.py               #   BaseHandler — enforces 5-stage pipeline
│   ├── registry.py           #   HandlerRegistry — extension → handler mapping
│   ├── pdf/                  #   PDF (default)
│   ├── pdf_plus/             #   PDF (advanced: table detection, complex layouts)
│   ├── docx/ doc/ pptx/ ppt/ #   Office documents
│   ├── xlsx/ xls/ csv/       #   Spreadsheets / data
│   ├── hwp/ hwpx/            #   Korean word processor
│   ├── rtf/ text/            #   RTF / text / code / config
│   └── image/                #   Image (OCR integration)
│
├── pipeline/                 # 5-Stage pipeline ABCs
│   ├── converter.py          #   Stage 1: Binary → Format Object
│   ├── preprocessor.py       #   Stage 2: Preprocessing
│   ├── metadata_extractor.py #   Stage 3: Metadata extraction
│   ├── content_extractor.py  #   Stage 4: Text / table / image / chart extraction
│   └── postprocessor.py      #   Stage 5: Final assembly & cleanup
│
├── services/                 # Shared services (DI)
│   ├── tag_service.py        #   Page / slide / sheet tag generation
│   ├── image_service.py      #   Image saving / tagging / deduplication
│   ├── chart_service.py      #   Chart data formatting
│   ├── table_service.py      #   Table HTML / MD rendering
│   ├── metadata_service.py   #   Metadata formatting
│   └── storage/              #   Storage backends (Local, MinIO, S3, ...)
│
├── chunking/                 # Chunking subsystem
│   ├── chunker.py            #   TextChunker — auto strategy selection
│   ├── constants.py          #   Protected region patterns
│   └── strategies/           #   4 chunking strategies
│       ├── plain_strategy.py     # Recursive splitting (default fallback)
│       ├── table_strategy.py     # Sheet / table-based splitting
│       ├── page_strategy.py      # Page-boundary splitting
│       └── protected_strategy.py # Protected region preservation
│
└── ocr/                      # OCR subsystem (optional)
    ├── base.py               #   BaseOCREngine ABC
    ├── processor.py          #   OCRProcessor — tag detection + engine call
    └── engines/              #   5 engine implementations
        ├── openai_engine.py
        ├── anthropic_engine.py
        ├── gemini_engine.py
        ├── bedrock_engine.py
        └── vllm_engine.py
```

## Requirements

- **Python** 3.12+
- Required dependencies are included in `pyproject.toml`
- **Optional**: LibreOffice (DOC/PPT/RTF conversion), Poppler (PDF image extraction)

## Documentation

| Document | Contents |
|----------|----------|
| [QUICKSTART.md](QUICKSTART.md) | Detailed usage guide & full API reference |
| [Process Logic.md](Process%20Logic.md) | Handler processing flow diagrams |
| [ARCHITECTURE.md](contextifier/ARCHITECTURE.md) | Internal architecture specification |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [Handler Comparison](docs/handler_comparison.md) | Handler feature support matrix |
| [Configuration Reference](docs/configuration.md) | All config options, defaults & examples |
| [Error Codes](docs/error_codes.md) | Exception hierarchy & troubleshooting |
| [OCR Guide](docs/ocr_guide.md) | OCR engine setup & customization |
| [Plugin Development](docs/plugin_development.md) | Custom handler development guide |

## License

Apache License 2.0 — see [LICENSE](LICENSE)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
