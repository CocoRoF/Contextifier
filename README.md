# Contextifier v2

**Contextifier** is a Python document processing library that converts documents of various formats into structured, AI-ready text. It applies a **uniform 5-stage pipeline** to every document format, ensuring consistent and predictable output.

## Key Features

- **Broad Format Support**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HWP, HWPX, RTF, CSV, TSV, TXT, MD, HTML, images, code files, and 80+ extensions
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
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()
text = processor.extract_text("document.pdf")
print(text)
```

### 2. Extract + Chunk in One Step

```python
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()
result = processor.extract_chunks("document.pdf")

for i, chunk in enumerate(result.chunks, 1):
    print(f"Chunk {i}: {chunk[:100]}...")

# Save as Markdown files
result.save_to_md("output/chunks")
```

### 3. Custom Configuration

```python
from contextifier_new import DocumentProcessor
from contextifier_new.config import ProcessingConfig, ChunkingConfig, TagConfig

config = ProcessingConfig(
    tags=TagConfig(page_prefix="<page>", page_suffix="</page>"),
    chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=300),
)

processor = DocumentProcessor(config=config)
text = processor.extract_text("report.xlsx")
```

### 4. OCR Integration

```python
from contextifier_new import DocumentProcessor
from contextifier_new.ocr.engines import OpenAIOCREngine

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
contextifier_new/
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
| [ARCHITECTURE.md](contextifier_new/ARCHITECTURE.md) | Internal architecture specification |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

## License

Apache License 2.0 — see [LICENSE](LICENSE)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
