# Contextifier

Convert raw documents into AI-ready text and chunks, and edit OOXML files losslessly.

Contextifier is a Python library with two views of a document:

- **AI-friendly view** — `DocumentProcessor` runs every supported format through the same 5-stage pipeline (convert, preprocess, extract metadata, extract content, postprocess) and returns normalized text (tables, image tags, page tags, charts, metadata) ready for LLMs and RAG, plus table-aware chunking.
- **Raw view** — `open_raw()` opens `.xlsx` / `.docx` / `.pptx` as an addressable, writable package. Saving keeps every untouched part byte-identical.

## Installation

Requires Python 3.12+.

```bash
pip install contextifier
# PDF support needs PyMuPDF (AGPL-3.0, so it is an explicit extra)
pip install "contextifier[pdf]"
```

`uv add contextifier` works too. Extras: `pdf` (PyMuPDF), `langchain` (LangChain integrations), `server` (pydantic and related), `all` (everything).

Optional system tools: LibreOffice (some legacy `.doc` / `.ppt` / `.xls` / `.rtf` conversion paths), Poppler (`pdf2image`), Tesseract (local OCR engine).

## Quick Start

```python
from contextifier import DocumentProcessor

processor = DocumentProcessor()
text = processor.extract_text("document.pdf")

result = processor.extract_chunks("document.pdf", chunk_size=1000)
for chunk in result.chunks:
    print(chunk[:100])
result.save_to_md("output/chunks")
```

### Fast scan

`extract_text_fast()` returns plain text only (no metadata block, image tags, chart blocks or table reconstruction). Use it when you only need to scan for words or patterns:

```python
text = processor.extract_text_fast("report.pdf")
```

### Raw read/write

```python
from contextifier import open_raw

raw = open_raw("report.xlsx")
raw.sheets["Sales"].set_cell("B3", 142)
raw.charts[0].set_data(categories=["Q1", "Q2"], series=[("Sales", [120, 135])])
raw.save("report-edited.xlsx")

doc = open_raw("paper.docx")
doc.set_paragraph_text(3, "Revised text")   # runs and inline images preserved
doc.tables[0].insert_row(2)

deck = open_raw("deck.pptx")
deck.slides[0].set_text(shape_id=2, new_text="New title")
```

Every raw model also exposes `.package` for part-level OPC access. `.xlsx`, `.docx` and `.pptx` are the supported raw formats.

### Configuration

`ProcessingConfig` is a frozen dataclass; sub-configs are `TagConfig`, `ImageConfig`, `ChartConfig`, `MetadataConfig`, `TableConfig`, `ChunkingConfig`, `OCRConfig` and `EncodingConfig`. Use `with_tags()`, `with_chunking()`, `with_ocr()` and similar to derive modified copies.

```python
from contextifier import DocumentProcessor
from contextifier.config import ProcessingConfig, ChunkingConfig, TagConfig

config = ProcessingConfig(
    tags=TagConfig(page_prefix="<page>", page_suffix="</page>"),
    chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=300),
)
processor = DocumentProcessor(config=config)
```

PDF has two content extractors, selected with `config.with_format_option("pdf", mode="default")` or `mode="plus"` (the default, with advanced table and layout analysis). See [docs/configuration.md](docs/configuration.md) for all options.

### OCR

```python
from contextifier import DocumentProcessor
from contextifier.ocr.engines import OpenAIOCREngine

ocr = OpenAIOCREngine.from_api_key("sk-...", model="gpt-4o")
processor = DocumentProcessor(ocr_engine=ocr)
text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

Engines in `contextifier.ocr.engines`: `OpenAIOCREngine`, `AnthropicOCREngine`, `GeminiOCREngine`, `BedrockOCREngine`, `VLLMOCREngine`, `DeepSeekOCREngine`, and the local `TesseractOCREngine`. See [docs/ocr_guide.md](docs/ocr_guide.md).

## API Overview

| API | Purpose |
|-----|---------|
| `DocumentProcessor` | `extract_text`, `extract_text_fast`, `process` (returns `ExtractionResult`), `extract_chunks` (returns `ChunkResult`), `chunk_text`, `open_raw`, `is_supported`, `supported_extensions` |
| `AsyncDocumentProcessor` | Async wrapper; adds `extract_batch(paths, max_concurrent=4)` |
| `CachedDocumentProcessor` | Wraps `DocumentProcessor` with a pluggable cache (`MemoryCacheBackend` default, `DiskCacheBackend` in `contextifier.cached_processor`) |
| `open_raw` | Lossless, writable access to xlsx / docx / pptx |
| `TextChunker` | Chunking with automatic strategy selection: plain (recursive), table, page-boundary, protected-region |
| `contextifier.integrations.langchain_loader.ContextifierLoader` | LangChain `BaseLoader` (needs the `langchain` extra) |

Encrypted Office files: pass `password=` to `extract_text`, `extract_text_fast`, `process` or `extract_chunks`.

## Supported Formats

`DocumentProcessor().supported_extensions` currently lists 83 extensions:

| Category | Extensions |
|----------|-----------|
| Documents | `pdf`, `docx`, `doc`, `hwp`, `hwpx`, `rtf` |
| Presentations | `pptx`, `ppt` |
| Spreadsheets / data | `xlsx`, `xls`, `csv`, `tsv` |
| Web | `html`, `htm`, `xhtml` |
| Text | `txt`, `md`, `markdown`, `log`, `rst`, and code / config extensions (`py`, `js`, `ts`, `java`, `go`, `rs`, `json`, `yaml`, `toml`, `ini`, `xml`, `env`, ...) |
| Images | `jpg`, `jpeg`, `png`, `gif`, `bmp`, `webp`, `tif`, `tiff`, `heic`, `heif`, `ico`, `svg` (text comes from OCR) |

## Project Layout

```
contextifier/
├── document_processor.py   # DocumentProcessor facade
├── async_processor.py      # AsyncDocumentProcessor
├── cached_processor.py     # CachedDocumentProcessor
├── config.py               # frozen ProcessingConfig and sub-configs
├── handlers/               # per-format handlers + registry (pdf, pdf_plus, docx, hwp, ...)
├── pipeline/               # 5-stage abstract base classes
├── services/               # tag, image, table, chart, metadata, storage
├── chunking/               # TextChunker and strategies
├── ocr/                    # OCR engines and processor
├── raw/                    # open_raw: OPC, xlsx, docx, pptx, chart
└── integrations/           # LangChain loader
```

## Development

```bash
git clone https://github.com/CocoRoF/Contextifier.git
cd Contextifier
uv sync --all-extras          # or: pip install -e ".[all]" pytest pytest-asyncio ruff
pytest tests/ -q
ruff check contextifier/ && ruff format --check contextifier/
```

CI runs ruff and the test suite on Python 3.12 and 3.13.

## Documentation

[QUICKSTART.md](QUICKSTART.md) (usage guide), [Process Logic](Process%20Logic.md), [ARCHITECTURE](contextifier/ARCHITECTURE.md), [CHANGELOG](CHANGELOG.md), [CONTRIBUTING](CONTRIBUTING.md), and in `docs/`: [handler comparison](docs/handler_comparison.md), [configuration](docs/configuration.md), [error codes](docs/error_codes.md), [OCR guide](docs/ocr_guide.md), [plugin development](docs/plugin_development.md).

## Related Projects

[edit2docs](https://github.com/CocoRoF/edit2docs) builds document editing on top of the raw layer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
