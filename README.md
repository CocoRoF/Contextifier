# Contextifier

**Contextifier** is a document processing library that converts raw documents into AI-understandable context. It analyzes, restructures, and normalizes content so that language models can reason over documents with higher accuracy and consistency.

## Features

- **Multi-format Support**: Process a wide variety of document formats including:
  - PDF (with table detection, OCR fallback, and complex layout handling)
  - Microsoft Office: DOCX, DOC, PPTX, PPT, XLSX, XLS
  - Korean documents: HWP, HWPX (Hangul Word Processor)
  - Text formats: TXT, MD, RTF, CSV, HTML
  - Code files: Python, JavaScript, TypeScript, and 20+ languages

- **Intelligent Text Extraction**: 
  - Preserves document structure (headings, paragraphs, lists)
  - Extracts tables as HTML with proper `rowspan`/`colspan` handling
  - Handles merged cells and complex table layouts
  - Extracts and processes inline images

- **OCR Integration**:
  - Pluggable OCR engine architecture
  - Supports OpenAI, Anthropic, Google Gemini, and vLLM backends
  - Automatic OCR fallback for scanned documents or image-based PDFs

- **Smart Chunking**:
  - Semantic text chunking with configurable size and overlap
  - Table-aware chunking that preserves table integrity
  - Protected regions for code blocks and special content

- **Metadata Extraction**:
  - Extracts document metadata (title, author, creation date, etc.)
  - Formats metadata in a structured, parseable format

## Installation

```bash
pip install contextifier
```

Or using uv:

```bash
uv add contextifier
```

## Quick Start

### Basic Usage

```python
from libs.core.document_processor import DocumentProcessor

# Create processor instance
processor = DocumentProcessor()

# Extract text from a document
text = processor.extract_text("document.pdf")
print(text)

# Chunk the extracted text
chunks = processor.chunk_text(text, chunk_size=1000, chunk_overlap=200)
for chunk in chunks:
    print(chunk)
```

### With OCR Processing

```python
from libs.core.document_processor import DocumentProcessor
from libs.ocr.ocr_engine import OpenAIOCR

# Initialize OCR engine
ocr_engine = OpenAIOCR(api_key="sk-...", model="gpt-4o")

# Create processor with OCR
processor = DocumentProcessor(ocr_engine=ocr_engine)

# Extract text with OCR processing enabled
text = processor.extract_text(
    "scanned_document.pdf",
    ocr_processing=True
)
```

## Supported Formats

| Category | Extensions |
|----------|------------|
| Documents | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.hwp`, `.hwpx` |
| Spreadsheets | `.xlsx`, `.xls`, `.csv`, `.tsv` |
| Text | `.txt`, `.md`, `.rtf` |
| Web | `.html`, `.htm`, `.xml` |
| Code | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, and more |
| Config | `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.env` |

## Architecture

```
libs/
├── core/
│   ├── document_processor.py    # Main entry point
│   ├── processor/               # Format-specific handlers
│   │   ├── pdf_handler.py       # PDF processing with V4 engine
│   │   ├── docx_handler.py      # DOCX processing
│   │   ├── ppt_handler.py       # PowerPoint processing
│   │   ├── excel_handler.py     # Excel processing
│   │   ├── hwp_processor.py     # HWP 5.0 OLE processing
│   │   ├── hwpx_processor.py    # HWPX (ZIP/XML) processing
│   │   └── ...
│   └── functions/
│       └── img_processor.py     # Image handling utilities
├── chunking/
│   ├── chunking.py              # Main chunking interface
│   ├── text_chunker.py          # Text-based chunking
│   ├── table_chunker.py         # Table-aware chunking
│   └── page_chunker.py          # Page-based chunking
└── ocr/
    ├── base.py                  # OCR base class
    ├── ocr_processor.py         # OCR processing utilities
    └── ocr_engine/              # OCR engine implementations
        ├── openai_ocr.py
        ├── anthropic_ocr.py
        ├── gemini_ocr.py
        └── vllm_ocr.py
```

## Requirements

- Python 3.12+
- Required dependencies are automatically installed (see `pyproject.toml`)

### System Dependencies

For full functionality, you may need:

- **Tesseract OCR**: For local OCR fallback
- **LibreOffice**: For DOC/RTF conversion (optional)
- **Poppler**: For PDF image extraction

## Configuration

```python
# Custom configuration
config = {
    "pdf": {
        "extract_images": True,
        "ocr_fallback": True,
    },
    "chunking": {
        "default_size": 1000,
        "default_overlap": 200,
    }
}

processor = DocumentProcessor(config=config)
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
