# Contextifier v2 — Detailed Usage Guide

A comprehensive guide to all APIs, configuration options, and usage patterns in Contextifier v2.

## Table of Contents

1. [Installation](#1-installation)
2. [Basic Usage](#2-basic-usage)
3. [Configuration System (ProcessingConfig)](#3-configuration-system-processingconfig)
4. [Text Extraction API](#4-text-extraction-api)
5. [Chunking API](#5-chunking-api)
6. [OCR Integration](#6-ocr-integration)
7. [Tag Customization](#7-tag-customization)
8. [Table Processing](#8-table-processing)
9. [Image Processing](#9-image-processing)
10. [Metadata Extraction](#10-metadata-extraction)
11. [Format-Specific Guide](#11-format-specific-guide)
12. [Batch Processing](#12-batch-processing)
13. [RAG Integration](#13-rag-integration)
14. [Error Handling](#14-error-handling)
15. [Full API Reference](#15-full-api-reference)

---

## 1. Installation

### pip

```bash
pip install contextifier
```

### uv

```bash
uv add contextifier
```

### Optional Dependencies

```bash
# Per-engine OCR dependencies (install only what you need)
pip install langchain-openai       # OpenAI OCR
pip install langchain-anthropic    # Anthropic OCR
pip install langchain-google-genai # Google Gemini OCR
pip install langchain-aws          # AWS Bedrock OCR
pip install langchain-community    # vLLM OCR
```

### System Dependencies (Optional)

| Tool | Purpose | Install |
|------|---------|---------|
| LibreOffice | DOC, PPT, RTF → DOCX/PPTX/HTML conversion | [libreoffice.org](https://www.libreoffice.org/) |
| Poppler | PDF → image conversion | `apt install poppler-utils` / `brew install poppler` |

---

## 2. Basic Usage

### 2.1 Simplest Usage

```python
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()

# Extract text
text = processor.extract_text("document.pdf")
print(text)
```

### 2.2 Extract + Chunk in One Step

```python
result = processor.extract_chunks("document.pdf")

print(f"Total {len(result)} chunks")
for i, chunk in enumerate(result, 1):
    print(f"--- Chunk {i} ---")
    print(chunk[:200])
```

### 2.3 Check Supported Extensions

```python
processor = DocumentProcessor()

# Check specific extension
print(processor.is_supported("pdf"))   # True
print(processor.is_supported("xyz"))   # False

# List all supported extensions
print(processor.supported_extensions)
# frozenset({'pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', ...})
```

---

## 3. Configuration System (ProcessingConfig)

All behavior in Contextifier v2 is controlled by `ProcessingConfig`.
Configs are frozen dataclasses — **immutable** after creation. Modification returns a new instance.

### 3.1 Configuration Hierarchy

```
ProcessingConfig (root)
├── TagConfig       — tag prefix/suffix settings
├── ImageConfig     — image save path / format / naming
├── ChartConfig     — chart formatting settings
├── MetadataConfig  — metadata language / format
├── TableConfig     — table output format (HTML / MD / Text)
├── ChunkingConfig  — chunk size / overlap / strategy
├── OCRConfig       — OCR enable / provider
└── format_options  — format-specific additional options
```

### 3.2 Zero Configuration (All Defaults)

```python
from contextifier_new.config import ProcessingConfig

# Uses all default values — works out of the box
config = ProcessingConfig()
processor = DocumentProcessor(config=config)
```

### 3.3 Explicit Construction

```python
from contextifier_new.config import (
    ProcessingConfig,
    TagConfig,
    ImageConfig,
    ChartConfig,
    MetadataConfig,
    TableConfig,
    ChunkingConfig,
    OCRConfig,
)
from contextifier_new.types import NamingStrategy, OutputFormat

config = ProcessingConfig(
    tags=TagConfig(
        page_prefix="<page>",
        page_suffix="</page>",
        image_prefix="[IMG:",
        image_suffix="]",
    ),
    images=ImageConfig(
        directory_path="output/images",
        naming_strategy=NamingStrategy.HASH,
        quality=95,
    ),
    tables=TableConfig(
        output_format=OutputFormat.HTML,
        preserve_merged_cells=True,
    ),
    chunking=ChunkingConfig(
        chunk_size=2000,
        chunk_overlap=300,
        preserve_tables=True,
    ),
    metadata=MetadataConfig(
        language="en",
    ),
)

processor = DocumentProcessor(config=config)
```

### 3.4 Fluent Builder Pattern

Use `with_*()` methods to create a modified copy while keeping the original unchanged.

```python
config = ProcessingConfig()

# Modify only tags
config2 = config.with_tags(page_prefix="<!-- Page ", page_suffix=" -->")

# Modify only chunking
config3 = config.with_chunking(chunk_size=3000, chunk_overlap=500)

# Chain multiple modifications
config4 = (
    config
    .with_tags(page_prefix="[P:", page_suffix="]")
    .with_chunking(chunk_size=1500)
    .with_tables(output_format=OutputFormat.MARKDOWN)
    .with_images(directory_path="my_images/")
)
```

### 3.5 Format-Specific Options

Set options that apply only to a specific format.

```python
config = ProcessingConfig().with_format_option(
    "pdf",
    table_detection="lattice",
    ocr_fallback=True,
)

# Retrieve option
val = config.get_format_option("pdf", "table_detection", default="stream")
```

### 3.6 Serialization / Deserialization

```python
# Convert to dict (for JSON storage)
d = config.to_dict()

import json
with open("config.json", "w") as f:
    json.dump(d, f, indent=2)

# Restore from dict
with open("config.json") as f:
    d = json.load(f)

config = ProcessingConfig.from_dict(d)
```

### 3.7 Config Class Defaults Reference

| Config Class | Key Fields | Defaults |
|-------------|-----------|----------|
| **TagConfig** | `page_prefix` / `page_suffix` | `"[Page Number: "` / `"]"` |
| | `slide_prefix` / `slide_suffix` | `"[Slide Number: "` / `"]"` |
| | `sheet_prefix` / `sheet_suffix` | `"[Sheet: "` / `"]"` |
| | `image_prefix` / `image_suffix` | `"[Image:"` / `"]"` |
| | `chart_prefix` / `chart_suffix` | `"[chart]"` / `"[/chart]"` |
| | `metadata_prefix` / `metadata_suffix` | `"<Document-Metadata>"` / `"</Document-Metadata>"` |
| **ImageConfig** | `directory_path` | `"temp/images"` |
| | `naming_strategy` | `NamingStrategy.HASH` |
| | `default_format` | `"png"` |
| | `quality` | `95` |
| | `skip_duplicate` | `True` |
| **ChartConfig** | `use_html_table` | `True` |
| | `include_chart_type` | `True` |
| | `include_chart_title` | `True` |
| **MetadataConfig** | `language` | `"ko"` |
| | `date_format` | `"%Y-%m-%d %H:%M:%S"` |
| **TableConfig** | `output_format` | `OutputFormat.HTML` |
| | `clean_whitespace` | `True` |
| | `preserve_merged_cells` | `True` |
| **ChunkingConfig** | `chunk_size` | `1000` |
| | `chunk_overlap` | `200` |
| | `preserve_tables` | `True` |
| | `strategy` | `"recursive"` |
| **OCRConfig** | `enabled` | `False` |
| | `provider` | `None` |
| | `prompt` | `None` (uses built-in default prompt) |

---

## 4. Text Extraction API

### 4.1 `extract_text()` — Returns Text String

```python
text = processor.extract_text(
    file_path="document.pdf",        # file path (required)
    file_extension=None,             # extension override (optional)
    extract_metadata=True,           # include metadata in output
    ocr_processing=False,            # apply OCR post-processing
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str \| Path` | — | Path to the document file |
| `file_extension` | `str \| None` | `None` | Extension override (e.g., `.pdf` or `pdf`) |
| `extract_metadata` | `bool` | `True` | Whether to include metadata in the output text |
| `ocr_processing` | `bool` | `False` | Whether to apply OCR on image tags |

**Returns:** `str` — extracted text

**Extension override example:**

```python
# Process a file with no extension as PDF
text = processor.extract_text("report_v2", file_extension="pdf")

# Process a .dat file as Excel
text = processor.extract_text("data.dat", file_extension="xlsx")
```

### 4.2 `process()` — Returns Structured Result

Returns an `ExtractionResult` containing text, metadata, tables, images, and charts.

```python
from contextifier_new.types import ExtractionResult

result: ExtractionResult = processor.process("document.pdf")

print(result.text)            # Full extracted text
print(result.metadata)        # DocumentMetadata object
print(result.tables)          # List of TableData
print(result.images)          # List of image paths
print(result.charts)          # List of ChartData
```

**`ExtractionResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Full extracted text |
| `metadata` | `DocumentMetadata \| None` | Document metadata |
| `tables` | `List[TableData]` | Extracted tables |
| `images` | `List[str]` | Saved image file paths |
| `charts` | `List[ChartData]` | Extracted charts |

---

## 5. Chunking API

### 5.1 `chunk_text()` — Split Text into Chunks

Directly chunk an already-extracted text string.

```python
text = processor.extract_text("document.pdf")

# Default chunking (uses config settings)
chunks = processor.chunk_text(text)
print(f"{len(chunks)} chunks")

# Override size/overlap
chunks = processor.chunk_text(
    text,
    chunk_size=2000,
    chunk_overlap=300,
)

# Include position metadata (returns Chunk objects)
chunks_with_meta = processor.chunk_text(
    text,
    include_position_metadata=True,
)
for chunk in chunks_with_meta:
    print(f"Index: {chunk.metadata.chunk_index}, Text: {chunk.text[:50]}...")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | Text to split |
| `chunk_size` | `int \| None` | `None` (config value) | Maximum characters per chunk |
| `chunk_overlap` | `int \| None` | `None` (config value) | Overlap characters between chunks |
| `file_extension` | `str` | `""` | Source file extension (used for strategy selection) |
| `preserve_tables` | `bool` | `True` | Whether to preserve table structures |
| `include_position_metadata` | `bool` | `False` | Whether to include position metadata |

**Returns:**
- `include_position_metadata=False` → `List[str]`
- `include_position_metadata=True` → `List[Chunk]`

### 5.2 `extract_chunks()` — Extract + Chunk in One Step

Extracts text from a file and immediately chunks it. **The most commonly used API.**

```python
result = processor.extract_chunks(
    file_path="document.pdf",
    chunk_size=1500,
    chunk_overlap=200,
    preserve_tables=True,
    include_position_metadata=True,
)

# Basic ChunkResult usage
print(f"Total {len(result)} chunks")
print(result[0])          # Index access
for chunk in result:      # Iteration
    print(chunk[:100])

# Access position metadata
if result.has_metadata:
    for chunk_obj in result.chunks_with_metadata:
        print(f"  chunk_index={chunk_obj.metadata.chunk_index}")
        print(f"  page={chunk_obj.metadata.page_number}")

# Source file info
print(result.source_file)
```

### 5.3 `ChunkResult.save_to_md()` — Save as Markdown Files

Saves each chunk as an individual Markdown file. Useful for RAG indexing.

```python
result = processor.extract_chunks("report.pdf", chunk_size=1000)

# Default save
created = result.save_to_md("output/chunks")
print(f"{len(created)} files created")
# output/chunks/chunk_0001.md, chunk_0002.md, ...

# Custom filename prefix and separator
created = result.save_to_md(
    "output/report_chunks",
    filename_prefix="report",
    separator="===",
)
# output/report_chunks/report_0001.md, report_0002.md, ...
```

**Example saved file (with metadata):**

```markdown
<!-- chunk_index: 0 -->
<!-- page: 1 -->
---

First chunk text content...
```

### 5.4 Chunking Strategies

Contextifier v2 **automatically selects the optimal strategy** based on content characteristics.

| Strategy | Priority | Trigger Condition | Description |
|----------|----------|-------------------|-------------|
| **TableChunkingStrategy** | 5 (highest) | Spreadsheet files (xlsx, xls, csv, tsv) | Sheet/table boundary-based splitting |
| **PageChunkingStrategy** | 10 | Page tags present | Page boundary-aware splitting |
| **ProtectedChunkingStrategy** | 20 | HTML tables / protected regions present | Protected region preservation |
| **PlainChunkingStrategy** | 100 (lowest) | Default fallback | Recursive character splitting |

> **Lower** priority number = **higher** precedence. If a strategy can't handle the content, it falls through to the next.

---

## 6. OCR Integration

Contextifier v2 supports 5 Vision LLM-based OCR engines. They process image tags (`[Image:...]`) in extracted text and replace them with recognized text.

### 6.1 OpenAI

```python
from contextifier_new.ocr.engines import OpenAIOCREngine

ocr = OpenAIOCREngine.from_api_key(
    "sk-...",
    model="gpt-4o",        # default
    temperature=0.0,       # default
    max_tokens=None,       # optional
)

processor = DocumentProcessor(ocr_engine=ocr)
text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

### 6.2 Anthropic

```python
from contextifier_new.ocr.engines import AnthropicOCREngine

ocr = AnthropicOCREngine.from_api_key(
    "sk-ant-...",
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.3 Google Gemini

```python
from contextifier_new.ocr.engines import GeminiOCREngine

ocr = GeminiOCREngine.from_api_key(
    "AIza...",
    model="gemini-2.0-flash",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.4 AWS Bedrock

```python
from contextifier_new.ocr.engines import BedrockOCREngine

ocr = BedrockOCREngine.from_api_key(
    "AKIA...",                          # AWS Access Key
    aws_secret_access_key="...",
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.5 vLLM (Local / Self-Hosted)

```python
from contextifier_new.ocr.engines import VLLMOCREngine

ocr = VLLMOCREngine.from_api_key(
    "dummy-key",                  # vLLM doesn't require auth
    model="llava-1.5-7b",
    base_url="http://localhost:8000/v1",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.6 Pass Existing LangChain Client

If you already have a LangChain client, pass it directly:

```python
from langchain_openai import ChatOpenAI
from contextifier_new.ocr.engines import OpenAIOCREngine

llm = ChatOpenAI(model="gpt-4o", api_key="sk-...")
ocr = OpenAIOCREngine(llm_client=llm)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.7 Custom OCR Prompt

```python
custom_prompt = """
Extract all text from this image.
If tables are present, convert them to HTML tables.
"""

ocr = OpenAIOCREngine.from_api_key(
    "sk-...",
    prompt=custom_prompt,
)

# Or modify after creation
ocr.prompt = custom_prompt
```

---

## 7. Tag Customization

Contextifier inserts structural tags into extracted text. Tag formats are fully customizable via `TagConfig`.

### 7.1 Default Tag Output Example

```
<Document-Metadata>
  Title: Annual Report 2024
  Author: John Doe
  Created: 2024-01-15 10:30:00
</Document-Metadata>

[Page Number: 1]

Hello, this is the first page.

[Image: temp/images/abc123.png]

[chart]
<table>
  <tr><th>Quarter</th><th>Revenue</th></tr>
  <tr><td>Q1</td><td>$100M</td></tr>
</table>
[/chart]

[Page Number: 2]
...
```

### 7.2 XML-Style Tags

```python
config = ProcessingConfig(
    tags=TagConfig(
        page_prefix="<page number=\"",
        page_suffix="\">",
        image_prefix="<image src=\"",
        image_suffix="\" />",
        metadata_prefix="<metadata>",
        metadata_suffix="</metadata>",
        chart_prefix="<chart>",
        chart_suffix="</chart>",
    ),
)
```

Output:
```
<metadata>
  ...
</metadata>
<page number="1">
Text...
<image src="temp/images/abc123.png" />
```

### 7.3 Markdown-Style Tags

```python
config = ProcessingConfig(
    tags=TagConfig(
        page_prefix="## Page ",
        page_suffix="",
        image_prefix="![image](",
        image_suffix=")",
    ),
)
```

---

## 8. Table Processing

### 8.1 Output Format Selection

```python
from contextifier_new.config import ProcessingConfig, TableConfig
from contextifier_new.types import OutputFormat

# HTML (default) — full merged cell and structure preservation
config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.HTML))

# Markdown — simple tables (no merged cell support)
config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.MARKDOWN))

# Plain Text — aligned text table
config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.TEXT))
```

### 8.2 HTML Output Example

```html
<table>
  <tr>
    <th colspan="2">2024 Revenue Summary</th>
  </tr>
  <tr>
    <td>Q1</td>
    <td>$100M</td>
  </tr>
  <tr>
    <td>Q2</td>
    <td>$120M</td>
  </tr>
</table>
```

### 8.3 Markdown Output Example

```markdown
| Quarter | Revenue |
|---------|---------|
| Q1 | $100M |
| Q2 | $120M |
```

---

## 9. Image Processing

### 9.1 Image Extraction Settings

```python
from contextifier_new.config import ProcessingConfig, ImageConfig
from contextifier_new.types import NamingStrategy, StorageType

config = ProcessingConfig(
    images=ImageConfig(
        directory_path="output/images",        # Save path
        naming_strategy=NamingStrategy.HASH,    # Filename: content hash (dedup)
        default_format="png",                  # Save format
        quality=95,                            # JPEG quality (1-100)
        skip_duplicate=True,                   # Skip identical images
        storage_type=StorageType.LOCAL,         # Local storage
    ),
)
```

### 9.2 Naming Strategies

| Strategy | Filename Example | Description |
|----------|-----------------|-------------|
| `HASH` | `a3f2c1d8.png` | Content hash — automatic deduplication |
| `UUID` | `550e8400-e29b.png` | Random UUID |
| `SEQUENTIAL` | `img_001.png` | Sequential numbering |
| `TIMESTAMP` | `20240115_103000.png` | Timestamp-based |

---

## 10. Metadata Extraction

### 10.1 Usage

```python
# Extract text with metadata included (default)
text = processor.extract_text("report.docx", extract_metadata=True)

# Exclude metadata
text = processor.extract_text("report.docx", extract_metadata=False)

# Access structured metadata
result = processor.process("report.docx")
meta = result.metadata

print(meta.title)           # "Annual Report 2024"
print(meta.author)          # "John Doe"
print(meta.create_time)     # datetime(2024, 1, 15, 10, 30, 0)
print(meta.page_count)      # 42
print(meta.to_dict())       # As dictionary
```

### 10.2 Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Document title |
| `subject` | `str` | Subject |
| `author` | `str` | Author |
| `keywords` | `str` | Keywords |
| `comments` | `str` | Description / comments |
| `last_saved_by` | `str` | Last modified by |
| `create_time` | `datetime` | Creation date/time |
| `last_saved_time` | `datetime` | Last saved date/time |
| `page_count` | `int` | Total page count |
| `word_count` | `int` | Total word count |
| `category` | `str` | Document category |
| `revision` | `str` | Revision number |

### 10.3 Metadata Language Setting

```python
# Korean (default)
config = ProcessingConfig(metadata=MetadataConfig(language="ko"))
# Output: "제목: ...", "작성자: ...", "생성일: ..."

# English
config = ProcessingConfig(metadata=MetadataConfig(language="en"))
# Output: "Title: ...", "Author: ...", "Created: ..."
```

---

## 11. Format-Specific Guide

### 11.1 PDF

```python
# Default PDF handler (suitable for most PDFs)
text = processor.extract_text("document.pdf")

# Image-based / scanned PDF (requires OCR)
ocr = OpenAIOCREngine.from_api_key("sk-...")
processor = DocumentProcessor(ocr_engine=ocr)
text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

### 11.2 Word Documents (DOCX / DOC)

```python
# DOCX — processed directly
text = processor.extract_text("document.docx")

# DOC — auto-converted via LibreOffice then processed
text = processor.extract_text("legacy.doc")
```

> **Note:** DOC files can contain various internal formats (OLE, HTML, DOCX, RTF). Contextifier auto-detects the format and uses the optimal conversion method.

### 11.3 PowerPoint (PPTX / PPT)

```python
text = processor.extract_text("presentation.pptx")
# Includes [Slide Number: N] tags for each slide
# Extracts notes, charts, and tables
```

### 11.4 Excel (XLSX / XLS)

```python
text = processor.extract_text("data.xlsx")
# Includes [Sheet: SheetName] tags for each sheet
# Includes chart data in [chart]...[/chart] tags
```

### 11.5 HWP / HWPX (Korean Word Processor)

```python
text = processor.extract_text("document.hwp")   # HWP 5.0
text = processor.extract_text("document.hwpx")  # HWPX
```

> **Note:** Only HWP 5.0 format is supported. Legacy HWP 3.0 is not supported.

### 11.6 CSV / TSV

```python
text = processor.extract_text("data.csv")
text = processor.extract_text("data.tsv")
# Converted to table format
```

### 11.7 Text / Code / Config Files

```python
# Auto encoding detection (UTF-8, EUC-KR, etc.)
text = processor.extract_text("readme.txt")
text = processor.extract_text("script.py")
text = processor.extract_text("config.yaml")
```

### 11.8 Images

```python
# Image files require an OCR engine
ocr = OpenAIOCREngine.from_api_key("sk-...")
processor = DocumentProcessor(ocr_engine=ocr)

text = processor.extract_text("chart.png", ocr_processing=True)
```

---

## 12. Batch Processing

### 12.1 Process All Files in a Directory

```python
from pathlib import Path
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()

input_dir = Path("documents/")
output_dir = Path("output/")
output_dir.mkdir(exist_ok=True)

for file_path in input_dir.iterdir():
    ext = file_path.suffix.lstrip(".")
    if not processor.is_supported(ext):
        continue

    try:
        text = processor.extract_text(str(file_path))
        out_path = output_dir / f"{file_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"OK  {file_path.name}")
    except Exception as e:
        print(f"ERR {file_path.name}: {e}")
```

### 12.2 Batch Chunking

```python
processor = DocumentProcessor()

files = ["report.pdf", "data.xlsx", "memo.docx"]

all_chunks = []
for f in files:
    result = processor.extract_chunks(f, chunk_size=1000)
    for chunk in result.chunks:
        all_chunks.append({
            "text": chunk,
            "source": f,
        })

print(f"Total {len(all_chunks)} chunks generated")
```

---

## 13. RAG Integration

### 13.1 LangChain Integration

```python
from contextifier_new import DocumentProcessor
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

processor = DocumentProcessor()

# Process document + chunk
result = processor.extract_chunks(
    "knowledge_base.pdf",
    chunk_size=1000,
    chunk_overlap=200,
    include_position_metadata=True,
)

# Convert to LangChain Documents
documents = []
for chunk_obj in result.chunks_with_metadata:
    doc = Document(
        page_content=chunk_obj.text,
        metadata={
            "source": result.source_file,
            "chunk_index": chunk_obj.metadata.chunk_index,
            "page_number": chunk_obj.metadata.page_number,
        },
    )
    documents.append(doc)

# Index into vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Search
results = vectorstore.similarity_search("revenue overview", k=3)
```

### 13.2 Save-to-File Indexing

```python
# Save as Markdown files → feed into external indexing pipeline
result = processor.extract_chunks("report.pdf", chunk_size=1000)
result.save_to_md("rag_index/report/")
```

---

## 14. Error Handling

### 14.1 Exception Hierarchy

```
ContextifierError (base exception)
├── FileNotFoundError        — file does not exist
├── UnsupportedFormatError   — unsupported file format
├── HandlerNotFoundError     — no handler found for extension
├── ConversionError          — file conversion failed
├── ExtractionError          — text extraction failed
└── OCRError                 — OCR processing failed
```

### 14.2 Usage Example

```python
from contextifier_new.errors import (
    ContextifierError,
    UnsupportedFormatError,
    FileNotFoundError as ContextifyFileNotFoundError,
)

try:
    text = processor.extract_text("document.xyz")
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e}")
except ContextifyFileNotFoundError as e:
    print(f"File not found: {e}")
except ContextifierError as e:
    print(f"Processing error: {e}")
```

---

## 15. Full API Reference

### DocumentProcessor

```python
class DocumentProcessor:
    def __init__(
        self,
        config: ProcessingConfig | None = None,
        *,
        ocr_engine: BaseOCREngine | None = None,
    ) -> None: ...

    def extract_text(
        self,
        file_path: str | Path,
        file_extension: str | None = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs,
    ) -> str: ...

    def process(
        self,
        file_path: str | Path,
        file_extension: str | None = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs,
    ) -> ExtractionResult: ...

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        file_extension: str = "",
        preserve_tables: bool = True,
        include_position_metadata: bool = False,
    ) -> list[str] | list[Chunk]: ...

    def extract_chunks(
        self,
        file_path: str | Path,
        file_extension: str | None = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        preserve_tables: bool = True,
        include_position_metadata: bool = False,
        **kwargs,
    ) -> ChunkResult: ...

    def is_supported(self, extension: str) -> bool: ...

    @property
    def supported_extensions(self) -> frozenset: ...

    @property
    def config(self) -> ProcessingConfig: ...

    @property
    def registry(self) -> HandlerRegistry: ...
```

### ChunkResult

```python
@dataclass
class ChunkResult:
    chunks: list[str]
    chunks_with_metadata: list[Chunk] | None
    source_file: str | None

    @property
    def has_metadata(self) -> bool: ...

    def save_to_md(
        self,
        output_dir: str | Path,
        *,
        filename_prefix: str = "chunk",
        separator: str = "---",
    ) -> list[str]: ...

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> str: ...
    def __iter__(self) -> Iterator[str]: ...
```

### ProcessingConfig

```python
@dataclass(frozen=True)
class ProcessingConfig:
    tags: TagConfig
    images: ImageConfig
    charts: ChartConfig
    metadata: MetadataConfig
    tables: TableConfig
    chunking: ChunkingConfig
    ocr: OCRConfig
    format_options: dict[str, dict[str, Any]]

    def with_tags(self, **kwargs) -> ProcessingConfig: ...
    def with_images(self, **kwargs) -> ProcessingConfig: ...
    def with_charts(self, **kwargs) -> ProcessingConfig: ...
    def with_metadata(self, **kwargs) -> ProcessingConfig: ...
    def with_tables(self, **kwargs) -> ProcessingConfig: ...
    def with_chunking(self, **kwargs) -> ProcessingConfig: ...
    def with_ocr(self, **kwargs) -> ProcessingConfig: ...
    def with_format_option(self, format_name: str, **kwargs) -> ProcessingConfig: ...
    def get_format_option(self, format_name: str, key: str, default=None) -> Any: ...

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> ProcessingConfig: ...
```

### OCR Engines

```python
# Common interface
class BaseOCREngine(ABC):
    def __init__(self, llm_client, *, prompt=None): ...
    def convert_image_to_text(self, image_path: str) -> str | None: ...

# Per-engine convenience constructors
OpenAIOCREngine.from_api_key(api_key, *, model="gpt-4o", ...)
AnthropicOCREngine.from_api_key(api_key, *, model="claude-sonnet-4-20250514", ...)
GeminiOCREngine.from_api_key(api_key, *, model="gemini-2.0-flash", ...)
BedrockOCREngine.from_api_key(api_key, *, aws_secret_access_key, region_name, model, ...)
VLLMOCREngine.from_api_key(api_key, *, model, base_url, ...)
```
