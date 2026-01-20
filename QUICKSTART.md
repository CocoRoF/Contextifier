# Quick Start Guide

A comprehensive guide to get started with **Contextifier**.

## Table of Contents

1. [Installation](#installation)
2. [Processing Pipeline Overview](#processing-pipeline-overview)
3. [Basic Usage](#basic-usage)
4. [OCR Configuration](#ocr-configuration)
5. [Tag Customization](#tag-customization)
6. [Supported Formats](#supported-formats)
7. [Configuration Options](#configuration-options)
8. [Common Use Cases](#common-use-cases)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Installation

```bash
pip install contextifier
```

Or using uv:

```bash
uv add contextifier
```

---

## Processing Pipeline Overview

Contextifier processes documents through a multi-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXTIFIER PROCESSING PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐          ┌──────────────┐          ┌──────────────┐
    │   FILE   │  ──────► │     TEXT     │  ──────► │    CHUNKS    │
    │ (Input)  │  Stage 1 │ (with tags)  │  Stage 3 │   (Output)   │
    └──────────┘          └──────────────┘          └──────────────┘
                                 │
                                 │ Stage 2 (Optional)
                                 ▼
                          ┌──────────────┐
                          │  OCR TEXT    │
                          │ (processed)  │
                          └──────────────┘
```

### Stage 1: File → Text (Extract)

Documents are converted to text with embedded tags for images:

```
Input: document.pdf (with images, tables, text)
                    ↓
Output: "Document content...\n[Image:temp/images/abc123.png]\nMore text..."
```

**What happens:**
- Text content is extracted from the document
- Images are saved to disk and replaced with `[Image:path]` tags
- Tables are converted to HTML format
- Metadata is extracted (author, date, title, etc.)
- Page/Slide/Sheet markers are added

### Stage 2: Image Tags → OCR Text (Optional)

If OCR is enabled, image tags are converted to actual text:

```
Input: "...[Image:temp/images/abc123.png]..."
                    ↓
Output: "...[Figure:표 1. 매출 현황\n<table>...</table>]..."
```

**What happens:**
- Image tags are detected using configurable patterns
- Each image is sent to the OCR engine (OpenAI, Anthropic, Gemini, etc.)
- OCR results replace the original image tags
- Tables in images are converted to HTML format

### Stage 3: Text → Chunks (Chunk)

The final text is split into manageable chunks:

```
Input: "Long document text with multiple pages..."
                    ↓
Output: ["Chunk 1...", "Chunk 2...", "Chunk 3...", ...]
```

**What happens:**
- Text is split based on chunk_size and chunk_overlap
- Tables and protected regions are preserved
- Page number metadata can be included
- Position metadata (line numbers, character positions) available

---

## Basic Usage

### 1. Simple Text Extraction (Stage 1 only)

```python
from contextifier import DocumentProcessor

# Create processor instance
processor = DocumentProcessor()

# Extract text from any supported document
# Images become [Image:path] tags
text = processor.extract_text("document.pdf")
print(text)
```

**Output example:**
```
<Document-Metadata>
  Title: Annual Report 2025
  Author: John Doe
  Created: 2025-01-15
</Document-Metadata>

[Page Number: 1]

Executive Summary

This report presents our annual performance...

[Image:temp/images/a1b2c3d4.png]

The chart above shows quarterly revenue growth...

[Page Number: 2]
...
```

### 2. Full Pipeline: File → Chunks (All Stages)

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR

# Create OCR engine for image processing
ocr_engine = OpenAIOCR(api_key="your-api-key", model="gpt-4o")

# Create processor with OCR support
processor = DocumentProcessor(ocr_engine=ocr_engine)

# Full pipeline: Extract → OCR → Chunk
result = processor.extract_chunks(
    "document.pdf",
    chunk_size=1000,
    chunk_overlap=200,
    ocr_processing=True  # Enable Stage 2 (OCR)
)

# Access chunks
for i, chunk in enumerate(result.chunks):
    print(f"Chunk {i + 1}: {len(chunk)} characters")
```

### 3. Step-by-Step Processing

For more control, process each stage separately:

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR

# Setup
ocr_engine = OpenAIOCR(api_key="your-api-key", model="gpt-4o")
processor = DocumentProcessor(ocr_engine=ocr_engine)

# Stage 1: Extract text (images become tags)
text_with_tags = processor.extract_text(
    "document.pdf",
    extract_metadata=True,
    ocr_processing=False  # Don't process images yet
)
print("=== Stage 1: Text with Image Tags ===")
print(text_with_tags[:500])

# Stage 2: Process image tags with OCR
text_ocred = processor.extract_text(
    "document.pdf",
    ocr_processing=True  # Now process images
)
print("\n=== Stage 2: Text with OCR Results ===")
print(text_ocred[:500])

# Stage 3: Split into chunks
chunks = processor.chunk_text(
    text_ocred,
    chunk_size=1000,
    chunk_overlap=200
)
print(f"\n=== Stage 3: {len(chunks)} Chunks ===")
```

---

## OCR Configuration

### Setting Up OCR Engine

OCR is required for:
1. Converting embedded images in documents to text
2. Processing standalone image files (jpg, png, etc.)

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR

# Initialize OCR engine
ocr_engine = OpenAIOCR(
    api_key="your-openai-api-key",
    model="gpt-4o",          # or "gpt-4o-mini" for cost savings
    temperature=0.0,          # Deterministic output
    max_tokens=4096           # Maximum response length
)

# Create processor with OCR
processor = DocumentProcessor(ocr_engine=ocr_engine)
```

### Available OCR Engines

```python
# OpenAI GPT-4 Vision (Recommended)
from contextifier.ocr.ocr_engine import OpenAIOCR
engine = OpenAIOCR(api_key="sk-...", model="gpt-4o")

# Anthropic Claude Vision
from contextifier.ocr.ocr_engine import AnthropicOCR
engine = AnthropicOCR(api_key="sk-ant-...", model="claude-3-5-sonnet-20241022")

# Google Gemini Vision
from contextifier.ocr.ocr_engine import GeminiOCR
engine = GeminiOCR(api_key="...", model="gemini-1.5-pro")

# AWS Bedrock (Claude on AWS)
from contextifier.ocr.ocr_engine import BedrockOCR
engine = BedrockOCR(
    aws_access_key_id="AKIA...",
    aws_secret_access_key="...",
    aws_region="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0"
)

# AWS Bedrock with environment credentials
engine = BedrockOCR(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

# AWS Bedrock with temporary credentials (session token)
engine = BedrockOCR(
    aws_access_key_id="ASIA...",
    aws_secret_access_key="...",
    aws_session_token="...",
    aws_region="ap-northeast-2"
)

# vLLM (Self-hosted)
from contextifier.ocr.ocr_engine import VllmOCR
engine = VllmOCR(base_url="http://localhost:8000", model="llava-1.5-7b")
```

### Processing Standalone Image Files

Image files (jpg, png, gif, etc.) are processed differently:

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR

# Without OCR: Returns image tag
processor = DocumentProcessor()
result = processor.extract_text("chart.png")
# Output: "[Image:C:/path/to/chart.png]"

# With OCR: Returns extracted text
processor = DocumentProcessor(ocr_engine=OpenAIOCR(api_key="...", model="gpt-4o"))
result = processor.extract_text("chart.png")
# Output: "[Figure:매출 추이 차트\n2024년 1분기: 100억\n2024년 2분기: 120억...]"
```

### OCR Processing Modes

```python
# Mode 1: OCR during extraction (recommended)
text = processor.extract_text("document.pdf", ocr_processing=True)

# Mode 2: OCR during chunk extraction
result = processor.extract_chunks("document.pdf", ocr_processing=True)

# Mode 3: Manual OCR processing
text_with_tags = processor.extract_text("document.pdf", ocr_processing=False)
# Later, if you have the OCR engine:
ocred_text = ocr_engine.process_text(text_with_tags)
```

### Setting OCR Engine After Initialization

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR

# Create processor without OCR
processor = DocumentProcessor()

# Extract text (images remain as tags)
text = processor.extract_text("document.pdf")
# Contains: [Image:temp/images/abc.png]

# Later, add OCR engine
processor.ocr_engine = OpenAIOCR(api_key="...", model="gpt-4o")

# Now extract with OCR
text_with_ocr = processor.extract_text("document.pdf", ocr_processing=True)
# Images are now converted to text
```

---

## Tag Customization

### Image Tag Format

Configure how extracted images are referenced:

```python
# Default format: [Image:path/to/image.png]
processor = DocumentProcessor()

# HTML format: <img src='path/to/image.png'/>
processor = DocumentProcessor(
    image_directory="output/images",
    image_tag_prefix="<img src='",
    image_tag_suffix="'/>"
)

# Markdown format: ![image](path/to/image.png)
processor = DocumentProcessor(
    image_tag_prefix="![image](",
    image_tag_suffix=")"
)

# Custom XML format: <image path="..."/>
processor = DocumentProcessor(
    image_tag_prefix='<image path="',
    image_tag_suffix='"/>'
)
```

**Important:** When using custom image tag formats, the OCR engine automatically uses the same pattern to detect and process images.

### Page/Slide/Sheet Tag Format

Configure page number markers:

```python
# Default format: [Page Number: 1]
processor = DocumentProcessor()

# XML format: <page>1</page>
processor = DocumentProcessor(
    page_tag_prefix="<page>",
    page_tag_suffix="</page>"
)

# Markdown comment: <!-- Page 1 -->
processor = DocumentProcessor(
    page_tag_prefix="<!-- Page ",
    page_tag_suffix=" -->"
)

# For presentations (slides)
processor = DocumentProcessor(
    slide_tag_prefix="<slide>",
    slide_tag_suffix="</slide>"
)
```

### Complete Custom Configuration

```python
processor = DocumentProcessor(
    # OCR Engine
    ocr_engine=OpenAIOCR(api_key="...", model="gpt-4o"),
    
    # Image settings
    image_directory="output/extracted_images",
    image_tag_prefix="{{IMAGE:",
    image_tag_suffix="}}",
    
    # Page tag settings
    page_tag_prefix="{{PAGE:",
    page_tag_suffix="}}",
    slide_tag_prefix="{{SLIDE:",
    slide_tag_suffix="}}"
)

# Extracted text will use custom format:
# {{PAGE:1}}
# Document content...
# {{IMAGE:output/extracted_images/abc123.png}}
# More content...
```

---

## Supported Formats

| Category | Extensions | Features |
|----------|------------|----------|
| **PDF** | `.pdf` | Table detection, image extraction, OCR fallback, complex layouts |
| **Word** | `.docx`, `.doc` | Tables, images, charts, styles, metadata |
| **Excel** | `.xlsx`, `.xls` | Multiple sheets, formulas, charts, images |
| **PowerPoint** | `.pptx`, `.ppt` | Slides, notes, embedded objects, images |
| **Hangul** | `.hwp`, `.hwpx` | Korean word processor (full support) |
| **Text** | `.txt`, `.md`, `.rtf` | Plain text, Markdown, Rich Text |
| **Web** | `.html`, `.htm` | HTML documents |
| **Data** | `.csv`, `.tsv`, `.json` | Structured data formats |
| **Code** | `.py`, `.js`, `.java`, etc. | 20+ programming languages |
| **Config** | `.yaml`, `.toml`, `.ini` | Configuration files |
| **Images** | `.jpg`, `.png`, `.gif`, `.bmp`, `.webp` | Requires OCR engine for text extraction |

### Image File Processing

```python
# Images require OCR engine for meaningful text extraction
processor = DocumentProcessor(ocr_engine=ocr_engine)

# Supported image formats
image_text = processor.extract_text("screenshot.png")  # PNG
image_text = processor.extract_text("photo.jpg")       # JPEG
image_text = processor.extract_text("diagram.gif")     # GIF
image_text = processor.extract_text("scan.bmp")        # BMP
image_text = processor.extract_text("chart.webp")      # WebP
```

---

## Configuration Options

### Chunk Size and Overlap

```python
result = processor.extract_chunks(
    "document.pdf",
    chunk_size=1000,      # Target size in characters
    chunk_overlap=200,    # Overlap for context continuity
    preserve_tables=True  # Keep tables intact (default)
)
```

**Guidelines:**
- **RAG/Semantic Search:** chunk_size=500-1000, overlap=100-200
- **Summarization:** chunk_size=2000-4000, overlap=200-400
- **LLM Context:** chunk_size=4000-8000, overlap=500-1000

### Position Metadata

Get detailed position information for each chunk:

```python
result = processor.extract_chunks(
    "document.pdf",
    chunk_size=1000,
    include_position_metadata=True
)

# Access metadata
if result.has_metadata:
    for chunk_data in result.chunks_with_metadata:
        print(f"Page: {chunk_data['page_number']}")
        print(f"Lines: {chunk_data['line_start']}-{chunk_data['line_end']}")
        print(f"Position: {chunk_data['global_start']}-{chunk_data['global_end']}")
        print(f"Text: {chunk_data['text'][:100]}...")
```

### Table Preservation

```python
# Preserve table structure during chunking (recommended)
result = processor.extract_chunks("report.pdf", preserve_tables=True)

# Force chunking even through tables (may break table structure)
result = processor.extract_chunks("report.pdf", preserve_tables=False)
```

---

## Common Use Cases

### Building a RAG System

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR
import chromadb

# Setup with OCR for complete content extraction
ocr_engine = OpenAIOCR(api_key="...", model="gpt-4o")
processor = DocumentProcessor(ocr_engine=ocr_engine)

# Vector database
client = chromadb.Client()
collection = client.create_collection("knowledge_base")

# Process document with full OCR
result = processor.extract_chunks(
    "knowledge_base.pdf",
    chunk_size=500,
    chunk_overlap=100,
    ocr_processing=True  # Convert images to text
)

# Index all chunks
for i, chunk in enumerate(result.chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"],
        metadatas=[{
            "source": result.source_file,
            "chunk_index": i
        }]
    )

print(f"Indexed {len(result.chunks)} chunks (including OCR content)")
```

### Processing Scanned Documents

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR

# Scanned PDFs need OCR
ocr_engine = OpenAIOCR(api_key="...", model="gpt-4o")
processor = DocumentProcessor(ocr_engine=ocr_engine)

# Process scanned document
text = processor.extract_text(
    "scanned_contract.pdf",
    ocr_processing=True
)

# The text now contains OCR results from all pages
print(text)
```

### Batch Processing with Progress

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR
from pathlib import Path
from tqdm import tqdm

# Setup
ocr_engine = OpenAIOCR(api_key="...", model="gpt-4o")
processor = DocumentProcessor(ocr_engine=ocr_engine)

doc_dir = Path("documents/")
output_dir = Path("processed/")
output_dir.mkdir(exist_ok=True)

# Get all files
files = list(doc_dir.glob("**/*.*"))

for file in tqdm(files, desc="Processing documents"):
    try:
        # Check if it's an image file (needs OCR)
        is_image = file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        
        result = processor.extract_chunks(
            file,
            chunk_size=1000,
            ocr_processing=is_image or file.suffix.lower() == '.pdf'
        )
        
        # Save chunks
        output_file = output_dir / f"{file.stem}_chunks.md"
        result.save_to_md(output_file)
        
    except Exception as e:
        print(f"\nError processing {file.name}: {e}")

print(f"\nProcessed {len(files)} documents")
```

### Multi-Modal Document Processing

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine import OpenAIOCR

# Complete multi-modal setup
processor = DocumentProcessor(
    ocr_engine=OpenAIOCR(api_key="...", model="gpt-4o"),
    image_directory="output/images",
    image_tag_prefix="[IMAGE:",
    image_tag_suffix="]"
)

# Process PowerPoint with images and charts
result = processor.extract_chunks(
    "presentation.pptx",
    chunk_size=1000,
    ocr_processing=True  # Charts and diagrams become text
)

# Each chunk may contain:
# - Slide text
# - OCR results from charts/diagrams
# - Table content (as HTML)
for chunk in result.chunks:
    print(chunk)
    print("-" * 80)
```

---

## API Reference

### DocumentProcessor

```python
class DocumentProcessor:
    def __init__(
        self,
        config: Optional[Dict] = None,
        ocr_engine: Optional[BaseOCR] = None,
        image_directory: Optional[str] = None,      # Default: "temp/images"
        image_tag_prefix: Optional[str] = None,     # Default: "[Image:"
        image_tag_suffix: Optional[str] = None,     # Default: "]"
        page_tag_prefix: Optional[str] = None,      # Default: "[Page Number: "
        page_tag_suffix: Optional[str] = None,      # Default: "]"
        slide_tag_prefix: Optional[str] = None,     # Default: "[Slide Number: "
        slide_tag_suffix: Optional[str] = None      # Default: "]"
    )
    
    def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        extract_metadata: bool = True,
        ocr_processing: bool = False    # Set True to convert images to text
    ) -> str
    
    def extract_chunks(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True,
        include_position_metadata: bool = False
    ) -> ChunkResult
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        file_extension: Optional[str] = None,
        preserve_tables: bool = True,
        include_position_metadata: bool = False
    ) -> Union[List[str], List[Dict]]
    
    # Properties
    @property
    def ocr_engine(self) -> Optional[BaseOCR]
    
    @ocr_engine.setter
    def ocr_engine(self, engine: Optional[BaseOCR]) -> None
    
    @property
    def image_config(self) -> Dict[str, Any]
    
    @property
    def page_tag_config(self) -> Dict[str, Any]
    
    @property
    def supported_extensions(self) -> List[str]
```

### ChunkResult

```python
class ChunkResult:
    @property
    def chunks(self) -> List[str]
    
    @property
    def chunks_with_metadata(self) -> Optional[List[Dict[str, Any]]]
    
    @property
    def has_metadata(self) -> bool
    
    @property
    def source_file(self) -> Optional[str]
    
    def save_to_md(
        self,
        path: Optional[str] = None,
        filename: str = "chunks.md",
        separator: str = "---",
        include_metadata: bool = True
    ) -> str
    
    def __len__(self) -> int
    def __iter__(self) -> Iterator[str]
    def __getitem__(self, index: int) -> str
```

### OCR Engines

```python
class BaseOCR(ABC):
    def convert_image_to_text(self, image_path: str) -> Optional[str]
    def process_text(self, text: str, image_pattern: Optional[Pattern] = None) -> str
    def set_image_pattern(self, pattern: Optional[Pattern] = None) -> None
    def set_image_pattern_from_string(self, pattern_string: str) -> None

class OpenAIOCR(BaseOCR):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None
    )

class AnthropicOCR(BaseOCR):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None
    )

class GeminiOCR(BaseOCR):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None
    )

class VllmOCR(BaseOCR):
    def __init__(
        self,
        base_url: str,
        model: str,
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None
    )

class BedrockOCR(BaseOCR):
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        llm_client: Optional[Any] = None,
        prompt: Optional[str] = None,
        connect_timeout: int = 60,
        read_timeout: int = 120,
        max_retries: int = 10
    )
```

---

## Troubleshooting

### Import Error

```python
# Use the package name for imports
from contextifier import DocumentProcessor

# Or use the full path
from contextifier.core.document_processor import DocumentProcessor
```

### OCR Not Working

```python
# Check if OCR engine is configured
processor = DocumentProcessor()
print(f"OCR Engine: {processor.ocr_engine}")  # None if not configured

# Add OCR engine
from contextifier.ocr.ocr_engine import OpenAIOCR
processor.ocr_engine = OpenAIOCR(api_key="...", model="gpt-4o")

# Now OCR will work
text = processor.extract_text("document.pdf", ocr_processing=True)
```

### Image Tags Not Being Processed

```python
# Check image tag pattern
print(f"Image config: {processor.image_config}")

# Make sure ocr_processing=True
text = processor.extract_text("document.pdf", ocr_processing=True)

# If using custom tags, OCR automatically uses the same pattern
processor = DocumentProcessor(
    ocr_engine=ocr_engine,
    image_tag_prefix="<img:",
    image_tag_suffix=">"
)
# OCR will look for <img:path> patterns
```

### Memory Issues with Large Files

```python
# Use smaller chunk sizes for large documents
result = processor.extract_chunks(
    "large_document.pdf",
    chunk_size=500,     # Smaller chunks
    chunk_overlap=100   # Less overlap
)

# Process chunks one at a time
for chunk in result:
    process_chunk(chunk)  # Your processing function
```

### Unsupported Format

```python
# Check supported extensions
processor = DocumentProcessor()
print("Supported:", processor.supported_extensions)

# Check if specific format is supported
print(processor.is_supported("pdf"))  # True
print(processor.is_supported("xyz"))  # False
```

---

## Next Steps

- Check the [full documentation](https://github.com/CocoRoF/Contextifier)
- Browse [examples](https://github.com/CocoRoF/Contextifier/tree/main/examples)
- Report issues on [GitHub](https://github.com/CocoRoF/Contextifier/issues)
- Contribute to the project via [Pull Requests](https://github.com/CocoRoF/Contextifier/pulls)
