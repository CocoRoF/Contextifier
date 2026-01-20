# Quick Start Guide

A comprehensive guide to get started with **Contextifier**.

## Installation

```bash
pip install contextifier
```

Or using uv:

```bash
uv add contextifier
```

## Basic Usage

### 1. Simple Text Extraction

```python
from contextifier import DocumentProcessor

# Create processor instance
processor = DocumentProcessor()

# Extract text from any supported document
text = processor.extract_text("document.pdf")
print(text)
```

### 2. Extract and Chunk in One Step

The `extract_chunks()` method combines extraction and chunking:

```python
from contextifier import DocumentProcessor

processor = DocumentProcessor()

# Extract text and split into chunks
result = processor.extract_chunks(
    "long_document.pdf",
    chunk_size=1000,      # Target chunk size in characters
    chunk_overlap=200     # Overlap between chunks for context continuity
)

# Access chunks
print(f"Total chunks: {len(result.chunks)}")

for i, chunk in enumerate(result.chunks):
    print(f"Chunk {i + 1}: {len(chunk)} characters")
    print(chunk[:200])  # Preview first 200 chars
    print("-" * 80)
```

### 3. Save Chunks to Markdown

The `ChunkResult` object provides convenient methods for saving:

```python
result = processor.extract_chunks("document.pdf", chunk_size=500)

# Save all chunks to a single markdown file
saved_path = result.save_to_md("output/chunks.md")
print(f"Saved to: {saved_path}")

# Or save to a directory with auto-generated filename
result.save_to_md("output/", filename="my_document_chunks.md")
```

### 4. Process Multiple Documents

```python
from contextifier import DocumentProcessor
from pathlib import Path

processor = DocumentProcessor()
documents_dir = Path("documents/")
all_chunks = []

for file_path in documents_dir.glob("*.*"):
    try:
        result = processor.extract_chunks(file_path, chunk_size=500)
        all_chunks.extend(result.chunks)
        print(f"✅ {file_path.name}: {len(result.chunks)} chunks")
    except ValueError as e:
        print(f"⚠️ {file_path.name}: Unsupported format")
    except Exception as e:
        print(f"❌ {file_path.name}: {e}")

print(f"\nTotal chunks: {len(all_chunks)}")
```

### 5. OCR for Scanned Documents

For scanned PDFs or image-based documents, use OCR processing:

```python
from contextifier import DocumentProcessor
from contextifier.ocr.ocr_engine.openai_ocr import OpenAIOCREngine

# Initialize OCR engine
ocr_engine = OpenAIOCREngine(api_key="your-api-key", model="gpt-4o")

# Create processor with OCR support
processor = DocumentProcessor(ocr_engine=ocr_engine)

# Extract text with OCR processing enabled
text = processor.extract_text(
    "scanned_document.pdf",
    ocr_processing=True  # Enable OCR for image tags
)
```

### 6. Custom Image Tag Format

Configure how extracted images are referenced in the output:

```python
# Default: [Image:path/to/image.png]
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
```

## Supported Formats

| Category | Extensions | Features |
|----------|------------|----------|
| **PDF** | `.pdf` | Table detection, OCR fallback, complex layouts |
| **Word** | `.docx`, `.doc` | Tables, images, charts, styles |
| **Excel** | `.xlsx`, `.xls` | Multiple sheets, formulas, charts |
| **PowerPoint** | `.pptx`, `.ppt` | Slides, notes, embedded objects |
| **Hangul** | `.hwp`, `.hwpx` | Korean word processor (full support) |
| **Text** | `.txt`, `.md`, `.rtf` | Plain text, Markdown, Rich Text |
| **Web** | `.html`, `.htm` | HTML documents |
| **Data** | `.csv`, `.tsv`, `.json` | Structured data formats |
| **Code** | `.py`, `.js`, `.java`, etc. | 20+ programming languages |
| **Config** | `.yaml`, `.toml`, `.ini` | Configuration files |

## Configuration Options

### Chunk Size and Overlap

```python
result = processor.extract_chunks(
    "document.pdf",
    chunk_size=1000,      # Smaller = more chunks, better for semantic search
    chunk_overlap=200,    # Overlap maintains context between chunks
    preserve_tables=True  # Keep tables intact (default)
)
```

### Metadata Extraction

```python
# Include document metadata (default: True)
text = processor.extract_text("document.docx", extract_metadata=True)

# Metadata is included at the beginning of extracted text:
# <Document-Metadata>
#   Author: John Doe
#   Created: 2024-01-15 10:30:00
#   Title: Sample Document
# </Document-Metadata>
```

### Table Preservation

```python
# Preserve table structure during chunking (recommended)
result = processor.extract_chunks("report.pdf", preserve_tables=True)

# Force chunking even through tables (may break table structure)
result = processor.extract_chunks("report.pdf", preserve_tables=False)
```

## OCR Engine Options

Contextifier supports multiple OCR backends for processing scanned documents:

```python
# OpenAI GPT-4 Vision
from contextifier.ocr.ocr_engine.openai_ocr import OpenAIOCREngine
engine = OpenAIOCREngine(api_key="...", model="gpt-4o")

# Anthropic Claude Vision
from contextifier.ocr.ocr_engine.anthropic_ocr import AnthropicOCREngine
engine = AnthropicOCREngine(api_key="...")

# Google Gemini Vision
from contextifier.ocr.ocr_engine.gemini_ocr import GeminiOCREngine
engine = GeminiOCREngine(api_key="...")

# vLLM (self-hosted)
from contextifier.ocr.ocr_engine.vllm_ocr import VLLMOCREngine
engine = VLLMOCREngine(base_url="http://localhost:8000")

# Use with DocumentProcessor
processor = DocumentProcessor(ocr_engine=engine)
```

## Common Use Cases

### Building a RAG System

```python
from contextifier import DocumentProcessor
import chromadb

processor = DocumentProcessor()
client = chromadb.Client()
collection = client.create_collection("documents")

# Process and chunk document
result = processor.extract_chunks("knowledge_base.pdf", chunk_size=500)

# Add chunks to vector database
for i, chunk in enumerate(result.chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"],
        metadatas=[{"source": result.source_file, "chunk_index": i}]
    )

print(f"Indexed {len(result.chunks)} chunks")
```

### Document Analysis

```python
from contextifier import DocumentProcessor

processor = DocumentProcessor()
text = processor.extract_text("report.pdf")

# Word count
word_count = len(text.split())
print(f"Word count: {word_count:,}")

# Keyword frequency
keywords = ["AI", "machine learning", "neural network", "deep learning"]
for keyword in keywords:
    count = text.lower().count(keyword.lower())
    if count > 0:
        print(f"{keyword}: {count} occurrences")
```

### Batch Processing with Progress

```python
from contextifier import DocumentProcessor
from pathlib import Path
from tqdm import tqdm

processor = DocumentProcessor()
doc_dir = Path("documents/")
output_dir = Path("processed/")
output_dir.mkdir(exist_ok=True)

# Get all files
files = list(doc_dir.glob("**/*.*"))

for file in tqdm(files, desc="Processing documents"):
    try:
        result = processor.extract_chunks(file, chunk_size=1000)
        
        # Save chunks
        output_file = output_dir / f"{file.stem}_chunks.md"
        result.save_to_md(output_file)
        
    except Exception as e:
        print(f"\nError processing {file.name}: {e}")

print(f"\nProcessed {len(files)} documents")
```

## API Reference

### DocumentProcessor

```python
class DocumentProcessor:
    def __init__(
        self,
        config: Optional[Dict] = None,
        ocr_engine: Optional[BaseOCR] = None,
        image_directory: Optional[str] = None,
        image_tag_prefix: Optional[str] = None,
        image_tag_suffix: Optional[str] = None
    )
    
    def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        extract_metadata: bool = True,
        ocr_processing: bool = False
    ) -> str
    
    def extract_chunks(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True
    ) -> ChunkResult
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_tables: bool = True
    ) -> List[str]
```

### ChunkResult

```python
class ChunkResult:
    @property
    def chunks(self) -> List[str]
    
    @property
    def source_file(self) -> Optional[str]
    
    def save_to_md(
        self,
        path: Optional[str] = None,
        filename: str = "chunks.md",
        separator: str = "---"
    ) -> str
    
    def __len__(self) -> int
    def __iter__(self) -> Iterator[str]
    def __getitem__(self, index: int) -> str
```

## Troubleshooting

### Import Error

```python
# Use the package name for imports
from contextifier import DocumentProcessor

# Or use the full path
from contextifier.core.document_processor import DocumentProcessor
```

### File Not Found

```python
from pathlib import Path

file_path = Path("document.pdf").resolve()
if file_path.exists():
    text = processor.extract_text(file_path)
else:
    print(f"File not found: {file_path}")
```

### Unsupported Format

```python
# Check supported extensions
from contextifier import DocumentProcessor

processor = DocumentProcessor()

# Supported document types
print("Documents:", processor.DOCUMENT_TYPES)
print("Text files:", processor.TEXT_TYPES)
print("Code files:", processor.CODE_TYPES)
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

## Next Steps

- Check the [full documentation](https://github.com/CocoRoF/Contextifier)
- Browse [examples](https://github.com/CocoRoF/Contextifier/tree/main/examples)
- Report issues on [GitHub](https://github.com/CocoRoF/Contextifier/issues)
- Contribute to the project via [Pull Requests](https://github.com/CocoRoF/Contextifier/pulls)
