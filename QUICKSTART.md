# Quick Start Guide

## Installation

```bash
pip install contextify
```

## Basic Usage

### 1. Simple Text Extraction

```python
from libs.core.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Extract text from any document
text = processor.extract_text("document.pdf")
print(text)
```

### 2. Extract with Chunking

```python
# Extract and split into chunks
result = processor.extract_chunks(
    "long_document.pdf",
    chunk_size=1000,      # Target chunk size in characters
    chunk_overlap=200     # Overlap between chunks
)

# Access chunks
for i, chunk in enumerate(result.chunks):
    print(f"Chunk {i + 1}:")
    print(chunk.text)
    print(f"Metadata: {chunk.metadata}")
    print("-" * 80)
```

### 3. Process Multiple Documents

```python
import os

documents_dir = "documents/"
all_chunks = []

for filename in os.listdir(documents_dir):
    filepath = os.path.join(documents_dir, filename)
    
    try:
        result = processor.extract_chunks(filepath, chunk_size=500)
        all_chunks.extend(result.chunks)
        print(f"✅ Processed: {filename} ({len(result.chunks)} chunks)")
    except Exception as e:
        print(f"❌ Failed: {filename} - {e}")

print(f"\nTotal chunks: {len(all_chunks)}")
```

### 4. With OCR for Scanned Documents

```python
from libs.ocr.ocr_processor import OCRProcessor
from libs.ocr.ocr_engine.openai_ocr import OpenAIOCREngine

# Configure OCR engine
ocr_engine = OpenAIOCREngine(api_key="your-api-key")
ocr_processor = OCRProcessor(ocr_engine=ocr_engine)

# Create processor with OCR support
processor = DocumentProcessor(ocr_processor=ocr_processor)

# Process scanned PDF
text = processor.extract_text("scanned_document.pdf")
```

### 5. Extract Metadata

```python
# The extracted text includes metadata at the beginning
text = processor.extract_text("document.docx")

# Metadata format:
# <Document-Metadata>
#   작성자: John Doe
#   작성일: 2024-01-15 10:30:00
#   제목: Sample Document
# </Document-Metadata>

# Parse metadata if needed
if "<Document-Metadata>" in text:
    metadata_section = text.split("</Document-Metadata>")[0]
    pure_text = text.split("</Document-Metadata>")[1].strip()
```

## Supported Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | With table detection and OCR fallback |
| Word | `.docx`, `.doc` | Including tables, images, charts |
| Excel | `.xlsx`, `.xls` | Multiple sheets, formulas, charts |
| PowerPoint | `.pptx`, `.ppt` | Slides, notes, embedded objects |
| Hangul | `.hwp`, `.hwpx` | Korean word processor |
| Text | `.txt`, `.md`, `.rtf` | Plain text and markdown |
| Web | `.html` | HTML documents |
| Data | `.csv`, `.json` | Structured data formats |
| Code | `.py`, `.js`, `.java`, etc. | 20+ programming languages |

## Configuration Options

### Chunk Size and Overlap

```python
result = processor.extract_chunks(
    "document.pdf",
    chunk_size=1000,    # Smaller = more chunks, better for search
    chunk_overlap=200   # Overlap helps maintain context
)
```

### OCR Engines

```python
# OpenAI GPT-4 Vision
from libs.ocr.ocr_engine.openai_ocr import OpenAIOCREngine
engine = OpenAIOCREngine(api_key="...")

# Anthropic Claude Vision
from libs.ocr.ocr_engine.anthropic_ocr import AnthropicOCREngine
engine = AnthropicOCREngine(api_key="...")

# Google Gemini Vision
from libs.ocr.ocr_engine.gemini_ocr import GeminiOCREngine
engine = GeminiOCREngine(api_key="...")

# vLLM (self-hosted)
from libs.ocr.ocr_engine.vllm_ocr import VLLMOCREngine
engine = VLLMOCREngine(base_url="http://localhost:8000")
```

## Common Use Cases

### Building a RAG System

```python
from libs.core.document_processor import DocumentProcessor
import chromadb

processor = DocumentProcessor()
client = chromadb.Client()
collection = client.create_collection("documents")

# Process documents
result = processor.extract_chunks("knowledge_base.pdf", chunk_size=500)

# Add to vector DB
for i, chunk in enumerate(result.chunks):
    collection.add(
        documents=[chunk.text],
        ids=[f"doc_{i}"],
        metadatas=[chunk.metadata]
    )
```

### Document Analysis

```python
# Extract and analyze
text = processor.extract_text("report.pdf")

# Count words
word_count = len(text.split())
print(f"Word count: {word_count}")

# Find keywords
keywords = ["AI", "machine learning", "neural network"]
for keyword in keywords:
    count = text.lower().count(keyword.lower())
    print(f"{keyword}: {count} occurrences")
```

### Batch Processing with Progress

```python
from pathlib import Path
from tqdm import tqdm

doc_dir = Path("documents/")
output_dir = Path("processed/")
output_dir.mkdir(exist_ok=True)

files = list(doc_dir.glob("**/*.*"))
for file in tqdm(files, desc="Processing"):
    try:
        text = processor.extract_text(str(file))
        
        # Save extracted text
        output_file = output_dir / f"{file.stem}.txt"
        output_file.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"Error processing {file.name}: {e}")
```

## Troubleshooting

### Import Error

If you get import errors after installation:

```python
# Make sure to use the full import path
from libs.core.document_processor import DocumentProcessor
```

### File Not Found

Always use absolute paths or check file existence:

```python
import os
file_path = os.path.abspath("document.pdf")
if os.path.exists(file_path):
    text = processor.extract_text(file_path)
```

### Encoding Issues

For text files with special characters:

```python
# Contextify auto-detects encoding, but you can verify:
import chardet

with open("file.txt", "rb") as f:
    result = chardet.detect(f.read())
    print(f"Detected encoding: {result['encoding']}")
```

## Next Steps

- Check [full documentation](https://github.com/CocoRoF/Contextify)
- See [examples directory](https://github.com/CocoRoF/Contextify/tree/main/examples)
- Report issues on [GitHub](https://github.com/CocoRoF/Contextify/issues)
