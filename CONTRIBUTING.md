# Contributing to Contextifier

Thank you for your interest in contributing to Contextifier! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Clone & Create Environment

```bash
git clone https://github.com/your-org/contextifier.git
cd contextifier

python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

pip install -e ".[dev]"
```

### 2. Project Structure

```
contextifier_new/           # v2 main package
├── document_processor.py   # Facade (public API)
├── config.py               # ProcessingConfig
├── types.py                # Shared types
├── errors.py               # Exception hierarchy
├── handlers/               # 14 format handlers
├── pipeline/               # 5-Stage ABCs
├── services/               # Shared services
├── chunking/               # Chunking subsystem
└── ocr/                    # OCR subsystem
```

## Coding Conventions

### General Rules

- **Python 3.12+** syntax
- Type hints required on all public APIs
- Docstrings required (Google style)
- `from __future__ import annotations` at the top of every module

### Architecture Rules

1. **All handlers must follow the 5-stage pipeline:**
   - `Converter` → `Preprocessor` → `MetadataExtractor` → `ContentExtractor` → `Postprocessor`
   - `BaseHandler.process()` enforces execution order — implement each stage only.

2. **Do not create services directly:**
   - `TagService`, `ImageService`, etc. are created by `DocumentProcessor` and injected.
   - Handlers access them via `self._services["tag_service"]`, etc.

3. **Pass all settings through `ProcessingConfig`:**
   - No hardcoded magic numbers.
   - If you need a new setting, add a field to the appropriate `*Config` class.

4. **Respect the Facade pattern:**
   - The only user-facing API is `DocumentProcessor`.
   - Do not instruct users to import internal modules directly (OCR engines excepted).

## Adding a New Handler

### 1. Create Directory

```
contextifier_new/handlers/myformat/
├── __init__.py
├── converter.py
├── preprocessor.py
├── metadata_extractor.py
├── content_extractor.py
└── postprocessor.py
```

### 2. Implement Each Pipeline Stage

```python
# converter.py
from contextifier_new.pipeline.converter import BaseConverter

class MyFormatConverter(BaseConverter):
    def convert(self, file_context, **kwargs):
        # Binary → Format-specific object
        return parsed_object
```

### 3. Register the Handler

Add to `contextifier_new/handlers/registry.py` in `register_defaults()`:

```python
from contextifier_new.handlers.myformat import MyFormatHandler
self.register(MyFormatHandler, extensions=["myf", "myformat"])
```

## Commit Convention

```
feat: add new feature
fix: bug fix
docs: documentation changes
refactor: refactoring (no behavior change)
test: add/modify tests
chore: build/config changes
```

Examples:
```
feat(handler): add EPUB handler with full pipeline
fix(chunking): preserve table structure in protected strategy
docs: update QUICKSTART with batch processing example
```

## Pull Request Guide

1. Create a feature branch from `main`
2. Implement changes and test
3. Include rationale and test results in PR description
4. Squash merge after review

## Reporting Issues

When reporting a bug, please include:

- Python version
- OS and version
- Input file format and size
- Full error message
- Reproduction code (if possible)

## License

All contributions are released under the project's Apache License 2.0.
