# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-19

### Added
- Initial release of Contextifier
- Multi-format document support (PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, HWP, HWPX)
- Intelligent text extraction with structure preservation
- Table detection and extraction with HTML formatting
- OCR integration (OpenAI, Anthropic, Google Gemini, vLLM)
- Smart chunking with semantic awareness
- Metadata extraction
- Support for 20+ code file formats
- Korean document support (HWP, HWPX)

### Features
- `DocumentProcessor` class for easy document processing
- Configurable chunk size and overlap
- Protected regions for code blocks
- Pluggable OCR engine architecture
- Automatic encoding detection for text files
- Chart and image extraction from Office documents

[0.1.0]: https://github.com/CocoRoF/Contextifier/releases/tag/v0.1.0
