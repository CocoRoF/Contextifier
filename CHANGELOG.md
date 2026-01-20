# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-01-20

### Added
- **BedrockOCR**: AWS Bedrock Vision model support for OCR processing
  - Supports Claude 3.5 Sonnet and other Bedrock vision models
  - Full AWS credential configuration (access key, secret key, session token, region)
  - Configurable timeouts and retry settings
- **ImageFileHandler**: New handler for standalone image files (jpg, png, gif, bmp, webp)
  - Automatically uses OCR engine when available
  - Returns image tag format when OCR is not configured for later processing
- **PageTagProcessor**: Centralized page/slide/sheet tag processing system
  - Unified tag generation across all document handlers
  - Configurable tag prefixes and suffixes
- **Image pattern support for OCR**: Custom image tag patterns now passed to OCR engine
  - `ImageProcessor.get_pattern_string()` method for regex pattern generation
  - `BaseOCR.set_image_pattern()` and `set_image_pattern_from_string()` methods
  - OCR engines now recognize custom image tag formats

### Changed
- **DocumentProcessor**: OCR engine setter now invalidates handler registry for proper refresh
- **Handler registry**: ImageFileHandler automatically registered with OCR engine support
- **QUICKSTART.md**: Complete rewrite with comprehensive documentation
  - 3-stage processing pipeline documentation (File → Text → OCR → Chunks)
  - Detailed OCR configuration guide for all 5 engines
  - Tag customization examples (image, page, slide, sheet)
  - Complete API reference with all parameters

### Improved
- All Korean comments and docstrings in `img_processor.py` converted to English
- Enhanced OCR integration with custom pattern matching support
- Better separation of concerns with PageTagProcessor

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

[0.1.2]: https://github.com/CocoRoF/Contextifier/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/CocoRoF/Contextifier/releases/tag/v0.1.0
