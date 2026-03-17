# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0-alpha] — 2025-07-15

### Breaking Changes — 전면 아키텍처 재설계

v2는 v1과 **호환되지 않는** 완전한 재설계입니다. 기존 `contextifier` 패키지는 `contextifier_new`로 이전되었습니다.

### Added

- **5단계 강제 파이프라인**: Convert → Preprocess → Metadata → Content → Postprocess
  - `BaseHandler`가 파이프라인 실행 순서를 강제하여, 모든 핸들러가 동일한 구조를 따름
  - 각 단계는 ABC(Abstract Base Class)로 정의: `Converter`, `Preprocessor`, `MetadataExtractor`, `ContentExtractor`, `Postprocessor`
- **14개 포맷 핸들러**: PDF, PDF-Plus, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV/TSV, HWP, HWPX, RTF, Text, Image
- **HandlerRegistry**: 확장자 → 핸들러 자동 매핑, `register_defaults()`로 빌트인 전체 등록
- **불변 설정 시스템**: frozen dataclass 기반 `ProcessingConfig`
  - `TagConfig`, `ImageConfig`, `ChartConfig`, `MetadataConfig`, `TableConfig`, `ChunkingConfig`, `OCRConfig`
  - Fluent builder: `config.with_tags()`, `config.with_chunking()`, ...
  - `to_dict()` / `from_dict()` 직렬화 지원
  - 포맷별 옵션: `config.with_format_option("pdf", ...)`
- **4가지 청킹 전략 자동 선택**:
  - `TableChunkingStrategy` (우선순위 5) — 스프레드시트 전용
  - `PageChunkingStrategy` (우선순위 10) — 페이지 경계 기반
  - `ProtectedChunkingStrategy` (우선순위 20) — HTML 테이블/Protected Region 보존
  - `PlainChunkingStrategy` (우선순위 100) — 재귀 분할 폴백
- **5종 OCR 엔진**: OpenAI, Anthropic, Google Gemini, AWS Bedrock, vLLM
  - 편의 생성자: `from_api_key()` 각 엔진별 지원
  - LangChain 클라이언트 직접 전달 가능
  - 커스텀 프롬프트 지원
- **5개 공유 서비스** (DI 패턴):
  - `TagService` — 페이지/슬라이드/시트 태그 생성
  - `ImageService` — 이미지 저장/태그/중복 제거/스토리지 백엔드
  - `ChartService` — 차트 데이터 포맷팅
  - `TableService` — 테이블 HTML/MD/Text 변환
  - `MetadataService` — 메타데이터 포맷팅 (한국어/영어)
- **통합 타입 시스템** (`types.py`):
  - `FileContext` TypedDict — 모든 핸들러의 표준 입력
  - `ExtractionResult` — 텍스트/메타데이터/테이블/이미지/차트 통합 결과
  - `DocumentMetadata`, `TableData`, `TableCell`, `ChartData` 등 공용 데이터클래스
  - `FileCategory`, `OutputFormat`, `NamingStrategy`, `StorageType` 등 Enum
- **통합 예외 계층** (`errors.py`):
  - `ContextifierError` 기반 예외 트리
  - `FileNotFoundError`, `UnsupportedFormatError`, `HandlerNotFoundError` 등
- **ChunkResult**: `save_to_md()`, `__len__`, `__iter__`, `__getitem__` 지원
- **DOC 핸들러 자동 감지**: OLE, HTML, DOCX, RTF 내부 형식 자동 판별

### Removed

- v1 `contextifier` 패키지의 모든 레거시 코드
  - `core/document_processor.py` (거대한 단일 파일)
  - `core/functions/` (utils.py, 개별 processor 모듈들)
  - `core/processor/` (핸들러별 파일이지만 통일된 구조 없음)
  - `chunking/` (단일 chunking.py에 모든 로직)
  - `ocr/ocr_engine/` (엔진별 파일이지만 일관성 없음)

### Architecture Changes

- **Facade 패턴**: `DocumentProcessor`가 유일한 공개 진입점
- **Strategy 패턴**: 청킹 전략 자동 선택
- **Template Method 패턴**: `BaseHandler.process()`가 5단계 순서를 강제
- **Dependency Injection**: 서비스가 생성 시 주입, 핸들러 간 공유
- **Registry 패턴**: 확장자 → 핸들러 매핑 자동화

---

## [0.1.2] — 2025-05-14

### Fixed
- `requirements.txt` path resolution for packaged installs.

---

## [0.1.0] — 2025-05-14

### Added
- Initial release of Contextifier v1.
- Support for PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV, TSV, HWP, HWPX, RTF, TXT, Image.
- OCR integration via OpenAI, Anthropic, Gemini, Bedrock.
- Basic text chunking with page/table awareness.
- Metadata extraction for common document formats.
