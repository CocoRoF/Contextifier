# Contextifier v2 — Architecture & Pipeline Specification

> **Version:** 2.0.0-alpha
> **Last Updated:** 2025-07
> **Status:** Interface skeleton complete, concrete implementations pending

---

## Table of Contents

1. [Overview](#1-overview)
2. [Package Structure](#2-package-structure)
3. [Core Type System](#3-core-type-system)
4. [Configuration System](#4-configuration-system)
5. [Error Hierarchy](#5-error-hierarchy)
6. [Service Layer](#6-service-layer)
7. [Pipeline Architecture](#7-pipeline-architecture)
8. [Handler Architecture](#8-handler-architecture)
9. [Chunking Subsystem](#9-chunking-subsystem)
10. [OCR Subsystem](#10-ocr-subsystem)
11. [Complete Execution Flow](#11-complete-execution-flow)
12. [Design Patterns Summary](#12-design-patterns-summary)
13. [Known Issues & Future Work](#13-known-issues--future-work)

---

## 1. Overview

Contextifier v2는 모든 문서 포맷에 대해 **동일한 5단계 파이프라인**을 강제하는 통합 문서 처리 라이브러리입니다. v1의 핸들러별 불일치와 임시방편적 처리 로직을 완전히 재설계하여, **일관된 인터페이스 계약**과 **명확한 관심사 분리**를 구현합니다.

### 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Enforced Pipeline** | 모든 핸들러는 동일한 5단계 파이프라인을 반드시 거침. process()는 오버라이드 불가 |
| **One Extension Per Handler** | 문서 포맷 핸들러는 정확히 1개의 확장자만 담당 (카테고리 핸들러 예외) |
| **Service Injection** | 공유 서비스(Image, Tag, Chart, Table, Metadata)는 DI로 주입 |
| **Immutable Config** | frozen dataclass로 설정 불변성 보장, `with_*()` fluent builder |
| **Format-Agnostic Services** | 포맷에 독립적인 로직은 서비스로, 포맷별 로직은 핸들러의 파이프라인 컴포넌트로 |
| **Explicit Delegation** | 핸들러 간 위임은 Stage 0 + Registry를 통해서만 가능 |

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DocumentProcessor (Facade)                       │
│                                                                         │
│   extract_text()  ─┐                                                    │
│   process()       ─┤──▶ HandlerRegistry ──▶ Handler.process()           │
│   extract_chunks() ┘         │                    │                     │
│                              │          ┌─────────┴──────────┐          │
│   chunk_text() ──────────────┼──▶       │  5-Stage Pipeline  │          │
│                              │          │  (enforced order)  │          │
│                              │          └────────────────────┘          │
│                              │                                          │
│   OCRProcessor (optional) ◄──┘                                          │
│                                                                         │
│   ┌──────────────────── Shared Services ──────────────────────────┐     │
│   │ TagService │ ImageService │ ChartService │ TableService │ ... │     │
│   └──────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Package Structure

```
contextifier_new/
├── __init__.py                    # 진입점: DocumentProcessor 노출
├── types.py                       # 모든 공유 타입, Enum, TypedDict, Dataclass
├── config.py                      # 불변 설정 시스템 (frozen dataclass)
├── errors.py                      # 통합 예외 계층
├── document_processor.py          # Facade: 서비스 생성, 레지스트리, 청킹, OCR 조율
│
├── pipeline/                      # 5-Stage 파이프라인 ABC
│   ├── __init__.py
│   ├── converter.py               # Stage 1: Binary → Format Object
│   ├── preprocessor.py            # Stage 2: Clean / Transform
│   ├── metadata_extractor.py      # Stage 3: Document Metadata
│   ├── content_extractor.py       # Stage 4: Text, Tables, Images, Charts
│   └── postprocessor.py           # Stage 5: Final Assembly & Cleanup
│
├── services/                      # 포맷 독립 공유 서비스
│   ├── __init__.py
│   ├── tag_service.py             # 구조적 태그 생성/탐지/제거
│   ├── image_service.py           # 이미지 저장, 중복제거, 태그 생성
│   ├── chart_service.py           # 차트 데이터 포맷팅
│   ├── table_service.py           # 테이블 HTML/MD/Text 포맷팅
│   ├── metadata_service.py        # 메타데이터 블록 포맷팅
│   └── storage/                   # 저장 백엔드
│       ├── base.py                # BaseStorageBackend ABC
│       └── local.py               # 로컬 파일시스템 구현
│
├── handlers/                      # 포맷별 핸들러
│   ├── __init__.py                # 핸들러 전체 export
│   ├── base.py                    # BaseHandler ABC (enforced pipeline)
│   ├── registry.py                # HandlerRegistry (ext → handler 매핑)
│   ├── pdf/handler.py             # .pdf   (PyMuPDF/fitz)
│   ├── docx/handler.py            # .docx  (python-docx)
│   ├── doc/handler.py             # .doc   (OLE2 + 위임)
│   ├── pptx/handler.py            # .pptx  (python-pptx)
│   ├── ppt/handler.py             # .ppt   (OLE2)
│   ├── xlsx/handler.py            # .xlsx  (openpyxl)
│   ├── xls/handler.py             # .xls   (xlrd)
│   ├── csv/handler.py             # .csv
│   ├── tsv/handler.py             # .tsv
│   ├── hwp/handler.py             # .hwp   (Korean OLE binary)
│   ├── hwpx/handler.py            # .hwpx  (Korean XML/ZIP)
│   ├── rtf/handler.py             # .rtf
│   ├── text/handler.py            # .txt/.md/.py/... (카테고리)
│   └── image/handler.py           # .jpg/.png/... (카테고리, OCR)
│
├── chunking/                      # 청킹 서브시스템
│   ├── __init__.py
│   ├── chunker.py                 # TextChunker Facade
│   ├── constants.py               # 패턴, 임계값, 데이터클래스
│   └── strategies/                # Strategy 패턴
│       ├── base.py                # BaseChunkingStrategy ABC
│       ├── table_strategy.py      # 테이블 기반 (priority 5)
│       ├── page_strategy.py       # 페이지 마커 기반 (priority 10)
│       ├── protected_strategy.py  # 보호 영역 인식 (priority 20)
│       └── plain_strategy.py      # 단순 재귀 분할 (priority 100)
│
└── ocr/                           # OCR 서브시스템
    ├── __init__.py
    ├── base.py                    # BaseOCREngine ABC
    ├── processor.py               # OCRProcessor 오케스트레이터
    └── engines/                   # Provider별 엔진
        ├── openai_engine.py       # GPT-4V / GPT-4o
        ├── anthropic_engine.py    # Claude Vision
        ├── gemini_engine.py       # Gemini Vision
        ├── bedrock_engine.py      # AWS Bedrock
        └── vllm_engine.py         # Self-hosted VLLM
```

---

## 3. Core Type System

> 정의 위치: `types.py`

모든 모듈이 공유하는 타입을 한 곳에 집중시켜 **타입 불일치를 원천 차단**합니다.

### 3.1 Enums

| Enum | 값 | 용도 |
|------|----|------|
| `FileCategory` | document, presentation, spreadsheet, text, code, config, data, script, log, web, image, unknown | 파일 분류 |
| `OutputFormat` | html, markdown, text | 테이블 출력 포맷 |
| `ImageFormat` | png, jpeg, jpg, gif, bmp, webp, tiff, unknown | 이미지 포맷 |
| `NamingStrategy` | hash, uuid, sequential, timestamp | 저장 파일명 전략 |
| `StorageType` | local, minio, s3, azure_blob, gcs | 저장 백엔드 종류 |
| `TagType` | page, slide, sheet | 구조적 태그 종류 |
| `PipelineStage` | convert, preprocess, extract_metadata, extract_content, postprocess | 파이프라인 단계 |
| `MetadataField` | title, subject, author, keywords, ... | 메타데이터 필드명 |

### 3.2 Data Structures

```
FileContext (TypedDict)
  ├── file_path: str         # 절대 경로
  ├── file_name: str         # 파일명 (확장자 포함)
  ├── file_extension: str    # 소문자, 점 없음 (예: "pdf")
  ├── file_category: str     # FileCategory.value
  ├── file_data: bytes       # 원본 바이너리
  ├── file_stream: BytesIO   # 재사용 가능 스트림
  └── file_size: int         # 바이트 크기

DocumentMetadata (dataclass)
  ├── title, subject, author, keywords, comments
  ├── last_saved_by, create_time, last_saved_time
  ├── page_count, word_count, category, revision
  └── custom: Dict[str, Any]
  메서드: to_dict(), from_dict(), is_empty()

TableCell (dataclass)
  ├── content: str
  ├── row_span, col_span: int
  ├── is_header: bool
  └── row_index, col_index, nested_table

TableData (dataclass)
  ├── rows: List[List[TableCell]]
  ├── num_rows, num_cols: int
  └── has_header, col_widths_percent, caption, metadata

ChartSeries (dataclass)
  └── name: str, values: List[Any]

ChartData (dataclass)
  └── chart_type, title, categories, series, raw_content

ExtractionResult (dataclass)   ← 파이프라인 최종 출력
  ├── text: str                 # 최종 텍스트
  ├── metadata: DocumentMetadata
  ├── tables: List[TableData]
  ├── charts: List[ChartData]
  ├── images: List[str]         # 이미지 태그 or 경로
  ├── page_count: int
  └── warnings: List[str]

PreprocessedData (dataclass)   ← Stage 2 → Stage 3,4 전달
  ├── content: Any              # 전처리된 주요 콘텐츠
  ├── raw_content: Any          # 원본 참조
  ├── encoding: str
  ├── resources: Dict           # 추출된 리소스 (이미지 등)
  └── properties: Dict          # 발견된 속성

Chunk (dataclass)
  ├── text: str
  └── metadata: ChunkMetadata
        ├── chunk_index, page_number
        └── line_start, line_end, global_start, global_end
```

### 3.3 Extension Registry

```python
EXTENSION_CATEGORIES: Dict[str, FileCategory]   # "pdf" → FileCategory.DOCUMENT
get_category("pdf")       → FileCategory.DOCUMENT
get_extensions(FileCategory.DOCUMENT) → frozenset({"pdf","docx","doc","rtf","hwp","hwpx"})
```

---

## 4. Configuration System

> 정의 위치: `config.py`

**frozen dataclass** 계층 구조로, 생성 후 변경 불가. `with_*()` fluent builder로 수정된 사본 생성.

```
ProcessingConfig (root, frozen)
├── tags: TagConfig
│   ├── page_prefix/suffix     # "[Page Number: " / "]"
│   ├── slide_prefix/suffix    # "[Slide Number: " / "]"
│   ├── sheet_prefix/suffix    # "[Sheet: " / "]"
│   ├── image_prefix/suffix    # "[Image:" / "]"
│   ├── chart_prefix/suffix    # "[chart]" / "[/chart]"
│   └── metadata_prefix/suffix # "<Document-Metadata>" / "</Document-Metadata>"
│
├── images: ImageConfig
│   ├── directory_path         # "temp/images"
│   ├── naming_strategy        # NamingStrategy.HASH
│   ├── default_format         # "png"
│   ├── quality                # 95
│   ├── skip_duplicate         # True
│   └── storage_type           # StorageType.LOCAL
│
├── charts: ChartConfig
│   ├── use_html_table         # True
│   ├── include_chart_type     # True
│   └── include_chart_title    # True
│
├── metadata: MetadataConfig
│   ├── language               # "ko"
│   ├── date_format            # "%Y-%m-%d %H:%M:%S"
│   └── indent                 # "  "
│
├── tables: TableConfig
│   ├── output_format          # OutputFormat.HTML
│   ├── clean_whitespace       # True
│   └── preserve_merged_cells  # True
│
├── chunking: ChunkingConfig
│   ├── chunk_size             # 1000
│   ├── chunk_overlap          # 200
│   ├── preserve_tables        # True
│   ├── include_position_metadata # False
│   └── strategy               # "recursive"
│
├── ocr: OCRConfig
│   ├── enabled                # False
│   ├── provider               # None ("openai"/"anthropic"/...)
│   └── prompt                 # None (default prompt 사용)
│
└── format_options: Dict[str, Dict[str, Any]]  # 포맷별 추가 옵션
```

**사용 예시:**
```python
# 기본 설정
config = ProcessingConfig()

# 커스텀 설정
config = ProcessingConfig(
    tags=TagConfig(page_prefix="<page>", page_suffix="</page>"),
    images=ImageConfig(directory_path="output/images"),
    chunking=ChunkingConfig(chunk_size=2000),
)

# Fluent builder
config = config.with_tags(page_prefix="<!-- Page ").with_chunking(chunk_size=2000)

# 직렬화
d = config.to_dict()
config2 = ProcessingConfig.from_dict(d)
```

---

## 5. Error Hierarchy

> 정의 위치: `errors.py`

모든 예외는 `ContextifierError`를 상속하며, `code`(머신 판독 가능), `context`(디버깅 정보), `cause`(원인 체인)를 제공합니다.

```
ContextifierError
├── ConfigurationError                    # 설정 오류
├── FileError                             # 파일 I/O
│   ├── FileNotFoundError                 # 파일 없음
│   ├── FileReadError                     # 읽기 실패
│   └── UnsupportedFormatError            # 지원하지 않는 포맷
├── PipelineError (+ stage, handler)      # 파이프라인 단계 실패
│   ├── ConversionError                   # Stage 1 실패
│   ├── PreprocessingError                # Stage 2 실패
│   ├── ExtractionError                   # Stage 3/4 실패
│   └── PostprocessingError               # Stage 5 실패
├── HandlerError                          # 핸들러 레벨
│   ├── HandlerNotFoundError              # 확장자에 대한 핸들러 없음
│   └── HandlerExecutionError             # 핸들러 실행 중 오류
├── ServiceError                          # 서비스 레벨
│   ├── ImageServiceError                 # 이미지 처리 실패
│   ├── StorageError                      # 저장소 실패
│   └── OCRError                          # OCR 실패
└── ChunkingError                         # 청킹 실패
```

**PipelineError** 특성: `stage`와 `handler` 필드를 추가로 포함하여 어느 단계에서 어떤 핸들러가 실패했는지 정확히 추적 가능.

---

## 6. Service Layer

> 정의 위치: `services/`

서비스는 **포맷에 독립적인 공유 기능**을 제공합니다. `DocumentProcessor`가 한 번 생성하고 모든 핸들러에 주입합니다.

### 6.1 Service Dependency Graph

```
DocumentProcessor._create_services()
  │
  ├── TagService(config)                          ← 독립
  │     │
  │     ├──▶ ImageService(config, storage, tag_service)    ← TagService 의존
  │     │       └── LocalStorageBackend(base_path)
  │     │
  │     └──▶ ChartService(config, tag_service)             ← TagService 의존
  │
  ├── TableService(config)                        ← 독립
  │
  └── MetadataService(config)                     ← 독립
```

**핵심 원칙:** TagService가 최초 생성되며, 태그 형식의 **Single Source of Truth** 역할. ImageService와 ChartService는 태그를 직접 생성하지 않고 TagService에 위임합니다.

### 6.2 Service Details

| Service | 역할 | 주요 API |
|---------|------|----------|
| **TagService** | 구조적 태그 생성/탐지/제거, Pre-compiled regex | `create_page_tag(n)`, `create_image_tag(path)`, `find_page_tags(text)`, `has_page_markers(text)` |
| **ImageService** | 이미지 저장/중복제거/태그 생성, SHA-256 해시 | `save(data)` → path, `save_and_tag(data)` → tag string |
| **ChartService** | ChartData → 태그 감싼 텍스트 블록 | `format_chart(data)`, `format_chart_fallback(...)` |
| **TableService** | TableData → HTML/Markdown/Text | `format_table(data)`, `format_as_html(data)` |
| **MetadataService** | DocumentMetadata → 태그 감싼 텍스트 블록 | `format_metadata(metadata)` |
| **StorageBackend** | 파일 저장/삭제/존재확인 | `save(data, path)`, `exists(path)`, `ensure_ready(dir)` |

### 6.3 ImageService 동작 상세

```
ContentExtractor.extract_images()
  │
  ├── image_bytes 추출 (포맷별 로직)
  │
  └── image_service.save_and_tag(image_bytes)
        │
        ├── Dedup check: SHA-256 해시 비교 → 중복이면 None 반환
        ├── Filename 생성: NamingStrategy에 따라 (hash/uuid/sequential/timestamp)
        ├── storage.save(data, path) → 파일 저장
        └── tag_service.create_image_tag(path) → "[Image:path/to/img.png]"
```

---

## 7. Pipeline Architecture

> 정의 위치: `pipeline/`

모든 핸들러는 이 5단계 파이프라인을 **반드시** 순서대로 실행합니다. 각 단계는 ABC로 정의되며, Null 구현이 제공됩니다.

### 7.1 Pipeline Stages

```
┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐   ┌───────────────┐
│  Stage 1     │   │  Stage 2     │   │  Stage 3          │   │  Stage 4         │   │  Stage 5      │
│  Converter   │──▶│ Preprocessor │──▶│ MetadataExtractor │──▶│ ContentExtractor │──▶│ Postprocessor │
│              │   │              │   │                   │   │                  │   │               │
│ FileContext  │   │ converted    │   │ preprocessed      │   │ preprocessed     │   │ Extraction    │
│ → format obj │   │ → Preproc.   │   │ → DocMetadata     │   │ → Extraction     │   │ Result → str  │
│              │   │   Data       │   │                   │   │   Result         │   │               │
└──────────────┘   └──────────────┘   └───────────────────┘   └──────────────────┘   └───────────────┘
```

### 7.2 Stage Definitions

#### Stage 1: Converter (`BaseConverter`)
| 항목 | 내용 |
|------|------|
| **입력** | `FileContext` (binary data + metadata) |
| **출력** | Format-specific object (fitz.Document, python-docx Document, etc.) |
| **추상 메서드** | `convert(file_context, **kwargs) → Any` |
| **추가 메서드** | `validate(file_context) → bool`, `close(obj) → None`, `get_format_name() → str` |
| **Null 구현** | `NullConverter` — raw bytes 그대로 반환 |
| **사용 예** | PDF: bytes → fitz.Document, DOCX: bytes → python-docx Document, Text: bytes → bytes |

#### Stage 2: Preprocessor (`BasePreprocessor`)
| 항목 | 내용 |
|------|------|
| **입력** | Converter가 반환한 format object |
| **출력** | `PreprocessedData(content, raw_content, encoding, resources, properties)` |
| **추상 메서드** | `preprocess(converted_data, **kwargs) → PreprocessedData` |
| **Null 구현** | `NullPreprocessor` — `PreprocessedData(content=input, raw_content=input)` |
| **사용 예** | PDF: 페이지 분석/복잡도 점수, DOCX: 네임스페이스 정규화, RTF: 컨트롤 코드 제거 |

#### Stage 3: MetadataExtractor (`BaseMetadataExtractor`)
| 항목 | 내용 |
|------|------|
| **입력** | `preprocessed.content` (format object 또는 전처리 결과) |
| **출력** | `DocumentMetadata` |
| **추상 메서드** | `extract(source) → DocumentMetadata` |
| **특성** | **Soft-fail**: 실패 시 metadata=None으로 설정하고 파이프라인 계속 진행 |
| **Null 구현** | `NullMetadataExtractor` — 빈 DocumentMetadata 반환 |
| **사용 예** | PDF: fitz.metadata, DOCX: core_properties, OLE: compound document properties |

#### Stage 4: ContentExtractor (`BaseContentExtractor`)
| 항목 | 내용 |
|------|------|
| **입력** | `PreprocessedData` + (optional) `DocumentMetadata` from Stage 3 |
| **출력** | `ExtractionResult` |
| **추상 메서드** | `extract_text(preprocessed) → str`, `get_format_name() → str` |
| **선택적 오버라이드** | `extract_tables() → List[TableData]`, `extract_images() → List[str]`, `extract_charts() → List[ChartData]` |
| **오케스트레이터** | `extract_all()` — 4개 메서드 호출 + 결과 조합 (보통 오버라이드 불요) |
| **서비스 의존** | `image_service`, `tag_service`, `chart_service`, `table_service` (생성자로 주입) |
| **Null 구현** | `NullContentExtractor` — 빈 문자열 반환 |

**extract_all() 오케스트레이션:**
```
extract_all(preprocessed, extract_metadata_result=metadata)
  │
  ├── 필수: extract_text(preprocessed) → str
  │   └── 실패 시: ExtractionError 발생 (파이프라인 중단)
  │
  ├── 선택: extract_tables(preprocessed) → List[TableData]
  │   └── 실패 시: warnings에 기록, 빈 리스트
  │
  ├── 선택: extract_images(preprocessed) → List[str]
  │   └── 실패 시: warnings에 기록, 빈 리스트
  │
  ├── 선택: extract_charts(preprocessed) → List[ChartData]
  │   └── 실패 시: warnings에 기록, 빈 리스트
  │
  └── return ExtractionResult(text, metadata, tables, charts, images, warnings)
```

#### Stage 5: Postprocessor (`BasePostprocessor`)
| 항목 | 내용 |
|------|------|
| **입력** | `ExtractionResult` |
| **출력** | `str` (최종 조립된 텍스트) |
| **추상 메서드** | `postprocess(result, include_metadata=True) → str` |
| **서비스 의존** | `metadata_service`, `tag_service` (생성자로 주입) |
| **기본 구현** | `DefaultPostprocessor`: 메타데이터 블록 prepend + whitespace 정규화 |
| **Null 구현** | `NullPostprocessor` — result.text 그대로 반환 |

**DefaultPostprocessor 처리:**
```
postprocess(result, include_metadata=True)
  │
  ├── 1. 메타데이터가 있고 include_metadata=True이면:
  │     metadata_service.format_metadata(result.metadata)
  │     text = metadata_block + "\n\n" + text
  │
  ├── 2. whitespace 정규화:
  │     - 3개 이상 연속 줄바꿈 → 2개로
  │     - 각 줄 끝 공백 제거
  │     - 전체 앞뒤 공백 제거
  │
  └── return text
```

### 7.3 Pipeline Component-Service Interaction Matrix

| Component | TagService | ImageService | ChartService | TableService | MetadataService |
|-----------|:----------:|:------------:|:------------:|:------------:|:---------------:|
| Converter | - | - | - | - | - |
| Preprocessor | - | - | - | - | - |
| MetadataExtractor | - | - | - | - | - |
| **ContentExtractor** | ✓ | ✓ | ✓ | ✓ | - |
| **Postprocessor** | ✓ | - | - | - | ✓ |

---

## 8. Handler Architecture

> 정의 위치: `handlers/`

### 8.1 BaseHandler (Template Method Pattern)

```python
class BaseHandler(ABC):
    # 생성자: 모든 핸들러 동일한 시그니처
    def __init__(self, config, *, image_service, tag_service,
                 chart_service, table_service, metadata_service):
        # 서비스 저장
        # 5개 팩토리 메서드 호출 → 파이프라인 컴포넌트 즉시 생성
        self._converter = self.create_converter()
        self._preprocessor = self.create_preprocessor()
        ...

    # ── 추상 팩토리 메서드 (서브클래스 반드시 구현) ──
    @abstractmethod
    def create_converter(self) -> BaseConverter: ...
    @abstractmethod
    def create_preprocessor(self) -> BasePreprocessor: ...
    @abstractmethod
    def create_metadata_extractor(self) -> BaseMetadataExtractor: ...
    @abstractmethod
    def create_content_extractor(self) -> BaseContentExtractor: ...
    @abstractmethod
    def create_postprocessor(self) -> BasePostprocessor: ...

    @property
    @abstractmethod
    def supported_extensions(self) -> FrozenSet[str]: ...

    @property
    @abstractmethod
    def handler_name(self) -> str: ...

    # ── 최종 메서드 (오버라이드 불가) ──
    def process(self, file_context, *, include_metadata=True, **kwargs) -> ExtractionResult:
        # Stage 0 → 1 → 2 → 3 → 4 → 5 (enforced)
        ...

    def extract_text(self, file_context, **kwargs) -> str:
        return self.process(file_context, **kwargs).text
```

### 8.2 Registered Handlers (14개)

| # | Handler | Extension(s) | Format | Stage 0 Delegation |
|---|---------|-------------|--------|-------------------|
| 1 | PDFHandler | pdf | PyMuPDF (fitz) | - |
| 2 | DOCXHandler | docx | python-docx (OOXML) | - |
| 3 | **DOCHandler** | doc | OLE2/CFBF | **ZIP→docx, RTF→rtf, HTML→future** |
| 4 | PPTXHandler | pptx | python-pptx (OOXML) | - |
| 5 | PPTHandler | ppt | OLE2 (LibreOffice) | - |
| 6 | XLSXHandler | xlsx | openpyxl (OOXML) | - |
| 7 | XLSHandler | xls | xlrd (BIFF) | - |
| 8 | CSVHandler | csv | stdlib csv | - |
| 9 | TSVHandler | tsv | stdlib csv (tab) | - |
| 10 | HWPHandler | hwp | Korean OLE binary | - |
| 11 | HWPXHandler | hwpx | Korean XML/ZIP | - |
| 12 | RTFHandler | rtf | RTF parser | - |
| 13 | TextHandler | txt, md, py, js, ... (60+) | Text (multi-ext category) | - |
| 14 | ImageFileHandler | jpg, png, gif, ... (12+) | Image (multi-ext category) | - |

### 8.3 One Extension Per Handler Rule

```
문서 포맷: 1 handler = 1 extension (strict)
  PPT(OLE) ≠ PPTX(OOXML) → 별도 핸들러
  XLS(BIFF) ≠ XLSX(OOXML) → 별도 핸들러
  CSV(comma) ≠ TSV(tab)   → 별도 핸들러

카테고리 핸들러: 1 handler = N extensions (예외)
  TextHandler: .txt/.md/.py/... → 모두 "plain text" 포맷
  ImageFileHandler: .jpg/.png/... → 모두 "raster image" 포맷
```

### 8.4 Delegation Pattern (Stage 0)

DOCHandler만이 _check_delegation()을 오버라이드합니다. `.doc` 파일은 실제로 4가지 포맷 중 하나일 수 있기 때문입니다.

```
DOCHandler._check_delegation(file_context)
  │
  ├── data[:2] == b"PK"      → ZIP magic → delegate to "docx"
  │     └── registry.get_handler("docx").process(file_context)
  │
  ├── data[:5] == b"{\rtf"   → RTF magic → delegate to "rtf"
  │     └── registry.get_handler("rtf").process(file_context)
  │
  ├── data[:256] starts with <html/<!DOCTYPE → HTML (현재 미구현, fallthrough)
  │
  └── OLE2 signature (D0CF11E0) 또는 미확인 → None (자체 파이프라인 진행)
```

**위임 메커니즘:**
```python
# BaseHandler._delegate_to()
def _delegate_to(self, extension, file_context, **kwargs):
    delegate = self._handler_registry.get_handler(extension)
    return delegate.process(file_context, **kwargs)
```

`HandlerRegistry.register()` 내에서 `handler.set_registry(self)`를 호출하여 핸들러에 레지스트리 참조를 주입합니다. 위임은 이 참조를 통해서만 가능합니다.

### 8.5 HandlerRegistry

```python
registry = HandlerRegistry(config, services={...})
registry.register_defaults()   # 14개 핸들러 importlib으로 로드

# 각 핸들러 등록 과정:
# 1. handler_class(config, **services)  → 인스턴스 생성
# 2. handler.set_registry(self)         → 레지스트리 주입
# 3. supported_extensions 순회          → ext → handler 매핑
```

---

## 9. Chunking Subsystem

> 정의 위치: `chunking/`

### 9.1 Architecture

```
TextChunker (Facade)
  │
  ├── chunk(text, file_extension, chunk_size, ...)
  │     │
  │     ├── Strategy 선택 (priority 순)
  │     │     ├── TableChunkingStrategy  (priority 5)  — CSV/TSV/XLSX/XLS
  │     │     ├── PageChunkingStrategy   (priority 10) — 페이지/슬라이드 마커 포함
  │     │     ├── ProtectedChunkingStrategy (priority 20) — 보호 영역 포함
  │     │     └── PlainChunkingStrategy  (priority 100) — 항상 True (fallback)
  │     │
  │     └── strategy.chunk(text, config) → List[str] | List[Chunk]
  │
  └── add_strategy(custom_strategy)  — 사용자 전략 추가 가능
```

### 9.2 Strategy Selection Logic

```python
for strategy in sorted_by_priority:
    if strategy.can_handle(text, config, file_extension=ext):
        return strategy.chunk(text, config)
# PlainChunkingStrategy always returns True → guaranteed fallback
```

### 9.3 Strategy Details

| Strategy | Priority | can_handle 조건 | 특징 |
|----------|:--------:|----------------|------|
| Table | 5 | ext ∈ {csv, tsv, xlsx, xls} | 헤더 복원, 테이블 청크 인덱싱, 오버랩 없음 |
| Page | 10 | text에 `[Page Number:` 또는 `[Slide Number:` 포함 | 페이지 경계 정렬, 1.5x 허용, 대형 페이지 재귀 분할 |
| Protected | 20 | text에 `<table`, `[chart]`, `[Image:` 등 포함 | 보호 영역 경계 존중, 대형 테이블 행 단위 분할 |
| Plain | 100 | 항상 True | 재귀적 문자 분할, 구분자: `\n\n` → `\n` → ` ` → `""` |

### 9.4 Protected Regions (constants.py)

분할 시 절대 깨뜨리지 않는 블록:

| 패턴 | 설명 |
|------|------|
| `<table>...</table>` | HTML 테이블 |
| `[chart]...[/chart]` | 차트 블록 |
| `[textbox]...[/textbox]` | 텍스트박스 |
| `[Image:...]` | 이미지 태그 |
| `[Page Number: N]` | 페이지 태그 |
| `[Slide Number: N]` | 슬라이드 태그 |
| `[Sheet: name]` | 시트 태그 |
| `<Document-Metadata>...</Document-Metadata>` | 메타데이터 블록 |
| `[Data Analysis]...[/Data Analysis]` | 데이터 분석 블록 |

---

## 10. OCR Subsystem

> 정의 위치: `ocr/`

### 10.1 Architecture

```
OCRProcessor (Orchestrator)
  │
  ├── process(text) → text (이미지 태그 → OCR 결과로 치환)
  │     │
  │     ├── 1. image tag 패턴 매칭 → 이미지 경로 추출
  │     ├── 2. 각 이미지에 대해:
  │     │     ├── 경로 검증 (존재, 0바이트 아닌지)
  │     │     ├── engine.convert_image_to_text(path)
  │     │     └── 태그를 OCR 결과로 치환
  │     └── 3. progress_callback 호출 (Protocol)
  │
  └── BaseOCREngine (ABC)
        │
        ├── convert_image_to_text(path) → str
        │     ├── encode_image_base64(path)
        │     ├── get_mime_type(path)
        │     ├── build_message_content(b64, mime, prompt) ← 추상 (provider별)
        │     └── llm_client.invoke([HumanMessage]) → response.content
        │
        ├── OpenAIOCREngine     → {"type":"image_url", "image_url":{"url":"data:..."}}
        ├── AnthropicOCREngine  → {"type":"image", "source":{"type":"base64",...}}
        ├── GeminiOCREngine     → {"type":"image_url", ...}
        ├── BedrockOCREngine    → Anthropic과 동일 형식
        └── VLLMOCREngine       → OpenAI와 동일 형식
```

### 10.2 Progress Reporting

```python
@dataclass(frozen=True)
class OCRProgressEvent:
    event_type: str       # 'tag_processing' | 'tag_processed' | 'completed'
    current_index: int    # 0-based
    total_count: int
    image_path: str
    status: str           # 'success' | 'failed' | ''
    error: str
```

---

## 11. Complete Execution Flow

### 11.1 Initialization Flow

```
DocumentProcessor(config=None, ocr_engine=None)
  │
  │  ┌─ 1. Config
  │  └── ProcessingConfig() (defaults if None)
  │
  │  ┌─ 2. Services
  │  └── _create_services()
  │       ├── tag_service     = TagService(config)
  │       ├── storage_backend = LocalStorageBackend(config.images.directory_path)
  │       ├── image_service   = ImageService(config, storage=..., tag_service=...)
  │       ├── chart_service   = ChartService(config, tag_service=...)
  │       ├── table_service   = TableService(config)
  │       └── metadata_service = MetadataService(config)
  │
  │  ┌─ 3. Registry
  │  └── HandlerRegistry(config, services=dict)
  │       └── register_defaults()
  │             ├── importlib PDFHandler → PDFHandler(config, **services)
  │             │     ├── create_converter()     → (format-specific)
  │             │     ├── create_preprocessor()  → (format-specific)
  │             │     ├── create_metadata_extractor() → (format-specific)
  │             │     ├── create_content_extractor()  → (format-specific)
  │             │     └── create_postprocessor() → DefaultPostprocessor(config, metadata_service, tag_service)
  │             │   set_registry(self) → 핸들러에 레지스트리 주입
  │             │   _handlers["pdf"] = handler
  │             ├── importlib DOCXHandler → ...
  │             ├── ... (12 more)
  │             └── _handlers = {"pdf":PDFHandler, "docx":DOCXHandler, "doc":DOCHandler, ...}
  │
  │  ┌─ 4. Chunker
  │  └── TextChunker(config)
  │       └── strategies = [Table(5), Page(10), Protected(20), Plain(100)]
  │
  │  ┌─ 5. OCR (optional)
  │  └── OCRProcessor(engine, config) if engine provided
  │
  └── Ready.
```

### 11.2 `extract_text("document.pdf")` — Full Trace

```
DocumentProcessor.extract_text("document.pdf", extract_metadata=True, ocr_processing=False)
  │
  ├── 1. 파일 검증
  │     os.path.exists("document.pdf") → True
  │
  ├── 2. 확장자 결정
  │     _resolve_extension("document.pdf", None) → "pdf"
  │
  ├── 3. FileContext 생성
  │     _create_file_context("document.pdf", "pdf")
  │     → FileContext {
  │         file_path:      "C:/.../document.pdf"
  │         file_name:      "document.pdf"
  │         file_extension: "pdf"
  │         file_category:  "document"
  │         file_data:      b"\x25\x50\x44\x46..."  (전체 바이너리)
  │         file_stream:    io.BytesIO(file_data)
  │         file_size:      12345
  │       }
  │
  ├── 4. 핸들러 조회
  │     registry.get_handler("pdf") → PDFHandler instance
  │
  ├── 5. 핸들러 파이프라인 실행
  │     handler.extract_text(file_context, include_metadata=True)
  │       └── handler.process(file_context, include_metadata=True)
  │             │
  │             ├── 【Stage 0: Delegation Check】
  │             │     _check_delegation(file_context) → None  (PDF는 위임 없음)
  │             │
  │             ├── 【Stage 1: Convert】
  │             │     converter.validate(file_context) → True
  │             │     converted = converter.convert(file_context)
  │             │     → fitz.Document  (예시: PyMuPDF 문서 객체)
  │             │
  │             ├── 【Stage 2: Preprocess】
  │             │     preprocessed = preprocessor.preprocess(converted)
  │             │     → PreprocessedData {
  │             │         content: fitz.Document (정리된)
  │             │         raw_content: fitz.Document (원본)
  │             │         encoding: "utf-8"
  │             │         resources: {images: [...]}
  │             │         properties: {page_complexities: [...]}
  │             │       }
  │             │
  │             ├── 【Stage 3: Metadata Extract】 (soft-fail)
  │             │     metadata = metadata_extractor.extract(preprocessed.content)
  │             │     → DocumentMetadata {
  │             │         title: "My Document"
  │             │         author: "John Doe"
  │             │         create_time: datetime(2024,1,15)
  │             │         page_count: 10
  │             │       }
  │             │     ⚠ 실패 시: metadata = None, 파이프라인 계속
  │             │
  │             ├── 【Stage 4: Content Extract】
  │             │     result = content_extractor.extract_all(preprocessed, metadata)
  │             │     │
  │             │     ├── text = extract_text(preprocessed)
  │             │     │   "[Page Number: 1]\n첫 번째 페이지 내용...\n[Image:img_001.png]\n..."
  │             │     │
  │             │     ├── tables = extract_tables(preprocessed)
  │             │     │   [TableData(rows=[...]), TableData(rows=[...])]
  │             │     │
  │             │     ├── images = extract_images(preprocessed)
  │             │     │   ["[Image:img_001.png]", "[Image:img_002.png]"]
  │             │     │
  │             │     ├── charts = extract_charts(preprocessed)
  │             │     │   [ChartData(chart_type="barChart", ...)]
  │             │     │
  │             │     └── return ExtractionResult {
  │             │           text: "[Page Number: 1]\n첫 번째 페이지..."
  │             │           metadata: DocumentMetadata(title="My Document",...)
  │             │           tables: [TableData(...)]
  │             │           charts: [ChartData(...)]
  │             │           images: ["[Image:img_001.png]"]
  │             │           warnings: []
  │             │         }
  │             │
  │             ├── 【Stage 5: Postprocess】
  │             │     final_text = postprocessor.postprocess(result, include_metadata=True)
  │             │     │
  │             │     ├── metadata_service.format_metadata(metadata)
  │             │     │   → "<Document-Metadata>\n  제목: My Document\n  작성자: John Doe\n  ..."
  │             │     │
  │             │     ├── text = metadata_block + "\n\n" + result.text
  │             │     │
  │             │     └── _normalize_text(text)
  │             │         → 3+ 줄바꿈 → 2개, 줄 끝 공백 제거, 앞뒤 공백 제거
  │             │
  │             ├── result.text = final_text
  │             │
  │             └── finally: converter.close(converted)
  │
  │   → return result.text  (문자열)
  │
  ├── 6. OCR 후처리 (ocr_processing=False이므로 건너뜀)
  │
  └── return text
```

### 11.3 `extract_text("legacy.doc")` — Delegation Flow

```
DocumentProcessor.extract_text("legacy.doc")
  │
  ├── extension = "doc"
  ├── handler = registry.get_handler("doc") → DOCHandler
  │
  └── handler.process(file_context)
        │
        ├── 【Stage 0: _check_delegation()】
        │     data = file_context["file_data"]
        │
        │     ┌── data[:2] == b"PK"?
        │     │   YES → "DOCX detected"
        │     │   delegate = registry.get_handler("docx") → DOCXHandler
        │     │   return DOCXHandler.process(file_context)
        │     │          │
        │     │          ├── Stage 0: None (DOCX는 위임 없음)
        │     │          ├── Stage 1: DOCX Converter
        │     │          ├── ...
        │     │          └── return ExtractionResult
        │     │
        │     ├── data[:5] == b"{\rtf"?
        │     │   YES → delegate to RTFHandler
        │     │
        │     ├── HTML markers?
        │     │   YES → (미구현, fallthrough)
        │     │
        │     └── OLE2 or unknown
        │         return None → DOC 자체 파이프라인 진행
        │
        ├── 【Stage 1-5: DOC Pipeline 실행】
        │     (OLE2 기반 처리)
        │
        └── return ExtractionResult
```

### 11.4 `extract_chunks("document.pdf", chunk_size=2000)` — Full Flow

```
DocumentProcessor.extract_chunks("document.pdf", chunk_size=2000, include_position_metadata=True)
  │
  ├── 1. extract_text("document.pdf", ...) → text (위 11.2와 동일)
  │
  ├── 2. _resolve_extension → "pdf"
  │
  ├── 3. chunk_text(text, chunk_size=2000, file_extension="pdf",
  │                  include_position_metadata=True)
  │     │
  │     └── chunker.chunk(text, chunk_size=2000, file_extension="pdf", ...)
  │           │
  │           ├── Strategy 선택:
  │           │   ├── TableStrategy.can_handle("pdf") → False (pdf ∉ TABLE_EXTENSIONS)
  │           │   ├── PageStrategy.can_handle() → "[Page Number:" in text → True ✓
  │           │   └── 선택: PageChunkingStrategy
  │           │
  │           └── page_strategy.chunk(text, config, ...)
  │                 → [Chunk(text="...", metadata=ChunkMetadata(chunk_index=0, page=1)),
  │                    Chunk(text="...", metadata=ChunkMetadata(chunk_index=1, page=3)),
  │                    ...]
  │
  └── 4. ChunkResult 조립
        → ChunkResult {
            chunks: ["...", "...", ...],
            chunks_with_metadata: [Chunk(...), Chunk(...), ...],
            source_file: "document.pdf"
          }
```

### 11.5 OCR Processing Flow

```
extract_text("document.pdf", ocr_processing=True)
  │
  ├── 1-5. 일반 추출 → text
  │     text = "...[Image:temp/images/img_001.png]...텍스트...[Image:temp/images/img_002.png]..."
  │
  └── 6. ocr_processor.process(text)
        │
        ├── _extract_image_paths(text) → ["temp/images/img_001.png", "temp/images/img_002.png"]
        │
        ├── for each image_path:
        │     ├── _resolve_image_path(path) → "/absolute/path/temp/images/img_001.png"
        │     │
        │     ├── engine.convert_image_to_text(absolute_path)
        │     │     ├── base64 인코딩
        │     │     ├── MIME type 결정 (png → image/png)
        │     │     ├── build_message_content(b64, mime, prompt)  ← provider별
        │     │     │
        │     │     │   OpenAI:    [{"type":"text","text":prompt},
        │     │     │               {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
        │     │     │
        │     │     │   Anthropic: [{"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}},
        │     │     │               {"type":"text","text":prompt}]
        │     │     │
        │     │     ├── llm_client.invoke([HumanMessage(content=payload)])
        │     │     └── return "[Figure:OCR 결과 텍스트]"
        │     │
        │     └── _replace_tag(text, path, ocr_result)
        │           "[Image:temp/images/img_001.png]" → "[Figure:표 형태의 데이터가 포함...]"
        │
        └── return text  (모든 이미지 태그가 OCR 결과로 치환됨)
```

---

## 12. Design Patterns Summary

| Pattern | 적용 위치 | 설명 |
|---------|----------|------|
| **Template Method** | `BaseHandler.process()` | 파이프라인 순서 고정, 서브클래스는 팩토리 메서드로만 커스터마이즈 |
| **Abstract Factory** | `BaseHandler.create_*()` | 각 핸들러가 자신의 파이프라인 컴포넌트 생성 |
| **Strategy** | `TextChunker` + `BaseChunkingStrategy` | 청킹 알고리즘 교체 가능, priority 기반 자동 선택 |
| **Facade** | `DocumentProcessor` | 복잡한 내부를 `extract_text()`/`process()` 단순 API로 은닉 |
| **Registry** | `HandlerRegistry` | 확장자 → 핸들러 매핑, 런타임 등록 |
| **Delegation Hook** | `_check_delegation()` | Stage 0에서 포맷 감지 후 다른 핸들러로 위임 |
| **Dependency Injection** | 서비스 → 핸들러 → 파이프라인 컴포넌트 | 모든 의존성 생성자 주입, 테스트 용이 |
| **Null Object** | `NullConverter`, `NullPreprocessor`, ... | 불필요한 단계를 안전하게 건너뜀 |
| **Builder** | `ProcessingConfig.with_*()` | frozen dataclass에 대한 fluent 수정 |
| **Observer / Callback** | `OCRProgressCallback` | OCR 진행률 보고용 Protocol |

---

## 13. Known Issues & Future Work

### 13.1 Minor Issues (⚠️)

| # | 이슈 | 설명 | 권장 조치 |
|---|------|------|-----------|
| 1 | **Chunking Constants vs Config** | `chunking/constants.py`의 태그 패턴이 하드코딩됨. `TagConfig` 변경 시 불일치 가능 | 전략 구현 시 config.tags 사용, constants는 기본값 참조만 |
| 2 | **OCR Pattern ↔ TagConfig** | `OCRProcessor`의 기본 이미지 태그 패턴이 `TagConfig`와 별도. 사용자가 태그 형식 변경 시 불일치 | `DocumentProcessor`에서 OCRProcessor 생성 시 tag_config 기반 패턴 전달 |
| 3 | **ImageService Dedup State** | `_processed_hashes`가 DocumentProcessor 수명 동안 누적. 다중 파일 처리 시 cross-file dedup 발생 | 파일 간 `clear_state()` 호출 또는 의도적 feature로 문서화 |

### 13.2 미구현 (TODO)

| 영역 | 상태 | 설명 |
|------|------|------|
| 14개 핸들러 파이프라인 컴포넌트 | 스켈레톤 (Null 구현) | 각 포맷의 Converter, Preprocessor, MetadataExtractor, ContentExtractor 구현 필요 |
| 4개 청킹 전략 | `NotImplementedError` | PageStrategy, TableStrategy, ProtectedStrategy, PlainStrategy 구현 필요 |
| HTML Handler | 미등록 | DOCHandler 위임 대상 중 HTML만 미구현 |
| Cloud Storage Backend | ABC만 존재 | MinIO, S3, Azure Blob, GCS |
| Config Validation | 미구현 | `ProcessingConfig`에 대한 값 범위 검증 (chunk_size > 0 등) |

### 13.3 Architecture Validation Summary

| 검증 항목 | 결과 | 비고 |
|----------|------|------|
| 모든 핸들러 동일 생성자 시그니처 | ✅ | `(config, *, image_service, tag_service, chart_service, table_service, metadata_service)` |
| Postprocessor 생성자 매칭 | ✅ | `DefaultPostprocessor(config, metadata_service=..., tag_service=...)` — 14개 핸들러 일치 |
| 서비스 의존 그래프 순환 없음 | ✅ | TagService → ImageService/ChartService, 역방향 의존 없음 |
| FileContext 필수 필드 완비 | ✅ | file_path, file_name, file_extension, file_category, file_data, file_stream, file_size |
| 파이프라인 단계 순서 강제 | ✅ | BaseHandler.process()에서 고정, 서브클래스 오버라이드 불가 |
| 에러 계층 커버리지 | ✅ | 파이프라인 5단계 각각에 대응하는 에러 클래스 존재 |
| One Extension Per Handler | ✅ | 12개 문서 핸들러: exact 1, 2개 카테고리 핸들러: multi (의도적 예외) |
| Config 불변성 | ✅ | frozen dataclass, with_*() → replace() |
| 21개 파일 문법 검증 | ✅ | 이전 세션에서 21/21 AST 통과 |

---

> **이 문서는 Contextifier v2의 인터페이스 골격(skeleton)에 대한 최종 아키텍처 명세입니다.**
> 모든 ABC, 타입, 서비스, 핸들러 등록, 파이프라인 순서가 확정되었으며,
> 이제 각 핸들러의 구체적인 파이프라인 컴포넌트 구현 단계(concrete implementation phase)로 진행할 수 있습니다.
