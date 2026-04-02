# Contextifier v0.2.5 — 아키텍처 및 현재 상태 분석

> 분석 일자: 2025-04-01
> 분석 대상: Contextifier v0.2.5 전체 코드베이스
> Python 파일 수: 181개 | 총 코드 라인: 26,536 LOC | 테스트: 93개 (796 LOC)

---

## 1. 프로젝트 개요

**Contextifier**는 다양한 문서 포맷(PDF, DOCX, PPTX, XLSX, HWP, RTF, CSV, 이미지 등)의 원본 파일을 AI/LLM이 이해할 수 있는 구조화된 텍스트로 변환하는 Python 라이브러리이다.

### 핵심 가치 제안
- **다중 포맷 지원**: 17개 핸들러로 30+ 파일 확장자 처리
- **통합 파이프라인**: 모든 포맷에 동일한 5단계 파이프라인 적용
- **AI 최적화 출력**: 태그 기반 구조화, 테이블 보존, 이미지 참조, 메타데이터 포함
- **유연한 청킹**: 테이블/페이지 보존형 전략적 텍스트 분할
- **OCR 통합**: 비전 LLM 기반 이미지-텍스트 변환

---

## 2. 아키텍처 전체 구조

### 2.1 상위 레벨 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   DocumentProcessor                      │
│                    (Facade Pattern)                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐ │
│  │HandlerRegistry│  │ TextChunker│  │  OCRProcessor   │ │
│  │  (Registry)   │  │ (Strategy) │  │  (Optional)     │ │
│  └──────┬───────┘  └────────────┘  └─────────────────┘ │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │              BaseHandler (ABC)                    │    │
│  │         Template Method Pattern                   │    │
│  │                                                   │    │
│  │  Stage 0: _check_delegation()                     │    │
│  │  Stage 1: Converter.convert()                     │    │
│  │  Stage 2: Preprocessor.preprocess()               │    │
│  │  Stage 3: MetadataExtractor.extract()             │    │
│  │  Stage 4: ContentExtractor.extract_all()          │    │
│  │  Stage 5: Postprocessor.postprocess()             │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Shared Services                      │    │
│  │  ImageService │ TagService │ ChartService         │    │
│  │  TableService │ MetadataService │ Storage         │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 대안 프로세서 (Wrapper 패턴)

```
DocumentProcessor (동기, 핵심)
    ├── AsyncDocumentProcessor (asyncio.to_thread 기반 비동기)
    └── CachedDocumentProcessor (콘텐츠 해시 기반 캐싱)
```

---

## 3. 모듈 구조 상세

### 3.1 루트 모듈 (`contextifier/`)

| 파일 | LOC | 역할 | 패턴 |
|------|-----|------|------|
| `__init__.py` | 62 | Public API 노출 | Facade |
| `config.py` | 248 | 통합 설정 시스템 (frozen dataclass, `with_*()` 빌더) | Builder, Immutable |
| `types.py` | 367 | 공유 타입, Enum, TypedDict, 데이터 클래스 | Value Object |
| `errors.py` | 169 | 통합 예외 계층 (15개 에러 클래스, 에러 코드) | Hierarchy, Fluent |
| `document_processor.py` | 465 | 메인 Facade (extract_text, process, chunk) | Facade, DI |
| `async_processor.py` | 155 | 비동기 래퍼 (asyncio.to_thread) | Adapter |
| `cached_processor.py` | 140 | 캐시 래퍼 (SHA256 + config hash) | Decorator |

### 3.2 설정 시스템 세부

```
ProcessingConfig (root, frozen)
├── TagConfig          — 페이지/슬라이드/시트/이미지/차트/메타데이터 태그 형식
├── ImageConfig        — 이미지 저장 경로, 네이밍 전략, 품질
├── ChartConfig        — 차트 포매팅 (HTML 테이블 사용 여부)
├── MetadataConfig     — 메타데이터 언어(ko/en), 날짜 형식
├── TableConfig        — 테이블 출력 형식 (HTML/Markdown/Text)
├── ChunkingConfig     — 청킹 크기/오버랩/전략
├── OCRConfig          — OCR 활성화/프로바이더/프롬프트
└── format_options     — 핸들러별 옵션 (MappingProxyType으로 동결)
```

### 3.3 파이프라인 모듈 (`contextifier/pipeline/`)

| 파일 | LOC | 역할 |
|------|-----|------|
| `converter.py` | 100 | Stage 1 ABC: 바이너리 → 포맷 객체 |
| `preprocessor.py` | 82 | Stage 2 ABC: 정규화/전처리 |
| `metadata_extractor.py` | ~60 | Stage 3 ABC: 메타데이터 추출 |
| `content_extractor.py` | 297 | Stage 4 ABC: 텍스트/테이블/이미지/차트 추출 |
| `postprocessor.py` | 148 | Stage 5 ABC + DefaultPostprocessor |

### 3.4 서비스 레이어 (`contextifier/services/`)

| 서비스 | LOC | 역할 | 특징 |
|--------|-----|------|------|
| `ImageService` | 244 | 이미지 저장/중복제거/태그 생성 | Thread-local 상태, Strategy 네이밍 |
| `TagService` | 167 | 구조 태그 생성/파싱 | 사전 컴파일된 regex, Config-driven |
| `ChartService` | 206 | 차트 데이터 포매팅 | OOXML 타입 매핑, 이중 렌더링 |
| `TableService` | 124 | 테이블 포매팅 (HTML/MD/Text) | HTML 이스케이프, Static 폴백 |
| `MetadataService` | 124 | 메타데이터 포매팅 | 이중 언어(ko/en), 날짜 형식 |
| `LocalStorageBackend` | ~50 | 로컬 파일 시스템 저장 | BaseStorageBackend 구현체 |
| `LibreOfficeHelper` | 133 | LibreOffice headless 변환 | ⚠️ 제거 대상 (외부 도구 의존) |

### 3.5 청킹 시스템 (`contextifier/chunking/`)

| 파일 | LOC | 역할 |
|------|-----|------|
| `chunker.py` | 202 | TextChunker Facade (전략 선택/편성) |
| `constants.py` | 178 | 공유 상수, regex 패턴, 데이터클래스 |
| `table_chunker.py` | 236 | 테이블 행 분할 (HTML & Markdown) |
| `table_parser.py` | 162 | 테이블 구조 파싱 |
| `strategies/base.py` | 87 | 전략 ABC |
| `strategies/plain_strategy.py` | 182 | 기본 텍스트 분할 |
| `strategies/page_strategy.py` | 264 | 페이지 경계 인식 분할 |
| `strategies/protected_strategy.py` | 480 | 보호 영역(테이블/태그) 보존 분할 |
| `strategies/table_strategy.py` | 304 | 테이블 포함 문서 특화 분할 |

**전략 우선순위:**
1. `TableChunkingStrategy` (p=5) — 테이블 포함 문서
2. `PageChunkingStrategy` (p=10) — 페이지 마커 기반
3. `ProtectedChunkingStrategy` (p=20) — 보호 영역 보존
4. `PlainChunkingStrategy` (p=100) — 기본 폴백

### 3.6 OCR 시스템 (`contextifier/ocr/`)

| 파일 | LOC | 역할 |
|------|-----|------|
| `base.py` | 150 | BaseOCREngine ABC (비전 LLM 인터페이스) |
| `processor.py` | 184 | OCRProcessor (이미지 태그 → OCR 텍스트 치환) |
| `engines/bedrock_engine.py` | 85 | AWS Bedrock 구현체 |
| `engines/openai_engine.py` | ~80 | OpenAI 구현체 |
| `engines/anthropic_engine.py` | ~80 | Anthropic 구현체 |
| `engines/gemini_engine.py` | ~80 | Google Gemini 구현체 |
| `engines/vllm_engine.py` | ~80 | vLLM 로컬 구현체 |

### 3.7 핸들러 (`contextifier/handlers/`)

```
handlers/
├── base.py        (509 LOC) — BaseHandler ABC
├── registry.py    (236 LOC) — HandlerRegistry + Plugin Discovery
├── pdf/           — PDF 라우터 (plus/default 선택)
├── pdf_default/   — 기본 PDF (PyMuPDF)
├── pdf_plus/      — 고급 PDF (12+ 전문 모듈, 복잡도 분석)
├── docx/          — OOXML Word
├── doc/           — OLE2 Word (레거시)
├── pptx/          — OOXML PowerPoint
├── ppt/           — OLE2 PowerPoint (레거시)
├── xlsx/          — OOXML Excel
├── xls/           — BIFF Excel (레거시)
├── csv/           — CSV
├── tsv/           — TSV (CSV 재사용)
├── hwp/           — HWP OLE2
├── hwpx/          — HWPX ZIP/XML
├── rtf/           — Rich Text Format
├── text/          — 플레인 텍스트 (40+ 확장자)
├── image/         — 이미지 (15+ 포맷)
└── html/          — HTML/XHTML
```

---

## 4. 핵심 설계 패턴

| 패턴 | 적용 위치 | 효과 |
|------|----------|------|
| **Facade** | DocumentProcessor | 복잡한 내부 구조를 단순 API로 노출 |
| **Template Method** | BaseHandler.process() | 파이프라인 순서 강제, 커스터마이즈 허용 |
| **Strategy** | 청킹 전략, StorageBackend, CacheBackend | 교체 가능한 알고리즘 |
| **Factory Method** | BaseHandler의 create_*() 메서드 | 핸들러별 파이프라인 컴포넌트 생성 |
| **Registry** | HandlerRegistry | 확장자→핸들러 매핑, 플러그인 디스커버리 |
| **Decorator** | CachedDocumentProcessor | 투명한 캐싱 레이어 |
| **Adapter** | AsyncDocumentProcessor | 동기→비동기 변환 |
| **DI (Dependency Injection)** | 서비스 → 핸들러 → 추출기 | 테스트 용이성, 유연성 |
| **Null Object** | NullConverter, NullPreprocessor 등 | 미구현 단계에 안전한 기본값 |
| **Builder/With** | ProcessingConfig.with_*() | Immutable 설정 변경 |
| **Value Object** | 모든 Config/Data 클래스 (frozen=True) | 불변성 보장 |
| **Observer** | OCRProgressCallback | 진행 상황 콜백 |

---

## 5. 타입 시스템

### 5.1 열거형 (Enum)
- `FileCategory` (12종): DOCUMENT, PRESENTATION, SPREADSHEET, TEXT, CODE, CONFIG, DATA, SCRIPT, LOG, WEB, IMAGE, UNKNOWN
- `OutputFormat` (3종): HTML, MARKDOWN, TEXT
- `ImageFormat` (8종): PNG, JPEG, JPG, GIF, BMP, WEBP, TIFF, UNKNOWN
- `NamingStrategy` (4종): HASH, UUID, SEQUENTIAL, TIMESTAMP
- `StorageType` (5종): LOCAL, MINIO, S3, AZURE_BLOB, GCS
- `TagType` (3종): PAGE, SLIDE, SHEET
- `PipelineStage` (5종): CONVERT, PREPROCESS, EXTRACT_METADATA, EXTRACT_CONTENT, POSTPROCESS
- `MetadataField` (12종): TITLE, SUBJECT, AUTHOR 등

### 5.2 데이터 클래스
- `FileContext` (TypedDict) — 7필드 표준 파일 입력
- `DocumentMetadata` — 12 표준 필드 + custom dict + to_dict/from_dict 직렬화
- `TableData` / `TableCell` — 테이블 구조 (행/열/병합/헤더)
- `ChartData` / `ChartSeries` — 차트 구조 (타입/제목/카테고리/시리즈)
- `ExtractionResult` — 파이프라인 출력 (text, metadata, tables, charts, images, warnings)
- `Chunk` / `ChunkMetadata` — 청킹 결과 (text, metadata, page_number, chunk_index)
- `PreprocessedData` — 전처리 결과 (content, raw_content, properties)

### 5.3 프로토콜
- `CacheBackend` — get/set/has 캐시 인터페이스
- `OCRProgressCallback` — 진행 콜백 프로토콜
- `BaseStorageBackend` — save/delete/exists 스토리지 인터페이스

---

## 6. 에러 계층

```
ContextifierError (base)
├── ConfigurationError
├── FileError
│   ├── FileNotFoundError
│   ├── FileReadError
│   └── UnsupportedFormatError
├── PipelineError
│   ├── ConversionError
│   ├── PreprocessingError
│   ├── ExtractionError
│   └── PostprocessingError
├── HandlerError
│   ├── HandlerNotFoundError
│   └── HandlerExecutionError
├── ServiceError
│   ├── ImageServiceError
│   ├── StorageError
│   └── OCRError
└── ChunkingError
```

**특징:**
- 자동 에러 코드 생성 (클래스명 → `E_CONVERSION` 형태)
- `context` dict로 디버그 정보 첨부
- `cause` 체인으로 원인 추적
- `with_context()` fluent API

---

## 7. 의존성 구조

### 7.1 Core Dependencies (pyproject.toml)
```
Core:        beautifulsoup4, chardet, langchain-text-splitters
PDF:         pymupdf, pdfplumber, pdfminer.six, pdf2image
Office:      python-docx, docx2pdf, python-pptx, openpyxl, xlrd
HWP:         pyhwp, olefile
RTF:         striprtf
Image/OCR:   pi-heif, pytesseract
```

### 7.2 Optional Dependencies
```
langchain:   langchain, langchain-aws/community/core/openai/anthropic/google-genai, langgraph, langsmith
server:      pydantic, pydantic-settings, python-dotenv, python-multipart, orjson, psutil
all:         langchain + server + pandas + cachetools
```

---

## 8. 테스트 현황

| 분류 | 테스트 수 | 상태 |
|------|----------|------|
| **단위 테스트** | 93 | ✅ 전체 통과 (0.98s) |
| **통합 테스트** | 0 | ❌ 미구현 |
| **성능 테스트** | 0 | ❌ 미구현 |
| **E2E 테스트** | 0 | ❌ 미구현 |

### 테스트 커버리지 분석
```
✅ 테스트 존재: config, types, errors, document_processor, handlers/base,
               services/tag_service, chunking strategies
❌ 테스트 부재: 개별 핸들러 파이프라인, 서비스 (image/chart/table/metadata),
               OCR, async/cached 프로세서, storage, 위임 경로
```

**테스트 비율**: 796 LOC / 26,536 LOC = **3.0%** (매우 낮음)

---

## 9. Public API 총정리

### DocumentProcessor (메인 진입점)
```python
processor = DocumentProcessor(config=None, ocr_engine=None)
text      = processor.extract_text(file_path, extract_metadata=True, ocr_processing=False)
result    = processor.process(file_path, extract_metadata=True, ocr_processing=False)
chunks    = processor.extract_chunks(file_path, chunk_size=1000, chunk_overlap=200)
text_chks = processor.chunk_text(text, chunk_size=1000, chunk_overlap=200)
supported = processor.is_supported(ext)
exts      = processor.supported_extensions
```

### AsyncDocumentProcessor
```python
async_proc = AsyncDocumentProcessor(config=None, ocr_engine=None)
text   = await async_proc.extract_text(file_path)
result = await async_proc.process(file_path)
chunks = await async_proc.extract_chunks(file_path)
batch  = await async_proc.extract_batch([path1, path2], max_concurrent=4)
```

### CachedDocumentProcessor
```python
cached = CachedDocumentProcessor(config=None, backend=MemoryCacheBackend())
text = cached.extract_text(file_path)  # 자동 캐싱
```

### TextChunker
```python
chunker = TextChunker(config)
chunks = chunker.chunk(text, chunk_size=1000, chunk_overlap=200)
```

---

## 10. Phase 0-5 이전 개선 완료 항목 요약

| Phase | 항목 수 | 내용 |
|-------|---------|------|
| Phase 0 | 6 | HTML 이스케이프, 설정 검증, Thread Safety, Lazy file_stream |
| Phase 1 | 7 | 에러 계층, 설정 시스템, ExtractionResult 확장 |
| Phase 2 | 6 | 파이프라인 ABC, 서비스 레이어, 기본 핸들러 |
| Phase 3 | 8 | 개별 핸들러 마이그레이션 (17개) |
| Phase 4 | 7 | 위임 깊이 제한, Public API 확장, 청킹 전략 |
| Phase 5 | 7 | 테스트 인프라, 플러그인 시스템, 캐싱, 비동기 |

**총 41개 항목 완료**, 93개 테스트, 0 실패

---

## 11. 핸들러 매트릭스 요약

| 핸들러 | 텍스트 | 테이블 | 이미지 | 차트 | 메타 | 위임 | 등급 |
|--------|--------|--------|--------|------|------|------|------|
| pdf_plus | ✓✓ | ✓✓ | ✓✓ | ✓ | ✓ | - | ★★★★★ |
| docx | ✓✓ | ✓✓ | ✓ | ✓ | ✓✓ | - | ★★★★★ |
| pptx | ✓✓ | ✓ | ✓ | ~ | ✓ | - | ★★★★ |
| xlsx | ✓ | ✓✓ | ✓ | ? | ✓ | - | ★★★★ |
| html | ✓ | ✓ | ~ | ✗ | ✓ | - | ★★★ |
| csv | ✓✓ | N/A | ✗ | ✗ | ~ | - | ★★★ |
| tsv | ✓✓ | N/A | ✗ | ✗ | ~ | csv | ★★★ |
| hwpx | ✓ | ~ | ✓ | ? | ✓ | - | ★★★ |
| pdf_default | ✓ | ~ | ✓ | ✗ | ✓ | - | ★★☆ |
| rtf | ✓ | ~ | ✓ | ✗ | ~ | - | ★★☆ |
| hwp | ~ | ~ | ~ | ✗ | ~ | hwpx | ★★☆ |
| xls | ✓ | ✓ | ✗ | ✗ | ~ | xlsx | ★★ |
| doc | ~ | ✗ | ~ | ✗ | ~ | docx | ★★ |
| ppt | ~ | ✗ | ~ | ✗ | ~ | pptx | ★★ |
| text | ✓✓ | ✗ | ✗ | ✗ | ✗ | - | ★★ |
| image | ? | ✗ | N/A | ✗ | ~ | - | ★☆ |

> ✓✓=우수, ✓=양호, ~=부분, ✗=미지원, ?=불명, N/A=해당없음

---

## 12. 코드 규모 분석

### 모듈별 LOC 분포

| 모듈 | 파일 수 | LOC | 비율 |
|------|---------|-----|------|
| handlers/ (전체) | ~130 | ~19,000 | 71.6% |
| ├─ pdf_plus/ | 18 | ~4,500 | 17.0% |
| ├─ docx/ | 9 | ~1,700 | 6.4% |
| ├─ pptx/ | 8 | ~1,400 | 5.3% |
| ├─ hwp/ | 11 | ~1,400 | 5.3% |
| ├─ rtf/ | 8 | ~1,500 | 5.7% |
| ├─ xlsx/ | 8 | ~1,300 | 4.9% |
| ├─ 기타 핸들러 | ~68 | ~6,700 | 25.2% |
| chunking/ | 8 | ~2,100 | 7.9% |
| services/ | 7 | ~1,050 | 4.0% |
| pipeline/ | 5 | ~690 | 2.6% |
| ocr/ | 7 | ~750 | 2.8% |
| root modules | 7 | ~1,960 | 7.4% |
| **합계** | **181** | **26,536** | **100%** |

### 상위 10 대형 파일

| 순위 | 파일 | LOC |
|------|------|-----|
| 1 | pdf_plus/_layout_block_detector.py | 633 |
| 2 | pdf_plus/_table_detection.py | 596 |
| 3 | handlers/base.py | 509 |
| 4 | pdf_plus/_types.py | 509 |
| 5 | hwpx/_section.py | 496 |
| 6 | pptx/content_extractor.py | 481 |
| 7 | chunking/strategies/protected_strategy.py | 480 |
| 8 | document_processor.py | 465 |
| 9 | rtf/_cleaner.py | 456 |
| 10 | rtf/_table_parser.py | 446 |

---

## 13. 현재 상태 평가 종합

### 강점
1. **견고한 아키텍처**: Template Method + Strategy + DI 조합으로 확장성 우수
2. **타입 안전성**: Frozen dataclass, Enum, TypedDict 전방위 활용
3. **에러 시스템**: 계층적 예외, 자동 코드, 컨텍스트 첨부
4. **설정 불변성**: MappingProxyType + frozen dataclass로 런타임 변이 차단
5. **서비스 분리**: 포맷별 추출과 공통 포매팅이 명확히 분리
6. **플러그인 시스템**: entry_points 기반 서드파티 핸들러 디스커버리

### 약점
1. **테스트 부족**: 3% 커버리지, 통합/E2E 테스트 없음
2. **핸들러 편차**: pdf_plus(★★★★★) vs ppt(★★) — 품질 차이 큼
3. **레거시 포맷 한계**: DOC/PPT/XLS의 테이블/이미지/차트 미지원
4. **메모리 비효율**: 대용량 파일 스트리밍 없음
5. **OCR 통합 부족**: pdf_default에서 스캔 감지 후 무시
6. **캐시 한계**: extract_text()만 캐싱, FIFO 방출 전략
