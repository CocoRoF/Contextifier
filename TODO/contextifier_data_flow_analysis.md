# Contextifier 핵심 비즈니스 로직 및 데이터 흐름 분석

> 분석 대상: CocoRoF/Contextifier v0.2.4
> 분석일: 2026-03-25
> 분석 범위: 전체 데이터 흐름, 핵심 비즈니스 로직, 도메인 모델, 고도화 범위 정의
> 분석자: 손성준 (Developer Agent)

---

## 목차

1. [시스템 전체 아키텍처 조감도](#1-시스템-전체-아키텍처-조감도)
2. [핵심 도메인 모델 (타입 시스템)](#2-핵심-도메인-모델)
3. [메인 데이터 흐름: 문서 → 텍스트](#3-메인-데이터-흐름)
4. [핸들러 파이프라인 상세 흐름](#4-핸들러-파이프라인-상세-흐름)
5. [청킹(Chunking) 서브시스템 흐름](#5-청킹-서브시스템-흐름)
6. [OCR 서브시스템 흐름](#6-ocr-서브시스템-흐름)
7. [서비스 레이어 역할 분석](#7-서비스-레이어-역할-분석)
8. [위임(Delegation) 패턴 전체 맵](#8-위임-패턴-전체-맵)
9. [핵심 비즈니스 로직 위치 매핑](#9-핵심-비즈니스-로직-위치-매핑)
10. [발견된 데이터 흐름 문제점](#10-발견된-데이터-흐름-문제점)
11. [고도화 범위 정의](#11-고도화-범위-정의)

---

## 1. 시스템 전체 아키텍처 조감도

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PUBLIC API (사용자 진입점)                        │
│                         DocumentProcessor                                │
│  extract_text(path)  ──▶  process(path)  ──▶  extract_chunks(path)       │
└─────────────────────────────────────────┬───────────────────────────────┘
                                          │
          ┌────────────────┬──────────────┼──────────────┐
          │                │              │              │
          ▼                ▼              ▼              ▼
   HandlerRegistry    TextChunker   OCRProcessor    Services
   (확장자→핸들러)     (청킹전략)     (선택적 OCR)   (이미지/태그/...)
          │
          ▼ get_handler(ext)
   ┌──────────────────────────────────────┐
   │           BaseHandler (Abstract)      │
   │  5단계 파이프라인 (Template Method)     │
   │  ┌─────────────────────────────────┐ │
   │  │ Stage 0: _check_delegation()   │ │  ← 위임 체크 (마법 바이트)
   │  │ Stage 1: Converter             │ │  ← bytes → 포맷 객체
   │  │ Stage 2: Preprocessor          │ │  ← 전처리 + 리소스 추출
   │  │ Stage 3: MetadataExtractor     │ │  ← 메타데이터 추출
   │  │ Stage 4: ContentExtractor      │ │  ← 텍스트/표/이미지/차트 추출
   │  │ Stage 5: Postprocessor         │ │  ← 최종 조립 + 정규화
   │  └─────────────────────────────────┘ │
   └──────────────────────────────────────┘
          │ (14개 핸들러 각각 구현)
          ├── PDFHandler (pdf_plus / pdf_default 선택)
          ├── DOCXHandler
          ├── DOCHandler → (위임 가능: RTF/DOCX)
          ├── PPTXHandler
          ├── PPTHandler → (위임 가능: PPTX)
          ├── XLSXHandler
          ├── XLSHandler → (위임 가능: XLSX)
          ├── CSVHandler
          ├── TSVHandler
          ├── HWPHandler → (위임 가능: HWPX)
          ├── HWPXHandler
          ├── RTFHandler
          ├── TextHandler (50+ 확장자)
          └── ImageFileHandler
```

---

## 2. 핵심 도메인 모델

### 2.1 파이프라인 입력: FileContext

```python
class FileContext(TypedDict):
    file_path: str          # "/path/to/document.pdf"
    file_name: str          # "document.pdf"
    file_extension: str     # "pdf"
    file_category: str      # FileCategory.DOCUMENT.value = "document"
    file_data: bytes        # 파일 전체 바이너리 (⚠ OOM 위험)
    file_stream: io.BytesIO # 동일 데이터의 스트림 (⚠ 메모리 2배)
    file_size: int          # 파일 크기 (bytes)
```

**설계 의도**: 파일을 한 번 읽어 불변 컨텍스트로 전달. 모든 파이프라인 단계가 동일한 원본 참조.

**실제 문제**: `file_data`(bytes)와 `file_stream`(BytesIO)이 동일 데이터를 메모리에 두 번 보유.

### 2.2 파이프라인 중간 데이터: PreprocessedData

```python
@dataclass
class PreprocessedData:
    content: Any                      # 포맷별 파싱 결과 (fitz.Document, python-docx Document, ...)
    raw_content: Any = None           # 원본 참조 (필요시 되돌아갈 수 있음)
    encoding: str = "utf-8"           # 감지된 인코딩 (텍스트 파일용)
    resources: Dict[str, Any] = {}    # 사전 추출된 리소스 (차트, 이미지 관계 등)
    properties: Dict[str, Any] = {}   # 전처리 중 발견된 속성들
```

`resources` 딕셔너리는 파이프라인의 **사전 계산 캐시** 역할:
- DOCX: `{"charts": ["차트 텍스트1", ...], "chart_count": 3}`
- XLSX: `{"images": {"xl/media/image1.png": bytes}, "charts": [...], "textboxes": {...}}`
- PPTX: `{"charts": [...], "slide_images": [...]}`

### 2.3 파이프라인 출력: ExtractionResult

```python
@dataclass
class ExtractionResult:
    text: str = ""                    # 최종 추출 텍스트 (태그 포함)
    metadata: Optional[DocumentMetadata] = None
    tables: List[TableData] = []      # 구조화된 표 데이터
    charts: List[ChartData] = []      # 구조화된 차트 데이터
    images: List[str] = []            # 이미지 태그 문자열 리스트
    page_count: int = 0
    warnings: List[str] = []          # 부분 실패 정보 (Stage 4에서 기록)
```

**중요**: `text`에는 이미지/표/차트/페이지 태그가 **인라인 삽입**되어 있음:
```
[Page Number: 1]
본문 텍스트입니다.

<table border='1'>...</table>

[Image: temp/images/abc123.png]

[chart]
Chart Type: Bar
...
[/chart]

[Page Number: 2]
```

### 2.4 메타데이터: DocumentMetadata

```python
@dataclass
class DocumentMetadata:
    title: Optional[str]           # 문서 제목
    author: Optional[str]          # 작성자
    create_time: Optional[datetime] # 생성일
    last_saved_time: Optional[datetime]
    page_count: Optional[int]
    word_count: Optional[int]
    keywords: Optional[str]
    custom: Dict[str, Any]          # 포맷별 추가 메타데이터
```

### 2.5 표 도메인 모델

```python
TableData
├── rows: List[List[TableCell]]
│   └── TableCell
│       ├── content: str
│       ├── row_span: int           # 병합 행 정보
│       ├── col_span: int           # 병합 열 정보
│       ├── is_header: bool
│       └── nested_table: Optional[TableData]  # 중첩 표 지원
├── num_rows: int
├── num_cols: int
├── has_header: bool
└── caption: Optional[str]
```

### 2.6 차트 도메인 모델

```python
ChartData
├── chart_type: Optional[str]       # "Bar", "Line", "Pie" 등
├── title: Optional[str]
├── categories: List[str]           # X축 레이블
├── series: List[ChartSeries]
│   └── ChartSeries
│       ├── name: Optional[str]
│       └── values: List[Any]
└── raw_content: Optional[str]      # 파싱 불가 시 원시 텍스트
```

### 2.7 파일 카테고리 시스템

```python
class FileCategory(str, Enum):
    DOCUMENT     = "document"      # pdf, docx, doc, rtf, hwp, hwpx
    PRESENTATION = "presentation"  # ppt, pptx
    SPREADSHEET  = "spreadsheet"   # xlsx, xls
    TEXT         = "text"          # txt, md, markdown
    CODE         = "code"          # py, js, html, css, ...
    CONFIG       = "config"        # json, yaml, xml, toml, ...
    DATA         = "data"          # csv, tsv
    SCRIPT       = "script"        # sh, bat, ps1
    LOG          = "log"           # log
    WEB          = "web"           # htm, xhtml  (⚠ "html" 제외!)
    IMAGE        = "image"         # jpg, png, gif, ...
    UNKNOWN      = "unknown"
```

**발견된 불일치**: `types.py`의 `EXTENSION_CATEGORIES`에서:
- `"html"` → `FileCategory.CODE` (CODE 카테고리에 매핑됨)
- `"htm"`, `"xhtml"` → `FileCategory.WEB`
- 반면 `TextHandler._TEXT_EXTENSIONS`은 `html`, `htm`, `xhtml` 모두 포함

두 시스템이 `html` 처리 방법을 다르게 정의하고 있음 → **일관성 없는 분류**.

---

## 3. 메인 데이터 흐름

### 3.1 extract_text() 전체 흐름

```
사용자 코드:
    processor = DocumentProcessor()
    text = processor.extract_text("report.pdf")

─────────────────────────────────────────────────────────
DocumentProcessor.extract_text("report.pdf")
    │
    ├─ 1. 파일 존재 확인 (os.path.exists)
    ├─ 2. 확장자 결정 (_resolve_extension → "pdf")
    ├─ 3. FileContext 생성 (_create_file_context)
    │       file_data = Path("report.pdf").read_bytes()  ← ⚠ 전체 파일 메모리 로드
    │       file_stream = io.BytesIO(file_data)          ← ⚠ 동일 데이터 복사
    │
    ├─ 4. ImageService 상태 초기화 (image_service.clear_state())
    │       per-file 이미지 중복 제거 초기화
    │
    ├─ 5. HandlerRegistry에서 핸들러 조회 (get_handler("pdf"))
    │       → PDFHandler 인스턴스 반환
    │
    ├─ 6. handler.extract_text(file_context)
    │       → handler.process() 내부 호출
    │       → ExtractionResult.text 반환
    │
    └─ 7. OCR 후처리 (ocr_processing=True인 경우)
            image 태그들에서 OCR 수행
            → [Image: path.png] 태그를 OCR 텍스트로 교체
```

### 3.2 process() 전체 흐름

```
DocumentProcessor.process("report.pdf")
    │ (extract_text와 동일 흐름)
    └─ 6. handler.process(file_context)
            → ExtractionResult (text + metadata + tables + charts + images) 반환
```

### 3.3 extract_chunks() 전체 흐름

```
DocumentProcessor.extract_chunks("report.pdf", chunk_size=1000)
    │
    ├─ 1~7. extract_text()와 동일
    │
    └─ 8. TextChunker.chunk(text, file_extension="pdf", chunk_size=1000)
            │
            ├─ 전략 선택 (_select_strategy)
            │   ├─ TableChunkingStrategy.can_handle()?  → 표 파일(csv/xlsx)이면 True
            │   ├─ PageChunkingStrategy.can_handle()?   → [Page Number: N] 태그 있으면 True
            │   ├─ ProtectedChunkingStrategy.can_handle()? → 보호 영역 있으면 True
            │   └─ PlainChunkingStrategy.can_handle()   → 항상 True (fallback)
            │
            └─ strategy.chunk(text, config)
                    → List[str] 또는 List[Chunk] 반환

    └─ ChunkResult(chunks=[...], source_file="report.pdf") 반환
```

---

## 4. 핸들러 파이프라인 상세 흐름

### 4.1 BaseHandler.process() 내부 흐름

```
BaseHandler.process(file_context)
    │
    │ [Stage 0] 위임 체크
    ├─ _check_delegation(file_context)
    │   ├─ None 반환 → 자체 파이프라인 계속
    │   └─ ExtractionResult 반환 → 파이프라인 스킵, 바로 반환
    │
    │ [Stage 1] 변환
    ├─ converter.validate(file_context)  ← 파일 유효성 검증
    ├─ converter.convert(file_context)   ← bytes → 포맷 객체
    │   반환: converted (fitz.Document / docx.Document / openpyxl.Workbook / ...)
    │
    │ [Stage 2] 전처리
    ├─ preprocessor.preprocess(converted)
    │   반환: PreprocessedData
    │           content = 처리된 객체 (or converted 그대로)
    │           resources = {"charts": [...], "images": {...}, ...}
    │
    │ [Stage 3] 메타데이터
    ├─ (include_metadata=True인 경우)
    │   metadata_extractor.extract(preprocessed.content)
    │   반환: DocumentMetadata (실패 시 None + warning)
    │
    │ [Stage 4] 콘텐츠 추출
    ├─ content_extractor.extract_all(preprocessed, metadata)
    │   내부 순서:
    │   ├─ extract_text() → str (필수, 실패 시 ExtractionError)
    │   ├─ extract_tables() → List[TableData] (실패 시 warning + 빈 리스트)
    │   ├─ extract_images() → List[str] (실패 시 warning + 빈 리스트)
    │   └─ extract_charts() → List[ChartData] (실패 시 warning + 빈 리스트)
    │   반환: ExtractionResult
    │
    │ [Stage 5] 후처리
    ├─ postprocessor.postprocess(result, include_metadata)
    │   ├─ 메타데이터 블록 텍스트 앞에 삽입 (MetadataService)
    │   └─ 텍스트 정규화 (과도한 개행 제거, 공백 정리)
    │   반환: str (최종 텍스트)
    │
    ├─ result.text = final_text  ← ExtractionResult.text 업데이트
    │
    └─ finally: converter.close(converted)  ← 리소스 해제
```

### 4.2 DOCX 핸들러 상세 데이터 흐름

```
FileContext(file_data=docx_bytes)
    │
    ▼ [Stage 1] DocxConverter
python-docx Document 객체
    │
    ▼ [Stage 2] DocxPreprocessor
PreprocessedData(
    content = Document,
    resources = {
        "charts": ["[chart]Bar Chart...[/chart]", ...],  # 차트 텍스트 사전 추출
        "chart_count": 3
    }
)
    │
    ▼ [Stage 3] DocxMetadataExtractor
DocumentMetadata(
    title = doc.core_properties.title,
    author = doc.core_properties.author,
    create_time = doc.core_properties.created,
    ...
)
    │
    ▼ [Stage 4] DocxContentExtractor.extract_text()
body 순회 (doc.element.body의 XML 요소들):
    "p" (단락) →
        process_paragraph(element) → (text, drawings, picts, has_break)
        drawings:
            DrawingKind.IMAGE → _extract_image_by_rel(rel_id)
                → ImageService.save_and_tag(image_bytes) → "[Image: path]"
            DrawingKind.CHART → charts[chart_index] (인덱스 기반 매칭 ⚠)
            DrawingKind.DIAGRAM → extract_diagram_text()
        has_break → make_page_tag(page_number)
    "tbl" (표) →
        extract_table(element) → TableData
        _format_table(table_data) → HTML 또는 텍스트
→ parts 리스트 조립 → "\n\n".join(parts)
→ import re; re.sub(r"\n{3,}", "\n\n", result) ← ⚠ 함수 내부 import

    ▼ [Stage 5] DefaultPostprocessor
metadata_service.format_metadata(metadata) → 메타데이터 블록
텍스트 정규화 (re.sub)
→ final text
```

### 4.3 XLSX 핸들러 상세 데이터 흐름

```
FileContext(file_data=xlsx_bytes)
    │
    ▼ [Stage 1] XlsxConverter
openpyxl.Workbook 객체
    │
    ▼ [Stage 2] XlsxPreprocessor
PreprocessedData(
    content = Workbook,
    resources = {
        "images": {                           # ZIP에서 직접 추출
            "xl/media/image1.png": bytes,
            "xl/media/image2.jpeg": bytes,
        },
        "charts": [                           # 차트 데이터 파싱
            {"chart_type": "bar", "title": "...", "series": [...], "categories": []},
        ],
        "textboxes": {                        # 드로잉 XML에서 텍스트박스
            "Sheet1": ["텍스트박스 내용1", ...],
        }
    }
)
    │
    ▼ [Stage 4] XlsxContentExtractor.extract_text()
시트 순서대로:
    ws = wb[sheet_name]
    → make_sheet_tag(sheet_name) → "[Sheet: Sheet1]"
    → layout_detect_range(ws) → LayoutRange
    → object_detect(ws, layout) → List[LayoutRange] (여러 표 감지)
    → convert_sheet_to_text(ws, region) → Markdown 표
    → ws._charts (내부API ⚠) → charts[chart_index] 매칭 (인덱스 기반 ⚠)
    → _extract_sheet_images(ws, images, processed_hashes)
        → getattr(ws, "_images", []) (내부API ⚠)
        → img._data() (내부API ⚠)
        → image_service.save_and_tag(img_data, custom_name="excel_sheet_img") (이름 충돌 ⚠)
    → textboxes.get(sheet_name, [])
```

---

## 5. 청킹(Chunking) 서브시스템 흐름

### 5.1 전략 선택 로직

```
TextChunker.chunk(text, file_extension="pdf")
    │
    ├─ 전략 우선순위 (낮을수록 높은 우선순위):
    │   TableChunkingStrategy     priority=5
    │   PageChunkingStrategy      priority=10
    │   ProtectedChunkingStrategy priority=20
    │   PlainChunkingStrategy     priority=100
    │
    ├─ TableChunkingStrategy.can_handle(text, config, file_extension="pdf")
    │   → file_extension이 {"csv","tsv","xlsx","xls"}이면 True
    │   → 또는 텍스트에 표 마커가 있으면 True
    │   → "pdf"이면 False
    │
    ├─ PageChunkingStrategy.can_handle(text, ...)
    │   → text에 "[Page Number: N]" 패턴이 있으면 True
    │   → PDF 처리 결과에는 이 태그가 삽입되어 있음 → True
    │
    └─ PageChunkingStrategy.chunk(text, config) 실행
```

### 5.2 PageChunkingStrategy 흐름

```
텍스트 (페이지 태그 포함):
    "[Page Number: 1]\n텍스트1\n[Page Number: 2]\n텍스트2..."
    │
    ├─ 페이지별 분리 (regex: \[Page Number: \d+\])
    ├─ chunk_size 기준으로 페이지를 묶어서 청크 생성
    │   → 한 청크가 chunk_size를 초과하면 다음 청크로
    │   → 페이지 경계에서만 분할 (페이지 내용 중간에서 자르지 않음)
    └─ ChunkMetadata(chunk_index=i, page_number=start_page) 생성
```

### 5.3 ProtectedChunkingStrategy 흐름

```
보호 영역 마커: "[chart]...[/chart]", "<table>...</table>", "[Image:...]"
    │
    ├─ 보호 영역 감지 → 해당 영역은 분할 금지
    ├─ 보호 영역 외부에서만 청크 경계 결정
    └─ 보호 영역이 chunk_size 초과하면? → 단일 청크로 유지 (사이즈 초과 허용)
```

---

## 6. OCR 서브시스템 흐름

### 6.1 두 가지 OCR 경로

```
경로 1: ImageHandler의 OCR (이미지 파일 처리 시)
    .jpg/.png 파일
        → ImageHandler.process()
        → ContentExtractor: OCR 엔진이 있으면 OCR → 텍스트 반환
                            OCR 엔진 없으면 "[Image: filename]" 태그만 반환

경로 2: DocumentProcessor의 OCR 후처리 (문서 내 이미지 OCR)
    extract_text(path, ocr_processing=True)
        → 텍스트 추출 완료
        → OCRProcessor.process(text)
            → "[Image: path/to/image.png]" 태그 검색
            → 각 이미지 파일 → OCR 엔진에 전송
            → 태그를 OCR 결과로 교체
```

### 6.2 OCR 엔진 추상화

```
BaseOCREngine (추상)
    ├── OpenAIVisionEngine   (GPT-4V)
    ├── AnthropicEngine      (Claude Vision)
    ├── GeminiEngine         (Gemini Pro Vision)
    ├── BedrockEngine        (AWS Bedrock Claude)
    └── VLLMEngine           (로컬 배포)
```

---

## 7. 서비스 레이어 역할 분석

### 7.1 서비스 의존성 그래프

```
DocumentProcessor._create_services()
    │
    ├─ TagService (standalone)
    │   역할: 태그 문자열 생성 표준화
    │   제공: make_page_tag(n), make_slide_tag(n), make_sheet_tag(name), make_image_tag(path)
    │   의존: ProcessingConfig.tags (TagConfig)
    │
    ├─ StorageBackend (LocalStorageBackend)
    │   역할: 이미지 파일 저장 추상화
    │   현재 구현: 로컬 파일시스템
    │   설계: S3, MinIO, Azure Blob 등으로 교체 가능
    │
    ├─ ImageService ──▶ TagService, StorageBackend
    │   역할: 이미지 저장 + 태그 생성 + 중복 제거
    │   주요 메서드:
    │       save_and_tag(image_bytes, custom_name) → "[Image: path]"
    │       clear_state()  ← per-file 처리 전 상태 초기화
    │   중복제거: NamingStrategy.HASH → 내용 해시 기반
    │
    ├─ ChartService ──▶ TagService
    │   역할: ChartData → 텍스트 표현 변환
    │   출력 예: "[chart]\nChart Type: Bar\nTitle: Sales\n..."
    │
    ├─ TableService
    │   역할: TableData → HTML/Markdown/텍스트 변환
    │   설정: TableConfig.output_format (HTML 기본)
    │
    └─ MetadataService
        역할: DocumentMetadata → 포맷화된 텍스트 블록 변환
        출력 예: "[Document-Metadata]\n제목: ...\n저자: ...\n[/Document-Metadata]"
```

### 7.2 서비스의 핵심 기능: 포맷 중립적 태그 생성

`TagService`를 통한 태그 표준화가 핵심 비즈니스 로직이다:

```python
# 모든 핸들러가 동일한 태그 포맷 사용 (TagConfig로 커스터마이즈 가능)
tag_service.make_page_tag(1)    → "[Page Number: 1]"
tag_service.make_slide_tag(1)   → "[Slide Number: 1]"
tag_service.make_sheet_tag("Sheet1") → "[Sheet: Sheet1]"
image_service.save_and_tag(...)  → "[Image: temp/images/abc123.png]"
chart_service.format_chart(...)  → "[chart]\n...\n[/chart]"
metadata_service.format_metadata(...) → "[Document-Metadata]\n...\n[/Document-Metadata]"
```

이 태그들이 텍스트에 인라인 삽입되어 AI 모델이 문서 구조를 인식할 수 있게 함.

---

## 8. 위임(Delegation) 패턴 전체 맵

```
파일 요청 → 등록된 확장자 핸들러 → 마법 바이트 확인 → 실제 처리 핸들러

.doc  요청 → DOCHandler._check_delegation()
    ├─ ZIP 마법 바이트 (50 4B 03 04) → DocxHandler 위임
    ├─ RTF 마법 바이트 (7B 5C 72 74) → RTFHandler 위임
    ├─ HTML 마법 바이트              → ⚠ TODO (미구현, 주석만 존재)
    └─ OLE2 마법 바이트 (D0 CF 11 E0) → DOC 자체 파이프라인

.ppt  요청 → PPTHandler._check_delegation()
    ├─ ZIP 마법 바이트 → PPTXHandler 위임
    └─ OLE2 마법 바이트 → PPT 자체 파이프라인

.xls  요청 → XLSHandler._check_delegation()
    ├─ ZIP 마법 바이트 → XLSXHandler 위임 (확장자는 .xls지만 실제 XLSX)
    └─ BIFF 마법 바이트 → XLS 자체 파이프라인

.hwp  요청 → HWPHandler._check_delegation()
    ├─ ZIP 마법 바이트 → HWPXHandler 위임 (HWP 포맷이 ZIP 기반인 HWPX)
    ├─ HWP 3.0 마법 바이트 → ConversionError (지원 안함)
    └─ OLE2 마법 바이트 → HWP 자체 파이프라인
```

### 위임 제약사항

```python
# _delegate_to()는 registry 없이 호출 불가
def _delegate_to(self, extension, file_context, ...):
    if self._handler_registry is None:
        raise HandlerExecutionError("Cannot delegate — no registry available")
    delegate = self._handler_registry.get_handler(extension)
    return delegate.process(file_context, ...)
```

`set_registry()`는 `HandlerRegistry.register()` 시 자동 호출되므로, 핸들러를 레지스트리 없이 독립적으로 사용하면 위임이 불가.

---

## 9. 핵심 비즈니스 로직 위치 매핑

| 비즈니스 기능 | 담당 모듈 | 핵심 코드 |
|-------------|----------|----------|
| 포맷 감지 및 라우팅 | `HandlerRegistry` | `register_defaults()`, `get_handler()` |
| 마법 바이트 기반 포맷 재감지 | 각 Handler `_check_delegation()` | DOCHandler, PPTHandler, HWPHandler, XLSHandler |
| 파이프라인 실행 | `BaseHandler.process()` | 5단계 Template Method |
| PDF 레이아웃 분석 | `pdf_plus/_layout_block_detector.py` | heuristic 블록 감지 (29KB) |
| PDF 표 감지 | `pdf_plus/_table_detection.py` | heuristic 경계 탐지 (25KB) |
| DOCX 요소 순회 | `docx/content_extractor.py` | XML element 순회, 단락/표/이미지 처리 |
| PPTX 시각적 읽기 순서 | `pptx/content_extractor.py` | `sorted(shapes, key=lambda s: (s.top, s.left))` |
| XLSX 레이아웃 감지 | `xlsx/_layout.py` | `layout_detect_range()`, `object_detect()` |
| 한국어 RTF 파싱 | `rtf/_cleaner.py` | `\ansicpg949` CJK 인코딩 처리 |
| 청킹 전략 선택 | `TextChunker._select_strategy()` | 우선순위 기반 전략 매칭 |
| 표 보호 청킹 | `ProtectedChunkingStrategy` | 보호 영역 경계에서만 분할 |
| 이미지 저장 + 태그 | `ImageService.save_and_tag()` | 해시 기반 중복제거 + 로컬 저장 |
| 메타데이터 블록 생성 | `MetadataService.format_metadata()` | DocumentMetadata → 텍스트 블록 |
| AI 친화적 태그 생성 | `TagService` | 모든 구조 태그의 단일 출처 |

### 핵심 비즈니스 규칙 (문서화된 것)

1. **1확장자 = 1핸들러 원칙**: 카테고리 핸들러(Text, Image) 예외
2. **파이프라인 불변성**: `process()` 오버라이드 금지 (주석으로만 강제됨)
3. **위임은 명시적으로**: `_check_delegation()` → `_delegate_to()` 경로만 허용
4. **서비스를 통한 출력 표준화**: 태그 직접 생성 금지, TagService/ImageService 경유
5. **설정은 불변**: `ProcessingConfig` frozen dataclass, 변경 시 새 인스턴스 생성

---

## 10. 발견된 데이터 흐름 문제점

### 10.1 데이터 흐름 중단점 (Critical Path Issues)

**FileContext 생성 시 OOM 위험**:
```
파일 경로 → read_bytes() → file_data (5GB)
                        └─ BytesIO(file_data) → file_stream (5GB 추가)
                        총 10GB RAM 즉시 소비
```

**XLSX 이미지 데이터 손실 경로**:
```
이미지 1 → save_and_tag(bytes, custom_name="excel_sheet_img")
           → 저장: temp/images/excel_sheet_img.png
이미지 2 → save_and_tag(bytes, custom_name="excel_sheet_img")
           → 덮어쓰기! temp/images/excel_sheet_img.png (이미지 1 손실)
```

**DOCX/PPTX/XLSX 차트 매핑 불안정**:
```
Preprocessor 수집 순서: [차트A, 차트B, 차트C]  ← XML 관계 파일 기준 순서
ContentExtractor 매핑: chart_index=0→차트A, 1→차트B, 2→차트C  ← 본문 순서

만약 두 순서가 다르면:
    차트B가 실제로 본문에서 첫 번째로 나와도 → 차트A로 잘못 매핑됨
```

### 10.2 데이터 흐름 불일치

**html 확장자 처리 경로 불일치**:
```
types.py get_category("html") → FileCategory.CODE
TextHandler._TEXT_EXTENSIONS → "html" 포함 (plain text 처리)

사용자가 "report.html"을 처리하면:
    → HandlerRegistry → TextHandler 사용 (registered)
    → HTML을 텍스트로 읽음 (BS4 파싱 없음)
    → file_category = "code" (but processed as plain text)
```

**표 데이터의 이중 표현**:
```
ExtractionResult.text      → HTML 표가 인라인으로 삽입된 텍스트
ExtractionResult.tables    → List[TableData] (구조화된 표 데이터)

두 표현이 동시에 존재하나, 청킹은 text만 사용
→ tables 필드는 현재 사용자가 직접 process()를 호출해야만 접근 가능
```

### 10.3 서비스 상태 관리 문제

```
DocumentProcessor가 하나의 ImageService 공유:

스레드 1: extract_text("doc1.pdf")
    → image_service.clear_state()    ← 상태 초기화
    → PDFHandler가 이미지 추출 중...

스레드 2: extract_text("doc2.docx")  ← 동시 실행
    → image_service.clear_state()    ← ⚠ 스레드 1의 상태 날려버림!
    → doc1.pdf 처리에서 이미지 중복이 발생함
```

---

## 11. 고도화 범위 정의

### 11.1 범위 구분 원칙

고도화 범위를 세 영역으로 구분:
1. **버그 수정** (Bug Fix): 현재 잘못된 동작 교정
2. **기능 완성** (Feature Completion): 설계된 기능이지만 미구현
3. **아키텍처 고도화** (Architecture Enhancement): 새로운 능력 추가

---

### 11.2 버그 수정 범위 (즉시)

| ID | 파일 | 문제 | 수정 방향 |
|----|------|------|----------|
| BF-1 | `xlsx/content_extractor.py:_extract_sheet_images()` | `custom_name="excel_sheet_img"` 고정 | `f"excel_{ws.title}_{content_hash[:8]}"` |
| BF-2 | `xlsx/content_extractor.py` | `ws._images`, `img._data()`, `ws._charts` 내부 API | ZIP 직접 파싱 또는 공개 API 사용 |
| BF-3 | `docx/content_extractor.py:extract_text()` | `import re` 함수 내부 | 모듈 레벨로 이동 + 컴파일된 패턴 재사용 |
| BF-4 | `pdf/handler.py:create_content_extractor()` | mode 유효성 검증 없음 | `if mode not in {"plus", "default"}: raise ValueError` |
| BF-5 | `handlers/registry.py:register_defaults()` | 등록 실패 `logger.info` | `logger.warning`으로 변경 |

---

### 11.3 기능 완성 범위 (단기)

| ID | 기능 | 현재 상태 | 완성 방향 |
|----|------|----------|----------|
| FC-1 | HTML 핸들러 | TextHandler가 plain text 처리 | `handlers/html/` 신규 구현, BS4 활용 |
| FC-2 | DOCX 헤더/푸터 | 미구현 | `section.header/footer` 파싱 추가 |
| FC-3 | DOCX 각주/미주 | 미구현 | `doc.part.footnotes_part` 파싱 추가 |
| FC-4 | DOCX 이미지 중복제거 | rel_id 기반 → 동일 이미지 중복 저장 | content hash 기반으로 변경 |
| FC-5 | PDF 암호화 지원 | 미구현 | `fitz.Document.needs_pass` + `authenticate()` |
| FC-6 | PDF 스캔 감지 | 미구현 | 텍스트 레이어 샘플링 → OCR 플래그 설정 |
| FC-7 | XLSX 숨김 시트 | 미구현 (항상 포함) | config 옵션으로 제어 |
| FC-8 | XLSX 수식 평가 | 수식 문자열 저장 | `openpyxl.load_workbook(data_only=True)` 옵션 |

---

### 11.4 아키텍처 고도화 범위 (중장기)

#### A. 메모리 효율화 (성능)

현재:
```python
file_data = Path(file_path).read_bytes()      # 전체 로드
file_stream = io.BytesIO(file_data)           # 복사
```

목표:
```python
# 대용량 파일은 지연 로드 (스트리밍 FileContext)
@dataclass
class LazyFileContext:
    file_path: str
    file_size: int
    _data: Optional[bytes] = None

    def read_bytes(self) -> bytes:
        if self._data is None:
            self._data = Path(self.file_path).read_bytes()
        return self._data

    def get_stream(self) -> io.BufferedReader:
        return open(self.file_path, "rb")  # 매번 새 스트림
```

#### B. Thread Safety (안정성)

현재: 하나의 서비스 인스턴스 공유
목표: per-call 격리 또는 ThreadLocal 서비스

```python
# 방안 1: per-call 격리
def extract_text(self, file_path, ...):
    call_image_service = self._services["image_service"].clone()  # per-call 복사본
    handler = self._registry.get_handler(ext)
    ...

# 방안 2: Thread-local 서비스 상태
class ImageService:
    def __init__(self):
        self._local = threading.local()

    def clear_state(self):
        self._local.processed = set()  # 스레드별 독립 상태
```

#### C. Async 지원 (확장성)

```python
class AsyncDocumentProcessor(DocumentProcessor):
    async def extract_text_async(self, file_path, **kwargs) -> str:
        return await asyncio.to_thread(self.extract_text, file_path, **kwargs)

    async def extract_batch_async(
        self,
        file_paths: List[str],
        *,
        max_concurrent: int = 4,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        async def one(path):
            async with semaphore:
                return await self.extract_text_async(path)
        return await asyncio.gather(*[one(p) for p in file_paths])
```

#### D. Timeout 지원 (안정성)

```python
def process(self, file_context, *, timeout: Optional[float] = None, ...):
    if timeout is None:
        return self._run_pipeline(file_context, ...)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(self._run_pipeline, file_context, ...)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise HandlerExecutionError(
                f"Processing timed out after {timeout}s",
                context={"file": file_context.get("file_name")},
            )
```

#### E. 플러그인 시스템 (확장성)

```python
# pyproject.toml에 등록
# [project.entry-points."contextifier.handlers"]
# my_format = "my_package.handlers:MyFormatHandler"

def register_plugins(self) -> None:
    import importlib.metadata
    eps = importlib.metadata.entry_points(group="contextifier.handlers")
    for ep in eps:
        try:
            handler_class = ep.load()
            self.register(handler_class)
            logger.info(f"Plugin handler registered: {ep.name}")
        except Exception as e:
            logger.warning(f"Plugin {ep.name} failed to load: {e}")
```

#### F. 캐싱 레이어 (성능)

```python
class CachedDocumentProcessor(DocumentProcessor):
    def __init__(self, *args, cache_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = cache_backend or {}

    def extract_text(self, file_path, **kwargs) -> str:
        file_hash = self._hash_file(file_path)
        config_hash = hash(frozenset(kwargs.items()))
        cache_key = f"{file_hash}:{config_hash}"

        if cache_key in self._cache:
            logger.debug(f"Cache hit: {file_path}")
            return self._cache[cache_key]

        result = super().extract_text(file_path, **kwargs)
        self._cache[cache_key] = result
        return result
```

#### G. LibreOffice 변환 레이어 (레거시 포맷 지원 강화)

```
현재 레거시 포맷 처리 방식:
    .ppt (OLE2) → 자체 바이너리 파서 (표/차트/이미지 미지원)
    .doc (OLE2) → 자체 파서 (복잡한 DOC 실패 가능)
    .hwp (OLE2) → pyhwp (유지보수 불안정)

목표: LibreOffice 변환 공통 레이어
    .ppt → LibreOffice → .pptx → PPTXHandler
    .doc → LibreOffice → .docx → DOCXHandler
    .hwp → LibreOffice → .docx → DOCXHandler
```

---

### 11.5 고도화 최종 우선순위 요약

```
즉시 (0~1주):
    BF-1 XLSX 이미지 이름 버그
    BF-2 XLSX 내부 API 교체
    BF-3 DOCX import re 위치
    BF-4 PDF mode 검증
    BF-5 registry 로그 레벨

단기 (1~4주):
    FC-1 HTML 핸들러 구현 (가장 임팩트 큰 기능 완성)
    FC-2 DOCX 헤더/푸터
    FC-4 DOCX 이미지 중복제거 content hash
    FC-5 PDF 암호화 지원
    FC-7 XLSX 수식 평가

중기 (1~2개월):
    A. 메모리 효율화 (LazyFileContext)
    B. Thread Safety (per-call 격리)
    D. Timeout 지원
    F. 캐싱 레이어

장기 (2개월+):
    C. Async 지원
    E. 플러그인 시스템
    G. LibreOffice 변환 레이어
    테스트 인프라 구축
```

---

*본 보고서는 실제 소스 코드 (`types.py`, `errors.py`, `pipeline/content_extractor.py`, `pipeline/postprocessor.py`, `chunking/chunker.py`, `document_processor.py`, 각 핸들러 파일들)를 직접 확인하여 작성되었습니다.*
*작성: 손성준 (Developer Agent) | 2026-03-25*
