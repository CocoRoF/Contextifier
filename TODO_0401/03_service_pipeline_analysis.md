# Contextifier v0.2.5 — 서비스 레이어 & 파이프라인 분석

> 분석 일자: 2025-04-01
> 분석 대상: services/ (7파일, ~1,050 LOC), pipeline/ (5파일, ~690 LOC), chunking/ (8파일, ~2,100 LOC), ocr/ (7파일, ~750 LOC)

---

## 1. 서비스 레이어 분석

### 1.1 아키텍처 개요

서비스 레이어는 핸들러별 추출 로직과 공통 포매팅/저장 로직을 분리하는 중간 계층이다.

```
Handler.ContentExtractor
    ├── ImageService      — 이미지 저장/중복제거/태그 생성
    │   └── StorageBackend — 실제 파일 I/O
    ├── TagService        — 구조 태그 생성/파싱
    ├── ChartService      — 차트 데이터 → 텍스트 블록
    ├── TableService      — TableData → HTML/Markdown/Text
    └── MetadataService   — DocumentMetadata → 태그 블록
```

### 1.2 ImageService (244 LOC)

**역할**: 이미지 데이터 수신 → SHA256 중복 검출 → 파일 저장 → 태그 생성

**핵심 메서드**:
| 메서드 | 역할 | 호출 빈도 |
|--------|------|----------|
| `save(data, filename)` | 파일 저장 + 경로 반환 | 높음 |
| `save_and_tag(data, filename)` | 저장 + 태그 한번에 | 가장 높음 |
| `extract_and_deduplicate(data, ...)` | 해시 기반 중복 제거 + 저장 + 태그 | 높음 |
| `clear_state()` | 파일별 중복 상태 초기화 | 파일당 1회 |
| `get_processed_count()` | 처리된 이미지 수 | 디버그 |

**Thread Safety 구현**:
```python
class ImageService:
    def __init__(self, ...):
        self._local = threading.local()  # Thread-local 상태

    def _ensure_state(self):
        if not hasattr(self._local, 'seen_hashes'):
            self._local.seen_hashes = set()
            self._local.saved_paths = []
            self._local.counter = 0
```

**네이밍 전략**: HASH (SHA256 16자), UUID, SEQUENTIAL, TIMESTAMP

**🟡 발견된 이슈**:

| # | 이슈 | 심각도 | 상세 |
|---|------|--------|------|
| IS1 | 이미지 크기 검증 없음 | 🟠 | 수백 MB 이미지도 무조건 저장 시도 |
| IS2 | SHA256 16자 잘림 | 🟡 | 이론적 해시 충돌 가능 (극히 낮은 확률) |
| IS3 | Thread-local 속성 동적 생성 | 🟡 | `hasattr` 패턴이 fragile |
| IS4 | 비동기 지원 없음 | 🟡 | AsyncDocumentProcessor에서 스레드 풀 필요 |
| IS5 | StorageBackend 에러 전파 | 🟡 | 저장 실패 시 전체 파이프라인 중단 |

**개선 제안**:
- `max_image_size` 설정 추가 (ImageConfig)
- `_ensure_state()` 대신 `__init__`에서 초기화 → `clear_state()`로 리셋
- 저장 실패를 경고로 처리 (이미지 태그를 에러 태그로 대체)

---

### 1.3 TagService (167 LOC)

**역할**: 구조 태그 생성/파싱 — 시스템 전체 태그 포맷의 단일 진실의 원천(Single Source of Truth)

**태그 종류**:
| 태그 | 예시 | 사용처 |
|------|------|--------|
| 페이지 | `[Page Number: 1]` | PDF, DOCX |
| 슬라이드 | `[Slide Number: 1]` | PPTX, PPT |
| 시트 | `[Sheet: Sheet1]` | XLSX, XLS |
| 이미지 | `[Image: path.png]` | 모든 핸들러 |
| 차트 | `[chart]...[/chart]` | 차트 지원 핸들러 |
| 메타데이터 | `[Document-Metadata]...[/Document-Metadata]` | 전체 |

**구현 강점**:
- `__init__`에서 regex 사전 컴파일 → 런타임 성능 보장
- `find_*_tags()` → `(start, end, value)` 튜플 반환 → 정밀 텍스트 조작
- Config-driven → `TagConfig` 변경 시 전체 시스템 자동 반영

**🟡 발견된 이슈**:

| # | 이슈 | 심각도 |
|---|------|--------|
| TS1 | 태그 타입 4종 하드코딩 | 🟡 — 확장 시 코드 변경 필요 |
| TS2 | 태그 검증 없음 | 🔵 — 잘못된 prefix/suffix 무시 |
| TS3 | deprecated alias 존재 | 🔵 — `remove_all_structural_markers` |

---

### 1.4 ChartService (206 LOC)

**역할**: ChartData → 태그로 감싼 텍스트 블록

**차트 타입 매핑** (OOXML → 표시명):
```
barChart → "Bar Chart"  |  lineChart → "Line Chart"
pieChart → "Pie Chart"  |  scatterChart → "Scatter Chart"
areaChart → "Area Chart" |  doughnutChart → "Doughnut Chart"
...15종 이상
```

**출력 예시**:
```
[chart]
Bar Chart: Revenue by Quarter
| Category | Series1 | Series2 |
|----------|---------|---------|
| Q1       | 100     | 200     |
| Q2       | 150     | 250     |
[/chart]
```

**🟡 발견된 이슈**:

| # | 이슈 | 심각도 |
|---|------|--------|
| CS1 | HTML 테이블 형식만 지원 (span 없음) | 🟡 |
| CS2 | 비표준 차트 타입 pass-through | 🟡 |
| CS3 | 차트→Markdown/ASCII 렌더링 없음 | 🔵 |

---

### 1.5 TableService (124 LOC)

**역할**: TableData → HTML/Markdown/Text 포매팅

**메서드 구조**:
| 메서드 | 출력 | 특징 |
|--------|------|------|
| `format_table(table)` | 설정 기반 자동 선택 | OutputFormat에 따라 분기 |
| `format_as_html(table)` | `<table>...</table>` | rowspan/colspan 속성 지원 |
| `format_as_markdown(table)` | `| ... | ... |` | 파이프 구분, 첫행 헤더 |
| `format_as_text(table)` | `\t` 구분 | 탭 구분 플레인 텍스트 |
| `format_as_html_simple(table)` (static) | `<table border='1'>` | **HTML 이스케이프 포함** — 폴백용 |

**🟠 발견된 이슈**:

| # | 이슈 | 심각도 | 상세 |
|---|------|--------|------|
| TBS1 | `format_as_html()`에서 셀 내용 HTML 이스케이프 없음 | 🟠 | `_clean_cell()`이 이스케이프하지 않음, XSS 가능성 |
| TBS2 | `format_as_html_simple()`은 이스케이프 있음 | - | static 폴백은 정상 |
| TBS3 | Markdown 출력에서 colspan/rowspan 정보 손실 | 🟡 | Markdown 테이블 문법 한계 |
| TBS4 | Text 출력에서 셀 정렬 없음 | 🔵 | 탭 기반이라 컬럼 정렬 불가 |

**🟠 TBS1 상세 (보안 이슈)**:
```python
def format_as_html(self, table: TableData) -> str:
    for cell in row_cells:
        content = self._clean_cell(cell.content)  # 이스케이프 없음!
        line_parts.append(f"<{tag}{attrs}>{content}</{tag}>")

# vs. format_as_html_simple (이스케이프 있음):
    content = html_mod.escape(content, quote=False)  # ✓ 안전
```
→ **사용자 입력이 포함된 문서(예: CSV의 셀, 사용자 편집 DOCX)에서 HTML 인젝션 가능**

---

### 1.6 MetadataService (124 LOC)

**역할**: DocumentMetadata → 태그로 감싼 텍스트 블록

**이중 언어 라벨**:
```python
_LABELS_KO = {
    "title": "제목", "author": "작성자", "subject": "주제",
    "keywords": "키워드", "create_time": "작성일시", ...
}
_LABELS_EN = {
    "title": "Title", "author": "Author", "subject": "Subject", ...
}
```

**평가**: ✅ 깔끔한 구현, 확장 용이

**🟡 미미한 이슈**:
- 필드 순서 하드코딩 (사용자 커스터마이즈 불가)
- 한국어/영어 외 추가 언어는 코드 변경 필요

---

### 1.7 LocalStorageBackend (~50 LOC)

**역할**: 로컬 파일 시스템에 이미지 저장

**인터페이스**: `save(filepath, data)`, `delete(filepath)`, `exists(filepath)`

**🟡 발견된 이슈**:

| # | 이슈 | 심각도 |
|---|------|--------|
| SB1 | 원자적 쓰기 없음 | 🟡 — 크래시 시 파일 손상 가능 |
| SB2 | 권한 검사 없음 | 🟡 — 권한 오류 시 무의미한 에러 |
| SB3 | 심볼릭 링크 미지원 | 🔵 |

---

## 2. 파이프라인 추상 계층 분석

### 2.1 5단계 파이프라인 ABC

```
┌────────────────┐   ┌────────────────┐   ┌███████████████████┐   ┌████████████████████┐   ┌████████████████┐
│  BaseConverter  │──▶│BasePreprocessor│──▶│BaseMetadataExtr.  │──▶│BaseContentExtractor│──▶│BasePostprocessor│
│  (100 LOC)      │   │  (82 LOC)      │   │   (~60 LOC)       │   │   (297 LOC)         │   │  (148 LOC)      │
│                 │   │                │   │                   │   │                     │   │                 │
│ convert()       │   │ preprocess()   │   │ extract()         │   │ extract_text()      │   │ postprocess()   │
│ validate()      │   │ validate()     │   │ get_format_name() │   │ extract_tables()    │   │ _normalize_text │
│ close()         │   │                │   │                   │   │ extract_images()    │   │                 │
│ _get_stream()   │   │                │   │                   │   │ extract_charts()    │   │                 │
│                 │   │                │   │                   │   │ extract_all()       │   │                 │
└────────────────┘   └────────────────┘   └███████████████████┘   └████████████████████┘   └████████████████┘
       │                     │                      │                        │                       │
  Null variant          Null variant           Null variant            Null variant          DefaultPostprocessor
```

### 2.2 BaseContentExtractor 상세 (297 LOC)

가장 중요한 파이프라인 컴포넌트 — 실제 데이터 추출의 핵심.

**공통 메서드**:
```python
class BaseContentExtractor(ABC):
    # 추상 (필수 구현)
    @abstractmethod
    def extract_text(preprocessed, **kwargs) -> str: ...

    # 선택 (기본값: 빈 리스트)
    def extract_tables(preprocessed, **kwargs) -> List[TableData]: return []
    def extract_images(preprocessed, **kwargs) -> List[str]: return []
    def extract_charts(preprocessed, **kwargs) -> List[ChartData]: return []

    # 편성자 (최종 — 재정의 금지)
    @final
    def extract_all(preprocessed, extract_metadata_result=None, **kwargs) -> ExtractionResult:
        text = self.extract_text(preprocessed, **kwargs)
        tables = self._safe_extract(self.extract_tables, preprocessed, "tables")
        images = self._safe_extract(self.extract_images, preprocessed, "images")
        charts = self._safe_extract(self.extract_charts, preprocessed, "charts")
        return ExtractionResult(text=text, ...)
```

**헬퍼 메서드**:
- `_safe_tag(tag_service, method_name, *args)` — 서비스 null 체크 래퍼
- `_format_chart_from_dict(chart_dict)` — dict → ChartData → 포매팅 통합

**서비스 접근자** (DI 기반):
```python
@property
def image_service(self) -> Optional[ImageService]: ...
@property
def tag_service(self) -> Optional[TagService]: ...
@property
def chart_service(self) -> Optional[ChartService]: ...
@property
def table_service(self) -> Optional[TableService]: ...
```

**🟡 이슈**:
- `extract_all()`에서 tables/images/charts 실패 시 경고 수집 → 파이프라인 계속
- 텍스트 추출 실패 시 전체 파이프라인 중단 (적절한 동작)
- 스트리밍 지원 없음 — 모든 데이터가 메모리에 로딩

### 2.3 DefaultPostprocessor (148 LOC)

**표준 후처리 파이프라인**:
1. 메타데이터 포매팅 + 텍스트 앞부분에 추가
2. 실제 텍스트 정상화 (3+ 연속 개행 → 2개로 축소)
3. 추출 경고를 HTML 주석으로 첨부

**🟡 이슈**:
- 경고가 HTML 주석 형식 (`<!-- Warning: ... -->`) — 혼합 포맷
- 메타데이터→텍스트 순서 고정 (역순 불가)
- 커스텀 후처리 단계 추가 메커니즘 없음

---

## 3. 청킹 시스템 분석

### 3.1 아키텍처

```
TextChunker (Facade)
    ├── Strategy Selection (priority ordering)
    │   ├── TableChunkingStrategy  (p=5)  — 테이블 포함 문서
    │   ├── PageChunkingStrategy   (p=10) — 페이지 마커 기반
    │   ├── ProtectedStrategy      (p=20) — 보호 영역 보존
    │   └── PlainChunkingStrategy  (p=100) — 기본 폴백
    │
    ├── TableChunker — HTML/Markdown 테이블 행 분할
    │   └── TableParser — 테이블 구조 파싱
    │
    └── Constants — 공유 패턴, 오버헤드 상수
```

### 3.2 전략별 분석

#### TableChunkingStrategy (304 LOC, p=5)
- **활성 조건**: 텍스트에 HTML 테이블 패턴 존재
- **동작**: 테이블 영역 분리 → 별도 청킹 → 재조립
- **특징**: 테이블-텍스트 혼합 구간 intelligent 분할

#### PageChunkingStrategy (264 LOC, p=10)
- **활성 조건**: 페이지/슬라이드/시트 마커 존재
- **동작**: 마커 기반 섹션 분할 → 섹션별 크기 검증 → 대형 섹션 재분할
- **특징**: 페이지 경계 보존 (LLM 참조 정확도 향상)

#### ProtectedChunkingStrategy (480 LOC, p=20) — **최대 규모**
- **활성 조건**: 보호 영역(테이블/태그) 존재
- **동작**: 보호 영역 식별 → 비보호 텍스트만 분할 → 재삽입
- **보호 대상**: HTML 테이블, 이미지 태그, 차트 블록, 페이지 마커
- **특징**: 구조화된 콘텐츠가 분할로 파괴되지 않도록 보호

#### PlainChunkingStrategy (182 LOC, p=100)
- **활성 조건**: 항상 (최종 폴백)
- **동작**: langchain RecursiveCharacterTextSplitter 활용
- **구분자 우선순위**: `\n\n` → `\n` → `. ` → ` ` → ``
- **특징**: 코드 파일용 Language 인식 분할, 오버랩 지원

### 3.3 테이블 청킹 (236 LOC)

```python
# HTML 테이블 청킹 흐름
chunk_html_table(table_html, max_chunk_size)
    ├── parse_html_table() → ParsedTable (headers, data_rows)
    ├── 헤더 크기 계산 (모든 청크에 복제)
    ├── 행 누적, max_chunk_size 초과 시 분할
    ├── rowspan 조정 (청크 경계에서 클램핑)
    └── [Table Chunk N/M] 주석 추가
```

### 3.4 발견된 이슈

| # | 이슈 | 심각도 | 상세 |
|---|------|--------|------|
| CH1 | 전략 선택 시 실패 이유 비로깅 | 🟡 | `can_handle()` 실패 추적 어려움 |
| CH2 | NotImplementedError를 "다음 시도"로 처리 | 🟡 | 취약한 폴백 메커니즘 |
| CH3 | regex 기반 HTML 파싱 | 🟡 | 복잡 테이블 (중첩, 속성) 실패 가능 |
| CH4 | colspan 처리 없음 (테이블 청커) | 🟡 | 복잡 colspan 테이블 오분할 |
| CH5 | 보호 전략 480 LOC | 🔵 | 리팩토링 가치 있음 |
| CH6 | 테이블 오버헤드 매직 넘버 | 🔵 | 30, 12, 10 — 주석 부족 |

---

## 4. OCR 시스템 분석

### 4.1 아키텍처

```
OCRProcessor (편성자)
    ├── 이미지 태그 탐색 (regex 패턴)
    ├── 이미지 파일 경로 해석
    ├── BaseOCREngine 호출
    │   ├── OpenAI Engine
    │   ├── Anthropic Engine
    │   ├── Google Gemini Engine
    │   ├── AWS Bedrock Engine
    │   └── vLLM Engine (로컬)
    └── 태그 치환 (이미지 태그 → OCR 텍스트)
```

### 4.2 BaseOCREngine (150 LOC)

**Template Method 패턴**:
```python
class BaseOCREngine(ABC):
    def convert_image_to_text(self, image_path: str) -> str:
        # 1. 이미지 읽기 + base64 인코딩  (공통)
        # 2. MIME 타입 결정               (공통)
        # 3. build_message_content()       (프로바이더별)
        # 4. LLM API 호출                 (프로바이더별)
        # 5. 결과 포매팅                   (공통)

    @abstractmethod
    def build_message_content(self, base64_data, mime_type) -> Any:
        """Provider-specific message format"""
```

**기본 프롬프트** (DEFAULT_OCR_PROMPT):
- 한국어 출력 요청 (하드코딩)
- 수학 공식/표 보존 지시
- 이미지 설명 포함

### 4.3 OCRProcessor (184 LOC)

**처리 흐름**:
```
입력 텍스트 → 이미지 태그 검색 → 각 태그에 대해:
    ├── 이미지 경로 추출
    ├── 파일 존재 확인
    ├── OCR 엔진 호출
    ├── [Figure: ...] 태그로 감싸기
    └── 원본 이미지 태그 치환
→ OCR 적용된 텍스트 반환
```

**진행 콜백**:
```python
class OCRProgressEvent:
    event_type: str    # "tag_processing", "tag_processed", "completed"
    tag_index: int
    total_tags: int
    image_path: Optional[str]
    result: Optional[str]
```

### 4.4 발견된 이슈

| # | 이슈 | 심각도 | 상세 |
|---|------|--------|------|
| OCR1 | 프롬프트 언어 하드코딩 (한국어) | 🟠 | 다국어 환경에서 부적합 |
| OCR2 | 순차 처리 (병렬화 없음) | 🟠 | 이미지 10장 = 10× API 호출 대기 |
| OCR3 | MIME 타입 맵 불완전 | 🟡 | .ico, .heic 등 미지원 |
| OCR4 | 태그 치환 regex 취약성 | 🟡 | 특수문자 포함 경로에서 실패 가능 |
| OCR5 | Tesseract 통합 없음 | 🟡 | pytesseract 의존성 있지만 엔진 구현 없음 |

---

## 5. 서비스 간 의존성 그래프

```
DocumentProcessor
    │
    ├── HandlerRegistry
    │       └── BaseHandler (17개)
    │               ├── Pipeline Components (각 핸들러별)
    │               │   ├── ImageService ←─── StorageBackend
    │               │   │       └── TagService
    │               │   ├── TagService (직접)
    │               │   ├── ChartService
    │               │   │       └── TagService
    │               │   ├── TableService
    │               │   └── MetadataService
    │               │           └── TagService
    │               └── _check_delegation() → HandlerRegistry (순환 but post-injection)
    │
    ├── TextChunker
    │       └── Strategies[] (TagService 패턴 참조)
    │
    └── OCRProcessor
            └── BaseOCREngine (외부 API)
```

**순환 의존성 분석**:
- `BaseHandler` ↔ `HandlerRegistry`: `set_registry()` post-injection으로 해결 ✅
- 서비스 간 순환: 없음 ✅
- 파이프라인 → 서비스: 단방향 ✅

---

## 6. 종합 평가

### 6.1 강점

| 영역 | 평가 |
|------|------|
| 서비스 분리 | ✅ 핸들러별 추출과 공통 포매팅이 명확히 분리 |
| DI 패턴 | ✅ 서비스 인스턴스가 Config를 통해 생성, 핸들러에 주입 |
| 태그 일관성 | ✅ TagService가 전체 태그 포맷의 SSOT |
| 청킹 전략 | ✅ 4단계 우선순위 폴백으로 다양한 문서 타입 대응 |
| OCR 확장성 | ✅ 5개 엔진 + 추가 가능한 ABC 인터페이스 |
| Postprocessor 통일 | ✅ 모든 핸들러가 동일한 후처리 파이프라인 사용 |

### 6.2 개선 필요 영역

| 영역 | 현재 상태 | 필요 조치 |
|------|----------|----------|
| **format_as_html() 이스케이프** | 🟠 미수행 | **즉시 수정** — `html.escape()` 추가 |
| **OCR 순차 처리** | 🟠 비효율 | asyncio 또는 ThreadPoolExecutor 병렬화 |
| **OCR 프롬프트 언어** | 🟠 하드코딩 | OCRConfig에 prompt_language 추가 |
| **이미지 크기 제한** | 🟡 없음 | ImageConfig에 max_file_size 추가 |
| **테이블 HTML 파싱** | 🟡 regex 기반 | BeautifulSoup 활용 고려 |
| **Tesseract 엔진** | 🟡 의존성만 존재 | 실제 엔진 구현 필요 |
| **메타데이터 필드 순서** | 🔵 고정 | MetadataConfig에 field_order 추가 (선택) |
| **chunk 전략 로깅** | 🔵 불충분 | 전략 선택/거부 이유 로깅 |
