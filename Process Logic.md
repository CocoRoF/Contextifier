# Contextifier v2 — 처리 흐름 (Process Logic)

이 문서는 Contextifier v2의 전체 문서 처리 흐름을 다이어그램으로 설명합니다.

---

## 1. 전체 흐름

```
사용자 코드
    │
    ▼
DocumentProcessor
    │
    ├─ extract_text(file_path) ──────────────────┐
    │                                             │
    ├─ process(file_path) ───────────────────────┤
    │                                             │
    ├─ extract_chunks(file_path) ────────────────┤
    │       │                                     │
    │       ├── extract_text() 호출               │
    │       └── chunk_text() 호출                 │
    │                                             │
    └─ chunk_text(text) ─── TextChunker           │
                                                  ▼
                                    HandlerRegistry
                                         │
                                         ▼
                                  확장자 → Handler 매핑
                                         │
                                         ▼
                               BaseHandler.process()
                                         │
                     ┌───────────────────┼───────────────────┐
                     ▼                   ▼                   ▼
            _check_delegation()    5-Stage Pipeline     ExtractionResult
            (다른 핸들러로 위임)         │                   반환
                                        ▼
                    ┌─────────────────────────────────────┐
                    │  Stage 1: Converter.convert()       │
                    │  Stage 2: Preprocessor.preprocess()  │
                    │  Stage 3: MetadataExtractor.extract() │
                    │  Stage 4: ContentExtractor.extract_all() │
                    │  Stage 5: Postprocessor.postprocess()│
                    └─────────────────────────────────────┘
                                        │
                                        ▼
                              OCRProcessor (선택)
                              이미지 태그 → 텍스트 변환
```

---

## 2. 5-Stage 파이프라인 상세

모든 핸들러는 `BaseHandler.process()`가 강제하는 동일한 5단계를 실행합니다.
핸들러는 **단계별 컴포넌트**만 구현하고, 실행 순서는 오버라이드할 수 없습니다.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Stage 1     │     │  Stage 2     │     │  Stage 3     │     │  Stage 4     │     │  Stage 5     │
│  CONVERT     │────▶│  PREPROCESS  │────▶│  METADATA    │────▶│  CONTENT     │────▶│  POSTPROCESS │
│              │     │              │     │              │     │              │     │              │
│  Binary →    │     │  정규화,     │     │  제목/작성자/ │     │  텍스트/표/  │     │  메타데이터  │
│  Format Obj  │     │  인코딩 변환 │     │  날짜/페이지 │     │  이미지/차트 │     │  태그 삽입   │
│              │     │  전처리      │     │  수 추출     │     │  추출        │     │  최종 조립   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Stage 1: Convert (변환)

바이너리 파일 데이터를 해당 포맷의 객체로 변환합니다.

| 핸들러 | 변환 내용 |
|--------|-----------|
| PDF | `file_data` → `pdfplumber.PDF` 객체 |
| DOCX | `file_data` → `docx.Document` 객체 |
| DOC | `file_data` → LibreOffice 변환 → DOCX/HTML |
| PPTX | `file_stream` → `pptx.Presentation` 객체 |
| PPT | `file_data` → LibreOffice 변환 → PPTX |
| XLSX | `file_data` → `openpyxl.Workbook` 객체 |
| XLS | `file_data` → `xlrd.Book` 객체 |
| CSV/TSV | `file_data` → 자동 인코딩 감지 → 문자열 |
| HWP | `file_data` → OLE 컴파운드 파일 파싱 |
| HWPX | `file_data` → ZIP 압축 해제 → XML DOM |
| RTF | `file_data` → LibreOffice 변환 → HTML |
| Text | `file_data` → 자동 인코딩 감지 → 문자열 |
| Image | `file_data` → 임시 파일 저장 (OCR용) |

### Stage 2: Preprocess (전처리)

변환된 객체에 대해 정규화 및 전처리를 수행합니다.

- 인코딩 통일
- 불필요한 메타데이터 스트림 제거
- 페이지/슬라이드 순서 검증
- 임시 파일 준비

### Stage 3: Metadata Extract (메타데이터 추출)

`DocumentMetadata` 구조체를 생성합니다.

```python
@dataclass
class DocumentMetadata:
    title: str | None
    subject: str | None
    author: str | None
    keywords: str | None
    comments: str | None
    last_saved_by: str | None
    create_time: datetime | None
    last_saved_time: datetime | None
    page_count: int | None
    word_count: int | None
    category: str | None
    revision: str | None
    custom: dict
```

### Stage 4: Content Extract (콘텐츠 추출)

텍스트, 테이블, 이미지, 차트를 추출하여 `ExtractionResult`에 저장합니다.

```python
@dataclass
class ExtractionResult:
    text: str                          # 전체 추출 텍스트
    metadata: DocumentMetadata | None  # Stage 3 결과
    tables: list[TableData]            # 추출된 테이블들
    images: list[str]                  # 저장된 이미지 경로들
    charts: list[ChartData]            # 추출된 차트들
```

이 단계에서 공유 서비스를 활용합니다:
- **TagService**: 페이지/슬라이드/시트 태그 생성
- **ImageService**: 이미지 저장 + 태그 생성 + 중복 제거
- **ChartService**: 차트 데이터 → HTML 테이블 포맷팅 + 태그 래핑
- **TableService**: `TableData` → HTML/Markdown/Text 렌더링

### Stage 5: Postprocess (후처리)

최종 텍스트를 조립합니다:

1. `MetadataService`로 메타데이터 포맷팅 → 태그로 래핑
2. 페이지별 텍스트 + 테이블 + 이미지 태그 + 차트 조합
3. 연속 빈 줄 정리
4. 최종 텍스트 문자열 반환

---

## 3. Delegation (위임) 흐름

일부 파일은 확장자와 실제 내부 형식이 다를 수 있습니다.
이 경우 핸들러가 자동으로 올바른 핸들러에게 **위임**합니다.

```
DOC Handler
    │
    ├─ _check_delegation(file_context)
    │     │
    │     ├─ OLE 서명 감지 → DOC Handler 계속 진행
    │     ├─ HTML 마커 감지 → HTML Reprocessor로 위임
    │     ├─ DOCX 서명 감지 → DOCX Handler로 위임
    │     └─ RTF 서명 감지 → RTF Handler로 위임
    │
    ▼
정상적인 DOC 처리 또는 위임된 핸들러의 결과 반환
```

위임이 발생하면 `BaseHandler._delegate_to(extension, file_context)` 메서드로
`HandlerRegistry`에서 적절한 핸들러를 조회하여 처리를 넘기며,
원래 핸들러의 파이프라인은 실행되지 않습니다.

---

## 4. 청킹 흐름

```
extract_chunks() 또는 chunk_text()
    │
    ▼
TextChunker
    │
    ├─ 등록된 전략 목록 (우선순위 순 정렬)
    │     │
    │     ├─ TableChunkingStrategy    (우선순위 5)
    │     ├─ PageChunkingStrategy     (우선순위 10)
    │     ├─ ProtectedChunkingStrategy (우선순위 20)
    │     └─ PlainChunkingStrategy    (우선순위 100)
    │
    ├─ 각 전략에 can_handle(text, extension) 질의
    │     │
    │     ├─ 스프레드시트(xlsx, xls, csv, tsv) → TableChunkingStrategy ✓
    │     ├─ 페이지 태그 존재 → PageChunkingStrategy ✓
    │     ├─ HTML 테이블 존재 → ProtectedChunkingStrategy ✓
    │     └─ 기본 → PlainChunkingStrategy ✓ (항상 True)
    │
    ▼
선택된 전략의 chunk(text, chunk_size, chunk_overlap) 실행
    │
    ▼
List[str] 또는 List[Chunk] 반환
```

### 전략별 로직

#### TableChunkingStrategy (스프레드시트)
1. `[Sheet: ...]` 태그로 시트 경계 분할
2. 시트 내 테이블을 개별 청크로 분리
3. 큰 테이블은 행 단위로 분할

#### PageChunkingStrategy (페이지 기반)
1. `[Page Number: ...]` 태그로 페이지 경계 분할
2. 한 페이지가 `chunk_size`를 초과하면 재귀 분할
3. 페이지 내 테이블은 보존

#### ProtectedChunkingStrategy (Protected Region)
1. HTML 테이블, 메타데이터 블록 등 **보호 영역** 식별
2. 보호 영역을 플레이스홀더로 치환
3. 나머지 텍스트를 재귀 분할
4. 플레이스홀더를 원래 콘텐츠로 복원

#### PlainChunkingStrategy (기본 폴백)
1. 재귀적 문자 분할 (LangChain `RecursiveCharacterTextSplitter` 방식)
2. 분할 기준: `\n\n` → `\n` → `. ` → ` ` → 문자
3. `chunk_overlap` 만큼 겹침 유지

---

## 5. OCR 처리 흐름

```
extract_text(..., ocr_processing=True)
    │
    ├─ 핸들러 파이프라인 실행 → 텍스트 얻기
    │     (이미지 위치에 [Image: path/to/img.png] 태그 삽입됨)
    │
    ▼
OCRProcessor.process(text)
    │
    ├─ 정규식으로 [Image: ...] 태그 검출
    │
    ├─ 각 태그에 대해:
    │     │
    │     ├─ 이미지 파일 경로 추출
    │     ├─ BaseOCREngine.convert_image_to_text(path)
    │     │     │
    │     │     ├─ 이미지 → Base64 인코딩
    │     │     ├─ build_message_content(b64, mime, prompt)
    │     │     │     (엔진별 LLM 메시지 페이로드 구성)
    │     │     ├─ LangChain HumanMessage 생성
    │     │     ├─ LLM 호출 (Vision API)
    │     │     └─ [Figure: 결과텍스트] 반환
    │     │
    │     └─ [Image: ...] 태그를 [Figure: ...] 텍스트로 교체
    │
    ▼
OCR 처리된 최종 텍스트 반환
```

### 지원 OCR 엔진

| 엔진 | 프로바이더 | 기본 모델 |
|------|-----------|-----------|
| `OpenAIOCREngine` | OpenAI | gpt-4o |
| `AnthropicOCREngine` | Anthropic | claude-sonnet-4-20250514 |
| `GeminiOCREngine` | Google | gemini-2.0-flash |
| `BedrockOCREngine` | AWS Bedrock | anthropic.claude-3-5-sonnet |
| `VLLMOCREngine` | 자체호스팅 | (사용자 지정) |

---

## 6. 서비스 의존성 그래프

```
DocumentProcessor
    │
    ├─ TagService (독립 — 의존성 없음)
    │     ├─ 페이지/슬라이드/시트 태그 생성
    │     └─ TagConfig로 prefix/suffix 커스터마이징
    │
    ├─ ImageService (TagService + StorageBackend 의존)
    │     ├─ 이미지 저장 (Local/MinIO/S3/Azure/GCS)
    │     ├─ 이미지 태그 생성 (TagService 위임)
    │     └─ 해시 기반 중복 제거
    │
    ├─ ChartService (TagService 의존)
    │     ├─ ChartData → HTML 테이블 변환
    │     └─ [chart]...[/chart] 태그 래핑
    │
    ├─ TableService (독립)
    │     └─ TableData → HTML/Markdown/Text 렌더링
    │
    └─ MetadataService (독립)
          ├─ DocumentMetadata → 포맷팅된 텍스트
          └─ 한국어/영어 라벨 지원
```

서비스는 `DocumentProcessor`가 생성 시 한 번 만들어, 모든 핸들러가 공유합니다.

---

## 7. 포맷별 핸들러 요약

| 핸들러 | 패키지 | 확장자 | 변환 방식 | 특이사항 |
|--------|--------|--------|-----------|----------|
| **PDFHandler** | `handlers/pdf/` | `.pdf` | pdfplumber | 기본 PDF 처리 |
| **PDFPlusHandler** | `handlers/pdf_plus/` | `.pdf` | pdfplumber + 고급 분석 | 테이블 감지, 텍스트 품질 분석, 복잡 레이아웃 |
| **DOCXHandler** | `handlers/docx/` | `.docx` | python-docx | 표/차트/이미지 직접 추출 |
| **DOCHandler** | `handlers/doc/` | `.doc` | LibreOffice 변환 | OLE/HTML/DOCX/RTF 자동 감지 + 위임 |
| **PPTXHandler** | `handlers/pptx/` | `.pptx` | python-pptx | 슬라이드/노트/차트 추출 |
| **PPTHandler** | `handlers/ppt/` | `.ppt` | LibreOffice 변환 → PPTX | PPTX Handler에 위임 |
| **XLSXHandler** | `handlers/xlsx/` | `.xlsx` | openpyxl | 다중 시트, 차트, 수식 |
| **XLSHandler** | `handlers/xls/` | `.xls` | xlrd | 다중 시트, 차트 |
| **CSVHandler** | `handlers/csv/` | `.csv`, `.tsv` | 자체 파싱 | 자동 인코딩/구분자 감지 |
| **HWPHandler** | `handlers/hwp/` | `.hwp` | OLE 파싱 | HWP 5.0 binary, 3.0 미지원 |
| **HWPXHandler** | `handlers/hwpx/` | `.hwpx` | ZIP + XML | OWPML 포맷 |
| **RTFHandler** | `handlers/rtf/` | `.rtf` | LibreOffice → HTML | HTML Reprocessor 활용 |
| **TextHandler** | `handlers/text/` | `.txt`, `.md`, `.py`, `.json`, ... | 자동 인코딩 감지 | 80+ 확장자 카테고리 핸들러 |
| **ImageHandler** | `handlers/image/` | `.jpg`, `.png`, `.gif`, ... | 임시 파일 저장 | OCR 엔진 필요 |

---

## 8. 설정 흐름

```
ProcessingConfig (불변, frozen dataclass)
    │
    ├─ DocumentProcessor.__init__()에서 수신
    │
    ├─ services 생성 시 전달
    │     ├─ TagService(config)
    │     ├─ ImageService(config, storage, tag_service)
    │     ├─ ChartService(config, tag_service)
    │     ├─ TableService(config)
    │     └─ MetadataService(config)
    │
    ├─ HandlerRegistry(config, services)
    │     └─ 각 핸들러 생성 시 config + services 전달
    │
    └─ TextChunker(config)
          └─ ChunkingConfig 참조
```

설정은 **한 번 생성되면 변경 불가**합니다.
다른 설정이 필요하면 새 `ProcessingConfig`를 만들어 새 `DocumentProcessor`를 생성하세요.

```python
config1 = ProcessingConfig(chunking=ChunkingConfig(chunk_size=1000))
config2 = config1.with_chunking(chunk_size=2000)  # 새 인스턴스

proc1 = DocumentProcessor(config=config1)  # chunk_size=1000
proc2 = DocumentProcessor(config=config2)  # chunk_size=2000
```
