# Contextifier 고도화 분석 보고서

> GitHub: https://github.com/CocoRoF/Contextifier
> 분석일: 2026-03-25
> 버전: v0.2.4
> 분석자: 손성준 (Developer Agent)

---

## 1. 프로젝트 개요

**Contextifier**는 PDF, DOCX, XLSX, HWP 등 80+ 확장자를 AI 친화적 텍스트로 변환하는 Python 문서 처리 라이브러리입니다.

### 핵심 구조

```
DocumentProcessor (Facade)
  ├── HandlerRegistry          # 확장자 → 핸들러 매핑
  ├── TextChunker              # 청킹 서브시스템
  ├── OCRProcessor             # OCR 서브시스템 (선택)
  └── Services (DI)
        ├── ImageService
        ├── TagService
        ├── ChartService
        ├── TableService
        └── MetadataService

Handlers (14개)
  각 핸들러마다 5-stage 파이프라인:
  [Convert] → [Preprocess] → [Metadata] → [ContentExtract] → [Postprocess]
```

### 강점 요약

- 명확한 5단계 파이프라인 (Template Method 패턴으로 강제)
- 서비스 DI(의존성 주입)로 테스트/확장 용이
- 한 확장자 = 한 핸들러 원칙 (명확한 책임 분리)
- Delegation 메커니즘 (`.doc` → RTF 위임 등)
- Frozen dataclass 기반 불변 설정 시스템

---

## 2. 핸들러 심층 분석 및 개선 방안

### 2.1 BaseHandler / 공통 파이프라인

#### 현재 문제점

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | 파일 전체를 메모리에 로드 (`read_bytes()`) | `document_processor.py:_create_file_context()` | 🔴 HIGH |
| 2 | Timeout 메커니즘 없음 (대용량 파일 시 무한 대기 가능) | `base.py:process()` | 🔴 HIGH |
| 3 | Thread safety 미보장 (서비스 인스턴스 공유) | 전체 서비스 레이어 | 🟡 MEDIUM |
| 4 | Async 지원 없음 (모두 동기) | 전체 | 🟡 MEDIUM |
| 5 | Progress 콜백 없음 (대용량 파일 처리 상태 불투명) | `process()` | 🟡 MEDIUM |
| 6 | `process()` 파이프라인이 `@final`로 보호되지 않음 | `base.py` | 🟢 LOW |

#### 개선 방안

**[HIGH] 스트리밍/메모리 효율화**
```python
# 현재: 파일 전체를 메모리에 로드
file_data = Path(file_path).read_bytes()  # 5GB PDF → OOM

# 개선안: 스트리밍 FileContext 지원
@dataclass
class FileContext:
    file_path: str
    file_stream: IO[bytes]       # 지연 로드
    file_size: int               # stat으로만 확인
    _data_cache: Optional[bytes] = None  # 필요시에만 로드

    def read_bytes(self) -> bytes:
        if self._data_cache is None:
            self.file_stream.seek(0)
            self._data_cache = self.file_stream.read()
        return self._data_cache
```

**[HIGH] Timeout 지원**
```python
def process(self, file_context, *, timeout: Optional[float] = None, ...) -> ExtractionResult:
    if timeout:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            future = ex.submit(self._run_pipeline, file_context, ...)
            return future.result(timeout=timeout)
```

**[MEDIUM] Async 지원**
```python
class BaseHandler(ABC):
    async def aprocess(self, file_context: FileContext, **kwargs) -> ExtractionResult:
        """Async wrapper using asyncio.to_thread"""
        import asyncio
        return await asyncio.to_thread(self.process, file_context, **kwargs)
```

**[MEDIUM] Progress 콜백**
```python
def process(self, file_context, *, on_progress: Optional[Callable[[PipelineStage, float], None]] = None, ...):
    stages = [Stage1, Stage2, Stage3, Stage4, Stage5]
    for i, stage_fn in enumerate(stages):
        result = stage_fn(...)
        if on_progress:
            on_progress(stage, (i+1)/5)
```

---

### 2.2 PDF Handler (`handlers/pdf/`)

#### 현재 구조
- `PDFHandler`: config의 `mode` 옵션으로 `PdfPlusContentExtractor` 또는 `PdfDefaultContentExtractor` 선택
- Converter/Preprocessor/MetadataExtractor는 양 모드 공유

#### 현재 문제점

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | mode 문자열 비교 (`mode == "default"`) — 오타 시 묵시적으로 plus 모드로 fallback | 🔴 HIGH |
| 2 | 암호화된 PDF 처리 없음 (즉시 예외 또는 빈 텍스트 반환) | 🔴 HIGH |
| 3 | 다단 컬럼 레이아웃 읽기 순서 미지원 (좌→우→하 순서 보장 안됨) | 🟡 MEDIUM |
| 4 | 스캔 PDF(텍스트 레이어 없음) 탐지 후 자동 OCR 전환 없음 | 🟡 MEDIUM |
| 5 | 대용량 PDF 페이지별 스트리밍 처리 없음 | 🟡 MEDIUM |

#### 개선 방안

**[HIGH] Mode 유효성 검증**
```python
# _constants.py에 추가
PDF_VALID_MODES = frozenset({PDF_MODE_DEFAULT, PDF_MODE_PLUS})

# handler.py
def create_content_extractor(self) -> BaseContentExtractor:
    mode = self._config.get_format_option(PDF_FORMAT_OPTION_KEY, PDF_MODE_OPTION, PDF_MODE_PLUS)
    if mode not in PDF_VALID_MODES:
        raise ValueError(f"Invalid PDF mode: {mode!r}. Valid: {PDF_VALID_MODES}")
    ...
```

**[HIGH] 암호화 PDF 지원**
```python
# converter.py
def convert(self, file_context: FileContext, *, password: Optional[str] = None, **kwargs):
    doc = fitz.open(stream=file_context.file_data, filetype="pdf")
    if doc.needs_pass:
        if password is None:
            raise ConversionError("PDF is password-protected. Provide 'password' kwarg.")
        if not doc.authenticate(password):
            raise ConversionError("Incorrect PDF password.")
    return doc
```

**[MEDIUM] 스캔 PDF 자동 OCR fallback**
```python
# preprocessor.py
def preprocess(self, converted, **kwargs):
    # 텍스트 레이어 있는지 샘플링
    total_text = sum(len(page.get_text()) for page in converted[:5])
    if total_text < 50 and self._ocr_engine:
        # OCR 플래그 설정
        return PreprocessedData(content=converted, resources={"needs_ocr": True})
```

---

### 2.3 DOCX Handler (`handlers/docx/`)

#### 현재 문제점

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | `re` 모듈을 `extract_text()` 내부에서 매 호출마다 import | `content_extractor.py:L107` | 🔴 HIGH |
| 2 | 이미지 중복 제거가 `rel_id` 기반 (같은 이미지 다른 rel_id면 중복 저장) | `content_extractor.py:_extract_image_by_rel()` | 🔴 HIGH |
| 3 | 차트를 index 순으로 매칭 — 문서 순서와 불일치 시 잘못된 차트 연결 | `content_extractor.py:_process_drawing()` | 🟡 MEDIUM |
| 4 | 헤더/푸터, 각주/미주, 텍스트박스 추출 없음 | `content_extractor.py` | 🟡 MEDIUM |
| 5 | 주석(Comments), 변경 추적(Track Changes) 미지원 | `content_extractor.py` | 🟡 MEDIUM |
| 6 | 1×1 단일 컬럼 테이블을 무조건 일반 텍스트로 변환 (컨텍스트 손실) | `_format_table()` | 🟢 LOW |

#### 개선 방안

**[HIGH] 모듈 레벨 import 수정**
```python
# content_extractor.py 상단으로 이동
import re
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# extract_text() 내에서
result = _EXCESS_NEWLINES.sub("\n\n", result)
```

**[HIGH] 이미지 중복제거: content hash 기반으로 변경**
```python
def _extract_image_by_rel(self, rel_id, doc, processed_images: Set[str]) -> Optional[str]:
    ...
    image_data: bytes = rel.target_part.blob
    # rel_id 대신 content hash로 중복 체크
    content_hash = hashlib.md5(image_data).hexdigest()
    if content_hash in processed_images:
        return None
    ...
    processed_images.add(content_hash)
```

**[MEDIUM] 헤더/푸터 추출**
```python
def _extract_headers_footers(self, doc) -> List[str]:
    """문서 섹션의 헤더/푸터 텍스트 추출"""
    results = []
    for section in doc.sections:
        for hf in [section.header, section.footer]:
            if hf and not hf.is_linked_to_previous:
                text = "\n".join(p.text for p in hf.paragraphs if p.text.strip())
                if text:
                    results.append(f"[Header/Footer]\n{text}")
    return results
```

**[MEDIUM] 차트 매칭 개선: 관계 ID 기반**
```python
# preprocessor.py에서 차트를 {rel_id: chart_text} 딕셔너리로 저장
# content_extractor.py에서 drawing.rel_id로 직접 조회
def _process_drawing(self, drawing, doc, charts_by_rel, ...):
    if drawing.kind == DrawingKind.CHART:
        return charts_by_rel.get(drawing.rel_id, "[Chart]")
```

---

### 2.4 XLSX Handler (`handlers/xlsx/`)

#### 현재 문제점

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | `ws._images` — openpyxl 내부 API 사용 (버전 업 시 깨짐) | `content_extractor.py:_extract_sheet_images()` | 🔴 HIGH |
| 2 | `img._data()` — openpyxl 내부 API 사용 | 동일 | 🔴 HIGH |
| 3 | 모든 시트 이미지에 동일한 `custom_name="excel_sheet_img"` 사용 (덮어쓰기 위험) | `content_extractor.py:L215` | 🔴 HIGH |
| 4 | 수식(Formula) 결과값 대신 수식 문자열 저장 (예: `=SUM(A1:A10)`) | `preprocessor.py` | 🟡 MEDIUM |
| 5 | 숨김 시트(hidden sheet) 처리 안됨 (포함 여부 제어 불가) | `content_extractor.py` | 🟡 MEDIUM |
| 6 | 스파스 스프레드시트(데이터가 드문드문) 레이아웃 감지 오류 가능 | `_layout.py` | 🟡 MEDIUM |

#### 개선 방안

**[HIGH] 고유 이미지 이름 생성**
```python
import uuid

def _extract_sheet_images(self, ws, all_images, processed_hashes):
    ...
    unique_name = f"excel_{ws.title}_{uuid.uuid4().hex[:8]}"
    tag = self._image_service.save_and_tag(
        image_bytes=img_data,
        custom_name=unique_name,
    )
```

**[HIGH] 내부 API 대체 (ZIP 직접 파싱)**
```python
def _extract_images_from_workbook_zip(file_data: bytes) -> dict:
    images = {}
    with ZipFile(BytesIO(file_data)) as zf:
        for name in zf.namelist():
            if name.startswith("xl/media/"):
                images[name] = zf.read(name)
    return images
```

**[MEDIUM] 수식 평가 모드 지원**
```python
# converter.py
def convert(self, file_context, *, evaluate_formulas=False, **kwargs):
    if evaluate_formulas:
        wb = openpyxl.load_workbook(BytesIO(data), data_only=True)
    else:
        wb = openpyxl.load_workbook(BytesIO(data))
    return wb
```

**[MEDIUM] 숨김 시트 처리**
```python
for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    if ws.sheet_state == "hidden":
        include_hidden = self._config.get_format_option("xlsx", "include_hidden_sheets", False)
        if not include_hidden:
            continue
```

---

### 2.5 Image Handler (`handlers/image/`)

#### 현재 문제점

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | OCR 없이는 이미지 태그만 출력 (텍스트 없음) — 사용자가 OCR engine 설정 필수 | 🟡 MEDIUM |
| 2 | 대용량 이미지의 OCR 전 리사이즈 없음 (고해상도 → 느린 OCR, 높은 API 비용) | 🟡 MEDIUM |
| 3 | EXIF 메타데이터 미추출 (촬영일, GPS, 카메라 정보 등) | 🟢 LOW |
| 4 | 이미지 포맷 변환 없음 (TIFF, HEIC 등 일부 포맷은 OCR 엔진 미지원) | 🟢 LOW |

#### 개선 방안

**[MEDIUM] OCR 전 자동 리사이즈**
```python
# preprocessor.py
def preprocess(self, converted, *, max_ocr_dimension=4096, **kwargs):
    img_data, fmt = converted
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img_data))
    if max(img.size) > max_ocr_dimension:
        img.thumbnail((max_ocr_dimension, max_ocr_dimension), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=fmt or "PNG")
        img_data = buf.getvalue()
    return PreprocessedData(content=img_data, resources={"format": fmt})
```

---

### 2.6 HWP Handler (`handlers/hwp/`)

#### 현재 문제점

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | pyhwp 라이브러리 의존성 — 유지보수 불안정 (최신 HWP 5.1+ 미지원 가능) | 🔴 HIGH |
| 2 | HWP 내 이미지 추출 완성도 낮음 (바이너리 OLE 구조 파싱 복잡) | 🟡 MEDIUM |
| 3 | HWP 표(Table) 추출 불완전 (vMerge/hMerge 처리 부재 가능) | 🟡 MEDIUM |

#### 개선 방안

**[HIGH] LibreOffice 변환 fallback**
```python
def _check_delegation(self, file_context, **kwargs):
    try:
        # pyhwp 직접 파싱 시도
        ...
    except Exception:
        if self._libreoffice_available():
            docx_path = self._convert_via_libreoffice(file_context)
            new_context = create_file_context(docx_path, "docx")
            return self._delegate_to("docx", new_context, **kwargs)
        raise
```

---

### 2.7 HandlerRegistry (`handlers/registry.py`)

#### 현재 문제점

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | `register_defaults()` 실패 시 `logger.info`만 출력 — 핸들러 미등록을 사용자가 모름 | 🟡 MEDIUM |
| 2 | 핸들러 언등록(unregister) API 없음 | 🟡 MEDIUM |
| 3 | 같은 핸들러 인스턴스를 여러 확장자가 공유 — 상태 공유 위험 | 🟡 MEDIUM |
| 4 | HTML 핸들러가 `register_defaults()`에 없음 (README엔 지원한다고 명시) | 🔴 HIGH |

#### 개선 방안

**[HIGH] HTML 핸들러 추가**
```python
# registry.py register_defaults()에 추가
("contextifier.handlers.html.handler", "HTMLHandler"),
```

**[MEDIUM] 등록 실패 가시성 향상**
```python
    logger.warning(f"Handler {class_name} failed to register: {e}")
```

---

## 3. 아키텍처 레벨 고도화 방안

### 3.1 비동기/병렬 처리 아키텍처

```python
class AsyncDocumentProcessor(DocumentProcessor):
    async def extract_text_async(self, file_path, **kwargs) -> str:
        import asyncio
        return await asyncio.to_thread(self.extract_text, file_path, **kwargs)

    async def extract_batch_async(self, file_paths: List[str], *, max_concurrent=4) -> List[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_one(path):
            async with semaphore:
                return await self.extract_text_async(path)
        return await asyncio.gather(*[process_one(p) for p in file_paths])
```

### 3.2 캐싱 레이어

```python
class CachedDocumentProcessor(DocumentProcessor):
    def __init__(self, *args, cache_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = cache_backend or {}

    def extract_text(self, file_path, **kwargs) -> str:
        file_hash = self._hash_file(file_path)
        cache_key = f"{file_hash}:{sorted(kwargs.items())}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = super().extract_text(file_path, **kwargs)
        self._cache[cache_key] = result
        return result
```

### 3.3 플러그인 시스템

```python
def register_plugins(self) -> None:
    import importlib.metadata
    for ep in importlib.metadata.entry_points(group="contextifier.handlers"):
        try:
            handler_class = ep.load()
            self.register(handler_class)
        except Exception as e:
            logger.warning(f"Plugin {ep.name} failed: {e}")
```

---

## 4. 테스트 인프라 부재

현재 레포에 **테스트 코드가 없습니다.** 이것이 가장 큰 리스크입니다.

### 필요한 테스트 구조

```
tests/
├── unit/
│   ├── handlers/
│   │   ├── test_pdf_handler.py
│   │   ├── test_docx_handler.py
│   │   ├── test_xlsx_handler.py
│   │   └── test_image_handler.py
│   ├── pipeline/
│   │   ├── test_base_handler.py
│   │   └── test_content_extractor.py
│   └── services/
│       ├── test_image_service.py
│       └── test_table_service.py
├── integration/
│   ├── fixtures/
│   └── test_e2e.py
└── conftest.py
```

---

## 5. 우선순위 로드맵

### Phase 1 — 버그 수정 (즉시)
| 항목 | 핸들러 | 설명 |
|------|--------|------|
| XLSX 이미지 이름 중복 | xlsx | `excel_sheet_img` → 고유 UUID 기반 이름 |
| DOCX re 모듈 import 위치 | docx | 모듈 레벨로 이동 |
| HTML 핸들러 누락 | registry | `register_defaults()`에 HTMLHandler 추가 |
| PDF mode 유효성 검증 | pdf | 잘못된 mode 즉시 에러 |

### Phase 2 — 기능 강화 (1~2주)
| 항목 | 핸들러 | 설명 |
|------|--------|------|
| DOCX 헤더/푸터 추출 | docx | section.header/footer 파싱 |
| DOCX 이미지 중복제거 개선 | docx | rel_id → content hash |
| XLSX 내부 API 제거 | xlsx | `_images`, `_data()` 대체 |
| 암호화 PDF 지원 | pdf | password kwarg |
| XLSX 숨김 시트 제어 | xlsx | config option |

### Phase 3 — 아키텍처 고도화 (1개월)
| 항목 | 설명 |
|------|------|
| 비동기 지원 | `AsyncDocumentProcessor` |
| 캐싱 레이어 | 파일 해시 기반 캐시 |
| 스트리밍 지원 | 대용량 파일 페이지별 처리 |
| 테스트 인프라 | pytest 기반 단위/통합 테스트 |
| 플러그인 시스템 | entry_points 기반 외부 핸들러 |
| 메모리 효율화 | 파일 스트리밍 (read_bytes 제거) |

---

## 6. 결론

Contextifier v2는 v1 대비 아키텍처적으로 크게 개선된 프로젝트입니다. 5단계 파이프라인 강제, DI 기반 서비스, 명확한 핸들러 분리 등 설계 원칙이 잘 구현되어 있습니다.

**핵심 개선 포인트:**
1. **XLSX 내부 API 의존성 제거** — 가장 시급한 안정성 문제
2. **HTML 핸들러 추가** — README와 실제 구현 불일치
3. **DOCX 헤더/푸터/각주 추출** — 문서 정보 손실 방지
4. **비동기/스트리밍** — 프로덕션 환경에서의 대용량 처리
5. **테스트 인프라 구축** — 리그레션 방지를 위한 필수 과제
