# Contextifier 고도화 방안 최종 보고서

> 프로젝트: CocoRoF/Contextifier v0.2.4
> 작성일: 2026-03-25
> 작성자: 손성준 (Developer Agent)
> 의뢰자: 장하렴 CEO

---

## Executive Summary

Contextifier는 **80+ 문서 포맷을 AI 친화적 텍스트로 변환**하는 Python 라이브러리입니다.
v0.2.4는 v1 대비 아키텍처가 크게 개선되었으나(Template Method 파이프라인, DI 서비스, 불변 설정),
**핸들러 구현 완성도 격차, 데이터 손실 버그, 프로덕션 안정성 취약점**이 존재합니다.

**분석 결과 요약**:
- 실제 코드에서 확인된 버그: **5건** (P0 즉시 수정)
- 발견된 전체 이슈: **28건** (코드품질/에러처리/확장성/재사용성/성능)
- 미구현 기능 (README 약속): **HTML 핸들러**
- 테스트 코드: **0개** (가장 큰 구조적 리스크)

---

## 1. 현황 진단

### 1.1 핸들러 품질 계층

| 계층 | 핸들러 | 수준 | 주요 특징 |
|------|--------|------|----------|
| **Tier 1** | `pdf_plus`, `pptx` | ⭐⭐⭐⭐⭐ | 레이아웃 분석, 시각적 읽기 순서, 그룹 도형 재귀 처리 |
| **Tier 2** | `docx`, `xlsx`, `rtf`, `csv/tsv` | ⭐⭐⭐ | 핵심 기능 동작, 엣지 케이스 미완성 |
| **Tier 3** | `doc`, `ppt`, `hwp`, `hwpx`, `xls`, `image` | ⭐⭐ | 텍스트 추출 위주, 표/이미지/차트 미흡 |
| **미구현** | `html` | ⭐ | TextHandler가 plain text로 처리 (README 불일치) |

### 1.2 전체 데이터 흐름

```
파일 경로
    │
    ▼ DocumentProcessor._create_file_context()
FileContext { file_data: bytes, file_stream: BytesIO, ... }   ← ⚠ OOM 위험
    │
    ▼ HandlerRegistry.get_handler(ext)
    │   [마법 바이트 감지 → 위임 또는 자체 처리]
    │
    ▼ BaseHandler.process()
    ┌──────────────────────────────────────────┐
    │ Stage 1: Converter  bytes → 포맷 객체     │
    │ Stage 2: Preprocessor → PreprocessedData  │
    │          resources = {charts, images, ...}│
    │ Stage 3: MetadataExtractor → DocumentMetadata
    │ Stage 4: ContentExtractor → ExtractionResult
    │          text (태그 포함) + tables + images + charts
    │ Stage 5: Postprocessor → 최종 텍스트      │
    └──────────────────────────────────────────┘
    │
    ▼ TextChunker (전략 선택)
    │   PageStrategy → ProtectedStrategy → PlainStrategy
    │
    ▼ ChunkResult → RAG/AI 입력
```

---

## 2. 확인된 버그 (P0 — 즉시 수정)

> 아래 5건은 실제 GitHub 소스 코드를 직접 디코딩하여 확인한 버그입니다.

### BUG-1: XLSX 이미지 이름 충돌 → 데이터 손실
**파일**: `contextifier/handlers/xlsx/content_extractor.py:_extract_sheet_images()`

```python
# 현재 코드 (버그)
tag = self._image_service.save_and_tag(
    image_bytes=img_data,
    custom_name=f"excel_sheet_img",   # ← 모든 이미지에 동일한 이름
)

# 수정 코드
import hashlib
content_hash = hashlib.md5(img_data).hexdigest()[:8]
unique_name = f"excel_{ws.title}_{content_hash}"
tag = self._image_service.save_and_tag(
    image_bytes=img_data,
    custom_name=unique_name,
)
```

**영향**: 워크북에 이미지가 2개 이상이면 마지막 이미지만 남고 나머지 손실.

---

### BUG-2: XLSX openpyxl 내부 API 3종 사용 → 버전 업 시 즉시 파손
**파일**: `contextifier/handlers/xlsx/content_extractor.py`

```python
# 현재 코드 (버그)
ws_images = getattr(ws, "_images", [])   # 내부 속성
img_data  = img._data()                  # 내부 메서드
ws_charts = getattr(ws, "_charts", [])   # 내부 속성

# 수정 방향: ZIP 직접 파싱으로 교체
from zipfile import ZipFile
from io import BytesIO

def _extract_images_from_workbook_zip(file_data: bytes) -> dict:
    images = {}
    with ZipFile(BytesIO(file_data)) as zf:
        for name in zf.namelist():
            if name.startswith("xl/media/"):
                images[name] = zf.read(name)
    return images
```

**영향**: openpyxl 버전 업(3.x→4.x)에서 `AttributeError`로 즉시 중단.

---

### BUG-3: DOCX `import re` 함수 내부 위치
**파일**: `contextifier/handlers/docx/content_extractor.py:extract_text()`

```python
# 현재 코드 (버그)
def extract_text(self, preprocessed, **kwargs):
    ...
    result = "\n\n".join(parts)
    import re                              # ← 매 호출마다 import
    result = re.sub(r"\n{3,}", "\n\n", result)

# 수정 코드 (모듈 레벨)
import re
_EXCESS_NEWLINES = re.compile(r"\n{3,}")   # 컴파일 1회

def extract_text(self, preprocessed, **kwargs):
    ...
    result = _EXCESS_NEWLINES.sub("\n\n", "\n\n".join(parts))
```

**동일 패턴이 `pptx/content_extractor.py`에도 존재** (2곳 동시 수정 필요).

---

### BUG-4: PDF mode 문자열 유효성 검증 없음
**파일**: `contextifier/handlers/pdf/handler.py:create_content_extractor()`

```python
# 현재 코드 (버그)
mode = self._config.get_format_option("pdf", "mode", "plus")
if mode == "default":
    return PdfDefaultContentExtractor(...)
return PdfPlusContentExtractor(...)  # ← 오타 포함 모든 값이 plus

# 수정 코드
VALID_MODES = frozenset({"plus", "default"})
if mode not in VALID_MODES:
    raise ConfigurationError(
        f"Invalid PDF mode '{mode}'. Valid: {VALID_MODES}",
        context={"mode": mode},
    )
```

---

### BUG-5: HandlerRegistry 등록 실패를 `logger.info`로만 로깅
**파일**: `contextifier/handlers/registry.py:register_defaults()`

```python
# 현재 코드 (버그)
except (ImportError, AttributeError) as e:
    logger.info(f"Handler {class_name} not available: {e}")  # ← INFO (기본 레벨서 숨김)

# 수정 코드
    logger.warning(f"Handler {class_name} failed to register: {e}")
```

**영향**: 의존성 미설치로 핸들러가 누락되어도 사용자가 런타임 전까지 모름.

---

## 3. 기능 완성 (P1 — 1~4주)

### FC-1: HTML 핸들러 신규 구현 【최우선】

**배경**: README에 "HTML structure preservation" 명시. `beautifulsoup4` 의존성 등록. DOCHandler에 `# TODO: delegate to 'html' handler once implemented` 주석 존재.

**구현 계획**:

```
handlers/html/ (신규 디렉토리)
├── handler.py           — HTMLHandler(BaseHandler)
├── converter.py         — bytes → BeautifulSoup 객체
├── preprocessor.py      — <meta>, <title> 사전 추출
├── metadata_extractor.py — <meta name="author"> 등 파싱
└── content_extractor.py  — 구조화 HTML → AI 친화적 텍스트
```

**핵심 구현**:

```python
# content_extractor.py
from bs4 import BeautifulSoup

class HTMLContentExtractor(BaseContentExtractor):
    def extract_text(self, preprocessed, **kwargs) -> str:
        soup: BeautifulSoup = preprocessed.content
        parts = []

        for tag in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
            level = tag.name.upper()
            text = tag.get_text(strip=True)
            if text:
                parts.append(f"[{level}]{text}[/{level}]")

        for table in soup.find_all("table"):
            table_data = self._parse_html_table(table)
            if self._table_service:
                parts.append(self._table_service.format_table(table_data))

        for lst in soup.find_all(["ul", "ol"]):
            parts.append(self._extract_list(lst))

        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text:
                parts.append(text)

        return "\n\n".join(filter(None, parts))
```

**등록**:
```python
# registry.py register_defaults()에 추가
("contextifier.handlers.html.handler", "HTMLHandler"),
```

---

### FC-2: DOCX 헤더/푸터/각주 추출

```python
def _extract_headers_footers(self, doc) -> List[str]:
    results = []
    for section in doc.sections:
        for label, hf in [("Header", section.header), ("Footer", section.footer)]:
            if hf and not hf.is_linked_to_previous:
                text = "\n".join(p.text.strip() for p in hf.paragraphs if p.text.strip())
                if text:
                    results.append(f"[{label}]\n{text}")
    return results

def _extract_footnotes(self, doc) -> List[str]:
    footnotes = []
    try:
        part = doc.part.footnotes_part
        for fn in part.footnotes:
            text = " ".join(p.text.strip() for p in fn.paragraphs if p.text.strip())
            if text:
                footnotes.append(f"[Footnote] {text}")
    except AttributeError:
        pass
    return footnotes
```

---

### FC-3: PDF 암호화 파일 지원

```python
# pdf/converter.py
def convert(self, file_context, *, password: Optional[str] = None, **kwargs):
    import fitz
    doc = fitz.open(stream=file_context["file_data"], filetype="pdf")
    if doc.needs_pass:
        if not password:
            raise ConversionError(
                "PDF is password-protected. Pass password= kwarg.",
                stage="convert",
            )
        if not doc.authenticate(password):
            raise ConversionError("Incorrect PDF password.")
    return doc
```

---

## 4. 아키텍처 고도화 (P2 — 1~2개월)

### ARCH-1: 메모리 효율화 — LazyFileContext

```python
MAX_EAGER_LOAD = 100 * 1024 * 1024  # 100MB 이하만 즉시 로드

@staticmethod
def _create_file_context(file_path: str, extension: str) -> FileContext:
    file_size = os.path.getsize(file_path)

    if file_size <= MAX_EAGER_LOAD:
        file_data = Path(file_path).read_bytes()
        file_stream = io.BytesIO(file_data)
    else:
        file_data = b""
        file_stream = open(file_path, "rb")

    return FileContext(
        file_path=file_path,
        file_name=os.path.basename(file_path),
        file_extension=extension,
        file_category=get_category(extension).value,
        file_data=file_data,
        file_stream=file_stream,
        file_size=file_size,
    )
```

### ARCH-2: Thread Safety — ThreadLocal 서비스 상태

```python
import threading

class ImageService:
    def __init__(self, ...):
        self._local = threading.local()

    def _get_processed(self) -> Set[str]:
        if not hasattr(self._local, "processed"):
            self._local.processed = set()
        return self._local.processed

    def clear_state(self) -> None:
        self._local.processed = set()
```

### ARCH-3: Timeout 지원

```python
def process(
    self,
    file_context: FileContext,
    *,
    include_metadata: bool = True,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> ExtractionResult:
    if timeout is None:
        return self._execute_pipeline(file_context, include_metadata=include_metadata, **kwargs)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            self._execute_pipeline, file_context,
            include_metadata=include_metadata, **kwargs
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise HandlerExecutionError(
                f"Processing timed out after {timeout}s",
                context={"file": file_context.get("file_name"), "timeout": timeout},
            )
```

---

## 5. 장기 고도화 (P3 — 2개월 이상)

### LONG-1: Async 지원

```python
class AsyncDocumentProcessor(DocumentProcessor):
    async def extract_text_async(self, file_path, **kwargs) -> str:
        import asyncio
        return await asyncio.to_thread(self.extract_text, file_path, **kwargs)

    async def extract_batch_async(
        self,
        file_paths: List[str],
        *,
        max_concurrent: int = 4,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_one(path: str) -> str:
            async with semaphore:
                return await self.extract_text_async(path)
        return await asyncio.gather(*[process_one(p) for p in file_paths])
```

### LONG-2: 플러그인 시스템

```python
def register_plugins(self) -> None:
    import importlib.metadata
    try:
        eps = importlib.metadata.entry_points(group="contextifier.handlers")
    except Exception:
        return
    for ep in eps:
        try:
            handler_class = ep.load()
            self.register(handler_class)
            logger.info(f"Plugin handler registered: {ep.name}")
        except Exception as e:
            logger.warning(f"Plugin '{ep.name}' failed: {e}")
```

### LONG-3: 캐싱 레이어

```python
class CachedDocumentProcessor(DocumentProcessor):
    def __init__(self, *args, cache_backend=None, ttl: int = 3600, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = cache_backend or {}

    def extract_text(self, file_path, **kwargs) -> str:
        file_hash = self._hash_file(file_path)
        config_hash = hash(str(sorted(kwargs.items())))
        cache_key = f"{file_hash}:{config_hash}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = super().extract_text(file_path, **kwargs)
        self._cache[cache_key] = result
        return result
```

### LONG-4: LibreOffice 변환 레이어

```python
def convert_via_libreoffice(
    file_path: str,
    output_format: str = "docx",
    timeout: float = 60.0,
) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", output_format,
             "--outdir", tmpdir, file_path],
            capture_output=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise ConversionError(f"LibreOffice conversion failed: {result.stderr}")
        out_path = next(Path(tmpdir).glob(f"*.{output_format}"))
        return out_path.read_bytes()
```

---

## 6. 우선순위 실행 로드맵

### Phase 0: 버그 수정 (1주, ~16 시간)

| 작업 | 파일 | 예상 시간 |
|------|------|----------|
| XLSX 이미지 이름 고유화 | `xlsx/content_extractor.py` | 1h |
| XLSX 내부 API → ZIP 직접 파싱 | `xlsx/content_extractor.py` | 4h |
| DOCX + PPTX `import re` 모듈 레벨 이동 | `docx/content_extractor.py`, `pptx/content_extractor.py` | 0.5h |
| PDF mode 유효성 검증 | `pdf/handler.py` | 0.5h |
| Registry 로그 레벨 수정 | `registry.py` | 0.5h |
| 회귀 테스트 작성 (위 5건) | `tests/unit/` | 4h |

### Phase 1: 기능 완성 (2~4주, ~80 시간)

| 작업 | 우선순위 | 예상 시간 |
|------|----------|----------|
| **HTML 핸들러 신규 구현** | 🔴 최우선 | 16h |
| DOCX 헤더/푸터/각주 추출 | 🔴 | 12h |
| DOCX 이미지 중복제거 content hash | 🟡 | 4h |
| PDF 암호화 파일 지원 | 🟡 | 8h |
| PDF 스캔 자동 OCR fallback | 🟡 | 8h |
| XLSX 수식 평가 모드 (`data_only`) | 🟢 | 2h |
| XLSX 숨김 시트 처리 옵션 | 🟢 | 2h |
| 차트 매칭 rel_id 기반으로 개선 (DOCX/PPTX) | 🟡 | 12h |

### Phase 2: 아키텍처 안정화 (1~2개월)

| 작업 | 임팩트 | 예상 시간 |
|------|--------|----------|
| Thread Safety (ThreadLocal 서비스 상태) | 🔴 프로덕션 필수 | 16h |
| Timeout 지원 | 🔴 안정성 | 12h |
| 메모리 최적화 (LazyFileContext) | 🟡 대용량 파일 | 20h |
| 통합 테스트 인프라 구축 | 🔴 필수 | 20h |

### Phase 3: 장기 고도화 (2개월 이상)

| 작업 | 가치 |
|------|------|
| Async 지원 (`AsyncDocumentProcessor`) | FastAPI/비동기 환경 대응 |
| 캐싱 레이어 (`CachedDocumentProcessor`) | RAG 파이프라인 성능 |
| 플러그인 시스템 (entry_points) | 서드파티 확장 생태계 |
| LibreOffice 변환 레이어 | 레거시 포맷 완성도 |

---

## 7. 고도화 임팩트 예측

| 항목 | 현재 | Phase 0 후 | Phase 1 후 | Phase 2 후 |
|------|------|------------|------------|------------|
| 버그 수 (확인됨) | 5개 | 0개 | 0개 | 0개 |
| 지원 포맷 | 14개 | 14개 | **15개** (HTML) | 15개 |
| 테스트 커버리지 | ~0% | ~20% | ~50% | ~80% |
| 대용량 파일(>1GB) | OOM | OOM | OOM | **정상 처리** |
| 멀티스레드 안전성 | ❌ | ❌ | ❌ | **✅** |
| HTML 구조 보존 | ❌ | ❌ | **✅** | ✅ |
| PDF 암호화 지원 | ❌ | ❌ | **✅** | ✅ |

---

## 8. 관련 분석 문서 목록

| 문서 | 파일명 | 내용 |
|------|--------|------|
| 전체 아키텍처 분석 | `contextifier_analysis.md` | 구조 개요 + 주요 개선 방안 코드 예시 |
| 핸들러 심층 분석 | `contextifier_handler_deep_analysis.md` | 14개 핸들러 역할/입출력/버그/개선방안 |
| 문제점 및 한계 | `contextifier_issues_analysis.md` | 5가지 관점 28개 이슈 (코드 증거 포함) |
| 데이터 흐름 분석 | `contextifier_data_flow_analysis.md` | 전체 데이터 흐름 + 비즈니스 로직 매핑 |
| **본 문서** | `contextifier_upgrade_roadmap.md` | 고도화 방안 + 우선순위 로드맵 |

---

*본 보고서는 CocoRoF/Contextifier GitHub 소스 코드를 직접 확인하여 작성되었습니다.*
*모든 버그는 실제 코드 라인을 base64 디코딩하여 검증했습니다.*
*작성: 손성준 (Developer Agent) | 2026-03-25*
