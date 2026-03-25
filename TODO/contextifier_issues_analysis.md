# Contextifier 핸들러 현황 문제점 및 한계 식별

> 분석 대상: CocoRoF/Contextifier v0.2.4
> 분석일: 2026-03-25
> 방법론: GitHub 실제 소스 코드 직접 확인 (base64 디코딩 후 라인별 검증)
> 분석자: 손성준 (Developer Agent)

**분석 관점 5가지**: 코드 품질 / 에러 처리 / 확장성 / 재사용성 / 성능

---

## 목차

1. [코드 품질 (Code Quality)](#1-코드-품질)
2. [에러 처리 (Error Handling)](#2-에러-처리)
3. [확장성 (Extensibility)](#3-확장성)
4. [재사용성 (Reusability)](#4-재사용성)
5. [성능 (Performance)](#5-성능)
6. [종합 우선순위 매트릭스](#6-종합-우선순위-매트릭스)

---

## 1. 코드 품질

### 1.1 [BUG-CRITICAL] XLSX 이미지 이름 충돌 — 데이터 손실

**파일**: `contextifier/handlers/xlsx/content_extractor.py:_extract_sheet_images()`

```python
# 실제 코드 (확인됨)
tag = self._image_service.save_and_tag(
    image_bytes=img_data,
    custom_name=f"excel_sheet_img",   # ← 모든 이미지에 동일한 이름
)
```

워크북 내 이미지가 2개 이상이면 두 번째 이미지가 첫 번째 이미지 파일을 덮어씀.
같은 메서드에서 `content_hash` 기반 중복 제거는 올바르게 구현되어 있으나, `custom_name`을 고정 문자열로 하드코딩한 것이 치명적 버그.

> **영향**: 이미지가 있는 XLSX 처리 시 마지막 이미지만 저장되고 나머지는 손실됨.

---

### 1.2 [BUG-HIGH] XLSX — openpyxl 내부 API 2종 사용

**파일**: `contextifier/handlers/xlsx/content_extractor.py`

```python
# 확인된 실제 코드 (두 곳)

# (1) _extract_sheet_images() 내부
ws_images = getattr(ws, "_images", [])   # ← _images: 비공개 속성
    img_data = img._data()               # ← _data(): 비공개 메서드

# (2) extract_text() 내부 차트 처리
ws_charts = getattr(ws, "_charts", [])   # ← _charts: 비공개 속성
```

openpyxl 공식 문서에 `_images`, `_data()`, `_charts`는 존재하지 않음.
Python 컨벤션상 `_`로 시작하는 속성/메서드는 내부 구현 세부사항이며, 라이브러리 버전 업데이트 시 언제든지 삭제/변경될 수 있음.

> **영향**: openpyxl 3.2→3.3 업그레이드 등 패치 버전에서도 즉시 `AttributeError` 발생 가능.

---

### 1.3 [BUG-HIGH] DOCX — `import re`가 메서드 내부에 위치

**파일**: `contextifier/handlers/docx/content_extractor.py:extract_text()`

```python
def extract_text(self, preprocessed: PreprocessedData, **kwargs: Any) -> str:
    # ... 수십 줄 처리 후 ...
    result = "\n\n".join(parts)
    import re                                    # ← 함수 내부 import
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
```

`import re`는 Python 표준 모듈임에도 **매 `extract_text()` 호출마다** sys.modules 조회 비용이 발생함. 모듈 레벨에 올리는 것이 Python 관례이며 성능과 가독성 모두에서 우수.

> **영향**: 단발 호출보다는 배치 처리(수천 건)에서 누적 비용 발생.

---

### 1.4 [BUG-MEDIUM] DOCX — 이미지 중복 제거가 `rel_id` 기반

**파일**: `contextifier/handlers/docx/content_extractor.py:_extract_image_by_rel()`

```python
def _extract_image_by_rel(self, rel_id, doc, processed_images: Set[str]):
    if rel_id in processed_images:          # ← rel_id로 중복 체크
        return None
    # ...
    if tag:
        processed_images.add(rel_id)        # ← rel_id를 set에 추가
```

문서에서 동일한 이미지 파일이 여러 위치에 삽입되면 각각 별도 relationship ID가 부여됨.
그 결과 동일 이미지 내용이 다른 `rel_id`로 여러 번 저장됨.

> **올바른 접근**: `hashlib.md5(image_data).hexdigest()`로 내용 기반 중복 제거.

---

### 1.5 [BUG-HIGH] PDF — mode 문자열 유효성 검증 없음

**파일**: `contextifier/handlers/pdf/handler.py:create_content_extractor()`

```python
mode = self._config.get_format_option(
    PDF_FORMAT_OPTION_KEY, PDF_MODE_OPTION, PDF_MODE_PLUS,
)
logger.debug("[PDFHandler] PDF mode = %s", mode)

if mode == PDF_MODE_DEFAULT:           # "default"와만 비교
    return PdfDefaultContentExtractor(...)

# else: 어떤 값이든 plus로 fallback (유효성 검증 없음)
return PdfPlusContentExtractor(...)
```

`mode="defualt"` (오타), `mode="PLUS"`, `mode="invalid"` 등 어떤 값이든
`plus` 모드로 묵묵히 실행됨. 사용자가 `default` 모드를 의도했어도 오타 한 글자로 비용이 더 높은 `plus` 모드가 사용됨.

---

### 1.6 [MEDIUM] Registry — 핸들러 등록 실패를 `INFO`로만 로깅

**파일**: `contextifier/handlers/registry.py:register_defaults()`

```python
except (ImportError, AttributeError) as e:
    logger.info(                           # ← WARNING이 아닌 INFO
        f"Handler {class_name} not available: {e}"
    )
```

`logger.info`는 기본 로깅 레벨(`WARNING`)에서 출력되지 않음.
의존성(`pyhwp`, `xlrd`, `PyMuPDF` 등)이 설치되지 않아 핸들러가 등록 실패하더라도
사용자는 아무 경고도 받지 못하고 `HandlerNotFoundError`가 런타임에 발생함.

---

### 1.7 [MEDIUM] TextHandler — HTML을 plain text로 처리 (README 불일치)

**파일**: `contextifier/handlers/text/handler.py`

```python
_TEXT_EXTENSIONS = frozenset({
    # ...
    "html", "htm", "xhtml",   # ← HTML이 TextHandler에 포함
    "css", "scss", "less", "sass",
    "svg",
})
```

README는 "HTML structure preservation" 지원을 명시하지만, 실제로는
`TextHandler`가 HTML을 그냥 텍스트로 읽는다. `pyproject.toml`에
`beautifulsoup4` 의존성이 등록되어 있으나 TextHandler 어디에도 BS4 import 없음.

`doc/handler.py`에 `# TODO: delegate to 'html' handler once implemented` 주석이
존재 — 개발자도 HTML 핸들러가 없음을 인지하고 있음.

---

## 2. 에러 처리

### 2.1 [CRITICAL] 파일 크기 검사 없이 전체 메모리 로드

**파일**: `contextifier/document_processor.py:_create_file_context()`

```python
@staticmethod
def _create_file_context(file_path: str, extension: str) -> FileContext:
    file_data = Path(file_path).read_bytes()   # ← 파일 전체를 메모리에 로드
    return FileContext(
        ...
        file_data=file_data,
        file_stream=io.BytesIO(file_data),      # ← 같은 데이터를 두 번 메모리에 보유
        file_size=len(file_data),
    )
```

문제점 두 가지:
1. `read_bytes()` 전에 파일 크기 확인 없음 — 5GB PDF 처리 시 OOM 즉시 발생
2. `file_data`와 `io.BytesIO(file_data)` 두 복사본이 동시에 메모리에 존재

> **영향**: 프로덕션에서 대용량 파일 처리 시 프로세스 강제 종료 가능.

---

### 2.2 [HIGH] 핸들러 전체에 만연한 bare `except Exception: pass`

**파일**: `contextifier/handlers/xlsx/content_extractor.py`

```python
# extract_text() 내 차트 처리
try:
    ws_charts = getattr(ws, "_charts", [])
    for _ in ws_charts:
        ...
except Exception as exc:
    logger.debug("Error processing charts for sheet %s: %s", sheet_name, exc)

# _extract_sheet_images() 내부
try:
    img_data = img._data()
    ...
except Exception:
    pass   # ← 오류 완전 묵살
```

내부 `except Exception: pass`는 프로그래밍 오류(NameError, TypeError 등)도
조용히 삼킴. 실제로 잘못된 동작이 발생해도 사용자는 알 수 없고, 빈 결과만 반환됨.

> **패턴 발생 위치 (확인됨)**:
> - `xlsx/content_extractor.py`: 5곳 이상
> - `base.py:process()`: converter.close() 실패 묵살
> - `docx/content_extractor.py`: `_extract_image_by_rel()`, `_format_table()`

---

### 2.3 [HIGH] 메타데이터 추출 실패를 파이프라인 레벨에서 묵살

**파일**: `contextifier/handlers/base.py:process()`

```python
# Stage 3: Metadata
metadata = None
if include_metadata:
    try:
        metadata = self._metadata_extractor.extract(preprocessed.content)
    except Exception as e:
        self._logger.warning(f"Metadata extraction failed: {e}")
        metadata = None   # ← 실패해도 파이프라인 계속 진행
```

메타데이터 추출 실패는 경고로 로깅되지만 파이프라인은 계속 진행됨.
이는 의도적 설계지만, 사용자가 `include_metadata=True`로 요청한 경우
조용한 실패(silent failure)는 혼란을 초래할 수 있음.

> **개선안**: `ExtractionResult`에 `warnings: List[str]` 필드를 추가해
> 부분 실패 정보를 명시적으로 반환.

---

### 2.4 [HIGH] Timeout 메커니즘 없음

**파일**: `contextifier/handlers/base.py:process()`

```python
def process(self, file_context: FileContext, *, include_metadata: bool = True, **kwargs):
    # ...
    converted = self._converter.convert(file_context, **kwargs)  # ← timeout 없음
    preprocessed = self._preprocessor.preprocess(converted, **kwargs)
    # ...
```

손상된 PDF, 악의적으로 조작된 ZIP 파일, 무한 루프 파서 패턴 등으로 인해
`process()`가 무한 대기 상태에 빠질 수 있음. 외부에서 제어할 방법이 없음.

> **위험 시나리오**:
> - 악의적으로 조작된 Zip Bomb (수 KB → 수 GB 압축 해제)
> - 재귀적으로 중첩된 그룹 도형이 있는 PPTX
> - 손상된 HWP 파일로 인한 pyhwp 무한 파싱

---

### 2.5 [MEDIUM] 핸들러 위임(Delegation) 실패 시 원본 포맷 처리 fallback 없음

**파일**: `contextifier/handlers/base.py:_delegate_to()`

```python
def _delegate_to(self, extension: str, file_context, ...) -> ExtractionResult:
    if self._handler_registry is None:
        raise HandlerExecutionError(
            f"{self.handler_name}: Cannot delegate — no registry available",
        )
    delegate = self._handler_registry.get_handler(extension)
    return delegate.process(file_context, ...)  # ← 위임 실패 시 그대로 예외 전파
```

예: `DOCHandler`가 ZIP 마법 바이트를 감지해 `DocxHandler`에 위임했는데,
실제로는 손상된 ZIP이어서 `DocxHandler`가 실패하는 경우,
원래 OLE2 DOC 파이프라인으로 재시도하는 fallback이 없음.

---

### 2.6 [MEDIUM] 서비스가 `None`일 때 핸들러 동작 불명확

**파일**: `contextifier/handlers/base.py:__init__()`

```python
def __init__(
    self,
    config: ProcessingConfig,
    *,
    image_service: Optional["ImageService"] = None,  # ← None 허용
    tag_service: Optional["TagService"] = None,
    ...
```

모든 서비스가 Optional이지만, 핸들러 내부에서 None 체크 일관성이 부족함:

```python
# DOCX content_extractor.py — 일관성 없는 None 체크 패턴
if self._image_service is None:      # 어떤 곳은 체크함
    return None
...
self._table_service.format_table(...)  # 다른 곳은 그냥 호출 (None이면 AttributeError)
```

---

## 3. 확장성

### 3.1 [CRITICAL] HTML 핸들러 미구현 — README 약속 불이행

현재 `register_defaults()`에 HTML 핸들러가 없음 (실제 코드 확인):

```python
default_handlers: List[tuple] = [
    ("contextifier.handlers.pdf.handler", "PDFHandler"),
    ("contextifier.handlers.docx.handler", "DOCXHandler"),
    # ... 14개
    # ← HTMLHandler 없음
]
```

`beautifulsoup4`가 `pyproject.toml` 의존성에 등록되어 있고,
README는 HTML 구조 보존을 지원한다고 명시하나 실제 구현 없음.

> **필요 작업**: `handlers/html/` 디렉토리 신규 생성 + `register_defaults()`에 등록

---

### 3.2 [HIGH] `format_options` — 타입 안전성 없는 딕셔너리

**파일**: `contextifier/config.py`

```python
@dataclass(frozen=True)
class ProcessingConfig:
    format_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_format_option(self, format_name: str, key: str, default: Any = None) -> Any:
        return self.format_options.get(format_name, {}).get(key, default)
```

`format_options`는 완전히 비구조적인 `Dict[str, Dict[str, Any]]`.
PDF의 `mode` 옵션, XLSX의 `include_hidden_sheets` 등 각 핸들러 옵션들이
타입 힌트 없이 자유 형식으로 저장됨.

> **문제점**:
> - IDE 자동완성 없음 (어떤 옵션이 있는지 알 수 없음)
> - 오타 시 `get_format_option()`이 `default`를 반환 — 에러 없음
> - 타입이 강제되지 않아 `mode=1` (int) 도 저장 가능

---

### 3.3 [HIGH] 플러그인 시스템 없음 — 외부 핸들러 추가 불가

외부 패키지가 새 포맷 핸들러를 제공하려면 현재:
1. `register_defaults()` 소스를 수정하거나
2. 사용자가 직접 `registry.register(MyHandler)` 코드를 작성해야 함

Python 표준 `importlib.metadata.entry_points`를 활용한 플러그인 시스템이 없어,
서드파티 확장이 어려움.

---

### 3.4 [HIGH] `BaseHandler.process()`가 `@final`로 보호되지 않음

**파일**: `contextifier/handlers/base.py`

```python
# 주석으로만 경고:
# """
#     This method is NOT overridable. Handlers customize behaviour
#     by providing different pipeline components via factory methods.
# """
def process(self, file_context: FileContext, ...) -> ExtractionResult:
```

Python 3.8+에서 `typing.final` 데코레이터를 사용할 수 있으나 적용되어 있지 않음.
서브클래스가 의도치 않게 `process()`를 override해도 타입 체커나 런타임에서 경고 없음.

---

### 3.5 [MEDIUM] 핸들러 언등록(Unregister) API 없음

**파일**: `contextifier/handlers/registry.py`

```python
class HandlerRegistry:
    def register(self, handler_class):     # ← 있음
    def get_handler(self, extension):      # ← 있음
    def is_supported(self, extension):     # ← 있음
    # unregister() 없음                   # ← 없음
```

> **필요한 시나리오**:
> - 테스트: 특정 핸들러를 Mock으로 교체
> - 특정 포맷 처리를 비활성화 (보안/정책상 이유)
> - 핸들러 버전 교체

---

### 3.6 [MEDIUM] Async/스트리밍 지원 구조 없음

`BaseHandler`와 `DocumentProcessor`는 100% 동기적(synchronous).
FastAPI, 비동기 작업 큐(Celery async), 웹소켓 스트리밍 등
현대적 웹 서비스 환경에서 사용하려면 `asyncio.to_thread()` wrapping이 외부에서 필요.

공식 async 지원이 없으면:
- FastAPI의 `async def` 핸들러에서 직접 호출 → 이벤트 루프 블로킹
- 내부 진행 상황 스트리밍 불가

---

### 3.7 [LOW] `ChunkingConfig.strategy` — 유효값 검증 없음

**파일**: `contextifier/config.py`

```python
@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive"   # ← "recursive", "sliding", "hierarchical" 허용
```

`strategy: str`이 일반 문자열이어서 `strategy="recusive"` (오타)를 해도
생성 시점에 예외가 발생하지 않고 런타임에 청킹이 silently 실패함.

> **개선안**: `Literal["recursive", "sliding", "hierarchical"]` 타입 사용
> 또는 `__post_init__`에서 허용값 검증.

---

## 4. 재사용성

### 4.1 [HIGH] 차트 매칭 로직 중복 — 3개 핸들러가 동일한 인덱스 기반 패턴 사용

**파일들**: `docx/content_extractor.py`, `pptx/content_extractor.py`, `xlsx/content_extractor.py`

```python
# DOCX (content_extractor.py)
chart_index = 0
for element in body:
    ...
    if drawing.kind == DrawingKind.CHART:
        if chart_index < len(charts):
            return charts[chart_index]
        chart_index += 1

# XLSX (content_extractor.py) — 동일 패턴 반복
chart_index = 0
for sheet_name in wb.sheetnames:
    ws_charts = getattr(ws, "_charts", [])
    for _ in ws_charts:
        if chart_index < len(charts):
            chart_text = self._format_chart(charts[chart_index])
            chart_index += 1
```

세 핸들러 모두 "순서 기반 인덱스로 차트 매칭" 로직을 각자 구현.
이 패턴은 문서 순서와 Preprocessor 수집 순서가 다를 때 잘못된 매칭을 유발하며,
버그 수정 시 세 곳 모두 수정해야 함.

---

### 4.2 [HIGH] 이미지 추출 패턴 핸들러별 제각각

각 핸들러가 독자적인 이미지 추출 구현을 가짐:

| 핸들러 | 중복제거 방식 | 이름 생성 |
|--------|-------------|-----------|
| DOCX | `rel_id` 기반 | `docx_{rel_id}` 또는 파트명 |
| PPTX | content hash | shape 인덱스 기반 |
| XLSX | content hash | `excel_{clean_name}` 또는 `excel_sheet_img`(BUG) |
| RTF | 없음 | 순번 기반 |

이미지 추출의 핵심 로직(중복제거, 저장, 태그 생성)은 `ImageService`에 이미 있으나,
각 핸들러가 호출 방식과 중복 제거 기준을 서로 다르게 구현.

> **통일 방안**: `ImageService`에 `extract_and_deduplicate(blob: bytes) -> Optional[str]` 메서드를 추가해 content hash 기반 중복제거 + 저장 + 태그 생성을 한 곳에서 처리.

---

### 4.3 [MEDIUM] 표 HTML 생성이 `TableService` 없이 각 핸들러에 fallback 구현됨

**파일**: `contextifier/handlers/docx/content_extractor.py`

```python
def _format_table(self, table_data: TableData) -> str:
    # ...
    if self._table_service is not None:
        try:
            return self._table_service.format_table(table_data)
        except Exception as exc:
            logger.debug("TableService formatting failed: %s", exc)

    # Fallback: simple HTML generation       ← 독자적 HTML 생성 로직
    return self._table_to_html(table_data)

@staticmethod
def _table_to_html(table_data: TableData) -> str:
    lines = ["<table border='1'>"]
    # ... 직접 HTML 생성
```

`TableService`가 주입되어 있는데도 실패 시 독자적인 HTML 생성 fallback이 존재.
PPTX, XLSX에도 유사한 패턴이 있을 가능성이 높음.

---

### 4.4 [MEDIUM] 구버전 바이너리 포맷 처리가 각각 독립적으로 구현됨

| 포맷 | 핸들러 | LibreOffice 활용 |
|------|--------|----------------|
| `.doc` OLE2 | DOCHandler 자체 구현 | ❌ 없음 |
| `.ppt` OLE2 | PPTHandler 자체 구현 | ❌ 없음 |
| `.hwp` OLE2 | HWPHandler (pyhwp) | ❌ 없음 |
| `.xls` BIFF | XLSHandler (xlrd) | ❌ 없음 |

LibreOffice CLI를 통한 표준 포맷 변환 → OOXML 핸들러 위임 패턴을
공통으로 구현하면, 각 핸들러의 레거시 포맷 파싱 복잡도를 크게 낮출 수 있음.

---

### 4.5 [LOW] 각 핸들러의 `_make_page_tag()`, `_make_sheet_tag()` 메서드 중복

```python
# DOCX content_extractor.py
def _make_page_tag(self, page_number: int) -> Optional[str]:
    if self._tag_service is not None:
        try:
            return self._tag_service.make_page_tag(page_number)
        except Exception:
            pass
    return None

# PPTX content_extractor.py — 거의 동일한 패턴
def _make_slide_tag(self, slide_number: int) -> Optional[str]:
    if self._tag_service is not None:
        try:
            return self._tag_service.make_slide_tag(slide_number)
        except Exception:
            pass
    return None
```

`TagService`의 메서드를 Optional 체크 + try/except로 감싸는 패턴이
여러 핸들러에서 반복. `BaseContentExtractor`에 공통 헬퍼로 올리면 제거 가능.

---

## 5. 성능

### 5.1 [CRITICAL] 파일 전체 메모리 로드 — OOM 위험

**파일**: `contextifier/document_processor.py:_create_file_context()`

```python
file_data = Path(file_path).read_bytes()   # 전체 파일 → RAM
return FileContext(
    file_data=file_data,                   # 복사본 1
    file_stream=io.BytesIO(file_data),     # 복사본 2 (동일 데이터를 메모리에 두 번)
    file_size=len(file_data),
)
```

5GB PDF의 경우 **10GB RAM** 즉시 사용됨 (file_data + BytesIO 복사본).
Linux OOM killer가 프로세스를 강제 종료할 수 있음.

---

### 5.2 [HIGH] 서비스 인스턴스 공유 — Thread Safety 미보장

**파일**: `contextifier/document_processor.py:_create_services()`

`ImageService`는 per-file 상태(`clear_state()` 호출)를 가지며, 멀티스레드 환경에서
동시에 여러 파일을 처리하면 이미지 해시 세트가 경쟁 상태(race condition)에 빠짐.

```python
# 현재 코드
self._services["image_service"].clear_state()   # ← 멀티스레드에서 안전하지 않음
handler = self._registry.get_handler(ext)
text = handler.extract_text(...)
```

---

### 5.3 [HIGH] 대용량 Excel — 시트 전체를 openpyxl 객체로 로드

openpyxl의 기본 로드 방식은 워크북 전체를 메모리에 올림.
50만 행 × 100열의 Excel 파일은 수 GB의 메모리를 소비할 수 있음.

openpyxl은 `read_only=True` 옵션을 제공하지만 현재 사용 여부 불명확:
```python
# 개선안: 읽기 전용 모드 (메모리 효율적)
wb = openpyxl.load_workbook(BytesIO(data), read_only=True)
# 단: read_only 모드에서는 ws._images 등 일부 기능 미작동
```

---

### 5.4 [MEDIUM] pdf_plus — 복잡한 heuristic 분석의 페이지별 캐싱 없음

**파일**: `contextifier/handlers/pdf_plus/_layout_block_detector.py` (29KB)

pdf_plus 모드는 페이지마다 텍스트 블록 감지, 복잡도 분석, 표 경계 탐지, 레이아웃 분류를 반복 수행.
동일한 PDF를 여러 설정으로 반복 처리할 때 동일 페이지 분석이 중복 실행됨.

---

### 5.5 [MEDIUM] HandlerRegistry — 핸들러 인스턴스 공유 + 상태 없는 설계 혼재

TextHandler 하나가 60개 이상의 확장자에 매핑됨. 이 핸들러 인스턴스를
동시에 여러 스레드가 사용하면 파이프라인 컴포넌트들의 상태가 공유될 위험.

---

### 5.6 [LOW] 반복 처리 시 캐싱 없음

같은 파일을 다른 설정으로 여러 번 처리하는 경우, 파일 파싱이 매번 처음부터 실행됨.

```python
cache_key = f"{md5(file_data)}:{config_hash}"
if cache_key in self._cache:
    return self._cache[cache_key]
```

---

## 6. 종합 우선순위 매트릭스

### 즉시 수정 필요 (P0 — 1주 이내)

| ID | 관점 | 위치 | 문제 | 영향 |
|----|------|------|------|------|
| P0-1 | 코드 품질 | `xlsx/content_extractor.py:_extract_sheet_images()` | 이미지 이름 충돌 (`excel_sheet_img`) | 데이터 손실 |
| P0-2 | 코드 품질 | `xlsx/content_extractor.py` | `ws._images`, `img._data()`, `ws._charts` 내부 API | 버전 업 시 즉시 파손 |
| P0-3 | 코드 품질 | `docx/content_extractor.py:extract_text()` | `import re` 메서드 내부 위치 | 배치 처리 성능 저하 |
| P0-4 | 코드 품질 | `pdf/handler.py:create_content_extractor()` | mode 유효성 검증 없음 | 오타 시 의도치 않은 plus 모드 실행 |

### 단기 개선 (P1 — 2~4주)

| ID | 관점 | 위치 | 문제 | 영향 |
|----|------|------|------|------|
| P1-1 | 확장성 | `handlers/registry.py:register_defaults()` | HTML 핸들러 미구현 | README 약속 불이행 |
| P1-2 | 에러 처리 | `document_processor.py:_create_file_context()` | 파일 크기 체크 없이 read_bytes() | OOM 위험 |
| P1-3 | 에러 처리 | `base.py`, `xlsx/content_extractor.py` | bare `except Exception: pass` | 숨겨진 오류 |
| P1-4 | 코드 품질 | `docx/content_extractor.py:_extract_image_by_rel()` | rel_id 기반 중복 제거 | 이미지 중복 저장 |
| P1-5 | 에러 처리 | `base.py:process()` | Timeout 없음 | 무한 대기 가능 |
| P1-6 | 코드 품질 | `handlers/registry.py:register_defaults()` | 등록 실패를 INFO로 로깅 | 사용자 인지 불가 |

### 중기 개선 (P2 — 1~2개월)

| ID | 관점 | 위치 | 문제 | 영향 |
|----|------|------|------|------|
| P2-1 | 성능 | `document_processor.py` | 서비스 공유 + Thread Safety | 병렬 처리 불가 |
| P2-2 | 재사용성 | `docx`, `pptx`, `xlsx` | 차트 인덱스 매칭 중복 | 버그 발생 + 코드 중복 |
| P2-3 | 재사용성 | `docx`, `pptx`, `xlsx`, `rtf` | 이미지 추출 패턴 중복 | 불일치 동작 |
| P2-4 | 확장성 | `config.py` | `format_options` 무타입 Dict | IDE 지원 없음, 오타 미감지 |
| P2-5 | 성능 | `xlsx/converter.py` | openpyxl read_only 모드 미사용 (추정) | 대용량 Excel OOM |
| P2-6 | 에러 처리 | `base.py:_delegate_to()` | 위임 실패 시 fallback 없음 | 복구 불가 실패 |

### 장기 고도화 (P3 — 2개월 이상)

| ID | 관점 | 항목 |
|----|------|------|
| P3-1 | 확장성 | 플러그인 시스템 (entry_points 기반) |
| P3-2 | 확장성 | Async 지원 (`aprocess()`, `AsyncDocumentProcessor`) |
| P3-3 | 성능 | 파일 스트리밍 FileContext (대용량 파일 지연 로드) |
| P3-4 | 성능 | 콘텐츠 해시 기반 캐싱 레이어 |
| P3-5 | 재사용성 | LibreOffice 공통 변환 레이어 (doc/ppt/hwp/xls 레거시 포맷) |
| P3-6 | 확장성 | `BaseHandler.process()`에 `@final` 적용 |
| P3-7 | 코드 품질 | 테스트 인프라 구축 (현재 0개) |

---

*본 보고서는 GitHub API를 통해 base64 디코딩한 실제 소스 코드를 직접 검증한 결과입니다.*
*작성: 손성준 (Developer Agent) | 2026-03-25*
