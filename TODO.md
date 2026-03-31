# Contextifier v0.2.4 — 종합 개선 계획

> 대상: CocoRoF/Contextifier v0.2.4
> 작성일: 2026-03-31
> 근거: `TODO/` 폴더 5건 분석 보고서 + 전체 소스 코드 직접 교차 검증
> Python: >=3.12

---

## 목차

- [Phase 0 — 즉시 수정 (확인된 버그)](#phase-0--즉시-수정-확인된-버그)
- [Phase 1 — 단기 안정화 (에러 처리 · 코드 품질)](#phase-1--단기-안정화-에러-처리--코드-품질)
- [Phase 2 — 기능 완성 (미구현 · README 약속 이행)](#phase-2--기능-완성-미구현--readme-약속-이행)
- [Phase 3 — 재사용성 · 구조 개선](#phase-3--재사용성--구조-개선)
- [Phase 4 — 아키텍처 고도화 (성능 · 확장성)](#phase-4--아키텍처-고도화-성능--확장성)
- [Phase 5 — 장기 로드맵](#phase-5--장기-로드맵)
- [참고: 핸들러 품질 현황 매트릭스](#참고-핸들러-품질-현황-매트릭스)

---

## Phase 0 — 즉시 수정 (확인된 버그)

> 데이터 손실 또는 런타임 파손을 유발하는 실증된 버그.
> 모든 항목은 실제 코드에서 라인 단위로 확인 완료.

### P0-1. XLSX 이미지 이름 충돌 → 데이터 손실

- **파일**: `contextifier/handlers/xlsx/content_extractor.py` L311
- **현상**: `_extract_sheet_images()`에서 모든 이미지에 `custom_name="excel_sheet_img"` 하드코딩.
  워크북에 이미지 2개 이상이면 마지막 이미지만 남고 나머지 덮어쓰기됨.
- **참고**: 같은 파일 L176 `extract_images()`에서는 `custom_name=f"excel_{clean_name}"`을 정상 사용하므로
  `_extract_sheet_images()`만 수정하면 됨.
- **수정 방향**:
  - 이미지 데이터의 content hash(sha256 앞 8자리) + 시트 이름으로 고유 이름 생성.
  ```python
  import hashlib
  content_hash = hashlib.sha256(img_data).hexdigest()[:8]
  unique_name = f"excel_{ws.title}_{content_hash}"
  ```

### P0-2. XLSX openpyxl 비공개 API 3종 사용

- **파일**: `contextifier/handlers/xlsx/content_extractor.py`
- **위치**:
  - L103: `ws_charts = getattr(ws, "_charts", [])` — 비공개 속성
  - L296: `ws_images = getattr(ws, "_images", [])` — 비공개 속성
  - L300: `img_data = img._data()` — 비공개 메서드
- **위험**: openpyxl 버전 업(3.x→4.x)에서 `AttributeError`로 즉시 중단.
  `getattr(... , [])` fallback으로 Chart/Image가 조용히 소실될 수 있음.
- **수정 방향**:
  - 이미지: XLSX ZIP 구조(`xl/media/*`)를 직접 파싱하여 이미지 바이트를 추출.
  - 차트: `xl/charts/*.xml`을 직접 파싱하거나, Preprocessor에서 ZIP 기반으로 사전 수집.
  ```python
  from zipfile import ZipFile
  from io import BytesIO

  def _extract_images_from_zip(file_data: bytes) -> dict[str, bytes]:
      images = {}
      with ZipFile(BytesIO(file_data)) as zf:
          for name in zf.namelist():
              if name.startswith("xl/media/"):
                  images[name] = zf.read(name)
      return images
  ```

### P0-3. DOCX/PPTX `import re` 메서드 내부 위치

- **파일**:
  - `contextifier/handlers/docx/content_extractor.py` L131
  - `contextifier/handlers/pptx/content_extractor.py` L111
- **현상**: `extract_text()` 함수 본문 내부에 `import re`. 매 호출마다 `sys.modules` 조회 발생.
- **수정 방향**: 모듈 최상단으로 이동 + 컴파일된 패턴 재사용.
  ```python
  import re
  _EXCESS_NEWLINES = re.compile(r"\n{3,}")
  # extract_text() 내부에서:
  result = _EXCESS_NEWLINES.sub("\n\n", result)
  ```
- **추가 확인**: PPTX `content_extractor.py` L373에도 `import hashlib`이 메서드 내부에 존재 — 함께 이동.

### P0-4. PDF mode 문자열 유효성 검증 없음

- **파일**: `contextifier/handlers/pdf/handler.py` L79-100
- **현상**: `mode == "default"` 단일 비교만 존재.
  `"defualt"`, `"PLUS"`, `"invalid"` 등 어떤 값이든 plus 모드로 묵묵히 fallback.
- **수정 방향**:
  ```python
  _VALID_MODES = frozenset({PDF_MODE_DEFAULT, PDF_MODE_PLUS})

  mode = self._config.get_format_option(PDF_FORMAT_OPTION_KEY, PDF_MODE_OPTION, PDF_MODE_PLUS)
  if mode not in _VALID_MODES:
      raise ConfigurationError(
          f"Invalid PDF mode '{mode}'. Valid options: {_VALID_MODES}",
          context={"mode": mode},
      )
  ```

### P0-5. Registry 핸들러 등록 실패를 `logger.info`로 로깅

- **파일**: `contextifier/handlers/registry.py` L182
- **현상**: `register_defaults()`에서 핸들러 import 실패(의존성 미설치)를 `logger.info`로 처리.
  기본 로깅 레벨(`WARNING`)에서 출력되지 않아, 핸들러가 누락되어도 사용자가 인지 불가.
  (참고: 핸들러 인스턴스화 실패는 L113에서 `logger.warning`으로 올바르게 처리됨.)
- **수정 방향**: `logger.info` → `logger.warning`으로 변경.

---

## Phase 1 — 단기 안정화 (에러 처리 · 코드 품질)

> 프로덕션 안정성에 직접 영향을 미치는 에러 처리 및 코드 품질 개선.

### P1-1. 파일 크기 검사 없이 전체 메모리 로드 (OOM 위험)

- **파일**: `contextifier/document_processor.py` L558-567
- **현상**:
  1. `Path(file_path).read_bytes()` — 크기 제한 없이 파일 전체를 메모리에 로드.
  2. `file_data`(bytes)와 `BytesIO(file_data)` 두 복사본이 동시에 메모리에 존재 → 파일 크기 × 2 RAM.
  5GB PDF 처리 시 ~10GB RAM 즉시 소비 → Linux OOM killer로 프로세스 종료 가능.
- **수정 방향**:
  1. 파일 크기 상한 설정 (기본 500MB, config으로 조정 가능).
  2. `file_data`와 `file_stream` 중복 보유 제거 — `file_stream`만 유지하고 필요 시 `file_stream.read()`.
  ```python
  MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

  @staticmethod
  def _create_file_context(file_path: str, extension: str) -> FileContext:
      file_size = os.path.getsize(file_path)
      if file_size > MAX_FILE_SIZE:
          raise FileError(
              f"File size ({file_size:,} bytes) exceeds limit ({MAX_FILE_SIZE:,} bytes)",
              context={"file_path": file_path, "file_size": file_size},
          )
      file_data = Path(file_path).read_bytes()
      return FileContext(
          ...
          file_data=file_data,
          file_stream=io.BytesIO(file_data),  # 동일 객체 래핑 (복사 아님)
          file_size=file_size,
      )
  ```

### P1-2. bare `except Exception: pass` 패턴 정리

- **발생 위치 (확인됨)**:
  - `handlers/xlsx/content_extractor.py` — L73, L222, L313, L315 (최소 4곳)
  - `handlers/base.py` — L349-351 (`converter.close()` 실패 묵살)
  - `handlers/docx/content_extractor.py` — `_extract_image_by_rel()`, `_format_table()`, `_make_page_tag()`
  - `handlers/pptx/content_extractor.py` — `_try_extract_chart_data()` 내 최소 8곳
- **수정 방향**:
  - 프로그래밍 오류(TypeError, NameError 등)를 삼키지 않도록 구체적인 예외 타입으로 교체.
  - 최소한 `except Exception as e: logger.debug(...)` 수준의 로깅 추가.
  - `base.py` `converter.close()`는 `except Exception as e: logger.debug("Close failed: %s", e)` 정도면 충분.

### P1-3. ImageService 스레드 안전성 미보장

- **파일**: `contextifier/services/image_service.py` L55, L86-88
- **현상**: docstring(L55)에 "Thread-safe for concurrent handler use" 주장이 있으나,
  `_processed_hashes`(set), `_processed_paths`(list), `_counter`(int) 모두 plain mutable state — lock 없음.
  `clear_state()`를 멀티스레드 환경에서 동시 호출하면 이미지 중복 제거 경쟁 상태 발생.
  `SEQUENTIAL` 네이밍의 `self._counter += 1`(L209)도 비원자적 연산.
- **수정 방향**:
  - 방안 A: `threading.local()` 사용 → per-thread 격리.
  - 방안 B: `DocumentProcessor`에서 per-call 격리 (서비스 clone 또는 신규 생성).
  - docstring 스레드 안전성 주장은 실제 구현에 맞게 수정.

### P1-4. DOCX 이미지 중복 제거가 `rel_id` 기반

- **파일**: `contextifier/handlers/docx/content_extractor.py` L257
- **현상**: 동일한 이미지 파일이 여러 위치에 삽입되면 각각 별도 `rel_id`가 부여됨 → 같은 이미지가 중복 저장.
- **수정 방향**: content hash 기반으로 변경.
  ```python
  import hashlib
  image_data = rel.target_part.blob
  content_hash = hashlib.sha256(image_data).hexdigest()
  if content_hash in processed_images:
      return existing_tag  # 기존 태그 재사용
  processed_images[content_hash] = tag
  ```

### P1-5. `process()` 파이프라인 Timeout 미지원

- **파일**: `contextifier/handlers/base.py` L223
- **현상**: 손상된 파일, Zip Bomb, 재귀적 중첩 도형 등으로 `process()`가 무한 대기 가능.
  외부에서 제어할 방법 없음.
- **수정 방향**: `timeout` 매개변수 추가 + `concurrent.futures.ThreadPoolExecutor` 기반.
  ```python
  def process(self, file_context, *, timeout: Optional[float] = None,
              include_metadata: bool = True, **kwargs) -> ExtractionResult:
      if timeout is None:
          return self._execute_pipeline(file_context, ...)
      with concurrent.futures.ThreadPoolExecutor(1) as ex:
          future = ex.submit(self._execute_pipeline, file_context, ...)
          try:
              return future.result(timeout=timeout)
          except concurrent.futures.TimeoutError:
              raise HandlerExecutionError(
                  f"Processing timed out after {timeout}s",
                  context={"file": file_context.get("file_name")},
              )
  ```

### P1-6. PPTX 그룹 도형 재귀 깊이 제한 없음

- **파일**: `contextifier/handlers/pptx/content_extractor.py` L258-268
- **현상**: `_process_group()` → `_process_shape()` → `_process_group()` 재귀에 깊이 제한 없음.
  악의적으로 조작된 PPTX의 깊은 그룹 도형 중첩 시 `RecursionError` 발생 가능 (DoS 벡터).
- **수정 방향**: `max_depth` 매개변수 추가 (기본값 20).
  ```python
  def _process_group(self, group_shape, *, depth: int = 0, max_depth: int = 20):
      if depth >= max_depth:
          logger.warning("Group shape nesting depth %d exceeds limit", depth)
          return []
      for child in group_shape.shapes:
          results.extend(self._process_shape(child, depth=depth + 1, max_depth=max_depth))
  ```

### P1-7. OCR 모듈 `__all__`에 존재하지 않는 심볼 노출

- **파일**: `contextifier/ocr/processor.py` L166
- **현상**: `__all__`에 `DEFAULT_IMAGE_TAG_PATTERN`이 포함되어 있으나 해당 파일에 정의되지 않음.
  `from contextifier.ocr.processor import *` 시 `AttributeError` 발생.
- **수정 방향**: `__all__`에서 제거하거나, 실제 상수를 정의.

### P1-8. `format_options` — frozen dataclass에 mutable dict

- **파일**: `contextifier/config.py` L175
- **현상**: `ProcessingConfig`는 `frozen=True`이지만 `format_options: Dict[str, Dict[str, Any]]`는
  mutable dict. `config.format_options["pdf"]["mode"] = "default"` 같은 in-place 변경이 가능하여
  frozen 계약을 위반함. `__post_init__`에서 딥 카피나 immutable 변환이 없음.
- **수정 방향**:
  - `__post_init__`에서 `types.MappingProxyType`으로 래핑하여 실질적 불변성 확보.
  - 또는 `tuple`/`frozenset` 기반 구조로 교체.

---

## Phase 2 — 기능 완성 (미구현 · README 약속 이행)

> README에 명시되어 있거나 코드에 TODO로 존재하지만 아직 미구현인 기능.

### P2-1. HTML 핸들러 신규 구현 【최우선】

- **현상**:
  - README에 "HTML structure preservation" 지원 명시.
  - `pyproject.toml`에 `beautifulsoup4>=4.12.0` 의존성 등록 — 사용처 없음.
  - `handlers/doc/handler.py` L92-95: `# TODO: delegate to 'html' handler once implemented` 주석.
  - 현재 HTML은 `TextHandler`에서 plain text로 처리 (태그 제거 없음, 파싱 없음).
  - `types.py`에서 `"html"` → `FileCategory.CODE`, `"htm"/"xhtml"` → `FileCategory.WEB` — 분류 불일치.
- **필요 작업**:
  1. `handlers/html/` 디렉토리 신규 생성:
     ```
     handlers/html/
     ├── __init__.py
     ├── handler.py           — HTMLHandler(BaseHandler)
     ├── converter.py         — bytes → BeautifulSoup 객체
     ├── preprocessor.py      — <meta>, <link>, <script> 사전 처리
     ├── metadata_extractor.py— <meta name="author">, <title> 등 파싱
     └── content_extractor.py — 구조화된 HTML → AI 친화적 텍스트
     ```
  2. `registry.py` `register_defaults()`에 `("contextifier.handlers.html.handler", "HTMLHandler")` 추가.
  3. `text/handler.py` `_TEXT_EXTENSIONS`에서 `"html"`, `"htm"`, `"xhtml"` 제거.
  4. `types.py` `EXTENSION_CATEGORIES`에서 `"html"` 분류를 `CODE` → `WEB`으로 통일.
  5. `doc/handler.py` HTML 마법 바이트 분기에서 `html` 핸들러로 위임 구현.

### P2-2. PDF 암호화 파일 지원

- **파일**: `contextifier/handlers/pdf/converter.py` L42-62
- **현상**: `fitz.open()` 호출 시 `password` 매개변수 없음.
  암호화된 PDF는 PyMuPDF 내부에서 일반 예외 발생 → 사용자에게 명확한 에러 메시지 없음.
- **수정 방향**:
  ```python
  def convert(self, file_context, *, password: Optional[str] = None, **kwargs):
      doc = fitz.open(stream=file_context["file_stream"], filetype="pdf")
      if doc.needs_pass:
          if not password:
              raise ConversionError("PDF is password-protected. Provide 'password' kwarg.")
          if not doc.authenticate(password):
              raise ConversionError("Incorrect PDF password.")
      return PdfConvertedData(doc=doc, file_data=file_context["file_data"])
  ```

### P2-3. DOCX 헤더/푸터/각주 추출

- **파일**: `contextifier/handlers/docx/content_extractor.py`
- **현상**: 현재 `doc.element.body` 요소만 순회. 섹션 헤더/푸터, 각주(Footnote), 미주(Endnote) 미추출.
- **수정 방향**:
  - 헤더/푸터: `doc.sections` 순회 후 `section.header.paragraphs` / `section.footer.paragraphs` 추출.
  - 각주: `doc.part.footnotes_part` 접근 (python-docx 지원 시).
  - 추출된 내용은 본문 텍스트 끝에 별도 섹션으로 첨부.

### P2-4. XLSX 수식 평가 모드: `data_only` 옵션

- **현상**: openpyxl 기본 모드는 수식 문자열(`=SUM(A1:A10)`)을 그대로 반환.
  `data_only=True`로 로드하면 마지막 계산 결과값을 반환.
- **수정 방향**:
  - `format_options`에 `"xlsx.data_only"` 옵션 추가.
  - Converter에서 `openpyxl.load_workbook(BytesIO(data), data_only=config_value)`.

### P2-5. XLSX 숨김 시트 처리 옵션

- **현상**: 숨김 시트(`ws.sheet_state == "hidden"`)도 무조건 포함하여 처리.
- **수정 방향**:
  - `format_options`에 `"xlsx.include_hidden_sheets"` (기본값 `False`) 추가.
  - `content_extractor.py`에서 시트 순회 시 옵션 확인.

### P2-6. PDF 스캔 문서 자동 OCR 전환

- **현상**: 텍스트 레이어가 없는 스캔 PDF에서 빈 텍스트 반환. OCR 엔진 설정되어 있어도 자동 전환 없음.
- **수정 방향**:
  - Preprocessor에서 처음 5페이지 텍스트 길이 샘플링.
  - 텍스트가 임계값 미만이면 `resources["needs_ocr"] = True` 플래그 설정.
  - ContentExtractor에서 플래그 확인 후 OCR 경로 진입.

### P2-7. DOCX/PPTX/XLSX 차트 매칭: 인덱스 기반 → 관계 ID 기반

- **파일**:
  - `docx/content_extractor.py` L109-111, L207
  - `pptx/content_extractor.py` (동일 패턴)
  - `xlsx/content_extractor.py` L102-108
- **현상**: Preprocessor 수집 순서와 ContentExtractor 본문 순서가 다르면 잘못된 차트가 매핑됨.
- **수정 방향**:
  - Preprocessor에서 `{relationship_id: chart_text}` 딕셔너리로 수집.
  - ContentExtractor에서 `drawing.rel_id`로 직접 조회.

---

## Phase 3 — 재사용성 · 구조 개선

> 코드 중복 제거, 일관성 확보, 유지보수성 향상.

### P3-1. 이미지 추출 패턴 통일

- **현상**: 핸들러별 중복제거 방식/이름 생성이 제각각:

  | 핸들러 | 중복제거 방식 | 이름 생성 |
  |--------|-------------|-----------|
  | DOCX | `rel_id` 기반 | `docx_{rel_id}` |
  | PPTX | content hash | shape 인덱스 기반 |
  | XLSX | content hash | `excel_{clean_name}` (정상) / `excel_sheet_img` (버그) |
  | RTF | 없음 | 순번 기반 |

- **수정 방향**: `ImageService`에 통합 메서드 추가.
  ```python
  def extract_and_deduplicate(self, image_bytes: bytes, source_hint: str) -> Optional[str]:
      """Content hash 기반 중복 제거 + 저장 + 태그 생성."""
      content_hash = hashlib.sha256(image_bytes).hexdigest()
      if content_hash in self._processed_hashes:
          return self._hash_to_tag.get(content_hash)
      tag = self.save_and_tag(image_bytes, custom_name=f"{source_hint}_{content_hash[:8]}")
      self._hash_to_tag[content_hash] = tag
      return tag
  ```

### P3-2. 차트 인덱스 매칭 로직 공통화 (3개 핸들러 중복)

- **현상**: DOCX, PPTX, XLSX 모두 "순서 기반 인덱스로 차트 매칭" 로직을 각자 구현.
  → 버그 수정 시 3곳 동시 수정 필요.
- **수정 방향**: `ChartService` 또는 `BaseContentExtractor`에 공통 차트 매칭 헬퍼 추가.

### P3-3. `_make_page_tag()` / `_make_slide_tag()` / `_make_sheet_tag()` 패턴 중복

- **현상**: 여러 핸들러에서 TagService Optional 체크 + `except Exception: pass` 래핑 패턴 반복.
- **수정 방향**: `BaseContentExtractor`에 공통 헬퍼 메서드로 추출.
  ```python
  # pipeline/content_extractor.py (BaseContentExtractor)
  def _safe_tag(self, tag_fn, *args) -> Optional[str]:
      if self._tag_service is None:
          return None
      try:
          return tag_fn(*args)
      except Exception as e:
          logger.debug("Tag creation failed: %s", e)
          return None
  ```

### P3-4. 표 포매팅 fallback 중복

- **현상**: DOCX에 `_table_to_html()` 독자적 구현 존재.
  `TableService` 실패 시 핸들러별로 각자 HTML 생성 fallback 보유.
  DOCX `_table_to_html`은 HTML 엔티티 이스케이프 미처리 (PPTX 구현은 이스케이프 처리).
- **수정 방향**: `TableService`에 fallback HTML 생성을 포함하여 단일 출처(Single Source of Truth) 확보.
  핸들러에서 독자 HTML 생성 로직 제거.

### P3-5. ImageService 해시 알고리즘 불일치

- **파일**: `contextifier/services/image_service.py` L207, L216
- **현상**: 파일 이름 생성에 md5(L207), 중복 제거에 sha256(L216) — 동일 목적에 다른 알고리즘.
- **수정 방향**: sha256으로 통일.

### P3-6. `config.py` 타입 안전성 향상

- **현상**:
  - `format_options: Dict[str, Dict[str, Any]]` — IDE 자동완성 없음, 오타 무감지.
  - `ChunkingConfig.strategy: str` — `"recusive"` 오타도 생성 시점에 에러 없음.
- **수정 방향**:
  - `ChunkingConfig.strategy`에 `Literal["recursive", "sliding", "hierarchical"]` 적용 또는 `__post_init__` 검증.
  - `format_options`는 중기적으로 per-format typed config(`PdfFormatOptions`, `XlsxFormatOptions` 등)로 마이그레이션.

### P3-7. `types.py` FileCategory 분류 불일치 수정

- **현상**: `"html"` → `FileCategory.CODE`(L397), `"htm"/"xhtml"` → `FileCategory.WEB`(L409).
  같은 포맷인데 다른 카테고리로 분류.
- **수정 방향**: `"html"`을 `FileCategory.WEB`으로 이동 (P2-1 HTML 핸들러 구현 시 함께 처리).

---

## Phase 4 — 아키텍처 고도화 (성능 · 확장성)

> 프로덕션 환경 대응을 위한 구조적 개선.

### P4-1. 메모리 효율화: LazyFileContext

- **현상**: 파일 전체가 `file_data`(bytes) + `file_stream`(BytesIO) 두 복사본으로 메모리 상주.
- **수정 방향**:
  - 대용량 파일(>100MB)은 지연 로드: `file_stream`만 유지, `file_data`는 필요 시 `stream.read()`.
  - 소형 파일은 현행 유지 (즉시 로드).
  ```python
  @dataclass
  class LazyFileContext:
      file_path: str
      file_size: int
      _data: Optional[bytes] = None

      def read_bytes(self) -> bytes:
          if self._data is None:
              self._data = Path(self.file_path).read_bytes()
          return self._data

      def get_stream(self) -> IO[bytes]:
          return open(self.file_path, "rb")
  ```

### P4-2. Thread Safety 확보: per-call 서비스 격리

- **현상**: `DocumentProcessor`가 서비스 인스턴스를 공유하며 `clear_state()`로 상태 초기화.
  멀티스레드 환경에서 경쟁 상태 발생.
- **수정 방향**:
  - 방안 A: per-call 격리 — `extract_text()` 호출 시 서비스 새 인스턴스 생성.
  - 방안 B: ThreadLocal — `ImageService` 내부적으로 `threading.local()` 사용.

### P4-3. `BaseHandler.process()`에 `@final` 데코레이터 적용

- **파일**: `contextifier/handlers/base.py` L223
- **현상**: 주석으로만 override 금지 명시. Python 3.12 환경이므로 `typing.final` 적용 가능.
- **수정 방향**: `from typing import final` + `@final` 데코레이터 적용.
  → mypy/pyright 등 타입 체커가 서브클래스 override 시 경고 발생.

### P4-4. 핸들러 Unregister API 추가

- **파일**: `contextifier/handlers/registry.py`
- **현상**: `register()`, `get_handler()`, `is_supported()` 존재하나 `unregister()` 없음.
- **필요 시나리오**: 테스트 Mock 교체, 보안/정책상 특정 포맷 비활성화 등.
- **수정 방향**:
  ```python
  def unregister(self, extension: str) -> bool:
      """지정 확장자의 핸들러 등록 해제. 성공 시 True 반환."""
      return self._handlers.pop(extension, None) is not None
  ```

### P4-5. 의존성 정리 — Optional Extras 분리

- **파일**: `pyproject.toml`
- **현상**: 35+ 패키지가 모두 필수 의존성. PDF만 사용하려는 사용자도 `langchain`, `pyhwp`, `pytesseract` 등 전체 설치 필요.
- **수정 방향**:
  ```toml
  [project.optional-dependencies]
  pdf = ["pymupdf", "pdfplumber", "pdfminer.six", "pdf2image"]
  docx = ["python-docx"]
  pptx = ["python-pptx"]
  excel = ["openpyxl", "xlrd"]
  hwp = ["pyhwp", "olefile"]
  ocr = ["pytesseract", "pi-heif"]
  langchain = ["langchain", "langchain-core", "langchain-community", ...]
  all = ["contextifier[pdf,docx,pptx,excel,hwp,ocr,langchain]"]
  ```

### P4-6. Postprocessor 경고 전달

- **파일**: `contextifier/pipeline/postprocessor.py` L119, L126
- **현상**: docstring에 "Warning comments (if any warnings from extraction)" 언급이 있으나
  실제 구현에서 `ExtractionResult.warnings`가 무시됨.
- **수정 방향**: warnings를 최종 텍스트 끝에 주석으로 포함하거나, `ExtractionResult`에 유지하여
  사용자가 `.warnings`로 접근 가능하도록 함.

### P4-7. `TagService.remove_all_structural_markers()` 범위 확대

- **파일**: `contextifier/services/tag_service.py` L147
- **현상**: 이름과 달리 page/slide/sheet 마커만 제거. image, chart, metadata 태그는 그대로 남음.
- **수정 방향**: 메서드 이름을 `remove_page_slide_sheet_markers()`로 변경하거나, 실제로 모든 구조 마커를 제거하도록 구현 확장.

---

## Phase 5 — 장기 로드맵

> 생태계 확장 및 현대적 아키텍처 전환.

### P5-1. ✅ Async 지원

- `AsyncDocumentProcessor` 래퍼 클래스.
- `asyncio.to_thread()` 기반으로 기존 동기 파이프라인 활용.
- 배치 처리: `extract_batch_async(file_paths, max_concurrent=4)`.

### P5-2. ✅ 플러그인 시스템 (entry_points 기반)

- `importlib.metadata.entry_points(group="contextifier.handlers")`로 외부 핸들러 자동 발견.
- 서드파티 패키지가 `pyproject.toml`에 entry point 선언만으로 핸들러 등록 가능.

### P5-3. ✅ LibreOffice 변환 레이어

- `.doc`, `.ppt`, `.hwp`, `.xls` 등 레거시 바이너리 포맷 → OOXML 변환 공통 레이어.
- `subprocess.run(["libreoffice", "--headless", "--convert-to", format, ...])`.
- 변환 후 해당 OOXML 핸들러에 위임 → 레거시 포맷 파서 복잡도 대폭 감소.

### P5-4. ✅ 콘텐츠 해시 기반 캐싱 레이어

- `CachedDocumentProcessor` — 파일 해시 + config 해시로 결과 캐싱.
- 동일 파일 반복 처리 시 파싱 비용 제거.
- 백엔드: in-memory dict / Redis / 디스크 캐시.

### P5-5. ✅ 테스트 인프라 구축

- **현황**: 테스트 코드 0개 (가장 큰 구조적 리스크).
- **필요 구조**:
  ```
  tests/
  ├── conftest.py              — 공통 fixtures (MockService, 테스트 파일 경로)
  ├── unit/
  │   ├── handlers/
  │   │   ├── test_pdf_handler.py
  │   │   ├── test_docx_handler.py
  │   │   ├── test_xlsx_handler.py
  │   │   ├── test_pptx_handler.py
  │   │   └── test_text_handler.py
  │   ├── services/
  │   │   ├── test_image_service.py
  │   │   ├── test_tag_service.py
  │   │   └── test_table_service.py
  │   ├── chunking/
  │   │   └── test_chunker.py
  │   └── test_config.py
  ├── integration/
  │   ├── fixtures/            — 각 포맷 샘플 파일
  │   └── test_document_processor.py
  └── regression/
      └── test_bug_fixes.py    — P0 버그 회귀 방지 테스트
  ```
- **Phase 0 수정과 함께 최소한 회귀 테스트(regression tests) 작성 필수**.

### P5-6. ✅ 위임 실패 시 원본 포맷 fallback

- **파일**: `contextifier/handlers/base.py` `_delegate_to()`
- **현상**: DOCHandler가 ZIP 매직바이트 감지하여 DocxHandler에 위임했으나 손상된 ZIP이면
  DocxHandler가 실패 → 원래 OLE2 DOC 파이프라인으로 재시도 없이 예외 전파.
- **수정 방향**: `_check_delegation()`에서 위임 실패 시 자체 파이프라인으로 fallback하는 옵션.

### P5-7. ✅ 공개 API 확대

- **파일**: `contextifier/__init__.py`
- **현상**: `DocumentProcessor`와 `__version__`만 export. `ProcessingConfig`, `TextChunker`,
  `ExtractionResult` 등은 서브모듈에서 직접 import 필요.
- **수정 방향**: 주요 public 타입을 `__init__.py`에서 re-export.

---

## 참고: 핸들러 품질 현황 매트릭스

| 핸들러 | 텍스트 | 표 | 이미지 | 차트 | 헤더/푸터 | 메타데이터 | 위임 | 품질 |
|--------|:------:|:---:|:------:|:----:|:---------:|:----------:|:----:|:----:|
| PDF (plus) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ⭐⭐⭐⭐ |
| PDF (default) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | — | ⭐⭐ |
| DOCX | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | — | ⭐⭐⭐ |
| DOC | ✅ | △ | △ | ❌ | ❌ | △ | DOCX,RTF | ⭐⭐ |
| PPTX | ✅ | ✅ | ✅ | ✅ | △ | ✅ | — | ⭐⭐⭐⭐ |
| PPT | ✅ | ❌ | ❌ | ❌ | ❌ | △ | PPTX | ⭐⭐ |
| XLSX | ✅ | ✅ | ⚠️ | △ | — | ✅ | — | ⭐⭐⭐ |
| XLS | ✅ | ✅ | ❌ | ❌ | — | △ | XLSX | ⭐⭐ |
| CSV | ✅ | ✅ | — | — | — | △ | — | ⭐⭐⭐ |
| TSV | ✅ | ✅ | — | — | — | △ | — | ⭐⭐⭐ |
| HWP | ✅ | △ | △ | ❌ | ❌ | △ | HWPX | ⭐⭐ |
| HWPX | ✅ | △ | △ | ❌ | ❌ | △ | — | ⭐⭐ |
| RTF | ✅ | ✅ | ✅ | ❌ | ❌ | △ | — | ⭐⭐⭐ |
| Text | ✅ | ❌ | — | — | — | ❌ | — | ⭐⭐ |
| Image | △ | — | ✅ | — | — | ✅ | — | ⭐⭐⭐ |
| **HTML** | **❌** | **❌** | — | — | — | **❌** | **미구현** | ⭐ |

> ✅ 완전 지원 · △ 부분 지원 · ❌ 미지원 · ⚠️ 버그 있음 · — 해당 없음

---

## 요약: Phase별 작업 수

| Phase | 범위 | 항목 수 | 핵심 |
|-------|------|:-------:|------|
| **Phase 0** | 즉시 버그 수정 | 5 | 데이터 손실, API 파손, 사일런트 오류 |
| **Phase 1** | 단기 안정화 | 8 | OOM 방지, 에러 처리, 스레드 안전성 |
| **Phase 2** | 기능 완성 | 7 | HTML 핸들러, PDF 암호화, DOCX 헤더/푸터 |
| **Phase 3** | 구조 개선 | 7 | 코드 중복 제거, 타입 안전성, 일관성 |
| **Phase 4** | 아키텍처 고도화 | 7 | 메모리 효율, 의존성 정리, API 개선 |
| **Phase 5** | 장기 로드맵 | 7 | Async, 플러그인, 캐싱, 테스트 |
| **합계** | | **41** | |
