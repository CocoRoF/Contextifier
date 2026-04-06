# Contextifier 고도화 작업 기록

> 시작일: 2025-04-01
> 기반: 06_final_todo.md (46개 항목)
> 베이스라인: v0.2.5, 93 tests passing (0.48s)
> Phase 0 완료 후: 110 tests passing (0.36s)
> Phase 1 완료 후: 172 tests passing (0.62s)
> Phase 2 완료 후: 260 tests passing (1.31s)
> Phase 3 완료 후: 374 tests passing (1.11s)
> Phase 4 완료 후: 423 tests passing (1.68s)
> Phase 5 완료 후: 423 tests passing (1.83s) — 문서화 단계, 테스트 변경 없음
> Phase 6 완료 후: 476 tests passing (2.04s) — 에코시스템 & 미래 대응

---

## Phase 0: 즉시 수정 (보안 & 데이터 무결성) ✅ DONE

### P0-1. TableService.format_as_html() HTML 이스케이프 추가 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/services/table_service.py` — `format_as_html()` 내 `_clean_cell()` 후 `html.escape(content, quote=False)` 추가 + `\n` → `<br>` 변환
- **테스트**: `tests/unit/services/test_table_service.py` — `test_escapes_html_special_chars` 추가
- **검증**: 9 tests passed (기존 8 + 신규 1)

### P0-2. HTML 핸들러 base64 이미지 크기 제한 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/html/preprocessor.py`
  - `_MAX_IMAGE_DECODE_BYTES = 50 * 1024 * 1024` (50MB) 상수 추가
  - base64 문자열 길이로 디코딩 크기 추정 (`len(b64_str) * 3 // 4`)
  - 초과 시 `logger.warning()` + `continue` (해당 이미지 스킵)
- **테스트**: `tests/unit/handlers/test_html_handler.py` — 3개 테스트
  - `test_small_image_is_extracted`: 정상 크기 이미지 추출 확인
  - `test_oversized_image_is_skipped`: 임계값 초과 시 스킵 + 경고 로그
  - `test_mix_of_small_and_oversized`: 혼합 시 정상 이미지만 추출

### P0-3. 저장 경로 순회(Path Traversal) 방어 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/services/storage/local.py` — `LocalStorageBackend.save()` 내
  - `os.path.realpath()` 로 `file_path`와 `base_path` 정규화
  - 정규화된 경로가 `base_path` 하위가 아니면 `StorageError` 발생
  - 기존 `open(file_path)` → `open(resolved_path)` 변경
- **테스트**: `tests/unit/services/test_storage_local.py` — 5개 테스트
  - `test_normal_save_succeeds`: 정상 저장
  - `test_subdirectory_save_succeeds`: 하위 디렉토리 저장
  - `test_traversal_blocked`: `../` 경로 차단
  - `test_traversal_deep_blocked`: `../../` 경로 차단
  - `test_absolute_path_outside_base`: base 외부 절대 경로 차단

### P0-4. ZIP Bomb 방어 ✅
- **상태**: ✅ 완료
- **수정**:
  - `contextifier/pipeline/converter.py` — `check_zip_bomb()` 유틸리티 함수 + `MAX_ZIP_DECOMPRESSED_BYTES = 1GB` 상수 추가
  - `contextifier/handlers/docx/converter.py` — `validate()` 에 `check_zip_bomb()` 통합
  - `contextifier/handlers/pptx/converter.py` — `validate()` 에 `check_zip_bomb()` 통합
  - `contextifier/handlers/xlsx/converter.py` — `validate()` 에 `check_zip_bomb()` 통합
  - `contextifier/handlers/hwpx/converter.py` — `validate()` 에 `check_zip_bomb()` 통합 (기존 magic-only → ZIP 열어서 검증)
- **설계**: `ZipInfo.file_size` 합산으로 사전 검증 (실제 압축해제 없이 메타데이터만 확인)
- **테스트**: `tests/unit/handlers/test_zip_bomb_defense.py` — 8개 테스트
  - 4개 단위 테스트: small/oversized/multi-entry sum/exact limit
  - 4개 통합 테스트: DOCX/PPTX/XLSX/HWPX 각 converter의 `validate()` 거부 확인

---

## Phase 1: 치명적 기능 갭 해소 ✅ DONE

> Phase 0 완료 후: 110 tests → Phase 1 완료 후: 172 tests (0.62s)

### P1-1. PDF Default: 스캔 문서 이미지 태그 삽입 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/pdf_default/content_extractor.py`
  - `extract_text()`: `needs_ocr` 플래그 확인 → `_extract_scan_pages()` 호출
  - `_extract_scan_pages()`: 150 DPI 렌더링 → PNG → ImageService 저장 → `[Image: scan_page_N.png]` 태그
- **테스트**: `tests/unit/handlers/test_pdf_scan_ocr.py` — 3개 테스트

### P1-2. DOC 핸들러: OLE2 네이티브 파싱 고도화 ✅
- **상태**: ✅ 완료
- **수정**:
  - `contextifier/handlers/doc/_fib.py` (신규 295 LOC) — FIB + Piece Table 파서
    - `parse_fib_text()`: FIB 헤더 → fcClx/lcbClx → Clx → PlcPcd → 피스별 텍스트 재구성
    - `_parse_clx()`: Prc 스킵 + Pcdt 파싱
    - `_parse_plc_pcd()`: CP/PCD 배열 파싱 (compressed cp1252 / Unicode UTF-16LE 구분)
    - `detect_tables_from_text()`: `\x07` 셀 마커 기반 테이블 감지
  - `contextifier/handlers/doc/content_extractor.py` — FIB 우선 + 휴리스틱 폴백
    - `extract_text()`: `parse_fib_text()` 시도 → 실패 시 기존 UTF-16LE 스캐닝
    - `extract_tables()`: 피스 테이블 텍스트에서 `\x07` 마커로 테이블 구조 복원 → `TableData` 변환
- **참조**: MS-DOC Binary File Format Specification (FIB → Clx → PlcPcd)
- **테스트**: `tests/unit/handlers/test_doc_fib.py` — 24개 테스트
  - FIB 유효성: 7개 | 피스 파싱: 3개 | Clx 파싱: 4개 | 테이블 감지: 5개 | 텍스트 정리: 5개

### P1-3. PPT 핸들러: 테이블/차트 추출 구현 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/ppt/content_extractor.py`
  - `extract_tables()`: 탭 문자(`\t`) 기반 표형 텍스트 패턴 감지
  - `extract_charts()`: OLE 디렉토리 스캔 → 임베디드 차트 OLE 객체 감지
  - `_detect_tabular_text()` + `_detect_ole_charts()` 헬퍼 함수
- **설계 노트**: PPT 97-2003에는 네이티브 테이블 레코드 없음 (그룹 도형 기반).
  탭 구분 텍스트 패턴 감지가 최선의 접근. 차트는 Microsoft Graph OLE 객체로 저장됨.
- **테스트**: `tests/unit/handlers/test_ppt_tables_charts.py` — 13개 테스트

### P1-4. XLS 핸들러: OLE2 기반 이미지/차트 추출 구현 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/xls/content_extractor.py`
  - `extract_images()`: raw bytes → olefile OLE2 재개봉 → 이미지 스트림 스캔 (DOC 패턴 재사용)
  - `extract_charts()`: xlrd `sheet_type()` API → BIFF 차트 시트(type 2) 감지
  - `_detect_image_format()`: PNG/JPEG/GIF/BMP/TIFF/EMF 시그니처 감지
  - `__init__`에 `image_service` 파라미터 추가, `olefile` 모듈 레벨 임포트
- **설계**: xlrd 텍스트/테이블 유지, OLE 레벨 미디어 보완
- **테스트**: `tests/unit/handlers/test_xls_images_charts.py` — 18개 테스트

### P1-5. libreoffice.py 모듈 제거 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/services/libreoffice.py` 삭제 (168 LOC, import 0건)

### P1-6. 위임 깊이 제한 추가 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/base.py` — `threading.local()` 깊이 카운터, 최대 3레벨
- **테스트**: `tests/unit/handlers/test_delegation.py` — 4개 테스트

---

## Phase 2: 핸들러 품질 향상 ✅ DONE

> Phase 1 완료 후: 172 tests → Phase 2 완료 후: 260 tests (1.31s)

### P2-1. RTF 테이블 병합 플래그 검증 ✅
- **상태**: ✅ 완료 (구현 정상 확인 — 코드 변경 없음, 테스트만 추가)
- **검증**: `contextifier/handlers/rtf/_table_parser.py` (~540 LOC)
  - `_parse_cell_definitions()`: `\clmgf/\clmrg/\clvmgf/\clvmrg` → `_CellDef` 정상 파싱
  - `_build_table_data()`: colspan/rowspan 정확히 계산 확인
  - `_extract_cells_with_merge()`: 수평/수직/복합 병합 모두 정상
- **테스트**: `tests/unit/handlers/test_rtf_table_merge.py` — 20개 테스트
  - 수평 병합 (2셀, 3셀), 수직 병합 (2행, 3행), 복합 2×2 블록
  - 고아 v_merge_cont, 불균등 행, is_real_table, single_column_to_text
  - 실제 RTF 행 문자열 기반 통합 테스트

### P2-2. HWPX 차트 추출 조사 및 문서화 ✅
- **상태**: ✅ 완료 (이미 구현됨 확인 — 독스트링 보강)
- **확인**:
  - `_section.py` → `_process_chart_ref()` → `_parse_ooxml_chart()` 경로로 OOXML 차트 인라인 처리
  - `extract_charts()`는 빈 리스트 반환 (차트가 `extract_text()` 내에서 처리됨)
- **수정**: `contextifier/handlers/hwpx/content_extractor.py` — `extract_charts()` 독스트링 업데이트
- **테스트**: `tests/unit/handlers/test_hwpx_charts.py` — 10개 테스트
  - OOXML 차트 XML 파싱 (기본, 제목 없음, 꺾은선, 다중 시리즈)
  - `_format_chart_simple`, 섹션 내 인라인 차트, `extract_charts()` 빈 반환

### P2-3. Image 핸들러 OCR 통합 명확화 ✅
- **상태**: ✅ 완료 (아키텍처 정상 확인 — 독스트링 보강)
- **확인**: Image 핸들러는 `[Image: ...]` 태그 생성 → `DocumentProcessor`가 `ocr_processing=True`일 때 `OCRProcessor`로 치환
- **수정**: `contextifier/handlers/image/content_extractor.py` — "OCR Integration Architecture" 독스트링 추가
- **테스트**: P2-4 테스트 파일에 통합 (image handler → tag 생성 + fallback 테스트)

### P2-4. Tesseract OCR 엔진 구현 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/ocr/engines/tesseract_engine.py` (신규)
  - `TesseractOCREngine` — `BaseOCREngine` 상속, LLM 미사용
  - `convert_image_to_text()` 직접 오버라이드 → `pytesseract.image_to_string()` 래핑
  - 생성자: `__init__(*, lang="eng", tesseract_cmd=None, prompt=None)`
  - `build_message_content()` → `NotImplementedError` (LLM 기반 아님)
  - `provider` → `"tesseract"`, 성공 시 `[Figure: ...]`, 실패 시 `[Image conversion error: ...]`
  - `pytesseract` 지연 임포트 (런타임에만 의존)
- **수정**: `contextifier/ocr/engines/__init__.py` — `TesseractOCREngine` 추가
- **테스트**: `tests/unit/ocr/test_tesseract_engine.py` — 13개 테스트
  - 엔진 생성, provider, build_message_content raises, repr
  - convert_image (성공/빈문자열/import_error/runtime_error)

### P2-5. PPTX 그룹 셰이프 재귀 추출 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/pptx/content_extractor.py`
  - `extract_tables()` → `_collect_tables(shape, depth=0)` 재귀 헬퍼
  - `extract_images()` → `_collect_images(shape, slide_idx, processed, depth=0)` 재귀 헬퍼
  - `extract_charts()` → `_collect_charts(shape, depth=0)` 재귀 헬퍼
  - 각 헬퍼: `hasattr(shape, "shapes")` 체크 → 그룹 내부 재귀, `_MAX_GROUP_DEPTH` 제한
- **테스트**: `tests/unit/handlers/test_pptx_group_shapes.py` — 8개 테스트
  - 테이블/이미지/차트: 최상위, 그룹 내부, 중첩 그룹, 깊이 제한

### P2-6. XLSX 차트 추출 확인 ✅
- **상태**: ✅ 완료 (이미 구현됨 확인 — 테스트만 추가)
- **확인**: `contextifier/handlers/xlsx/content_extractor.py`
  - `extract_charts()`: `preprocessed.resources["charts_by_sheet"]` 에서 차트 데이터 로드
  - `preprocessor.py`에서 `openpyxl.chart` API로 사전 추출 → 리소스 맵핑
- **테스트**: `tests/unit/handlers/test_xlsx_charts.py` — 7개 테스트
  - 단일/다중 차트, 빈 리소스, 시리즈 데이터, 유효하지 않은 dict, 카테고리

### P2-7. OCR 프롬프트 언어 설정화 ✅
- **상태**: ✅ 완료
- **수정**:
  - `contextifier/ocr/base.py`
    - `_OCR_PROMPT_TEMPLATE`: `{language_rule}` 플레이스홀더 기반 템플릿
    - `OCR_LANGUAGE_RULES`: 언어별 규칙 dict (`ko`, `en`, `ja`)
    - `get_ocr_prompt(language: str = "ko") -> str`: 언어 코드로 프롬프트 생성
    - `DEFAULT_OCR_PROMPT` = `get_ocr_prompt("ko")` (하위 호환성 유지)
  - `contextifier/config.py`
    - `OCRConfig.prompt_language: str = "ko"` 필드 추가
- **테스트**: `tests/unit/ocr/test_ocr_prompt_language.py` — 8개 테스트
  - 한국어 기본, 영어, 일본어, 미지원 언어 폴백, 커스텀 프롬프트 우선
  - OCRConfig.prompt_language, 하위 호환성, DEFAULT_OCR_PROMPT 값

### P2-8. 하드코딩된 임계값 설정화 ✅
- **상태**: ✅ 완료
- **설계**: `BaseContentExtractor.__init__`에 `config: Optional[ProcessingConfig] = None` 파라미터 추가 (하위 호환)
  - 핸들러의 `create_content_extractor()`에서 `config=self._config` 전달
  - 각 추출기가 `self._config.get_format_option()` 으로 임계값 읽기, 기본값 폴백
- **수정**:
  - `contextifier/pipeline/content_extractor.py` — `config` 파라미터 추가
  - `contextifier/handlers/pdf/handler.py` — 양쪽 모드에 `config=self._config` 전달
  - `contextifier/handlers/pdf_default/content_extractor.py`
    - `render_dpi`: `format_options["pdf"]["render_dpi"]` (기본 150)
    - `min_image_size`: `format_options["pdf"]["min_image_size"]` (기본 50)
    - `min_image_area`: `format_options["pdf"]["min_image_area"]` (기본 2500)
  - `contextifier/handlers/pdf_plus/content_extractor.py` — `config` 파라미터 추가
  - `contextifier/handlers/pptx/handler.py` + `content_extractor.py`
    - `max_group_depth`: `format_options["pptx"]["max_group_depth"]` (기본 20)
  - `contextifier/handlers/csv/handler.py` + `preprocessor.py`
    - `delimiter_candidates`: `format_options["csv"]["delimiter_candidates"]` (기본 `[",", "\t", ";", "|"]`)
    - `_detect_delimiter()` 에 `candidates` 파라미터 추가
  - `contextifier/handlers/doc/handler.py` + `content_extractor.py`
    - `min_text_fragment_length`: `format_options["doc"]["min_text_fragment_length"]` (기본 4)
    - 인스턴스 변수 `_min_text_fragment_length`, `_min_unicode_bytes` 로 변환
- **사용 예시**:
  ```python
  config = ProcessingConfig().with_format_option("pdf", render_dpi=200, min_image_size=30)
  config = ProcessingConfig().with_format_option("pptx", max_group_depth=5)
  config = ProcessingConfig().with_format_option("csv", delimiter_candidates=[",", ";", ":"])
  config = ProcessingConfig().with_format_option("doc", min_text_fragment_length=8)
  ```
- **테스트**: `tests/unit/handlers/test_configurable_thresholds.py` — 22개 테스트
  - BaseContentExtractor config 전파 (2개)
  - PDF default: render_dpi, min_image_size, min_image_area, 스캔 DPI 적용, 필터링 (6개)
  - PPTX: 기본값, config 오버라이드, None config (3개)
  - CSV: 기본 후보, 커스텀 감지, 전처리기 저장, handler 전달 (5개)
  - DOC: 기본값, config 오버라이드, None, handler 전달 (4개)
  - PDF handler config 전파: default/plus 모드 (2개)

---

## Phase 3: 테스트 인프라 확장 ✅ DONE

> Phase 2 완료 후: 260 tests → Phase 3 완료 후: 374 tests (1.11s)
> 신규 테스트: 114개 추가

### P3-1. ImageService 단위 테스트 확장 ✅
- **상태**: ✅ 완료
- **수정**: `tests/unit/services/test_image_service.py` — 14 → 24 테스트
- **추가 테스트 영역**:
  - `TestNamingStrategy` (4): hash deterministic, sequential increments/resets, custom override
  - `TestTagBuilding` (2): tag_service used, fallback without tag_service
  - `TestDeduplication` 확장 (+2): config default, skip_false allows duplicates
  - `TestClearState` 확장 (+1): resets processed paths
  - `TestExtractAndDeduplicate` 확장 (+1): different data → different tags
  - `TestBasicSave` 확장 (+2): storage backend called, storage error raises
  - `TestThreadIsolation` 확장 (+1): counter independence

### P3-2. ChartService 단위 테스트 ✅
- **상태**: ✅ 완료
- **신규**: `tests/unit/services/test_chart_service.py` — 20 테스트
- **테스트 영역**:
  - `TestFormatChart` (7): basic bar, empty, raw_content, no type, no title, text table, HTML multi-series
  - `TestFormatChartFallback` (3): type+title, message, default message
  - `TestChartTypeName` (6): 5 parametrized known types + unknown passthrough
  - `TestChartPatterns` (3): has_chart_blocks, no_chart_blocks, find_chart_blocks
  - `TestTagFallback` (1): config tags without tag_service

### P3-3. MetadataService 단위 테스트 ✅
- **상태**: ✅ 완료
- **신규**: `tests/unit/services/test_metadata_service.py` — 14 테스트
- **테스트 영역**:
  - `TestFormatMetadata` (6): full KO, full EN, None, empty, custom fields, field order
  - `TestValueFormatting` (5): datetime, custom date format, integer, whitespace skipped, zero included
  - `TestFormatMetadataDict` (3): dict input, ISO datetime, empty dict

### P3-4. CachedDocumentProcessor 단위 테스트 ✅
- **상태**: ✅ 완료
- **신규**: `tests/unit/test_cached_processor.py` — 18 테스트
- **테스트 영역**:
  - `TestMemoryCacheBackend` (5): get missing, put/get, overwrite, FIFO eviction, eviction order
  - `TestDiskCacheBackend` (5): put/get, missing key, corrupt JSON, file persists, unicode value
  - `TestCachedDocumentProcessor` (8): cache hit, different files, custom backend, is_supported, supported_extensions, config property, repr, config change invalidation

### P3-5. AsyncDocumentProcessor 단위 테스트 ✅
- **상태**: ✅ 완료
- **신규**: `tests/unit/test_async_processor.py` — 9 테스트 (pytest-asyncio 추가 설치)
- **테스트 영역**:
  - `TestExtractText` (2): basic extraction, missing file error
  - `TestProcess` (1): returns ExtractionResult
  - `TestExtractChunks` (1): basic chunking
  - `TestExtractBatch` (3): all success, with failure, empty list
  - `TestUtility` (2): is_supported, config property

### P3-6. 핸들러 위임(Delegation) 경로 테스트 ✅
- **상태**: ✅ 완료
- **수정**: `tests/unit/handlers/test_delegation.py` — 4 → 14 테스트
- **추가 테스트 영역**:
  - `TestDOCDelegation` (4): ZIP→docx, RTF→rtf, HTML→html, OLE2→no delegation
  - `TestPPTDelegation` (2): ZIP→pptx, OLE2→no delegation
  - `TestXLSDelegation` (2): ZIP→xlsx, BIFF→no delegation
  - `TestHWPDelegation` (2): ZIP→hwpx, OLE2→no delegation

### P3-7. 통합 테스트 프레임워크 확장 ✅
- **상태**: ✅ 완료
- **수정**: `tests/integration/test_document_processor.py` — 10 → 23 테스트
- **추가 테스트 영역**:
  - `TestProcessEndToEnd` (2): ExtractionResult 반환, metadata 포함
  - `TestRoundTripConsistency` (2): deterministic extraction, deterministic chunks
  - `TestConfigIntegration` (3): metadata language ko/en, chunk_size affects count
  - `TestMultiFormatSupport` (4): CSV, TSV, HTML, JSON extraction
  - `TestSupportedExtensions` (2): common formats, unknown format

### P3-8. 보안 테스트 통합 ✅
- **상태**: ✅ 완료
- **신규**: `tests/unit/test_security.py` — 17 테스트
- **테스트 영역**:
  - `TestFileSizeLimits` (3): oversized rejected, within limit, zero disables
  - `TestZipBombDefense` (4): small passes, oversized blocked, cumulative checked, default 1GB
  - `TestPathTraversalDefense` (4): dotdot, absolute path, nested dotdot, normal subdir allowed
  - `TestDelegationDepthSecurity` (2): reasonable constant, overflow prevented
  - `TestInputBoundary` (4): null bytes, empty file rejected, binary garbage, nonexistent file

---

## Phase 4: 성능 & 안정성 ✅ DONE

> Phase 3 완료 후: 374 tests → Phase 4 완료 후: 423 tests (1.68s)
> 신규 테스트: 49개 추가

### P4-1. 대형 CSV 스트리밍 처리 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/csv/preprocessor.py`
  - `CsvParsedData`: `truncated: bool = False` 필드 추가
  - `CsvPreprocessor.__init__()`: `max_rows` 매개변수 추가 (기본값 = `MAX_ROWS`)
  - `_parse_csv_content()`: `max_rows` 매개변수 추가, `(rows, truncated)` 튜플 반환
  - `_parse_csv_simple()`: 동일 변경
  - `preprocess()`: `truncated` 플래그를 `CsvParsedData` 및 `properties`에 전파
- **수정**: `contextifier/handlers/csv/handler.py` — `format_options["csv"]["max_rows"]` 읽기
- **수정**: `contextifier/handlers/tsv/handler.py` — `format_options["tsv"]["max_rows"]` 읽기
- **테스트**: `tests/unit/handlers/test_csv_streaming.py` — 15 테스트
  - `TestParseMaxRows` (4): no truncation, truncation at limit, exact limit, default constant
  - `TestParseSimpleMaxRows` (2): truncation, no truncation
  - `TestCsvPreprocessorMaxRows` (4): default, custom, preprocess truncates, preprocess no truncation
  - `TestCsvHandlerMaxRowsOption` (2): handler passes max_rows, handler default
  - `TestTsvHandlerMaxRowsOption` (1): TSV handler passes max_rows
  - `TestCsvParsedDataTruncated` (2): default false, explicit true

### P4-2. XLSX read_only 모드 옵션 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/xlsx/converter.py`
  - `XlsxConverter.__init__()`: `read_only: bool = False` 매개변수 추가
  - `convert()`: `openpyxl.load_workbook(stream, data_only=..., read_only=...)` 전달
- **수정**: `contextifier/handlers/xlsx/handler.py` — `format_options["xlsx"]["read_only"]` 읽기
- **테스트**: `tests/unit/handlers/test_xlsx_handler.py` — 3 테스트 추가
  - `test_read_only_default_false`, `test_read_only_from_format_options`, `test_read_only_passed_to_load_workbook`

### P4-3. OCR 병렬 처리 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/ocr/processor.py`
  - `OCRProcessor.__init__()`: `max_workers: int = 1` 매개변수 추가
  - `process()`: 2-phase 구조 개편 (Phase 1: 병렬 OCR → Phase 2: 순차 교체)
  - `_run_ocr_batch()`: `ThreadPoolExecutor` 기반 병렬 실행 (max_workers > 1일 때)
  - `_ocr_single()`: 개별 이미지 OCR + 진행률 콜백
  - `_replace_tag()`: Windows 경로 역슬래시 regex 버그 수정 (lambda 사용)
- **테스트**: `tests/unit/ocr/test_ocr_parallel.py` — 9 테스트
  - `TestOCRProcessorMaxWorkers` (3): default 1, custom, minimum 1
  - `TestOCRSequentialMode` (2): empty text, no tags
  - `TestOCRParallelMode` (3): same result as sequential, progress callbacks, handles failures
  - `TestOCRBatchRun` (1): sequential ordered

### P4-4. ThreadPoolExecutor 재사용 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/base.py`
  - `_timeout_executor`: `ClassVar` 공유 `ThreadPoolExecutor` (max_workers=4)
  - `_get_timeout_executor()`: 더블 체크 락킹 패턴으로 지연 생성
  - `_shutdown_timeout_executor()`: `atexit` 등록, 프로세스 종료 시 정리
  - `_process_with_timeout()`: 공유 executor 사용 (매번 생성 → 재사용)
- **테스트**: `tests/unit/handlers/test_tpe_reuse.py` — 6 테스트
  - `TestSharedTimeoutExecutor` (6): lazy init, creates executor, same instance, shutdown clears, recreated after shutdown, handler uses shared

### P4-5. CachedDocumentProcessor 확장 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/cached_processor.py`
  - `process()`: `ExtractionResult` 캐싱 (JSON 직렬화/역직렬화)
  - `extract_chunks()`: `ChunkResult` 캐싱 (JSON 직렬화/역직렬화)
  - `_make_key()`: `suffix` 매개변수 추가 (text/process/chunks 구분)
  - 직렬화 헬퍼: `_serialize/deserialize_extraction_result`, `_serialize/deserialize_chunk_result`
  - `_table_to_dict/_dict_to_table`, `_chart_to_dict/_dict_to_chart` 보조 함수
- **테스트**: `tests/unit/test_cached_processor.py` — 8 테스트 추가 (18 → 28)
  - `TestCachedProcess` (3): returns ExtractionResult, cache hit, custom backend
  - `TestCachedExtractChunks` (2): returns ChunkResult, cache hit
  - `TestSerializationRoundTrip` (3): ExtractionResult round-trip, ChunkResult round-trip, no metadata

### P4-6. MemoryCacheBackend LRU 전환 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/cached_processor.py`
  - `MemoryCacheBackend._store`: `dict` → `collections.OrderedDict`
  - `get()`: `move_to_end(key)` 호출로 LRU 위치 갱신
  - `put()`: 기존 키 갱신 시 `move_to_end()`, 초과 시 `popitem(last=False)` (LRU 제거)
- **테스트**: `tests/unit/test_cached_processor.py` — 2 테스트 추가
  - `test_lru_eviction`: 접근한 항목 보존 확인
  - `test_lru_put_update_refreshes`: 덮어쓰기 시 LRU 위치 갱신 확인

### P4-7. ImageService 크기 제한 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/config.py`
  - `ImageConfig`: `max_file_size_mb: Optional[float] = None` 필드 추가
- **수정**: `contextifier/services/image_service.py`
  - `save()`: 크기 제한 체크 로직 추가 (초과 시 경고 + `None` 반환)
- **테스트**: `tests/unit/services/test_image_service.py` — 6 테스트 추가 (24 → 30)
  - `TestImageSizeLimit` (6): default no limit, within limit, exceeds limit, exact limit, None limit, save_and_tag respects

---

## Phase 5: 문서화 & DX ✅ DONE

> Phase 4 완료 후: 423 tests → Phase 5 완료 후: 423 tests (변경 없음)
> 신규 문서: 5개 생성, README.md·CHANGELOG.md 업데이트

### P5-1. 핸들러 기능 비교표 문서 ✅
- **상태**: ✅ 완료
- **생성**: `docs/handler_comparison.md`
  - 15개 핸들러의 텍스트/테이블/이미지/차트/메타데이터 지원 매트릭스
  - 추출 아키텍처 비교 (분리형/인라인형/하이브리드)
  - 위임(Delegation) 관계도
  - `format_options` 지원 옵션 전체 목록

### P5-2. 설정 옵션 레퍼런스 문서 ✅
- **상태**: ✅ 완료
- **생성**: `docs/configuration.md`
  - 8개 Config 클래스 전체 필드·타입·기본값·설명
  - ProcessingConfig fluent API (`with_*()` 메서드)
  - `format_options` 핸들러별 세부 설정 (PDF, PPTX, DOC, CSV, TSV, XLSX)
  - 직렬화/역직렬화 (`to_dict()`, `from_dict()`)
  - 설정 조합 예시 (고성능, AI/LLM, 최소 출력)

### P5-3. 에러 코드 레퍼런스 문서 ✅
- **상태**: ✅ 완료
- **생성**: `docs/error_codes.md`
  - 19개 예외 클래스의 에러 코드·원인·해결법
  - 예외 계층 트리 다이어그램
  - 에러 처리 패턴 (기본, 컨텍스트 추가, 파이프라인 에러)

### P5-4. OCR 설정 가이드 ✅
- **상태**: ✅ 완료
- **생성**: `docs/ocr_guide.md`
  - 6개 엔진별 설치·설정 예시 (OpenAI, Anthropic, Gemini, Bedrock, vLLM, Tesseract)
  - 프롬프트 커스터마이징 (언어 설정, 커스텀 프롬프트)
  - OCR 병렬 처리 (`max_workers`)
  - OCR 동작 원리 (이미지 태그 → OCR → 치환)
  - 트러블슈팅 (Tesseract 미발견, API 인증, Rate Limit)

### P5-5. 플러그인 개발 가이드 ✅
- **상태**: ✅ 완료
- **생성**: `docs/plugin_development.md`
  - 5단계 파이프라인 컴포넌트 구현 가이드 (Converter → Preprocessor → MetadataExtractor → ContentExtractor → Postprocessor)
  - 핸들러 조립 (BaseHandler 상속)
  - 등록 방법 (직접 등록, entry_points)
  - 규칙 및 제약 (하나의 확장자, process() 오버라이드 금지, 균일한 생성자)
  - 테스트 작성 예시

### P5-6. CHANGELOG v0.3.0 업데이트 ✅
- **상태**: ✅ 완료
- **수정**: `CHANGELOG.md`
  - Phase 0 (Security), Phase 1 (Fixed), Phase 2 (Improved), Phase 3 (Added), Phase 4 (Performance), Phase 5 (Documentation) 전체 반영
- **수정**: `README.md`
  - Documentation 테이블에 5개 신규 문서 링크 추가
  - `contextifier_new` → `contextifier` 임포트 경로 7건 수정
  - ARCHITECTURE.md 링크 경로 수정

---

## Phase 6: 에코시스템 & 미래 ✅ DONE

> Phase 5 완료 후: 423 tests → Phase 6 완료 후: 476 tests (2.04s)
> 신규 테스트: 53개 추가 (LangChain 12 + Crypto 10 + Delimiter 14 + Encoding 17)

### P6-1. LangChain Document Loader ✅
- **상태**: ✅ 완료
- **생성**: `contextifier/integrations/__init__.py`, `contextifier/integrations/langchain_loader.py`
  - `ContextifierLoader(BaseLoader)` — LangChain 문서 로더
  - `lazy_load()` — 단일 문서 모드 / 청크 모드 (chunk=True)
  - 메타데이터: source, file_name, file_extension, chunk_index, page_number
  - OCR 통합 지원 (ocr_engine, ocr_processing 매개변수)
- **테스트**: `tests/unit/integrations/test_langchain_loader.py` (12 tests)
  - TestSingleDocument (5), TestChunkedDocument (3), TestWithConfig (2), TestOCRIntegration (2)

### P6-2. CI/CD 파이프라인 ✅
- **상태**: ✅ 완료
- **생성**: `.github/workflows/ci.yml`
  - lint (ruff check + format), test (Python 3.12/3.13 매트릭스)
  - type-check (mypy, continue-on-error), publish (PyPI trusted publishing)

### P6-3. Dockerfile ✅
- **상태**: ✅ 완료
- **생성**: `Dockerfile`
  - Multi-stage 빌드 (builder → runtime)
  - python:3.12-slim + tesseract-ocr + tesseract-ocr-kor + poppler-utils
  - Non-root user (`contextifier`), WORKDIR /data

### P6-4. 암호화 파일 지원 ✅
- **상태**: ✅ 완료
- **생성**: `contextifier/services/crypto_service.py`
  - `is_encrypted(file_data) -> bool` — msoffcrypto로 암호화 여부 확인
  - `decrypt_if_encrypted(file_data, password=None) -> bytes` — 복호화
  - msoffcrypto 미설치 시 graceful fallback (원본 데이터 반환)
- **수정**: `contextifier/document_processor.py`
  - `_create_file_context()`: `password` 매개변수 추가 + `decrypt_if_encrypted()` 호출
  - `extract_text()`, `process()`, `extract_chunks()`: `password` kwarg 전달
- **테스트**: `tests/unit/services/test_crypto_service.py` (10 tests)
  - TestIsEncrypted (4), TestDecryptIfEncrypted (6)
- **의존성**: `msoffcrypto-tool>=6.0.0` (pip install)

### P6-5. PyMuPDF AGPL 라이선스 검토 ✅
- **상태**: ✅ 완료
- **생성**: `docs/license_review.md`
  - AGPL-3.0 vs Apache-2.0 비호환성 분석
  - 영향받는 파일 목록 (pdf/converter.py, pdf_default, pdf_plus)
  - 조치: pymupdf를 optional dependency (`pip install contextifier[pdf]`)로 전환
- **수정**: `pyproject.toml`
  - `pymupdf>=1.24.0`를 `dependencies`에서 `[project.optional-dependencies].pdf`로 이동
  - `all` extra에 `pdf` 포함
- **수정**: `contextifier/handlers/pdf/converter.py`
  - `import fitz` → try/except 가드 + fitz 미설치 시 ConversionError
- **수정**: `contextifier/handlers/pdf_default/content_extractor.py`
  - `import fitz` → try/except 가드

### P6-6. CSV 구분자 신뢰도 점수 ✅
- **상태**: ✅ 완료
- **수정**: `contextifier/handlers/csv/preprocessor.py`
  - `_detect_delimiter()` 반환값: `str` → `tuple[str, float]` (delimiter, confidence)
  - Sniffer 성공: 행 일관성 검증 후 0.7~1.0 신뢰도
  - 수동 스코어링: 완벽 일관성 1.0, 변동 시 0.0~0.7
  - `CsvParsedData`: `delimiter_confidence: float = 1.0` 필드 추가
  - `preprocess()`: `delimiter_confidence`를 `properties`에 포함
- **테스트**: `tests/unit/handlers/test_csv_delimiter_confidence.py` (14 tests)
  - TestDelimiterConfidence (10), TestCsvParsedDataConfidence (2), TestPreprocessorConfidenceOutput (2)

### P6-7. 인코딩 감지 설정 통합 ✅
- **상태**: ✅ 완료
- **생성**: `contextifier/config.py`에 `EncodingConfig` 추가
  - `fallback_encodings`: 인코딩 시도 순서 (기본: utf-8, utf-8-sig, cp949, euc-kr, latin-1, ascii)
  - `force_encoding`: 강제 인코딩 지정 (감지 건너뜀)
  - `min_confidence`: chardet 최소 신뢰도 (향후 통합용, 기본 0.7)
- **수정**: `ProcessingConfig`에 `encoding: EncodingConfig` 필드 + `with_encoding()` 메서드
  - `to_dict()` / `from_dict()` 직렬화 지원
- **수정**: CSV/TSV/Text 핸들러 컨버터
  - `CsvConverter(encoding_config=)`, `TextConverter(encoding_config=)` 지원
  - `force_encoding` 최우선 → `encodings` 명시 → `fallback_encodings` 순
  - CSVHandler, TSVHandler, TextHandler: `create_converter()`에서 `self._config.encoding` 전달
- **테스트**: `tests/unit/test_encoding_config.py` (17 tests)
  - TestEncodingConfigDefaults (4), TestEncodingConfigCustom (3), TestProcessingConfigEncoding (4), TestConverterEncodingConfig (6)
