# Contextifier v0.2.5 → v0.3.0 고도화 TODO

> 작성일: 2025-04-01
> 기반: 01~05 분석 리포트 종합
> 총 항목: 46개 (Phase 0~6)
> 전제: Phase 0-5 (41항목) 모두 완료된 상태에서의 차기 고도화 계획

---

## Phase 0: 즉시 수정 (보안 & 데이터 무결성)

> 심각한 보안 취약점과 데이터 손실 문제. 즉시 수정 필요.

### P0-1. TableService.format_as_html() HTML 이스케이프 추가
- **파일**: `contextifier/services/table_service.py`
- **이슈**: `_clean_cell()` 또는 `format_as_html()`에서 셀 내용을 HTML 이스케이프하지 않음
- **위험**: XSS / HTML 인젝션 (사용자 입력 포함 CSV, DOCX 등)
- **수정**: `html.escape(content, quote=False)` 추가 (format_as_html_simple과 동일하게)
- **근거**: 03_service_pipeline_analysis.md > TBS1
- **검증**: 기존 TableService 테스트 + HTML 특수문자 포함 테스트 추가

### P0-2. HTML 핸들러 base64 이미지 크기 제한
- **파일**: `contextifier/handlers/html/content_extractor.py`
- **이슈**: base64 인코딩 이미지 디코딩 시 크기 제한 없음 → 메모리 DoS
- **수정**: `MAX_IMAGE_DECODE_SIZE` 상수 추가 (기본 50MB), 초과 시 스킵 + 경고
- **근거**: 02_handler_quality_analysis.md > H5, 04_performance_security_edge_cases.md > R2

### P0-3. 저장 경로 순회(Path Traversal) 방어
- **파일**: `contextifier/services/storage/local.py`, `contextifier/services/image_service.py`
- **이슈**: 이미지 파일명에 `../` 포함 시 base 디렉토리 밖 쓰기 가능
- **수정**: `os.path.commonpath()` 검증 추가 — 저장 경로가 반드시 base_directory 하위인지 확인
- **근거**: 04_performance_security_edge_cases.md > R3

### P0-4. ZIP Bomb 방어
- **파일**: DOCX/PPTX/XLSX/HWPX 각 converter.py
- **이슈**: ZIP 기반 포맷에서 압축 해제 전 크기 검증 없음
- **수정**: 해제 후 크기 임계값(기본 1GB) 체크, 초과 시 ConversionError
- **근거**: 04_performance_security_edge_cases.md > R4

---

## Phase 1: 치명적 기능 갭 해소

> 실제 사용 환경에서 데이터 손실을 유발하는 핸들러 한계 해결.

### P1-1. PDF Default: 스캔 문서 이미지 태그 삽입 (OCR 연계)
- **파일**: `contextifier/handlers/pdf_default/content_extractor.py`
- **이슈**: preprocessor에서 `needs_ocr=True` 감지 후 content_extractor에서 무시 → 스캔 PDF 텍스트 0%
- **설계 원칙**: OCR은 핸들러 외부(DocumentProcessor)의 후처리 경로에 존재.
  핸들러 내부에 OCRProcessor를 주입하면 서비스 주입 구조를 훼손하므로,
  기존 아키텍처(이미지 태그 → 후처리 OCR 치환)를 그대로 활용한다.
- **수정**:
  1. `needs_ocr=True`일 때 페이지를 이미지로 렌더링 (`page.get_pixmap()`)
  2. 렌더링된 이미지를 ImageService로 저장 → `[Image: scan_page_N.png]` 태그 삽입
  3. 사용자가 `ocr_processing=True`로 호출하면 DocumentProcessor 후처리에서 자동 OCR 치환
  4. OCR 엔진 미설정 시: 이미지 태그는 그대로 유지 + 경고 로그
     (`"OCR requested but no engine configured. Skipping."` — 기존 로직)
- **영향받지 않는 것**: BaseContentExtractor 서비스 주입 구조, OCRProcessor 위치, 파이프라인 계층
- **근거**: 02_handler_quality_analysis.md > C1
- **영향**: 스캔 PDF에서 이미지 태그 생성 → `ocr_processing=True` 시 텍스트 복구 가능

### P1-2. DOC 핸들러: OLE2 네이티브 파싱 고도화
- **파일**: `contextifier/handlers/doc/content_extractor.py`, `contextifier/handlers/doc/preprocessor.py`
- **이슈**: 휴리스틱 UTF-16LE 바이트 스캐닝만 → 테이블/복잡 구조 미지원
- **수정**:
  1. FIB(File Information Block) 파싱 → Piece Table 기반 정확한 텍스트 추출
  2. Table Stream(0Table/1Table) 구조 파싱 → `extract_tables()` 구현
  3. OLE Data 스트림 내 임베디드 객체 추출 강화
- **참조 규격**: MS-DOC Binary File Format (.doc) Structure Specification
- **근거**: 02_handler_quality_analysis.md > C2

### P1-3. PPT 핸들러: OLE2 레코드 기반 테이블/차트 추출 구현
- **파일**: `contextifier/handlers/ppt/content_extractor.py`
- **이슈**: `extract_tables()`, `extract_charts()` 모두 `return []` 스턱
- **수정**:
  1. Table record (recType 0x0F00~0x0F0F) 파싱 → 테이블 셀/행/열 복원
  2. Chart Drawing Group 레코드 파싱 → 차트 데이터 추출
  3. 기존 Pictures 스트림 파싱(이미 구현) 패턴을 테이블/차트로 확장
- **참조 규격**: MS-PPT Binary File Format Specification
- **근거**: 02_handler_quality_analysis.md > C3

### P1-4. XLS 핸들러: OLE2 기반 이미지/차트 추출 구현
- **파일**: `contextifier/handlers/xls/content_extractor.py`, `contextifier/handlers/xls/converter.py`
- **이슈**: xlrd가 이미지/차트 API 미제공
- **수정**:
  1. olefile로 OLE 스트림 접근 → 임베디드 이미지 시그니처 스캔 (DOC 핸들러 패턴 재사용)
  2. BIFF 레코드(CHART 타입 0x1002~) 파싱 → 차트 데이터 추출
  3. xlrd의 텍스트/테이블 기능은 그대로 유지, OLE 레벨에서 미디어만 보완
- **참조 규격**: BIFF (Binary Interchange File Format) Specification
- **근거**: 02_handler_quality_analysis.md > C4

### P1-5. libreoffice.py 모듈 제거
- **파일**: `contextifier/services/libreoffice.py` (133 LOC 삭제)
- **이슈**: 외부 도구 의존 모듈이 불필요하게 존재
- **수정**: 파일 삭제 + 다른 모듈에서의 import 제거
- **근거**: 모든 레거시 포맷을 네이티브 바이너리 파싱으로 처리하는 원칙

### P1-6. 위임 깊이 제한 추가
- **파일**: `contextifier/handlers/base.py`
- **이슈**: 위임 체인 무한 루프 이론적 가능 (DOC→RTF→...→DOC)
- **수정**: `_delegate_to()`에 `_delegation_depth` 카운터 추가, 최대 3레벨 제한
- **근거**: 04_performance_security_edge_cases.md > R7

---

## Phase 2: 핸들러 품질 향상

> 부분 구현된 핸들러의 기능 보완 및 검증.

### P2-1. RTF 테이블 병합 플래그 검증 및 수정
- **파일**: `contextifier/handlers/rtf/_table_parser.py`
- **이슈**: `\clmgf`, `\clmrg`, `\clvmgf`, `\clvmrg` 구현 미검증
- **수정**: 테스트 파일 기반 병합 셀 동작 검증, 누락 시 구현
- **근거**: 02_handler_quality_analysis.md > H1
- **검증**: 병합 셀 포함 RTF 파일로 통합 테스트

### P2-2. HWPX 차트 추출 조사 및 구현
- **파일**: `contextifier/handlers/hwpx/content_extractor.py`
- **이슈**: HWPX 차트 XML 구조 미조사, 차트 데이터 손실 가능
- **수정**: HWPX 스펙에서 차트 요소 확인 → 구현 또는 "미지원" 명시적 문서화
- **근거**: 02_handler_quality_analysis.md > H2

### P2-3. Image 핸들러 OCR 통합 명확화
- **파일**: `contextifier/handlers/image/content_extractor.py`
- **이슈**: OCR 텍스트 추출 경로 불명확
- **수정**: pytesseract 기반 Tesseract 엔진 구현 + Image 핸들러 연동
- **근거**: 02_handler_quality_analysis.md > H6

### P2-4. Tesseract OCR 엔진 구현
- **파일**: `contextifier/ocr/engines/tesseract_engine.py` (신규)
- **이슈**: pytesseract 의존성은 있지만 실제 엔진 구현 없음
- **수정**: `BaseOCREngine` 상속 → `pytesseract.image_to_string()` 래핑
- **근거**: 03_service_pipeline_analysis.md > OCR5

### P2-5. PPTX 그룹 셰이프 내 차트 추출
- **파일**: `contextifier/handlers/pptx/content_extractor.py`
- **이슈**: 그룹 셰이프 내부의 차트가 누락될 수 있음
- **수정**: 재귀 셰이프 순회 시 차트 추출 로직 확인/보완
- **근거**: 02_handler_quality_analysis.md > M4

### P2-6. XLSX 차트 추출 확인 및 보완
- **파일**: `contextifier/handlers/xlsx/content_extractor.py`
- **이슈**: 차트 추출 구현 여부 불명확
- **수정**: openpyxl 차트 API 확인 → 구현 또는 안내 문서화
- **근거**: 02_handler_quality_analysis.md > xlsx 차트 ★2

### P2-7. OCR 프롬프트 언어 설정화
- **파일**: `contextifier/ocr/base.py`
- **이슈**: DEFAULT_OCR_PROMPT가 한국어 출력 하드코딩
- **수정**: `OCRConfig`에 `prompt_language` 추가, 언어별 프롬프트 매핑
- **근거**: 03_service_pipeline_analysis.md > OCR1

### P2-8. 하드코딩된 임계값 설정화
- **파일**: 해당 핸들러 각각
- **이슈**:
  - pdf_default: `TABLE_SIZE = 50`
  - pptx: `MAX_GROUP_DEPTH = 20`
  - csv: 구분자 후보 목록
  - doc: `MIN_TEXT_FRAGMENT_LENGTH = 20`
- **수정**: 각각 `format_options`에서 읽도록 변경, 기본값 유지
- **근거**: 02_handler_quality_analysis.md > G1

---

## Phase 3: 테스트 인프라 확장

> 현재 3% 커버리지 → 목표 60%+

### P3-1. ImageService 단위 테스트
- **파일**: `tests/unit/services/test_image_service.py` (신규)
- **범위**: save, save_and_tag, extract_and_deduplicate, clear_state, 네이밍 전략
- **항목**: 최소 15개 테스트

### P3-2. ChartService 단위 테스트
- **파일**: `tests/unit/services/test_chart_service.py` (신규)
- **범위**: format_chart, format_chart_fallback, OOXML 타입 매핑
- **항목**: 최소 10개 테스트

### P3-3. MetadataService 단위 테스트
- **파일**: `tests/unit/services/test_metadata_service.py` (신규)
- **범위**: format_metadata, 이중 언어, 날짜 포매팅, 빈 메타데이터
- **항목**: 최소 8개 테스트

### P3-4. CachedDocumentProcessor 단위 테스트
- **파일**: `tests/unit/test_cached_processor.py` (신규)
- **범위**: MemoryCacheBackend, DiskCacheBackend, 캐시 히트/미스, 설정 변경 무효화
- **항목**: 최소 12개 테스트

### P3-5. AsyncDocumentProcessor 단위 테스트
- **파일**: `tests/unit/test_async_processor.py` (신규)
- **범위**: extract_text, extract_batch, 동시성 제한
- **항목**: 최소 8개 테스트

### P3-6. 위임 경로 테스트
- **파일**: `tests/unit/handlers/test_delegation.py` (신규)
- **범위**: DOC→RTF, PPT→PPTX, XLS→XLSX, HWP→HWPX 시그니처 기반 위임 경로
- **항목**: 최소 10개 테스트

### P3-7. 핸들러 통합 테스트 프레임워크
- **파일**: `tests/integration/` 디렉토리 구조 + conftest.py
- **범위**: 각 핸들러별 실제 샘플 파일 처리 검증
- **의존**: 테스트용 샘플 파일 수집/생성

### P3-8. 보안 테스트
- **파일**: `tests/unit/test_security.py` (신규)
- **범위**: HTML 이스케이프, 경로 순회, ZIP bomb 방어, base64 크기 제한
- **항목**: 최소 10개 테스트

---

## Phase 4: 성능 & 안정성

> 대형 파일 처리, 메모리 효율, 동시성 개선.

### P4-1. 대형 CSV 스트리밍 처리
- **파일**: `contextifier/handlers/csv/content_extractor.py`
- **이슈**: 전체 파일 인메모리 로딩
- **수정**: 청크 단위 읽기 (`csv.reader` + 행 제한), `format_options["csv"]["max_rows"]` 옵션
- **근거**: 04_performance_security_edge_cases.md

### P4-2. XLSX read_only 모드 옵션
- **파일**: `contextifier/handlers/xlsx/converter.py`
- **이슈**: openpyxl 전체 워크북 로딩 → 대형 파일 OOM
- **수정**: `format_options["xlsx"]["read_only"]` 옵션, read_only=True 시 수식 무시
- **근거**: 04_performance_security_edge_cases.md

### P4-3. OCR 병렬 처리
- **파일**: `contextifier/ocr/processor.py`
- **이슈**: 이미지 순차 처리 → 10장 = 10× 지연
- **수정**: `asyncio.gather()` 또는 `ThreadPoolExecutor`로 병렬 OCR 호출
- **근거**: 03_service_pipeline_analysis.md > OCR2

### P4-4. ThreadPoolExecutor 재사용
- **파일**: `contextifier/handlers/base.py`
- **이슈**: `_process_with_timeout()`에서 매 호출 TPE 생성
- **수정**: 클래스 레벨 또는 인스턴스 레벨 TPE 재사용
- **근거**: 04_performance_security_edge_cases.md > R6

### P4-5. CachedDocumentProcessor 확장
- **파일**: `contextifier/cached_processor.py`
- **이슈**: `extract_text()`만 캐싱
- **수정**: `process()`, `extract_chunks()`도 캐싱 지원
- **근거**: 05_api_dx_ecosystem.md

### P4-6. MemoryCacheBackend LRU 전환
- **파일**: `contextifier/cached_processor.py`
- **이슈**: FIFO 방출 전략 → 빈번 접근 항목도 방출
- **수정**: `collections.OrderedDict` 기반 LRU 또는 `functools.lru_cache` 활용
- **근거**: 이전 분석 세션

### P4-7. ImageService 크기 제한 추가
- **파일**: `contextifier/services/image_service.py`, `contextifier/config.py`
- **이슈**: 대형 이미지 무제한 저장
- **수정**: `ImageConfig`에 `max_file_size_mb` 추가, 초과 시 경고 + 스킵
- **근거**: 03_service_pipeline_analysis.md > IS1

---

## Phase 5: 문서화 & DX

> 개발자 경험 향상, 문서 완성.

### P5-1. 핸들러 기능 비교표 문서
- **파일**: `README.md` 또는 `docs/handler_comparison.md`
- **내용**: 핸들러별 텍스트/테이블/이미지/차트/메타데이터 지원 매트릭스

### P5-2. 설정 옵션 레퍼런스 문서
- **파일**: `docs/configuration.md` (신규)
- **내용**: 모든 Config 클래스의 모든 옵션, 기본값, 사용 예시

### P5-3. 에러 코드 레퍼런스 문서
- **파일**: `docs/error_codes.md` (신규)
- **내용**: 모든 E_* 에러 코드, 원인, 해결 방법

### P5-4. OCR 설정 가이드
- **파일**: `docs/ocr_guide.md` (신규)
- **내용**: OCR 엔진별 설정 방법, Tesseract 설치, 클라우드 API 키 설정

### P5-5. 플러그인 개발 가이드
- **파일**: `docs/plugin_development.md` (신규)
- **내용**: BaseHandler 상속, entry_points 설정, 테스트 방법

### P5-6. CHANGELOG v0.3.0 업데이트
- **파일**: `CHANGELOG.md`
- **내용**: Phase 0-6 전체 변경 사항 기록

---

## Phase 6: 에코시스템 & 미래

> 외부 프레임워크 통합, CI/CD, 로드맵.

### P6-1. LangChain Document Loader 구현
- **파일**: `contextifier/integrations/langchain_loader.py` (신규)
- **내용**: `BaseLoader` 상속, `load()` → `List[Document]` 반환
- **근거**: 05_api_dx_ecosystem.md > 7.1

### P6-2. CI/CD 파이프라인 구축
- **파일**: `.github/workflows/ci.yml` (신규)
- **내용**: lint (ruff), test (pytest), type-check (mypy), publish (twine)

### P6-3. Docker 이미지 정의
- **파일**: `Dockerfile` (신규)
- **내용**: Python 3.12 + Tesseract + Poppler (LibreOffice 불필요 — 네이티브 파싱)

### P6-4. 비밀번호 보호 파일 지원 조사
- **범위**: DOCX/PPTX/XLSX 비밀번호 처리
- **방법**: msoffcrypto-tool 라이브러리 평가 + 통합 설계
- **근거**: 04_performance_security_edge_cases.md > R9

### P6-5. pymupdf AGPL 라이선스 검토
- **범위**: pymupdf(AGPL-3.0)과 contextifier(Apache-2.0) 라이선스 호환성
- **방법**: 법적 검토 + 필요 시 pymupdf 대체 (pymupdf4llm, pdfminer.six only)
- **근거**: 05_api_dx_ecosystem.md > 5.4

### P6-6. CSV 구분자 신뢰도 점수 도입
- **파일**: `contextifier/handlers/csv/preprocessor.py`
- **이슈**: csv.Sniffer 결과에 신뢰도 없음
- **수정**: 다중 구분자 시도 + 행 일관성 점수화 + 최저 신뢰도 설정
- **근거**: 02_handler_quality_analysis.md > M1

### P6-7. 인코딩 감지 설정 통합
- **파일**: `contextifier/config.py`
- **이슈**: 핸들러마다 인코딩 감지 방식 상이
- **수정**: `EncodingConfig` 추가 (min_confidence, fallback_encodings, force_encoding)
- **근거**: 04_performance_security_edge_cases.md > 4절

---

## 실행 우선순위 매트릭스

### 긴급도 × 영향도

```
                   높은 영향                         낮은 영향
         ┌────────────────────────┬────────────────────────┐
  긴급   │ P0-1 (HTML 이스케이프)  │ P0-3 (경로 순회)       │
         │ P0-2 (base64 크기)     │ P0-4 (ZIP bomb)        │
         │ P1-1 (PDF OCR 폴백)    │ P1-6 (위임 깊이)       │
         ├────────────────────────┼────────────────────────┤
  중요   │ P1-2 (DOC FIB 파싱)     │ P2-7 (OCR 프롬프트)    │
         │ P1-3 (PPT 레코드 파싱)  │ P2-8 (하드코딩 설정화) │
         │ P1-4 (XLS OLE 보완)    │ P4-4 (TPE 재사용)      │
         │ P1-5 (libreoffice 제거) │ P4-6 (LRU 캐시)        │
         │ P2-4 (Tesseract 엔진)  │ P4-6 (LRU 캐시)        │
         │ P3-1~8 (테스트)        │                         │
         ├────────────────────────┼────────────────────────┤
  보통   │ P4-1 (CSV 스트리밍)    │ P5-1~6 (문서화)        │
         │ P4-2 (XLSX read_only)  │ P6-1 (LangChain)       │
         │ P4-3 (OCR 병렬)       │ P6-2 (CI/CD)           │
         │ P2-1 (RTF 병합)       │ P6-3 (Docker)          │
         │ P2-2 (HWPX 차트)      │ P6-6 (CSV 신뢰도)      │
         └────────────────────────┴────────────────────────┘
```

---

## 추정 규모 요약

| Phase | 항목 수 | 주요 내용 |
|-------|---------|----------|
| Phase 0 | 4 | 보안 취약점 즉시 수정 |
| Phase 1 | 6 | 치명적 데이터 손실 해소 + 외부 의존 제거 |
| Phase 2 | 8 | 핸들러 기능 보완 |
| Phase 3 | 8 | 테스트 인프라 확장 |
| Phase 4 | 7 | 성능 & 안정성 |
| Phase 5 | 6 | 문서화 & DX |
| Phase 6 | 7 | 에코시스템 & 미래 |
| **합계** | **46** | |

---

## 버전 릴리스 계획 (안)

| 버전 | Phase | 핵심 테마 |
|------|-------|----------|
| **v0.2.6** | Phase 0 | 보안 패치 |
| **v0.3.0** | Phase 1 + Phase 2 일부 | 레거시 포맷 네이티브 파싱 고도화, OCR 완성, 외부 의존 제거 |
| **v0.3.1** | Phase 2 나머지 + Phase 3 | 핸들러 품질 향상 + 테스트 확충 |
| **v0.4.0** | Phase 4 + Phase 5 | 성능 최적화 + 문서화 완성 |
| **v0.5.0** | Phase 6 | 에코시스템 통합, CI/CD |

---

## 참조 분석 리포트

| 리포트 | 파일 | 핵심 내용 |
|--------|------|----------|
| 아키텍처 & 현재 상태 | `01_architecture_current_state.md` | 모듈 구조, 설계 패턴, 타입, 에러, LOC 통계 |
| 핸들러 품질 분석 | `02_handler_quality_analysis.md` | 17개 핸들러 심층 분석, 기능 매트릭스, 이슈 |
| 서비스 & 파이프라인 | `03_service_pipeline_analysis.md` | 서비스 6개, 파이프라인 5단계, 청킹/OCR |
| 성능 & 보안 & 엣지 케이스 | `04_performance_security_edge_cases.md` | 메모리, 스레드, 보안, 인코딩, 리소스 |
| API & DX & 에코시스템 | `05_api_dx_ecosystem.md` | API 설계, 문서화, 의존성, 플러그인, CI/CD |
