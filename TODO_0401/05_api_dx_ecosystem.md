# Contextifier v0.2.5 — API 설계, 개발자 경험, 에코시스템 분석

> 분석 일자: 2025-04-01
> 분석 범위: Public API, DX(Developer Experience), 문서화, 테스트, 의존성, 플러그인 시스템, 배포, 에코시스템 통합

---

## 1. Public API 설계 분석

### 1.1 3대 프로세서 API

| 프로세서 | 용도 | 핵심 메서드 |
|---------|------|------------|
| `DocumentProcessor` | 동기 처리 (핵심) | `extract_text()`, `process()`, `extract_chunks()`, `chunk_text()` |
| `AsyncDocumentProcessor` | 비동기 래퍼 | 위와 동일 (async) + `extract_batch()` |
| `CachedDocumentProcessor` | 캐시 래퍼 | `extract_text()` (캐시 적용) |

### 1.2 API 일관성 분석

**✅ 양호한 점**:
- `DocumentProcessor`의 모든 메서드가 `file_path`를 첫 번째 인수로 받음
- `extract_text()`, `process()` 시그니처 통일
- `is_supported()`, `supported_extensions` 속성 제공

**🟡 비일관 영역**:

| 이슈 | 상세 |
|------|------|
| `process()` 반환이 `extract_text()`와 다름 | `ExtractionResult` vs `str` — 예상 가능하지만 초보자 혼동 |
| `extract_chunks()` 반환이 `ChunkResult` | 별도 래퍼 클래스 — `List[str]`이 더 직관적? |
| `chunk_text()` vs `extract_chunks()` | 전자는 텍스트→청크, 후자는 파일→청크 — 명명 혼동 가능 |
| `CachedDocumentProcessor` — `process()` 미캐싱 | `extract_text()`만 캐싱 → 불완전한 래핑 |

### 1.3 ChunkResult API

```python
@dataclass
class ChunkResult:
    chunks: List[str]                           # 텍스트 청크
    chunks_with_metadata: Optional[List[Chunk]]  # 메타데이터 포함 청크
    source_file: Optional[str]

    def __len__(self): ...
    def __getitem__(self, index): ...
    def __iter__(self): ...
    def save_to_md(self, output_dir, ...): ...
```

**✅ 강점**: `__iter__`, `__len__`, `__getitem__` 구현으로 리스트처럼 사용 가능
**🟡 이슈**: `has_metadata` 속성이 외부에서 직관적이지 않음

### 1.4 설정 API

```python
# 제로 설정 (합리적 기본값)
processor = DocumentProcessor()

# 커스텀 설정
from contextifier.config import ProcessingConfig, TagConfig, ImageConfig
config = ProcessingConfig(
    tags=TagConfig(page_prefix="<page>", page_suffix="</page>"),
    images=ImageConfig(directory_path="output/images"),
    chunking=ChunkingConfig(chunk_size=2000),
)
processor = DocumentProcessor(config=config)

# 부분 변경 (builder 패턴)
new_config = config.with_chunking(chunk_size=3000)
```

**✅ 강점**: 제로 설정으로 즉시 사용 가능, `with_*()` 메서드로 불변 변경
**🟡 이슈**: `format_options` dict가 타입 안전하지 않음 — `Dict[str, Dict[str, Any]]`

---

## 2. 개발자 경험 (DX) 분석

### 2.1 Import 편의성

```python
# 최소 임포트
from contextifier import DocumentProcessor

# 설정 포함
from contextifier import DocumentProcessor, ProcessingConfig

# 전체 타입
from contextifier import (
    DocumentProcessor, AsyncDocumentProcessor, CachedDocumentProcessor,
    ProcessingConfig, ChunkingConfig,
    ExtractionResult, FileContext, Chunk, ChunkMetadata,
    TextChunker, ContextifierError, UnsupportedFormatError,
)
```

**✅**: `__init__.py`의 `__all__`이 핵심 타입 13개로 깔끔

### 2.2 에러 메시지 품질

```python
# 좋은 예
"[E_UNSUPPORTED_FORMAT] Unsupported file format: .xyz\n  file_path: /path/to/file.xyz"

# 예외 체인
"[E_HANDLER_EXECUTION] Handler 'PDF Handler' failed: ...\n  Caused by: fitz.FileDataError: ..."
```

**✅**: 에러 코드 + 메시지 + 컨텍스트 + 원인 체인 — 우수한 디버깅 경험

### 2.3 로깅

```python
logger = logging.getLogger("contextifier")
# 핸들러별: logging.getLogger(f"contextifier.handler.{cls_name}")
```

**🟡 이슈**:
- 로그 레벨 제어가 모듈 단위 — 파이프라인 단계별 세분화 불가
- DEBUG 로그가 과도할 수 있음 (Stage 1/5, Stage 2/5... 매 파일마다)
- 구조화 로깅(structured logging) 미지원

### 2.4 타입 힌트 품질

| 영역 | 품질 | 상세 |
|------|------|------|
| Public API | ✅ 우수 | 모든 메서드 완전 타입 힌팅 |
| Config | ✅ 우수 | frozen dataclass로 자동 타입 |
| Types | ✅ 우수 | TypedDict, Enum, dataclass |
| 내부 구현 | 🟡 혼합 | `Any` 사용 (fitz.Document, chart objects 등) |
| 반환 타입 | ✅ | 모두 명시적 |

---

## 3. 문서화 상태 분석

### 3.1 문서 인벤토리

| 문서 | 상태 | 내용 |
|------|------|------|
| `README.md` | ✅ | 프로젝트 개요, 설치, 기본 사용법 |
| `QUICKSTART.md` | ✅ | 빠른 시작 가이드 |
| `CONTRIBUTING.md` | ✅ | 기여 가이드 |
| `CHANGELOG.md` | ✅ | 변경 이력 |
| `LICENSE` | ✅ | Apache 2.0 |
| `ARCHITECTURE.md` | ✅ | 아키텍처 설명 |
| `Process Logic.md` | ✅ | 처리 흐름 설명 |

### 3.2 문서화 갭

| 항목 | 현재 상태 | 필요성 |
|------|----------|--------|
| **API Reference** (자동 생성) | ❌ 없음 | 🟠 높음 |
| **핸들러별 지원 기능 표** | ❌ 없음 | 🟠 높음 |
| **설정 옵션 전체 레퍼런스** | ❌ 없음 | 🟠 높음 |
| **마이그레이션 가이드 (v1→v2)** | ❌ 없음 | 🟡 중간 |
| **OCR 설정 가이드** | ❌ 없음 | 🟡 중간 |
| **플러그인 개발 가이드** | ❌ 없음 | 🟡 중간 |
| **성능 튜닝 가이드** | ❌ 없음 | 🔵 낮음 |
| **트러블슈팅 가이드** | ❌ 없음 | 🔵 낮음 |

### 3.3 코드 내 문서화

**✅ 우수한 점**:
- 모든 모듈에 상세한 모듈 docstring
- `base.py` 509 LOC 중 ~200 LOC가 문서/주석
- Config 클래스에 사용 예시 포함

**🟡 부족한 점**:
- 핸들러별 지원 범위/한계 문서 없음
- 헬퍼 함수 docstring 부족 (특히 `_` prefix 함수)
- 에러 코드 레퍼런스 문서 없음

---

## 4. 테스트 전략 분석

### 4.1 현재 테스트 현황

```
tests/
├── __init__.py
├── conftest.py
├── unit/
│   ├── test_config.py
│   ├── test_types.py
│   ├── test_document_processor.py
│   ├── handlers/
│   │   └── test_base_handler.py
│   ├── services/
│   │   ├── test_tag_service.py
│   │   └── test_table_service.py
│   └── chunking/
│       └── test_strategies.py
```

**통계**: 93 테스트, 796 LOC, 전체 통과 (0.98s)

### 4.2 테스트 커버리지 갭 분석

| 영역 | 테스트 유무 | 우선순위 |
|------|-----------|---------|
| ProcessingConfig 직렬화 | ✅ | - |
| ChunkingConfig 검증 | ✅ | - |
| BaseHandler 파이프라인 | ✅ | - |
| TagService 태그 생성/파싱 | ✅ | - |
| TableService 포매팅 | ✅ | - |
| **ImageService** | ❌ | 🟠 높음 |
| **ChartService** | ❌ | 🟡 중간 |
| **MetadataService** | ❌ | 🟡 중간 |
| **개별 핸들러** (17개) | ❌ | 🟠 높음 |
| **OCR 시스템** | ❌ | 🟡 중간 |
| **AsyncDocumentProcessor** | ❌ | 🟡 중간 |
| **CachedDocumentProcessor** | ❌ | 🟡 중간 |
| **LocalStorageBackend** | ❌ | 🟡 중간 |
| **위임 경로** | ❌ | 🟠 높음 |
| **통합 테스트 (실제 파일)** | ❌ | 🟠 높음 |
| **성능/메모리 테스트** | ❌ | 🔵 낮음 |

### 4.3 테스트 전략 제안

**Tier 1 (단위 테스트 — 즉시)**:
- 모든 서비스 클래스 (ImageService, ChartService, MetadataService)
- 캐시 백엔드 (MemoryCacheBackend, DiskCacheBackend)
- 위임 경로 로직 (DOC→RTF, PPT→PPTX 등)
- TableService.format_as_html() HTML 이스케이프 검증

**Tier 2 (통합 테스트 — 단기)**:
- 각 핸들러별 실제 파일 처리 (샘플 파일 필요)
- PDF 스캔 감지 + OCR 폴백
- 대형 파일 메모리 사용량 검증
- 비밀번호 보호 파일 에러 처리

**Tier 3 (E2E/성능 — 중기)**:
- 전체 파이프라인 roundtrip (파일 → 텍스트 → 청크)
- 동시 처리 (AsyncDocumentProcessor.extract_batch)
- 메모리 프로파일링

---

## 5. 의존성 분석

### 5.1 Core 의존성 위험도

| 패키지 | 버전 | 유지보수 상태 | 위험도 | 비고 |
|--------|------|-------------|--------|------|
| beautifulsoup4 | >=4.12.0 | ✅ 활발 | 낮음 | |
| chardet | >=5.0.0 | ✅ 활발 | 낮음 | |
| langchain-text-splitters | >=1.0.0 | ✅ 활발 | 🟡 | LangChain 생태계 변동 잦음 |
| pymupdf | >=1.24.0 | ✅ 활발 | 낮음 | AGPL 라이선스 주의 |
| pdfplumber | >=0.11.0 | ✅ 활발 | 낮음 | |
| pdfminer.six | >=20231228 | ✅ 활발 | 낮음 | |
| pdf2image | >=1.17.0 | ✅ 활발 | 낮음 | poppler 시스템 의존성 |
| python-docx | >=1.1.0 | ✅ 활발 | 낮음 | |
| docx2pdf | >=0.1.8 | 🟡 느림 | 중간 | Word/LibreOffice 필요 |
| python-pptx | >=1.0.0 | ✅ 활발 | 낮음 | |
| openpyxl | >=3.1.0 | ✅ 활발 | 낮음 | |
| xlrd | >=2.0.0 | 🟡 안정 | 낮음 | XLS-only, 더 이상 활발하지 않음 |
| pyhwp | >=0.1b15 | 🟠 제한 | 높음 | 한국 커뮤니티 전용, 0.1b 상태 |
| olefile | >=0.47 | ✅ 활발 | 낮음 | |
| striprtf | >=0.0.29 | 🟡 안정 | 낮음 | |
| pi-heif | >=1.0.0 | ✅ 활발 | 낮음 | |
| pytesseract | >=0.3.10 | ✅ 활발 | 🟡 | Tesseract 시스템 의존성 |

### 5.2 시스템 의존성

| 의존성 | 필요 핸들러 | 선택/필수 |
|--------|-----------|----------|
| **Tesseract OCR** | image, pdf 스캔 | 선택적 (pytesseract 래퍼) |
| **Poppler** | pdf2image 사용 시 | 선택적 |
| **LibreOffice** | DOC/PPT/XLS 변환 | 선택적 (미래) |

### 5.3 의존성 최적화 기회

| 제안 | 효과 |
|------|------|
| `docx2pdf` 분리 | Word/LibreOffice 불필요 시 제거 가능 |
| `pdfplumber` + `pdfminer.six` 조건부 | pdf_plus에서만 필요 |
| `pdf2image` 조건부 | OCR 사용 시에만 필요 |

### 5.4 라이선스 주의사항

| 패키지 | 라이선스 | 주의 |
|--------|---------|------|
| **pymupdf** | AGPL-3.0 | **🟠 AGPL 전파 위험** — 상업적 사용 시 오픈소스 공개 의무 또는 별도 라이선스 필요 |
| contextifier | Apache-2.0 | AGPL과의 호환성 검토 필요 |

---

## 6. 플러그인 시스템 분석

### 6.1 현재 구현

```python
# HandlerRegistry.register_defaults()
def _discover_plugins(self):
    """
    Third-party handlers via entry-point group.
    pyproject.toml:
        [project.entry-points."contextifier.handlers"]
        my_format = "my_package.handler:MyHandler"
    """
    import importlib.metadata
    for ep in importlib.metadata.entry_points(group="contextifier.handlers"):
        handler_class = ep.load()
        self.register(handler_class)
```

### 6.2 플러그인 시스템 평가

**✅ 강점**:
- 표준 Python entry_points 메커니즘 사용
- BaseHandler ABC 상속으로 인터페이스 강제
- `register()` 메서드로 수동 등록도 가능

**🟡 개선 영역**:

| 항목 | 현재 | 필요 |
|------|------|------|
| 플러그인 검증 | TypeError만 체크 | 버전 호환성, 인터페이스 검증 |
| 플러그인 격리 | 없음 | 플러그인 에러가 전체 영향 |
| 플러그인 문서 | 없음 | 개발 가이드 필요 |
| 플러그인 검색 | entry_points만 | 디렉토리 스캔 추가 가능 |
| 플러그인 우선순위 | 덮어쓰기 | 명시적 우선순위 |

---

## 7. 에코시스템 통합

### 7.1 LangChain 통합

**현재 상태**:
- `langchain-text-splitters`는 core 의존성 (청킹에 사용)
- `langchain` 자체는 optional 의존성
- LangChain Document Loader 인터페이스 미구현

**🟡 필요한 통합**:
```python
# LangChain Document Loader 호환
from langchain.document_loaders.base import BaseLoader

class ContextifierLoader(BaseLoader):
    def __init__(self, file_path, config=None):
        self.processor = DocumentProcessor(config)
        self.file_path = file_path

    def load(self) -> List[Document]:
        result = self.processor.process(self.file_path)
        return [Document(page_content=result.text, metadata=result.metadata.to_dict())]
```

### 7.2 FastAPI/Server 통합

**optional 의존성**: pydantic, python-multipart, orjson 등이 `[server]` 그룹에 포함
**현재 서버 구현**: ❌ 없음 (의존성만 존재)

**🟡 필요한 구현**:
- REST API 엔드포인트 (파일 업로드 → 텍스트/청크 반환)
- WebSocket 지원 (OCR 진행 상황 스트리밍)
- 배치 처리 엔드포인트

### 7.3 기타 통합 기회

| 프레임워크 | 통합 방식 | 우선순위 |
|-----------|----------|---------|
| **LlamaIndex** | Document Reader 구현 | 🟠 |
| **Haystack** | Converter 구현 | 🟡 |
| **Unstructured** | 포맷 호환 | 🔵 |
| **Apache Airflow** | Operator 구현 | 🔵 |

---

## 8. 배포 & 패키징

### 8.1 현재 구성

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
version = "0.2.5"
requires-python = ">=3.12"
```

### 8.2 배포 이슈

| 항목 | 현재 | 이슈 |
|------|------|------|
| 버전 관리 | 수동 (pyproject.toml) | 자동화 필요 (bumpversion/commitizen) |
| CI/CD | ❌ 없음 | GitHub Actions 필요 |
| PyPI 배포 | ❌ 수동 | `twine upload` 자동화 |
| Docker 이미지 | ❌ 없음 | Tesseract/Poppler/LibreOffice 포함 이미지 |
| Python 3.14 호환 | ⚠️ 경고 있음 | Pydantic V1 호환성 경고 |

### 8.3 Python 3.14 호환성 경고

```
UserWarning: Core Pydantic V1 functionality isn't compatible
with Python 3.14 or greater.
```

→ langchain-core의 Pydantic V1 의존성 문제. LangChain 업데이트 대기 필요.

---

## 9. 종합 개선 제안

### 9.1 API 개선

| 항목 | 현재 | 제안 |
|------|------|------|
| CachedDocumentProcessor 완성 | extract_text만 캐싱 | process(), extract_chunks() 추가 |
| format_options 타입 안전화 | `Dict[str, Dict[str, Any]]` | 포맷별 TypedDict 정의 |
| ExtractionResult 활용 | 내부 전달용 | Public API에서 더 적극 노출 |

### 9.2 DX 개선

| 항목 | 제안 | 우선순위 |
|------|------|---------|
| API Reference 자동 생성 | Sphinx/MkDocs + autodoc | 🟠 |
| 핸들러 기능 비교표 문서 | README 또는 별도 문서 | 🟠 |
| 설정 레퍼런스 문서 | 모든 Config 옵션 열거 | 🟠 |
| 플러그인 개발 가이드 | BaseHandler 상속 가이드 | 🟡 |
| 에러 코드 레퍼런스 | 모든 E_* 코드 설명 | 🟡 |

### 9.3 에코시스템 개선

| 항목 | 제안 | 우선순위 |
|------|------|---------|
| LangChain Loader | ContextifierLoader 구현 | 🟠 |
| CI/CD | GitHub Actions (lint, test, publish) | 🟠 |
| Docker 이미지 | 시스템 의존성 포함 | 🟡 |
| LlamaIndex Reader | Document Reader 구현 | 🟡 |
