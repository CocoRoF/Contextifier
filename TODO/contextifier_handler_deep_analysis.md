# Contextifier 핸들러 심층 분석 보고서

> 분석 대상: [CocoRoF/Contextifier](https://github.com/CocoRoF/Contextifier) v0.2.4
> 분석일: 2026-03-25
> 분석 범위: 14개 핸들러 전체 + 공통 인프라 (BaseHandler, Registry)
> 분석자: 손성준 (Developer Agent)

---

## 목차

1. [공통 인프라 (BaseHandler / Registry)](#1-공통-인프라)
2. [PDF Handler](#2-pdf-handler)
3. [DOCX Handler](#3-docx-handler)
4. [DOC Handler](#4-doc-handler)
5. [PPTX Handler](#5-pptx-handler)
6. [PPT Handler](#6-ppt-handler)
7. [XLSX Handler](#7-xlsx-handler)
8. [XLS Handler](#8-xls-handler)
9. [CSV Handler](#9-csv-handler)
10. [TSV Handler](#10-tsv-handler)
11. [HWP Handler](#11-hwp-handler)
12. [HWPX Handler](#12-hwpx-handler)
13. [RTF Handler](#13-rtf-handler)
14. [Text Handler (Category)](#14-text-handler-category)
15. [Image Handler (Category)](#15-image-handler-category)
16. [누락된 핸들러 분석](#16-누락된-핸들러-분석)
17. [핸들러 전체 비교 매트릭스](#17-핸들러-전체-비교-매트릭스)

---

## 1. 공통 인프라

### 1.1 BaseHandler (`handlers/base.py`)

#### 역할
모든 핸들러의 추상 기반 클래스. Template Method 패턴으로 5단계 파이프라인을 강제한다.

#### 5단계 파이프라인 구조

```
FileContext (raw bytes + metadata)
     │
     ▼
[Stage 1] Converter          → 원시 바이트를 처리 가능한 객체로 변환
     │                          (PDF→fitz.Document, DOCX→docx.Document, ...)
     ▼
[Stage 2] Preprocessor       → 핸들러별 전처리 (이미지/차트/메타데이터 사전 추출 등)
     │
     ▼
[Stage 3] MetadataExtractor  → 문서 메타데이터 추출 (제목, 저자, 생성일 등)
     │
     ▼
[Stage 4] ContentExtractor   → 핵심 텍스트/이미지/표 추출 → ExtractionResult
     │
     ▼
[Stage 5] Postprocessor      → 텍스트 정규화, 후처리
     │
     ▼
ExtractionResult (text, metadata, images, tables, ...)
```

#### 의존성 주입 구조

```python
BaseHandler.__init__(
    config: ProcessingConfig,
    image_service: ImageService,
    tag_service: TagService,
    chart_service: ChartService,
    table_service: TableService,
    metadata_service: MetadataService,
)
```

모든 서비스가 생성자에서 주입된다. 이로 인해 테스트 시 Mock 교체가 용이하다.

#### 위임(Delegation) 메커니즘

```python
def _check_delegation(self, file_context, **kwargs) -> Optional[ExtractionResult]:
    """마법 바이트 기반으로 실제 포맷 감지 후 적절한 핸들러에 위임"""
    pass  # 서브클래스가 override

def _delegate_to(self, handler_name: str, file_context, **kwargs) -> ExtractionResult:
    """레지스트리에서 대상 핸들러 조회 후 process() 호출"""
    pass
```

위임 흐름:
- `.doc` 확장자로 접근 → 마법 바이트가 ZIP이면 → DocxHandler에 위임
- `.ppt` → ZIP이면 → PptxHandler에 위임
- `.hwp` → ZIP이면 → HwpxHandler에 위임
- `.xls` → ZIP이면 → XlsxHandler에 위임 (XLSX는 ZIP 기반)
- `.doc` → RTF 마법 바이트이면 → RtfHandler에 위임

#### 발견된 문제점

| # | 문제 | 위치 | 심각도 | 상세 |
|---|------|------|--------|------|
| 1 | `process()` 미보호 — `@final` 없음 | `base.py:process()` | 🟡 MEDIUM | 서브클래스가 파이프라인을 임의로 Override 가능 |
| 2 | 파일 전체 메모리 로드 | `document_processor.py:_create_file_context()` | 🔴 HIGH | `Path.read_bytes()` — 5GB PDF는 OOM 유발 |
| 3 | Timeout 없음 | `base.py:process()` | 🔴 HIGH | 손상된 파일 파싱 시 무한 대기 가능 |
| 4 | 서비스 인스턴스 공유 (Thread safety) | `DocumentProcessor` | 🟡 MEDIUM | 동일 ImageService 인스턴스를 여러 호출이 공유 |
| 5 | Progress 콜백 없음 | `process()` | 🟡 MEDIUM | 대용량 파일 처리 상태 불투명 |
| 6 | Async 미지원 | 전체 | 🟡 MEDIUM | 모든 처리가 동기적 |

---

### 1.2 HandlerRegistry (`handlers/registry.py`)

#### 역할
확장자 문자열 → 핸들러 인스턴스 매핑을 관리. 지연 임포트(lazy import)로 선택적 의존성 지원.

#### 동작 방식

```python
# 등록: 확장자 → 핸들러 클래스 이름
registry.register("pdf", PDFHandler)

# 조회: 확장자로 핸들러 인스턴스 반환
handler = registry.get_handler("pdf")  # PDFHandler 인스턴스

# 기본 핸들러 등록 (importlib 사용)
registry.register_defaults()
```

#### register_defaults() 흐름

```python
default_handlers = [
    ("contextifier.handlers.pdf.handler", "PDFHandler"),
    ("contextifier.handlers.docx.handler", "DocxHandler"),
    # ... 14개
]

for module_path, class_name in default_handlers:
    try:
        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)
        self.register(handler_class)
    except Exception as e:
        logger.info(f"...")  # ⚠ 실패를 INFO로만 로깅 — 사용자 인지 불가
```

#### 발견된 문제점

| # | 문제 | 심각도 | 상세 |
|---|------|--------|------|
| 1 | HTML 핸들러 누락 | 🔴 HIGH | `register_defaults()`에 HTML/HTM/XHTML 핸들러 없음. README는 HTML 구조 보존 지원 명시 |
| 2 | 등록 실패 무음 처리 | 🟡 MEDIUM | `logger.info`만 출력 — 의존성 미설치 시 핸들러 누락을 사용자가 알 수 없음 |
| 3 | 핸들러 언등록 API 없음 | 🟡 MEDIUM | 테스트/커스터마이징 시 특정 핸들러 제거 불가 |
| 4 | 확장자 충돌 허용 | 🟢 LOW | 중복 등록 시 warning만 출력, 덮어쓰기 허용 |
| 5 | Health check 없음 | 🟢 LOW | 등록 후 핸들러 실제 동작 검증 미지원 |

---

## 2. PDF Handler

### 2.1 구조

```
handlers/pdf/
├── handler.py           # PDFHandler: mode 선택 로직
├── converter.py         # PyMuPDF(fitz)로 PDF 로드
├── preprocessor.py      # 페이지 정보 사전 처리
├── metadata_extractor.py # PDF 메타데이터 추출
├── _content_extractor_factory.py  # pdf_plus vs pdf_default 분기
└── _constants.py        # 상수 정의

handlers/pdf_plus/       # 고급 레이아웃 분석 (15+ 파일)
├── _layout_block_detector.py  (29KB) — 레이아웃 블록 감지
├── _table_detection.py        (25KB) — 표 감지 알고리즘
├── _types.py                  (24KB) — 타입 정의
├── _complexity_analyzer.py    — 페이지 복잡도 분석
├── _cell_analysis.py          — 셀 내용 분석
├── _line_analysis.py          — 텍스트 라인 분석
├── _page_analyzer.py          — 페이지 종합 분석
├── _graphic_detector.py       — 그래픽 요소 감지
├── _image_extractor.py        — 이미지 추출
├── _table_processor.py        — 표 처리
├── _table_validator.py        — 표 유효성 검증
├── _table_quality_analyzer.py — 표 품질 분석
├── _text_extractor.py         — 텍스트 추출
├── _text_quality_analyzer.py  — 텍스트 품질 분석
└── _vector_text_ocr.py        — 벡터 텍스트 OCR

handlers/pdf_default/    # 기본 텍스트 추출
└── content_extractor.py
```

### 2.2 처리 흐름

```
PDF bytes
    │
    ▼ Converter
fitz.Document (PyMuPDF)
    │
    ▼ Preprocessor
페이지 수, 리소스 목록 수집
    │
    ▼ MetadataExtractor
author, title, subject, keywords, creator, producer, creationDate
    │
    ▼ ContentExtractor (mode 선택)
    ├── pdf_plus: 레이아웃 분석 → 블록 감지 → 표/이미지/텍스트 구조화 추출
    └── pdf_default: 페이지별 단순 텍스트 추출 (fitz.page.get_text())
    │
    ▼ Postprocessor
텍스트 정규화
```

### 2.3 pdf_plus vs pdf_default 선택 로직

```python
# handler.py
def create_content_extractor(self):
    mode = self._config.get_format_option("pdf", "mode", "plus")
    if mode == "default":
        return PdfDefaultContentExtractor(...)
    return PdfPlusContentExtractor(...)   # 기본값 (오타 포함 모든 경우)
```

**문제**: `mode == "default"` 비교만 있어, 오타("defaut", "PLUS" 등) 시 묵시적으로 `plus` 모드로 fallback.

### 2.4 발견된 문제점

| # | 문제 | 심각도 | 코드 위치 |
|---|------|--------|-----------|
| 1 | `mode` 문자열 유효성 검증 없음 — 오타 시 묵시적 `plus` fallback | 🔴 HIGH | `handler.py:create_content_extractor()` |
| 2 | 암호화 PDF 지원 없음 | 🔴 HIGH | `converter.py` |
| 3 | 스캔 PDF 자동 OCR 전환 없음 | 🟡 MEDIUM | `preprocessor.py` |
| 4 | 다단 컬럼 레이아웃 읽기 순서 불안정 | 🟡 MEDIUM | `pdf_plus/_layout_block_detector.py` |
| 5 | 대용량 PDF 페이지별 스트리밍 없음 | 🟡 MEDIUM | `content_extractor.py` |
| 6 | pdf_plus 레이아웃 분석 heuristic에 단위 테스트 없음 | 🟡 MEDIUM | `pdf_plus/` 전체 |

### 2.5 개선 방안

**[HIGH] Mode 유효성 검증**
```python
PDF_VALID_MODES = frozenset({"plus", "default"})

mode = self._config.get_format_option("pdf", "mode", "plus")
if mode not in PDF_VALID_MODES:
    raise ValueError(f"Invalid PDF mode '{mode}'. Valid options: {PDF_VALID_MODES}")
```

**[HIGH] 암호화 PDF 지원**
```python
def convert(self, file_context, *, password=None, **kwargs):
    doc = fitz.open(stream=file_context.file_data, filetype="pdf")
    if doc.needs_pass:
        if not password:
            raise ConversionError("PDF is password-protected. Pass 'password=' kwarg.")
        if not doc.authenticate(password):
            raise ConversionError("Incorrect PDF password.")
```

**[MEDIUM] 스캔 PDF 자동 OCR fallback**
```python
sample_text = sum(len(p.get_text()) for p in doc[:5])
if sample_text < 50:
    preprocessed.resources["needs_ocr"] = True
```

---

## 3. DOCX Handler

### 3.1 구조

```
handlers/docx/
├── handler.py
├── converter.py             # python-docx로 .docx 로드 (ZIP 파일)
├── preprocessor.py          # 차트/이미지 관계 사전 수집
├── metadata_extractor.py    # 코어 속성 추출
└── content_extractor.py     # 단락/표/이미지/차트 추출 (12.8KB)
```

### 3.2 핵심 추출 로직

**단락 처리 — Run 단위 텍스트 결합 + 스타일 태그**:
```
Heading 1 → [H1]텍스트[/H1]
Bold run  → [B]텍스트[/B]
Link      → [LINK href="url"]텍스트[/LINK]
```

**이미지 처리 (`_extract_image_by_rel`)**:
- Relationship ID로 이미지 파트 조회
- 중복 제거: `processed_images: Set[str]`에 **rel_id** 저장
- ⚠️ 같은 이미지가 다른 rel_id로 참조되면 중복 저장

**차트 처리 (`_process_drawing`)**:
- ⚠️ 문서 내 순서와 Preprocessor 수집 순서 불일치 시 잘못된 차트 연결

### 3.3 발견된 문제점

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | `import re`가 `extract_text()` 내부에 있음 (매 호출마다 import) | `content_extractor.py` | 🔴 HIGH |
| 2 | 이미지 중복 제거가 rel_id 기반 (내용 동일 이미지 중복 저장) | `_extract_image_by_rel()` | 🔴 HIGH |
| 3 | 차트 매칭이 인덱스 기반 (순서 불일치 시 잘못된 매핑) | `_process_drawing()` | 🟡 MEDIUM |
| 4 | 헤더/푸터 추출 없음 | `content_extractor.py` | 🟡 MEDIUM |
| 5 | 각주(Footnote)/미주(Endnote) 추출 없음 | `content_extractor.py` | 🟡 MEDIUM |
| 6 | 텍스트 박스(TextBox) 추출 없음 | `content_extractor.py` | 🟡 MEDIUM |
| 7 | 변경 추적(Track Changes) 미처리 | `content_extractor.py` | 🟢 LOW |
| 8 | 1×1 표를 무조건 일반 텍스트 변환 | `_format_table()` | 🟢 LOW |

### 3.4 개선 방안

**[HIGH] `import re` 모듈 레벨로 이동**
```python
import re
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# extract_text() 내에서
result = _EXCESS_NEWLINES.sub("\n\n", result)
```

**[HIGH] 이미지 중복제거: content hash 기반**
```python
import hashlib

def _extract_image_by_rel(self, rel_id, doc, processed_hashes: Set[str]):
    image_data = rel.target_part.blob
    content_hash = hashlib.md5(image_data).hexdigest()
    if content_hash in processed_hashes:
        return None
    processed_hashes.add(content_hash)
```

**[MEDIUM] 헤더/푸터 추출**
```python
def _extract_headers_footers(self, doc) -> List[str]:
    results = []
    for section in doc.sections:
        for hf_type, hf in [("Header", section.header), ("Footer", section.footer)]:
            if hf and not hf.is_linked_to_previous:
                text = "\n".join(p.text.strip() for p in hf.paragraphs if p.text.strip())
                if text:
                    results.append(f"[{hf_type}]\n{text}")
    return results
```

**[MEDIUM] 차트 매칭: rel_id 기반으로 개선**
```python
# preprocessor.py: {rel_id: chart_text} 딕셔너리 생성
charts_by_rel: Dict[str, str] = {}
for rel in doc.part.rels.values():
    if "chart" in rel.target_ref:
        chart_text = extract_chart_text(rel.target_part)
        charts_by_rel[rel.rId] = chart_text

# content_extractor.py: rel_id로 직접 조회
def _process_drawing(self, drawing, charts_by_rel):
    if drawing.kind == DrawingKind.CHART:
        return charts_by_rel.get(drawing.chart_rel_id, "[Chart]")
```

---

## 4. DOC Handler

### 4.1 처리 흐름 (위임 우선)

```
.doc 파일
    │
    ▼ _check_delegation()
    ├── ZIP 마법 바이트 (50 4B) → DocxHandler에 위임
    ├── RTF 마법 바이트 (7B 5C) → RtfHandler에 위임
    ├── HTML 마법 바이트      → ⚠️ TODO: HTMLHandler에 위임 (미구현)
    └── OLE2 마법 바이트 (D0 CF) → 자체 DOC 파이프라인
```

**핵심 발견**: `handler.py`에 `# TODO: delegate to 'html' handler once implemented` 주석 존재.

### 4.2 발견된 문제점

| # | 문제 | 심각도 | 상세 |
|---|------|--------|------|
| 1 | HTML 위임 미구현 (TODO로 방치) | 🔴 HIGH | .doc 파일이 HTML 포맷이면 처리 불가 |
| 2 | 바이너리 DOC 표(Table) 추출 완성도 | 🟡 MEDIUM | 구버전 DOC 포맷 표 구조 복잡 |
| 3 | 복잡한 DOC (매크로, 양식) 파싱 실패 가능 | 🟡 MEDIUM | OLE2 구조의 다양성 |
| 4 | LibreOffice 변환 fallback 없음 | 🟡 MEDIUM | pyhwp/oletools 실패 시 복구 수단 없음 |

---

## 5. PPTX Handler

### 5.1 구조

```
handlers/pptx/
├── handler.py
├── converter.py             # python-pptx로 .pptx 로드
├── preprocessor.py          # 슬라이드 차트/이미지 관계 사전 수집
├── metadata_extractor.py
└── content_extractor.py     # 슬라이드 처리 (21KB — 가장 복잡한 content extractor)
```

### 5.2 강점 — 시각적 읽기 순서

```python
# shapes를 시각적 위치 기준으로 정렬 (읽기 순서 보장)
sorted_shapes = sorted(
    slide.shapes,
    key=lambda s: (s.top, s.left)
)
```

### 5.3 그룹 도형 재귀 처리

```python
def _process_shape(self, shape) -> List[str]:
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        results = []
        for child in shape.shapes:
            results.extend(self._process_shape(child))  # 재귀
        return results
```

깊게 중첩된 그룹에서는 재귀 깊이 제한이 없다는 잠재적 문제가 있다.

### 5.4 발견된 문제점

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | `import re`가 `extract_text()` 내부에 있음 (DOCX와 동일 패턴) | `content_extractor.py` | 🔴 HIGH |
| 2 | 깊게 중첩된 그룹 도형 재귀 깊이 제한 없음 | `_process_shape()` | 🟡 MEDIUM |
| 3 | 마스터/레이아웃 슬라이드 텍스트 미추출 (헤더/배경 텍스트 누락) | `content_extractor.py` | 🟡 MEDIUM |

---

## 6. PPT Handler

### 6.1 바이너리 PPT 파이프라인 한계

OLE2 기반 구버전 PPT(`.ppt`) 포맷 파싱:
- 텍스트만 heuristic하게 추출 (python-pptx가 미지원)
- 표(Table) 추출 없음
- 차트 추출 없음
- 이미지 추출 없음 또는 불완전

**권장 개선**: LibreOffice CLI를 통해 `.ppt` → `.pptx` 변환 후 PptxHandler에 위임.

---

## 7. XLSX Handler

### 7.1 핵심 버그: XLSX 이미지 이름 충돌

```python
# content_extractor.py (현재 코드)
tag = self._image_service.save_and_tag(
    image_bytes=img_data,
    custom_name=f"excel_sheet_img"   # ⚠️ 모든 이미지가 같은 이름!
)
```

하나의 워크북에 이미지가 여러 개면 모두 `excel_sheet_img`라는 이름으로 저장 시도 → 이전 이미지를 덮어씀. **데이터 손실 버그**.

### 7.2 핵심 버그: openpyxl 내부 API 사용

```python
for img in ws._images:          # ⚠️ _images: 내부 속성
    img_data = img._data()      # ⚠️ _data(): 내부 메서드

ws_charts = getattr(ws, "_charts", [])  # ⚠️ _charts: 내부 속성
```

### 7.3 발견된 문제점

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | 모든 이미지에 동일한 이름 `excel_sheet_img` → 덮어쓰기 | `content_extractor.py` | 🔴 HIGH (버그) |
| 2 | `ws._images` 내부 API 사용 | `_extract_sheet_images()` | 🔴 HIGH |
| 3 | `img._data()` 내부 API 사용 | `_extract_sheet_images()` | 🔴 HIGH |
| 4 | 수식 → 문자열로 저장 (계산값 아님) | `preprocessor.py` | 🟡 MEDIUM |
| 5 | 숨김 시트 처리 제어 불가 | `content_extractor.py` | 🟡 MEDIUM |
| 6 | 차트 매칭 인덱스 기반 | `content_extractor.py` | 🟡 MEDIUM |

### 7.4 개선 방안

**[HIGH] 고유 이미지 이름 생성**
```python
content_hash = hashlib.md5(img_data).hexdigest()[:8]
unique_name = f"excel_{ws.title}_{content_hash}"
tag = self._image_service.save_and_tag(image_bytes=img_data, custom_name=unique_name)
```

**[HIGH] 내부 API 대체: ZIP 직접 접근**
```python
from zipfile import ZipFile
from io import BytesIO

def _extract_images_from_zip(self, file_data: bytes) -> List[bytes]:
    images = []
    with ZipFile(BytesIO(file_data)) as zf:
        for name in zf.namelist():
            if name.startswith("xl/media/"):
                images.append(zf.read(name))
    return images
```

**[MEDIUM] 수식 평가 지원**
```python
wb = openpyxl.load_workbook(BytesIO(file_data), data_only=data_only)
```

---

## 8. XLS Handler

```
.xls 파일
    │
    ▼ _check_delegation()
    ├── ZIP 마법 바이트 → XlsxHandler에 위임 (잘못된 확장자)
    └── BIFF 마법 바이트 → 자체 xlrd 파이프라인
```

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | 이미지 추출 없음 (xlrd는 이미지 미지원) | 🟡 MEDIUM |
| 2 | 차트 추출 없음 | 🟡 MEDIUM |
| 3 | LibreOffice 변환 fallback 없음 | 🟢 LOW |

---

## 9. CSV Handler

**처리 흐름**:
```
CSV bytes → 인코딩 감지 → 구분자 자동 감지 → 전체 CSV → Markdown 표 변환
```

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | 대용량 CSV (수십만 행) 전체 메모리 로드 | 🟡 MEDIUM |
| 2 | 인코딩 감지 실패 시 fallback 전략 불명확 | 🟡 MEDIUM |

---

## 10. TSV Handler

TSV는 CSV와 동일한 파이프라인을 사용하되, `delimiter='\t'`로 강제 설정. CSV와 동일한 문제 공유.

---

## 11. HWP Handler

```
.hwp 파일
    │
    ▼ _check_delegation()
    ├── ZIP 마법 바이트 (50 4B) → HwpxHandler에 위임
    ├── HWP 3.0 마법 바이트 → ConversionError 발생 (지원 안함)
    └── OLE2 마법 바이트 (D0 CF) → 자체 HWP OLE2 파이프라인
```

### pyhwp 의존성 리스크

`pyhwp`는 커뮤니티 기반으로 유지보수되며, 최신 HWP 5.1+ 포맷 지원 여부 불명확.

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | pyhwp 라이브러리 유지보수 불안정 (최신 HWP 5.1+ 미지원 가능) | 🔴 HIGH |
| 2 | HWP 내 이미지 추출 완성도 | 🟡 MEDIUM |
| 3 | LibreOffice 변환 fallback 없음 | 🟡 MEDIUM |

**권장 개선**: LibreOffice로 HWP → DOCX 변환 후 DocxHandler에 위임.

---

## 12. HWPX Handler

```
handlers/hwpx/
├── handler.py      — HWPX ZIP 구조 파싱
└── _section.py     (19KB) — 섹션별 콘텐츠 추출
```

HWPX는 DOCX와 유사하게 ZIP+XML 기반이지만, 한컴만의 독자적인 XML 스키마를 사용한다.

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | HWPX 스키마가 한컴 독자 — 공식 문서 없음 (리버스 엔지니어링 기반) | 🟡 MEDIUM |
| 2 | 한글 특수 기능 (글맵시, 수식, OLE 객체) 처리 불완전 가능 | 🟡 MEDIUM |

---

## 13. RTF Handler

```
handlers/rtf/
├── handler.py
├── _cleaner.py     (18KB) — RTF 컨트롤 워드 파싱 / CJK 인코딩 처리
├── _table_parser.py (19KB) — RTF 표 구조 파싱
└── [이미지 추출 로직]
```

### 강점 — CJK/Korean 인코딩 지원

`_cleaner.py`가 RTF 표준의 한국어 인코딩을 처리:
- `\ansicpg949`: EUC-KR (한국어)
- `\ansicpg932`: Shift-JIS (일본어)
- `\ansicpg936`: GBK (중국어 간체)
- 16진수 이스케이프 (`\'xx`) → 실제 문자 변환

---

## 14. Text Handler (Category Handler)

### TextHandler는 단일 핸들러가 50+ 확장자를 처리한다

```python
_TEXT_EXTENSIONS = frozenset({
    "txt", "text", "log", "md", "markdown", "rst",
    "py", "js", "ts", "java", "c", "cpp",
    "html", "htm", "xhtml",   # ⚠️ HTML을 plain text 처리!
    "css", "scss", "sass",    # ⚠️ CSS를 plain text 처리!
    "svg",                    # ⚠️ SVG를 plain text 처리!
    "json", "yaml", "yml", "toml", "xml", "ini", "cfg",
})
```

### 핵심 문제: HTML이 Plain Text로 처리

**README의 주장**: "Table/structure preservation for HTML"
**실제 구현**: HTML을 그냥 읽어서 텍스트 추출 시 태그 제거

`BeautifulSoup4`가 `pyproject.toml`에 의존성으로 명시되어 있으나 TextHandler에서 사용되지 않음.

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | HTML → plain text 처리 (README 약속 불이행) | 🔴 HIGH |
| 2 | BS4 의존성 있으나 미사용 | 🔴 HIGH |
| 3 | CSS/SVG를 텍스트로 처리 (AI에게 의미없는 코드 전달) | 🟡 MEDIUM |
| 4 | 50+ 확장자 일괄 처리 → 포맷별 최적화 불가 | 🟡 MEDIUM |

### 개선 방안: HTML 핸들러 분리

```python
# handlers/html/handler.py (신규)
class HTMLHandler(BaseHandler):
    supported_extensions = frozenset({"html", "htm", "xhtml"})
    handler_name = "HTML Handler"

# handlers/html/content_extractor.py
from bs4 import BeautifulSoup

class HTMLContentExtractor:
    def extract_text(self, preprocessed, **kwargs) -> str:
        soup = preprocessed.content
        result_parts = []
        for tag in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
            level = tag.name.upper()
            result_parts.append(f"[{level}]{tag.get_text(strip=True)}[/{level}]")
        for table in soup.find_all("table"):
            result_parts.append(self._table_service.to_markdown(table))
        for p in soup.find_all("p"):
            result_parts.append(p.get_text(strip=True))
        return "\n\n".join(filter(None, result_parts))
```

---

## 15. Image Handler (Category Handler)

### OCR 서브시스템 연동

ImageHandler는 5가지 OCR 엔진을 지원:
- **OpenAI Vision** (GPT-4V)
- **Anthropic Claude** (claude-3-haiku, claude-3-opus)
- **Google Gemini** (gemini-pro-vision)
- **AWS Bedrock** (claude on bedrock)
- **vLLM** (로컬 배포)

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | OCR 없이는 이미지 내용 완전히 손실 (태그만 반환) | 🟡 MEDIUM |
| 2 | 고해상도 이미지 OCR 전 리사이즈 없음 (API 비용 증가) | 🟡 MEDIUM |
| 3 | EXIF 메타데이터 미추출 (촬영일, GPS 등) | 🟢 LOW |

---

## 16. 누락된 핸들러 분석

### 16.1 HTML 핸들러 — 계획됨, 미구현 (🔴 HIGH)

**근거**:
1. `handlers/doc/handler.py`: `# TODO: delegate to 'html' handler once implemented`
2. `pyproject.toml`: `beautifulsoup4` 의존성 등록 (사용처 없음)
3. README: HTML 구조 보존 지원 명시
4. `handlers/text/handler.py`: `html`, `htm`, `xhtml`이 plain text 처리

**필요 작업**:
```
handlers/html/ (신규 디렉토리)
├── handler.py           — HTMLHandler class
├── converter.py         — bytes → BeautifulSoup
├── preprocessor.py
├── metadata_extractor.py — <meta> 태그 파싱
└── content_extractor.py — 구조화된 HTML → AI 친화적 텍스트
```

### 16.2 Email 핸들러 (.eml, .msg)

비즈니스 문서 처리에서 이메일 파싱 수요가 높다:
- `.eml`: Python 표준 라이브러리 `email` 모듈로 처리 가능
- `.msg`: `extract-msg` 라이브러리 (Outlook MSG 포맷)
- 첨부파일 재귀 처리 (이메일 내 PDF → PDFHandler에 위임)

### 16.3 ZIP/Archive 핸들러 (.zip)

아카이브 내 문서 일괄 처리:
- ZIP → 내부 파일 목록 → 각 파일을 해당 핸들러로 처리
- 중첩 ZIP 처리 (재귀 깊이 제한 필요)

---

## 17. 핸들러 전체 비교 매트릭스

| 핸들러 | 텍스트 | 표 | 이미지 | 차트 | 헤더/푸터 | 각주 | 메타데이터 | 위임 | 완성도 |
|--------|--------|-----|--------|------|-----------|------|------------|------|--------|
| PDF (plus) | ✅ | ✅ | ✅ | ✅ | ✅ | - | ✅ | - | ⭐⭐⭐⭐ |
| PDF (default) | ✅ | ❌ | ❌ | ❌ | ❌ | - | ✅ | - | ⭐⭐ |
| DOCX | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | - | ⭐⭐⭐ |
| DOC | ✅ | △ | △ | ❌ | ❌ | ❌ | △ | ZIP→DOCX, RTF→RTF | ⭐⭐ |
| PPTX | ✅ | ✅ | ✅ | ✅ | △ | - | ✅ | - | ⭐⭐⭐⭐ |
| PPT | ✅ | ❌ | ❌ | ❌ | ❌ | - | △ | ZIP→PPTX | ⭐⭐ |
| XLSX | ✅ | ✅ | ⚠️ | △ | - | - | ✅ | ZIP→XLSX | ⭐⭐⭐ |
| XLS | ✅ | ✅ | ❌ | ❌ | - | - | △ | ZIP→XLSX | ⭐⭐ |
| CSV | ✅ | ✅ | - | - | - | - | △ | - | ⭐⭐⭐ |
| TSV | ✅ | ✅ | - | - | - | - | △ | - | ⭐⭐⭐ |
| HWP | ✅ | △ | △ | ❌ | ❌ | ❌ | △ | ZIP→HWPX | ⭐⭐ |
| HWPX | ✅ | △ | △ | ❌ | ❌ | ❌ | △ | - | ⭐⭐ |
| RTF | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | △ | - | ⭐⭐⭐ |
| Text | ✅ | ❌ | - | - | - | - | ❌ | - | ⭐⭐ |
| Image | △ | - | ✅ | - | - | - | ✅ | - | ⭐⭐⭐ |
| **HTML** | **❌** | **❌** | **-** | **-** | **-** | **-** | **❌** | **미구현** | **⭐** |

> ✅ 완전 지원 | △ 부분 지원 | ❌ 미지원 | ⚠️ 버그 있음 | - 해당 없음

---

## 18. 우선순위별 개선 로드맵 (핸들러 중심)

### Phase 1: 버그 수정 (1주 이내)

| 우선순위 | 작업 | 핸들러 | 예상 공수 |
|----------|------|--------|-----------|
| P0 | XLSX 이미지 이름 중복 버그 수정 | xlsx | 0.5일 |
| P0 | DOCX `import re` 모듈 레벨로 이동 | docx | 0.5시간 |
| P0 | PPTX `import re` 모듈 레벨로 이동 | pptx | 0.5시간 |
| P0 | PDF mode 유효성 검증 추가 | pdf | 0.5일 |
| P1 | XLSX `ws._images` / `img._data()` 내부 API 대체 | xlsx | 1~2일 |

### Phase 2: 기능 강화 (2~4주)

| 우선순위 | 작업 | 핸들러 | 예상 공수 |
|----------|------|--------|-----------|
| P1 | **HTML 핸들러 신규 구현** | html (신규) | 3~5일 |
| P1 | DOCX 이미지 중복제거 content hash 기반으로 변경 | docx | 1일 |
| P1 | DOCX 헤더/푸터/각주 추출 추가 | docx | 2~3일 |
| P2 | PDF 암호화 파일 지원 | pdf | 1~2일 |
| P2 | PDF 스캔 자동 OCR fallback | pdf | 1~2일 |
| P2 | DOCX/PPTX 차트 매칭 rel_id 기반으로 개선 | docx, pptx | 2일 |
| P2 | XLSX 수식 평가 모드 지원 (data_only) | xlsx | 0.5일 |
| P2 | XLSX 숨김 시트 처리 옵션 추가 | xlsx | 0.5일 |

### Phase 3: 아키텍처 고도화 (1~2개월)

| 우선순위 | 작업 | 설명 |
|----------|------|------|
| P2 | HWP/PPT/DOC LibreOffice fallback | 복잡한 레거시 포맷 처리 안정성 향상 |
| P2 | PPT→PPTX LibreOffice 변환 | 바이너리 PPT 파서 대체 |
| P3 | 테스트 인프라 구축 | 각 핸들러 단위/통합 테스트 |
| P3 | 비동기 처리 지원 | `aprocess()` 추가 |
| P3 | 대용량 파일 스트리밍 | `read_bytes()` → 스트리밍 FileContext |
| P3 | 플러그인 시스템 | entry_points 기반 외부 핸들러 지원 |

---

## 19. 결론

Contextifier의 핸들러 아키텍처는 **잘 설계된 골격** 위에 **구현 완성도 차이**가 큰 14개 핸들러들로 이루어져 있다.

### 핸들러 품질 계층

**Tier 1 (고품질)**:
- `pdf_plus`: 코드베이스에서 가장 정교한 구현. 레이아웃 분석, 표 감지 알고리즘 포함.
- `pptx`: 시각적 읽기 순서 정렬, 그룹 도형 재귀 처리 등 세밀한 구현.

**Tier 2 (중간)**:
- `docx`, `xlsx`, `rtf`, `csv`, `tsv`: 핵심 기능은 구현, 엣지 케이스와 고급 기능 부족.

**Tier 3 (기본)**:
- `doc`, `ppt`, `hwp`, `hwpx`, `xls`, `image`: 기본 텍스트 추출만 가능.

**미구현**:
- `html`: README에 명시되어 있으나 실제로는 TextHandler가 plain text로 처리.

### 가장 시급한 3가지

1. **XLSX 이미지 이름 버그** — 데이터 손실을 유발하는 실제 버그
2. **HTML 핸들러 구현** — README와 코드 불일치 해소, BS4 의존성 활용
3. **XLSX 내부 API(openpyxl) 교체** — openpyxl 버전 업 시 즉시 깨지는 시한폭탄

---

*본 보고서는 CocoRoF/Contextifier v0.2.4 GitHub 소스 코드 직접 분석을 기반으로 작성되었습니다.*
*작성: 손성준 (Developer Agent) | 2026-03-25*
