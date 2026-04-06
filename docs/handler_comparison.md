# Handler Feature Comparison

> Contextifier v0.3.0 — 15개 핸들러 기능 지원 매트릭스

## 지원 포맷 총괄

| 카테고리 | 확장자 | 핸들러 |
|----------|--------|--------|
| **문서** | `.pdf`, `.docx`, `.doc`, `.hwp`, `.hwpx`, `.rtf` | PDFHandler, DOCXHandler, DOCHandler, HWPHandler, HWPXHandler, RTFHandler |
| **프레젠테이션** | `.pptx`, `.ppt` | PPTXHandler, PPTHandler |
| **스프레드시트** | `.xlsx`, `.xls`, `.csv`, `.tsv` | XLSXHandler, XLSHandler, CSVHandler, TSVHandler |
| **웹** | `.html`, `.htm`, `.xhtml` | HtmlHandler |
| **텍스트/코드** | `.txt`, `.md`, `.log`, `.rst`, `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs` 등 (60+) | TextHandler |
| **이미지** | `.jpg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff` 등 (12+) | ImageFileHandler |

---

## 기능 매트릭스

| 핸들러 | 텍스트 | 테이블 | 이미지 | 차트 | 메타데이터 | 비고 |
|--------|:------:|:------:|:------:|:----:|:----------:|------|
| **PDFHandler** | ✅ | ✅ | ✅¹ | ❌ | ✅ | ¹이미지 태그 인라인 삽입 (OCR 연계) |
| **DOCXHandler** | ✅ | ✅ | ✅ | ❌ | ✅ | python-docx 기반 |
| **DOCHandler** | ✅ | ✅ | ✅ | ❌ | ✅ | OLE2 네이티브 파싱, 자동 포맷 감지 (RTF/DOCX/HTML 위임) |
| **PPTXHandler** | ✅ | ✅ | ✅ | ✅ | ✅ | python-pptx 기반, 그룹 셰이프 재귀 탐색 |
| **PPTHandler** | ✅ | ✅² | ✅ | ✅² | ✅ | OLE2 바이너리 파싱 ²휴리스틱 기반 |
| **XLSXHandler** | ✅ | ✅ | ✅ | ✅ | ✅ | openpyxl 기반, read_only 모드 지원 |
| **XLSHandler** | ✅ | ✅ | ✅ | ✅ | ✅ | xlrd + OLE 바이너리 확장 |
| **CSVHandler** | ✅ | ✅ | — | — | ✅ | 자동 구분자/인코딩 감지, max_rows 스트리밍 |
| **TSVHandler** | ✅ | ✅ | — | — | ✅ | CSVHandler 기반 (탭 구분자 고정) |
| **HWPHandler** | ✅ | ✅³ | ✅³ | —³ | ✅ | OLE2 네이티브 파싱 ³인라인 렌더링 |
| **HWPXHandler** | ✅ | ✅³ | ✅³ | ✅³ | ✅ | OOXML(ZIP) 파싱 ³인라인 렌더링 |
| **RTFHandler** | ✅ | ✅ | ✅ | ❌ | ✅ | RTF 파서 내장, 병합 셀 지원 |
| **HtmlHandler** | ✅ | ✅ | ✅⁴ | ❌ | ✅ | BeautifulSoup 기반 ⁴base64 임베디드 이미지 |
| **TextHandler** | ✅ | — | — | — | ⚠️ | 60+ 확장자 지원 (코드, 설정 파일 포함) |
| **ImageFileHandler** | ✅⁵ | — | ✅ | — | ✅ | ⁵OCR 엔진 연계 시 텍스트 추출 |

### 범례

- ✅ = 완전 지원
- ⚠️ = 제한적 지원 (기본 정보만)
- ❌ = 미지원 (포맷 특성상 불가 또는 미구현)
- — = 해당 없음 (포맷 특성)

### 주석

1. **PDF 이미지**: `extract_text()` 시 이미지를 저장하고 `[Image: ...]` 태그를 텍스트에 삽입. `ocr_processing=True`로 OCR 치환 가능.
2. **PPT 테이블/차트**: OLE2 바이너리 레코드 파싱. 복잡한 구조는 휴리스틱 기반 추출.
3. **HWP/HWPX 인라인**: 테이블·이미지·차트가 텍스트 추출 과정에서 인라인으로 렌더링됨. `extract_tables()` 등은 `[]` 반환.
4. **HTML 이미지**: base64 인코딩 인라인 이미지만 추출. 외부 URL 이미지는 태그로 유지.
5. **이미지 텍스트**: OCR 엔진 미설정 시 이미지 태그만 생성. 엔진 설정 시 텍스트 자동 추출.

---

## 추출 아키텍처 비교

### 분리형 (Separate Extraction)
테이블·이미지·차트를 `extract_tables()`, `extract_images()`, `extract_charts()` 메서드로 개별 추출:
- **DOCX, PPTX, XLSX, XLS, RTF, HTML, CSV/TSV, Image**

### 인라인형 (Inline Rendering)
모든 콘텐츠가 `extract_text()` 과정에서 HTML/텍스트로 인라인 렌더링:
- **HWP, HWPX**

### 하이브리드 (Hybrid)
이미지는 인라인 태그로 삽입하고, 테이블은 별도 추출도 가능:
- **PDF, DOC, PPT**

---

## 위임 (Delegation) 관계

```
DOCHandler ─── 콘텐츠 감지 ──→ RTFHandler (RTF 내용일 때)
           └── 콘텐츠 감지 ──→ DOCXHandler (OOXML 내용일 때)
           └── 콘텐츠 감지 ──→ HtmlHandler (HTML 내용일 때)
```

DOCHandler는 `.doc` 파일의 실제 포맷을 자동 감지하여, 해당 포맷의 전문 핸들러에 위임합니다.
위임 깊이는 최대 3단계로 제한됩니다 (`MAX_DELEGATION_DEPTH`).

---

## format_options 지원

| 핸들러 | 옵션 키 | 설명 | 기본값 |
|--------|---------|------|--------|
| PDFHandler | `render_dpi` | 이미지 렌더링 DPI | `150` |
| PDFHandler | `min_image_size` | 최소 이미지 크기 (px) | `100` |
| PDFHandler | `min_image_area` | 최소 이미지 면적 (px²) | `10000` |
| PPTXHandler | `max_group_depth` | 그룹 셰이프 최대 재귀 깊이 | `20` |
| DOCHandler | `min_text_fragment_length` | 최소 텍스트 프래그먼트 길이 | `20` |
| CSVHandler | `max_rows` | 최대 처리 행 수 | `10000` |
| CSVHandler | `delimiter_candidates` | 구분자 후보 목록 | `[",", "\t", "|", ";"]` |
| CSVHandler | `encodings` | 인코딩 후보 목록 | `["utf-8", "cp949", ...]` |
| TSVHandler | `max_rows` | 최대 처리 행 수 | `10000` |
| XLSXHandler | `read_only` | openpyxl read_only 모드 | `False` |
| PDFHandler | `table_size` | 테이블 감지 최소 크기 | `50` |
