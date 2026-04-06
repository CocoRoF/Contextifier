# License Compatibility Review: PyMuPDF (AGPL-3.0)

**Date**: 2025-07-19
**Project License**: Apache-2.0
**Dependency**: PyMuPDF (`pymupdf` / `fitz`) — AGPL-3.0

---

## 1. 문제 요약 (Executive Summary)

Contextifier는 Apache-2.0 라이선스로 배포되며, PyMuPDF는 AGPL-3.0 라이선스이다.
AGPL-3.0은 copyleft 라이선스로, **PyMuPDF를 링크/사용하는 프로그램 전체가 AGPL-3.0 조건을 따라야** 할 수 있다.
이는 Apache-2.0과 호환되지 않는다.

### 판정

| 항목 | 상태 |
|------|------|
| Contextifier 라이선스 | Apache-2.0 (permissive) |
| PyMuPDF 라이선스 | AGPL-3.0 (copyleft) |
| **호환성** | **⚠️ 비호환 — 주의 필요** |
| 영향 범위 | PDF 처리 핸들러 (pdf, pdf_default, pdf_plus) |

---

## 2. AGPL-3.0 핵심 조건

1. **소스코드 공개 의무**: PyMuPDF를 사용하는 전체 프로그램의 소스코드를 AGPL-3.0으로 배포해야 함
2. **네트워크 사용 조항 (Section 13)**: 서버에서 실행 시에도 소스코드 접근권을 제공해야 함
3. **라이선스 전파 (copyleft)**: 파생 저작물에 동일 라이선스 적용 의무

### Apache-2.0과의 충돌

- Apache-2.0은 파생 저작물에 다른 라이선스를 허용함
- AGPL-3.0은 파생 저작물에 동일 라이선스를 요구함
- **두 라이선스는 양립할 수 없음** (FSF 공식 입장)

---

## 3. 조치 사항 (Implemented)

### 3.1 pymupdf를 선택적 의존성으로 전환

`pyproject.toml`에서 pymupdf를 `dependencies`에서 `[project.optional-dependencies]`로 이동:

```toml
# Before (required):
dependencies = [
    "pymupdf>=1.24.0",
    ...
]

# After (optional):
[project.optional-dependencies]
pdf = ["pymupdf>=1.24.0"]
```

이를 통해:
- 기본 설치(`pip install contextifier`)에는 pymupdf가 포함되지 않음
- 사용자가 명시적으로 `pip install contextifier[pdf]`로 설치해야 함
- 사용자가 AGPL-3.0 조건을 인지하고 선택할 수 있음

### 3.2 가드 임포트 패턴 적용

모든 `import fitz` 호출을 try/except로 보호하여, pymupdf가 없을 때 명확한 에러 메시지 제공:

```python
try:
    import fitz
except ImportError:
    fitz = None  # type: ignore[assignment]
```

### 3.3 대안 유지

pymupdf 없이도 기본적인 PDF 텍스트 추출이 가능한 라이브러리를 유지:
- `pdfplumber` (MIT) — 테이블 추출
- `pdfminer.six` (MIT/X11) — 텍스트 추출
- `pdf2image` (MIT) — 이미지 변환

---

## 4. 사용자 안내

### PyMuPDF 사용 시 (AGPL-3.0 수용)

```bash
pip install contextifier[pdf]
```

- PDF 핸들러(pdf, pdf_default, pdf_plus)가 전체 기능으로 동작
- **사용자의 프로젝트가 AGPL-3.0 조건에 영향을 받을 수 있음**

### PyMuPDF 미사용 시 (Apache-2.0 유지)

```bash
pip install contextifier
```

- PDF 핸들러 사용 시 `ImportError`와 함께 설치 안내 메시지 표시
- pdfplumber, pdfminer.six를 통한 기본 텍스트 추출은 여전히 가능

---

## 5. PyMuPDF 상업 라이선스 옵션

Artifex Software는 PyMuPDF의 상업 라이선스도 제공한다.
AGPL-3.0 조건을 피하면서 PyMuPDF를 사용하려면 상업 라이선스 구매를 고려할 수 있다.

- https://pymupdf.readthedocs.io/en/latest/about.html#license

---

## 6. 영향받는 파일 목록

| 파일 | fitz 사용 방식 |
|------|---------------|
| `contextifier/handlers/pdf/converter.py` | `fitz.open()` — PDF 문서 열기 |
| `contextifier/handlers/pdf_default/content_extractor.py` | fitz 페이지 텍스트 추출 |
| `contextifier/handlers/pdf_plus/_block_image_engine.py` | 블록 이미지 처리 (이미 가드 적용) |
| `contextifier/handlers/pdf_plus/_image_extractor.py` | 이미지 추출 |
| `contextifier/handlers/pdf_plus/_vector_text_ocr.py` | 벡터 텍스트 OCR 렌더링 |
