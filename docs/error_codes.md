# Error Codes Reference

> Contextifier v0.3.0 — 예외 계층 및 에러 코드 레퍼런스

## 개요

모든 Contextifier 예외는 `ContextifierError` 기반 클래스를 상속합니다.
각 예외는 다음 속성을 제공합니다:

| 속성 | 타입 | 설명 |
|------|------|------|
| `message` | `str` | 사람이 읽을 수 있는 에러 설명 |
| `code` | `str` | 기계 판독 가능한 에러 코드 (예: `E_CONVERSION`) |
| `context` | `dict` | 디버깅용 추가 컨텍스트 |
| `cause` | `Exception \| None` | 원인이 된 원래 예외 |

### 에러 코드 생성 규칙

클래스명에서 `Error` 접미사를 제거하고 CamelCase를 UPPER_SNAKE로 변환합니다:
- `ConversionError` → `E_CONVERSION`
- `HandlerNotFoundError` → `E_HANDLER_NOT_FOUND`
- `ImageServiceError` → `E_IMAGE_SERVICE`

---

## 예외 계층

```
ContextifierError (E_CONTEXTIFIER)
├── ConfigurationError (E_CONFIGURATION)
├── FileError (E_FILE)
│   ├── FileNotFoundError (E_FILE_NOT_FOUND)
│   ├── FileReadError (E_FILE_READ)
│   └── UnsupportedFormatError (E_UNSUPPORTED_FORMAT)
├── PipelineError (E_PIPELINE)
│   ├── ConversionError (E_CONVERSION)
│   ├── PreprocessingError (E_PREPROCESSING)
│   ├── ExtractionError (E_EXTRACTION)
│   └── PostprocessingError (E_POSTPROCESSING)
├── HandlerError (E_HANDLER)
│   ├── HandlerNotFoundError (E_HANDLER_NOT_FOUND)
│   └── HandlerExecutionError (E_HANDLER_EXECUTION)
├── ServiceError (E_SERVICE)
│   ├── ImageServiceError (E_IMAGE_SERVICE)
│   ├── StorageError (E_STORAGE)
│   └── OCRError (E_OCR)
└── ChunkingError (E_CHUNKING)
```

---

## 에러 코드 상세

### E_CONFIGURATION — ConfigurationError

**원인**: 잘못된 설정값이 제공되었을 때

```python
from contextifier.config import ChunkingConfig

# 잘못된 전략 이름
ChunkingConfig(strategy="invalid")
# → ConfigurationError: Invalid chunking strategy 'invalid'
```

**해결**: 올바른 설정값을 사용하세요. `ChunkingConfig.strategy`는 `"recursive"`, `"sliding"`, `"hierarchical"` 중 하나여야 합니다.

---

### E_FILE_NOT_FOUND — FileNotFoundError

**원인**: 지정된 경로에 파일이 존재하지 않을 때

```python
processor.extract_text("nonexistent.pdf")
# → FileNotFoundError: File not found: nonexistent.pdf
```

**해결**: 파일 경로를 확인하세요. 상대경로/절대경로를 정확히 지정하세요.

---

### E_FILE_READ — FileReadError

**원인**: 파일 읽기 실패 (권한 부족, 파일 손상 등)

**해결**: 파일 접근 권한을 확인하세요. 파일이 손상되지 않았는지 확인하세요.

---

### E_UNSUPPORTED_FORMAT — UnsupportedFormatError

**원인**: 지원하지 않는 파일 확장자

```python
processor.extract_text("data.xyz")
# → UnsupportedFormatError: Unsupported format: xyz
```

**해결**: `processor.is_supported("xyz")`로 지원 여부를 사전 확인하세요.
지원 확장자 목록: `processor.supported_extensions`

---

### E_CONVERSION — ConversionError

**원인**: 바이너리 데이터를 포맷별 객체로 변환 실패 (파이프라인 Stage 1)

- 손상된 ZIP 구조 (DOCX, PPTX, XLSX, HWPX)
- ZIP bomb 감지 (압축 해제 크기 > 1GB)
- 잘못된 OLE2 구조 (DOC, PPT, XLS, HWP)

```python
# 컨텍스트에 stage, handler 정보가 포함됩니다
except ConversionError as e:
    print(e.context)  # {"stage": "convert", "handler": "DOCX Handler"}
```

**해결**: 파일이 손상되지 않았는지 확인하세요. ZIP 기반 포맷은 유효한 ZIP 파일이어야 합니다.

---

### E_PREPROCESSING — PreprocessingError

**원인**: 전처리 단계 실패 (파이프라인 Stage 2)

- 인코딩 감지 실패 (CSV, TXT)
- 잘못된 파일 구조

**해결**: 파일 인코딩을 확인하세요. `format_options`에서 인코딩 목록을 지정할 수 있습니다.

---

### E_EXTRACTION — ExtractionError

**원인**: 콘텐츠/메타데이터 추출 실패 (파이프라인 Stage 3-4)

**해결**: 파일이 올바른 포맷인지 확인하세요. 손상된 내부 구조가 원인일 수 있습니다.

---

### E_POSTPROCESSING — PostprocessingError

**원인**: 후처리/최종 조립 실패 (파이프라인 Stage 5)

**해결**: 일반적으로 내부 오류입니다. 재현 가능하면 이슈를 보고해주세요.

---

### E_HANDLER_NOT_FOUND — HandlerNotFoundError

**원인**: 요청한 확장자에 등록된 핸들러가 없을 때

```python
registry.get_handler("xyz")
# → HandlerNotFoundError: No handler for extension 'xyz'
```

**해결**: `registry.is_supported("xyz")`로 확인하세요. 커스텀 핸들러를 등록할 수 있습니다.

---

### E_HANDLER_EXECUTION — HandlerExecutionError

**원인**: 핸들러 실행 중 런타임 에러 (타임아웃 포함)

```python
# 타임아웃 예시
handler.process(file_context, timeout=5.0)
# → HandlerExecutionError: Processing timed out after 5.0s
```

**해결**: `timeout` 값을 늘리거나, 파일 크기를 확인하세요. 대형 파일은 `format_options`에서 리소스 제한 옵션을 설정하세요.

---

### E_IMAGE_SERVICE — ImageServiceError

**원인**: 이미지 처리 또는 저장 실패

**해결**: 이미지 저장 디렉토리에 쓰기 권한이 있는지 확인하세요. 디스크 공간을 확인하세요.

---

### E_STORAGE — StorageError

**원인**: 스토리지 백엔드 작업 실패 (로컬/S3/MinIO)

**해결**: 스토리지 설정과 접근 권한을 확인하세요.

---

### E_OCR — OCRError

**원인**: OCR 처리 실패

- API 키 인증 실패
- 네트워크 오류
- 이미지 대역폭 초과
- Tesseract 바이너리 미발견

**해결**: OCR 엔진 설정을 확인하세요. 자세한 내용은 [OCR 설정 가이드](ocr_guide.md)를 참조하세요.

---

### E_CHUNKING — ChunkingError

**원인**: 텍스트 청킹 실패

**해결**: 입력 텍스트와 청킹 설정을 확인하세요.

---

## 에러 처리 패턴

### 기본 에러 처리

```python
from contextifier import DocumentProcessor
from contextifier.errors import (
    ContextifierError,
    FileNotFoundError,
    UnsupportedFormatError,
)

processor = DocumentProcessor()

try:
    text = processor.extract_text("document.pdf")
except FileNotFoundError as e:
    print(f"파일 없음: {e.context.get('file_path')}")
except UnsupportedFormatError as e:
    print(f"미지원 포맷: {e.code}")
except ContextifierError as e:
    # 모든 Contextifier 예외를 포괄
    print(f"처리 실패 [{e.code}]: {e.message}")
    if e.cause:
        print(f"원인: {e.cause}")
```

### 컨텍스트 추가 (Fluent API)

```python
try:
    result = handler.process(file_context)
except ContextifierError as e:
    raise e.with_context(
        user_id="abc123",
        batch_id="batch-42",
    )
```

### 파이프라인 에러 정보

```python
from contextifier.errors import PipelineError

try:
    result = handler.process(file_context)
except PipelineError as e:
    print(f"실패 단계: {e.context.get('stage')}")
    print(f"핸들러: {e.context.get('handler')}")
```
