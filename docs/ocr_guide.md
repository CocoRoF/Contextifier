# OCR Setup Guide

> Contextifier v0.3.0 — OCR 엔진 설정 가이드

## 개요

Contextifier는 6개의 OCR 엔진을 지원합니다:

| 엔진 | 유형 | 의존성 | 특징 |
|------|------|--------|------|
| **OpenAI** | 클라우드 (Vision LLM) | `langchain-openai` | GPT-4o 기반, 최고 정확도 |
| **Anthropic** | 클라우드 (Vision LLM) | `langchain-anthropic` | Claude 기반 |
| **Google Gemini** | 클라우드 (Vision LLM) | `langchain-google-genai` | Gemini Pro Vision |
| **AWS Bedrock** | 클라우드 (Vision LLM) | `langchain-aws` | AWS 관리형 |
| **vLLM** | 셀프호스팅 | `langchain-openai` | 자체 서버 |
| **Tesseract** | 로컬 | `pytesseract`, `Pillow` | 오프라인, 무료 |

---

## 빠른 시작

### 1. OpenAI (권장)

```bash
pip install langchain-openai
```

```python
from contextifier import DocumentProcessor
from contextifier.ocr.engines import OpenAIOCREngine

# API 키로 간편 생성
ocr = OpenAIOCREngine.from_api_key(
    "sk-your-api-key",
    model="gpt-4o",           # 기본값
    temperature=0.0,           # 기본값
)

processor = DocumentProcessor(ocr_engine=ocr)
text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

### 2. Anthropic

```bash
pip install langchain-anthropic
```

```python
from contextifier.ocr.engines import AnthropicOCREngine

ocr = AnthropicOCREngine.from_api_key(
    "sk-ant-your-api-key",
    model="claude-sonnet-4-20250514",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 3. Google Gemini

```bash
pip install langchain-google-genai
```

```python
from contextifier.ocr.engines import GeminiOCREngine

ocr = GeminiOCREngine.from_api_key(
    "your-google-api-key",
    model="gemini-2.5-pro",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 4. AWS Bedrock

```bash
pip install langchain-aws
```

```python
from contextifier.ocr.engines import BedrockOCREngine
from langchain_aws import ChatBedrock

# LangChain 클라이언트 직접 주입
client = ChatBedrock(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1",
)
ocr = BedrockOCREngine(client)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 5. vLLM (셀프호스팅)

```bash
pip install langchain-openai
```

```python
from contextifier.ocr.engines import VLLMOCREngine
from langchain_openai import ChatOpenAI

# vLLM 서버를 OpenAI 호환 엔드포인트로 연결
client = ChatOpenAI(
    model="your-model-name",
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM은 API 키 불필요
)
ocr = VLLMOCREngine(client)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6. Tesseract (로컬/오프라인)

```bash
pip install pytesseract Pillow
```

**Tesseract 바이너리 설치:**

| OS | 설치 방법 |
|----|----------|
| **Ubuntu/Debian** | `sudo apt install tesseract-ocr tesseract-ocr-kor` |
| **macOS** | `brew install tesseract tesseract-lang` |
| **Windows** | [UB Mannheim 설치파일](https://github.com/UB-Mannheim/tesseract/wiki) 다운로드 후 PATH 등록 |

```python
from contextifier.ocr.engines import TesseractOCREngine

# 기본 영어
ocr = TesseractOCREngine()

# 한국어 + 영어
ocr = TesseractOCREngine(lang="kor+eng")

# Tesseract 경로 직접 지정 (Windows)
ocr = TesseractOCREngine(
    lang="kor+eng",
    tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    config="--psm 6",  # 단일 텍스트 블록 모드
)

processor = DocumentProcessor(ocr_engine=ocr)
```

---

## OCR 프롬프트 커스터마이징

### 기본 프롬프트

기본 프롬프트는 다국어를 지원합니다:
- `"ko"`: 한국어 출력 (기본값)
- `"en"`: 영어 출력

```python
from contextifier.config import ProcessingConfig

# 프롬프트 언어 변경
config = ProcessingConfig().with_ocr(prompt_language="en")
```

### 커스텀 프롬프트

```python
custom_prompt = """
이 이미지에서 텍스트를 추출하세요.
- 표가 있으면 HTML 테이블로 변환
- 수식이 있으면 LaTeX로 변환
- 출력은 한국어로
"""

ocr = OpenAIOCREngine.from_api_key("sk-...", prompt=custom_prompt)
```

---

## OCR 병렬 처리

대량의 이미지가 포함된 문서에서 OCR 속도를 향상시킬 수 있습니다:

```python
from contextifier.ocr.processor import OCRProcessor

# 기본: 순차 처리 (max_workers=1)
processor = OCRProcessor(engine=ocr, config=config)

# 병렬 처리: ThreadPoolExecutor 사용
processor = OCRProcessor(engine=ocr, config=config, max_workers=4)
```

> **참고**: 클라우드 API의 rate limit에 주의하세요. `max_workers`를 API 제한에 맞게 설정하세요.

---

## OCR 동작 원리

1. **텍스트 추출**: 핸들러가 문서를 처리하면서 이미지를 발견하면 `[Image: path/to/image.png]` 태그를 삽입
2. **OCR 호출**: `ocr_processing=True`일 때, OCRProcessor가 텍스트에서 `[Image: ...]` 태그를 인식
3. **이미지 → 텍스트**: 각 이미지에 대해 OCR 엔진이 텍스트를 추출
4. **태그 치환**: 이미지 태그가 추출된 텍스트로 교체

```
[원본 텍스트]
제1장 서론
[Image: temp/images/page_1.png]
제2장 본론
...

[OCR 처리 후]
제1장 서론
[Figure]
이미지에서 추출된 텍스트 내용...
[/Figure]
제2장 본론
...
```

---

## 포맷별 OCR 적용

| 포맷 | OCR 적용 시나리오 |
|------|-------------------|
| **PDF** (스캔) | `needs_ocr=True` 감지 시 페이지를 이미지로 렌더링 → 이미지 태그 삽입 → OCR |
| **이미지** (jpg, png 등) | 이미지 자체가 이미지 태그 → OCR |
| **DOCX/PPTX** | 임베디드 이미지에 이미지 태그 삽입 → OCR |
| **HWP/HWPX** | 임베디드 이미지에 이미지 태그 삽입 → OCR |

---

## 트러블슈팅

### Tesseract를 찾을 수 없음

```
pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or not in PATH
```

→ Tesseract 바이너리를 설치하고 PATH에 등록하거나, `tesseract_cmd` 매개변수로 경로를 직접 지정하세요.

### API 인증 실패

```
AuthenticationError: Incorrect API key provided
```

→ API 키가 올바른지, 유효기간이 만료되지 않았는지 확인하세요.

### Rate Limit 초과

```
RateLimitError: Rate limit exceeded
```

→ `max_workers`를 줄이거나, API 사용량을 확인하세요. 재시도 로직은 LangChain 클라이언트에서 제공합니다.

### OCR 결과가 비어 있음

→ 이미지가 실제로 텍스트를 포함하는지 확인하세요.
→ Tesseract의 경우 `lang` 매개변수가 올바른지 확인하세요 (`kor+eng` 등).
→ 이미지 해상도가 너무 낮으면 (< 150 DPI) 정확도가 떨어질 수 있습니다.
