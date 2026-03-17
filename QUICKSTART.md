# Contextifier v2 — 상세 사용 가이드

Contextifier v2의 전체 API와 사용 방법을 상세히 설명합니다.

## 목차

1. [설치](#1-설치)
2. [기본 사용법](#2-기본-사용법)
3. [설정 시스템 (ProcessingConfig)](#3-설정-시스템-processingconfig)
4. [텍스트 추출 API](#4-텍스트-추출-api)
5. [청킹 API](#5-청킹-api)
6. [OCR 연동](#6-ocr-연동)
7. [태그 커스터마이징](#7-태그-커스터마이징)
8. [테이블 처리](#8-테이블-처리)
9. [이미지 처리](#9-이미지-처리)
10. [메타데이터 추출](#10-메타데이터-추출)
11. [포맷별 가이드](#11-포맷별-가이드)
12. [배치 처리](#12-배치-처리)
13. [RAG 연동](#13-rag-연동)
14. [에러 핸들링](#14-에러-핸들링)
15. [전체 API 레퍼런스](#15-전체-api-레퍼런스)

---

## 1. 설치

### pip

```bash
pip install contextifier
```

### uv

```bash
uv add contextifier
```

### 선택 의존성

```bash
# OCR 엔진별 의존성 (필요한 것만 설치)
pip install langchain-openai       # OpenAI OCR
pip install langchain-anthropic    # Anthropic OCR
pip install langchain-google-genai # Google Gemini OCR
pip install langchain-aws          # AWS Bedrock OCR
pip install langchain-community    # vLLM OCR
```

### 시스템 의존성 (선택)

| 도구 | 용도 | 설치 |
|------|------|------|
| LibreOffice | DOC, PPT, RTF → DOCX/PPTX/HTML 변환 | [libreoffice.org](https://www.libreoffice.org/) |
| Poppler | PDF → 이미지 변환 | `apt install poppler-utils` / `brew install poppler` |

---

## 2. 기본 사용법

### 2.1 가장 간단한 사용

```python
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()

# 텍스트 추출
text = processor.extract_text("document.pdf")
print(text)
```

### 2.2 추출 + 청킹 한 번에

```python
result = processor.extract_chunks("document.pdf")

print(f"총 {len(result)}개 청크")
for i, chunk in enumerate(result, 1):
    print(f"--- Chunk {i} ---")
    print(chunk[:200])
```

### 2.3 지원 확장자 확인

```python
processor = DocumentProcessor()

# 특정 확장자 지원 여부
print(processor.is_supported("pdf"))   # True
print(processor.is_supported("xyz"))   # False

# 전체 지원 확장자 목록
print(processor.supported_extensions)
# frozenset({'pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', ...})
```

---

## 3. 설정 시스템 (ProcessingConfig)

Contextifier v2의 모든 동작은 `ProcessingConfig`로 제어합니다.
설정은 frozen dataclass로 **불변(immutable)**이며, 변경 시 새 인스턴스를 반환합니다.

### 3.1 설정 계층 구조

```
ProcessingConfig (루트)
├── TagConfig       — 태그 prefix/suffix 설정
├── ImageConfig     — 이미지 저장 경로/형식/네이밍
├── ChartConfig     — 차트 포맷팅 설정
├── MetadataConfig  — 메타데이터 언어/형식
├── TableConfig     — 테이블 출력 형식 (HTML/MD/Text)
├── ChunkingConfig  — 청킹 크기/오버랩/전략
├── OCRConfig       — OCR 활성화/프로바이더
└── format_options  — 포맷별 추가 옵션
```

### 3.2 기본 설정 (제로 설정)

```python
from contextifier_new.config import ProcessingConfig

# 모든 기본값 사용 — 설정 없이 바로 사용 가능
config = ProcessingConfig()
processor = DocumentProcessor(config=config)
```

### 3.3 직접 생성

```python
from contextifier_new.config import (
    ProcessingConfig,
    TagConfig,
    ImageConfig,
    ChartConfig,
    MetadataConfig,
    TableConfig,
    ChunkingConfig,
    OCRConfig,
)

config = ProcessingConfig(
    tags=TagConfig(
        page_prefix="<page>",
        page_suffix="</page>",
        image_prefix="[IMG:",
        image_suffix="]",
    ),
    images=ImageConfig(
        directory_path="output/images",
        naming_strategy=NamingStrategy.HASH,
        quality=95,
    ),
    tables=TableConfig(
        output_format=OutputFormat.HTML,
        preserve_merged_cells=True,
    ),
    chunking=ChunkingConfig(
        chunk_size=2000,
        chunk_overlap=300,
        preserve_tables=True,
    ),
    metadata=MetadataConfig(
        language="ko",
    ),
)

processor = DocumentProcessor(config=config)
```

### 3.4 Fluent Builder 패턴

기존 설정을 기반으로 일부만 변경할 때 `with_*()` 메서드를 사용합니다.
원본은 변경되지 않고, 새 `ProcessingConfig` 인스턴스가 반환됩니다.

```python
config = ProcessingConfig()

# 태그만 변경
config2 = config.with_tags(page_prefix="<!-- Page ", page_suffix=" -->")

# 청킹만 변경
config3 = config.with_chunking(chunk_size=3000, chunk_overlap=500)

# 체이닝 가능
config4 = (
    config
    .with_tags(page_prefix="[P:", page_suffix="]")
    .with_chunking(chunk_size=1500)
    .with_tables(output_format=OutputFormat.MARKDOWN)
    .with_images(directory_path="my_images/")
)
```

### 3.5 포맷별 옵션

특정 포맷에만 적용되는 옵션을 설정합니다.

```python
config = ProcessingConfig().with_format_option(
    "pdf",
    table_detection="lattice",
    ocr_fallback=True,
)

# 옵션 조회
val = config.get_format_option("pdf", "table_detection", default="stream")
```

### 3.6 설정 직렬화/역직렬화

```python
# Dict로 변환 (JSON 저장용)
d = config.to_dict()

import json
with open("config.json", "w") as f:
    json.dump(d, f, indent=2)

# Dict에서 복원
with open("config.json") as f:
    d = json.load(f)

config = ProcessingConfig.from_dict(d)
```

### 3.7 개별 설정 클래스 기본값 참조

| 설정 클래스 | 주요 필드 | 기본값 |
|-------------|-----------|--------|
| **TagConfig** | `page_prefix` / `page_suffix` | `"[Page Number: "` / `"]"` |
| | `slide_prefix` / `slide_suffix` | `"[Slide Number: "` / `"]"` |
| | `sheet_prefix` / `sheet_suffix` | `"[Sheet: "` / `"]"` |
| | `image_prefix` / `image_suffix` | `"[Image:"` / `"]"` |
| | `chart_prefix` / `chart_suffix` | `"[chart]"` / `"[/chart]"` |
| | `metadata_prefix` / `metadata_suffix` | `"<Document-Metadata>"` / `"</Document-Metadata>"` |
| **ImageConfig** | `directory_path` | `"temp/images"` |
| | `naming_strategy` | `NamingStrategy.HASH` |
| | `default_format` | `"png"` |
| | `quality` | `95` |
| | `skip_duplicate` | `True` |
| **ChartConfig** | `use_html_table` | `True` |
| | `include_chart_type` | `True` |
| | `include_chart_title` | `True` |
| **MetadataConfig** | `language` | `"ko"` |
| | `date_format` | `"%Y-%m-%d %H:%M:%S"` |
| **TableConfig** | `output_format` | `OutputFormat.HTML` |
| | `clean_whitespace` | `True` |
| | `preserve_merged_cells` | `True` |
| **ChunkingConfig** | `chunk_size` | `1000` |
| | `chunk_overlap` | `200` |
| | `preserve_tables` | `True` |
| | `strategy` | `"recursive"` |
| **OCRConfig** | `enabled` | `False` |
| | `provider` | `None` |
| | `prompt` | `None` (내장 기본 프롬프트 사용) |

---

## 4. 텍스트 추출 API

### 4.1 `extract_text()` — 텍스트 문자열 반환

```python
text = processor.extract_text(
    file_path="document.pdf",        # 파일 경로 (필수)
    file_extension=None,             # 확장자 오버라이드 (선택)
    extract_metadata=True,           # 메타데이터 포함 여부
    ocr_processing=False,            # OCR 후처리 여부
)
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `file_path` | `str \| Path` | — | 문서 파일 경로 |
| `file_extension` | `str \| None` | `None` | 확장자 오버라이드 (`.pdf` 형태 또는 `pdf`) |
| `extract_metadata` | `bool` | `True` | 메타데이터를 텍스트에 포함할지 여부 |
| `ocr_processing` | `bool` | `False` | 이미지 태그를 OCR로 후처리할지 여부 |

**반환값:** `str` — 추출된 텍스트

**확장자 오버라이드 활용 예:**

```python
# 확장자가 없는 파일을 PDF로 처리
text = processor.extract_text("report_v2", file_extension="pdf")

# .dat 파일을 Excel로 처리
text = processor.extract_text("data.dat", file_extension="xlsx")
```

### 4.2 `process()` — 구조화된 결과 반환

텍스트뿐 아니라 메타데이터, 테이블, 이미지, 차트 정보를 모두 포함한 `ExtractionResult`를 반환합니다.

```python
from contextifier_new.types import ExtractionResult

result: ExtractionResult = processor.process("document.pdf")

print(result.text)            # 추출된 전체 텍스트
print(result.metadata)        # DocumentMetadata 객체
print(result.tables)          # 추출된 TableData 리스트
print(result.images)          # 이미지 경로 리스트
print(result.charts)          # ChartData 리스트
```

**`ExtractionResult` 필드:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `text` | `str` | 추출된 전체 텍스트 |
| `metadata` | `DocumentMetadata \| None` | 문서 메타데이터 |
| `tables` | `List[TableData]` | 추출된 테이블 목록 |
| `images` | `List[str]` | 저장된 이미지 경로 목록 |
| `charts` | `List[ChartData]` | 추출된 차트 목록 |

---

## 5. 청킹 API

### 5.1 `chunk_text()` — 텍스트를 청크로 분할

이미 추출한 텍스트 문자열을 직접 청킹합니다.

```python
text = processor.extract_text("document.pdf")

# 기본 청킹 (config 설정 사용)
chunks = processor.chunk_text(text)
print(f"{len(chunks)}개 청크")

# 크기/오버랩 오버라이드
chunks = processor.chunk_text(
    text,
    chunk_size=2000,
    chunk_overlap=300,
)

# 위치 메타데이터 포함 (Chunk 객체 반환)
chunks_with_meta = processor.chunk_text(
    text,
    include_position_metadata=True,
)
for chunk in chunks_with_meta:
    print(f"Index: {chunk.metadata.chunk_index}, Text: {chunk.text[:50]}...")
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `text` | `str` | — | 분할할 텍스트 |
| `chunk_size` | `int \| None` | `None` (config 값) | 청크당 최대 문자 수 |
| `chunk_overlap` | `int \| None` | `None` (config 값) | 청크 간 겹침 문자 수 |
| `file_extension` | `str` | `""` | 소스 파일 확장자 (전략 선택에 사용) |
| `preserve_tables` | `bool` | `True` | 테이블 구조 보존 여부 |
| `include_position_metadata` | `bool` | `False` | 위치 메타데이터 포함 여부 |

**반환값:**
- `include_position_metadata=False` → `List[str]`
- `include_position_metadata=True` → `List[Chunk]`

### 5.2 `extract_chunks()` — 추출 + 청킹 한 번에

파일에서 텍스트를 추출하고 바로 청킹까지 수행합니다. **가장 많이 사용하는 API**입니다.

```python
result = processor.extract_chunks(
    file_path="document.pdf",
    chunk_size=1500,
    chunk_overlap=200,
    preserve_tables=True,
    include_position_metadata=True,
)

# ChunkResult 기본 사용
print(f"총 {len(result)}개 청크")
print(result[0])          # 인덱스 접근
for chunk in result:      # 이터레이션
    print(chunk[:100])

# 위치 메타데이터 접근
if result.has_metadata:
    for chunk_obj in result.chunks_with_metadata:
        print(f"  chunk_index={chunk_obj.metadata.chunk_index}")
        print(f"  page={chunk_obj.metadata.page_number}")

# 소스 파일 정보
print(result.source_file)
```

### 5.3 `ChunkResult.save_to_md()` — Markdown 파일로 저장

각 청크를 개별 Markdown 파일로 저장합니다. RAG 인덱싱에 유용합니다.

```python
result = processor.extract_chunks("report.pdf", chunk_size=1000)

# 기본 저장
created = result.save_to_md("output/chunks")
print(f"{len(created)}개 파일 생성")
# output/chunks/chunk_0001.md, chunk_0002.md, ...

# 커스텀 파일명 접두사 및 구분자
created = result.save_to_md(
    "output/report_chunks",
    filename_prefix="report",
    separator="===",
)
# output/report_chunks/report_0001.md, report_0002.md, ...
```

**저장된 파일 예시 (메타데이터 포함 시):**

```markdown
<!-- chunk_index: 0 -->
<!-- page: 1 -->
---

첫 번째 청크의 텍스트 내용...
```

### 5.4 청킹 전략

Contextifier v2는 콘텐츠 특성에 따라 4가지 전략 중 **자동으로 최적 전략을 선택**합니다.

| 전략 | 우선순위 | 적용 조건 | 설명 |
|------|----------|-----------|------|
| **TableChunkingStrategy** | 5 (최고) | 스프레드시트(xlsx, xls, csv, tsv) | 시트/테이블 경계 기반 분할 |
| **PageChunkingStrategy** | 10 | 페이지 태그 존재 시 | 페이지 경계 존중 분할 |
| **ProtectedChunkingStrategy** | 20 | HTML 테이블/Protected Region 존재 시 | Protected region 보존 분할 |
| **PlainChunkingStrategy** | 100 (최저) | 기본 폴백 | 재귀적 문자 분할 |

> 우선순위 숫자가 **작을수록** 먼저 선택됩니다. 해당 전략이 적용 불가능하면 다음 전략으로 넘어갑니다.

---

## 6. OCR 연동

Contextifier v2는 5종의 Vision LLM 기반 OCR 엔진을 지원합니다.
이미지 태그(`[Image:...]`)가 포함된 텍스트에서 해당 이미지를 읽어 텍스트로 변환합니다.

### 6.1 OpenAI

```python
from contextifier_new.ocr.engines import OpenAIOCREngine

# 간편 생성
ocr = OpenAIOCREngine.from_api_key(
    "sk-...",
    model="gpt-4o",        # 기본값
    temperature=0.0,       # 기본값
    max_tokens=None,       # 선택
)

processor = DocumentProcessor(ocr_engine=ocr)
text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

### 6.2 Anthropic

```python
from contextifier_new.ocr.engines import AnthropicOCREngine

ocr = AnthropicOCREngine.from_api_key(
    "sk-ant-...",
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.3 Google Gemini

```python
from contextifier_new.ocr.engines import GeminiOCREngine

ocr = GeminiOCREngine.from_api_key(
    "AIza...",
    model="gemini-2.0-flash",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.4 AWS Bedrock

```python
from contextifier_new.ocr.engines import BedrockOCREngine

ocr = BedrockOCREngine.from_api_key(
    "AKIA...",                          # AWS Access Key
    aws_secret_access_key="...",
    region_name="us-east-1",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.5 vLLM (로컬 / 자체 호스팅)

```python
from contextifier_new.ocr.engines import VLLMOCREngine

ocr = VLLMOCREngine.from_api_key(
    "dummy-key",                  # vLLM은 인증 불필요 시 아무 값
    model="llava-1.5-7b",
    base_url="http://localhost:8000/v1",
)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.6 LangChain 클라이언트 직접 전달

이미 LangChain 클라이언트가 있다면 직접 전달할 수 있습니다:

```python
from langchain_openai import ChatOpenAI
from contextifier_new.ocr.engines import OpenAIOCREngine

llm = ChatOpenAI(model="gpt-4o", api_key="sk-...")
ocr = OpenAIOCREngine(llm_client=llm)

processor = DocumentProcessor(ocr_engine=ocr)
```

### 6.7 커스텀 OCR 프롬프트

```python
custom_prompt = """
이미지의 텍스트를 추출하세요.
표가 있으면 HTML 테이블로 변환하세요.
"""

ocr = OpenAIOCREngine.from_api_key(
    "sk-...",
    prompt=custom_prompt,
)

# 또는 생성 후 변경
ocr.prompt = custom_prompt
```

---

## 7. 태그 커스터마이징

Contextifier는 추출된 텍스트에 구조적 태그를 삽입합니다.
태그 형식은 `TagConfig`로 자유롭게 변경할 수 있습니다.

### 7.1 기본 태그 출력 예시

```
<Document-Metadata>
  제목: 연간 보고서 2024
  작성자: 홍길동
  생성일: 2024-01-15 10:30:00
</Document-Metadata>

[Page Number: 1]

안녕하세요, 이것은 첫 번째 페이지입니다.

[Image: temp/images/abc123.png]

[chart]
<table>
  <tr><th>분기</th><th>매출</th></tr>
  <tr><td>Q1</td><td>100M</td></tr>
</table>
[/chart]

[Page Number: 2]
...
```

### 7.2 XML 스타일 태그

```python
config = ProcessingConfig(
    tags=TagConfig(
        page_prefix="<page number=\"",
        page_suffix="\">",
        image_prefix="<image src=\"",
        image_suffix="\" />",
        metadata_prefix="<metadata>",
        metadata_suffix="</metadata>",
        chart_prefix="<chart>",
        chart_suffix="</chart>",
    ),
)
```

출력:
```
<metadata>
  ...
</metadata>
<page number="1">
텍스트...
<image src="temp/images/abc123.png" />
```

### 7.3 Markdown 스타일 태그

```python
config = ProcessingConfig(
    tags=TagConfig(
        page_prefix="## Page ",
        page_suffix="",
        image_prefix="![image](",
        image_suffix=")",
    ),
)
```

---

## 8. 테이블 처리

### 8.1 출력 형식 선택

```python
from contextifier_new.config import ProcessingConfig, TableConfig
from contextifier_new.types import OutputFormat

# HTML (기본값) — 셀 병합, 구조 완벽 보존
config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.HTML))

# Markdown — 단순 테이블 (병합 셀 미지원)
config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.MARKDOWN))

# Plain Text — 정렬된 텍스트 테이블
config = ProcessingConfig(tables=TableConfig(output_format=OutputFormat.TEXT))
```

### 8.2 HTML 출력 예시

```html
<table>
  <tr>
    <th colspan="2">2024년 매출 요약</th>
  </tr>
  <tr>
    <td>Q1</td>
    <td>100억</td>
  </tr>
  <tr>
    <td>Q2</td>
    <td>120억</td>
  </tr>
</table>
```

### 8.3 Markdown 출력 예시

```markdown
| 분기 | 매출 |
|------|------|
| Q1 | 100억 |
| Q2 | 120억 |
```

---

## 9. 이미지 처리

### 9.1 이미지 추출 설정

```python
from contextifier_new.config import ProcessingConfig, ImageConfig
from contextifier_new.types import NamingStrategy, StorageType

config = ProcessingConfig(
    images=ImageConfig(
        directory_path="output/images",       # 저장 경로
        naming_strategy=NamingStrategy.HASH,   # 파일명: 해시 (중복 제거)
        default_format="png",                 # 저장 형식
        quality=95,                           # JPEG 품질 (1-100)
        skip_duplicate=True,                  # 동일 이미지 스킵
        storage_type=StorageType.LOCAL,        # 로컬 스토리지
    ),
)
```

### 9.2 네이밍 전략

| 전략 | 파일명 예시 | 설명 |
|------|-------------|------|
| `HASH` | `a3f2c1d8.png` | 콘텐츠 해시 — 중복 자동 제거 |
| `UUID` | `550e8400-e29b.png` | 랜덤 UUID |
| `SEQUENTIAL` | `img_001.png` | 순번 |
| `TIMESTAMP` | `20240115_103000.png` | 타임스탬프 |

---

## 10. 메타데이터 추출

### 10.1 사용법

```python
# 메타데이터 포함 텍스트 추출 (기본)
text = processor.extract_text("report.docx", extract_metadata=True)

# 메타데이터 제외
text = processor.extract_text("report.docx", extract_metadata=False)

# 구조화된 메타데이터 접근
result = processor.process("report.docx")
meta = result.metadata

print(meta.title)           # "연간 보고서 2024"
print(meta.author)          # "홍길동"
print(meta.create_time)     # datetime(2024, 1, 15, 10, 30, 0)
print(meta.page_count)      # 42
print(meta.to_dict())       # dict 형태
```

### 10.2 메타데이터 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `title` | `str` | 문서 제목 |
| `subject` | `str` | 주제 |
| `author` | `str` | 작성자 |
| `keywords` | `str` | 키워드 |
| `comments` | `str` | 설명/코멘트 |
| `last_saved_by` | `str` | 마지막 수정자 |
| `create_time` | `datetime` | 생성 일시 |
| `last_saved_time` | `datetime` | 마지막 저장 일시 |
| `page_count` | `int` | 총 페이지 수 |
| `word_count` | `int` | 총 단어 수 |
| `category` | `str` | 문서 카테고리 |
| `revision` | `str` | 리비전 번호 |

### 10.3 메타데이터 언어 설정

```python
# 한국어 (기본값)
config = ProcessingConfig(metadata=MetadataConfig(language="ko"))
# 출력: "제목: ...", "작성자: ...", "생성일: ..."

# 영어
config = ProcessingConfig(metadata=MetadataConfig(language="en"))
# 출력: "Title: ...", "Author: ...", "Created: ..."
```

---

## 11. 포맷별 가이드

### 11.1 PDF

```python
# 기본 PDF 핸들러 (대부분의 PDF에 적합)
text = processor.extract_text("document.pdf")

# 이미지 기반 PDF (OCR 필요)
ocr = OpenAIOCREngine.from_api_key("sk-...")
processor = DocumentProcessor(ocr_engine=ocr)
text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

### 11.2 Word 문서 (DOCX/DOC)

```python
# DOCX — 직접 처리
text = processor.extract_text("document.docx")

# DOC — LibreOffice로 DOCX 변환 후 처리 (자동)
text = processor.extract_text("legacy.doc")
```

> **참고:** DOC 파일은 OLE, HTML, DOCX, RTF 등 다양한 내부 형식을 가질 수 있습니다.
> Contextifier가 자동으로 감지하여 최적의 방법으로 변환합니다.

### 11.3 PowerPoint (PPTX/PPT)

```python
text = processor.extract_text("presentation.pptx")
# 각 슬라이드별 [Slide Number: N] 태그 포함
# 노트, 차트, 표 모두 추출
```

### 11.4 Excel (XLSX/XLS)

```python
text = processor.extract_text("data.xlsx")
# 각 시트별 [Sheet: SheetName] 태그 포함
# 차트 데이터 [chart]...[/chart] 태그 포함
```

### 11.5 한글 (HWP/HWPX)

```python
text = processor.extract_text("document.hwp")   # HWP 5.0
text = processor.extract_text("document.hwpx")  # HWPX
```

> **참고:** HWP 5.0 형식만 지원합니다. HWP 3.0 등 구 버전은 미지원입니다.

### 11.6 CSV/TSV

```python
text = processor.extract_text("data.csv")
text = processor.extract_text("data.tsv")
# 테이블 형식으로 변환
```

### 11.7 텍스트/코드/설정 파일

```python
# 자동 인코딩 감지 (UTF-8, EUC-KR 등)
text = processor.extract_text("readme.txt")
text = processor.extract_text("script.py")
text = processor.extract_text("config.yaml")
```

### 11.8 이미지

```python
# 이미지 파일은 OCR 엔진이 필요합니다
ocr = OpenAIOCREngine.from_api_key("sk-...")
processor = DocumentProcessor(ocr_engine=ocr)

text = processor.extract_text("chart.png", ocr_processing=True)
```

---

## 12. 배치 처리

### 12.1 디렉토리 내 모든 파일 처리

```python
from pathlib import Path
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()

input_dir = Path("documents/")
output_dir = Path("output/")
output_dir.mkdir(exist_ok=True)

for file_path in input_dir.iterdir():
    ext = file_path.suffix.lstrip(".")
    if not processor.is_supported(ext):
        continue

    try:
        text = processor.extract_text(str(file_path))
        out_path = output_dir / f"{file_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"✓ {file_path.name}")
    except Exception as e:
        print(f"✗ {file_path.name}: {e}")
```

### 12.2 배치 청킹

```python
processor = DocumentProcessor()

files = ["report.pdf", "data.xlsx", "memo.docx"]

all_chunks = []
for f in files:
    result = processor.extract_chunks(f, chunk_size=1000)
    for chunk in result.chunks:
        all_chunks.append({
            "text": chunk,
            "source": f,
        })

print(f"총 {len(all_chunks)}개 청크 생성")
```

---

## 13. RAG 연동

### 13.1 LangChain 연동

```python
from contextifier_new import DocumentProcessor
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

processor = DocumentProcessor()

# 문서 처리 + 청킹
result = processor.extract_chunks(
    "knowledge_base.pdf",
    chunk_size=1000,
    chunk_overlap=200,
    include_position_metadata=True,
)

# LangChain Document 변환
documents = []
for chunk_obj in result.chunks_with_metadata:
    doc = Document(
        page_content=chunk_obj.text,
        metadata={
            "source": result.source_file,
            "chunk_index": chunk_obj.metadata.chunk_index,
            "page_number": chunk_obj.metadata.page_number,
        },
    )
    documents.append(doc)

# 벡터 스토어에 인덱싱
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 검색
results = vectorstore.similarity_search("매출 현황", k=3)
```

### 13.2 파일 저장 후 인덱싱

```python
# Markdown 파일로 저장 → 외부 인덱싱 파이프라인에서 사용
result = processor.extract_chunks("report.pdf", chunk_size=1000)
result.save_to_md("rag_index/report/")
```

---

## 14. 에러 핸들링

### 14.1 예외 계층

```
ContextifierError (기본 예외)
├── FileNotFoundError        — 파일이 존재하지 않음
├── UnsupportedFormatError   — 지원하지 않는 파일 형식
├── HandlerNotFoundError     — 확장자에 맞는 핸들러 없음
├── ConversionError          — 파일 변환 실패
├── ExtractionError          — 텍스트 추출 실패
└── OCRError                 — OCR 처리 실패
```

### 14.2 사용 예

```python
from contextifier_new.errors import (
    ContextifierError,
    UnsupportedFormatError,
    FileNotFoundError as ContextifyFileNotFoundError,
)

try:
    text = processor.extract_text("document.xyz")
except UnsupportedFormatError as e:
    print(f"미지원 형식: {e}")
except ContextifyFileNotFoundError as e:
    print(f"파일 없음: {e}")
except ContextifierError as e:
    print(f"처리 오류: {e}")
```

---

## 15. 전체 API 레퍼런스

### DocumentProcessor

```python
class DocumentProcessor:
    def __init__(
        self,
        config: ProcessingConfig | None = None,
        *,
        ocr_engine: BaseOCREngine | None = None,
    ) -> None: ...

    def extract_text(
        self,
        file_path: str | Path,
        file_extension: str | None = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs,
    ) -> str: ...

    def process(
        self,
        file_path: str | Path,
        file_extension: str | None = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        **kwargs,
    ) -> ExtractionResult: ...

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        file_extension: str = "",
        preserve_tables: bool = True,
        include_position_metadata: bool = False,
    ) -> list[str] | list[Chunk]: ...

    def extract_chunks(
        self,
        file_path: str | Path,
        file_extension: str | None = None,
        *,
        extract_metadata: bool = True,
        ocr_processing: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        preserve_tables: bool = True,
        include_position_metadata: bool = False,
        **kwargs,
    ) -> ChunkResult: ...

    def is_supported(self, extension: str) -> bool: ...

    @property
    def supported_extensions(self) -> frozenset: ...

    @property
    def config(self) -> ProcessingConfig: ...

    @property
    def registry(self) -> HandlerRegistry: ...
```

### ChunkResult

```python
@dataclass
class ChunkResult:
    chunks: list[str]
    chunks_with_metadata: list[Chunk] | None
    source_file: str | None

    @property
    def has_metadata(self) -> bool: ...

    def save_to_md(
        self,
        output_dir: str | Path,
        *,
        filename_prefix: str = "chunk",
        separator: str = "---",
    ) -> list[str]: ...

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> str: ...
    def __iter__(self) -> Iterator[str]: ...
```

### ProcessingConfig

```python
@dataclass(frozen=True)
class ProcessingConfig:
    tags: TagConfig
    images: ImageConfig
    charts: ChartConfig
    metadata: MetadataConfig
    tables: TableConfig
    chunking: ChunkingConfig
    ocr: OCRConfig
    format_options: dict[str, dict[str, Any]]

    def with_tags(self, **kwargs) -> ProcessingConfig: ...
    def with_images(self, **kwargs) -> ProcessingConfig: ...
    def with_charts(self, **kwargs) -> ProcessingConfig: ...
    def with_metadata(self, **kwargs) -> ProcessingConfig: ...
    def with_tables(self, **kwargs) -> ProcessingConfig: ...
    def with_chunking(self, **kwargs) -> ProcessingConfig: ...
    def with_ocr(self, **kwargs) -> ProcessingConfig: ...
    def with_format_option(self, format_name: str, **kwargs) -> ProcessingConfig: ...
    def get_format_option(self, format_name: str, key: str, default=None) -> Any: ...

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> ProcessingConfig: ...
```

### OCR Engines

```python
# 공통 인터페이스
class BaseOCREngine(ABC):
    def __init__(self, llm_client, *, prompt=None): ...
    def convert_image_to_text(self, image_path: str) -> str | None: ...

# 엔진별 편의 생성자
OpenAIOCREngine.from_api_key(api_key, *, model="gpt-4o", ...)
AnthropicOCREngine.from_api_key(api_key, *, model="claude-sonnet-4-20250514", ...)
GeminiOCREngine.from_api_key(api_key, *, model="gemini-2.0-flash", ...)
BedrockOCREngine.from_api_key(api_key, *, aws_secret_access_key, region_name, model, ...)
VLLMOCREngine.from_api_key(api_key, *, model, base_url, ...)
```
