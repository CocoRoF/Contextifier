# Configuration Reference

> Contextifier v0.3.0 — 전체 설정 옵션 레퍼런스

## 개요

Contextifier는 **불변(frozen) 데이터클래스** 기반의 설정 시스템을 사용합니다.
모든 설정은 `ProcessingConfig` 루트 객체에 집중되며, 생성 후 수정 불가합니다.
변경이 필요하면 `with_*()` 메서드로 수정된 복사본을 만듭니다.

```python
from contextifier.config import ProcessingConfig

# 기본 설정 (모든 옵션이 기본값)
config = ProcessingConfig()

# 커스텀 설정
config = ProcessingConfig().with_chunking(chunk_size=2000, chunk_overlap=300)
```

---

## 설정 계층 구조

```
ProcessingConfig (루트)
├── TagConfig         — 구조 태그 형식 설정
├── ImageConfig       — 이미지 저장/처리 설정
├── ChartConfig       — 차트 포맷 설정
├── MetadataConfig    — 메타데이터 포맷 설정
├── TableConfig       — 테이블 출력 형식 설정
├── ChunkingConfig    — 텍스트 청킹 설정
├── OCRConfig         — OCR 엔진 설정
└── format_options    — 핸들러별 세부 설정 (dict)
```

---

## ProcessingConfig

루트 설정 객체. 모든 하위 설정을 포함합니다.

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
    format_options: Dict[str, Dict[str, Any]]
```

### Fluent 수정 메서드

| 메서드 | 설명 |
|--------|------|
| `with_tags(**kwargs)` | TagConfig 수정 |
| `with_images(**kwargs)` | ImageConfig 수정 |
| `with_charts(**kwargs)` | ChartConfig 수정 |
| `with_metadata(**kwargs)` | MetadataConfig 수정 |
| `with_tables(**kwargs)` | TableConfig 수정 |
| `with_chunking(**kwargs)` | ChunkingConfig 수정 |
| `with_ocr(**kwargs)` | OCRConfig 수정 |
| `with_format_option(format_name, **kwargs)` | 핸들러별 옵션 수정 |
| `get_format_option(format_name, key, default)` | 핸들러별 옵션 조회 |

### 직렬화

```python
# Dict 변환
d = config.to_dict()

# Dict에서 복원
config = ProcessingConfig.from_dict(d)
```

---

## TagConfig

구조 태그(페이지, 슬라이드, 시트, 이미지, 차트, 메타데이터)의 접두사/접미사를 제어합니다.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `page_prefix` | `str` | `"[Page Number: "` | 페이지 태그 접두사 |
| `page_suffix` | `str` | `"]"` | 페이지 태그 접미사 |
| `slide_prefix` | `str` | `"[Slide Number: "` | 슬라이드 태그 접두사 |
| `slide_suffix` | `str` | `"]"` | 슬라이드 태그 접미사 |
| `sheet_prefix` | `str` | `"[Sheet: "` | 시트 태그 접두사 |
| `sheet_suffix` | `str` | `"]"` | 시트 태그 접미사 |
| `image_prefix` | `str` | `"[Image:"` | 이미지 태그 접두사 |
| `image_suffix` | `str` | `"]"` | 이미지 태그 접미사 |
| `chart_prefix` | `str` | `"[chart]"` | 차트 블록 접두사 |
| `chart_suffix` | `str` | `"[/chart]"` | 차트 블록 접미사 |
| `metadata_prefix` | `str` | `"[Document-Metadata]"` | 메타데이터 블록 접두사 |
| `metadata_suffix` | `str` | `"[/Document-Metadata]"` | 메타데이터 블록 접미사 |

```python
config = ProcessingConfig().with_tags(
    page_prefix="<page>",
    page_suffix="</page>",
    image_prefix="![Image:",
    image_suffix="]",
)
```

---

## ImageConfig

이미지 추출 및 저장 설정.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `directory_path` | `str` | `"temp/images"` | 이미지 저장 디렉토리 |
| `naming_strategy` | `NamingStrategy` | `HASH` | 파일 명명 전략: `HASH`, `SEQUENTIAL`, `ORIGINAL` |
| `default_format` | `str` | `"png"` | 기본 이미지 포맷 |
| `quality` | `int` | `95` | JPEG 품질 (1-100) |
| `skip_duplicate` | `bool` | `True` | 해시 기반 중복 이미지 스킵 |
| `storage_type` | `StorageType` | `LOCAL` | 스토리지 백엔드: `LOCAL`, `S3`, `MINIO` |
| `max_file_size_mb` | `Optional[float]` | `None` | 최대 이미지 파일 크기 (MB). 초과 시 스킵 + 경고 |

```python
config = ProcessingConfig().with_images(
    directory_path="output/images",
    max_file_size_mb=10.0,
    naming_strategy=NamingStrategy.SEQUENTIAL,
)
```

---

## ChartConfig

차트 데이터 포맷 설정.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `use_html_table` | `bool` | `True` | HTML 테이블 형식으로 차트 데이터 포맷 |
| `include_chart_type` | `bool` | `True` | 차트 타입 포함 |
| `include_chart_title` | `bool` | `True` | 차트 제목 포함 |

```python
config = ProcessingConfig().with_charts(
    use_html_table=False,
    include_chart_type=True,
)
```

---

## MetadataConfig

메타데이터 포맷 설정.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `language` | `str` | `"ko"` | 메타데이터 출력 언어: `"ko"`, `"en"` |
| `date_format` | `str` | `"%Y-%m-%d %H:%M:%S"` | 날짜 형식 문자열 |
| `indent` | `str` | `"  "` | 메타데이터 들여쓰기 문자열 |

```python
config = ProcessingConfig().with_metadata(
    language="en",
    date_format="%Y/%m/%d",
)
```

---

## TableConfig

테이블 출력 형식 설정.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `output_format` | `OutputFormat` | `HTML` | 출력 형식: `HTML`, `MARKDOWN`, `TEXT` |
| `clean_whitespace` | `bool` | `True` | 셀 내 여백 정리 |
| `preserve_merged_cells` | `bool` | `True` | 병합 셀 보존 (rowspan/colspan) |

```python
from contextifier.types import OutputFormat

config = ProcessingConfig().with_tables(
    output_format=OutputFormat.MARKDOWN,
    preserve_merged_cells=True,
)
```

---

## ChunkingConfig

텍스트 청킹 설정.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `chunk_size` | `int` | `1000` | 청크 최대 문자 수 |
| `chunk_overlap` | `int` | `200` | 청크 간 겹침 문자 수 |
| `preserve_tables` | `bool` | `True` | 테이블 구조 보존 (분할 방지) |
| `include_position_metadata` | `bool` | `False` | 위치 메타데이터 포함 |
| `strategy` | `str` | `"recursive"` | 청킹 전략: `"recursive"`, `"sliding"`, `"hierarchical"` |

```python
config = ProcessingConfig().with_chunking(
    chunk_size=2000,
    chunk_overlap=300,
    strategy="recursive",
    preserve_tables=True,
)
```

### 자동 전략 선택

실제로는 `TextChunker`가 파일 확장자와 콘텐츠를 분석하여 최적 전략을 자동 선택합니다:

| 전략 | 우선순위 | 적용 조건 |
|------|:--------:|-----------|
| `TableChunkingStrategy` | 5 | 스프레드시트 (csv, tsv, xlsx, xls) |
| `PageChunkingStrategy` | 10 | 페이지 경계 태그 존재 시 (PDF, PPTX 등) |
| `ProtectedChunkingStrategy` | 20 | HTML 테이블 / 보호 영역 존재 시 |
| `PlainChunkingStrategy` | 100 | 폴백 (재귀 분할) |

---

## OCRConfig

OCR 엔진 설정.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `enabled` | `bool` | `False` | OCR 활성화 여부 |
| `provider` | `Optional[str]` | `None` | 엔진 이름: `"openai"`, `"anthropic"`, `"gemini"`, `"bedrock"`, `"vllm"`, `"tesseract"` |
| `prompt` | `Optional[str]` | `None` | 커스텀 OCR 프롬프트 (None = 기본 프롬프트) |
| `prompt_language` | `str` | `"ko"` | 프롬프트 출력 언어: `"ko"`, `"en"` |

```python
config = ProcessingConfig().with_ocr(
    enabled=True,
    provider="openai",
    prompt_language="en",
)
```

> **참고**: OCR 엔진 인스턴스는 `DocumentProcessor` 생성자에 직접 전달합니다.
> `OCRConfig`는 프롬프트/언어 설정만 담당합니다.

---

## format_options (핸들러별 세부 설정)

핸들러마다 고유한 설정을 `format_options` 딕셔너리로 관리합니다.

### 사용법

```python
# 설정 시
config = ProcessingConfig().with_format_option("pdf", render_dpi=200)

# 조회 시
dpi = config.get_format_option("pdf", "render_dpi", default=150)
```

### 지원 옵션 전체

#### PDF

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `render_dpi` | `int` | `150` | 이미지 렌더링 DPI |
| `min_image_size` | `int` | `100` | 최소 이미지 크기 (px) |
| `min_image_area` | `int` | `10000` | 최소 이미지 면적 (px²) |
| `table_size` | `int` | `50` | 테이블 감지 최소 크기 |

#### PPTX

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `max_group_depth` | `int` | `20` | 그룹 셰이프 최대 재귀 깊이 |

#### DOC

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `min_text_fragment_length` | `int` | `20` | 최소 텍스트 프래그먼트 길이 |

#### CSV

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `max_rows` | `int` | `10000` | 최대 처리 행 수 (초과 시 truncated) |
| `delimiter_candidates` | `list` | `[",", "\t", "\|", ";"]` | 구분자 감지 후보 목록 |
| `encodings` | `list` | `["utf-8", "cp949", ...]` | 인코딩 감지 후보 목록 |

#### TSV

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `max_rows` | `int` | `10000` | 최대 처리 행 수 |

#### XLSX

| 키 | 타입 | 기본값 | 설명 |
|----|------|--------|------|
| `read_only` | `bool` | `False` | openpyxl read_only 모드 (대형 파일 메모리 절약) |

---

## 설정 조합 예시

### 고성능 대형 파일 처리

```python
config = (
    ProcessingConfig()
    .with_images(max_file_size_mb=5.0)
    .with_format_option("csv", max_rows=50000)
    .with_format_option("xlsx", read_only=True)
    .with_format_option("pdf", render_dpi=72)
)
```

### AI/LLM 통합용

```python
config = (
    ProcessingConfig()
    .with_chunking(chunk_size=2000, chunk_overlap=200)
    .with_tables(output_format=OutputFormat.MARKDOWN)
    .with_metadata(language="en")
    .with_ocr(prompt_language="en")
)
```

### 최소 출력 (텍스트만)

```python
config = (
    ProcessingConfig()
    .with_tags(
        metadata_prefix="", metadata_suffix="",
        chart_prefix="", chart_suffix="",
    )
    .with_images(skip_duplicate=True)
)

text = processor.extract_text("doc.pdf", extract_metadata=False)
```
