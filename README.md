# Contextifier v2

**Contextifier**는 다양한 형식의 문서를 AI가 이해할 수 있는 구조화된 텍스트로 변환하는 Python 문서 처리 라이브러리입니다.
모든 문서 포맷에 대해 **동일한 5단계 파이프라인**을 적용하여, 일관된 결과를 보장합니다.

## 주요 기능

- **광범위한 포맷 지원**: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HWP, HWPX, RTF, CSV, TSV, TXT, MD, HTML, 이미지, 코드 파일 등 80+ 확장자
- **지능형 텍스트 추출**: 문서 구조(제목, 표, 이미지 위치) 보존 및 메타데이터 자동 추출
- **테이블 처리**: HTML/Markdown/Text 형식으로 테이블 변환, 병합 셀(`rowspan`/`colspan`) 지원
- **OCR 통합**: OpenAI, Anthropic, Google Gemini, AWS Bedrock, vLLM 5종 Vision LLM 엔진 지원
- **스마트 청킹**: 테이블 보존, Protected Region 인식, 페이지 경계 존중 등 4가지 전략 자동 선택
- **불변 설정 시스템**: frozen dataclass 기반 `ProcessingConfig`로 모든 동작 제어

## 설치

```bash
pip install contextifier
```

또는

```bash
uv add contextifier
```

## 빠른 시작

### 1. 기본 텍스트 추출

```python
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()
text = processor.extract_text("document.pdf")
print(text)
```

### 2. 추출 + 청킹 한 번에

```python
from contextifier_new import DocumentProcessor

processor = DocumentProcessor()
result = processor.extract_chunks("document.pdf")

for i, chunk in enumerate(result.chunks, 1):
    print(f"Chunk {i}: {chunk[:100]}...")

# Markdown 파일로 저장
result.save_to_md("output/chunks")
```

### 3. 설정 커스터마이징

```python
from contextifier_new import DocumentProcessor
from contextifier_new.config import ProcessingConfig, ChunkingConfig, TagConfig

config = ProcessingConfig(
    tags=TagConfig(page_prefix="<page>", page_suffix="</page>"),
    chunking=ChunkingConfig(chunk_size=2000, chunk_overlap=300),
)

processor = DocumentProcessor(config=config)
text = processor.extract_text("report.xlsx")
```

### 4. OCR 연동

```python
from contextifier_new import DocumentProcessor
from contextifier_new.ocr.engines import OpenAIOCREngine

ocr = OpenAIOCREngine.from_api_key("sk-...", model="gpt-4o")
processor = DocumentProcessor(ocr_engine=ocr)

text = processor.extract_text("scanned.pdf", ocr_processing=True)
```

## 지원 형식

| 카테고리 | 확장자 | 비고 |
|----------|--------|------|
| **문서** | `.pdf`, `.docx`, `.doc`, `.hwp`, `.hwpx`, `.rtf` | HWP 5.0+, HWPX 지원 |
| **프레젠테이션** | `.pptx`, `.ppt` | 슬라이드/노트/차트 추출 |
| **스프레드시트** | `.xlsx`, `.xls`, `.csv`, `.tsv` | 다중 시트, 수식, 차트 |
| **텍스트** | `.txt`, `.md`, `.log`, `.rst` | 자동 인코딩 감지 |
| **웹** | `.html`, `.htm`, `.xhtml` | 테이블/구조 보존 |
| **코드** | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs` 등 20+ | 언어별 하이라이팅 |
| **설정** | `.json`, `.yaml`, `.toml`, `.ini`, `.xml`, `.env` | 구조 보존 |
| **이미지** | `.jpg`, `.png`, `.gif`, `.bmp`, `.webp`, `.tiff` | OCR 엔진 필요 |

## 아키텍처

```
contextifier_new/
├── document_processor.py     # Facade: 유일한 공개 진입점
├── config.py                 # 불변 설정 시스템 (ProcessingConfig)
├── types.py                  # 공유 타입/Enum/TypedDict
├── errors.py                 # 통합 예외 계층
│
├── handlers/                 # 14개 포맷별 핸들러
│   ├── base.py               #   BaseHandler — 5단계 파이프라인 강제
│   ├── registry.py           #   HandlerRegistry — 확장자 → 핸들러 매핑
│   ├── pdf/                  #   PDF (default)
│   ├── pdf_plus/             #   PDF (advanced: 테이블 감지, 복잡 레이아웃)
│   ├── docx/ doc/ pptx/ ppt/ #   오피스 문서
│   ├── xlsx/ xls/ csv/       #   스프레드시트/데이터
│   ├── hwp/ hwpx/            #   한글 문서
│   ├── rtf/ text/            #   RTF/텍스트/코드/설정
│   └── image/                #   이미지 (OCR 연동)
│
├── pipeline/                 # 5-Stage 파이프라인 ABC
│   ├── converter.py          #   Stage 1: Binary → Format Object
│   ├── preprocessor.py       #   Stage 2: 전처리
│   ├── metadata_extractor.py #   Stage 3: 메타데이터 추출
│   ├── content_extractor.py  #   Stage 4: 텍스트/표/이미지 추출
│   └── postprocessor.py      #   Stage 5: 최종 조립 및 정리
│
├── services/                 # 공유 서비스 (DI)
│   ├── tag_service.py        #   페이지/슬라이드/시트 태그 생성
│   ├── image_service.py      #   이미지 저장/태그/중복 제거
│   ├── chart_service.py      #   차트 데이터 포맷팅
│   ├── table_service.py      #   테이블 HTML/MD 변환
│   ├── metadata_service.py   #   메타데이터 포맷팅
│   └── storage/              #   스토리지 백엔드 (Local, MinIO, S3, ...)
│
├── chunking/                 # 청킹 서브시스템
│   ├── chunker.py            #   TextChunker — 전략 자동 선택/실행
│   ├── constants.py          #   Protected region 패턴
│   └── strategies/           #   4가지 청킹 전략
│       ├── plain_strategy.py     # 재귀 분할 (기본)
│       ├── table_strategy.py     # 시트/테이블 기반
│       ├── page_strategy.py      # 페이지 경계 기반
│       └── protected_strategy.py # Protected region 보존
│
└── ocr/                      # OCR 서브시스템 (선택사항)
    ├── base.py               #   BaseOCREngine ABC
    ├── processor.py          #   OCRProcessor — 태그 검출 + 엔진 호출
    └── engines/              #   5종 엔진 구현
        ├── openai_engine.py
        ├── anthropic_engine.py
        ├── gemini_engine.py
        ├── bedrock_engine.py
        └── vllm_engine.py
```

## 시스템 요구사항

- **Python** 3.12+
- 필수 의존성은 `pyproject.toml`에 자동 포함
- **선택 의존성**: LibreOffice (DOC/PPT/RTF 변환), Poppler (PDF 이미지 추출)

## 문서

| 문서 | 내용 |
|------|------|
| [QUICKSTART.md](QUICKSTART.md) | 상세 사용 가이드 및 전체 API 레퍼런스 |
| [Process Logic.md](Process%20Logic.md) | 핸들러별 처리 흐름 다이어그램 |
| [ARCHITECTURE.md](contextifier_new/ARCHITECTURE.md) | 내부 아키텍처 상세 문서 |
| [CHANGELOG.md](CHANGELOG.md) | 버전별 변경 사항 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 기여 가이드라인 |

## 라이선스

Apache License 2.0 — [LICENSE](LICENSE) 참조

## 기여

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.
