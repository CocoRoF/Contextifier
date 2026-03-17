# Contributing to Contextifier

Contextifier에 기여해 주셔서 감사합니다! 이 문서는 기여 가이드라인을 설명합니다.

## 개발 환경 설정

### 1. 저장소 클론 및 환경 생성

```bash
git clone https://github.com/your-org/contextifier.git
cd contextifier

python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

pip install -e ".[dev]"
```

### 2. 프로젝트 구조

```
contextifier_new/           # v2 메인 패키지
├── document_processor.py   # Facade (공개 API)
├── config.py               # ProcessingConfig
├── types.py                # 공유 타입
├── errors.py               # 예외 계층
├── handlers/               # 14개 포맷 핸들러
├── pipeline/               # 5-Stage ABC
├── services/               # 공유 서비스
├── chunking/               # 청킹 서브시스템
└── ocr/                    # OCR 서브시스템
```

## 코딩 컨벤션

### 일반 규칙

- **Python 3.12+** 문법 사용
- Type hints 필수 (모든 public API)
- docstring 필수 (Google style)
- `from __future__ import annotations` 모든 모듈 상단에 추가

### 아키텍처 규칙

1. **모든 핸들러는 5단계 파이프라인을 따라야 합니다:**
   - `Converter` → `Preprocessor` → `MetadataExtractor` → `ContentExtractor` → `Postprocessor`
   - `BaseHandler.process()`가 순서를 강제하므로, 각 단계만 구현하면 됩니다.

2. **서비스를 직접 생성하지 마세요:**
   - `TagService`, `ImageService` 등은 `DocumentProcessor`가 생성하여 주입합니다.
   - 핸들러에서는 `self._services["tag_service"]` 등으로 접근합니다.

3. **설정은 `ProcessingConfig`를 통해 전달하세요:**
   - 하드코딩된 매직 넘버 금지
   - 새로운 설정이 필요하면 적절한 `*Config` 클래스에 필드를 추가하세요.

4. **Facade 패턴 준수:**
   - 외부 사용자가 접근하는 API는 `DocumentProcessor`뿐입니다.
   - 내부 모듈을 직접 import하도록 안내하지 마세요 (OCR 엔진 제외).

## 새 핸들러 추가 가이드

### 1. 디렉토리 생성

```
contextifier_new/handlers/myformat/
├── __init__.py
├── converter.py
├── preprocessor.py
├── metadata_extractor.py
├── content_extractor.py
└── postprocessor.py
```

### 2. 각 파이프라인 단계 구현

```python
# converter.py
from contextifier_new.pipeline.converter import Converter

class MyFormatConverter(Converter):
    def convert(self, file_context, **kwargs):
        # Binary → Format-specific object
        return parsed_object
```

### 3. 핸들러 등록

`contextifier_new/handlers/registry.py`의 `register_defaults()`에 추가:

```python
from contextifier_new.handlers.myformat import MyFormatHandler
self.register(MyFormatHandler, extensions=["myf", "myformat"])
```

## 커밋 컨벤션

```
feat: 새 기능 추가
fix: 버그 수정
docs: 문서 변경
refactor: 리팩토링 (기능 변경 없음)
test: 테스트 추가/수정
chore: 빌드/설정 변경
```

예시:
```
feat(handler): add EPUB handler with full pipeline
fix(chunking): preserve table structure in protected strategy
docs: update QUICKSTART with batch processing example
```

## Pull Request 가이드

1. `main` 브랜치에서 feature 브랜치 생성
2. 변경사항 구현 및 테스트
3. PR 설명에 변경 이유와 테스트 결과 포함
4. 리뷰 후 squash merge

## 이슈 보고

버그를 발견하면 다음 정보를 포함해 주세요:

- Python 버전
- OS 및 버전
- 입력 파일 형식 및 크기
- 에러 메시지 전문
- 재현 코드 (가능하다면)

## 라이선스

기여하신 코드는 프로젝트의 Apache License 2.0 하에 배포됩니다.
