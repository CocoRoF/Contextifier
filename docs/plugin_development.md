# Plugin Development Guide

> Contextifier v0.3.0 — 커스텀 핸들러 개발 가이드

## 개요

Contextifier는 `BaseHandler`를 상속하여 새로운 파일 포맷을 지원할 수 있습니다.
핸들러는 5단계 파이프라인을 **반드시** 따르며, 각 단계에 대응하는 컴포넌트를 구현합니다.

```
process(file_context) → ExtractionResult
    ├── Stage 1: Converter.convert()        — 바이너리 → 포맷 객체
    ├── Stage 2: Preprocessor.preprocess()  — 전처리/정리
    ├── Stage 3: MetadataExtractor.extract() — 메타데이터 추출
    ├── Stage 4: ContentExtractor.extract_all() — 콘텐츠 추출
    └── Stage 5: Postprocessor.postprocess() — 최종 조립
```

---

## 디렉토리 구조

```
contextifier/handlers/
└── myformat/
    ├── __init__.py
    ├── handler.py           # MyFormatHandler (BaseHandler 상속)
    ├── converter.py         # MyFormatConverter (BaseConverter 상속)
    ├── preprocessor.py      # MyFormatPreprocessor (BasePreprocessor 상속)
    ├── metadata_extractor.py # MyFormatMetadataExtractor (BaseMetadataExtractor 상속)
    └── content_extractor.py # MyFormatContentExtractor (BaseContentExtractor 상속)
```

---

## Step 1: Converter 구현

바이너리 데이터를 포맷별 작업 객체로 변환합니다.

```python
# contextifier/handlers/myformat/converter.py
from typing import Any
from contextifier.pipeline.converter import BaseConverter
from contextifier.types import FileContext
from contextifier.errors import ConversionError


class MyFormatConverter(BaseConverter):
    def convert(self, file_context: FileContext, **kwargs: Any) -> Any:
        """바이너리 데이터를 포맷 객체로 변환."""
        data = file_context.get("file_data", b"")
        if not data:
            raise ConversionError("Empty file data")

        # 포맷별 파싱 로직
        try:
            parsed = parse_my_format(data)  # 실제 파싱 구현
            return parsed
        except Exception as e:
            raise ConversionError(f"Conversion failed: {e}", cause=e)

    def get_format_name(self) -> str:
        return "myformat"

    def validate(self, file_context: FileContext) -> bool:
        """파일 데이터 유효성 사전 검증 (선택)."""
        data = file_context.get("file_data", b"")
        return len(data) > 0

    def close(self, converted: Any) -> None:
        """리소스 정리 (선택). 파일 핸들 등을 닫습니다."""
        if hasattr(converted, 'close'):
            converted.close()
```

> **팁**: 변환이 불필요한 포맷은 `NullConverter`를 사용하세요:
> ```python
> from contextifier.pipeline.converter import NullConverter
> ```

---

## Step 2: Preprocessor 구현

변환된 객체를 정리/변환하여 `PreprocessedData`를 생성합니다.

```python
# contextifier/handlers/myformat/preprocessor.py
from typing import Any
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.types import PreprocessedData


class MyFormatPreprocessor(BasePreprocessor):
    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """변환 결과를 전처리."""
        # 정리/정규화 로직
        cleaned = clean_data(converted_data)

        return PreprocessedData(
            content=cleaned,         # 메인 콘텐츠 (Stage 3-4에서 사용)
            resources={},            # 추출된 리소스 (이미지, 차트 등)
            properties={             # 핸들러별 메타 속성
                "page_count": cleaned.page_count,
            },
        )

    def get_format_name(self) -> str:
        return "myformat"
```

> **팁**: 전처리가 불필요하면 `NullPreprocessor`를 사용하세요.

---

## Step 3: MetadataExtractor 구현

문서 메타데이터(제목, 작성자, 날짜 등)를 추출합니다.

```python
# contextifier/handlers/myformat/metadata_extractor.py
from typing import Any, Optional
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.types import DocumentMetadata


class MyFormatMetadataExtractor(BaseMetadataExtractor):
    def extract(self, content: Any, **kwargs: Any) -> Optional[DocumentMetadata]:
        """메타데이터 추출."""
        if not hasattr(content, 'metadata'):
            return None

        return DocumentMetadata(
            title=content.metadata.get("title"),
            author=content.metadata.get("author"),
            created_date=content.metadata.get("created"),
            page_count=content.metadata.get("pages"),
        )

    def get_format_name(self) -> str:
        return "myformat"
```

> **팁**: 메타데이터가 없는 포맷은 `NullMetadataExtractor`를 사용하세요.

---

## Step 4: ContentExtractor 구현

텍스트, 테이블, 이미지, 차트를 추출합니다.

```python
# contextifier/handlers/myformat/content_extractor.py
from typing import Any, List, Optional
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.types import (
    ChartData,
    DocumentMetadata,
    PreprocessedData,
    TableData,
)


class MyFormatContentExtractor(BaseContentExtractor):
    def extract_text(
        self,
        preprocessed: PreprocessedData,
        *,
        extract_metadata_result: Optional[DocumentMetadata] = None,
        **kwargs: Any,
    ) -> str:
        """텍스트 추출 (필수)."""
        content = preprocessed.content
        # 텍스트 추출 로직
        return extract_text_from_content(content)

    def extract_tables(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[TableData]:
        """테이블 추출 (선택). 미구현 시 빈 리스트 반환."""
        # TableData 생성 예시:
        # TableData(
        #     rows=[[TableCell(value="A1"), TableCell(value="B1")], ...],
        #     headers=["Column A", "Column B"],
        # )
        return []

    def extract_images(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[str]:
        """이미지 추출 (선택). ImageService로 저장 후 경로 반환."""
        saved_paths = []
        for img_data in preprocessed.resources.get("images", []):
            if self._image_service:
                tag = self._image_service.save(img_data, "image.png")
                if tag:
                    saved_paths.append(tag)
        return saved_paths

    def extract_charts(
        self, preprocessed: PreprocessedData, **kwargs: Any
    ) -> List[ChartData]:
        """차트 추출 (선택). 미구현 시 빈 리스트 반환."""
        return []

    def get_format_name(self) -> str:
        return "myformat"
```

---

## Step 5: Handler 조립

5개 컴포넌트를 조합하는 핸들러를 작성합니다.

```python
# contextifier/handlers/myformat/handler.py
from typing import FrozenSet
from contextifier.handlers.base import BaseHandler
from contextifier.pipeline.converter import BaseConverter
from contextifier.pipeline.preprocessor import BasePreprocessor
from contextifier.pipeline.metadata_extractor import BaseMetadataExtractor
from contextifier.pipeline.content_extractor import BaseContentExtractor
from contextifier.pipeline.postprocessor import BasePostprocessor, DefaultPostprocessor

from .converter import MyFormatConverter
from .preprocessor import MyFormatPreprocessor
from .metadata_extractor import MyFormatMetadataExtractor
from .content_extractor import MyFormatContentExtractor


class MyFormatHandler(BaseHandler):
    """Custom handler for .myformat files."""

    @property
    def supported_extensions(self) -> FrozenSet[str]:
        # 문서 핸들러는 반드시 1개 확장자만
        return frozenset({"myformat"})

    @property
    def handler_name(self) -> str:
        return "MyFormat Handler"

    def create_converter(self) -> BaseConverter:
        return MyFormatConverter()

    def create_preprocessor(self) -> BasePreprocessor:
        return MyFormatPreprocessor()

    def create_metadata_extractor(self) -> BaseMetadataExtractor:
        return MyFormatMetadataExtractor()

    def create_content_extractor(self) -> BaseContentExtractor:
        return MyFormatContentExtractor(
            image_service=self._image_service,
            tag_service=self._tag_service,
            chart_service=self._chart_service,
            table_service=self._table_service,
            config=self._config,
        )

    def create_postprocessor(self) -> BasePostprocessor:
        return DefaultPostprocessor(
            config=self._config,
            metadata_service=self._metadata_service,
            tag_service=self._tag_service,
        )
```

---

## Step 6: 핸들러 등록

### 방법 A: 직접 등록

```python
from contextifier import DocumentProcessor
from contextifier.config import ProcessingConfig
from my_package.handler import MyFormatHandler

processor = DocumentProcessor()
processor.registry.register(MyFormatHandler)

# 이제 .myformat 파일 처리 가능
text = processor.extract_text("document.myformat")
```

### 방법 B: Entry Points (패키지 배포 시)

`pyproject.toml`에 entry point를 등록하면 자동으로 발견됩니다:

```toml
[project.entry-points."contextifier.handlers"]
myformat = "my_package.handler:MyFormatHandler"
```

`register_defaults()` 호출 시 자동으로 등록됩니다.

---

## 규칙 및 제약

### 필수 규칙

1. **하나의 확장자, 하나의 핸들러**: 문서 포맷 핸들러는 `supported_extensions`에 정확히 1개의 확장자만 반환해야 합니다. 카테고리 핸들러(Text, Image)만 예외.

2. **process() 오버라이드 금지**: `process()`와 `extract_text()`는 `@final`로 선언되어 있어 오버라이드할 수 없습니다.

3. **균일한 생성자**: 핸들러 생성자에 추가 매개변수를 넣지 마세요. 핸들러별 설정은 `config.format_options`를 사용하세요:
   ```python
   def create_preprocessor(self):
       my_opts = dict(self._config.format_options.get("myformat", {}))
       threshold = my_opts.get("threshold", 50)
       return MyFormatPreprocessor(threshold=threshold)
   ```

4. **서비스 주입 패턴**: 이미지/태그/차트/테이블/메타데이터 서비스는 생성자에서 주입받습니다. 직접 생성하지 마세요.

### 선택 사항

- **위임 지원**: `_check_delegation()`을 오버라이드하여 다른 핸들러에 처리를 위임할 수 있습니다.
- **타임아웃**: `process(file_context, timeout=30.0)`으로 처리 제한 시간을 설정할 수 있습니다.

---

## 테스트 작성

```python
# tests/unit/handlers/test_myformat.py
import pytest
from unittest.mock import MagicMock
from contextifier.config import ProcessingConfig
from my_package.handler import MyFormatHandler


@pytest.fixture
def handler():
    config = ProcessingConfig()
    return MyFormatHandler(config=config)


class TestMyFormatHandler:
    def test_supported_extensions(self, handler):
        assert handler.supported_extensions == frozenset({"myformat"})

    def test_handler_name(self, handler):
        assert handler.handler_name == "MyFormat Handler"

    def test_process_valid_file(self, handler):
        file_context = {
            "file_name": "test.myformat",
            "file_extension": "myformat",
            "file_data": b"valid data...",
            "file_stream": None,
        }
        result = handler.process(file_context)
        assert result.text  # 텍스트가 추출되었는지 확인

    def test_process_empty_file(self, handler):
        from contextifier.errors import ConversionError
        file_context = {
            "file_name": "empty.myformat",
            "file_extension": "myformat",
            "file_data": b"",
            "file_stream": None,
        }
        with pytest.raises(ConversionError):
            handler.process(file_context)
```

---

## 참고

- [ARCHITECTURE.md](../contextifier/ARCHITECTURE.md) — 전체 아키텍처 명세
- [CONTRIBUTING.md](../CONTRIBUTING.md) — 코딩 규칙
- [Error Codes](error_codes.md) — 예외 계층 및 에러 코드
- [Configuration](configuration.md) — 설정 레퍼런스
