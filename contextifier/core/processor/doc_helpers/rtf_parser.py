# service/document_processor/processor/doc_helpers/rtf_parser.py
"""
RTF Parser - RTF 파일 바이너리 직접 파싱 (리팩터링 버전)

LibreOffice 없이 RTF 파일을 직접 분석하여:
- 텍스트 추출 (원래 위치 유지)
- 테이블을 HTML로 변환 (인라인 배치)
- 병합 셀 처리 (clmgf/clmrg/clvmgf/clvmrg)
- 메타데이터 추출
- 이미지 추출

RTF 1.5+ 스펙 기반 구현

이 파일은 기능별로 분리된 모듈들을 조합하여 사용합니다:
- rtf_constants.py: 상수 정의
- rtf_models.py: 데이터 모델 (RTFCellInfo, RTFTable, RTFContentPart, RTFDocument)
- rtf_decoder.py: 인코딩/디코딩 유틸리티
- rtf_text_cleaner.py: 텍스트 정리 유틸리티
- rtf_metadata_extractor.py: 메타데이터 추출
- rtf_table_extractor.py: 테이블 추출/파싱
- rtf_content_extractor.py: 인라인 콘텐츠 추출
- rtf_region_finder.py: 제외 영역 탐색
- rtf_bin_processor.py: 바이너리 전처리
"""
import logging
from typing import Optional, Set

from contextifier.core.functions.img_processor import ImageProcessor

# 모델 임포트 (외부에서 사용할 수 있도록)
from contextifier.core.processor.doc_helpers.rtf_models import (
    RTFCellInfo,
    RTFTable,
    RTFContentPart,
    RTFDocument,
)

# 디코더 임포트
from contextifier.core.processor.doc_helpers.rtf_decoder import (
    detect_encoding,
    decode_content,
    decode_hex_escapes,
)

# Text cleaner import
from contextifier.core.processor.doc_helpers.rtf_text_cleaner import (
    clean_rtf_text,
    remove_shprslt_blocks,
)

# Metadata extractor import
from contextifier.core.processor.doc_helpers.rtf_metadata_extractor import (
    DOCMetadataExtractor,
    RTFSourceInfo,
)

# 테이블 추출기 임포트
from contextifier.core.processor.doc_helpers.rtf_table_extractor import (
    extract_tables_with_positions,
)

# 콘텐츠 추출기 임포트
from contextifier.core.processor.doc_helpers.rtf_content_extractor import (
    extract_inline_content,
    extract_text_legacy,
)

# 바이너리 처리기 임포트
from contextifier.core.processor.doc_helpers.rtf_bin_processor import (
    preprocess_rtf_binary,
)

logger = logging.getLogger("document-processor")


class RTFParser:
    """
    RTF 파일 파서 (리팩터링 버전)

    RTF 바이너리를 직접 파싱하여 텍스트, 테이블, 메타데이터를 추출합니다.

    기능별로 분리된 모듈들을 조합하여 사용합니다.
    """

    def __init__(
        self,
        encoding: str = "cp949",
        processed_images: Optional[Set[str]] = None,
        image_processor: ImageProcessor = None
    ):
        """
        Args:
            encoding: 기본 인코딩 (한글 문서는 보통 cp949)
            processed_images: 처리된 이미지 해시 집합 (중복 방지)
            image_processor: 이미지 처리기
        """
        self.encoding = encoding
        self.processed_images = processed_images if processed_images is not None else set()
        self.image_processor = image_processor
        self.document = RTFDocument(encoding=encoding)

        # 파싱 상태
        self._content: str = ""
        self._raw_content: bytes = b""  # 원본 바이너리
        self._image_tags = {}  # 위치 -> 이미지 태그

    def parse(self, content: bytes) -> RTFDocument:
        """
        RTF 바이너리를 파싱합니다.

        Args:
            content: RTF 파일 바이트 데이터

        Returns:
            파싱된 RTFDocument 객체
        """
        self._raw_content = content

        # 바이너리 데이터 전처리 (\bin 태그 처리, 이미지 추출)
        clean_content, self._image_tags = preprocess_rtf_binary(
            content,
            processed_images=self.processed_images,
            image_processor=self.image_processor
        )

        # 이미지 태그를 문서에 저장 (유효한 태그만)
        self.document.image_tags = [
            tag for tag in self._image_tags.values()
            if tag and tag.strip() and '/uploads/.' not in tag
        ]

        # 인코딩 감지 및 디코딩
        self.encoding = detect_encoding(clean_content, self.encoding)
        self._content = decode_content(clean_content, self.encoding)

        # \shprslt block removal (prevent duplicate content)
        self._content = remove_shprslt_blocks(self._content)

        # Metadata extraction using DOCMetadataExtractor
        extractor = DOCMetadataExtractor()
        source_info = RTFSourceInfo(content=self._content, encoding=self.encoding)
        metadata_obj = extractor.extract(source_info)
        self.document.metadata = metadata_obj.to_dict()

        # 테이블 추출 (위치 정보 포함)
        tables, table_regions = extract_tables_with_positions(
            self._content,
            self.encoding
        )
        self.document.tables = tables

        # 인라인 콘텐츠 추출 (테이블 위치 유지)
        self.document.content_parts = extract_inline_content(
            self._content,
            table_regions,
            self.encoding
        )

        # 호환성을 위해 기존 text_content도 설정
        self.document.text_content = extract_text_legacy(
            self._content,
            self.encoding
        )

        return self.document


def parse_rtf(
    content: bytes,
    encoding: str = "cp949",
    processed_images: Optional[Set[str]] = None,
    image_processor: ImageProcessor = None
) -> RTFDocument:
    """
    RTF 파일을 파싱합니다.

    바이너리 이미지 데이터를 로컬에 저장하고 태그로 변환합니다.

    Args:
        content: RTF 파일 바이트 데이터
        encoding: 기본 인코딩
        processed_images: 처리된 이미지 해시 집합 (중복 방지, optional)
        image_processor: 이미지 처리기

    Returns:
        파싱된 RTFDocument 객체
    """
    parser = RTFParser(
        encoding=encoding,
        processed_images=processed_images,
        image_processor=image_processor
    )
    return parser.parse(content)


# 하위 호환성을 위한 re-export
__all__ = [
    'RTFParser',
    'RTFDocument',
    'RTFTable',
    'RTFCellInfo',
    'RTFContentPart',
    'parse_rtf',
]
