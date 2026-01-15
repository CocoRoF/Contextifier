# libs/core/document_processor_new.py
"""
DocumentProcessor - 신규 문서 처리 클래스

Contextify 라이브러리의 메인 문서 처리 클래스입니다.
다양한 문서 형식(PDF, DOCX, PPT, Excel, HWP 등)에서 텍스트를 추출하고,
청킹을 수행하는 통합 인터페이스를 제공합니다.

이 클래스는 라이브러리 사용 시 권장되는 진입점입니다.

사용 예시:
    from libs.core.document_processor_new import DocumentProcessor

    # 인스턴스 생성
    processor = DocumentProcessor()

    # 파일에서 텍스트 추출
    text = await processor.extract_text(file_path, file_extension)

    # 텍스트 청킹
    chunks = processor.chunk_text(text, chunk_size=1000)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("contextify")


class DocumentProcessor:
    """
    Contextify 문서 처리 메인 클래스

    다양한 문서 형식을 처리하고 텍스트를 추출하는 통합 인터페이스입니다.

    Attributes:
        config: 설정 딕셔너리 또는 ConfigComposer 인스턴스
        supported_extensions: 지원되는 파일 확장자 목록

    Example:
        >>> processor = DocumentProcessor()
        >>> text = await processor.extract_text("document.pdf", "pdf")
        >>> chunks = processor.chunk_text(text, chunk_size=1000)
    """

    # === 지원 파일 타입 분류 ===
    DOCUMENT_TYPES = frozenset(['pdf', 'docx', 'doc', 'pptx', 'ppt', 'hwp', 'hwpx'])
    TEXT_TYPES = frozenset(['txt', 'md', 'markdown', 'rtf'])
    CODE_TYPES = frozenset([
        'py', 'js', 'ts', 'java', 'cpp', 'c', 'h', 'cs', 'go', 'rs',
        'php', 'rb', 'swift', 'kt', 'scala', 'dart', 'r', 'sql',
        'html', 'css', 'jsx', 'tsx', 'vue', 'svelte'
    ])
    CONFIG_TYPES = frozenset(['json', 'yaml', 'yml', 'xml', 'toml', 'ini', 'cfg', 'conf', 'properties', 'env'])
    DATA_TYPES = frozenset(['csv', 'tsv', 'xlsx', 'xls'])
    SCRIPT_TYPES = frozenset(['sh', 'bat', 'ps1', 'zsh', 'fish'])
    LOG_TYPES = frozenset(['log'])
    WEB_TYPES = frozenset(['htm', 'xhtml'])
    IMAGE_TYPES = frozenset(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'])

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        **kwargs
    ):
        """
        DocumentProcessor 초기화

        Args:
            config: 설정 딕셔너리 또는 ConfigComposer 인스턴스
                   - Dict: 직접 설정 딕셔너리 전달
                   - ConfigComposer: 기존 config_composer 인스턴스
                   - None: 기본 설정 사용
            **kwargs: 추가 설정 옵션
        """
        self._config = config
        self._kwargs = kwargs
        self._supported_extensions: Optional[List[str]] = None

        # 로거 설정
        self._logger = logging.getLogger("contextify.processor")

        # 라이브러리 가용성 체크 결과 캐시
        self._library_availability: Optional[Dict[str, bool]] = None

    # =========================================================================
    # 공개 속성 (Properties)
    # =========================================================================

    @property
    def supported_extensions(self) -> List[str]:
        """지원되는 모든 파일 확장자 목록"""
        if self._supported_extensions is None:
            self._supported_extensions = self._build_supported_extensions()
        return self._supported_extensions.copy()

    @property
    def config(self) -> Optional[Union[Dict[str, Any], Any]]:
        """현재 설정"""
        return self._config

    # =========================================================================
    # 공개 메서드 - 텍스트 추출
    # =========================================================================

    async def extract_text(
        self,
        file_path: Union[str, Path],
        file_extension: Optional[str] = None,
        *,
        process_type: str = "default",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        파일에서 텍스트를 추출합니다.

        Args:
            file_path: 파일 경로
            file_extension: 파일 확장자 (None인 경우 file_path에서 자동 추출)
            process_type: 처리 유형 ('default', 'enhanced', 'enhanced_v4', 'enhanced_ocr' 등)
            extract_metadata: 메타데이터 추출 여부
            **kwargs: 핸들러별 추가 옵션

        Returns:
            추출된 텍스트 문자열

        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우
            ValueError: 지원되지 않는 파일 형식인 경우
        """
        # TODO: 구현 예정
        raise NotImplementedError("extract_text method is not yet implemented")

    async def extract_text_batch(
        self,
        file_paths: List[Union[str, Path]],
        *,
        process_type: str = "default",
        extract_metadata: bool = True,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        여러 파일에서 텍스트를 배치로 추출합니다.

        Args:
            file_paths: 파일 경로 목록
            process_type: 처리 유형
            extract_metadata: 메타데이터 추출 여부
            max_concurrent: 최대 동시 처리 수
            **kwargs: 핸들러별 추가 옵션

        Returns:
            추출 결과 딕셔너리 목록
            [{"file_path": str, "text": str, "success": bool, "error": Optional[str]}, ...]
        """
        # TODO: 구현 예정
        raise NotImplementedError("extract_text_batch method is not yet implemented")

    # =========================================================================
    # 공개 메서드 - 텍스트 청킹
    # =========================================================================

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        file_extension: Optional[str] = None,
        preserve_tables: bool = True,
        **kwargs
    ) -> List[str]:
        """
        텍스트를 청크로 분할합니다.

        Args:
            text: 분할할 텍스트
            chunk_size: 청크 크기 (문자 수)
            chunk_overlap: 청크 간 오버랩 크기
            file_extension: 파일 확장자 (테이블 기반 파일 처리에 사용)
            preserve_tables: 테이블 구조 보존 여부
            **kwargs: 추가 청킹 옵션

        Returns:
            청크 문자열 목록
        """
        # TODO: 구현 예정
        raise NotImplementedError("chunk_text method is not yet implemented")

    # =========================================================================
    # 공개 메서드 - 유틸리티
    # =========================================================================

    def get_file_category(self, file_extension: str) -> str:
        """
        파일 확장자의 카테고리를 반환합니다.

        Args:
            file_extension: 파일 확장자

        Returns:
            카테고리 문자열 ('document', 'text', 'code', 'data', 등)
        """
        ext = file_extension.lower().lstrip('.')

        if ext in self.DOCUMENT_TYPES:
            return 'document'
        if ext in self.TEXT_TYPES:
            return 'text'
        if ext in self.CODE_TYPES:
            return 'code'
        if ext in self.CONFIG_TYPES:
            return 'config'
        if ext in self.DATA_TYPES:
            return 'data'
        if ext in self.SCRIPT_TYPES:
            return 'script'
        if ext in self.LOG_TYPES:
            return 'log'
        if ext in self.WEB_TYPES:
            return 'web'
        if ext in self.IMAGE_TYPES:
            return 'image'

        return 'unknown'

    def is_supported(self, file_extension: str) -> bool:
        """
        파일 확장자가 지원되는지 확인합니다.

        Args:
            file_extension: 파일 확장자

        Returns:
            지원 여부
        """
        ext = file_extension.lower().lstrip('.')
        return ext in self.supported_extensions

    @staticmethod
    def clean_text(text: str) -> str:
        """
        텍스트를 정리합니다.

        Args:
            text: 정리할 텍스트

        Returns:
            정리된 텍스트
        """
        from libs.core.functions.utils import clean_text as _clean_text
        return _clean_text(text)

    @staticmethod
    def clean_code_text(text: str) -> str:
        """
        코드 텍스트를 정리합니다.

        Args:
            text: 정리할 코드 텍스트

        Returns:
            정리된 코드 텍스트
        """
        from libs.core.functions.utils import clean_code_text as _clean_code_text
        return _clean_code_text(text)

    # =========================================================================
    # 비공개 메서드
    # =========================================================================

    def _build_supported_extensions(self) -> List[str]:
        """지원되는 확장자 목록 구성"""
        extensions = list(
            self.DOCUMENT_TYPES |
            self.TEXT_TYPES |
            self.CODE_TYPES |
            self.CONFIG_TYPES |
            self.DATA_TYPES |
            self.SCRIPT_TYPES |
            self.LOG_TYPES |
            self.WEB_TYPES |
            self.IMAGE_TYPES
        )

        # 라이브러리 가용성에 따른 필터링
        availability = self._check_library_availability()

        if not availability.get('openpyxl') and not availability.get('xlrd'):
            extensions = [e for e in extensions if e not in ['xlsx', 'xls']]
            self._logger.warning("openpyxl/xlrd not available. Excel processing disabled.")

        if not availability.get('langchain_openai'):
            extensions = [e for e in extensions if e not in self.IMAGE_TYPES]
            self._logger.warning("langchain_openai not available. Image processing disabled.")

        return sorted(extensions)

    def _check_library_availability(self) -> Dict[str, bool]:
        """필수 라이브러리 가용성 체크"""
        if self._library_availability is not None:
            return self._library_availability

        availability = {}

        # openpyxl
        try:
            from openpyxl import load_workbook  # noqa
            availability['openpyxl'] = True
        except ImportError:
            availability['openpyxl'] = False

        # xlrd
        try:
            import xlrd  # noqa
            availability['xlrd'] = True
        except ImportError:
            availability['xlrd'] = False

        # langchain_openai
        try:
            from langchain_openai import ChatOpenAI  # noqa
            availability['langchain_openai'] = True
        except ImportError:
            availability['langchain_openai'] = False

        # pdfminer
        try:
            from pdfminer.high_level import extract_text  # noqa
            availability['pdfminer'] = True
        except ImportError:
            availability['pdfminer'] = False

        # pdf2image
        try:
            from pdf2image import convert_from_path  # noqa
            availability['pdf2image'] = True
        except ImportError:
            availability['pdf2image'] = False

        # python-pptx
        try:
            from pptx import Presentation  # noqa
            availability['python_pptx'] = True
        except ImportError:
            availability['python_pptx'] = False

        # PIL
        try:
            from PIL import Image  # noqa
            availability['pil'] = True
        except ImportError:
            availability['pil'] = False

        self._library_availability = availability
        return availability

    # =========================================================================
    # 컨텍스트 매니저 지원
    # =========================================================================

    async def __aenter__(self) -> "DocumentProcessor":
        """비동기 컨텍스트 매니저 진입"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """비동기 컨텍스트 매니저 종료"""
        # 리소스 정리가 필요한 경우 여기서 수행
        pass

    def __enter__(self) -> "DocumentProcessor":
        """동기 컨텍스트 매니저 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """동기 컨텍스트 매니저 종료"""
        pass

    # =========================================================================
    # 문자열 표현
    # =========================================================================

    def __repr__(self) -> str:
        return f"DocumentProcessor(supported_extensions={len(self.supported_extensions)})"

    def __str__(self) -> str:
        return f"Contextify DocumentProcessor ({len(self.supported_extensions)} supported formats)"


# === 모듈 레벨 편의 함수 ===

def create_processor(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    **kwargs
) -> DocumentProcessor:
    """
    DocumentProcessor 인스턴스를 생성합니다.

    Args:
        config: 설정 딕셔너리 또는 ConfigComposer 인스턴스
        **kwargs: 추가 설정 옵션

    Returns:
        DocumentProcessor 인스턴스

    Example:
        >>> processor = create_processor()
        >>> processor = create_processor(config={"vision_model": "gpt-4-vision"})
    """
    return DocumentProcessor(config=config, **kwargs)


__all__ = [
    "DocumentProcessor",
    "create_processor",
]
