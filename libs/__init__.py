# libs/__init__.py
"""
Contextify Library

문서 처리 및 청킹을 위한 라이브러리입니다.

패키지 구조:
- core: 문서 처리 핵심 모듈
    - DocumentProcessor: 메인 문서 처리 클래스
    - processor: 개별 문서 타입별 핸들러 (PDF, DOCX, PPT, Excel, HWP 등)
    - functions: 유틸리티 함수

- chunking: 텍스트 청킹 모듈
    - 텍스트 분할 및 청킹 로직
    - 테이블 보호 청킹
    - 페이지 기반 청킹

사용 예시:
    from libs.core import DocumentProcessor
    from libs.chunking import chunk_plain_text, split_text_preserving_html_blocks
"""

__version__ = "1.0.0"

# 핵심 클래스 최상위 노출
from libs.core import DocumentProcessor

# 명시적 서브패키지
from libs import core
from libs import chunking

__all__ = [
    "__version__",
    # 핵심 클래스
    "DocumentProcessor",
    # 서브패키지
    "core",
    "chunking",
]
