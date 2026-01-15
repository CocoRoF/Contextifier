# service/document_processor/processor/docx_helper/docx_metadata.py
"""
DOCX 메타데이터 추출 유틸리티

DOCX 문서의 core_properties에서 메타데이터를 추출하고 포맷팅합니다.
- extract_docx_metadata: 메타데이터 딕셔너리 추출
- format_metadata: 메타데이터를 읽기 쉬운 문자열로 변환
"""
import logging
from datetime import datetime
from typing import Any, Dict

from docx import Document

logger = logging.getLogger("document-processor")


def extract_docx_metadata(doc: Document) -> Dict[str, Any]:
    """
    DOCX 문서에서 메타데이터를 추출합니다.

    python-docx의 core_properties를 통해 다음 정보를 추출합니다:
    - 제목 (title)
    - 주제 (subject)
    - 작성자 (author)
    - 키워드 (keywords)
    - 설명 (comments)
    - 마지막 수정자 (last_modified_by)
    - 작성일 (created)
    - 수정일 (modified)

    Args:
        doc: python-docx Document 객체

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        props = doc.core_properties

        if props.title:
            metadata['title'] = props.title.strip()
        if props.subject:
            metadata['subject'] = props.subject.strip()
        if props.author:
            metadata['author'] = props.author.strip()
        if props.keywords:
            metadata['keywords'] = props.keywords.strip()
        if props.comments:
            metadata['comments'] = props.comments.strip()
        if props.last_modified_by:
            metadata['last_saved_by'] = props.last_modified_by.strip()
        if props.created:
            metadata['create_time'] = props.created
        if props.modified:
            metadata['last_saved_time'] = props.modified

        logger.debug(f"Extracted DOCX metadata: {list(metadata.keys())}")

    except Exception as e:
        logger.warning(f"Failed to extract DOCX metadata: {e}")

    return metadata


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터 딕셔너리를 읽기 쉬운 문자열로 변환합니다.

    Args:
        metadata: 메타데이터 딕셔너리

    Returns:
        포맷된 메타데이터 문자열
    """
    if not metadata:
        return ""

    lines = ["<Document-Metadata>"]

    field_names = {
        'title': '제목',
        'subject': '주제',
        'author': '작성자',
        'keywords': '키워드',
        'comments': '설명',
        'last_saved_by': '마지막 저장자',
        'create_time': '작성일',
        'last_saved_time': '수정일',
    }

    for key, label in field_names.items():
        if key in metadata and metadata[key]:
            value = metadata[key]

            # datetime 객체 포맷팅
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')

            lines.append(f"  {label}: {value}")

    lines.append("</Document-Metadata>")

    return "\n".join(lines)


__all__ = [
    'extract_docx_metadata',
    'format_metadata',
]
