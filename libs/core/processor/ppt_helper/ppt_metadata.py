"""
PPT 메타데이터 추출 모듈

포함 함수:
- extract_ppt_metadata(): PPT에서 메타데이터 추출
- format_metadata(): 메타데이터를 읽기 쉬운 문자열로 변환
"""
import logging
from datetime import datetime
from typing import Any, Dict

from pptx import Presentation

logger = logging.getLogger("document-processor")


def extract_ppt_metadata(prs: Presentation) -> Dict[str, Any]:
    """
    PPT 파일에서 메타데이터를 추출합니다.

    python-pptx의 core_properties를 통해 다음 정보를 추출합니다:
    - 제목 (title)
    - 주제 (subject)
    - 작성자 (author)
    - 키워드 (keywords)
    - 설명 (comments)
    - 마지막 수정자 (last_modified_by)
    - 작성일 (created)
    - 수정일 (modified)

    Args:
        prs: python-pptx Presentation 객체

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        props = prs.core_properties

        if props.title:
            metadata['title'] = props.title
        if props.subject:
            metadata['subject'] = props.subject
        if props.author:
            metadata['author'] = props.author
        if props.keywords:
            metadata['keywords'] = props.keywords
        if props.comments:
            metadata['comments'] = props.comments
        if props.last_modified_by:
            metadata['last_saved_by'] = props.last_modified_by
        if props.created:
            metadata['create_time'] = props.created
        if props.modified:
            metadata['last_saved_time'] = props.modified

        logger.info(f"Extracted PPT metadata: {metadata}")

    except Exception as e:
        logger.warning(f"Failed to extract PPT metadata: {e}")

    return metadata


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터 딕셔너리를 읽기 쉬운 문자열로 변환합니다.

    Args:
        metadata: 메타데이터 딕셔너리

    Returns:
        포맷팅된 메타데이터 문자열
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
