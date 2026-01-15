"""
XLSX/XLS 메타데이터 추출 모듈

Excel 문서에서 메타데이터(제목, 작성자, 주제, 키워드, 작성일, 수정일 등)를 추출합니다.
"""

import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger("document-processor")


def extract_xlsx_metadata(wb) -> Dict[str, Any]:
    """
    XLSX 문서에서 메타데이터를 추출합니다.

    openpyxl의 properties를 통해 다음 정보를 추출합니다:
    - 제목 (title)
    - 주제 (subject)
    - 작성자 (creator)
    - 키워드 (keywords)
    - 설명 (description)
    - 마지막 수정자 (lastModifiedBy)
    - 작성일 (created)
    - 수정일 (modified)

    Args:
        wb: openpyxl Workbook 객체

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        props = wb.properties

        if props.title:
            metadata['title'] = props.title.strip()
        if props.subject:
            metadata['subject'] = props.subject.strip()
        if props.creator:
            metadata['author'] = props.creator.strip()
        if props.keywords:
            metadata['keywords'] = props.keywords.strip()
        if props.description:
            metadata['comments'] = props.description.strip()
        if props.lastModifiedBy:
            metadata['last_saved_by'] = props.lastModifiedBy.strip()
        if props.created:
            metadata['create_time'] = props.created
        if props.modified:
            metadata['last_saved_time'] = props.modified

        logger.debug(f"Extracted XLSX metadata: {list(metadata.keys())}")

    except Exception as e:
        logger.warning(f"Failed to extract XLSX metadata: {e}")

    return metadata


def extract_xls_metadata(wb) -> Dict[str, Any]:
    """
    XLS 문서에서 메타데이터를 추출합니다.

    xlrd는 제한된 메타데이터만 지원합니다.

    Args:
        wb: xlrd Workbook 객체

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        # xlrd는 제한된 메타데이터 접근만 가능
        if hasattr(wb, 'user_name') and wb.user_name:
            metadata['author'] = wb.user_name

        logger.debug(f"Extracted XLS metadata: {list(metadata.keys())}")

    except Exception as e:
        logger.warning(f"Failed to extract XLS metadata: {e}")

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
