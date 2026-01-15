# csv_helper/csv_metadata.py
"""
CSV 메타데이터 추출 및 포맷팅

CSV 파일의 메타데이터를 추출하고 읽기 쉬운 형식으로 변환합니다.
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from .csv_constants import DELIMITER_NAMES

logger = logging.getLogger("document-processor")


def format_file_size(size_bytes: int) -> str:
    """
    파일 크기를 읽기 쉬운 형식으로 변환합니다.

    Args:
        size_bytes: 파일 크기 (바이트)

    Returns:
        포맷된 파일 크기 문자열 (예: "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_delimiter_name(delimiter: str) -> str:
    """
    구분자를 읽기 쉬운 이름으로 변환합니다.

    Args:
        delimiter: 구분자 문자

    Returns:
        구분자의 읽기 쉬운 이름 (예: "쉼표 (,)")
    """
    return DELIMITER_NAMES.get(delimiter, repr(delimiter))


def extract_csv_metadata(
    file_path: str,
    encoding: str,
    delimiter: str,
    rows: List[List[str]],
    has_header: bool
) -> Dict[str, Any]:
    """
    CSV 파일에서 메타데이터를 추출합니다.

    Args:
        file_path: 파일 경로
        encoding: 감지된 인코딩
        delimiter: 감지된 구분자
        rows: 파싱된 행 데이터
        has_header: 헤더 존재 여부

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        # 파일 정보
        file_stat = os.stat(file_path)
        file_name = os.path.basename(file_path)

        metadata['file_name'] = file_name
        metadata['file_size'] = format_file_size(file_stat.st_size)
        metadata['modified_time'] = datetime.fromtimestamp(file_stat.st_mtime)

        # CSV 구조 정보
        metadata['encoding'] = encoding
        metadata['delimiter'] = get_delimiter_name(delimiter)
        metadata['row_count'] = len(rows)
        metadata['col_count'] = len(rows[0]) if rows else 0
        metadata['has_header'] = '예' if has_header else '아니오'

        # 헤더 정보 (있는 경우)
        if has_header and rows:
            headers = [h.strip() for h in rows[0] if h.strip()]
            if headers:
                metadata['columns'] = ', '.join(headers[:10])  # 최대 10개
                if len(rows[0]) > 10:
                    metadata['columns'] += f' 외 {len(rows[0]) - 10}개'

        logger.debug(f"Extracted CSV metadata: {list(metadata.keys())}")

    except Exception as e:
        logger.warning(f"Failed to extract CSV metadata: {e}")

    return metadata


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터 딕셔너리를 읽기 쉬운 문자열로 변환합니다.

    Args:
        metadata: 메타데이터 딕셔너리

    Returns:
        포맷된 메타데이터 문자열 (<Document-Metadata> 태그 형식)
    """
    if not metadata:
        return ""

    lines = ["<Document-Metadata>"]

    field_names = {
        'file_name': '파일명',
        'file_size': '파일 크기',
        'modified_time': '수정일',
        'encoding': '인코딩',
        'delimiter': '구분자',
        'row_count': '행 수',
        'col_count': '열 수',
        'has_header': '헤더 존재',
        'columns': '컬럼 목록',
    }

    for key, label in field_names.items():
        if key in metadata and metadata[key] is not None:
            value = metadata[key]

            # datetime 객체 포맷팅
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')

            lines.append(f"  {label}: {value}")

    lines.append("</Document-Metadata>")

    return "\n".join(lines)
