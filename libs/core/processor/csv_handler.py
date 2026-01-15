# your_package/document_processor/csv_handler.py
"""
CSV/TSV Handler - 구분자 기반 텍스트 데이터 처리기

주요 기능:
- 메타데이터 추출 (<Document-Metadata> 태그 형식)
- CSV/TSV 파일 파싱
- 자동 인코딩 감지 (BOM, chardet)
- 자동 구분자 감지
- 헤더 자동 감지
- 병합셀 유무에 따라 Markdown 또는 HTML 테이블 변환

모든 처리는 표준 라이브러리 csv 모듈을 사용합니다.

리팩터링된 구조:
- 기능별 로직은 csv_helper/ 모듈로 분리
- csv_handler는 조합 및 조율 역할
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# csv_helper에서 필요한 것들 import
from libs.core.processor.csv_helper import (
    # Constants
    CSVMetadata,
    # Metadata
    extract_csv_metadata,
    format_metadata,
    # Encoding
    read_file_with_encoding,
    # Parser
    detect_delimiter,
    parse_csv_content,
    detect_header,
    # Table
    convert_rows_to_table,
)

logger = logging.getLogger("document-processor")


# === 메인 함수 ===

async def extract_text_from_csv(
    file_path: str,
    current_config: Dict[str, Any] = None,
    encoding: str = None,
    delimiter: str = None,
    extract_default_metadata: bool = True
) -> str:
    """
    CSV/TSV 파일에서 텍스트를 추출합니다.

    주요 기능:
    - 메타데이터 추출 (<Document-Metadata> 태그 형식)
    - 자동 인코딩 감지 (BOM, chardet, 휴리스틱)
    - 자동 구분자 감지
    - 헤더 자동 감지
    - 병합셀 유무에 따라 Markdown 또는 HTML 테이블 변환

    Args:
        file_path: CSV/TSV 파일 경로
        current_config: 설정 딕셔너리 (현재 미사용, 향후 확장용)
        encoding: 인코딩 (None이면 자동 감지)
        delimiter: 구분자 (None이면 자동 감지)
        extract_default_metadata: 기본 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (메타데이터, 테이블 포함)
    """
    _ = current_config  # 향후 확장용

    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"CSV processing: {file_path}, ext: {ext}")

    # TSV 파일이면 기본 구분자를 탭으로 설정
    if ext == '.tsv' and delimiter is None:
        delimiter = '\t'

    try:
        result_parts = []

        # 파일 읽기 (인코딩 자동 감지)
        content, detected_encoding = await read_file_with_encoding(file_path, encoding)

        # 구분자 자동 감지
        if delimiter is None:
            delimiter = detect_delimiter(content)

        logger.info(f"CSV: encoding={detected_encoding}, delimiter={repr(delimiter)}")

        # CSV 파싱
        rows = parse_csv_content(content, delimiter)

        if not rows:
            return ""

        # 헤더 여부 감지
        has_header = detect_header(rows)

        # 메타데이터 추출 및 추가
        if extract_default_metadata:
            metadata = extract_csv_metadata(file_path, detected_encoding, delimiter, rows, has_header)
            metadata_str = format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")
                logger.info(f"CSV metadata extracted: {list(metadata.keys())}")

        # 테이블 생성 (병합셀 유무에 따라 Markdown 또는 HTML)
        table = convert_rows_to_table(rows, has_header)
        if table:
            result_parts.append(table)

        result = "".join(result_parts)
        logger.info(f"CSV processing completed: {len(rows)} rows, {len(rows[0]) if rows else 0} cols")

        return result

    except Exception as e:
        logger.error(f"Error extracting text from CSV {file_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


# === 유틸리티 함수 ===

async def get_csv_metadata(file_path: str) -> CSVMetadata:
    """
    CSV 파일의 메타데이터를 반환합니다.

    Args:
        file_path: CSV 파일 경로

    Returns:
        CSVMetadata 객체
    """
    try:
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        modified_time = datetime.fromtimestamp(file_stat.st_mtime)
        file_name = os.path.basename(file_path)

        content, encoding = await read_file_with_encoding(file_path)
        delimiter = detect_delimiter(content)
        rows = parse_csv_content(content, delimiter)
        has_header = detect_header(rows) if rows else False

        return CSVMetadata(
            encoding=encoding,
            delimiter=delimiter,
            has_header=has_header,
            row_count=len(rows),
            col_count=len(rows[0]) if rows else 0,
            file_size=file_size,
            file_name=file_name,
            modified_time=modified_time
        )

    except Exception as e:
        logger.error(f"Error getting CSV metadata: {e}")
        raise


async def validate_csv(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    CSV 파일의 유효성을 검증합니다.

    Args:
        file_path: CSV 파일 경로

    Returns:
        (is_valid, error_message) 튜플
    """
    try:
        content, _ = await read_file_with_encoding(file_path)
        delimiter = detect_delimiter(content)
        rows = parse_csv_content(content, delimiter)

        if not rows:
            return False, "파일이 비어있습니다."

        # 열 수 일관성 확인
        col_counts = [len(row) for row in rows]
        if len(set(col_counts)) > 1:
            return False, f"열 수가 일관되지 않습니다: {set(col_counts)}"

        return True, None

    except Exception as e:
        return False, str(e)
