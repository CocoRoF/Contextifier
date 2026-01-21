# contextifier/core/processor/csv_helper/csv_metadata.py
"""
CSV Metadata Extraction Module

Provides CSVMetadataExtractor class for extracting metadata from CSV files.
Implements BaseMetadataExtractor interface.

CSV differs from regular documents - it provides file structure information as metadata:
- File name, file size, modification time
- Encoding, delimiter
- Row/column count, header information
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
    MetadataFormatter,
)
from contextifier.core.processor.csv_helper.csv_constants import DELIMITER_NAMES

logger = logging.getLogger("document-processor")


def format_file_size(size_bytes: int) -> str:
    """
    Convert file size to human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted file size string (e.g., "1.5 MB")
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
    Convert delimiter to human-readable name.

    Args:
        delimiter: Delimiter character

    Returns:
        Human-readable delimiter name (e.g., "Comma (,)")
    """
    return DELIMITER_NAMES.get(delimiter, repr(delimiter))


@dataclass
class CSVSourceInfo:
    """
    Source information for CSV metadata extraction.
    
    Container for data passed to CSVMetadataExtractor.extract().
    """
    file_path: str
    encoding: str
    delimiter: str
    rows: List[List[str]]
    has_header: bool


class CSVMetadataExtractor(BaseMetadataExtractor):
    """
    CSV Metadata Extractor.
    
    CSV 파일의 구조 정보를 메타데이터로 추출합니다.
    
    지원 필드 (custom 필드에 저장):
    - file_name, file_size, modified_time
    - encoding, delimiter
    - row_count, col_count, has_header, columns
    
    사용법:
        extractor = CSVMetadataExtractor()
        source = CSVSourceInfo(
            file_path="data.csv",
            encoding="utf-8",
            delimiter=",",
            rows=parsed_rows,
            has_header=True
        )
        metadata = extractor.extract(source)
        text = extractor.format(metadata)
    """
    
    # CSV 특화 필드 라벨
    CSV_FIELD_LABELS = {
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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # CSV용 커스텀 포맷터 설정
        self._formatter.field_labels.update(self.CSV_FIELD_LABELS)
    
    def extract(self, source: CSVSourceInfo) -> DocumentMetadata:
        """
        CSV 파일에서 메타데이터를 추출합니다.
        
        Args:
            source: CSVSourceInfo 객체 (파일 경로, 인코딩, 구분자, 행 데이터, 헤더 여부)
            
        Returns:
            추출된 메타데이터가 담긴 DocumentMetadata 인스턴스
        """
        custom_fields: Dict[str, Any] = {}

        try:
            # 파일 정보
            file_stat = os.stat(source.file_path)
            file_name = os.path.basename(source.file_path)

            custom_fields['file_name'] = file_name
            custom_fields['file_size'] = format_file_size(file_stat.st_size)
            custom_fields['modified_time'] = datetime.fromtimestamp(file_stat.st_mtime)

            # CSV 구조 정보
            custom_fields['encoding'] = source.encoding
            custom_fields['delimiter'] = get_delimiter_name(source.delimiter)
            custom_fields['row_count'] = len(source.rows)
            custom_fields['col_count'] = len(source.rows[0]) if source.rows else 0
            custom_fields['has_header'] = '예' if source.has_header else '아니오'

            # 헤더 정보 (있는 경우)
            if source.has_header and source.rows:
                headers = [h.strip() for h in source.rows[0] if h.strip()]
                if headers:
                    custom_fields['columns'] = ', '.join(headers[:10])  # 최대 10개
                    if len(source.rows[0]) > 10:
                        custom_fields['columns'] += f' 외 {len(source.rows[0]) - 10}개'

            self.logger.debug(f"Extracted CSV metadata: {list(custom_fields.keys())}")

        except Exception as e:
            self.logger.warning(f"Failed to extract CSV metadata: {e}")
        
        # CSV는 표준 필드가 없고 모두 custom 필드
        return DocumentMetadata(custom=custom_fields)


__all__ = [
    'CSVMetadataExtractor',
    'CSVSourceInfo',
    'format_file_size',
    'get_delimiter_name',
]
