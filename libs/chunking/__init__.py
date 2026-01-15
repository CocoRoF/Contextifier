# libs/chunking/__init__.py
"""
Chunking - 텍스트 청킹 모듈

이 패키지는 문서 텍스트를 적절한 크기의 청크로 분할하는 기능을 제공합니다.

모듈 구조:
- chunking: 메인 청킹 함수 (split_text_preserving_html_blocks 등)
- constants: 상수, 패턴, 데이터 클래스
- table_parser: HTML 테이블 파싱
- table_chunker: 테이블 청킹 핵심 로직
- protected_regions: 보호 영역 처리
- page_chunker: 페이지 기반 청킹
- text_chunker: 텍스트 청킹
- sheet_processor: 시트 및 메타데이터 처리

사용 예시:
    from libs.chunking import split_text_preserving_html_blocks, chunk_plain_text
    from libs.chunking import TableRow, ParsedTable
"""

# === 메인 청킹 함수 (chunking.py) ===
from libs.chunking.chunking import (
    split_text_preserving_html_blocks,
    split_table_based_content,
    is_table_based_file_type,
)

# constants
from libs.chunking.constants import (
    # 상수
    LANGCHAIN_CODE_LANGUAGE_MAP,
    HTML_TABLE_PATTERN,
    CHART_BLOCK_PATTERN,
    TEXTBOX_BLOCK_PATTERN,
    IMAGE_TAG_PATTERN,
    MARKDOWN_TABLE_PATTERN,
    TABLE_WRAPPER_OVERHEAD,
    CHUNK_INDEX_OVERHEAD,
    TABLE_SIZE_THRESHOLD_MULTIPLIER,
    TABLE_BASED_FILE_TYPES,
    # 데이터 클래스
    TableRow,
    ParsedTable,
)

# table_parser
from libs.chunking.table_parser import (
    parse_html_table,
    extract_cell_spans,
    extract_cell_spans_with_positions,
    has_complex_spans,
)

# table_chunker
from libs.chunking.table_chunker import (
    calculate_available_space,
    adjust_rowspan_in_chunk,
    build_table_chunk,
    update_chunk_metadata,
    split_table_into_chunks,
    split_table_preserving_rowspan,
    chunk_large_table,
)

# protected_regions
from libs.chunking.protected_regions import (
    find_protected_regions,
    get_protected_region_positions,
    ensure_protected_region_integrity,
    split_with_protected_regions,
    split_large_chunk_with_protected_regions,
    # 하위 호환성 별칭
    ensure_table_integrity,
    split_large_chunk_with_table_protection,
)

# page_chunker
from libs.chunking.page_chunker import (
    split_into_pages,
    merge_pages,
    get_overlap_content,
    chunk_by_pages,
)

# text_chunker
from libs.chunking.text_chunker import (
    chunk_plain_text,
    chunk_text_without_tables,
    chunk_with_row_protection,
    chunk_with_row_protection_simple,
    clean_chunks,
    chunk_code_text,
    reconstruct_text_from_chunks,
    find_overlap_length,
    estimate_chunks_count,
)

# sheet_processor
from libs.chunking.sheet_processor import (
    extract_document_metadata,
    prepend_metadata_to_chunks,
    extract_sheet_sections,
    extract_content_segments,
    chunk_multi_sheet_content,
    chunk_single_table_content,
)


__all__ = [
    # === 메인 청킹 함수 ===
    "split_text_preserving_html_blocks",
    "split_table_based_content",
    "is_table_based_file_type",
    # constants
    "LANGCHAIN_CODE_LANGUAGE_MAP",
    "HTML_TABLE_PATTERN",
    "CHART_BLOCK_PATTERN",
    "TEXTBOX_BLOCK_PATTERN",
    "IMAGE_TAG_PATTERN",
    "MARKDOWN_TABLE_PATTERN",
    "TABLE_WRAPPER_OVERHEAD",
    "CHUNK_INDEX_OVERHEAD",
    "TABLE_SIZE_THRESHOLD_MULTIPLIER",
    "TABLE_BASED_FILE_TYPES",
    "TableRow",
    "ParsedTable",
    # table_parser
    "parse_html_table",
    "extract_cell_spans",
    "extract_cell_spans_with_positions",
    "has_complex_spans",
    # table_chunker
    "calculate_available_space",
    "adjust_rowspan_in_chunk",
    "build_table_chunk",
    "update_chunk_metadata",
    "split_table_into_chunks",
    "split_table_preserving_rowspan",
    "chunk_large_table",
    # protected_regions
    "find_protected_regions",
    "get_protected_region_positions",
    "ensure_protected_region_integrity",
    "split_with_protected_regions",
    "split_large_chunk_with_protected_regions",
    "ensure_table_integrity",
    "split_large_chunk_with_table_protection",
    # page_chunker
    "split_into_pages",
    "merge_pages",
    "get_overlap_content",
    "chunk_by_pages",
    # text_chunker
    "chunk_plain_text",
    "chunk_text_without_tables",
    "chunk_with_row_protection",
    "chunk_with_row_protection_simple",
    "clean_chunks",
    "chunk_code_text",
    "reconstruct_text_from_chunks",
    "find_overlap_length",
    "estimate_chunks_count",
    # sheet_processor
    "extract_document_metadata",
    "prepend_metadata_to_chunks",
    "extract_sheet_sections",
    "extract_content_segments",
    "chunk_multi_sheet_content",
    "chunk_single_table_content",
]
