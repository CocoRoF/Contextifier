# your_package/document_processor/chunking.py
"""
Document Chunking Module - 고도화된 텍스트 청킹 시스템

주요 기능:
- HTML 테이블 보존 청킹
- 대용량 테이블 데이터(CSV/Excel) 지능형 분할
- 테이블 구조 복원 (헤더 보존)
- 페이지 기반 청킹
- 코드 파일 언어별 청킹

핵심 개선사항 (테이블 청킹 고도화):
- 대용량 테이블을 chunk_size에 맞게 행 단위로 분할
- 각 청크에 테이블 헤더를 자동으로 복원
- 테이블 구조 무결성 보장
- 청크 인덱싱 메타데이터 추가

리팩토링:
- 핵심 로직은 chunking_helper 서브모듈로 분리
- 이 파일은 공개 API와 통합 로직만 유지
"""
import logging
import re
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunking_helper에서 필요한 모듈 import
from libs.chunking.chunking_helper import (
    # constants
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
    TableRow,
    ParsedTable,
    # table_parser
    parse_html_table as _parse_html_table,
    extract_cell_spans as _extract_cell_spans,
    extract_cell_spans_with_positions as _extract_cell_spans_with_positions,
    has_complex_spans as _has_complex_spans,
    # table_chunker
    calculate_available_space as _calculate_available_space,
    adjust_rowspan_in_chunk as _adjust_rowspan_in_chunk,
    build_table_chunk as _build_table_chunk,
    update_chunk_metadata as _update_chunk_metadata,
    split_table_into_chunks as _split_table_into_chunks,
    split_table_preserving_rowspan as _split_table_preserving_rowspan,
    chunk_large_table as _chunk_large_table,
    # protected_regions
    find_protected_regions as _find_protected_regions,
    get_protected_region_positions as _get_protected_region_positions,
    ensure_protected_region_integrity as _ensure_protected_region_integrity,
    split_with_protected_regions as _split_with_protected_regions,
    split_large_chunk_with_protected_regions as _split_large_chunk_with_protected_regions,
    ensure_table_integrity as _ensure_table_integrity,
    split_large_chunk_with_table_protection as _split_large_chunk_with_table_protection,
    # page_chunker
    split_into_pages as _split_into_pages,
    merge_pages as _merge_pages,
    get_overlap_content as _get_overlap_content,
    chunk_by_pages as _chunk_by_pages,
    # text_chunker
    chunk_plain_text as _chunk_plain_text,
    chunk_text_without_tables,
    chunk_with_row_protection,
    chunk_with_row_protection_simple,
    clean_chunks as _clean_chunks,
    chunk_code_text,
    reconstruct_text_from_chunks,
    find_overlap_length,
    estimate_chunks_count,
    # sheet_processor
    extract_document_metadata as _extract_document_metadata,
    prepend_metadata_to_chunks as _prepend_metadata_to_chunks,
    extract_sheet_sections as _extract_sheet_sections,
    extract_content_segments as _extract_content_segments,
    chunk_multi_sheet_content,
    chunk_single_table_content,
)

logger = logging.getLogger("document-processor")


# ============================================================================
# 공개 API - 외부에서 사용되는 주요 함수들
# ============================================================================

def split_table_based_content(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    테이블 기반 콘텐츠(CSV/Excel)를 청킹합니다.

    대용량 테이블을 chunk_size에 맞게 분할하고,
    각 청크에서 테이블 구조를 복원합니다.

    다중 시트 Excel의 경우 시트별로 분리하여 처리합니다.

    Args:
        text: 전체 텍스트 (메타데이터 + 테이블)
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹침 크기

    Returns:
        청크 리스트
    """
    if not text or not text.strip():
        return [""]

    # 메타데이터 추출
    metadata_block, text_without_metadata = _extract_document_metadata(text)

    # 데이터 분석 블록 추출
    analysis_pattern = r'(\[데이터 분석\].*?\[/데이터 분석\])\s*'
    analysis_match = re.search(analysis_pattern, text_without_metadata, re.DOTALL)
    analysis_block = ""

    if analysis_match:
        analysis_block = analysis_match.group(1)
        text_without_analysis = (
            text_without_metadata[:analysis_match.start()] +
            text_without_metadata[analysis_match.end():]
        ).strip()
    else:
        text_without_analysis = text_without_metadata

    # 다중 시트 확인 (Excel)
    sheets = _extract_sheet_sections(text_without_analysis)

    if sheets:
        logger.info(f"Multi-sheet Excel detected: {len(sheets)} sheets")
        return chunk_multi_sheet_content(
            sheets, metadata_block, analysis_block, chunk_size, chunk_overlap,
            _chunk_plain_text, _chunk_large_table
        )

    # 단일 테이블/시트 처리
    return chunk_single_table_content(
        text_without_analysis, metadata_block, analysis_block, chunk_size, chunk_overlap,
        _chunk_plain_text, _chunk_large_table
    )


def split_text_preserving_html_blocks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    file_extension: Optional[str] = None,
    force_chunking: Optional[bool] = False
) -> List[str]:
    """
    HTML 테이블을 보존하고, 페이지 단위를 고려하여 텍스트를 청킹합니다.

    핵심 전략:
    1. file_extension이 CSV/Excel이면 테이블 기반 청킹 적용
    2. 페이지 마커가 있으면 페이지 기반 청킹 우선 적용
    3. 페이지들을 chunk_size 기준으로 병합 (1.5배까지 허용)
    4. 테이블 중간에서는 절대 자르지 않음
    5. overlap 정상 적용

    Args:
        text: 원본 텍스트
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹침 크기
        file_extension: 파일 확장자 (csv, xlsx, pdf 등) - 테이블 기반 처리 결정에 사용
        force_chunking: 강제 청킹 여부 (테이블 기반 파일 제외하고 강제 청킹 적용)

    Returns:
        청크 리스트
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return [""]

    # === 테이블 기반 콘텐츠 확인 (CSV/Excel 파일만) ===
    # file_extension을 기반으로 명시적으로 판단 (텍스트 내용 추측 X)
    is_table_based = file_extension and file_extension.lower() in TABLE_BASED_FILE_TYPES

    # 테이블 보호 해제 조건: is_table_based 또는 force_chunking이 True
    disable_table_protection = is_table_based or force_chunking

    if is_table_based:
        # 대용량 테이블 확인
        table_pattern = r'<table\s+border=["\']1["\']>.*?</table>'
        table_matches = list(re.finditer(table_pattern, text, re.DOTALL | re.IGNORECASE))

        # 테이블이 chunk_size보다 크면 분할 필요
        has_large_table = any(
            (m.end() - m.start()) > chunk_size * TABLE_SIZE_THRESHOLD_MULTIPLIER
            for m in table_matches
        )

        if has_large_table:
            logger.info(f"Large table detected in {file_extension} file, using table-based chunking")
            return split_table_based_content(text, chunk_size, chunk_overlap)

    # 메타데이터 추출
    metadata_block, text_without_metadata = _extract_document_metadata(text)
    text = text_without_metadata

    # === 페이지 마커 확인 ===
    page_marker_pattern = r'<페이지\s*번호>\s*(\d+)\s*</페이지\s*번호>'
    has_page_markers = bool(re.search(page_marker_pattern, text))

    if has_page_markers:
        # 페이지 기반 청킹
        logger.debug("Page markers found, using page-based chunking")
        chunks = _chunk_by_pages(text, chunk_size, chunk_overlap, is_table_based, force_chunking)
    else:
        # 보호 영역 찾기 (HTML 테이블, 차트 블록, Markdown 테이블)
        # force_chunking 시 테이블 보호 해제 (차트는 항상 보호)
        protected_regions = _find_protected_regions(text, is_table_based, force_chunking)
        protected_positions = _get_protected_region_positions(protected_regions)

        if protected_positions:
            region_types = set(r[2] for r in protected_regions)
            logger.info(f"Found {len(protected_positions)} protected regions in document: {region_types}")
            chunks = _split_with_protected_regions(text, protected_positions, chunk_size, chunk_overlap, force_chunking)
        else:
            # 보호 영역 없을 때: force_chunking이면 row 단위 청킹 적용
            if disable_table_protection:
                logger.debug("Force chunking enabled, using row-preserving chunking")
                chunks = _chunk_with_row_protection(text, chunk_size, chunk_overlap, force_chunking)
            else:
                logger.debug("No protected blocks found, using standard chunking")
                return _chunk_text_without_tables(text, chunk_size, chunk_overlap, metadata_block)

    # 청크 정제
    cleaned_chunks = _clean_chunks(chunks)

    # 메타데이터 추가
    cleaned_chunks = _prepend_metadata_to_chunks(cleaned_chunks, metadata_block)

    logger.info(f"Final text split into {len(cleaned_chunks)} chunks")

    return cleaned_chunks


def is_table_based_file_type(file_extension: Optional[str]) -> bool:
    """
    파일 확장자가 테이블 기반 파일 타입인지 확인합니다.

    Args:
        file_extension: 파일 확장자

    Returns:
        테이블 기반 파일 타입이면 True
    """
    if not file_extension:
        return False
    return file_extension.lower() in TABLE_BASED_FILE_TYPES


# ============================================================================
# 내부 래퍼 함수 - 기존 호출 패턴 호환성 유지
# ============================================================================

def _chunk_text_without_tables(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[str]
) -> List[str]:
    """
    테이블이 없는 텍스트에 대한 기존 청킹 로직.
    chunk_text_without_tables의 래퍼 함수.
    """
    return chunk_text_without_tables(
        text, chunk_size, chunk_overlap, metadata,
        _prepend_metadata_to_chunks
    )


def _chunk_with_row_protection(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    force_chunking: bool = False
) -> List[str]:
    """
    테이블 보호 해제 시 row 단위로는 보호하면서 청킹합니다.
    chunk_with_row_protection의 래퍼 함수.
    """
    # force_chunking을 전달하기 위한 래퍼 함수
    def split_with_protected_regions_wrapper(text, regions, chunk_size, chunk_overlap):
        return _split_with_protected_regions(text, regions, chunk_size, chunk_overlap, force_chunking)

    return chunk_with_row_protection(
        text, chunk_size, chunk_overlap,
        split_with_protected_regions_wrapper, _chunk_large_table
    )
