# chunking_helper/page_chunker.py
"""
Page Chunker - 페이지 기반 청킹

주요 기능:
- 페이지 단위 텍스트 분리
- 페이지 병합 및 청킹
- overlap 처리
"""
import logging
import re
from typing import List, Tuple

from libs.chunking.protected_regions import (
    find_protected_regions, get_protected_region_positions,
    ensure_protected_region_integrity, split_large_chunk_with_protected_regions
)

logger = logging.getLogger("document-processor")


def split_into_pages(text: str, page_marker_pattern: str) -> List[Tuple[int, str]]:
    """
    텍스트를 페이지별로 분리합니다.
    빈 페이지(페이지 마커만 있는 경우)는 제외합니다.

    Returns:
        [(page_num, page_content), ...] 리스트
    """
    pages = []

    # 페이지 마커 위치 찾기
    markers = list(re.finditer(page_marker_pattern, text))

    if not markers:
        return []

    for i, match in enumerate(markers):
        page_num = int(match.group(1))
        start = match.start()

        # 다음 페이지 마커까지 또는 텍스트 끝까지
        if i + 1 < len(markers):
            end = markers[i + 1].start()
        else:
            end = len(text)

        # 페이지 내용 (마커 포함)
        page_content = text[start:end].strip()

        # 빈 페이지 체크: 페이지 마커만 있는지 확인
        if page_content:
            content_without_marker = re.sub(page_marker_pattern, '', page_content).strip()

            if content_without_marker:
                # 실제 내용이 있는 페이지만 추가
                pages.append((page_num, page_content))
            else:
                # 빈 페이지는 스킵
                logger.debug(f"Skipping empty page {page_num}")

    # 첫 페이지 마커 이전 내용이 있으면 추가
    if markers and markers[0].start() > 0:
        before_content = text[:markers[0].start()].strip()
        if before_content:
            pages.insert(0, (0, before_content))

    return pages


def merge_pages(pages: List[Tuple[int, str]]) -> str:
    """
    페이지들을 하나의 문자열로 병합합니다.
    """
    return '\n\n'.join(content for _, content in pages)


def get_overlap_content(pages: List[Tuple[int, str]], overlap_size: int) -> str:
    """
    마지막 페이지에서 overlap 크기만큼 추출합니다.
    """
    if not pages:
        return ""

    _, last_content = pages[-1]
    if len(last_content) <= overlap_size:
        return last_content

    return last_content[-overlap_size:]


def chunk_by_pages(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    is_table_based: bool = False,
    force_chunking: bool = False,
    page_tag_processor = None
) -> List[str]:
    """
    페이지 단위로 텍스트를 청킹합니다.

    알고리즘:
    1. 텍스트를 페이지별로 분리
    2. 페이지들을 순서대로 병합 시도
    3. 병합 후 크기가 chunk_size 이하면 계속 병합
    4. chunk_size 초과 시:
       - 1.5배 이하면 허용
       - 1.5배 초과면 이전까지만 청크로 확정
    5. 보호 영역(테이블, 차트, Markdown 테이블)이 페이지 경계에 걸쳐있으면 함께 유지
       (단, force_chunking 시 테이블은 row 단위만 보호, 차트는 항상 보호)
    
    Args:
        text: Original text
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap size between chunks
        is_table_based: Whether the file is table-based
        force_chunking: Force chunking (disable table protection)
        page_tag_processor: PageTagProcessor instance for custom patterns
    """
    # Build page marker patterns from PageTagProcessor or use defaults
    if page_tag_processor is not None:
        page_marker_patterns = [
            page_tag_processor.get_pattern_string(),  # Page pattern
        ]
        config = page_tag_processor.config
        if config.slide_prefix != config.tag_prefix:
            from libs.core.functions.page_tag_processor import PageTagType
            page_marker_patterns.append(page_tag_processor.get_pattern_string(PageTagType.SLIDE))
    else:
        page_marker_patterns = [
            r'\[Page Number:\s*(\d+)\]',  # Default page format
            r'\[Slide Number:\s*(\d+)\]',  # Default slide format
        ]
    
    # Find first matching pattern
    pages = []
    for page_marker_pattern in page_marker_patterns:
        pages = split_into_pages(text, page_marker_pattern)
        if pages:
            break

    if not pages:
        # 페이지 분리 실패 시 일반 청킹
        from .text_chunker import chunk_plain_text
        return chunk_plain_text(text, chunk_size, chunk_overlap)

    logger.debug(f"Split into {len(pages)} pages")

    # 보호 영역 위치 식별 (HTML 테이블, 차트 블록, Markdown 테이블)
    # force_chunking 시 테이블 보호 해제 (차트는 항상 보호)
    protected_regions = find_protected_regions(text, is_table_based, force_chunking)
    protected_positions = get_protected_region_positions(protected_regions)

    # 페이지 병합하여 청크 생성
    chunks = []
    max_size = int(chunk_size * 1.5)  # 1.5배까지 허용

    current_chunk_pages = []  # 현재 청크에 포함된 페이지들
    current_size = 0

    for page_idx, (page_num, page_content) in enumerate(pages):
        page_size = len(page_content)

        if not current_chunk_pages:
            # 첫 페이지
            current_chunk_pages.append((page_num, page_content))
            current_size = page_size
            continue

        # 병합 시도
        # 페이지 사이에 \n\n 추가 (4자)
        potential_size = current_size + 4 + page_size

        if potential_size <= chunk_size:
            # chunk_size 이하: 병합
            current_chunk_pages.append((page_num, page_content))
            current_size = potential_size
        elif potential_size <= max_size:
            # chunk_size 초과 but 1.5배 이하: 병합 허용
            current_chunk_pages.append((page_num, page_content))
            current_size = potential_size

            # 이 청크 확정 (더 이상 추가 안 함)
            chunk_content = merge_pages(current_chunk_pages)

            # 보호 영역 무결성 확인: 청크 끝이 보호 영역 중간이면 경고
            chunk_content = ensure_protected_region_integrity(chunk_content)

            chunks.append(chunk_content)

            # overlap 처리: 마지막 페이지 일부를 다음 청크에 포함
            overlap_content = get_overlap_content(current_chunk_pages, chunk_overlap)
            current_chunk_pages = []
            current_size = 0

            if overlap_content:
                # overlap은 다음 청크 시작에 추가됨 (아래서 처리)
                pass
        else:
            # 1.5배 초과: 현재까지 청크로 확정, 새 페이지는 다음 청크로
            if current_chunk_pages:
                chunk_content = merge_pages(current_chunk_pages)
                chunk_content = ensure_protected_region_integrity(chunk_content)
                chunks.append(chunk_content)

            # 새 청크 시작
            current_chunk_pages = [(page_num, page_content)]
            current_size = page_size

    # 남은 페이지 처리
    if current_chunk_pages:
        chunk_content = merge_pages(current_chunk_pages)
        chunk_content = ensure_protected_region_integrity(chunk_content)
        chunks.append(chunk_content)

    # 너무 큰 청크는 추가 분할 (보호 영역 보호하면서)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_size * 1.5:
            # 매우 큰 청크: 보호 영역 보호하면서 분할
            sub_chunks = split_large_chunk_with_protected_regions(chunk, chunk_size, chunk_overlap, is_table_based, force_chunking)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks
