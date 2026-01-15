# chunking_helper/protected_regions.py
"""
Protected Regions - 보호 영역 감지 및 처리

주요 기능:
- 청킹 시 분할되지 않아야 하는 보호 영역 감지
- 보호 영역을 보호하면서 텍스트 분할
- 대형 테이블의 효율적 처리
"""
import logging
import re
from typing import List, Tuple

from .constants import (
    HTML_TABLE_PATTERN, CHART_BLOCK_PATTERN, TEXTBOX_BLOCK_PATTERN,
    IMAGE_TAG_PATTERN, MARKDOWN_TABLE_PATTERN
)
# 순환 import 방지를 위해 함수 내에서 import
# from .table_chunker import chunk_large_table

logger = logging.getLogger("document-processor")


def find_protected_regions(
    text: str,
    is_table_based: bool = False,
    force_chunking: bool = False
) -> List[Tuple[int, int, str]]:
    """
    청킹 시 분할되지 않아야 하는 보호 영역을 찾습니다.

    보호 영역:
    1. HTML 테이블: <table border='1'>...</table> (force_chunking/테이블 기반에서는 row 단위만 보호)
    2. 차트 블록: [chart]...[/chart] - 항상 보호 (어떤 조건에서도 분할 불가)
    3. 텍스트박스 블록: [textbox]...[/textbox] - 항상 보호 (어떤 조건에서도 분할 불가)
    4. 이미지 태그: [image:...], [Image: {...}] 등 - 항상 보호 (어떤 조건에서도 분할 불가)
    5. Markdown 테이블: |...|\n|---|...|\n|...| (force_chunking/테이블 기반에서는 row 단위만 보호)

    Args:
        text: 검색할 텍스트
        is_table_based: 테이블 기반 파일 여부 (True이면 테이블 전체 보호 대신 row 단위 보호)
        force_chunking: 강제 청킹 여부 (True이면 테이블 기반과 동일하게 row 단위 보호)

    Returns:
        [(start, end, type), ...] - 정렬된 보호 영역 리스트
    """
    regions: List[Tuple[int, int, str]] = []

    # 테이블 보호 해제 조건: is_table_based 또는 force_chunking이 True인 경우
    disable_table_protection = is_table_based or force_chunking

    # 1. HTML 테이블 (보호 해제 시 row 단위만 보호)
    if not disable_table_protection:
        for match in re.finditer(HTML_TABLE_PATTERN, text, re.DOTALL | re.IGNORECASE):
            regions.append((match.start(), match.end(), 'html_table'))
    # else: HTML 테이블은 row 단위 청킹 허용 (_chunk_html_table_by_rows에서 처리)

    # 2. 차트 블록 - 항상 보호 (어떤 조건에서도 분할 불가)
    for match in re.finditer(CHART_BLOCK_PATTERN, text, re.DOTALL):
        regions.append((match.start(), match.end(), 'chart'))

    # 3. 텍스트박스 블록 - 항상 보호 (어떤 조건에서도 분할 불가)
    for match in re.finditer(TEXTBOX_BLOCK_PATTERN, text, re.DOTALL):
        regions.append((match.start(), match.end(), 'textbox'))

    # 4. 이미지 태그 - 항상 보호 (어떤 조건에서도 분할 불가)
    # 형식: [image:path], [Image: {path}], [image : path] 등
    for match in re.finditer(IMAGE_TAG_PATTERN, text):
        regions.append((match.start(), match.end(), 'image_tag'))

    # 5. Markdown 테이블 (보호 해제 시 row 단위만 보호)
    if not disable_table_protection:
        for match in re.finditer(MARKDOWN_TABLE_PATTERN, text, re.MULTILINE):
            # 전체 매치에서 테이블 시작 위치 찾기
            table_start = match.start()
            if match.group(0).startswith('\n'):
                table_start += 1
            table_end = match.end()
            regions.append((table_start, table_end, 'markdown_table'))
    # else: Markdown 테이블은 row 단위 청킹 허용 (_split_markdown_table_by_rows에서 처리)

    # 시작 위치로 정렬
    regions.sort(key=lambda x: x[0])

    # 겹치는 영역 병합
    merged_regions: List[Tuple[int, int, str]] = []
    for start, end, region_type in regions:
        if merged_regions and start < merged_regions[-1][1]:
            # 이전 영역과 겹침 -> 병합
            prev_start, prev_end, prev_type = merged_regions[-1]
            merged_regions[-1] = (prev_start, max(prev_end, end), f"{prev_type}+{region_type}")
        else:
            merged_regions.append((start, end, region_type))

    return merged_regions


def get_protected_region_positions(regions: List[Tuple[int, int, str]]) -> List[Tuple[int, int]]:
    """
    보호 영역에서 (start, end) 튜플만 추출합니다.
    """
    return [(start, end) for start, end, _ in regions]


def ensure_protected_region_integrity(content: str) -> str:
    """
    청크 내 보호 영역(HTML 테이블, 차트 블록, Markdown 테이블)이 완전한지 확인합니다.
    불완전한 보호 영역이 있으면 경고만 로깅 (내용은 유지).
    """
    # HTML 테이블 무결성 검사
    open_tables = len(re.findall(r'<table\s+border=["\']1["\']>', content, re.IGNORECASE))
    close_tables = len(re.findall(r'</table>', content, re.IGNORECASE))
    if open_tables != close_tables:
        logger.warning(f"Incomplete HTML table detected in chunk: {open_tables} open, {close_tables} close tags")

    # 차트 블록 무결성 검사
    open_charts = len(re.findall(r'\[chart\]', content))
    close_charts = len(re.findall(r'\[/chart\]', content))
    if open_charts != close_charts:
        logger.warning(f"Incomplete chart block detected in chunk: {open_charts} open, {close_charts} close tags")

    return content


def split_with_protected_regions(
    text: str,
    protected_regions: List[Tuple[int, int]],
    chunk_size: int,
    chunk_overlap: int,
    force_chunking: bool = False
) -> List[str]:
    """
    보호 영역(HTML 테이블, 차트 블록, Markdown 테이블)을 보호하면서 텍스트를 chunk_size 기준으로 분할합니다.

    알고리즘:
    1. 현재 위치에서 chunk_size만큼 앞으로 이동
    2. 그 지점이 보호 영역 안이면 → 보호 영역 시작점 직전에서 자르거나, 보호 영역 끝까지 포함
    3. 보호 영역이 chunk_size보다 크면:
       - HTML 테이블이면 → _chunk_large_table로 효율적 분할
       - 그 외(차트 등)면 → 보호 영역 단독 청크
    4. overlap 적용하여 다음 청크 시작점 계산

    이미지 태그 처리:
    - 이미지 태그는 중간 분할만 방지 (chunk_size 지점이 이미지 중간이면 이미지 끝까지 확장)
    - 이미지로 끝나면 overlap 없이 다음 청크 시작
    - 여러 이미지가 연속되어도 chunk_size 내에서 함께 포함 가능

    force_chunking 처리:
    - force_chunking=True일 때 protected_regions에 테이블이 없더라도
    - 텍스트 내 HTML 테이블을 직접 스캔하여 테이블 중간에서 자르지 않음
    - 큰 테이블은 chunk_large_table로 분할
    """
    # 이미지 태그 위치 별도 추출 (중간 분할 방지용)
    image_regions = []
    for match in re.finditer(IMAGE_TAG_PATTERN, text):
        image_regions.append((match.start(), match.end()))

    # 테이블/차트 등 블록 보호 영역 (이미지 제외)
    # protected_regions에서 이미지 태그가 아닌 것만 필터링
    block_regions = []
    for t_start, t_end in protected_regions:
        is_image = False
        for img_start, img_end in image_regions:
            if t_start == img_start and t_end == img_end:
                is_image = True
                break
        if not is_image:
            block_regions.append((t_start, t_end))

    # force_chunking 시 HTML 테이블 위치 직접 스캔
    # (protected_regions에 등록되지 않은 테이블도 처리하기 위함)
    html_table_regions = []
    if force_chunking:
        for match in re.finditer(HTML_TABLE_PATTERN, text, re.DOTALL | re.IGNORECASE):
            t_start, t_end = match.start(), match.end()
            # 이미 block_regions에 있는지 확인
            already_in_block = any(
                bs <= t_start and be >= t_end
                for bs, be in block_regions
            )
            if not already_in_block:
                html_table_regions.append((t_start, t_end))

    # block_regions에 HTML 테이블 영역 추가 (중복 없이)
    all_block_regions = block_regions + html_table_regions
    # 시작 위치로 정렬
    all_block_regions.sort(key=lambda x: x[0])

    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        # 남은 텍스트가 chunk_size 이하면 마지막 청크
        remaining = text_len - current_pos
        if remaining <= chunk_size:
            chunk = text[current_pos:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # chunk_size 지점 계산
        tentative_end = current_pos + chunk_size

        # 이 범위 내에 블록 보호영역(테이블/차트)이 있는지 확인
        block_in_range = None
        for t_start, t_end in all_block_regions:
            if t_start < tentative_end and t_end > current_pos:
                block_in_range = (t_start, t_end)
                break

        if block_in_range:
            t_start, t_end = block_in_range
            table_size = t_end - t_start

            if t_start <= current_pos:
                # 현재 위치가 이미 테이블 안 또는 테이블 시작점
                if table_size > chunk_size:
                    # 테이블이 chunk_size보다 큼
                    table_content = text[t_start:t_end].strip()

                    # HTML 테이블인지 확인하고 효율적으로 분할
                    if table_content.startswith('<table'):
                        from .table_chunker import chunk_large_table
                        table_chunks = chunk_large_table(table_content, chunk_size, chunk_overlap, "")
                        chunks.extend(table_chunks)
                    else:
                        # 차트 등 → 단독 청크
                        if table_content:
                            chunks.append(table_content)

                    current_pos = t_end
                else:
                    # 테이블이 chunk_size 이하 → 테이블 + 뒤 텍스트 포함 시도
                    end_pos = min(t_end + (chunk_size - table_size), text_len)

                    # 다음 블록 보호영역(테이블/차트)과 충돌하는지 확인 (이미지는 제외)
                    for next_t_start, next_t_end in all_block_regions:
                        if next_t_start > t_end and next_t_start < end_pos:
                            end_pos = next_t_start
                            break

                    # end_pos가 이미지 태그 중간이면 이미지 끝까지 확장
                    end_pos, ends_with_image = _adjust_for_image_boundary(end_pos, image_regions, text_len)

                    chunk = text[current_pos:end_pos].strip()
                    if chunk:
                        chunks.append(chunk)

                    # 이미지로 끝나면 overlap 없이, 아니면 overlap 적용
                    if ends_with_image:
                        current_pos = end_pos
                    else:
                        current_pos = max(t_end, end_pos - chunk_overlap)
            else:
                # 테이블이 청크 중간에 있음
                space_before_table = t_start - current_pos
                space_with_table = t_end - current_pos

                if space_with_table <= chunk_size:
                    # 테이블 전체를 포함할 수 있음 → 테이블까지 포함
                    end_pos = t_end

                    # 남은 공간에 뒤 텍스트도 추가 가능한지
                    remaining_space = chunk_size - space_with_table
                    if remaining_space > 0:
                        potential_end = min(t_end + remaining_space, text_len)

                        # 다음 블록 보호영역과 충돌 확인 (이미지는 제외)
                        for next_t_start, next_t_end in all_block_regions:
                            if next_t_start > t_end and next_t_start < potential_end:
                                potential_end = next_t_start
                                break

                        end_pos = potential_end

                    # end_pos가 이미지 태그 중간이면 이미지 끝까지 확장
                    end_pos, ends_with_image = _adjust_for_image_boundary(end_pos, image_regions, text_len)

                    chunk = text[current_pos:end_pos].strip()
                    if chunk:
                        chunks.append(chunk)

                    # 이미지로 끝나면 overlap 없이
                    if ends_with_image:
                        current_pos = end_pos
                    else:
                        current_pos = max(t_end, end_pos - chunk_overlap)
                else:
                    # 테이블 전체를 포함할 수 없음
                    if space_before_table > chunk_overlap:
                        # 테이블 앞 텍스트를 먼저 청크로 분리
                        end_pos = t_start
                        # end_pos가 이미지 태그 중간이면 이미지 끝까지 확장
                        end_pos, ends_with_image = _adjust_for_image_boundary(end_pos, image_regions, text_len)

                        chunk = text[current_pos:end_pos].strip()
                        if chunk:
                            chunks.append(chunk)

                        if ends_with_image:
                            current_pos = end_pos
                        else:
                            current_pos = max(current_pos + 1, t_start - chunk_overlap)
                    else:
                        # 테이블 앞 공간이 너무 작음 → 테이블 처리
                        table_content = text[t_start:t_end].strip()

                        # 테이블이 chunk_size보다 크면 분할
                        if table_size > chunk_size and table_content.startswith('<table'):
                            from .table_chunker import chunk_large_table
                            table_chunks = chunk_large_table(table_content, chunk_size, chunk_overlap, "")
                            chunks.extend(table_chunks)
                        else:
                            # 테이블 단독 청크 또는 차트
                            if table_content:
                                chunks.append(table_content)
                        current_pos = t_end
        else:
            # 블록 보호영역 없음 → 정상 분할점 찾기
            best_split = tentative_end

            # 문단 구분자 찾기
            search_start = max(current_pos, tentative_end - 200)
            para_match = None
            for m in re.finditer(r'\n\s*\n', text[search_start:tentative_end]):
                para_match = m

            if para_match:
                best_split = search_start + para_match.end()
            else:
                # 줄바꿈 찾기
                newline_pos = text.rfind('\n', current_pos, tentative_end)
                if newline_pos > current_pos + chunk_size // 2:
                    best_split = newline_pos + 1
                else:
                    # 공백 찾기
                    space_pos = text.rfind(' ', current_pos, tentative_end)
                    if space_pos > current_pos + chunk_size // 2:
                        best_split = space_pos + 1

            # best_split이 이미지 태그 중간이면 이미지 끝까지 확장
            best_split, ends_with_image = _adjust_for_image_boundary(best_split, image_regions, text_len)

            chunk = text[current_pos:best_split].strip()
            if chunk:
                chunks.append(chunk)

            # 이미지로 끝나면 overlap 없이, 아니면 overlap 적용
            if ends_with_image:
                current_pos = best_split
            else:
                current_pos = best_split - chunk_overlap
                if current_pos < 0:
                    current_pos = best_split

    return chunks


def _adjust_for_image_boundary(
    pos: int,
    image_regions: List[Tuple[int, int]],
    text_len: int
) -> Tuple[int, bool]:
    """
    주어진 위치가 이미지 태그 중간인지 확인하고, 중간이면 이미지 끝으로 조정합니다.

    Args:
        pos: 현재 분할 위치
        image_regions: 이미지 태그 위치 리스트 [(start, end), ...]
        text_len: 전체 텍스트 길이

    Returns:
        (adjusted_pos, ends_with_image): 조정된 위치와 이미지로 끝나는지 여부
    """
    for img_start, img_end in image_regions:
        # 분할 위치가 이미지 태그 중간에 있으면
        if img_start < pos < img_end:
            # 이미지 끝까지 확장
            return min(img_end, text_len), True
        # 분할 위치가 이미지 태그 바로 뒤 (공백/줄바꿈 포함)이면
        if img_end <= pos <= img_end + 5:
            return pos, True
    return pos, False


def split_large_chunk_with_protected_regions(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    is_table_based: bool = False,
    force_chunking: bool = False
) -> List[str]:
    """
    큰 청크를 보호 영역(HTML 테이블, 차트, Markdown 테이블)을 보호하면서 분할합니다.
    force_chunking 시 테이블 보호 해제 (차트는 항상 보호, row 단위는 보호)

    force_chunking=True일 때:
    - find_protected_regions에서 테이블이 보호 영역으로 등록되지 않음
    - 하지만 split_with_protected_regions에서 테이블을 직접 스캔하여 처리
    """
    protected_regions = find_protected_regions(text, is_table_based, force_chunking)
    protected_positions = get_protected_region_positions(protected_regions)

    # force_chunking 시에도 split_with_protected_regions가 테이블을 처리함
    # (split_with_protected_regions가 force_chunking=True일 때 테이블을 직접 스캔)
    return split_with_protected_regions(text, protected_positions, chunk_size, chunk_overlap, force_chunking)


# 하위 호환성을 위한 별칭
def ensure_table_integrity(content: str, table_pattern: str) -> str:
    """Deprecated: Use ensure_protected_region_integrity instead."""
    return ensure_protected_region_integrity(content)


def split_large_chunk_with_table_protection(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """Deprecated: Use split_large_chunk_with_protected_regions instead."""
    return split_large_chunk_with_protected_regions(text, chunk_size, chunk_overlap, False)
