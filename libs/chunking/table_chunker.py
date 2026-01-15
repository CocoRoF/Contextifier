# chunking_helper/table_chunker.py
"""
Table Chunker - 테이블 청킹 핵심 로직

주요 기능:
- 대용량 HTML 테이블을 chunk_size에 맞게 분할
- 테이블 구조(헤더) 보존 및 복원
- rowspan/colspan 인식 분할
- rowspan 재조정
"""
import logging
import re
from typing import Dict, List

from .constants import (
    ParsedTable, TableRow,
    TABLE_WRAPPER_OVERHEAD, CHUNK_INDEX_OVERHEAD
)
from .table_parser import (
    parse_html_table, extract_cell_spans_with_positions, has_complex_spans
)

logger = logging.getLogger("document-processor")


def calculate_available_space(
    chunk_size: int,
    header_size: int,
    chunk_index: int = 0,
    total_chunks: int = 1
) -> int:
    """
    청크에서 데이터 행을 위해 사용 가능한 공간을 계산합니다.

    Args:
        chunk_size: 전체 청크 크기
        header_size: 헤더 크기
        chunk_index: 현재 청크 인덱스 (0부터)
        total_chunks: 예상 총 청크 수

    Returns:
        데이터 행을 위해 사용 가능한 문자 수
    """
    # 고정 오버헤드
    overhead = TABLE_WRAPPER_OVERHEAD

    # 청크 인덱스 메타데이터 오버헤드 (총 청크가 2개 이상일 때만)
    if total_chunks > 1:
        overhead += CHUNK_INDEX_OVERHEAD

    # 헤더 오버헤드 (첫 번째 청크가 아닐 때도 헤더 포함)
    overhead += header_size

    available = chunk_size - overhead

    return max(available, 100)  # 최소 100자는 보장


def adjust_rowspan_in_chunk(rows_html: List[str], total_rows_in_chunk: int) -> List[str]:
    """
    청크 내 행들의 rowspan을 재조정합니다.

    청크에 포함된 행 수에 맞게 rowspan 값을 조정하여
    테이블이 올바르게 렌더링되도록 합니다.

    Args:
        rows_html: 청크에 포함된 행들의 HTML 리스트
        total_rows_in_chunk: 청크 내 총 행 수

    Returns:
        rowspan이 조정된 행들의 HTML 리스트
    """
    if not rows_html:
        return rows_html

    adjusted_rows = []

    for row_idx, row_html in enumerate(rows_html):
        remaining_rows = total_rows_in_chunk - row_idx

        def adjust_cell_rowspan(match):
            """셀의 rowspan을 조정하는 콜백 함수"""
            tag = match.group(1)  # td 또는 th
            attrs = match.group(2)
            content = match.group(3)

            # 현재 rowspan 추출
            rowspan_match = re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
            if rowspan_match:
                original_rowspan = int(rowspan_match.group(1))

                # 남은 행 수보다 크면 조정
                adjusted_rowspan = min(original_rowspan, remaining_rows)

                if adjusted_rowspan <= 1:
                    # rowspan=1이면 속성 제거
                    new_attrs = re.sub(r'\s*rowspan=["\']?\d+["\']?', '', attrs, flags=re.IGNORECASE)
                else:
                    # rowspan 값 조정
                    new_attrs = re.sub(
                        r'rowspan=["\']?\d+["\']?',
                        f"rowspan='{adjusted_rowspan}'",
                        attrs,
                        flags=re.IGNORECASE
                    )

                return f'<{tag}{new_attrs}>{content}</{tag}>'

            return match.group(0)

        # 셀 패턴: <td ...>...</td> 또는 <th ...>...</th>
        cell_pattern = r'<(td|th)([^>]*)>(.*?)</\1>'
        adjusted_row = re.sub(cell_pattern, adjust_cell_rowspan, row_html, flags=re.DOTALL | re.IGNORECASE)

        adjusted_rows.append(adjusted_row)

    return adjusted_rows


def build_table_chunk(
    header_html: str,
    data_rows: List[TableRow],
    chunk_index: int = 0,
    total_chunks: int = 1,
    context_prefix: str = ""
) -> str:
    """
    청크용 완전한 테이블 HTML을 구성합니다.

    rowspan이 청크 범위를 초과하는 경우 자동으로 조정합니다.

    Args:
        header_html: 헤더 행들의 HTML
        data_rows: 데이터 행들
        chunk_index: 현재 청크 인덱스 (0부터)
        total_chunks: 총 청크 수
        context_prefix: 컨텍스트 정보 (메타데이터, 시트 정보 등) - 모든 청크에 포함

    Returns:
        완전한 테이블 HTML
    """
    parts = []

    # 컨텍스트 정보 (메타데이터, 시트 정보 등) - 모든 청크에 포함
    if context_prefix:
        parts.append(context_prefix)

    # 청크 인덱스 메타데이터 (2개 이상일 때만)
    if total_chunks > 1:
        parts.append(f"[테이블 청크 {chunk_index + 1}/{total_chunks}]")

    # 테이블 시작
    parts.append("<table border='1'>")

    # 헤더 (있는 경우)
    if header_html:
        parts.append(header_html)

    # 데이터 행들의 HTML 추출
    rows_html = [row.html for row in data_rows]

    # rowspan 조정
    adjusted_rows = adjust_rowspan_in_chunk(rows_html, len(data_rows))

    # 조정된 행들 추가
    for row_html in adjusted_rows:
        parts.append(row_html)

    # 테이블 종료
    parts.append("</table>")

    return "\n".join(parts)


def update_chunk_metadata(chunks: List[str], total_chunks: int) -> List[str]:
    """
    청크들의 메타데이터(총 청크 수)를 업데이트합니다.
    """
    updated_chunks = []

    for idx, chunk in enumerate(chunks):
        # 기존 메타데이터 패턴
        old_pattern = r'\[테이블 청크 \d+/\d+\]'
        new_metadata = f"[테이블 청크 {idx + 1}/{total_chunks}]"

        if re.search(old_pattern, chunk):
            updated_chunk = re.sub(old_pattern, new_metadata, chunk)
        else:
            # 메타데이터가 없으면 추가
            updated_chunk = f"{new_metadata}\n{chunk}"

        updated_chunks.append(updated_chunk)

    return updated_chunks


def split_table_into_chunks(
    parsed_table: ParsedTable,
    chunk_size: int,
    chunk_overlap: int = 0,
    context_prefix: str = ""
) -> List[str]:
    """
    파싱된 테이블을 chunk_size에 맞게 분할합니다.
    각 청크는 완전한 테이블 구조를 가집니다 (헤더 포함).

    주의: 테이블 청킹에서는 overlap이 적용되지 않습니다.
    테이블 데이터의 중복은 검색 품질을 저하시키므로 의도적으로 제외합니다.

    Args:
        parsed_table: 파싱된 테이블 정보
        chunk_size: 청크 최대 크기
        chunk_overlap: 사용되지 않음 (호환성을 위해 유지)
        context_prefix: 컨텍스트 정보 (메타데이터, 시트 정보 등) - 모든 청크에 포함

    Returns:
        분할된 테이블 HTML 청크 리스트
    """
    data_rows = parsed_table.data_rows
    header_html = parsed_table.header_html
    header_size = parsed_table.header_size

    # 컨텍스트 크기 계산
    context_size = len(context_prefix) + 2 if context_prefix else 0  # 줄바꿈 포함

    if not data_rows:
        # 데이터 행이 없으면 원본 반환
        return [parsed_table.original_html]

    # 예상 청크 수 계산 (대략적)
    total_data_size = sum(row.char_length for row in data_rows)
    available_per_chunk = calculate_available_space(chunk_size, header_size + context_size, 0, 1)
    estimated_chunks = max(1, (total_data_size + available_per_chunk - 1) // available_per_chunk)

    # 실제 청크 수로 다시 계산
    available_per_chunk = calculate_available_space(chunk_size, header_size + context_size, 0, estimated_chunks)

    chunks: List[str] = []
    current_rows: List[TableRow] = []
    current_size = 0
    # 테이블 청킹에서는 overlap을 적용하지 않음 (데이터 중복 방지)

    for row_idx, row in enumerate(data_rows):
        row_size = row.char_length + 1  # 줄바꿈 포함

        # 현재 청크에 추가 가능한지 확인
        if current_rows and (current_size + row_size > available_per_chunk):
            # 현재 청크 완료
            chunk_html = build_table_chunk(
                header_html,
                current_rows,
                chunk_index=len(chunks),
                total_chunks=estimated_chunks,
                context_prefix=context_prefix
            )
            chunks.append(chunk_html)

            # 새 청크 시작 (테이블에서는 overlap 없음)
            current_rows = []
            current_size = 0

        current_rows.append(row)
        current_size += row_size

    # 마지막 청크 처리
    if current_rows:
        chunk_html = build_table_chunk(
            header_html,
            current_rows,
            chunk_index=len(chunks),
            total_chunks=max(len(chunks) + 1, estimated_chunks),
            context_prefix=context_prefix
        )
        chunks.append(chunk_html)

    # 총 청크 수 업데이트하여 메타데이터 수정
    if len(chunks) != estimated_chunks and len(chunks) > 1:
        chunks = update_chunk_metadata(chunks, len(chunks))

    logger.info(f"Table split into {len(chunks)} chunks (original: {len(parsed_table.original_html)} chars)")

    return chunks


def split_table_preserving_rowspan(
    parsed_table: ParsedTable,
    chunk_size: int,
    chunk_overlap: int,
    context_prefix: str = ""
) -> List[str]:
    """
    rowspan을 고려하여 테이블을 분할합니다.

    rowspan으로 연결된 행들은 의미론적 블록 단위로 함께 유지합니다.

    주의: 테이블 청킹에서는 overlap이 적용되지 않습니다.
    테이블 데이터의 중복은 검색 품질을 저하시키므로 의도적으로 제외합니다.

    알고리즘:
    1. 각 행에서 활성 rowspan을 추적 (열 위치별, colspan 고려)
    2. 이전 행에서 모든 rowspan이 끝나고 새 rowspan이 시작되면 새 블록
    3. 블록들을 chunk_size에 맞게 조합

    Args:
        parsed_table: 파싱된 테이블
        chunk_size: 청크 크기
        chunk_overlap: 사용되지 않음 (호환성을 위해 유지)
        context_prefix: 컨텍스트 정보 (메타데이터, 시트 정보 등)

    Returns:
        분할된 테이블 청크 리스트
    """
    data_rows = parsed_table.data_rows
    header_html = parsed_table.header_html
    header_size = parsed_table.header_size

    # 컨텍스트 크기 계산
    context_size = len(context_prefix) + 2 if context_prefix else 0

    if not data_rows:
        if context_prefix:
            return [f"{context_prefix}\n{parsed_table.original_html}"]
        return [parsed_table.original_html]

    # === rowspan 블록 식별 ===
    # 블록 = rowspan으로 연결된 연속 행들의 그룹
    active_rowspans: Dict[int, int] = {}  # 열_위치 -> 남은_행_수 (현재 행 포함)
    row_block_ids: List[int] = []  # 각 행의 블록 ID
    current_block_id = -1

    for row_idx, row in enumerate(data_rows):
        # 1. 이전 행 처리 후 남은 rowspan 감소 (첫 행 제외)
        if row_idx > 0:
            finished_cols = []
            for col in list(active_rowspans.keys()):
                active_rowspans[col] -= 1
                if active_rowspans[col] <= 0:
                    finished_cols.append(col)
            for col in finished_cols:
                del active_rowspans[col]

        # 감소 후 상태 (새 span 추가 전)
        had_active_before_new = len(active_rowspans) > 0

        # 2. 현재 행에서 시작하는 새 rowspan 추가
        new_spans = extract_cell_spans_with_positions(row.html)
        for col, span in new_spans.items():
            # 기존 rowspan보다 크면 갱신 (더 긴 span이 우선)
            if col not in active_rowspans or span > active_rowspans[col]:
                active_rowspans[col] = span

        has_active_now = len(active_rowspans) > 0
        has_new_span = len(new_spans) > 0

        # 블록 결정 로직:
        # - 활성 rowspan이 없으면 독립 블록
        # - 이전 행 처리 후 활성이 없었는데 새 span이 시작되면 새 블록
        # - 그 외 기존 블록 유지
        if not has_active_now:
            # rowspan 없음 - 독립 행
            current_block_id += 1
            row_block_ids.append(current_block_id)
        elif not had_active_before_new and has_new_span:
            # 이전 rowspan 모두 끝나고 새 rowspan 시작 - 새 블록
            current_block_id += 1
            row_block_ids.append(current_block_id)
        else:
            # 기존 블록 유지
            row_block_ids.append(current_block_id)

    # 블록별로 행 그룹화
    block_groups: Dict[int, List[int]] = {}
    for row_idx, block_id in enumerate(row_block_ids):
        if block_id not in block_groups:
            block_groups[block_id] = []
        block_groups[block_id].append(row_idx)

    # 정렬된 블록 순서로 row_groups 생성
    row_groups: List[List[int]] = [
        block_groups[block_id]
        for block_id in sorted(block_groups.keys())
    ]

    # === 그룹들을 청크로 조합 ===
    chunks: List[str] = []
    current_rows: List[TableRow] = []
    current_size = 0

    available_space = calculate_available_space(chunk_size, header_size + context_size, 0, 1)

    for group in row_groups:
        group_rows = [data_rows[idx] for idx in group]
        group_size = sum(row.char_length + 1 for row in group_rows)

        if current_rows and current_size + group_size > available_space:
            # 현재 청크 완료
            chunks.append(build_table_chunk(
                header_html, current_rows, len(chunks), len(chunks) + 2,
                context_prefix=context_prefix
            ))
            current_rows = []
            current_size = 0

        current_rows.extend(group_rows)
        current_size += group_size

    # 마지막 청크
    if current_rows:
        chunks.append(build_table_chunk(
            header_html, current_rows, len(chunks), len(chunks) + 1,
            context_prefix=context_prefix
        ))

    # 청크 수 업데이트
    if len(chunks) > 1:
        chunks = update_chunk_metadata(chunks, len(chunks))

    return chunks


def chunk_large_table(
    table_html: str,
    chunk_size: int,
    chunk_overlap: int,
    context_prefix: str = ""
) -> List[str]:
    """
    대용량 테이블을 chunk_size에 맞게 분할합니다.
    테이블 구조(헤더)를 각 청크에서 복원합니다.

    rowspan이 있는 복잡한 테이블도 처리합니다.

    주의: 테이블 청킹에서는 overlap이 적용되지 않습니다.
    테이블 데이터의 중복은 검색 품질을 저하시키므로 의도적으로 제외합니다.

    Args:
        table_html: HTML 테이블 문자열
        chunk_size: 청크 최대 크기
        chunk_overlap: 사용되지 않음 (호환성을 위해 유지)
        context_prefix: 컨텍스트 정보 (메타데이터, 시트 정보 등) - 모든 청크에 포함

    Returns:
        분할된 테이블 청크 리스트
    """
    # 테이블 파싱
    parsed = parse_html_table(table_html)

    if not parsed:
        logger.warning("Failed to parse table, returning original")
        if context_prefix:
            return [f"{context_prefix}\n{table_html}"]
        return [table_html]

    # 테이블 크기가 chunk_size 이하면 분할 불필요
    if len(table_html) + len(context_prefix) <= chunk_size:
        if context_prefix:
            return [f"{context_prefix}\n{table_html}"]
        return [table_html]

    # 데이터 행이 없으면 분할 불필요
    if not parsed.data_rows:
        if context_prefix:
            return [f"{context_prefix}\n{table_html}"]
        return [table_html]

    # rowspan이 있는지 확인
    if has_complex_spans(table_html):
        logger.info("Complex table with rowspan detected, using span-aware splitting")
        return split_table_preserving_rowspan(parsed, chunk_size, chunk_overlap, context_prefix)

    # 일반 테이블 분할
    chunks = split_table_into_chunks(parsed, chunk_size, chunk_overlap, context_prefix)

    return chunks
