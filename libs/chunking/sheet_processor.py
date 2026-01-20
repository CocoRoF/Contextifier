# chunking_helper/sheet_processor.py
"""
Sheet Processor - 시트 및 메타데이터 처리

주요 기능:
- 문서 메타데이터 추출
- 시트 섹션 추출
- 다중 시트 콘텐츠 청킹
- 단일 테이블 콘텐츠 청킹
"""
import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger("document-processor")


def extract_document_metadata(text: str) -> Tuple[Optional[str], str]:
    """
    텍스트에서 Document-Metadata 블록을 추출합니다.

    Args:
        text: 원본 텍스트

    Returns:
        (metadata_block, remaining_text) 튜플
    """
    metadata_pattern = r'<Document-Metadata>.*?</Document-Metadata>\s*'
    match = re.search(metadata_pattern, text, re.DOTALL)

    if match:
        metadata_block = match.group(0).strip()
        remaining_text = text[:match.start()] + text[match.end():]
        return metadata_block, remaining_text.strip()

    return None, text


def prepend_metadata_to_chunks(chunks: List[str], metadata: Optional[str]) -> List[str]:
    """
    각 청크에 메타데이터를 추가합니다.

    Args:
        chunks: 청크 리스트
        metadata: 메타데이터 블록

    Returns:
        메타데이터가 추가된 청크 리스트
    """
    if not metadata:
        return chunks
    return [f"{metadata}\n\n{chunk}" for chunk in chunks]


def extract_sheet_sections(text: str) -> List[Tuple[str, str]]:
    """
    Excel 시트 섹션을 추출합니다.

    Args:
        text: 전체 텍스트

    Returns:
        [(sheet_name, sheet_content), ...] 리스트
    """
    # Sheet marker pattern - only standard format from PageTagProcessor
    sheet_pattern = r'\[Sheet:\s*([^\]]+)\]'
    marker_template = '[Sheet: {name}]'

    matches = list(re.finditer(sheet_pattern, text))

    if not matches:
        return []

    sheets = []

    for i, match in enumerate(matches):
        sheet_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start:end].strip()
        if content:
            # 시트 마커를 콘텐츠에 포함
            sheet_marker = marker_template.format(name=sheet_name)
            full_content = f"{sheet_marker}\n{content}"
            sheets.append((sheet_name, full_content))

    return sheets


def extract_content_segments(content: str) -> List[Tuple[str, str]]:
    """
    콘텐츠에서 다양한 타입의 세그먼트를 추출합니다.

    세그먼트 타입:
    - table: HTML 테이블 또는 Markdown 테이블 (테이블 마커 포함)
    - textbox: [textbox]...[/textbox] 블록
    - chart: [chart]...[/chart] 블록
    - image: [image:...] 태그
    - text: 일반 텍스트

    Args:
        content: 파싱할 콘텐츠

    Returns:
        [(segment_type, segment_content), ...] 리스트
    """
    segments: List[Tuple[str, str]] = []

    # 각 특수 블록 패턴 정의
    # [테이블 N] 마커와 함께 테이블을 하나의 블록으로 인식
    patterns = [
        # [테이블 N] + HTML 테이블
        ('table', r'(?:\[테이블\s*\d+\]\s*)?<table\s+border=["\']1["\']>.*?</table>'),
        # [테이블 N] + Markdown 테이블 (| 로 시작하는 여러 줄, 마지막 행 줄바꿈 없어도 매칭)
        ('table', r'\[테이블\s*\d+\]\s*\n(?:\|[^\n]*\|(?:\s*\n|$))+'),
        # 단독 Markdown 테이블 (| 로 시작하고 --- 구분선이 있는 형태, 마지막 행 줄바꿈 없어도 매칭)
        ('table', r'(?:^|\n)(\|[^\n]*\|\s*\n\|[\s\-:]*\|[^\n]*(?:\n\|[^\n]*\|)*)'),
        ('textbox', r'\[textbox\].*?\[/textbox\]'),
        ('chart', r'\[chart\].*?\[/chart\]'),
        ('image', r'\[(?i:image)\s*:\s*[^\]]+\]'),
    ]

    # 모든 특수 블록 위치 찾기
    all_matches: List[Tuple[int, int, str, str]] = []  # (start, end, type, content)

    for segment_type, pattern in patterns:
        for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE | re.MULTILINE):
            matched_content = match.group(0).strip()
            # 빈 매치 무시
            if not matched_content:
                continue
            all_matches.append((match.start(), match.end(), segment_type, matched_content))

    # 시작 위치로 정렬
    all_matches.sort(key=lambda x: x[0])

    # 겹치는 매치 제거 (더 긴 매치 우선)
    filtered_matches: List[Tuple[int, int, str, str]] = []
    last_end = 0
    for start, end, segment_type, segment_content in all_matches:
        if start >= last_end:
            filtered_matches.append((start, end, segment_type, segment_content))
            last_end = end

    # 세그먼트 구성 (특수 블록 + 그 사이의 일반 텍스트)
    current_pos = 0
    for start, end, segment_type, segment_content in filtered_matches:
        # 특수 블록 전의 일반 텍스트
        if start > current_pos:
            text_between = content[current_pos:start].strip()
            # [테이블 N] 마커만 있는 텍스트는 다음 테이블과 합치기 위해 건너뜀
            if text_between and not re.match(r'^\[테이블\s*\d+\]\s*$', text_between):
                segments.append(('text', text_between))

        # 특수 블록
        segments.append((segment_type, segment_content))
        current_pos = end

    # 마지막 특수 블록 후의 일반 텍스트
    if current_pos < len(content):
        remaining_text = content[current_pos:].strip()
        # [테이블 N] 마커만 있는 텍스트는 무시
        if remaining_text and not re.match(r'^\[테이블\s*\d+\]\s*$', remaining_text):
            segments.append(('text', remaining_text))

    return segments


def chunk_multi_sheet_content(
    sheets: List[Tuple[str, str]],
    metadata_block: Optional[str],
    analysis_block: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_plain_text_func,
    chunk_large_table_func
) -> List[str]:
    """
    다중 시트 콘텐츠를 청킹합니다.

    각 시트를 독립적으로 처리하고, 필요시 분할합니다.
    모든 청크에 메타데이터와 시트 정보를 포함합니다.
    테이블뿐 아니라 테이블 전후의 추가 콘텐츠(textbox, chart, image 등)도 처리합니다.

    Args:
        sheets: [(sheet_name, sheet_content), ...] 리스트
        metadata_block: 메타데이터 블록
        analysis_block: 분석 블록
        chunk_size: 청크 크기
        chunk_overlap: 청크 겹침
        chunk_plain_text_func: 일반 텍스트 청킹 함수
        chunk_large_table_func: 대용량 테이블 청킹 함수

    Returns:
        청크 리스트
    """
    all_chunks: List[str] = []

    # 공통 메타데이터 구성 (모든 청크에 포함)
    common_metadata_parts = []
    if metadata_block:
        common_metadata_parts.append(metadata_block)
    if analysis_block:
        common_metadata_parts.append(analysis_block)
    common_metadata = "\n\n".join(common_metadata_parts) if common_metadata_parts else ""

    for sheet_idx, (sheet_name, sheet_content) in enumerate(sheets):
        # Extract sheet marker - only standard format
        sheet_marker_match = re.match(r'(\[Sheet:\s*[^\]]+\])', sheet_content)
        sheet_marker = sheet_marker_match.group(1) if sheet_marker_match else f"[Sheet: {sheet_name}]"

        # 이 시트에 대한 컨텍스트 구성 (메타데이터 + 시트 정보)
        context_parts = []
        if common_metadata:
            context_parts.append(common_metadata)
        context_parts.append(sheet_marker)
        context_prefix = "\n\n".join(context_parts) if context_parts else ""

        # 시트 콘텐츠에서 시트 마커 제거
        content_after_marker = sheet_content
        if sheet_marker_match:
            content_after_marker = sheet_content[sheet_marker_match.end():].strip()

        # === 시트 콘텐츠를 세그먼트로 분리 ===
        # 세그먼트: 테이블, textbox, chart, image 등의 블록과 일반 텍스트
        segments = extract_content_segments(content_after_marker)

        if not segments:
            # 빈 시트는 건너뛰기
            continue

        # 각 세그먼트 처리
        for segment_type, segment_content in segments:
            if not segment_content.strip():
                continue

            segment_size = len(segment_content)

            if segment_type == 'table':
                # 테이블 처리
                if segment_size + len(context_prefix) <= chunk_size:
                    all_chunks.append(f"{context_prefix}\n{segment_content}")
                else:
                    # 대용량 테이블: 분할
                    table_chunks = chunk_large_table_func(
                        segment_content, chunk_size, chunk_overlap,
                        context_prefix=context_prefix
                    )
                    all_chunks.extend(table_chunks)

            elif segment_type in ('textbox', 'chart', 'image'):
                # 보호 블록: 분할하지 않고 하나의 청크로
                if len(context_prefix) + segment_size > chunk_size:
                    # 청크 크기 초과해도 분할하지 않음 (보호 블록)
                    logger.warning(f"{segment_type} block exceeds chunk_size, but keeping it intact")
                all_chunks.append(f"{context_prefix}\n{segment_content}")

            else:
                # 일반 텍스트
                if len(context_prefix) + segment_size <= chunk_size:
                    all_chunks.append(f"{context_prefix}\n{segment_content}")
                else:
                    # 긴 일반 텍스트는 분할
                    text_chunks = chunk_plain_text_func(segment_content, chunk_size, chunk_overlap)
                    for chunk in text_chunks:
                        all_chunks.append(f"{context_prefix}\n{chunk}")

    logger.info(f"Multi-sheet content split into {len(all_chunks)} chunks")

    return all_chunks


def chunk_single_table_content(
    text: str,
    metadata_block: Optional[str],
    analysis_block: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_plain_text_func,
    chunk_large_table_func
) -> List[str]:
    """
    단일 테이블 콘텐츠를 청킹합니다.
    모든 청크에 메타데이터를 포함합니다.

    Args:
        text: 테이블 포함 텍스트
        metadata_block: 메타데이터 블록
        analysis_block: 분석 블록
        chunk_size: 청크 크기
        chunk_overlap: 청크 겹침
        chunk_plain_text_func: 일반 텍스트 청킹 함수
        chunk_large_table_func: 대용량 테이블 청킹 함수

    Returns:
        청크 리스트
    """
    # 컨텍스트 구성 (모든 청크에 포함)
    context_parts = []
    if metadata_block:
        context_parts.append(metadata_block)
    if analysis_block:
        context_parts.append(analysis_block)
    context_prefix = "\n\n".join(context_parts) if context_parts else ""

    # 테이블 추출
    table_pattern = r'<table\s+border=["\']1["\']>.*?</table>'
    table_matches = list(re.finditer(table_pattern, text, re.DOTALL | re.IGNORECASE))

    if not table_matches:
        # 테이블이 없으면 일반 청킹
        full_text = text
        if context_prefix:
            full_text = f"{context_prefix}\n\n{full_text}"
        return chunk_plain_text_func(full_text, chunk_size, chunk_overlap)

    # 결과 청크들
    all_chunks: List[str] = []

    # 각 테이블 처리
    for match in table_matches:
        table_html = match.group(0)
        table_size = len(table_html)

        logger.debug(f"Processing table: {table_size} chars")

        if table_size + len(context_prefix) <= chunk_size:
            # 작은 테이블: 컨텍스트와 함께
            if context_prefix:
                all_chunks.append(f"{context_prefix}\n\n{table_html}")
            else:
                all_chunks.append(table_html)
        else:
            # 대용량 테이블: 분할 (컨텍스트를 모든 청크에 포함)
            table_chunks = chunk_large_table_func(
                table_html, chunk_size, chunk_overlap,
                context_prefix=context_prefix
            )
            all_chunks.extend(table_chunks)

    logger.info(f"Single table content split into {len(all_chunks)} chunks")

    return all_chunks
