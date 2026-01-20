# chunking_helper/text_chunker.py
"""
Text Chunker - 텍스트 청킹

주요 기능:
- 일반 텍스트 청킹
- 테이블 없는 텍스트 청킹
- 행(row) 보호 청킹
- 코드 텍스트 청킹
"""
import logging
import re
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from libs.chunking.constants import LANGCHAIN_CODE_LANGUAGE_MAP

logger = logging.getLogger("document-processor")


def chunk_plain_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    일반 텍스트를 RecursiveCharacterTextSplitter로 청킹합니다.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_text(text)


def chunk_text_without_tables(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[str],
    prepend_metadata_func
) -> List[str]:
    """
    테이블이 없는 텍스트에 대한 기존 청킹 로직.

    Args:
        text: 청킹할 텍스트
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹침 크기
        metadata: 청크에 추가할 메타데이터
        prepend_metadata_func: 메타데이터 추가 함수

    Returns:
        청크 리스트
    """
    # HTML 코드 블록 처리
    html_code_pattern = r'```html\s*(.*?)\s*```'
    table_section_pattern = r'<div class="table-section">(.*?)</div>'
    combined_pattern = f'({html_code_pattern}|{table_section_pattern})'

    html_chunks = []
    matches = list(re.finditer(combined_pattern, text, re.DOTALL))

    if matches:
        current_pos = 0
        for m in matches:
            s, e = m.span()
            before = text[current_pos:s].strip()
            if before:
                html_chunks.append(('text', before))
            html_chunks.append(('html', text[s:e]))
            current_pos = e
        after = text[current_pos:].strip()
        if after:
            html_chunks.append(('text', after))
    else:
        html_chunks = [('text', text)]

    final_chunks: List[str] = []

    for kind, content in html_chunks:
        if kind == 'html':
            final_chunks.append(content)
            continue

        # 섹션 마커 처리
        section_markers = [
            "[색션 구분]", "[표 구분]", "[섹션 구분]",
            "<!-- [섹션 구분] -->", "<!-- [표 구분] -->", "<!-- [section] -->",
            '<div class="table-section">'
        ]

        if any(m in content for m in section_markers):
            pattern = r'(\[색션 구분\]|\[표 구분\]|\[섹션 구분\]|<!-- \[섹션 구분\] -->|<!-- \[표 구분\] -->|<!-- \[section\] -->|<div class="table-section">)'
            parts = re.split(pattern, content)
            majors = [p.strip() for p in parts if p.strip() and not any(m in p for m in section_markers)]

            merged: List[str] = []
            current = ""
            prev_tail = ""

            for sec in majors:
                pot = (current + "\n\n" + sec) if current else sec
                if len(pot) <= chunk_size:
                    current = pot
                else:
                    if current:
                        prev_tail = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                        merged.append(current)
                        current = (prev_tail + "\n\n" + sec) if prev_tail else sec
                    else:
                        current = sec

            if current:
                if prev_tail and len(merged) > 0 and not current.startswith(prev_tail):
                    current = prev_tail + "\n\n" + current
                merged.append(current)

            for sec in merged:
                if len(sec) > chunk_size * 2:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                        length_function=len, separators=["\n\n", "\n", " ", ""]
                    )
                    final_chunks.extend(splitter.split_text(sec))
                else:
                    final_chunks.append(sec)
        else:
            text_chunks = chunk_plain_text(content, chunk_size, chunk_overlap)
            final_chunks.extend(text_chunks)

    cleaned_chunks = clean_chunks(final_chunks)
    cleaned_chunks = prepend_metadata_func(cleaned_chunks, metadata)

    return cleaned_chunks


def chunk_with_row_protection(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    split_with_protected_regions_func,
    chunk_large_table_func
) -> List[str]:
    """
    테이블 보호 해제 시 row 단위로는 보호하면서 청킹합니다.

    HTML 테이블은 chunk_large_table_func로 처리하여 테이블 구조를 유지합니다.
    Markdown 테이블의 |...| 행은 중간에서 자르지 않고 보호합니다.

    Args:
        text: 청킹할 텍스트
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹침 크기
        split_with_protected_regions_func: 보호 영역 분할 함수
        chunk_large_table_func: 큰 테이블 청킹 함수

    Returns:
        청크 리스트
    """
    if not text or not text.strip():
        return []

    # === HTML 테이블을 먼저 추출하여 별도 처리 ===
    # HTML 테이블을 찾고, 테이블/비테이블 세그먼트로 분리
    table_pattern = r'<table[^>]*>.*?</table>'

    segments: List[Tuple[str, str]] = []  # [(type, content), ...]
    last_end = 0

    for match in re.finditer(table_pattern, text, re.DOTALL | re.IGNORECASE):
        # 테이블 이전 텍스트
        if match.start() > last_end:
            before_text = text[last_end:match.start()].strip()
            if before_text:
                segments.append(('text', before_text))

        # 테이블
        segments.append(('table', match.group(0)))
        last_end = match.end()

    # 마지막 테이블 이후 텍스트
    if last_end < len(text):
        after_text = text[last_end:].strip()
        if after_text:
            segments.append(('text', after_text))

    # 테이블이 없으면 기존 로직 사용
    if not any(seg_type == 'table' for seg_type, _ in segments):
        return chunk_with_row_protection_simple(
            text, chunk_size, chunk_overlap, split_with_protected_regions_func
        )

    # === 세그먼트별로 처리 ===
    all_chunks: List[str] = []

    for seg_type, content in segments:
        if seg_type == 'table':
            # HTML 테이블 → chunk_large_table_func로 효율적 분할
            table_chunks = chunk_large_table_func(content, chunk_size, chunk_overlap, "")
            all_chunks.extend(table_chunks)
        else:
            # 일반 텍스트 → Markdown row 보호하면서 청킹
            text_chunks = chunk_with_row_protection_simple(
                content, chunk_size, chunk_overlap, split_with_protected_regions_func
            )
            all_chunks.extend(text_chunks)

    return all_chunks


def chunk_with_row_protection_simple(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    split_with_protected_regions_func
) -> List[str]:
    """
    Markdown 테이블 행을 보호하면서 청킹합니다.
    HTML 테이블은 이미 분리되어 있다고 가정합니다.

    Args:
        text: 청킹할 텍스트
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹침 크기
        split_with_protected_regions_func: 보호 영역 분할 함수

    Returns:
        청크 리스트
    """
    if not text or not text.strip():
        return []

    # Markdown 테이블 행만 보호 (HTML 테이블은 이미 분리됨)
    row_patterns = [
        r'\|[^\n]+\|',  # Markdown 테이블 행 (헤더, 데이터, 구분선 모두 포함)
    ]

    # 모든 행 위치 찾기
    row_positions: List[Tuple[int, int]] = []
    for pattern in row_patterns:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            row_positions.append((match.start(), match.end()))

    # 위치로 정렬
    row_positions.sort(key=lambda x: x[0])

    # 겹치는 영역 병합
    merged_rows: List[Tuple[int, int]] = []
    for start, end in row_positions:
        if merged_rows and start < merged_rows[-1][1]:
            # 겹침 -> 병합
            prev_start, prev_end = merged_rows[-1]
            merged_rows[-1] = (prev_start, max(prev_end, end))
        else:
            merged_rows.append((start, end))

    if not merged_rows:
        # 행이 없으면 일반 청킹
        return chunk_plain_text(text, chunk_size, chunk_overlap)

    # 행을 보호하면서 청킹
    return split_with_protected_regions_func(text, merged_rows, chunk_size, chunk_overlap)


def clean_chunks(chunks: List[str]) -> List[str]:
    """
    Clean chunks: remove empty chunks and chunks with only page markers.
    """
    cleaned_chunks = []

    # Only recognize the standard format from PageTagProcessor
    page_marker_patterns = [
        r"\[Page Number:\s*\d+(\s*\(OCR[+Ref]*\))?\]",
        r"\[Slide Number:\s*\d+(\s*\(OCR\))?\]",
    ]

    for chunk in chunks:
        if not chunk.strip():
            continue

        # 페이지 마커만 있는지 확인
        is_page_marker_only = False
        for pattern in page_marker_patterns:
            if re.fullmatch(pattern, chunk.strip()):
                is_page_marker_only = True
                break

        if not is_page_marker_only:
            cleaned_chunks.append(chunk)

    return cleaned_chunks


def chunk_code_text(
    text: str,
    file_type: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 300
) -> List[str]:
    """
    코드 텍스트를 언어별 splitter로 청킹합니다.

    Args:
        text: 코드 텍스트
        file_type: 파일 확장자 (e.g., 'py', 'js')
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹침 크기

    Returns:
        청크 리스트
    """
    if not text or not text.strip():
        return [""]

    lang = LANGCHAIN_CODE_LANGUAGE_MAP.get(file_type.lower())

    if lang:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len, separators=["\n\n", "\n", " ", ""]
        )

    chunks = splitter.split_text(text)
    logger.info(f"Code text split into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")

    return chunks


def reconstruct_text_from_chunks(chunks: List[str], chunk_overlap: int) -> str:
    """
    청크들을 원본 텍스트로 재조합합니다.
    overlap 부분을 제거하여 중복 없이 합칩니다.

    Args:
        chunks: 청크 리스트
        chunk_overlap: 청크 간 겹침 크기

    Returns:
        재조합된 텍스트
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    out = chunks[0]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        cur = chunks[i]
        ov = find_overlap_length(prev, cur, chunk_overlap)
        out += cur[ov:] if ov > 0 else cur

    return out


def find_overlap_length(c1: str, c2: str, max_overlap: int) -> int:
    """
    두 청크 간 실제 overlap 길이를 찾습니다.

    Args:
        c1: 이전 청크
        c2: 현재 청크
        max_overlap: 최대 overlap 크기

    Returns:
        실제 overlap 길이
    """
    max_check = min(len(c1), len(c2), max_overlap)
    for ov in range(max_check, 0, -1):
        if c1[-ov:] == c2[:ov]:
            return ov
    return 0


def estimate_chunks_count(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
    """
    텍스트를 청킹했을 때 예상 청크 수를 계산합니다.

    Args:
        text: 텍스트
        chunk_size: 청크 최대 크기
        chunk_overlap: 청크 간 겹침 크기

    Returns:
        예상 청크 수
    """
    if not text:
        return 0
    if len(text) <= chunk_size:
        return 1

    eff = chunk_size - chunk_overlap
    return max(1, (len(text) - chunk_overlap) // eff + 1)
