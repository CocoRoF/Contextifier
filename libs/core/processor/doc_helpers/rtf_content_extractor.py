# service/document_processor/processor/doc_helpers/rtf_content_extractor.py
"""
RTF 콘텐츠 추출기

RTF 문서에서 인라인 콘텐츠(텍스트 + 테이블)를 추출하는 기능을 제공합니다.
"""
import logging
import re
from typing import List, Tuple

from libs.core.processor.doc_helpers.rtf_models import (
    RTFTable,
    RTFContentPart,
)
from libs.core.processor.doc_helpers.rtf_decoder import (
    decode_hex_escapes,
)
from libs.core.processor.doc_helpers.rtf_text_cleaner import (
    clean_rtf_text,
    remove_destination_groups,
    remove_shape_groups,
    remove_shape_property_groups,
)
from libs.core.processor.doc_helpers.rtf_region_finder import (
    find_excluded_regions,
)

logger = logging.getLogger("document-processor")


def extract_inline_content(
    content: str,
    table_regions: List[Tuple[int, int, RTFTable]],
    encoding: str = "cp949"
) -> List[RTFContentPart]:
    """
    RTF에서 인라인 콘텐츠를 추출합니다.

    테이블은 원래 위치에 배치됩니다.

    Args:
        content: RTF 문자열 콘텐츠
        table_regions: 테이블 영역 리스트 [(start, end, table), ...]
        encoding: 사용할 인코딩

    Returns:
        콘텐츠 파트 리스트
    """
    content_parts = []

    # 헤더 영역 제거 (fonttbl, colortbl, stylesheet, info 등)
    # 첫 번째 \pard 이전은 헤더로 간주
    header_end = 0
    pard_match = re.search(r'\\pard\b', content)
    if pard_match:
        header_end = pard_match.start()

    # 제외 영역 찾기 (header, footer, footnote 등)
    excluded_regions = find_excluded_regions(content)

    def clean_segment(segment: str, start_pos: int) -> str:
        """세그먼트를 정리하되 제외 영역은 건너뜁니다."""
        if not excluded_regions:
            # 제외 영역이 없으면 전체 정리
            segment = remove_destination_groups(segment)
            decoded = decode_hex_escapes(segment, encoding)
            return clean_rtf_text(decoded, encoding)

        # 세그먼트 내에서 제외 영역을 마스킹
        result_parts = []
        seg_pos = 0

        for excl_start, excl_end in excluded_regions:
            # 세그먼트 기준 상대 위치로 변환
            rel_start = excl_start - start_pos
            rel_end = excl_end - start_pos

            # 세그먼트 범위 내에 있는지 확인
            if rel_end <= 0 or rel_start >= len(segment):
                continue  # 범위 밖

            # 범위 조정
            rel_start = max(0, rel_start)
            rel_end = min(len(segment), rel_end)

            # 제외 영역 전 텍스트 처리
            if rel_start > seg_pos:
                part = segment[seg_pos:rel_start]
                part = remove_destination_groups(part)
                decoded = decode_hex_escapes(part, encoding)
                clean = clean_rtf_text(decoded, encoding)
                if clean.strip():
                    result_parts.append(clean)

            seg_pos = rel_end

        # 마지막 제외 영역 이후 텍스트
        if seg_pos < len(segment):
            part = segment[seg_pos:]
            part = remove_destination_groups(part)
            decoded = decode_hex_escapes(part, encoding)
            clean = clean_rtf_text(decoded, encoding)
            if clean.strip():
                result_parts.append(clean)

        return ' '.join(result_parts)

    # 테이블 영역이 없으면 전체 텍스트만 추출
    if not table_regions:
        clean = clean_segment(content[header_end:], header_end)
        if clean.strip():
            content_parts.append(RTFContentPart(
                content_type="text",
                position=0,
                text=clean
            ))
        return content_parts

    # 헤더 오프셋 적용
    adjusted_regions = []
    for start_pos, end_pos, table in table_regions:
        # 헤더 이후 영역만 처리
        if end_pos > header_end:
            adj_start = max(start_pos, header_end)
            adjusted_regions.append((adj_start, end_pos, table))

    # 콘텐츠 파트 생성
    last_end = header_end

    for start_pos, end_pos, table in adjusted_regions:
        # 테이블 전 텍스트
        if start_pos > last_end:
            segment = content[last_end:start_pos]
            clean = clean_segment(segment, last_end)
            if clean.strip():
                content_parts.append(RTFContentPart(
                    content_type="text",
                    position=last_end,
                    text=clean
                ))

        # 테이블
        content_parts.append(RTFContentPart(
            content_type="table",
            position=start_pos,
            table=table
        ))

        last_end = end_pos

    # 마지막 부분 (테이블 이후 텍스트)
    if last_end < len(content):
        segment = content[last_end:]
        clean = clean_segment(segment, last_end)
        if clean.strip():
            content_parts.append(RTFContentPart(
                content_type="text",
                position=last_end,
                text=clean
            ))

    return content_parts


def extract_text_legacy(content: str, encoding: str = "cp949") -> str:
    """
    RTF에서 일반 텍스트를 추출합니다.
    테이블 영역은 제외하고 추출합니다.
    (레거시 호환성을 위해 유지)

    Args:
        content: RTF 문자열 콘텐츠
        encoding: 사용할 인코딩

    Returns:
        추출된 텍스트
    """
    # 헤더 영역 제거 (fonttbl, colortbl, stylesheet 등)
    pard_match = re.search(r'\\pard\b', content)
    if pard_match:
        content = content[pard_match.start():]

    # destination 그룹 제거 (latentstyles, themedata 등)
    content = remove_destination_groups(content)

    # Shape 그룹 처리 (shptxt 내용은 보존)
    content = remove_shape_groups(content)

    # Shape 속성 그룹 제거
    content = remove_shape_property_groups(content)

    # 테이블 영역 찾기 및 마킹
    table_regions = []
    for match in re.finditer(r'\\trowd.*?\\row', content, re.DOTALL):
        table_regions.append((match.start(), match.end()))

    # 테이블 영역을 병합 (인접한 테이블들)
    merged_regions = []
    for start, end in table_regions:
        if merged_regions and start - merged_regions[-1][1] < 100:
            merged_regions[-1] = (merged_regions[-1][0], end)
        else:
            merged_regions.append((start, end))

    # 테이블 영역을 제외한 텍스트 추출
    text_parts = []
    last_end = 0

    for start, end in merged_regions:
        if start > last_end:
            segment = content[last_end:start]
            decoded = decode_hex_escapes(segment, encoding)
            clean = clean_rtf_text(decoded, encoding)
            if clean:
                text_parts.append(clean)
        last_end = end

    # 마지막 부분
    if last_end < len(content):
        segment = content[last_end:]
        decoded = decode_hex_escapes(segment, encoding)
        clean = clean_rtf_text(decoded, encoding)
        if clean:
            text_parts.append(clean)

    # 연속된 빈 줄 정리
    text = '\n'.join(text_parts)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()
