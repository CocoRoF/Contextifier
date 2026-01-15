# service/document_processor/processor/doc_helpers/rtf_region_finder.py
"""
RTF 영역 탐색기

RTF 문서에서 제외해야 할 영역(헤더, 푸터, 각주 등)을 찾는 기능을 제공합니다.
"""
import logging
import re
from typing import List, Tuple

from libs.core.processor.doc_helpers.rtf_constants import (
    EXCLUDE_DESTINATION_KEYWORDS,
)

logger = logging.getLogger("document-processor")


def find_excluded_regions(content: str) -> List[Tuple[int, int]]:
    r"""
    문서 본문이 아닌 제외 영역을 찾습니다.

    RTF에서 \header, \footer, \footnote 등의 그룹은 본문이 아니므로
    테이블 및 텍스트 추출에서 제외해야 합니다.

    주의: RTF 테이블은 \trowd에서 시작하여 \row로 끝나는데,
    footer/header 그룹이 \trowd만 포함하고 셀 내용과 \row는 그룹 밖에
    있을 수 있습니다. 따라서 footer/header 그룹 안에서 시작하는 테이블의
    전체 범위(\row까지)를 제외해야 합니다.

    제외 대상:
    - \header, \headerf, \headerl, \headerr (헤더)
    - \footer, \footerf, \footerl, \footerr (푸터)
    - \footnote, \ftnsep, \ftnsepc, \aftncn, \aftnsep, \aftnsepc (각주)
    - \pntext, \pntxta, \pntxtb (번호 매기기 텍스트)
    - 위 그룹 안에서 시작하는 테이블의 전체 범위 (\trowd ~ \row)

    Args:
        content: RTF 콘텐츠

    Returns:
        제외 영역 리스트 [(start, end), ...]
    """
    excluded_regions = []

    pattern = '|'.join(EXCLUDE_DESTINATION_KEYWORDS)

    for match in re.finditer(pattern, content):
        keyword_start = match.start()
        keyword_end = match.end()

        # 이 키워드가 속한 그룹의 시작점('{') 찾기
        group_start = keyword_start
        search_back = min(keyword_start, 50)  # 최대 50자 뒤로 검색
        for i in range(keyword_start - 1, keyword_start - search_back - 1, -1):
            if i < 0:
                break
            if content[i] == '{':
                group_start = i
                break
            elif content[i] == '}':
                # 다른 그룹이 끝났으면 중단
                break

        # 그룹의 끝('}') 찾기 - 중첩 괄호 처리
        depth = 1
        i = keyword_end
        while i < len(content) and depth > 0:
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
            i += 1
        group_end = i

        # footer/header 그룹 안에 \trowd가 있으면, \row까지 확장
        group_content = content[group_start:group_end]
        if '\\trowd' in group_content:
            # 이 그룹 끝 이후에 매칭되는 \row 찾기
            row_match = re.search(r'\\row(?![a-z])', content[group_end:])
            if row_match:
                # \row의 끝까지 제외 영역 확장
                extended_end = group_end + row_match.end()
                group_end = extended_end
                logger.debug(f"Extended excluded region to include table row: {group_start}~{group_end}")

        excluded_regions.append((group_start, group_end))

    # 겹치는 영역 병합 및 정렬
    if not excluded_regions:
        return []

    excluded_regions.sort(key=lambda x: x[0])
    merged = [excluded_regions[0]]

    for start, end in excluded_regions[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # 겹치면 병합
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    logger.debug(f"Found {len(merged)} excluded regions (header/footer/footnote)")
    return merged


def is_in_excluded_region(pos: int, excluded_regions: List[Tuple[int, int]]) -> bool:
    """
    주어진 위치가 제외 영역 안에 있는지 확인합니다.

    Args:
        pos: 확인할 위치
        excluded_regions: 제외 영역 리스트

    Returns:
        제외 영역 안에 있으면 True
    """
    for start, end in excluded_regions:
        if start <= pos < end:
            return True
    return False
