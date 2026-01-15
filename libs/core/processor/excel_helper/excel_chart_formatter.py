"""
Excel 차트 데이터 포맷팅 모듈

차트 데이터를 Markdown 테이블 형식으로 변환합니다.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("document-processor")


def format_chart_data_as_table(chart_info: Dict[str, Any]) -> Optional[str]:
    """
    차트 데이터를 Markdown 테이블 형식으로 포맷합니다.

    데이터가 충분하면 테이블 문자열 반환, 없으면 None 반환.
    None 반환 시 이미지 폴백이 트리거됩니다.
    
    Args:
        chart_info: 차트 정보 딕셔너리
        
    Returns:
        Markdown 테이블 형식의 문자열 또는 None
    """
    if not chart_info:
        return None

    categories = chart_info.get('categories', [])
    series_list = chart_info.get('series', [])

    # 데이터가 없으면 None 반환 (이미지 폴백 필요)
    if not series_list or all(len(s.get('values', [])) == 0 for s in series_list):
        return None

    result_parts = ["[chart]"]

    if chart_info.get('title'):
        result_parts.append(f"제목: {chart_info['title']}")

    if chart_info.get('chart_type'):
        result_parts.append(f"유형: {chart_info['chart_type']}")

    result_parts.append("")

    # 테이블 헤더 생성
    header = ["카테고리"] + [s.get('name', f'시리즈 {i+1}') for i, s in enumerate(series_list)]
    result_parts.append("| " + " | ".join(str(h) for h in header) + " |")
    result_parts.append("| " + " | ".join(["---"] * len(header)) + " |")

    # 데이터 행 생성
    max_len = max(
        len(categories),
        max((len(s.get('values', [])) for s in series_list), default=0)
    )

    for i in range(max_len):
        row = []
        
        # 카테고리
        if i < len(categories):
            row.append(str(categories[i]))
        else:
            row.append(f"항목 {i+1}")

        # 시리즈 값
        for series in series_list:
            values = series.get('values', [])
            if i < len(values):
                val = values[i]
                if isinstance(val, float):
                    row.append(f"{val:,.2f}")
                elif val is not None:
                    row.append(str(val))
                else:
                    row.append("")
            else:
                row.append("")

        result_parts.append("| " + " | ".join(row) + " |")

    result_parts.append("[/chart]")
    return "\n".join(result_parts)


def format_chart_fallback(chart_info: Dict[str, Any]) -> str:
    """
    차트 정보만 출력하는 폴백 포맷터.
    
    테이블/이미지 변환 모두 실패 시 사용됩니다.
    
    Args:
        chart_info: 차트 정보 딕셔너리
        
    Returns:
        [chart]...[/chart] 형태의 기본 문자열
    """
    result_parts = ["[chart]"]
    
    if chart_info and chart_info.get('title'):
        result_parts.append(f"제목: {chart_info['title']}")
    if chart_info and chart_info.get('chart_type'):
        result_parts.append(f"유형: {chart_info['chart_type']}")
    
    result_parts.append("[/chart]")
    return "\n".join(result_parts)
