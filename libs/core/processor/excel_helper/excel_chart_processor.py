"""
Excel 차트 처리 메인 모듈

차트 처리의 진입점 역할을 합니다.
1순위: 데이터를 표(테이블)로 변환
2순위: 실패 시 matplotlib로 이미지 생성
"""

import logging
from typing import Any, Dict, Set

from .excel_chart_formatter import format_chart_data_as_table, format_chart_fallback
from .excel_chart_renderer import render_chart_to_image

logger = logging.getLogger("document-processor")


def process_chart(
    chart_info: Dict[str, Any],
    app_db=None,
    processed_images: Set[str] = None,
    upload_func=None
) -> str:
    """
    차트를 처리합니다.

    1순위: 데이터를 표(테이블)로 변환 - LLM이 직접 해석 가능
    2순위: 실패 시 matplotlib로 이미지 생성 후 MinIO 업로드

    Args:
        chart_info: 차트 정보 딕셔너리
        app_db: 데이터베이스 연결
        processed_images: 이미 처리된 이미지 해시 집합
        upload_func: 이미지 업로드 함수

    Returns:
        [chart]...[/chart] 형태의 문자열
    """
    # 1순위: 테이블로 변환 시도
    table_result = format_chart_data_as_table(chart_info)
    if table_result:
        logger.debug("Chart converted to table successfully")
        return table_result

    # 2순위: 이미지로 렌더링
    image_result = render_chart_to_image(chart_info, app_db, processed_images, upload_func)
    if image_result:
        logger.debug("Chart rendered to image successfully")
        return image_result

    # 최종 폴백: 기본 정보만 출력
    return format_chart_fallback(chart_info)
