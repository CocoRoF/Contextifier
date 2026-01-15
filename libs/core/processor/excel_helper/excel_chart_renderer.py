"""
Excel 차트 이미지 렌더링 모듈

matplotlib를 사용하여 차트 데이터를 이미지로 렌더링합니다.
테이블 변환 실패 시 폴백으로 사용됩니다.
"""

import io
import logging
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("document-processor")

# matplotlib (차트 이미지 생성용)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def render_chart_to_image(
    chart_info: Dict[str, Any],
    app_db=None,
    processed_images: Set[str] = None,
    upload_func=None
) -> Optional[str]:
    """
    차트 데이터를 matplotlib로 이미지로 렌더링하고 MinIO에 업로드합니다.

    테이블 변환 실패 시 폴백으로 사용됩니다.

    Args:
        chart_info: 차트 정보 딕셔너리
        app_db: 데이터베이스 연결
        processed_images: 이미 처리된 이미지 해시 집합
        upload_func: 이미지 업로드 함수

    Returns:
        [chart] 태그로 감싸진 이미지 참조 문자열, 실패 시 None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available for chart rendering")
        return None

    if not chart_info:
        return None

    try:
        categories = chart_info.get('categories', [])
        series_list = chart_info.get('series', [])
        chart_type = chart_info.get('chart_type', '')
        title = chart_info.get('title', '차트')

        if not series_list:
            return None

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6))

        # 차트 유형에 따른 렌더링
        if '파이' in chart_type or 'pie' in chart_type.lower():
            _render_pie_chart(ax, series_list, categories)
        elif '선' in chart_type or 'line' in chart_type.lower():
            _render_line_chart(ax, series_list, categories)
        elif '영역' in chart_type or 'area' in chart_type.lower():
            _render_area_chart(ax, series_list, categories)
        else:
            # 기본: 막대 차트
            _render_bar_chart(ax, series_list, categories)

        ax.set_title(title)
        plt.tight_layout()

        # 이미지를 바이트로 저장
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_data = img_buffer.getvalue()
        plt.close(fig)

        # MinIO에 업로드
        if processed_images is None:
            processed_images = set()

        if upload_func:
            minio_path = upload_func(img_data, app_db=app_db, processed_images=processed_images)

            if minio_path:
                result_parts = ["[chart]"]
                if title:
                    result_parts.append(f"제목: {title}")
                if chart_type:
                    result_parts.append(f"유형: {chart_type}")
                result_parts.append(f"[image:{minio_path}]")
                result_parts.append("[/chart]")
                return "\n".join(result_parts)

        return None

    except Exception as e:
        logger.warning(f"Error rendering chart to image: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def _render_pie_chart(ax, series_list, categories):
    """
    파이 차트를 렌더링합니다.
    """
    if series_list and series_list[0].get('values'):
        values = series_list[0]['values']
        labels = categories if categories else [f'항목 {i+1}' for i in range(len(values))]
        ax.pie(values, labels=labels, autopct='%1.1f%%')


def _render_line_chart(ax, series_list, categories):
    """
    선 차트를 렌더링합니다.
    """
    x = categories if categories else list(range(len(series_list[0].get('values', []))))
    for series in series_list:
        values = series.get('values', [])
        name = series.get('name', '시리즈')
        if values:
            ax.plot(x[:len(values)], values, marker='o', label=name)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _render_area_chart(ax, series_list, categories):
    """
    영역 차트를 렌더링합니다.
    """
    for series in series_list:
        values = series.get('values', [])
        name = series.get('name', '시리즈')
        if values:
            ax.fill_between(range(len(values)), values, alpha=0.5, label=name)
            ax.plot(values, marker='o', label=f'{name} (선)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _render_bar_chart(ax, series_list, categories):
    """
    막대 차트를 렌더링합니다.
    """
    x = categories if categories else [f'항목 {i+1}' for i in range(len(series_list[0].get('values', [])))]
    width = 0.8 / len(series_list) if len(series_list) > 1 else 0.6

    for idx, series in enumerate(series_list):
        values = series.get('values', [])
        name = series.get('name', f'시리즈 {idx+1}')
        if values:
            offset = (idx - len(series_list) / 2 + 0.5) * width
            positions = [i + offset for i in range(len(values))]
            ax.bar(positions, values, width=width, label=name)

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
