"""
PPT 차트 처리 모듈

포함 함수:
- extract_chart_data(): 차트에서 데이터 추출

차트 유형에 따라 다음 정보를 추출:
- 차트 제목
- 카테고리 (X축 레이블)
- 시리즈 이름 및 값
- 데이터 테이블 형식으로 출력
"""
import logging

logger = logging.getLogger("document-processor")


def extract_chart_data(chart) -> str:
    """
    차트에서 데이터를 추출합니다.

    차트 유형에 따라 다음 정보를 추출합니다:
    - 차트 제목
    - 카테고리 (X축 레이블)
    - 시리즈 이름 및 값
    - 데이터 테이블 형식으로 출력
    
    Args:
        chart: python-pptx Chart 객체
        
    Returns:
        차트 데이터 텍스트 (Markdown 테이블 형식)
    """
    try:
        result_parts = ["[차트]"]

        # 차트 제목
        if chart.has_title and chart.chart_title:
            try:
                title_text = chart.chart_title.text_frame.text if chart.chart_title.has_text_frame else ""
                if title_text:
                    result_parts.append(f"제목: {title_text}")
            except Exception:
                pass

        # 차트 유형
        try:
            chart_type = str(chart.chart_type).split('.')[-1] if hasattr(chart, 'chart_type') else "Unknown"
            result_parts.append(f"유형: {chart_type}")
        except Exception:
            pass

        # 카테고리 추출
        categories = []
        try:
            if hasattr(chart, 'plots') and chart.plots:
                for plot in chart.plots:
                    if hasattr(plot, 'categories') and plot.categories:
                        categories = list(plot.categories)
                        break
        except Exception:
            pass

        # 시리즈 데이터 추출
        series_data = []
        try:
            for series in chart.series:
                series_info = {
                    'name': series.name if hasattr(series, 'name') else f"시리즈 {len(series_data) + 1}",
                    'values': []
                }

                # 시리즈 값 추출
                try:
                    if hasattr(series, 'values') and series.values:
                        series_info['values'] = list(series.values)
                except Exception:
                    pass

                series_data.append(series_info)
        except Exception:
            pass

        # 데이터 테이블 형식으로 출력
        if categories or series_data:
            result_parts.append("")

            # 헤더 행: 카테고리 | 시리즈1 | 시리즈2 | ...
            header = ["카테고리"] + [s['name'] for s in series_data]
            result_parts.append("| " + " | ".join(str(h) for h in header) + " |")
            result_parts.append("| " + " | ".join(["---"] * len(header)) + " |")

            # 데이터 행
            max_len = max(
                len(categories),
                max((len(s['values']) for s in series_data), default=0)
            ) if categories or series_data else 0

            for i in range(max_len):
                row = []
                # 카테고리
                if i < len(categories):
                    row.append(str(categories[i]))
                else:
                    row.append("")

                # 시리즈 값
                for series in series_data:
                    if i < len(series['values']):
                        val = series['values'][i]
                        # 숫자 포맷팅
                        if isinstance(val, float):
                            row.append(f"{val:,.2f}")
                        elif val is not None:
                            row.append(str(val))
                        else:
                            row.append("")
                    else:
                        row.append("")

                result_parts.append("| " + " | ".join(row) + " |")

        result_parts.append("[/차트]")
        return "\n".join(result_parts)

    except Exception as e:
        logger.debug(f"Error extracting chart data: {e}")
        return "[차트 객체]"
