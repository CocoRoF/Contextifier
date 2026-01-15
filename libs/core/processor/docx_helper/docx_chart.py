# service/document_processor/processor/docx_helper/docx_chart.py
"""
DOCX 차트 처리 유틸리티

OOXML DrawingML Chart 사양(ISO/IEC 29500 / ECMA 376)에 따라 차트 데이터를 파싱합니다.
- parse_ooxml_chart_xml: OOXML 차트 XML 파싱
- extract_chart_series: 차트 시리즈 데이터 추출
- format_chart_data: 차트 데이터를 텍스트로 포맷팅
- parse_chart_data_basic: 기본 차트 정보 추출 (폴백용)
"""
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

from lxml import etree

from .docx_constants import CHART_TYPE_MAP

logger = logging.getLogger("document-processor")


def parse_ooxml_chart_xml(chart_xml: bytes) -> Optional[Dict[str, Any]]:
    """
    OOXML 차트 XML을 파싱하여 차트 데이터를 추출합니다.

    OOXML 차트는 ISO/IEC 29500 / ECMA 376 DrawingML Chart 사양을 따릅니다.

    Args:
        chart_xml: 차트 XML 바이트

    Returns:
        차트 데이터 딕셔너리 (title, type, series, categories)
    """
    try:
        # OOXML 네임스페이스
        ns = {
            'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        }

        # XML 파싱
        try:
            root = ET.fromstring(chart_xml)
        except ET.ParseError:
            # BOM이나 잘못된 문자 제거 시도
            try:
                chart_str = chart_xml.decode('utf-8-sig', errors='ignore')
                root = ET.fromstring(chart_str)
            except:
                return None

        chart_info = {
            'type': 'ooxml',
            'chart_type': None,
            'title': None,
            'series': [],
            'categories': []
        }

        # chart 요소 찾기
        chart_elem = root.find('.//c:chart', ns)
        if chart_elem is None:
            chart_elem = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}chart')

        if chart_elem is None:
            # root가 chart일 수 있음
            if root.tag.endswith('}chart') or root.tag == 'chart':
                chart_elem = root
            else:
                return None

        # 차트 제목 추출
        title_elem = chart_elem.find('.//c:title//c:tx//c:rich//a:t', ns)
        if title_elem is not None and title_elem.text:
            chart_info['title'] = title_elem.text.strip()
        else:
            # 대체 경로 시도
            title_elem = chart_elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}tx//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
            if title_elem is not None and title_elem.text:
                chart_info['title'] = title_elem.text.strip()

        # 차트 유형 및 시리즈 데이터 추출
        plot_area = chart_elem.find('.//c:plotArea', ns)
        if plot_area is None:
            plot_area = chart_elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}plotArea')

        if plot_area is not None:
            for chart_tag, chart_name in CHART_TYPE_MAP.items():
                elem = plot_area.find(f'.//c:{chart_tag}', ns)
                if elem is None:
                    elem = plot_area.find(f'.//{{{ns["c"]}}}{chart_tag}')
                if elem is not None:
                    chart_info['chart_type'] = chart_name
                    extract_chart_series(elem, chart_info, ns)
                    break

        return chart_info if chart_info['series'] else None

    except Exception as e:
        logger.debug(f"Error parsing OOXML chart: {e}")
        return None


def extract_chart_series(chart_type_elem, chart_info: Dict[str, Any], ns: Dict[str, str]):
    """
    차트 요소에서 시리즈 데이터를 추출합니다.

    Args:
        chart_type_elem: 차트 타입 XML 요소
        chart_info: 채울 차트 정보 딕셔너리
        ns: XML 네임스페이스
    """
    ns_c = ns.get('c', 'http://schemas.openxmlformats.org/drawingml/2006/chart')

    # 모든 시리즈 찾기 (c:ser)
    series_elements = chart_type_elem.findall('.//c:ser', ns)
    if not series_elements:
        series_elements = chart_type_elem.findall(f'.//{{{ns_c}}}ser')

    categories_extracted = False

    for ser_elem in series_elements:
        series_data = {
            'name': None,
            'values': [],
        }

        # 시리즈 이름 추출 (c:tx)
        tx_elem = ser_elem.find('.//c:tx//c:v', ns)
        if tx_elem is None:
            tx_elem = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}v')
        if tx_elem is not None and tx_elem.text:
            series_data['name'] = tx_elem.text.strip()
        else:
            # strRef 경로 시도
            str_ref = ser_elem.find('.//c:tx//c:strRef//c:strCache//c:pt//c:v', ns)
            if str_ref is None:
                str_ref = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}strRef//{{{ns_c}}}strCache//{{{ns_c}}}pt//{{{ns_c}}}v')
            if str_ref is not None and str_ref.text:
                series_data['name'] = str_ref.text.strip()

        # 카테고리 레이블 추출 (c:cat) - 첫 시리즈에서만
        if not categories_extracted:
            cat_elem = ser_elem.find('.//c:cat', ns)
            if cat_elem is None:
                cat_elem = ser_elem.find(f'.//{{{ns_c}}}cat')

            if cat_elem is not None:
                # strRef 시도 (문자열 레이블)
                str_cache = cat_elem.find('.//c:strCache', ns)
                if str_cache is None:
                    str_cache = cat_elem.find(f'.//{{{ns_c}}}strCache')

                if str_cache is not None:
                    pts = str_cache.findall('.//c:pt', ns)
                    if not pts:
                        pts = str_cache.findall(f'.//{{{ns_c}}}pt')

                    for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                        v_elem = pt.find('c:v', ns)
                        if v_elem is None:
                            v_elem = pt.find(f'{{{ns_c}}}v')
                        if v_elem is not None and v_elem.text:
                            chart_info['categories'].append(v_elem.text.strip())

                # numCache 시도 (숫자 레이블)
                if not chart_info['categories']:
                    num_cache = cat_elem.find('.//c:numCache', ns)
                    if num_cache is None:
                        num_cache = cat_elem.find(f'.//{{{ns_c}}}numCache')

                    if num_cache is not None:
                        pts = num_cache.findall('.//c:pt', ns)
                        if not pts:
                            pts = num_cache.findall(f'.//{{{ns_c}}}pt')

                        for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                            v_elem = pt.find('c:v', ns)
                            if v_elem is None:
                                v_elem = pt.find(f'{{{ns_c}}}v')
                            if v_elem is not None and v_elem.text:
                                chart_info['categories'].append(v_elem.text.strip())

                categories_extracted = True

        # 값 추출 (c:val)
        val_elem = ser_elem.find('.//c:val', ns)
        if val_elem is None:
            val_elem = ser_elem.find(f'.//{{{ns_c}}}val')

        if val_elem is not None:
            num_cache = val_elem.find('.//c:numCache', ns)
            if num_cache is None:
                num_cache = val_elem.find(f'.//{{{ns_c}}}numCache')

            if num_cache is not None:
                pts = num_cache.findall('.//c:pt', ns)
                if not pts:
                    pts = num_cache.findall(f'.//{{{ns_c}}}pt')

                for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                    v_elem = pt.find('c:v', ns)
                    if v_elem is None:
                        v_elem = pt.find(f'{{{ns_c}}}v')
                    if v_elem is not None and v_elem.text:
                        try:
                            series_data['values'].append(float(v_elem.text))
                        except ValueError:
                            series_data['values'].append(v_elem.text)

        # yVal 확인 (scatter/bubble 차트용)
        if not series_data['values']:
            yval_elem = ser_elem.find('.//c:yVal', ns)
            if yval_elem is None:
                yval_elem = ser_elem.find(f'.//{{{ns_c}}}yVal')

            if yval_elem is not None:
                num_cache = yval_elem.find('.//c:numCache', ns)
                if num_cache is None:
                    num_cache = yval_elem.find(f'.//{{{ns_c}}}numCache')

                if num_cache is not None:
                    pts = num_cache.findall('.//c:pt', ns)
                    if not pts:
                        pts = num_cache.findall(f'.//{{{ns_c}}}pt')

                    for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                        v_elem = pt.find('c:v', ns)
                        if v_elem is None:
                            v_elem = pt.find(f'{{{ns_c}}}v')
                        if v_elem is not None and v_elem.text:
                            try:
                                series_data['values'].append(float(v_elem.text))
                            except ValueError:
                                series_data['values'].append(v_elem.text)

        if series_data['values']:
            chart_info['series'].append(series_data)


def format_chart_data(chart_info: Dict[str, Any]) -> str:
    """
    차트 데이터를 읽기 쉬운 텍스트 형식으로 포맷합니다.

    Args:
        chart_info: 차트 정보 딕셔너리

    Returns:
        포맷된 차트 데이터 문자열
    """
    if not chart_info:
        return "[차트]"

    result_parts = ["[차트]"]

    # 차트 제목
    if chart_info.get('title'):
        result_parts.append(f"제목: {chart_info['title']}")

    # 차트 유형
    if chart_info.get('chart_type'):
        result_parts.append(f"유형: {chart_info['chart_type']}")

    # 데이터 테이블
    categories = chart_info.get('categories', [])
    series_list = chart_info.get('series', [])

    if categories or series_list:
        result_parts.append("")

        # 헤더 행: 카테고리 | 시리즈1 | 시리즈2 | ...
        header = ["카테고리"] + [s.get('name', f'시리즈 {i+1}') for i, s in enumerate(series_list)]
        result_parts.append("| " + " | ".join(str(h) for h in header) + " |")
        result_parts.append("| " + " | ".join(["---"] * len(header)) + " |")

        # 데이터 행
        max_len = max(
            len(categories),
            max((len(s.get('values', [])) for s in series_list), default=0)
        ) if categories or series_list else 0

        for i in range(max_len):
            row = []
            # 카테고리
            if i < len(categories):
                row.append(str(categories[i]))
            else:
                row.append("")

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

    result_parts.append("[/차트]")
    return "\n".join(result_parts)


def parse_chart_data_basic(chart_xml: bytes) -> str:
    """
    차트 XML에서 기본 정보를 추출합니다 (폴백용).

    Args:
        chart_xml: 차트 XML 바이트

    Returns:
        기본 차트 정보 문자열
    """
    try:
        root = etree.fromstring(chart_xml)

        result_parts = ["[차트]"]

        # 차트 제목 추출
        title_elem = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}title')
        if title_elem is not None:
            title_texts = title_elem.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
            title = ''.join([t.text or '' for t in title_texts])
            if title.strip():
                result_parts.append(f"제목: {title.strip()}")

        # 시리즈 데이터 추출 시도
        series_elems = root.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}ser')
        for i, ser in enumerate(series_elems):
            # 시리즈 이름
            tx = ser.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}tx')
            if tx is not None:
                v = tx.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}v')
                if v is not None and v.text:
                    result_parts.append(f"시리즈 {i+1}: {v.text}")

        result_parts.append("[/차트]")
        return "\n".join(result_parts)

    except Exception as e:
        logger.debug(f"Error parsing chart data (basic): {e}")
        return "[차트]"


__all__ = [
    'parse_ooxml_chart_xml',
    'extract_chart_series',
    'format_chart_data',
    'parse_chart_data_basic',
]
