"""
Excel 차트 OOXML 파싱 모듈

XLSX 파일의 차트 XML을 파싱하여 데이터를 추출합니다.
"""

import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .excel_chart_constants import CHART_TYPE_MAP, CHART_NAMESPACES

logger = logging.getLogger("document-processor")


def extract_charts_from_xlsx(file_path: str) -> List[Dict[str, Any]]:
    """
    XLSX 파일에서 차트 데이터를 추출합니다.

    XLSX 차트는 xl/charts/ 폴더에 chart*.xml 파일로 저장됩니다.

    Args:
        file_path: XLSX 파일 경로

    Returns:
        차트 정보 딕셔너리 리스트
    """
    charts = []

    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            for name in zf.namelist():
                if name.startswith('xl/charts/chart') and name.endswith('.xml'):
                    try:
                        chart_xml = zf.read(name)
                        chart_info = parse_ooxml_chart_xml(chart_xml)
                        if chart_info:
                            charts.append(chart_info)
                    except Exception as e:
                        logger.debug(f"Error parsing chart {name}: {e}")

        logger.info(f"Extracted {len(charts)} charts from XLSX")

    except Exception as e:
        logger.warning(f"Error extracting charts from XLSX: {e}")

    return charts


def parse_ooxml_chart_xml(chart_xml: bytes) -> Optional[Dict[str, Any]]:
    """
    OOXML 차트 XML을 파싱하여 차트 데이터를 추출합니다.

    Args:
        chart_xml: 차트 XML 바이트

    Returns:
        차트 데이터 딕셔너리
    """
    try:
        ns = CHART_NAMESPACES

        try:
            root = ET.fromstring(chart_xml)
        except ET.ParseError:
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
            if root.tag.endswith('}chart') or root.tag == 'chart':
                chart_elem = root
            else:
                return None

        # 차트 제목 추출
        title_elem = chart_elem.find('.//c:title//c:tx//c:rich//a:t', ns)
        if title_elem is not None and title_elem.text:
            chart_info['title'] = title_elem.text.strip()
        else:
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
                    _extract_chart_series(elem, chart_info, ns)
                    break

        return chart_info if chart_info['series'] else None

    except Exception as e:
        logger.debug(f"Error parsing OOXML chart: {e}")
        return None


def _extract_chart_series(chart_type_elem, chart_info: Dict[str, Any], ns: Dict[str, str]):
    """
    차트 요소에서 시리즈 데이터를 추출합니다.
    
    Args:
        chart_type_elem: 차트 타입 XML 요소
        chart_info: 차트 정보 딕셔너리 (수정됨)
        ns: XML 네임스페이스 딕셔너리
    """
    ns_c = ns.get('c', 'http://schemas.openxmlformats.org/drawingml/2006/chart')

    series_elements = chart_type_elem.findall('.//c:ser', ns)
    if not series_elements:
        series_elements = chart_type_elem.findall(f'.//{{{ns_c}}}ser')

    categories_extracted = False

    for ser_elem in series_elements:
        series_data = {
            'name': None,
            'values': [],
        }

        # 시리즈 이름 추출
        tx_elem = ser_elem.find('.//c:tx//c:v', ns)
        if tx_elem is None:
            tx_elem = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}v')
        if tx_elem is not None and tx_elem.text:
            series_data['name'] = tx_elem.text.strip()
        else:
            str_ref = ser_elem.find('.//c:tx//c:strRef//c:strCache//c:pt//c:v', ns)
            if str_ref is None:
                str_ref = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}strRef//{{{ns_c}}}strCache//{{{ns_c}}}pt//{{{ns_c}}}v')
            if str_ref is not None and str_ref.text:
                series_data['name'] = str_ref.text.strip()

        # 카테고리 레이블 추출
        if not categories_extracted:
            _extract_categories(ser_elem, chart_info, ns, ns_c)
            categories_extracted = True

        # 값 추출
        _extract_values(ser_elem, series_data, ns, ns_c)

        if series_data['values']:
            chart_info['series'].append(series_data)


def _extract_categories(ser_elem, chart_info: Dict[str, Any], ns: Dict[str, str], ns_c: str):
    """
    시리즈 요소에서 카테고리 레이블을 추출합니다.
    """
    cat_elem = ser_elem.find('.//c:cat', ns)
    if cat_elem is None:
        cat_elem = ser_elem.find(f'.//{{{ns_c}}}cat')

    if cat_elem is None:
        return

    # strCache에서 추출
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

    # numCache에서 추출 (폴백)
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


def _extract_values(ser_elem, series_data: Dict[str, Any], ns: Dict[str, str], ns_c: str):
    """
    시리즈 요소에서 값을 추출합니다.
    """
    # val 요소에서 추출
    val_elem = ser_elem.find('.//c:val', ns)
    if val_elem is None:
        val_elem = ser_elem.find(f'.//{{{ns_c}}}val')

    if val_elem is not None:
        _extract_num_cache_values(val_elem, series_data, ns, ns_c)

    # yVal 확인 (scatter/bubble 차트용)
    if not series_data['values']:
        yval_elem = ser_elem.find('.//c:yVal', ns)
        if yval_elem is None:
            yval_elem = ser_elem.find(f'.//{{{ns_c}}}yVal')

        if yval_elem is not None:
            _extract_num_cache_values(yval_elem, series_data, ns, ns_c)


def _extract_num_cache_values(val_elem, series_data: Dict[str, Any], ns: Dict[str, str], ns_c: str):
    """
    numCache에서 숫자 값을 추출합니다.
    """
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


def extract_chart_info_basic(chart, ws) -> str:
    """
    차트 정보를 추출합니다 (openpyxl 객체에서 기본 추출).
    OOXML 파싱 실패 시 폴백으로 사용됩니다.
    
    Args:
        chart: openpyxl Chart 객체
        ws: openpyxl Worksheet 객체
        
    Returns:
        [chart]...[/chart] 형태의 문자열
    """
    try:
        result_parts = ["[chart]"]

        # 차트 타입
        chart_type = type(chart).__name__
        result_parts.append(f"유형: {chart_type}")

        # 차트 제목
        if chart.title:
            title_text = _extract_chart_title(chart.title)
            if title_text:
                result_parts.append(f"제목: {title_text}")

        # 시리즈 데이터
        if hasattr(chart, 'series'):
            for i, series in enumerate(chart.series):
                series_info = _extract_series_info(series, ws, i)
                if series_info:
                    result_parts.append(series_info)

        result_parts.append("[/chart]")
        return "\n".join(result_parts)

    except Exception as e:
        logger.debug(f"Error extracting chart info: {e}")
        return "[chart][/chart]"


def _extract_chart_title(title_obj) -> str:
    """
    차트 제목을 추출합니다.
    """
    try:
        if hasattr(title_obj, 'tx') and title_obj.tx:
            if hasattr(title_obj.tx, 'rich') and title_obj.tx.rich:
                # RichText에서 텍스트 추출
                texts = []
                if hasattr(title_obj.tx.rich, 'p'):
                    for p in title_obj.tx.rich.p:
                        if hasattr(p, 'r'):
                            for r in p.r:
                                if hasattr(r, 't') and r.t:
                                    texts.append(r.t)
                return "".join(texts)
        return ""
    except Exception:
        return ""


def _extract_series_info(series, ws, index: int) -> str:
    """
    차트 시리즈 정보를 추출합니다.
    """
    try:
        parts = [f"시리즈 {index + 1}:"]

        # 시리즈 이름
        if hasattr(series, 'title') and series.title:
            if hasattr(series.title, 'strRef') and series.title.strRef:
                ref = series.title.strRef.f
                parts.append(f"  이름 참조: {ref}")

        # 데이터 참조
        if hasattr(series, 'val') and series.val:
            if hasattr(series.val, 'numRef') and series.val.numRef:
                ref = series.val.numRef.f
                parts.append(f"  데이터 참조: {ref}")

                # 실제 데이터 값 추출 시도
                try:
                    values = _get_range_values(ws, ref)
                    if values:
                        parts.append(f"  데이터: {values[:10]}{'...' if len(values) > 10 else ''}")
                except Exception:
                    pass

        return "\n".join(parts) if len(parts) > 1 else ""

    except Exception:
        return ""


def _get_range_values(ws, ref: str) -> List[Any]:
    """
    셀 범위 참조에서 값을 추출합니다.
    """
    try:
        # 참조 형식: 'Sheet1'!$A$1:$A$10 또는 Sheet1!A1:A10
        match = re.search(r"['\"]?([^'\"!]+)['\"]?!\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)", ref)
        if not match:
            return []

        _, start_col, start_row, end_col, end_row = match.groups()
        start_row, end_row = int(start_row), int(end_row)

        values = []
        for row in range(start_row, end_row + 1):
            cell = ws[f"{start_col}{row}"]
            if cell.value is not None:
                values.append(cell.value)

        return values

    except Exception:
        return []
