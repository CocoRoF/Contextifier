"""
차트 처리 공통 헬퍼

주요 기능:
- HWP/HWPX 차트 데이터 추출 (OOXML 및 레거시 형식)
- 차트를 테이블로 변환 (1순위)
- 차트를 이미지로 렌더링 (2순위, 테이블 변환 실패 시)
"""
import io
import os
import zlib
import struct
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List, Set

import olefile

# matplotlib (차트 이미지 생성용)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# MinIO 업로드
try:
    from libs.core.functions.utils import upload_image_to_minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

logger = logging.getLogger("document-processor")


class ChartHelper:
    """차트 추출 및 포맷팅 관련 공통 유틸리티"""

    @staticmethod
    def extract_chart_from_ole_stream(ole_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Extract chart data from an OLE compound file stored in BinData.

        HWP stores charts as OLE objects in BinData. The chart data can be in:
        1. 'Contents' stream - Legacy HWP chart format (한글 Neo)
        2. 'OOXMLChartContents' stream - OOXML chart format (한글 2018+)

        Note: HWP BinData streams often have a 4-byte header before the OLE data.

        Args:
            ole_data: Raw bytes of the OLE compound file from BinData

        Returns:
            Dictionary containing chart data or None if not a chart
        """
        try:
            # Check if it's an OLE file
            if not ole_data or len(ole_data) < 12:
                return None

            # OLE magic number: D0 CF 11 E0 A1 B1 1A E1
            OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

            # Check for OLE magic at offset 0 or 4 (HWP often has 4-byte header)
            offset = 0
            if ole_data[:8] == OLE_MAGIC:
                offset = 0
            elif ole_data[4:12] == OLE_MAGIC:
                offset = 4
                logger.debug("Found OLE magic at offset 4 (HWP 4-byte header detected)")
            else:
                # Scan for OLE magic in first 16 bytes
                for i in range(16):
                    if ole_data[i:i+8] == OLE_MAGIC:
                        offset = i
                        logger.debug(f"Found OLE magic at offset {i}")
                        break
                else:
                    return None

            # Skip header bytes
            ole_data = ole_data[offset:]

            # Open as OLE file from bytes
            ole_stream = io.BytesIO(ole_data)

            try:
                ole = olefile.OleFileIO(ole_stream)
            except Exception as e:
                logger.debug(f"Not a valid OLE file: {e}")
                return None

            try:
                chart_data = None

                # Check for OOXML chart first (한글 2018+)
                if ole.exists('OOXMLChartContents'):
                    logger.debug("Found OOXMLChartContents stream - OOXML chart format")
                    stream = ole.openstream('OOXMLChartContents')
                    ooxml_data = stream.read()
                    chart_data = ChartHelper.parse_ooxml_chart(ooxml_data)

                # Fallback to legacy Contents stream
                elif ole.exists('Contents'):
                    logger.debug("Found Contents stream - Legacy HWP chart format")
                    stream = ole.openstream('Contents')
                    contents_data = stream.read()
                    chart_data = ChartHelper.parse_legacy_hwp_chart(contents_data)

                return chart_data

            finally:
                ole.close()

        except Exception as e:
            logger.debug(f"Error extracting chart from OLE stream: {e}")
            return None

    @staticmethod
    def parse_ooxml_chart(ooxml_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse OOXML chart data (DrawingML Chart format).

        OOXML charts follow ISO/IEC 29500 / ECMA 376 DrawingML Chart specification.
        The root element is 'chartSpace' with namespace:
        http://schemas.openxmlformats.org/drawingml/2006/chart

        Key elements:
        - c:chart/c:plotArea - Contains chart series data
        - c:ser - Individual data series
        - c:cat - Category axis data (labels)
        - c:val - Value axis data (numbers)
        - c:tx - Series name/title

        Args:
            ooxml_data: Raw XML bytes of the OOXML chart

        Returns:
            Dictionary with chart title, type, series data, and categories
        """
        try:
            # OOXML namespaces
            namespaces = {
                'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
                'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            }

            # Try to decompress if compressed
            try:
                decompressed = zlib.decompress(ooxml_data)
                ooxml_data = decompressed
            except:
                pass

            # Parse XML
            try:
                root = ET.fromstring(ooxml_data)
            except ET.ParseError:
                # Try removing BOM or invalid chars
                try:
                    ooxml_str = ooxml_data.decode('utf-8-sig', errors='ignore')
                    root = ET.fromstring(ooxml_str)
                except:
                    return None

            chart_info = {
                'type': 'ooxml',
                'chart_type': None,
                'title': None,
                'series': [],
                'categories': []
            }

            # Find chart element
            chart_elem = root.find('.//c:chart', namespaces)
            if chart_elem is None:
                # Try without namespace
                chart_elem = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}chart')

            if chart_elem is None:
                # Root might be chart itself
                if root.tag.endswith('}chart') or root.tag == 'chart':
                    chart_elem = root
                else:
                    return None

            # Extract chart title
            title_elem = chart_elem.find('.//c:title//c:tx//c:rich//a:t', namespaces)
            if title_elem is not None and title_elem.text:
                chart_info['title'] = title_elem.text.strip()
            else:
                # Try alternative path
                title_elem = chart_elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}tx//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
                if title_elem is not None and title_elem.text:
                    chart_info['title'] = title_elem.text.strip()

            # Determine chart type by looking for specific chart type elements
            plot_area = chart_elem.find('.//c:plotArea', namespaces)
            if plot_area is None:
                plot_area = chart_elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}plotArea')

            if plot_area is not None:
                # Check for different chart types
                chart_type_map = {
                    'barChart': '막대 차트',
                    'bar3DChart': '3D 막대 차트',
                    'lineChart': '선 차트',
                    'line3DChart': '3D 선 차트',
                    'pieChart': '파이 차트',
                    'pie3DChart': '3D 파이 차트',
                    'doughnutChart': '도넛 차트',
                    'areaChart': '영역 차트',
                    'area3DChart': '3D 영역 차트',
                    'scatterChart': '분산형 차트',
                    'radarChart': '방사형 차트',
                    'bubbleChart': '거품형 차트',
                    'stockChart': '주식형 차트',
                    'surfaceChart': '표면 차트',
                    'surface3DChart': '3D 표면 차트',
                }

                for chart_tag, chart_name in chart_type_map.items():
                    elem = plot_area.find(f'.//c:{chart_tag}', namespaces)
                    if elem is None:
                        elem = plot_area.find(f'.//{{{namespaces["c"]}}}{chart_tag}')
                    if elem is not None:
                        chart_info['chart_type'] = chart_name
                        # Extract series from this chart type element
                        ChartHelper._extract_ooxml_series(elem, chart_info, namespaces)
                        break

            return chart_info if chart_info['series'] else None

        except Exception as e:
            logger.debug(f"Error parsing OOXML chart: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def _extract_ooxml_series(chart_type_elem, chart_info: Dict[str, Any], namespaces: Dict[str, str]):
        """
        Extract series data from OOXML chart type element.

        Args:
            chart_type_elem: The chart type XML element (e.g., barChart, lineChart)
            chart_info: Dictionary to populate with series data
            namespaces: XML namespaces
        """
        ns_c = namespaces.get('c', 'http://schemas.openxmlformats.org/drawingml/2006/chart')

        # Find all series (c:ser)
        series_elements = chart_type_elem.findall(f'.//c:ser', namespaces)
        if not series_elements:
            series_elements = chart_type_elem.findall(f'.//{{{ns_c}}}ser')

        categories_extracted = False

        for ser_elem in series_elements:
            series_data = {
                'name': None,
                'values': [],
                'categories': []
            }

            # Extract series name (c:tx)
            tx_elem = ser_elem.find('.//c:tx//c:v', namespaces)
            if tx_elem is None:
                tx_elem = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}v')
            if tx_elem is not None and tx_elem.text:
                series_data['name'] = tx_elem.text.strip()
            else:
                # Try strRef path
                str_ref = ser_elem.find('.//c:tx//c:strRef//c:strCache//c:pt//c:v', namespaces)
                if str_ref is None:
                    str_ref = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}strRef//{{{ns_c}}}strCache//{{{ns_c}}}pt//{{{ns_c}}}v')
                if str_ref is not None and str_ref.text:
                    series_data['name'] = str_ref.text.strip()

            # Extract category labels (c:cat) - only need once
            if not categories_extracted:
                cat_elem = ser_elem.find('.//c:cat', namespaces)
                if cat_elem is None:
                    cat_elem = ser_elem.find(f'.//{{{ns_c}}}cat')

                if cat_elem is not None:
                    # Try strRef first (string labels)
                    str_cache = cat_elem.find('.//c:strCache', namespaces)
                    if str_cache is None:
                        str_cache = cat_elem.find(f'.//{{{ns_c}}}strCache')

                    if str_cache is not None:
                        pts = str_cache.findall('.//c:pt', namespaces)
                        if not pts:
                            pts = str_cache.findall(f'.//{{{ns_c}}}pt')

                        for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                            v_elem = pt.find('c:v', namespaces)
                            if v_elem is None:
                                v_elem = pt.find(f'{{{ns_c}}}v')
                            if v_elem is not None and v_elem.text:
                                chart_info['categories'].append(v_elem.text.strip())

                    # Try numCache (numeric labels)
                    if not chart_info['categories']:
                        num_cache = cat_elem.find('.//c:numCache', namespaces)
                        if num_cache is None:
                            num_cache = cat_elem.find(f'.//{{{ns_c}}}numCache')

                        if num_cache is not None:
                            pts = num_cache.findall('.//c:pt', namespaces)
                            if not pts:
                                pts = num_cache.findall(f'.//{{{ns_c}}}pt')

                            for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                                v_elem = pt.find('c:v', namespaces)
                                if v_elem is None:
                                    v_elem = pt.find(f'{{{ns_c}}}v')
                                if v_elem is not None and v_elem.text:
                                    chart_info['categories'].append(v_elem.text.strip())

                    categories_extracted = True

            # Extract values (c:val)
            val_elem = ser_elem.find('.//c:val', namespaces)
            if val_elem is None:
                val_elem = ser_elem.find(f'.//{{{ns_c}}}val')

            if val_elem is not None:
                num_cache = val_elem.find('.//c:numCache', namespaces)
                if num_cache is None:
                    num_cache = val_elem.find(f'.//{{{ns_c}}}numCache')

                if num_cache is not None:
                    pts = num_cache.findall('.//c:pt', namespaces)
                    if not pts:
                        pts = num_cache.findall(f'.//{{{ns_c}}}pt')

                    for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                        v_elem = pt.find('c:v', namespaces)
                        if v_elem is None:
                            v_elem = pt.find(f'{{{ns_c}}}v')
                        if v_elem is not None and v_elem.text:
                            try:
                                series_data['values'].append(float(v_elem.text))
                            except ValueError:
                                series_data['values'].append(v_elem.text)

            # Also check for yVal (scatter/bubble charts)
            if not series_data['values']:
                yval_elem = ser_elem.find('.//c:yVal', namespaces)
                if yval_elem is None:
                    yval_elem = ser_elem.find(f'.//{{{ns_c}}}yVal')

                if yval_elem is not None:
                    num_cache = yval_elem.find('.//c:numCache', namespaces)
                    if num_cache is None:
                        num_cache = yval_elem.find(f'.//{{{ns_c}}}numCache')

                    if num_cache is not None:
                        pts = num_cache.findall('.//c:pt', namespaces)
                        if not pts:
                            pts = num_cache.findall(f'.//{{{ns_c}}}pt')

                        for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                            v_elem = pt.find('c:v', namespaces)
                            if v_elem is None:
                                v_elem = pt.find(f'{{{ns_c}}}v')
                            if v_elem is not None and v_elem.text:
                                try:
                                    series_data['values'].append(float(v_elem.text))
                                except ValueError:
                                    series_data['values'].append(v_elem.text)

            if series_data['values']:
                chart_info['series'].append(series_data)

    @staticmethod
    def parse_legacy_hwp_chart(contents_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse legacy HWP chart format from Contents stream.

        Legacy HWP charts store data in a proprietary binary format with
        ChartObj structures containing DataGrid, Series, and other objects.

        The structure follows the HWP Chart specification with:
        - VtChart object containing overall chart properties
        - DataGrid object containing row/column data
        - Series collection with individual data points

        Args:
            contents_data: Raw bytes from the Contents stream

        Returns:
            Dictionary with chart title, type, series data, and categories
        """
        try:
            chart_info = {
                'type': 'legacy',
                'chart_type': None,
                'title': None,
                'series': [],
                'categories': []
            }

            if not contents_data or len(contents_data) < 100:
                return None

            # Try to decompress if compressed
            try:
                decompressed = zlib.decompress(contents_data, -15)
                contents_data = decompressed
            except:
                try:
                    decompressed = zlib.decompress(contents_data)
                    contents_data = decompressed
                except:
                    pass

            # Parse the binary chart structure
            # The structure is based on VtChart Object specification
            pos = 0
            data_len = len(contents_data)

            # Look for string patterns that might indicate chart data
            # Common patterns in HWP charts: row labels, column labels, numeric data

            # Try to find DataGrid-like structures
            # Search for patterns: RowCount, ColumnCount followed by data

            # Extract strings that might be labels or titles
            strings_found = []
            numeric_data = []

            # Scan for Unicode strings (common in HWP)
            i = 0
            while i < data_len - 4:
                # Check for potential string length marker
                if i + 4 < data_len:
                    try:
                        str_len = struct.unpack('<H', contents_data[i:i+2])[0]
                        if 0 < str_len < 200 and i + 2 + str_len * 2 <= data_len:
                            # Try to decode as UTF-16LE
                            potential_str = contents_data[i+2:i+2+str_len*2].decode('utf-16le', errors='ignore')
                            # Filter for meaningful strings
                            if potential_str and len(potential_str) > 0:
                                # Check if it's printable
                                if all(c.isprintable() or c.isspace() for c in potential_str):
                                    cleaned = potential_str.strip('\x00').strip()
                                    if cleaned and len(cleaned) >= 1:
                                        strings_found.append((i, cleaned))
                    except:
                        pass

                # Look for floating-point numbers (chart data)
                if i + 8 <= data_len:
                    try:
                        val = struct.unpack('<d', contents_data[i:i+8])[0]
                        # Check for reasonable numeric values
                        if -1e10 < val < 1e10 and val != 0:
                            # Additional check: not NaN or Inf
                            import math
                            if not (math.isnan(val) or math.isinf(val)):
                                numeric_data.append((i, val))
                    except:
                        pass

                i += 1

            # Try to identify chart structure from found data
            if strings_found:
                # First meaningful string might be title
                for pos, s in strings_found[:5]:
                    if len(s) > 2 and not s.replace('.', '').replace('-', '').isdigit():
                        chart_info['title'] = s
                        break

                # Look for potential category labels
                # They often appear in sequence
                label_candidates = []
                for pos, s in strings_found:
                    if len(s) >= 1 and len(s) <= 50:
                        label_candidates.append(s)

                # Deduplicate while preserving order
                seen = set()
                unique_labels = []
                for label in label_candidates:
                    if label not in seen and label != chart_info.get('title'):
                        seen.add(label)
                        unique_labels.append(label)

                if len(unique_labels) >= 2:
                    chart_info['categories'] = unique_labels[:20]  # Limit categories

            # Group numeric data into potential series
            if numeric_data:
                # Try to find patterns in numeric data positions
                values = [v for pos, v in numeric_data]

                # If we have categories, try to split values accordingly
                num_categories = len(chart_info['categories']) if chart_info['categories'] else 1
                if num_categories > 0 and len(values) >= num_categories:
                    # Assume series are grouped
                    num_series = len(values) // num_categories
                    for i in range(min(num_series, 10)):  # Limit to 10 series
                        start = i * num_categories
                        end = start + num_categories
                        if end <= len(values):
                            series_values = values[start:end]
                            chart_info['series'].append({
                                'name': f'계열 {i + 1}',
                                'values': series_values
                            })
                else:
                    # Just put all values in one series
                    chart_info['series'].append({
                        'name': '데이터',
                        'values': values[:50]  # Limit values
                    })

            return chart_info if (chart_info['series'] or chart_info['categories']) else None

        except Exception as e:
            logger.debug(f"Error parsing legacy HWP chart: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def format_chart_data_as_table(chart_data: Dict[str, Any]) -> Optional[str]:
        """
        Format extracted chart data as a readable markdown table.

        Returns table string if data is sufficient, None if fallback to image is needed.

        Args:
            chart_data: Dictionary containing chart title, type, categories, and series

        Returns:
            Formatted [chart]...[/chart] string with data table, or None if insufficient data
        """
        if not chart_data:
            return None

        # Check if we have actual data to display
        series = chart_data.get('series', [])
        if not series or all(len(s.get('values', [])) == 0 for s in series):
            return None  # No data, need image fallback

        categories = chart_data.get('categories', [])

        lines = ["[chart]"]

        # Chart title
        if chart_data.get('title'):
            lines.append(f"제목: {chart_data['title']}")

        # Chart type
        if chart_data.get('chart_type'):
            lines.append(f"유형: {chart_data['chart_type']}")

        lines.append("")  # Empty line before table

        # Build header row
        header = ['범주']
        for i, s in enumerate(series):
            name = s.get('name', f'계열 {i + 1}')
            header.append(str(name))

        # Calculate column widths for alignment
        max_values = max(len(s.get('values', [])) for s in series) if series else 0
        num_rows = max(len(categories), max_values)

        # Build data rows
        data_rows = []
        for row_idx in range(num_rows):
            row = []

            # Category label
            if row_idx < len(categories):
                row.append(str(categories[row_idx]))
            else:
                row.append(f'항목 {row_idx + 1}')

            # Series values
            for s in series:
                values = s.get('values', [])
                if row_idx < len(values):
                    val = values[row_idx]
                    if isinstance(val, float):
                        # Format with appropriate precision
                        if val == int(val):
                            row.append(str(int(val)))
                        else:
                            row.append(f'{val:.2f}')
                    else:
                        row.append(str(val))
                else:
                    row.append('-')

            data_rows.append(row)

        # Format as markdown table
        if header and data_rows:
            # Header row
            header_str = '| ' + ' | '.join(header) + ' |'
            lines.append(header_str)

            # Separator row
            sep_str = '| ' + ' | '.join(['---'] * len(header)) + ' |'
            lines.append(sep_str)

            # Data rows
            for row in data_rows:
                row_str = '| ' + ' | '.join(row) + ' |'
                lines.append(row_str)

        lines.append("[/chart]")

        return '\n'.join(lines)

    @staticmethod
    def format_chart_data(chart_data: Dict[str, Any]) -> Optional[str]:
        """
        Alias for format_chart_data_as_table for backward compatibility.
        """
        return ChartHelper.format_chart_data_as_table(chart_data)

    @staticmethod
    def render_chart_to_image(chart_data: Dict[str, Any], app_db=None, processed_images: Set[str] = None) -> Optional[str]:
        """
        차트 데이터를 matplotlib로 이미지로 렌더링하고 MinIO에 업로드합니다.

        테이블 변환 실패 시 폴백으로 사용됩니다.

        Args:
            chart_data: 차트 정보 딕셔너리
            app_db: 데이터베이스 연결
            processed_images: 이미 처리된 이미지 해시 집합

        Returns:
            [chart] 태그로 감싸진 이미지 참조 문자열, 실패 시 None
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available for chart rendering")
            return None

        if not MINIO_AVAILABLE:
            logger.warning("MinIO upload not available for chart rendering")
            return None

        if not chart_data:
            return None

        try:
            categories = chart_data.get('categories', [])
            series_list = chart_data.get('series', [])
            chart_type = chart_data.get('chart_type', '')
            title = chart_data.get('title', '차트')

            if not series_list:
                return None

            # 그래프 생성
            fig, ax = plt.subplots(figsize=(10, 6))

            # 차트 유형에 따른 렌더링
            if '파이' in chart_type or 'pie' in chart_type.lower():
                # 파이 차트
                if series_list and series_list[0].get('values'):
                    values = series_list[0]['values']
                    labels = categories if categories else [f'항목 {i+1}' for i in range(len(values))]
                    ax.pie(values, labels=labels, autopct='%1.1f%%')

            elif '선' in chart_type or 'line' in chart_type.lower():
                # 선 차트
                x = categories if categories else list(range(len(series_list[0].get('values', []))))
                for series in series_list:
                    values = series.get('values', [])
                    name = series.get('name', '시리즈')
                    if values:
                        ax.plot(x[:len(values)], values, marker='o', label=name)
                ax.legend()
                ax.grid(True, alpha=0.3)

            elif '영역' in chart_type or 'area' in chart_type.lower():
                # 영역 차트
                x = list(range(len(series_list[0].get('values', []))))
                for series in series_list:
                    values = series.get('values', [])
                    name = series.get('name', '시리즈')
                    if values:
                        ax.fill_between(x[:len(values)], values, alpha=0.5, label=name)
                ax.legend()
                ax.grid(True, alpha=0.3)

            else:
                # 기본: 막대 차트
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

            minio_path = upload_image_to_minio(img_data, app_db=app_db, processed_images=processed_images)

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

    @staticmethod
    def process_chart(chart_data: Dict[str, Any], app_db=None, processed_images: Set[str] = None) -> str:
        """
        차트를 처리합니다.

        1순위: 데이터를 표(테이블)로 변환 - LLM이 직접 해석 가능
        2순위: 실패 시 matplotlib로 이미지 생성 후 MinIO 업로드

        Args:
            chart_data: 차트 정보 딕셔너리
            app_db: 데이터베이스 연결
            processed_images: 이미 처리된 이미지 해시 집합

        Returns:
            [chart]...[/chart] 형태의 문자열
        """
        # 1순위: 테이블로 변환 시도
        table_result = ChartHelper.format_chart_data_as_table(chart_data)
        if table_result:
            logger.debug("Chart converted to table successfully")
            return table_result

        # 2순위: 이미지로 렌더링
        image_result = ChartHelper.render_chart_to_image(chart_data, app_db, processed_images)
        if image_result:
            logger.debug("Chart rendered to image successfully")
            return image_result

        # 최종 폴백: 기본 정보만 출력
        result_parts = ["[chart]"]
        if chart_data and chart_data.get('title'):
            result_parts.append(f"제목: {chart_data['title']}")
        if chart_data and chart_data.get('chart_type'):
            result_parts.append(f"유형: {chart_data['chart_type']}")
        result_parts.append("[/chart]")
        return "\n".join(result_parts)


__all__ = [
    'ChartHelper',
]
