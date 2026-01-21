"""
DOCX Chart Extractor

Extracts all chart data from DOCX files.
Parses OOXML DrawingML Chart format (ISO/IEC 29500).

Structure:
- Charts are stored in word/charts/chart*.xml
- Referenced via relationships in document.xml
"""
import io
import logging
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Union, BinaryIO

from contextifier.core.functions.chart_extractor import BaseChartExtractor, ChartData

logger = logging.getLogger("document-processor")

# OOXML namespaces
OOXML_NS = {
    'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
}

# Chart type mapping
CHART_TYPE_MAP = {
    'barChart': 'Bar Chart',
    'bar3DChart': '3D Bar Chart',
    'lineChart': 'Line Chart',
    'line3DChart': '3D Line Chart',
    'pieChart': 'Pie Chart',
    'pie3DChart': '3D Pie Chart',
    'doughnutChart': 'Doughnut Chart',
    'areaChart': 'Area Chart',
    'area3DChart': '3D Area Chart',
    'scatterChart': 'Scatter Chart',
    'bubbleChart': 'Bubble Chart',
    'radarChart': 'Radar Chart',
    'surfaceChart': 'Surface Chart',
    'surface3DChart': '3D Surface Chart',
    'stockChart': 'Stock Chart',
}


class DOCXChartExtractor(BaseChartExtractor):
    """
    Chart extractor for DOCX files.
    
    Extracts all charts from DOCX by parsing word/charts/*.xml files.
    """
    
    # ========================================================================
    # Main Interface
    # ========================================================================
    
    def extract(self, chart_element: Any) -> ChartData:
        """
        Extract chart data from various input types.
        
        Args:
            chart_element: One of:
                - bytes: Raw chart XML
                - object with 'blob' attribute: Chart part object
                
        Returns:
            ChartData with extracted information
        """
        if not chart_element:
            return ChartData()
        
        # Handle chart XML bytes
        if isinstance(chart_element, bytes):
            return self._parse_ooxml_chart(chart_element)
        
        # Handle object with blob attribute (chart part)
        if hasattr(chart_element, 'blob'):
            return self._parse_ooxml_chart(chart_element.blob)
        
        return ChartData()
    
    def extract_all_from_file(
        self, 
        file_source: Union[str, bytes, BinaryIO]
    ) -> List[ChartData]:
        """
        Extract all charts from a DOCX file.
        
        Args:
            file_source: File path, bytes, or file-like object
            
        Returns:
            List of ChartData for all charts in the file (in document order)
        """
        charts = []
        
        try:
            # Prepare file-like object
            if isinstance(file_source, str):
                zf = zipfile.ZipFile(file_source, 'r')
            elif isinstance(file_source, bytes):
                zf = zipfile.ZipFile(io.BytesIO(file_source), 'r')
            else:
                file_source.seek(0)
                zf = zipfile.ZipFile(file_source, 'r')
            
            try:
                # Find all chart XML files in word/charts/
                chart_files = sorted([
                    name for name in zf.namelist()
                    if name.startswith('word/charts/chart') and name.endswith('.xml')
                ])
                
                for chart_file in chart_files:
                    try:
                        chart_xml = zf.read(chart_file)
                        chart_data = self._parse_ooxml_chart(chart_xml)
                        if chart_data.has_data():
                            charts.append(chart_data)
                        else:
                            # Even empty charts should be tracked for position matching
                            charts.append(chart_data)
                    except Exception as e:
                        logger.debug(f"Error parsing chart {chart_file}: {e}")
                        charts.append(ChartData())  # Placeholder for failed chart
                        
            finally:
                zf.close()
                
            logger.debug(f"Extracted {len(charts)} charts from DOCX file")
            
        except Exception as e:
            logger.warning(f"Error extracting charts from DOCX: {e}")
        
        return charts
    
    def process_all_from_file(
        self, 
        file_source: Union[str, bytes, BinaryIO]
    ) -> List[str]:
        """
        Extract and format all charts from a DOCX file.
        
        Args:
            file_source: File path, bytes, or file-like object
            
        Returns:
            List of formatted chart strings
        """
        results = []
        
        for chart_data in self.extract_all_from_file(file_source):
            formatted = self._format_chart_data(chart_data)
            if formatted:
                results.append(formatted)
        
        return results
    
    # ========================================================================
    # OOXML Chart Parsing
    # ========================================================================
    
    def _parse_ooxml_chart(self, chart_xml: bytes) -> ChartData:
        """Parse OOXML chart XML."""
        try:
            # Parse XML
            root = self._parse_xml(chart_xml)
            if root is None:
                return ChartData()
            
            # Find chart element
            chart_elem = self._find_chart_element(root)
            if chart_elem is None:
                return ChartData()
            
            # Extract title
            title = self._extract_title(chart_elem)
            
            # Extract chart type and series data
            chart_type, categories, series = self._extract_plot_data(chart_elem)
            
            return ChartData(
                chart_type=chart_type,
                title=title,
                categories=categories if categories else None,
                series=series if series else None
            )
            
        except Exception as e:
            logger.debug(f"Error parsing OOXML chart: {e}")
            return ChartData()
    
    def _parse_xml(self, chart_xml: bytes) -> Optional[ET.Element]:
        """Parse XML with BOM and encoding handling."""
        try:
            return ET.fromstring(chart_xml)
        except ET.ParseError:
            try:
                chart_str = chart_xml.decode('utf-8-sig', errors='ignore')
                return ET.fromstring(chart_str)
            except:
                return None
    
    def _find_chart_element(self, root: ET.Element) -> Optional[ET.Element]:
        """Find the chart element in the XML tree."""
        chart_elem = root.find('.//c:chart', OOXML_NS)
        if chart_elem is not None:
            return chart_elem
        
        chart_elem = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}chart')
        if chart_elem is not None:
            return chart_elem
        
        if root.tag.endswith('}chart') or root.tag == 'chart':
            return root
        
        return None
    
    def _extract_title(self, chart_elem: ET.Element) -> Optional[str]:
        """Extract chart title."""
        # Primary path
        title_elem = chart_elem.find('.//c:title//c:tx//c:rich//a:t', OOXML_NS)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        
        # Alternative path
        title_elem = chart_elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}tx//{http://schemas.openxmlformats.org/drawingml/2006/main}t')
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        
        return None
    
    def _extract_plot_data(self, chart_elem: ET.Element) -> tuple:
        """Extract chart type, categories, and series."""
        plot_area = chart_elem.find('.//c:plotArea', OOXML_NS)
        if plot_area is None:
            plot_area = chart_elem.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}plotArea')
        
        if plot_area is None:
            return "Chart", [], []
        
        for chart_tag, type_name in CHART_TYPE_MAP.items():
            elem = plot_area.find(f'.//c:{chart_tag}', OOXML_NS)
            if elem is None:
                elem = plot_area.find(f'.//{{{OOXML_NS["c"]}}}{chart_tag}')
            if elem is not None:
                categories, series = self._extract_series_data(elem)
                return type_name, categories, series
        
        return "Chart", [], []
    
    def _extract_series_data(self, chart_type_elem: ET.Element) -> tuple:
        """Extract categories and series data."""
        ns_c = OOXML_NS['c']
        categories = []
        series = []
        categories_extracted = False
        
        series_elements = chart_type_elem.findall('.//c:ser', OOXML_NS)
        if not series_elements:
            series_elements = chart_type_elem.findall(f'.//{{{ns_c}}}ser')
        
        for idx, ser_elem in enumerate(series_elements):
            series_data = {
                'name': self._extract_series_name(ser_elem, idx),
                'values': []
            }
            
            if not categories_extracted:
                categories = self._extract_categories(ser_elem)
                categories_extracted = True
            
            series_data['values'] = self._extract_values(ser_elem)
            
            if series_data['values']:
                series.append(series_data)
        
        return categories, series
    
    def _extract_series_name(self, ser_elem: ET.Element, idx: int) -> str:
        """Extract series name."""
        ns_c = OOXML_NS['c']
        
        tx_elem = ser_elem.find('.//c:tx//c:v', OOXML_NS)
        if tx_elem is None:
            tx_elem = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}v')
        if tx_elem is not None and tx_elem.text:
            return tx_elem.text.strip()
        
        str_ref = ser_elem.find('.//c:tx//c:strRef//c:strCache//c:pt//c:v', OOXML_NS)
        if str_ref is None:
            str_ref = ser_elem.find(f'.//{{{ns_c}}}tx//{{{ns_c}}}strRef//{{{ns_c}}}strCache//{{{ns_c}}}pt//{{{ns_c}}}v')
        if str_ref is not None and str_ref.text:
            return str_ref.text.strip()
        
        return f"Series {idx + 1}"
    
    def _extract_categories(self, ser_elem: ET.Element) -> List[str]:
        """Extract category labels."""
        ns_c = OOXML_NS['c']
        categories = []
        
        cat_elem = ser_elem.find('.//c:cat', OOXML_NS)
        if cat_elem is None:
            cat_elem = ser_elem.find(f'.//{{{ns_c}}}cat')
        
        if cat_elem is None:
            return categories
        
        # Try string cache
        str_cache = cat_elem.find('.//c:strCache', OOXML_NS)
        if str_cache is None:
            str_cache = cat_elem.find(f'.//{{{ns_c}}}strCache')
        
        if str_cache is not None:
            categories = self._extract_point_values(str_cache, as_string=True)
        
        # Fallback to numeric cache
        if not categories:
            num_cache = cat_elem.find('.//c:numCache', OOXML_NS)
            if num_cache is None:
                num_cache = cat_elem.find(f'.//{{{ns_c}}}numCache')
            
            if num_cache is not None:
                categories = self._extract_point_values(num_cache, as_string=True)
        
        return categories
    
    def _extract_values(self, ser_elem: ET.Element) -> List[Any]:
        """Extract series values."""
        ns_c = OOXML_NS['c']
        values = []
        
        val_elem = ser_elem.find('.//c:val', OOXML_NS)
        if val_elem is None:
            val_elem = ser_elem.find(f'.//{{{ns_c}}}val')
        
        if val_elem is not None:
            num_cache = val_elem.find('.//c:numCache', OOXML_NS)
            if num_cache is None:
                num_cache = val_elem.find(f'.//{{{ns_c}}}numCache')
            
            if num_cache is not None:
                values = self._extract_point_values(num_cache, as_string=False)
        
        # Try yVal for scatter/bubble charts
        if not values:
            yval_elem = ser_elem.find('.//c:yVal', OOXML_NS)
            if yval_elem is None:
                yval_elem = ser_elem.find(f'.//{{{ns_c}}}yVal')
            
            if yval_elem is not None:
                num_cache = yval_elem.find('.//c:numCache', OOXML_NS)
                if num_cache is None:
                    num_cache = yval_elem.find(f'.//{{{ns_c}}}numCache')
                
                if num_cache is not None:
                    values = self._extract_point_values(num_cache, as_string=False)
        
        return values
    
    def _extract_point_values(self, cache_elem: ET.Element, as_string: bool = False) -> List[Any]:
        """Extract values from cache element."""
        ns_c = OOXML_NS['c']
        values = []
        
        pts = cache_elem.findall('.//c:pt', OOXML_NS)
        if not pts:
            pts = cache_elem.findall(f'.//{{{ns_c}}}pt')
        
        for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
            v_elem = pt.find('c:v', OOXML_NS)
            if v_elem is None:
                v_elem = pt.find(f'{{{ns_c}}}v')
            
            if v_elem is not None and v_elem.text:
                text = v_elem.text.strip()
                if as_string:
                    values.append(text)
                else:
                    try:
                        values.append(float(text))
                    except ValueError:
                        values.append(text)
        
        return values
    
    # ========================================================================
    # Formatting
    # ========================================================================
    
    def _format_chart_data(self, chart_data: ChartData) -> str:
        """Format ChartData using ChartProcessor."""
        if chart_data.has_data():
            return self._chart_processor.format_chart_data(
                chart_type=chart_data.chart_type,
                title=chart_data.title,
                categories=chart_data.categories,
                series=chart_data.series
            )
        else:
            return self._chart_processor.format_chart_fallback(
                chart_type=chart_data.chart_type,
                title=chart_data.title
            )


__all__ = ['DOCXChartExtractor']
