"""
HWPX Chart Extractor

Extracts chart data from HWPX files.
HWPX uses OOXML-based chart format similar to Office documents.

Provides:
- extract(): Single chart XML extraction
- extract_all_from_file(): Extract all charts from HWPX file
"""
import io
import logging
import xml.etree.ElementTree as ET
import zipfile
import zlib
from typing import Any, BinaryIO, Dict, List, Optional, Union

import olefile

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
}

# OLE file magic signature
OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

# Image extensions to skip
SKIP_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.wmf', '.emf'}


class HWPXChartExtractor(BaseChartExtractor):
    """
    Chart extractor for HWPX files.
    
    HWPX is the Open Document format for Hangul.
    Charts are stored as:
    - OOXML XML in Chart/, Charts/, or Contents/Charts/ directory
    - OLE objects in BinData/ directory
    """
    
    # ========================================================================
    # Main Interface
    # ========================================================================
    
    def extract(self, chart_element: Any) -> ChartData:
        """
        Extract chart data from HWPX chart XML or OLE data.
        
        Args:
            chart_element: Chart XML bytes or OLE bytes from HWPX archive
                
        Returns:
            ChartData with extracted information
        """
        if not chart_element:
            return ChartData()
        
        if isinstance(chart_element, bytes):
            # Try as OOXML first
            result = self._parse_chart_xml(chart_element)
            if result.has_data():
                return result
            # Try as OLE
            return self._extract_from_ole(chart_element)
        elif isinstance(chart_element, str):
            return self._parse_chart_xml(chart_element.encode('utf-8'))
        
        return ChartData()
    
    def extract_all_from_file(
        self,
        file_source: Union[str, bytes, BinaryIO]
    ) -> List[ChartData]:
        """
        Extract all charts from an HWPX file.
        
        Args:
            file_source: File path, bytes, or file-like object
            
        Returns:
            List of ChartData for all charts in the file
        """
        charts = []
        processed_hashes = set()
        
        try:
            # Prepare file-like object
            if isinstance(file_source, str):
                with open(file_source, 'rb') as f:
                    file_obj = io.BytesIO(f.read())
            elif isinstance(file_source, bytes):
                file_obj = io.BytesIO(file_source)
            else:
                file_source.seek(0)
                file_obj = file_source
            
            with zipfile.ZipFile(file_obj, 'r') as zf:
                namelist = zf.namelist()
                
                # 1. Extract OOXML charts
                charts.extend(self._extract_ooxml_charts(zf, namelist, processed_hashes))
                
                # 2. Extract OLE charts from BinData
                charts.extend(self._extract_ole_charts(zf, namelist, processed_hashes))
            
            logger.info(f"Extracted {len(charts)} charts from HWPX file")
            
        except Exception as e:
            logger.error(f"Error extracting charts from HWPX: {e}")
        
        return charts

    def extract_all_with_refs(
        self,
        file_source: Union[str, bytes, BinaryIO]
    ) -> Dict[str, ChartData]:
        """
        Extract all charts from an HWPX file with their chartIDRefs.
        
        This method returns a dictionary mapping chartIDRef (e.g., "Chart/chart1.xml")
        to ChartData, allowing for inline chart processing in document order.
        
        Args:
            file_source: File path, bytes, or file-like object
            
        Returns:
            Dictionary mapping chartIDRef -> ChartData
        """
        chart_map: Dict[str, ChartData] = {}
        processed_hashes = set()
        
        try:
            # Prepare file-like object
            if isinstance(file_source, str):
                with open(file_source, 'rb') as f:
                    file_obj = io.BytesIO(f.read())
            elif isinstance(file_source, bytes):
                file_obj = io.BytesIO(file_source)
            else:
                file_source.seek(0)
                file_obj = file_source
            
            with zipfile.ZipFile(file_obj, 'r') as zf:
                namelist = zf.namelist()
                
                # Extract OOXML charts with their references
                chart_files = [
                    f for f in namelist
                    if (f.startswith('Chart/') and f.endswith('.xml'))
                    or (f.startswith('Contents/Charts/') and f.endswith('.xml'))
                    or (f.startswith('Charts/') and f.endswith('.xml'))
                ]
                
                for chart_file in sorted(chart_files):
                    try:
                        with zf.open(chart_file) as f:
                            chart_xml = f.read()
                        
                        chart_data = self._parse_chart_xml(chart_xml)
                        
                        if chart_data.has_data():
                            # Duplicate check
                            chart_hash = f"{chart_data.title}|{chart_data.series}"
                            if chart_hash in processed_hashes:
                                continue
                            processed_hashes.add(chart_hash)
                            
                            # Map by chartIDRef (e.g., "Chart/chart1.xml")
                            chart_map[chart_file] = chart_data
                            logger.debug(f"Mapped chart: {chart_file}")
                            
                    except Exception as e:
                        logger.debug(f"Error reading chart file {chart_file}: {e}")
            
            logger.info(f"Extracted {len(chart_map)} charts with refs from HWPX file")
            
        except Exception as e:
            logger.error(f"Error extracting charts from HWPX: {e}")
        
        return chart_map
    
    def _parse_chart_xml(self, chart_xml: bytes) -> ChartData:
        """Parse OOXML chart XML."""
        try:
            root = ET.fromstring(chart_xml)
            
            # Find chart element
            chart_elem = root.find('.//c:chart', OOXML_NS)
            if chart_elem is None:
                chart_elem = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}chart')
            if chart_elem is None:
                if root.tag.endswith('}chart') or root.tag == 'chart':
                    chart_elem = root
                else:
                    return ChartData()
            
            # Extract title
            title = self._extract_title(chart_elem)
            
            # Extract plot data
            chart_type, categories, series = self._extract_plot_data(chart_elem)
            
            return ChartData(
                chart_type=chart_type,
                title=title,
                categories=categories,
                series=series
            )
            
        except Exception as e:
            logger.debug(f"Error parsing HWPX chart: {e}")
            return ChartData()
    
    def _extract_title(self, chart_elem) -> Optional[str]:
        """Extract chart title."""
        title_elem = chart_elem.find('.//c:title//c:tx//c:rich//a:t', OOXML_NS)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        return None
    
    def _extract_plot_data(self, chart_elem) -> tuple:
        """Extract chart type, categories, and series."""
        plot_area = chart_elem.find('.//c:plotArea', OOXML_NS)
        if plot_area is None:
            return "Chart", [], []
        
        chart_type = "Chart"
        categories = []
        series = []
        
        for chart_tag, type_name in CHART_TYPE_MAP.items():
            elem = plot_area.find(f'.//c:{chart_tag}', OOXML_NS)
            if elem is not None:
                chart_type = type_name
                categories, series = self._extract_series_data(elem)
                break
        
        return chart_type, categories, series
    
    def _extract_series_data(self, chart_type_elem) -> tuple:
        """Extract series and categories from chart type element."""
        categories = []
        series = []
        categories_extracted = False
        
        series_elements = chart_type_elem.findall('.//c:ser', OOXML_NS)
        
        for idx, ser_elem in enumerate(series_elements):
            # Extract series name
            name = f"Series {idx + 1}"
            tx_elem = ser_elem.find('.//c:tx//c:v', OOXML_NS)
            if tx_elem is not None and tx_elem.text:
                name = tx_elem.text.strip()
            
            # Extract categories from first series
            if not categories_extracted:
                cat_elem = ser_elem.find('.//c:cat', OOXML_NS)
                if cat_elem is not None:
                    categories = self._extract_string_cache(cat_elem)
                categories_extracted = True
            
            # Extract values
            values = []
            val_elem = ser_elem.find('.//c:val', OOXML_NS)
            if val_elem is not None:
                values = self._extract_num_cache(val_elem)
            
            if values:
                series.append({'name': name, 'values': values})
        
        return categories, series
    
    def _extract_string_cache(self, cat_elem) -> List[str]:
        """Extract string cache values."""
        values = []
        str_cache = cat_elem.find('.//c:strCache', OOXML_NS)
        if str_cache is not None:
            pts = str_cache.findall('.//c:pt', OOXML_NS)
            for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                v = pt.find('c:v', OOXML_NS)
                if v is not None and v.text:
                    values.append(v.text.strip())
        return values
    
    def _extract_num_cache(self, val_elem) -> List[Any]:
        """Extract numeric cache values."""
        values = []
        num_cache = val_elem.find('.//c:numCache', OOXML_NS)
        if num_cache is not None:
            pts = num_cache.findall('.//c:pt', OOXML_NS)
            for pt in sorted(pts, key=lambda x: int(x.get('idx', 0))):
                v = pt.find('c:v', OOXML_NS)
                if v is not None and v.text:
                    try:
                        values.append(float(v.text))
                    except ValueError:
                        values.append(v.text)
        return values
    
    # ========================================================================
    # File-Level Extraction Helpers
    # ========================================================================
    
    def _extract_ooxml_charts(
        self,
        zf: zipfile.ZipFile,
        namelist: List[str],
        processed_hashes: set
    ) -> List[ChartData]:
        """Extract OOXML charts from ZIP archive."""
        charts = []
        
        chart_files = [
            f for f in namelist
            if (f.startswith('Chart/') and f.endswith('.xml'))
            or (f.startswith('Contents/Charts/') and f.endswith('.xml'))
            or (f.startswith('Charts/') and f.endswith('.xml'))
        ]
        
        for chart_file in sorted(chart_files):
            try:
                with zf.open(chart_file) as f:
                    chart_xml = f.read()
                
                chart_data = self._parse_chart_xml(chart_xml)
                
                if chart_data.has_data():
                    # Duplicate check
                    chart_hash = f"{chart_data.title}|{chart_data.series}"
                    if chart_hash in processed_hashes:
                        continue
                    processed_hashes.add(chart_hash)
                    
                    charts.append(chart_data)
                    logger.debug(f"Extracted chart from: {chart_file}")
                    
            except Exception as e:
                logger.debug(f"Error reading chart file {chart_file}: {e}")
        
        return charts
    
    def _extract_ole_charts(
        self,
        zf: zipfile.ZipFile,
        namelist: List[str],
        processed_hashes: set
    ) -> List[ChartData]:
        """Extract OLE charts from BinData directory."""
        charts = []
        
        bindata_files = [
            f for f in namelist
            if f.startswith('BinData/') and not f.endswith('/')
        ]
        
        for bindata_file in bindata_files:
            import os
            ext = os.path.splitext(bindata_file)[1].lower()
            
            if ext in SKIP_IMAGE_EXTENSIONS:
                continue
            
            try:
                with zf.open(bindata_file) as f:
                    data = f.read()
                
                # Try decompression
                try:
                    data = zlib.decompress(data, -15)
                except:
                    try:
                        data = zlib.decompress(data)
                    except:
                        pass
                
                chart_data = self._extract_from_ole(data)
                
                if chart_data.has_data():
                    # Duplicate check
                    chart_hash = f"{chart_data.title}|{chart_data.series}"
                    if chart_hash in processed_hashes:
                        continue
                    processed_hashes.add(chart_hash)
                    
                    charts.append(chart_data)
                    logger.debug(f"Extracted OLE chart from: {bindata_file}")
                    
            except Exception as e:
                logger.debug(f"Error reading bindata file {bindata_file}: {e}")
        
        return charts
    
    def _extract_from_ole(self, ole_data: bytes) -> ChartData:
        """Extract chart from OLE compound file."""
        if len(ole_data) < 12:
            return ChartData()
        
        # Find OLE magic
        offset = 0
        if ole_data[:8] == OLE_MAGIC:
            offset = 0
        elif len(ole_data) > 12 and ole_data[4:12] == OLE_MAGIC:
            offset = 4
        else:
            for i in range(16):
                if ole_data[i:i+8] == OLE_MAGIC:
                    offset = i
                    break
            else:
                return ChartData()
        
        try:
            ole_stream = io.BytesIO(ole_data[offset:])
            ole = olefile.OleFileIO(ole_stream)
            
            try:
                # Try OOXML format first
                if ole.exists('OOXMLChartContents'):
                    stream = ole.openstream('OOXMLChartContents')
                    ooxml_data = stream.read()
                    return self._parse_chart_xml(ooxml_data)
                
                # Try Contents stream
                if ole.exists('Contents'):
                    stream = ole.openstream('Contents')
                    contents_data = stream.read()
                    # Try as OOXML first
                    result = self._parse_chart_xml(contents_data)
                    if result.has_data():
                        return result
                
                return ChartData()
                
            finally:
                ole.close()
                
        except Exception as e:
            logger.debug(f"Error extracting chart from OLE: {e}")
            return ChartData()


__all__ = ['HWPXChartExtractor']
