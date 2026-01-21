"""
HWP Chart Extractor

Extracts chart data from HWP files.
Supports both OOXML charts (한글 2018+) and legacy HWP charts.

Provides:
- extract(): Single chart extraction from OLE bytes
- extract_all_from_file(): Extract all charts from HWP file
"""
import io
import logging
import os
import struct
import xml.etree.ElementTree as ET
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

# OLE magic signature
OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

# Image extensions to skip
SKIP_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.wmf', '.emf'}


class HWPChartExtractor(BaseChartExtractor):
    """
    Chart extractor for HWP files.
    
    HWP stores charts as OLE objects in BinData streams.
    Supports:
    - OOXML chart format (한글 2018+) via 'OOXMLChartContents' stream
    - Legacy HWP chart format via 'Contents' stream
    """
    
    # ========================================================================
    # Main Interface
    # ========================================================================
    
    def extract(self, chart_element: Any) -> ChartData:
        """
        Extract chart data from HWP OLE stream data.
        
        Args:
            chart_element: Raw bytes of OLE compound file from BinData
                
        Returns:
            ChartData with extracted information
        """
        if not chart_element or not isinstance(chart_element, bytes):
            return ChartData()
        
        ole_data = self._prepare_ole_data(chart_element)
        if not ole_data:
            return ChartData()
        
        return self._extract_from_ole(ole_data)
    
    def extract_all_from_file(
        self,
        file_source: Union[str, bytes, BinaryIO]
    ) -> List[ChartData]:
        """
        Extract all charts from an HWP file.
        
        Args:
            file_source: File path, bytes, or file-like object
            
        Returns:
            List of ChartData for all charts in the file
        """
        charts = []
        
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
            
            # Check if valid OLE file
            file_obj.seek(0)
            header = file_obj.read(8)
            file_obj.seek(0)
            
            if header != OLE_MAGIC:
                logger.debug("Not a valid HWP OLE file")
                return charts
            
            ole = olefile.OleFileIO(file_obj)
            
            try:
                # Find all BinData streams
                bindata_streams = [
                    e for e in ole.listdir()
                    if len(e) >= 2 and e[0] == "BinData"
                ]
                
                for stream_path in bindata_streams:
                    stream_name = stream_path[-1]
                    ext = os.path.splitext(stream_name)[1].lower()
                    
                    # Skip image files
                    if ext in SKIP_IMAGE_EXTENSIONS:
                        continue
                    
                    chart_data = self._process_chart_stream(ole, stream_path)
                    if chart_data.has_data():
                        charts.append(chart_data)
                        logger.debug(f"Extracted chart from: {'/'.join(stream_path)}")
                        
            finally:
                ole.close()
            
            logger.info(f"Extracted {len(charts)} charts from HWP file")
            
        except Exception as e:
            logger.error(f"Error extracting charts from HWP: {e}")
        
        return charts
    
    def _process_chart_stream(self, ole, stream_path: List[str]) -> ChartData:
        """Process a single BinData stream for chart data."""
        try:
            stream = ole.openstream(stream_path)
            ole_data = stream.read()
            
            # Try decompression
            try:
                ole_data = zlib.decompress(ole_data, -15)
            except:
                try:
                    ole_data = zlib.decompress(ole_data)
                except:
                    pass
            
            return self.extract(ole_data)
            
        except Exception as e:
            logger.debug(f"Error processing chart stream: {e}")
            return ChartData()
    
    def _prepare_ole_data(self, raw_data: bytes) -> Optional[bytes]:
        """Prepare OLE data by finding and extracting OLE compound file."""
        if len(raw_data) < 12:
            return None
        
        OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
        
        # Find OLE magic
        offset = 0
        if raw_data[:8] == OLE_MAGIC:
            offset = 0
        elif raw_data[4:12] == OLE_MAGIC:
            offset = 4  # HWP often has 4-byte header
        else:
            for i in range(16):
                if raw_data[i:i+8] == OLE_MAGIC:
                    offset = i
                    break
            else:
                return None
        
        return raw_data[offset:]
    
    def _extract_from_ole(self, ole_data: bytes) -> ChartData:
        """Extract chart data from OLE compound file."""
        try:
            ole_stream = io.BytesIO(ole_data)
            ole = olefile.OleFileIO(ole_stream)
            
            try:
                # Try OOXML format first (한글 2018+)
                if ole.exists('OOXMLChartContents'):
                    stream = ole.openstream('OOXMLChartContents')
                    ooxml_data = stream.read()
                    return self._parse_ooxml_chart(ooxml_data)
                
                # Fallback to legacy format
                if ole.exists('Contents'):
                    stream = ole.openstream('Contents')
                    contents_data = stream.read()
                    return self._parse_legacy_chart(contents_data)
                
                return ChartData()
                
            finally:
                ole.close()
                
        except Exception as e:
            logger.debug(f"Error extracting chart from OLE: {e}")
            return ChartData()
    
    def _parse_ooxml_chart(self, ooxml_data: bytes) -> ChartData:
        """Parse OOXML chart format."""
        try:
            root = ET.fromstring(ooxml_data)
            
            # Find chart element
            chart_elem = root.find('.//c:chart', OOXML_NS)
            if chart_elem is None:
                chart_elem = root.find('.//{http://schemas.openxmlformats.org/drawingml/2006/chart}chart')
            if chart_elem is None:
                return ChartData()
            
            # Extract title
            title = self._extract_ooxml_title(chart_elem)
            
            # Extract plot data
            chart_type, categories, series = self._extract_ooxml_plot_data(chart_elem)
            
            return ChartData(
                chart_type=chart_type,
                title=title,
                categories=categories,
                series=series
            )
            
        except Exception as e:
            logger.debug(f"Error parsing OOXML chart: {e}")
            return ChartData()
    
    def _extract_ooxml_title(self, chart_elem) -> Optional[str]:
        """Extract title from OOXML chart."""
        title_elem = chart_elem.find('.//c:title//c:tx//c:rich//a:t', OOXML_NS)
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        return None
    
    def _extract_ooxml_plot_data(self, chart_elem) -> tuple:
        """Extract chart type, categories, and series from OOXML."""
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
                categories, series = self._extract_ooxml_series(elem)
                break
        
        return chart_type, categories, series
    
    def _extract_ooxml_series(self, chart_type_elem) -> tuple:
        """Extract series data from OOXML chart type element."""
        ns_c = OOXML_NS['c']
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
                    categories = self._extract_ooxml_string_cache(cat_elem)
                categories_extracted = True
            
            # Extract values
            values = []
            val_elem = ser_elem.find('.//c:val', OOXML_NS)
            if val_elem is not None:
                values = self._extract_ooxml_num_cache(val_elem)
            
            if values:
                series.append({'name': name, 'values': values})
        
        return categories, series
    
    def _extract_ooxml_string_cache(self, cat_elem) -> List[str]:
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
    
    def _extract_ooxml_num_cache(self, val_elem) -> List[Any]:
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
    
    def _parse_legacy_chart(self, contents_data: bytes) -> ChartData:
        """Parse legacy HWP chart format."""
        try:
            # Legacy format uses record-based structure
            # Try to extract basic info
            title = None
            categories = []
            series = []
            
            # Scan for UTF-16LE text patterns
            try:
                text = contents_data.decode('utf-16le', errors='ignore')
                # Look for title-like strings
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                if lines:
                    title = lines[0][:50]  # First line might be title
            except:
                pass
            
            return ChartData(
                chart_type="Chart",
                title=title,
                categories=categories,
                series=series
            )
            
        except Exception as e:
            logger.debug(f"Error parsing legacy chart: {e}")
            return ChartData()


__all__ = ['HWPChartExtractor']
