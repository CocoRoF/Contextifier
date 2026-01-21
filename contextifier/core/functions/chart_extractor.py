"""
Chart Extractor Base Module

Abstract base class for chart extraction across different file formats.
Each file handler should have its own ChartExtractor implementation.

Output format:
    {chart_prefix}
    Title: {chart_title}
    Chart Type: {chart_type}
    <table>...</table>
    {chart_suffix}
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from contextifier.core.functions.chart_processor import ChartProcessor


@dataclass
class ChartData:
    """
    Standardized chart data structure.
    
    All chart extractors should convert their format-specific chart data
    into this common structure before formatting.
    
    Attributes:
        chart_type: Type of chart (e.g., "Bar Chart", "Line Chart", "Pie Chart")
        title: Chart title (optional)
        categories: List of category labels (X-axis values)
        series: List of series data, each containing 'name' and 'values'
    """
    chart_type: str = "Chart"
    title: Optional[str] = None
    categories: Optional[List[str]] = None
    series: Optional[List[Dict[str, Any]]] = None
    
    def has_data(self) -> bool:
        """Check if chart has extractable data."""
        if not self.series:
            return False
        return any(s.get('values') for s in self.series)


class BaseChartExtractor(ABC):
    """
    Abstract base class for chart extraction.
    
    Each file format handler should implement its own ChartExtractor
    that inherits from this class.
    
    Usage:
        class ExcelChartExtractor(BaseChartExtractor):
            def extract(self, chart_element) -> ChartData:
                # Excel-specific extraction logic
                ...
    """
    
    def __init__(self, chart_processor: "ChartProcessor"):
        """
        Initialize chart extractor.
        
        Args:
            chart_processor: ChartProcessor instance for formatting output
        """
        self._chart_processor = chart_processor
    
    @property
    def chart_processor(self) -> "ChartProcessor":
        """ChartProcessor instance."""
        return self._chart_processor
    
    @abstractmethod
    def extract(self, chart_element: Any) -> ChartData:
        """
        Extract chart data from format-specific chart element.
        
        Args:
            chart_element: Format-specific chart object/element
            
        Returns:
            ChartData with extracted information
        """
        pass
    
    def process(self, chart_element: Any) -> str:
        """
        Extract and format chart data.
        
        This is the main entry point for chart processing.
        Extracts data using format-specific logic, then formats using ChartProcessor.
        
        Args:
            chart_element: Format-specific chart object/element
            
        Returns:
            Formatted chart string with tags
        """
        try:
            chart_data = self.extract(chart_element)
            
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
        except Exception as e:
            return self._chart_processor.format_chart_fallback(
                chart_type="Unknown",
                message=f"Error extracting chart: {str(e)}"
            )


class NullChartExtractor(BaseChartExtractor):
    """
    Null implementation for handlers that don't support charts.
    
    Use this for file formats like PDF, CSV, TXT that don't contain charts.
    """
    
    def extract(self, chart_element: Any) -> ChartData:
        """Return empty chart data."""
        return ChartData(chart_type="Unsupported")
    
    def process(self, chart_element: Any) -> str:
        """Return empty string for unsupported formats."""
        return ""


__all__ = [
    'ChartData',
    'BaseChartExtractor',
    'NullChartExtractor',
]
