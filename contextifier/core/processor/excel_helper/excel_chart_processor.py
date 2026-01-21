"""
Excel Chart Processing Module

Extracts chart data from Excel files and formats using ChartProcessor.
Output format:
    {chart_prefix}
    {chart_title}
    {chart_type}
    <table>...</table>
    {chart_suffix}
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from contextifier.core.functions.chart_processor import ChartProcessor

logger = logging.getLogger("document-processor")


def process_chart(
    chart_info: Dict[str, Any],
    chart_processor: "ChartProcessor"
) -> str:
    """
    Process a chart using ChartProcessor.

    Args:
        chart_info: Chart information dictionary containing:
            - chart_type: Type of chart (bar, line, pie, etc.)
            - title: Chart title (optional)
            - categories: List of category labels
            - series: List of series dicts with 'name' and 'values'
        chart_processor: ChartProcessor instance for formatting

    Returns:
        Formatted chart string with tags
    """
    if not chart_info:
        return chart_processor.format_chart_fallback(chart_type="Unknown")
    
    chart_type = chart_info.get('chart_type', 'Unknown')
    title = chart_info.get('title')
    categories = chart_info.get('categories', [])
    series_list = chart_info.get('series', [])
    
    # Check if we have valid data
    has_data = series_list and any(len(s.get('values', [])) > 0 for s in series_list)
    
    if has_data:
        result = chart_processor.format_chart_data(
            chart_type=chart_type,
            series_data=series_list,
            title=title,
            categories=categories
        )
        logger.debug(f"Chart '{title}' converted to table successfully")
        return result
    
    # Fallback: no data available
    return chart_processor.format_chart_fallback(chart_type=chart_type, title=title)
