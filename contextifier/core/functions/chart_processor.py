# libs/core/functions/chart_processor.py
"""
Chart Processor Module

Provides functionality for generating and formatting chart content in extracted text.
This module standardizes chart tag format across all document handlers.

=== Architecture Overview ===

1. Creation:
   - ChartProcessor instance is created when DocumentProcessor is initialized.
   - Created via DocumentProcessor.__init__() calling _create_chart_processor() method.

2. Propagation:
   - The created ChartProcessor is passed to ALL handlers.
   - In DocumentProcessor._get_handler_registry(), each handler is created with
     chart_processor=self._chart_processor parameter.

3. Access from Handlers:
   - Each Handler inherits from BaseHandler and can access via self.chart_processor.
   - Use format_chart_data() to convert chart data to standardized format.

4. Output Format:
   {chart_tag_prefix}
   Chart Type: {type}
   <table>...</table>
   {chart_tag_suffix}

=== Usage Examples ===

    # Custom settings at DocumentProcessor level
    from contextifier.core.document_processor import DocumentProcessor
    
    processor = DocumentProcessor(
        chart_tag_prefix="<chart>",
        chart_tag_suffix="</chart>"
    )
    
    # Usage inside Handler (BaseHandler subclass)
    class MyHandler(BaseHandler):
        def extract_text(self, ...):
            chart_content = self.chart_processor.format_chart_data(
                chart_type="Bar Chart",
                title="Sales Report",
                categories=["Q1", "Q2", "Q3"],
                series=[{"name": "Revenue", "values": [100, 150, 200]}]
            )

=== Default Tag Format ===

    [chart]
    Chart Type: Bar Chart
    Title: Sales Report
    <table border='1'>
    <tr><th>Category</th><th>Revenue</th></tr>
    <tr><td>Q1</td><td>100</td></tr>
    ...
    </table>
    [/chart]

"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger("document-processor")


# Chart type mapping (OOXML chart type names to human-readable names)
CHART_TYPE_MAP = {
    # Bar/Column charts
    'barChart': 'Bar Chart',
    'bar3DChart': '3D Bar Chart',
    'colChart': 'Column Chart',
    'col3DChart': '3D Column Chart',
    
    # Line charts
    'lineChart': 'Line Chart',
    'line3DChart': '3D Line Chart',
    'stockChart': 'Stock Chart',
    
    # Pie charts
    'pieChart': 'Pie Chart',
    'pie3DChart': '3D Pie Chart',
    'doughnutChart': 'Doughnut Chart',
    'ofPieChart': 'Pie of Pie Chart',
    
    # Area charts
    'areaChart': 'Area Chart',
    'area3DChart': '3D Area Chart',
    
    # Scatter/Bubble charts
    'scatterChart': 'Scatter Chart',
    'bubbleChart': 'Bubble Chart',
    
    # Radar charts
    'radarChart': 'Radar Chart',
    
    # Surface charts
    'surfaceChart': 'Surface Chart',
    'surface3DChart': '3D Surface Chart',
    
    # Combo/Other
    'comboChart': 'Combo Chart',
    'unknownChart': 'Chart',
}


@dataclass
class ChartProcessorConfig:
    """
    ChartProcessor configuration.

    Attributes:
        tag_prefix: Chart tag prefix (e.g., "[chart]")
        tag_suffix: Chart tag suffix (e.g., "[/chart]")
        use_html_table: Whether to use HTML table format (True) or Markdown (False)
        include_type: Whether to include chart type in output
        include_title: Whether to include chart title in output
    """
    tag_prefix: str = "[chart]"
    tag_suffix: str = "[/chart]"
    use_html_table: bool = True
    include_type: bool = True
    include_title: bool = True


class ChartProcessor:
    """
    Chart Processor Class

    Generates and formats chart content for document text extraction.
    Provides a standardized interface for all document handlers.

    Args:
        tag_prefix: Chart tag prefix (default: "[chart]")
        tag_suffix: Chart tag suffix (default: "[/chart]")
        use_html_table: Use HTML table format (default: True)
        config: ChartProcessorConfig instance (overrides individual parameters)

    Examples:
        >>> processor = ChartProcessor()
        >>> content = processor.format_chart_data(
        ...     chart_type="Bar Chart",
        ...     title="Sales",
        ...     categories=["Q1", "Q2"],
        ...     series=[{"name": "Revenue", "values": [100, 200]}]
        ... )
        '[chart]\\nChart Type: Bar Chart\\nTitle: Sales\\n<table>...</table>\\n[/chart]'
    """

    def __init__(
        self,
        tag_prefix: Optional[str] = None,
        tag_suffix: Optional[str] = None,
        use_html_table: Optional[bool] = None,
        config: Optional[ChartProcessorConfig] = None
    ):
        """Initialize ChartProcessor with configuration."""
        if config is not None:
            self._config = config
        else:
            self._config = ChartProcessorConfig(
                tag_prefix=tag_prefix if tag_prefix is not None else ChartProcessorConfig.tag_prefix,
                tag_suffix=tag_suffix if tag_suffix is not None else ChartProcessorConfig.tag_suffix,
                use_html_table=use_html_table if use_html_table is not None else ChartProcessorConfig.use_html_table,
            )

        # Pre-compile regex pattern for parsing
        self._chart_pattern: Optional[Pattern] = None

    @property
    def config(self) -> ChartProcessorConfig:
        """Current configuration."""
        return self._config

    @property
    def tag_prefix(self) -> str:
        """Chart tag prefix."""
        return self._config.tag_prefix

    @property
    def tag_suffix(self) -> str:
        """Chart tag suffix."""
        return self._config.tag_suffix

    @property
    def chart_pattern(self) -> Pattern:
        """Compiled regex pattern for matching chart blocks."""
        if self._chart_pattern is None:
            escaped_prefix = re.escape(self._config.tag_prefix)
            escaped_suffix = re.escape(self._config.tag_suffix)
            self._chart_pattern = re.compile(
                f'{escaped_prefix}(.*?){escaped_suffix}',
                re.DOTALL | re.IGNORECASE
            )
        return self._chart_pattern

    def get_pattern_string(self) -> str:
        """
        Get regex pattern string for matching chart blocks.

        Returns:
            Regex pattern string for matching chart blocks
        """
        escaped_prefix = re.escape(self._config.tag_prefix)
        escaped_suffix = re.escape(self._config.tag_suffix)
        return f'{escaped_prefix}.*?{escaped_suffix}'

    def get_chart_type_name(self, ooxml_type: str) -> str:
        """
        Convert OOXML chart type to human-readable name.

        Args:
            ooxml_type: OOXML chart type (e.g., 'barChart', 'pieChart')

        Returns:
            Human-readable chart type name
        """
        return CHART_TYPE_MAP.get(ooxml_type, ooxml_type or 'Chart')

    def format_chart_data(
        self,
        chart_type: Optional[str] = None,
        title: Optional[str] = None,
        categories: Optional[List[Any]] = None,
        series: Optional[List[Dict[str, Any]]] = None,
        raw_content: Optional[str] = None
    ) -> str:
        """
        Format chart data into standardized output format.

        Creates a formatted chart block with the configured tags, containing:
        - Chart type (if available)
        - Chart title (if available)
        - Data table in HTML format

        Args:
            chart_type: Chart type name (e.g., "Bar Chart", "Pie Chart")
            title: Chart title
            categories: List of category labels (x-axis values)
            series: List of series data, each containing:
                - 'name': Series name
                - 'values': List of values
            raw_content: Raw content to include (if no structured data)

        Returns:
            Formatted chart block string

        Example:
            >>> processor = ChartProcessor()
            >>> result = processor.format_chart_data(
            ...     chart_type="Bar Chart",
            ...     title="Quarterly Sales",
            ...     categories=["Q1", "Q2", "Q3", "Q4"],
            ...     series=[
            ...         {"name": "Product A", "values": [100, 150, 200, 180]},
            ...         {"name": "Product B", "values": [80, 120, 160, 140]}
            ...     ]
            ... )
        """
        parts = [self._config.tag_prefix]

        # Add chart type
        if chart_type and self._config.include_type:
            parts.append(f"Chart Type: {chart_type}")

        # Add title
        if title and self._config.include_title:
            parts.append(f"Title: {title}")

        # Add data table or raw content
        if series and any(s.get('values') for s in series):
            table = self._build_data_table(categories, series)
            if table:
                parts.append("")  # Empty line before table
                parts.append(table)
        elif raw_content:
            parts.append("")
            parts.append(raw_content)

        parts.append(self._config.tag_suffix)
        return "\n".join(parts)

    def format_chart_fallback(
        self,
        chart_type: Optional[str] = None,
        title: Optional[str] = None,
        message: Optional[str] = None
    ) -> str:
        """
        Format a fallback chart block when data extraction fails.

        Args:
            chart_type: Chart type name
            title: Chart title
            message: Optional message about the chart

        Returns:
            Minimal chart block string
        """
        parts = [self._config.tag_prefix]

        if chart_type:
            parts.append(f"Chart Type: {chart_type}")
        if title:
            parts.append(f"Title: {title}")
        if message:
            parts.append(message)
        elif not chart_type and not title:
            parts.append("(Chart content could not be extracted)")

        parts.append(self._config.tag_suffix)
        return "\n".join(parts)

    def _build_data_table(
        self,
        categories: Optional[List[Any]],
        series: List[Dict[str, Any]]
    ) -> str:
        """
        Build an HTML table from chart data.

        Args:
            categories: Category labels
            series: Series data list

        Returns:
            HTML table string
        """
        if not series:
            return ""

        categories = categories or []

        if self._config.use_html_table:
            return self._build_html_table(categories, series)
        else:
            return self._build_markdown_table(categories, series)

    def _build_html_table(
        self,
        categories: List[Any],
        series: List[Dict[str, Any]]
    ) -> str:
        """Build HTML table from chart data."""
        rows = []
        rows.append("<table border='1'>")

        # Header row
        header_cells = ["<th>Category</th>"]
        for i, s in enumerate(series):
            name = s.get('name') or f"Series {i+1}"
            header_cells.append(f"<th>{self._escape_html(str(name))}</th>")
        rows.append(f"<tr>{''.join(header_cells)}</tr>")

        # Data rows
        max_len = max(
            len(categories),
            max((len(s.get('values', [])) for s in series), default=0)
        )

        for i in range(max_len):
            cells = []

            # Category cell
            if i < len(categories):
                cat = self._escape_html(str(categories[i]))
            else:
                cat = f"Item {i+1}"
            cells.append(f"<td>{cat}</td>")

            # Value cells
            for s in series:
                values = s.get('values', [])
                if i < len(values) and values[i] is not None:
                    val = values[i]
                    if isinstance(val, float):
                        formatted = f"{val:,.2f}"
                    else:
                        formatted = self._escape_html(str(val))
                    cells.append(f"<td>{formatted}</td>")
                else:
                    cells.append("<td></td>")

            rows.append(f"<tr>{''.join(cells)}</tr>")

        rows.append("</table>")
        return "\n".join(rows)

    def _build_markdown_table(
        self,
        categories: List[Any],
        series: List[Dict[str, Any]]
    ) -> str:
        """Build Markdown table from chart data."""
        rows = []

        # Header row
        header = ["Category"] + [s.get('name', f'Series {i+1}') for i, s in enumerate(series)]
        rows.append("| " + " | ".join(str(h) for h in header) + " |")
        rows.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Data rows
        max_len = max(
            len(categories),
            max((len(s.get('values', [])) for s in series), default=0)
        )

        for i in range(max_len):
            row = []

            # Category
            if i < len(categories):
                row.append(str(categories[i]))
            else:
                row.append(f"Item {i+1}")

            # Values
            for s in series:
                values = s.get('values', [])
                if i < len(values) and values[i] is not None:
                    val = values[i]
                    if isinstance(val, float):
                        row.append(f"{val:,.2f}")
                    else:
                        row.append(str(val))
                else:
                    row.append("")

            rows.append("| " + " | ".join(row) + " |")

        return "\n".join(rows)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def has_chart_blocks(self, text: str) -> bool:
        """
        Check if text contains chart blocks.

        Args:
            text: Text to check

        Returns:
            True if chart blocks found
        """
        return bool(self.chart_pattern.search(text))

    def find_chart_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find all chart blocks in text.

        Args:
            text: Text to search

        Returns:
            List of tuples: (start_pos, end_pos, content)
        """
        results = []
        for match in self.chart_pattern.finditer(text):
            results.append((match.start(), match.end(), match.group(1)))
        return results

    def remove_chart_blocks(self, text: str) -> str:
        """
        Remove all chart blocks from text.

        Args:
            text: Text with chart blocks

        Returns:
            Text with chart blocks removed
        """
        return self.chart_pattern.sub('', text)

    def __repr__(self) -> str:
        return (
            f"ChartProcessor(tag_prefix={self._config.tag_prefix!r}, "
            f"tag_suffix={self._config.tag_suffix!r})"
        )


# Default instance for convenience
_default_processor: Optional[ChartProcessor] = None


def get_default_chart_processor() -> ChartProcessor:
    """Get the default ChartProcessor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = ChartProcessor()
    return _default_processor


def create_chart_processor(
    tag_prefix: Optional[str] = None,
    tag_suffix: Optional[str] = None,
    use_html_table: bool = True
) -> ChartProcessor:
    """
    Factory function to create a ChartProcessor instance.

    Args:
        tag_prefix: Chart tag prefix (default: "[chart]")
        tag_suffix: Chart tag suffix (default: "[/chart]")
        use_html_table: Use HTML table format (default: True)

    Returns:
        ChartProcessor instance
    """
    return ChartProcessor(
        tag_prefix=tag_prefix,
        tag_suffix=tag_suffix,
        use_html_table=use_html_table
    )


__all__ = [
    "ChartProcessorConfig",
    "ChartProcessor",
    "CHART_TYPE_MAP",
    "get_default_chart_processor",
    "create_chart_processor",
]
