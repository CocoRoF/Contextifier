# contextifier/core/processor/hwp5_helper/hwp5_table_processor.py
"""
HWP 5.0 Table Processor - Table Output Formatting

Handles conversion of HWP 5.0 tables to various output formats (HTML, Markdown, Text).
Works with HWP5TableExtractor to extract and format tables.

Processor Responsibilities:
- TableData → HTML conversion (with merge cell support)
- TableData → Markdown conversion (no merge support)
- TableData → Plain Text conversion
- Container table handling (1x1, single-column flattening)

Output Format Selection:
- Multi-column tables → HTML format (rowspan/colspan support)
- 1x1 tables → Plain text (container table flattening)
- Single-column tables → Newline-joined text

Usage:
    from contextifier.core.processor.hwp5_helper.hwp5_table_processor import (
        HWP5TableProcessor,
    )

    processor = HWP5TableProcessor()
    
    # From TableData
    html = processor.format_table(table_data)
    
    # Auto-select based on structure
    output = processor.format_table_auto(table_data)
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
    TableOutputFormat,
)
from contextifier.core.functions.table_extractor import TableData, TableCell

logger = logging.getLogger("document-processor.HWP5")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HWP5TableProcessorConfig(TableProcessorConfig):
    """Configuration specific to HWP 5.0 table processing.
    
    Attributes:
        flatten_container_tables: Whether to flatten 1x1 tables to plain text
        flatten_single_column: Whether to flatten single-column tables
        escape_newline_as_br: Whether to convert newlines to <br> in HTML
        preserve_whitespace: Whether to preserve whitespace in cells
    """
    flatten_container_tables: bool = True
    flatten_single_column: bool = True
    escape_newline_as_br: bool = True
    preserve_whitespace: bool = False


# ============================================================================
# HWP 5.0 Table Processor Class
# ============================================================================

class HWP5TableProcessor:
    """HWP 5.0 format-specific table processor.
    
    Handles conversion of HWP 5.0 tables to various output formats.
    Uses TableData from HWP5TableExtractor.
    
    HWP 5.0-Specific Features:
    - Container table flattening (1x1 tables → plain text)
    - Single-column table flattening (→ newline-joined text)
    - Cell merge support with rowspan/colspan
    
    Usage:
        processor = HWP5TableProcessor()
        html = processor.format_table(table_data)
    """
    
    def __init__(self, config: Optional[HWP5TableProcessorConfig] = None):
        """Initialize HWP 5.0 table processor.
        
        Args:
            config: HWP 5.0 table processing configuration
        """
        self.config = config or HWP5TableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor.HWP5")
    
    # ========================================================================
    # Main Public Methods
    # ========================================================================
    
    def _to_table_data(self, table_data: Union[TableData, List[List[str]]]) -> TableData:
        """Convert 2D list to TableData if necessary.
        
        Args:
            table_data: Either TableData or 2D list of strings
            
        Returns:
            TableData object
        """
        if isinstance(table_data, TableData):
            return table_data
        
        # Convert 2D list to TableData
        rows = []
        for row_data in table_data:
            row = [TableCell(content=str(cell)) for cell in row_data]
            rows.append(row)
        
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0
        
        return TableData(rows=rows, num_rows=num_rows, num_cols=num_cols)
    
    def format_table(self, table_data: Union[TableData, List[List[str]]]) -> str:
        """Format TableData to output string (HTML by default).
        
        Args:
            table_data: TableData object or 2D list of strings
            
        Returns:
            Formatted table string
        """
        table_data = self._to_table_data(table_data)
        if not table_data or not table_data.rows:
            return ""
        
        return self.format_table_as_html(table_data)
    
    def format_table_auto(self, table_data: Union[TableData, List[List[str]]]) -> str:
        """Format TableData with automatic format selection.
        
        Automatically chooses format based on table structure:
        - 1x1 table → Plain text (container table)
        - Single column → Newline-joined text
        - Multi-column → HTML
        
        Args:
            table_data: TableData object or 2D list
            
        Returns:
            Formatted table string
        """
        table_data = self._to_table_data(table_data)
        if not table_data or not table_data.rows:
            return ""
        
        # 1x1 table -> return cell content only
        if self.config.flatten_container_tables:
            if table_data.num_rows == 1 and table_data.num_cols == 1:
                if table_data.rows and table_data.rows[0]:
                    return table_data.rows[0][0].content
                return ""
        
        # Single column table -> join with newlines
        if self.config.flatten_single_column:
            if table_data.num_cols == 1:
                text_parts = []
                for row in table_data.rows:
                    if row and row[0].content:
                        text_parts.append(row[0].content)
                return "\n\n".join(text_parts)
        
        # Multi-column -> HTML
        return self.format_table_as_html(table_data)
    
    def format_table_as_html(self, table_data: Union[TableData, List[List[str]]]) -> str:
        """Format TableData to HTML.
        
        Supports merged cells with rowspan/colspan attributes.
        
        Args:
            table_data: TableData object or 2D list
            
        Returns:
            HTML table string
        """
        table_data = self._to_table_data(table_data)
        if not table_data or not table_data.rows:
            return ""
        
        html_parts = ["<table>"]
        skip_map = set()  # Track cells to skip due to spans
        
        for row_idx, row in enumerate(table_data.rows):
            html_parts.append("  <tr>")
            
            col_idx = 0
            for cell in row:
                # Skip cells covered by previous spans
                while (row_idx, col_idx) in skip_map:
                    col_idx += 1
                
                # Get span values
                row_span = getattr(cell, 'row_span', 1) or 1
                col_span = getattr(cell, 'col_span', 1) or 1
                
                # Skip merged-away cells (span = 0)
                if row_span == 0 or col_span == 0:
                    col_idx += 1
                    continue
                
                # Mark spanned cells
                for rs in range(row_span):
                    for cs in range(col_span):
                        if rs == 0 and cs == 0:
                            continue
                        skip_map.add((row_idx + rs, col_idx + cs))
                
                # Build attributes
                attrs = []
                if row_span > 1:
                    attrs.append(f"rowspan='{row_span}'")
                if col_span > 1:
                    attrs.append(f"colspan='{col_span}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                # Get content
                content = cell.content if hasattr(cell, 'content') else str(cell)
                content = self._escape_html(content)
                
                # Handle newlines
                if self.config.escape_newline_as_br:
                    content = content.replace('\n', '<br>')
                
                # Use th for header cells (first row)
                is_header = getattr(cell, 'is_header', row_idx == 0)
                tag = "th" if is_header else "td"
                
                html_parts.append(f"    <{tag}{attr_str}>{content}</{tag}>")
                col_idx += col_span
            
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        return "\n".join(html_parts)
    
    def format_table_as_markdown(self, table_data: Union[TableData, List[List[str]]]) -> str:
        """Format TableData to Markdown.
        
        Note: Markdown does not support merged cells.
        
        Args:
            table_data: TableData object or 2D list
            
        Returns:
            Markdown table string
        """
        table_data = self._to_table_data(table_data)
        if not table_data or not table_data.rows:
            return ""
        
        lines = []
        
        for row_idx, row in enumerate(table_data.rows):
            cells = []
            for cell in row:
                content = cell.content if hasattr(cell, 'content') else str(cell)
                # Escape pipe characters
                content = content.replace('|', '\\|').replace('\n', ' ')
                cells.append(content)
            
            line = "| " + " | ".join(cells) + " |"
            lines.append(line)
            
            # Add separator after header row
            if row_idx == 0:
                separator = "| " + " | ".join(['---'] * len(cells)) + " |"
                lines.append(separator)
        
        return "\n".join(lines)
    
    def format_table_as_text(self, table_data: Union[TableData, List[List[str]]]) -> str:
        """Format TableData to plain text.
        
        Args:
            table_data: TableData object or 2D list
            
        Returns:
            Plain text representation
        """
        table_data = self._to_table_data(table_data)
        if not table_data or not table_data.rows:
            return ""
        
        lines = []
        
        for row in table_data.rows:
            cells = []
            for cell in row:
                content = cell.content if hasattr(cell, 'content') else str(cell)
                cells.append(content)
            
            line = "\t".join(cells)
            lines.append(line)
        
        return "\n".join(lines)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.
        
        Args:
            text: Raw text
            
        Returns:
            HTML-escaped text
        """
        if not text:
            return ""
        
        # Don't escape if already contains HTML tags
        if '<table' in text.lower() or '<td' in text.lower():
            return text
        
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))


__all__ = [
    'HWP5TableProcessor',
    'HWP5TableProcessorConfig',
]
