# contextifier/core/processor/hwp_helper/hwp_table_processor.py
"""
HWP Legacy (2.0/3.0) Table Processor - Table Output Formatting

Handles conversion of HWP legacy tables to various output formats (HTML, Markdown, Text).
Works with HWPLegacyTableExtractor to extract and format tables.

Processor Responsibilities:
- TableData → HTML conversion
- TableData → Markdown conversion
- TableData → Plain Text conversion

Usage:
    from contextifier.core.processor.hwp_helper.hwp_table_processor import (
        HWPLegacyTableProcessor,
    )

    processor = HWPLegacyTableProcessor()
    
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

logger = logging.getLogger("document-processor.HWP-Legacy")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HWPLegacyTableProcessorConfig(TableProcessorConfig):
    """Configuration specific to HWP legacy table processing.
    
    Attributes:
        use_th_for_headers: Whether to use <th> tags for header row
        include_border_attr: Whether to include border attribute in <table>
    """
    use_th_for_headers: bool = True
    include_border_attr: bool = True


# ============================================================================
# HWP Legacy Table Processor Class
# ============================================================================

class HWPLegacyTableProcessor:
    """HWP legacy (2.0/3.0) format-specific table processor.
    
    Handles conversion of HWP legacy tables to various output formats.
    Uses TableData from HWPLegacyTableExtractor.
    
    Usage:
        processor = HWPLegacyTableProcessor()
        html = processor.format_table(table_data)
    """
    
    def __init__(self, config: Optional[HWPLegacyTableProcessorConfig] = None):
        """Initialize HWP legacy table processor.
        
        Args:
            config: HWP legacy table processing configuration
        """
        self.config = config or HWPLegacyTableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor.HWP-Legacy")
    
    # ========================================================================
    # Conversion Utilities
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
    
    # ========================================================================
    # Main Public Methods
    # ========================================================================
    
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
        
        Args:
            table_data: TableData object or 2D list
            
        Returns:
            Formatted table string
        """
        table_data = self._to_table_data(table_data)
        if not table_data or not table_data.rows:
            return ""
        
        # For HWP legacy, always use HTML
        return self.format_table_as_html(table_data)
    
    def format_table_as_html(self, table_data: Union[TableData, List[List[str]]]) -> str:
        """Format TableData to HTML.
        
        Args:
            table_data: TableData object or 2D list
            
        Returns:
            HTML table string
        """
        table_data = self._to_table_data(table_data)
        if not table_data or not table_data.rows:
            return ""
        
        # Build HTML
        if self.config.include_border_attr:
            html_parts = ["<table border='1'>"]
        else:
            html_parts = ["<table>"]
        
        for row_idx, row in enumerate(table_data.rows):
            html_parts.append("  <tr>")
            
            for cell in row:
                content = cell.content if hasattr(cell, 'content') else str(cell)
                
                # Use <th> for header row if configured
                if row_idx == 0 and self.config.use_th_for_headers:
                    html_parts.append(f"    <th>{content}</th>")
                else:
                    html_parts.append(f"    <td>{content}</td>")
            
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        return '\n'.join(html_parts)
    
    def format_table_as_markdown(self, table_data: Union[TableData, List[List[str]]]) -> str:
        """Format TableData to Markdown.
        
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
            lines.append("\t".join(cells))
        
        return "\n".join(lines)


# ============================================================================
# Convenience Aliases
# ============================================================================

# Alias for backward compatibility
HWPTableProcessor = HWPLegacyTableProcessor
HWPTableProcessorConfig = HWPLegacyTableProcessorConfig


__all__ = [
    'HWPLegacyTableProcessor',
    'HWPLegacyTableProcessorConfig',
    # Aliases
    'HWPTableProcessor',
    'HWPTableProcessorConfig',
]
