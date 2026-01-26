# contextifier/core/processor/csv_helper/csv_table_processor.py
"""
CSV Table Processor - CSV Format-Specific Table Processing

Handles conversion of CSV tables to various output formats (HTML, Markdown, Text).
Works with CSVTableExtractor to extract and format tables.

Processor Responsibilities:
- TableData → HTML conversion (with merge cell support)
- TableData → Markdown conversion (no merge support)
- TableData → Plain Text conversion
- Direct rows → table string conversion (for backward compatibility)

Output Format Selection:
- Merged cells present → HTML format (rowspan/colspan support)
- No merged cells → Markdown format (simpler, more readable)

Usage:
    from contextifier.core.processor.csv_helper.csv_table_processor import (
        CSVTableProcessor,
    )

    processor = CSVTableProcessor()
    
    # From TableData
    html = processor.format_table(table_data)
    
    # Direct from rows (backward compatible)
    html = processor.format_rows(rows, has_header=True)
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
    TableOutputFormat,
)
from contextifier.core.functions.table_extractor import TableData, TableCell

logger = logging.getLogger("document-processor")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CSVTableProcessorConfig(TableProcessorConfig):
    """Configuration specific to CSV table processing.
    
    Attributes:
        auto_select_format: Whether to auto-select format based on merge cells
        prefer_markdown: Whether to prefer Markdown when no merge cells
        escape_pipe_in_markdown: Whether to escape | in Markdown output
        escape_newline_as_br: Whether to convert newlines to <br> in HTML
        escape_newline_as_space: Whether to convert newlines to space in Markdown
    """
    auto_select_format: bool = True
    prefer_markdown: bool = True
    escape_pipe_in_markdown: bool = True
    escape_newline_as_br: bool = True
    escape_newline_as_space: bool = True


# ============================================================================
# CSV Table Processor Class
# ============================================================================

class CSVTableProcessor:
    """CSV format-specific table processor.
    
    Handles conversion of CSV tables to various output formats.
    Uses TableData from CSVTableExtractor or processes rows directly.
    
    Features:
    - Auto-selects HTML vs Markdown based on merge cells
    - Handles merged cells with rowspan/colspan in HTML
    - Escapes special characters appropriately
    
    Usage:
        processor = CSVTableProcessor()
        html = processor.format_table(table_data)
        # or
        markdown = processor.format_rows(rows, has_header=True)
    """
    
    def __init__(self, config: Optional[CSVTableProcessorConfig] = None):
        """Initialize CSV table processor.
        
        Args:
            config: CSV table processing configuration
        """
        self.config = config or CSVTableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor")
    
    # ========================================================================
    # Main Public Methods - TableData Processing
    # ========================================================================
    
    def format_table(self, table_data: TableData) -> str:
        """Format TableData to output string.
        
        Automatically chooses format based on table structure:
        - Merged cells present: HTML
        - No merged cells: Markdown (if prefer_markdown) or HTML
        
        Args:
            table_data: TableData object from CSVTableExtractor
            
        Returns:
            Formatted table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        # Check if auto-select is enabled
        if self.config.auto_select_format:
            has_merges = self._has_merged_cells(table_data)
            if has_merges:
                return self.format_table_as_html(table_data)
            elif self.config.prefer_markdown:
                return self.format_table_as_markdown(table_data)
        
        # Default to configured format
        return self._table_processor.format_table(table_data)
    
    def format_table_as_html(self, table_data: TableData) -> str:
        """Format TableData to HTML.
        
        Supports merged cells with rowspan/colspan attributes.
        
        Args:
            table_data: TableData object
            
        Returns:
            HTML table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_html(table_data)
    
    def format_table_as_markdown(self, table_data: TableData) -> str:
        """Format TableData to Markdown.
        
        Note: Markdown doesn't support merged cells, they are flattened.
        
        Args:
            table_data: TableData object
            
        Returns:
            Markdown table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_markdown(table_data)
    
    def format_table_as_text(self, table_data: TableData) -> str:
        """Format TableData to plain text.
        
        Args:
            table_data: TableData object
            
        Returns:
            Tab-separated text representation
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self._format_table_to_text(table_data)
    
    # ========================================================================
    # Direct Row Processing (Backward Compatible)
    # ========================================================================
    
    def format_rows(self, rows: List[List[str]], has_header: bool = True) -> str:
        """Format parsed rows directly to table string.
        
        Convenience method for direct row formatting without going through
        CSVTableExtractor. Auto-selects HTML or Markdown based on merges.
        
        Args:
            rows: Parsed row data (2D list of strings)
            has_header: Whether first row is header
            
        Returns:
            Formatted table string
        """
        if not rows:
            return ""
        
        if self._has_merged_cells_in_rows(rows):
            self.logger.debug("Merged cells detected, using HTML format")
            return self._format_rows_to_html(rows, has_header)
        else:
            self.logger.debug("No merged cells, using Markdown format")
            return self._format_rows_to_markdown(rows, has_header)
    
    def format_rows_as_html(self, rows: List[List[str]], has_header: bool = True) -> str:
        """Format parsed rows to HTML.
        
        Args:
            rows: Parsed row data
            has_header: Whether first row is header
            
        Returns:
            HTML table string
        """
        if not rows:
            return ""
        return self._format_rows_to_html(rows, has_header)
    
    def format_rows_as_markdown(self, rows: List[List[str]], has_header: bool = True) -> str:
        """Format parsed rows to Markdown.
        
        Args:
            rows: Parsed row data
            has_header: Whether first row is header (affects separator)
            
        Returns:
            Markdown table string
        """
        if not rows:
            return ""
        return self._format_rows_to_markdown(rows, has_header)
    
    # ========================================================================
    # Private Helper Methods - TableData Formatting
    # ========================================================================
    
    def _has_merged_cells(self, table_data: TableData) -> bool:
        """Check if TableData has merged cells."""
        for row in table_data.rows:
            for cell in row:
                if cell.row_span > 1 or cell.col_span > 1:
                    return True
        return False
    
    def _format_table_to_html(self, table_data: TableData) -> str:
        """Format TableData to HTML with merge support."""
        html_parts = ["<table border='1'>"]
        
        for row_idx, row in enumerate(table_data.rows):
            html_parts.append("<tr>")
            
            for cell in row:
                # Skip cells that are part of a merge (span = 0)
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                
                # Determine tag
                tag = "th" if cell.is_header else "td"
                
                # Build attributes
                attrs = []
                if cell.col_span > 1:
                    attrs.append(f"colspan='{cell.col_span}'")
                if cell.row_span > 1:
                    attrs.append(f"rowspan='{cell.row_span}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                # Escape and format content
                content = self._escape_html(cell.content)
                
                html_parts.append(f"<{tag}{attr_str}>{content}</{tag}>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    def _format_table_to_markdown(self, table_data: TableData) -> str:
        """Format TableData to Markdown (no merge support)."""
        md_parts = []
        row_count = 0
        
        for row in table_data.rows:
            cells = []
            for cell in row:
                # Skip cells that are part of a merge
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                
                content = cell.content
                if self.config.escape_pipe_in_markdown:
                    content = content.replace("|", "\\|")
                if self.config.escape_newline_as_space:
                    content = content.replace("\n", " ")
                cells.append(content)
            
            if not cells:
                continue
            
            row_str = "| " + " | ".join(cells) + " |"
            md_parts.append(row_str)
            row_count += 1
            
            # Add separator after first row (header)
            if row_count == 1:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                md_parts.append(separator)
        
        return "\n".join(md_parts)
    
    def _format_table_to_text(self, table_data: TableData) -> str:
        """Format TableData to plain text."""
        lines = []
        
        for row in table_data.rows:
            cells = []
            for cell in row:
                if cell.col_span == 0 or cell.row_span == 0:
                    continue
                content = cell.content.replace("\n", " ").replace("\t", " ")
                cells.append(content)
            
            if cells:
                lines.append("\t".join(cells))
        
        return "\n".join(lines)
    
    # ========================================================================
    # Private Helper Methods - Direct Row Formatting
    # ========================================================================
    
    def _has_merged_cells_in_rows(self, rows: List[List[str]]) -> bool:
        """Check if rows have merged cell patterns."""
        if not rows or len(rows) < 2:
            return False
        
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell_value = cell.strip() if cell else ""
                
                if not cell_value:
                    if row_idx > 0 and col_idx == 0:
                        return True
                    if col_idx > 0:
                        prev_cell = row[col_idx - 1].strip() if col_idx - 1 < len(row) else ""
                        if prev_cell:
                            return True
        
        return False
    
    def _analyze_row_merge_info(self, rows: List[List[str]]) -> List[List[Dict[str, Any]]]:
        """Analyze merge information from rows."""
        if not rows:
            return []
        
        row_count = len(rows)
        col_count = max(len(row) for row in rows) if rows else 0
        
        # Initialize
        merge_info = []
        for row_idx, row in enumerate(rows):
            row_info = []
            for col_idx in range(col_count):
                cell_value = row[col_idx].strip() if col_idx < len(row) else ""
                row_info.append({
                    'value': cell_value,
                    'colspan': 1,
                    'rowspan': 1,
                    'skip': False,
                })
            merge_info.append(row_info)
        
        # Pass 1: colspan
        for row_idx in range(row_count):
            col_idx = 0
            while col_idx < col_count:
                cell_info = merge_info[row_idx][col_idx]
                if cell_info['skip'] or not cell_info['value']:
                    col_idx += 1
                    continue
                
                colspan = 1
                next_col = col_idx + 1
                while next_col < col_count:
                    next_cell = merge_info[row_idx][next_col]
                    if not next_cell['value'] and not next_cell['skip']:
                        colspan += 1
                        next_cell['skip'] = True
                        next_col += 1
                    else:
                        break
                
                cell_info['colspan'] = colspan
                col_idx = next_col
        
        # Pass 2: rowspan
        for col_idx in range(col_count):
            row_idx = 0
            while row_idx < row_count:
                cell_info = merge_info[row_idx][col_idx]
                if cell_info['skip'] or not cell_info['value']:
                    row_idx += 1
                    continue
                
                rowspan = 1
                next_row = row_idx + 1
                while next_row < row_count:
                    next_cell = merge_info[next_row][col_idx]
                    if not next_cell['value'] and not next_cell['skip']:
                        rowspan += 1
                        next_cell['skip'] = True
                        next_row += 1
                    else:
                        break
                
                cell_info['rowspan'] = rowspan
                row_idx = next_row
        
        return merge_info
    
    def _format_rows_to_html(self, rows: List[List[str]], has_header: bool) -> str:
        """Format rows to HTML with merge analysis."""
        if not rows:
            return ""
        
        merge_info = self._analyze_row_merge_info(rows)
        
        html_parts = ["<table border='1'>"]
        
        for row_idx, row_info in enumerate(merge_info):
            html_parts.append("<tr>")
            
            for cell_info in row_info:
                if cell_info['skip']:
                    continue
                
                cell_value = cell_info['value']
                cell_value = self._escape_html(cell_value)
                
                tag = "th" if (has_header and row_idx == 0) else "td"
                
                attrs = []
                if cell_info['colspan'] > 1:
                    attrs.append(f"colspan='{cell_info['colspan']}'")
                if cell_info['rowspan'] > 1:
                    attrs.append(f"rowspan='{cell_info['rowspan']}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                html_parts.append(f"<{tag}{attr_str}>{cell_value}</{tag}>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
    def _format_rows_to_markdown(self, rows: List[List[str]], _has_header: bool) -> str:
        """Format rows to Markdown."""
        if not rows:
            return ""
        
        md_parts = []
        
        for row_idx, row in enumerate(rows):
            cells = []
            for cell in row:
                cell_value = cell.strip() if cell else ""
                cell_value = cell_value.replace("|", "\\|")
                cell_value = cell_value.replace("\n", " ")
                cells.append(cell_value)
            
            row_str = "| " + " | ".join(cells) + " |"
            md_parts.append(row_str)
            
            # Header separator after first row
            if row_idx == 0:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                md_parts.append(separator)
        
        return "\n".join(md_parts)
    
    # ========================================================================
    # Private Helper Methods - Utility
    # ========================================================================
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        
        if self.config.escape_newline_as_br:
            text = text.replace("\n", "<br>")
        
        return text


# ============================================================================
# Backward Compatibility Functions
# ============================================================================

def convert_rows_to_table(rows: List[List[str]], has_header: bool) -> str:
    """Convert CSV rows to table string.
    
    Backward compatible function.
    
    Args:
        rows: Parsed row data
        has_header: Whether first row is header
        
    Returns:
        Table string (HTML or Markdown)
    """
    processor = CSVTableProcessor()
    return processor.format_rows(rows, has_header)


def convert_rows_to_markdown(rows: List[List[str]], has_header: bool) -> str:
    """Convert CSV rows to Markdown.
    
    Backward compatible function.
    
    Args:
        rows: Parsed row data
        has_header: Whether first row is header
        
    Returns:
        Markdown table string
    """
    processor = CSVTableProcessor()
    return processor.format_rows_as_markdown(rows, has_header)


def convert_rows_to_html(rows: List[List[str]], has_header: bool) -> str:
    """Convert CSV rows to HTML.
    
    Backward compatible function.
    
    Args:
        rows: Parsed row data
        has_header: Whether first row is header
        
    Returns:
        HTML table string
    """
    processor = CSVTableProcessor()
    return processor.format_rows_as_html(rows, has_header)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    "CSVTableProcessorConfig",
    # Processor
    "CSVTableProcessor",
    # Backward compatibility
    "convert_rows_to_table",
    "convert_rows_to_markdown",
    "convert_rows_to_html",
]
