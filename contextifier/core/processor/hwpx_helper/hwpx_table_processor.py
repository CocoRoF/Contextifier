# contextifier/core/processor/hwpx_helper/hwpx_table_processor.py
"""
HWPX Table Processor - HWPX Format-Specific Table Processing

Handles conversion of HWPX tables to various output formats (HTML, Markdown, Text).
Works with HWPXTableExtractor to extract and format tables.

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
    from contextifier.core.processor.hwpx_helper.hwpx_table_processor import (
        HWPXTableProcessor,
    )

    processor = HWPXTableProcessor()
    
    # From TableData
    html = processor.format_table(table_data)
    
    # Auto-select based on structure
    output = processor.format_table_auto(table_data)
"""
import logging
from dataclasses import dataclass
from typing import Optional

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
class HWPXTableProcessorConfig(TableProcessorConfig):
    """Configuration specific to HWPX table processing.
    
    Attributes:
        flatten_container_tables: Whether to flatten 1x1 tables to plain text
        flatten_single_column: Whether to flatten single-column tables
        use_html_for_nested: Whether to use HTML for nested table content
        escape_newline_as_br: Whether to convert newlines to <br> in HTML
    """
    flatten_container_tables: bool = True
    flatten_single_column: bool = True
    use_html_for_nested: bool = True
    escape_newline_as_br: bool = True


# ============================================================================
# HWPX Table Processor Class
# ============================================================================

class HWPXTableProcessor:
    """HWPX format-specific table processor.
    
    Handles conversion of HWPX tables to various output formats.
    Uses TableData from HWPXTableExtractor.
    
    HWPX-Specific Features:
    - Container table flattening (1x1 tables → plain text)
    - Single-column table flattening (→ newline-joined text)
    - Nested table HTML rendering
    
    Usage:
        processor = HWPXTableProcessor()
        html = processor.format_table(table_data)
    """
    
    def __init__(self, config: Optional[HWPXTableProcessorConfig] = None):
        """Initialize HWPX table processor.
        
        Args:
            config: HWPX table processing configuration
        """
        self.config = config or HWPXTableProcessorConfig()
        self._table_processor = TableProcessor(self.config)
        self.logger = logging.getLogger("document-processor")
    
    # ========================================================================
    # Main Public Methods
    # ========================================================================
    
    def format_table(self, table_data: TableData) -> str:
        """Format TableData to output string (HTML by default).
        
        Args:
            table_data: TableData object from HWPXTableExtractor
            
        Returns:
            Formatted table string
        """
        if not table_data or not table_data.rows:
            return ""
        
        return self.format_table_as_html(table_data)
    
    def format_table_auto(self, table_data: TableData) -> str:
        """Format TableData with automatic format selection.
        
        Automatically chooses format based on table structure:
        - 1x1 table → Plain text (container table)
        - Single column → Newline-joined text
        - Multi-column → HTML
        
        Args:
            table_data: TableData object
            
        Returns:
            Formatted table string
        """
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
        
        html_parts = ["<table border='1'>"]
        
        for row in table_data.rows:
            html_parts.append("<tr>")
            
            for cell in row:
                # Skip merged-away cells (span = 0)
                if cell.row_span == 0 or cell.col_span == 0:
                    continue
                
                # Build attributes
                attrs = []
                if cell.row_span > 1:
                    attrs.append(f"rowspan='{cell.row_span}'")
                if cell.col_span > 1:
                    attrs.append(f"colspan='{cell.col_span}'")
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                # Handle nested tables
                content = cell.content
                if cell.nested_table is not None and self.config.use_html_for_nested:
                    nested_html = self.format_table_as_html(cell.nested_table)
                    if nested_html and nested_html not in content:
                        content = content + nested_html if content else nested_html
                
                # Escape content
                content = self._escape_html(content)
                
                # Use th for header cells
                tag = "th" if cell.is_header else "td"
                html_parts.append(f"<{tag}{attr_str}>{content}</{tag}>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        
        return "\n".join(html_parts)
    
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
        
        md_parts = []
        row_count = 0
        
        for row in table_data.rows:
            cells = []
            for cell in row:
                # Skip merged-away cells
                if cell.row_span == 0 or cell.col_span == 0:
                    continue
                
                content = cell.content
                # Escape pipe for Markdown
                content = content.replace("|", "\\|")
                # Replace newlines with space
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
    
    def format_table_as_text(self, table_data: TableData) -> str:
        """Format TableData to plain text.
        
        Args:
            table_data: TableData object
            
        Returns:
            Tab-separated text representation
        """
        if not table_data or not table_data.rows:
            return ""
        
        lines = []
        
        for row in table_data.rows:
            cells = []
            for cell in row:
                # Skip merged-away cells
                if cell.row_span == 0 or cell.col_span == 0:
                    continue
                content = cell.content.replace("\n", " ").replace("\t", " ")
                cells.append(content)
            
            if cells:
                lines.append("\t".join(cells))
        
        return "\n".join(lines)
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        
        # Don't double-escape if already contains HTML tags
        if "<table" in text.lower() or "<td" in text.lower():
            # Already contains HTML, just escape newlines if needed
            if self.config.escape_newline_as_br:
                # But not inside existing tags
                pass
            return text
        
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        
        if self.config.escape_newline_as_br:
            text = text.replace("\n", "<br>")
        
        return text
    
    def _has_merged_cells(self, table_data: TableData) -> bool:
        """Check if TableData has merged cells."""
        for row in table_data.rows:
            for cell in row:
                if cell.row_span > 1 or cell.col_span > 1:
                    return True
        return False


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Configuration
    "HWPXTableProcessorConfig",
    # Processor
    "HWPXTableProcessor",
]
