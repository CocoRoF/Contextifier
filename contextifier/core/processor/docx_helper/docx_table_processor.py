# contextifier/core/processor/docx_helper/docx_table_processor.py
"""
DOCX Table Processor

Formats TableData into HTML/Markdown/Text output for DOCX documents.
Extends the base TableProcessor with DOCX-specific formatting options.

Key Features:
- HTML output with border attributes for backward compatibility
- Special handling for 1x1 container tables
- Special handling for single column tables
- Post-processing for DOCX-specific requirements

Usage:
    from contextifier.core.processor.docx_helper.docx_table_processor import (
        DOCXTableProcessor,
        create_docx_table_processor,
    )
    
    processor = DOCXTableProcessor()
    html = processor.format_table(table_data)
"""
import logging
from dataclasses import dataclass
from typing import Optional

from contextifier.core.functions.table_extractor import TableData
from contextifier.core.functions.table_processor import (
    TableProcessor,
    TableProcessorConfig,
    TableOutputFormat,
)

logger = logging.getLogger("document-processor")


@dataclass
class DOCXTableProcessorConfig(TableProcessorConfig):
    """Configuration for DOCX table processing.
    
    Extends TableProcessorConfig with DOCX-specific options.
    
    Attributes:
        add_border: Whether to add border='1' attribute to HTML tables
        collapse_single_cell: Whether to collapse 1x1 tables to plain text
        collapse_single_column: Whether to collapse single-column tables to line-separated text
    """
    add_border: bool = True
    collapse_single_cell: bool = True
    collapse_single_column: bool = True


class DOCXTableProcessor(TableProcessor):
    """DOCX-specific table processor.
    
    Extends TableProcessor with DOCX-specific formatting:
    - Adds border='1' to HTML tables for backward compatibility
    - Collapses 1x1 container tables to plain text
    - Collapses single-column tables to line-separated text
    
    Usage:
        processor = DOCXTableProcessor()
        html = processor.format_table(table_data)
    """
    
    def __init__(self, config: Optional[DOCXTableProcessorConfig] = None):
        """Initialize the DOCX table processor.
        
        Args:
            config: DOCX table processing configuration
        """
        if config is None:
            config = DOCXTableProcessorConfig()
        super().__init__(config)
        self.docx_config = config
    
    def format_table(self, table: TableData) -> str:
        """Format a table with DOCX-specific handling.
        
        Handles special cases before delegating to base class:
        - 1x1 tables: Return cell content only (container tables)
        - Single column tables: Return as line-separated text
        
        Args:
            table: TableData to format
            
        Returns:
            Formatted table string
        """
        if not table or not table.rows:
            return ""
        
        # Special case: 1x1 table (container table)
        if (self.docx_config.collapse_single_cell and 
            table.num_rows == 1 and table.num_cols == 1):
            if table.rows and table.rows[0]:
                return table.rows[0][0].content
            return ""
        
        # Special case: Single column table
        if (self.docx_config.collapse_single_column and 
            table.num_cols == 1):
            text_items = []
            for row in table.rows:
                if row and row[0].content:
                    text_items.append(row[0].content)
            if text_items:
                return "\n\n".join(text_items)
            return ""
        
        # Normal table processing
        return super().format_table(table)
    
    def format_table_as_html(self, table: TableData) -> str:
        """Format table as HTML with DOCX-specific attributes.
        
        Adds border='1' attribute for backward compatibility.
        
        Args:
            table: TableData to format
            
        Returns:
            HTML table string
        """
        # Check for special cases first
        if not table or not table.rows:
            return ""
        
        # 1x1 table handling
        if (self.docx_config.collapse_single_cell and 
            table.num_rows == 1 and table.num_cols == 1):
            if table.rows and table.rows[0]:
                return table.rows[0][0].content
            return ""
        
        # Single column table handling
        if (self.docx_config.collapse_single_column and 
            table.num_cols == 1):
            text_items = []
            for row in table.rows:
                if row and row[0].content:
                    text_items.append(row[0].content)
            if text_items:
                return "\n\n".join(text_items)
            return ""
        
        # Generate HTML using base class
        html = super().format_table_as_html(table)
        
        # Post-process: Add border attribute
        if self.docx_config.add_border:
            html = html.replace("<table>", "<table border='1'>")
        
        return html


# Default configuration
DEFAULT_DOCX_PROCESSOR_CONFIG = DOCXTableProcessorConfig(
    output_format=TableOutputFormat.HTML,
    clean_whitespace=True,
    preserve_merged_cells=True,
    add_border=True,
    collapse_single_cell=True,
    collapse_single_column=True,
)


# Module-level default processor (lazy initialized)
_default_processor: Optional[DOCXTableProcessor] = None


def get_default_processor() -> DOCXTableProcessor:
    """Get or create the default DOCX table processor.
    
    Returns:
        Configured DOCXTableProcessor instance
    """
    global _default_processor
    if _default_processor is None:
        _default_processor = DOCXTableProcessor(DEFAULT_DOCX_PROCESSOR_CONFIG)
    return _default_processor


def create_docx_table_processor(
    config: Optional[DOCXTableProcessorConfig] = None
) -> DOCXTableProcessor:
    """Create a DOCX table processor instance.
    
    Args:
        config: DOCX table processing configuration
        
    Returns:
        Configured DOCXTableProcessor instance
    """
    return DOCXTableProcessor(config)


def format_table_as_html(table: TableData) -> str:
    """Convenience function to format a table as HTML.
    
    Uses the default DOCX table processor.
    
    Args:
        table: TableData to format
        
    Returns:
        HTML table string
    """
    processor = get_default_processor()
    return processor.format_table_as_html(table)


__all__ = [
    'DOCXTableProcessor',
    'DOCXTableProcessorConfig',
    'DEFAULT_DOCX_PROCESSOR_CONFIG',
    'create_docx_table_processor',
    'get_default_processor',
    'format_table_as_html',
]
