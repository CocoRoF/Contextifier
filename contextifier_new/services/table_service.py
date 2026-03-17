# contextifier_new/services/table_service.py
"""
TableService — Table Data Formatting

Replaces old TableProcessor with a unified service that formats
TableData instances into HTML, Markdown, or plain text strings.

This service is format-agnostic — every handler's ContentExtractor
produces TableData, and this service renders it according to config.

Key improvement: The old code had TableProcessor as a concrete class
but also had ad-hoc table formatting in several handlers. Now ALL
table formatting goes through this one service.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from contextifier_new.config import ProcessingConfig, TableConfig
from contextifier_new.types import OutputFormat, TableCell, TableData


class TableService:
    """
    Formats TableData into string representations.

    Supports HTML, Markdown, and plain text output.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        self._config = config
        self._table_config: TableConfig = config.tables
        self._logger = logging.getLogger("contextifier.services.table")

    def format_table(self, table: TableData) -> str:
        """
        Format a table using the configured output format.

        Args:
            table: TableData instance.

        Returns:
            Formatted string (HTML, Markdown, or Text).
        """
        fmt = self._table_config.output_format
        if fmt == OutputFormat.HTML:
            return self.format_as_html(table)
        elif fmt == OutputFormat.MARKDOWN:
            return self.format_as_markdown(table)
        else:
            return self.format_as_text(table)

    def format_as_html(self, table: TableData) -> str:
        """Render table as HTML."""
        if not table.rows:
            return ""

        lines: List[str] = ["<table>"]

        for row_cells in table.rows:
            line_parts: List[str] = []
            for cell in row_cells:
                content = self._clean_cell(cell.content)
                tag = "th" if cell.is_header else "td"
                attrs = ""
                if cell.row_span > 1:
                    attrs += f' rowspan="{cell.row_span}"'
                if cell.col_span > 1:
                    attrs += f' colspan="{cell.col_span}"'
                line_parts.append(f"<{tag}{attrs}>{content}</{tag}>")
            lines.append(f"<tr>{''.join(line_parts)}</tr>")

        lines.append("</table>")
        return "\n".join(lines)

    def format_as_markdown(self, table: TableData) -> str:
        """Render table as Markdown pipe table."""
        if not table.rows:
            return ""

        lines: List[str] = []
        num_cols = table.num_cols or (
            max(len(row) for row in table.rows) if table.rows else 0
        )

        for i, row_cells in enumerate(table.rows):
            cells_text = []
            for cell in row_cells:
                content = self._clean_cell(cell.content).replace("|", "\\|")
                cells_text.append(content)
            # Pad to num_cols
            while len(cells_text) < num_cols:
                cells_text.append("")

            lines.append("| " + " | ".join(cells_text) + " |")

            # Add separator after first row (header)
            if i == 0:
                sep = "| " + " | ".join(["---"] * num_cols) + " |"
                lines.append(sep)

        return "\n".join(lines)

    def format_as_text(self, table: TableData) -> str:
        """Render table as plain text with tab separation."""
        if not table.rows:
            return ""

        lines: List[str] = []
        for row_cells in table.rows:
            cells_text = [self._clean_cell(cell.content) for cell in row_cells]
            lines.append("\t".join(cells_text))

        return "\n".join(lines)

    def _clean_cell(self, content: str) -> str:
        """Clean cell content."""
        if not content:
            return ""
        if self._table_config.clean_whitespace:
            content = " ".join(content.split())
        return content.strip()


__all__ = ["TableService"]
