# contextifier/services/table_service.py
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

import html as html_mod
import logging
from typing import List

from contextifier.config import ProcessingConfig, TableConfig
from contextifier.types import OutputFormat, TableData


def _flatten_table(table: "TableData") -> str:
    """
    Reduce a nested table to a single line of its values.

    Markdown and plain text cannot nest a table inside a cell, so the choice
    is between flattening and losing the sub-table. Rows are separated by
    ``;`` and cells by ``/``, which keeps the grouping readable without
    pretending to be structure.
    """
    rows: List[str] = []
    for row_cells in table.rows:
        values = [" ".join((cell.content or "").split()) for cell in row_cells]
        values = [v for v in values if v]
        if values:
            rows.append(" / ".join(values))
    return "; ".join(rows)


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
                content = html_mod.escape(self._clean_cell(cell.content), quote=False)
                if cell.nested_table is not None:
                    # Rendered, not escaped: a sub-table is markup, and the
                    # chunker's table scanner counts depth so a nested table
                    # stays inside its parent's protected region.
                    inner = self.format_as_html(cell.nested_table)
                    if inner:
                        content = f"{content}{inner}" if content else inner
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
                content = self._clean_cell(cell.content)
                if cell.nested_table is not None:
                    # Markdown has no nesting; the sub-table's values are
                    # flattened into the cell so none of them are lost.
                    flat = _flatten_table(cell.nested_table)
                    if flat:
                        content = f"{content} {flat}" if content else flat
                cells_text.append(content.replace("|", "\\|"))
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
            cells_text = []
            for cell in row_cells:
                content = self._clean_cell(cell.content)
                if cell.nested_table is not None:
                    flat = _flatten_table(cell.nested_table)
                    if flat:
                        content = f"{content} {flat}" if content else flat
                cells_text.append(content)
            lines.append("\t".join(cells_text))

        return "\n".join(lines)

    def _clean_cell(self, content: str) -> str:
        """Clean cell content."""
        if not content:
            return ""
        if self._table_config.clean_whitespace:
            content = " ".join(content.split())
        return content.strip()

    @staticmethod
    def format_as_html_simple(table_data: "TableData") -> str:
        """Fallback HTML table generation for use without a TableService instance.

        All content extractors that need a no-config fallback should call
        this method instead of maintaining their own ``_table_to_html``.
        """
        if not table_data.rows:
            return ""

        lines: List[str] = ["<table border='1'>"]
        for row in table_data.rows:
            lines.append("  <tr>")
            for cell in row:
                tag = "th" if cell.is_header else "td"
                attrs = ""
                if cell.row_span > 1:
                    attrs += f" rowspan='{cell.row_span}'"
                if cell.col_span > 1:
                    attrs += f" colspan='{cell.col_span}'"
                content = cell.content or ""
                content = html_mod.escape(content, quote=False)
                content = content.replace("\n", "<br>")
                lines.append(f"    <{tag}{attrs}>{content}</{tag}>")
            lines.append("  </tr>")
        lines.append("</table>")
        return "\n".join(lines)


__all__ = ["TableService"]
