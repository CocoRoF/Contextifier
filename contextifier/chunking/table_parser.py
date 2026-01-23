# chunking_helper/table_parser.py
"""
Table Parser - HTML table parsing functions

Main Features:
- HTML table parsing and structure analysis
- Cell span information extraction (rowspan, colspan)
- Table complexity analysis
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from contextifier.chunking.constants import ParsedTable, TableRow

logger = logging.getLogger("document-processor")


def parse_html_table(table_html: str) -> Optional[ParsedTable]:
    """
    Parse an HTML table and extract structured information.

    Args:
        table_html: HTML table string

    Returns:
        ParsedTable object or None (if parsing fails)
    """
    try:
        # Extract rows
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        row_matches = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

        if not row_matches:
            logger.debug("No rows found in table")
            return None

        header_rows: List[TableRow] = []
        data_rows: List[TableRow] = []
        max_cols = 0

        for row_content in row_matches:
            # Extract cells
            th_cells = re.findall(r'<th[^>]*>(.*?)</th>', row_content, re.DOTALL | re.IGNORECASE)
            td_cells = re.findall(r'<td[^>]*>(.*?)</td>', row_content, re.DOTALL | re.IGNORECASE)

            is_header = len(th_cells) > 0 and len(td_cells) == 0
            cell_count = len(th_cells) if is_header else len(td_cells)
            max_cols = max(max_cols, cell_count)

            # Reconstruct original row HTML
            row_html = f"<tr>{row_content}</tr>"
            row_length = len(row_html)

            table_row = TableRow(
                html=row_html,
                is_header=is_header,
                cell_count=cell_count,
                char_length=row_length
            )

            if is_header and not data_rows:
                # Header row before any data rows
                header_rows.append(table_row)
            else:
                data_rows.append(table_row)

        # Build header HTML
        if header_rows:
            header_html = "\n".join(row.html for row in header_rows)
            header_size = sum(row.char_length for row in header_rows) + len(header_rows)  # Including newlines
        else:
            header_html = ""
            header_size = 0

        return ParsedTable(
            header_rows=header_rows,
            data_rows=data_rows,
            total_cols=max_cols,
            original_html=table_html,
            header_html=header_html,
            header_size=header_size
        )

    except Exception as e:
        logger.warning(f"Failed to parse HTML table: {e}")
        return None


def extract_cell_spans(row_html: str) -> List[Tuple[int, int]]:
    """
    Extract rowspan/colspan information from cells in a row.

    Args:
        row_html: Row HTML

    Returns:
        [(rowspan, colspan), ...] list
    """
    spans = []

    # Find th and td cells
    cell_pattern = r'<(th|td)([^>]*)>'

    for match in re.finditer(cell_pattern, row_html, re.IGNORECASE):
        attrs = match.group(2)

        # Extract rowspan
        rowspan_match = re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        rowspan = int(rowspan_match.group(1)) if rowspan_match else 1

        # Extract colspan
        colspan_match = re.search(r'colspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        colspan = int(colspan_match.group(1)) if colspan_match else 1

        spans.append((rowspan, colspan))

    return spans


def extract_cell_spans_with_positions(row_html: str) -> Dict[int, int]:
    """
    Extract rowspan information by column position from a row (considering colspan).

    Args:
        row_html: Row HTML

    Returns:
        {column_position: rowspan} dictionary (only cells with rowspan > 1)
    """
    spans: Dict[int, int] = {}
    cell_pattern = r'<(th|td)([^>]*)>'

    current_col = 0
    for match in re.finditer(cell_pattern, row_html, re.IGNORECASE):
        attrs = match.group(2)

        # Extract rowspan
        rowspan_match = re.search(r'rowspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        rowspan = int(rowspan_match.group(1)) if rowspan_match else 1

        # Extract colspan
        colspan_match = re.search(r'colspan=["\']?(\d+)["\']?', attrs, re.IGNORECASE)
        colspan = int(colspan_match.group(1)) if colspan_match else 1

        if rowspan > 1:
            spans[current_col] = rowspan

        current_col += colspan

    return spans


def has_complex_spans(table_html: str) -> bool:
    """
    Check if a table has complex rowspan.
    (colspan does not affect row splitting, only rowspan is problematic)

    Args:
        table_html: Table HTML

    Returns:
        True if there are cells with rowspan > 1
    """
    rowspan_pattern = r'rowspan=["\']?(\d+)["\']?'
    matches = re.findall(rowspan_pattern, table_html, re.IGNORECASE)

    for val in matches:
        if int(val) > 1:
            return True

    return False
