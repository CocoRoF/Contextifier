# contextifier_new/handlers/csv/content_extractor.py
"""
CsvContentExtractor — Stage 4: Parsed CSV → Table content

Takes PreprocessedData (containing CsvParsedData) and produces:
- extract_text(): Formatted table string (HTML, Markdown, or text
  depending on TableConfig.output_format)
- extract_tables(): Structured TableData list for programmatic access

Merged cell detection:
  CSV files exported from spreadsheets may have empty cells that
  represent merged regions. The extractor detects these patterns
  and assigns proper colspan/rowspan to TableCell objects, which
  the TableService then renders correctly in HTML mode.

v1.0 logic ported:
- has_merged_cells: empty cell adjacency pattern detection
- analyze_merge_info: colspan/rowspan calculation from empty cells
- convert_rows_to_table: auto-select markdown/html based on merges

v2 improvements:
- Produces standardized TableData instead of raw strings
- Output format is config-driven (not hard-wired to merged-cell detection)
- TableService handles rendering, keeping the extractor focused on structure
- Merged cell info is encoded in TableCell.row_span/col_span attributes
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from contextifier_new.pipeline.content_extractor import BaseContentExtractor
from contextifier_new.types import PreprocessedData, TableCell, TableData
from contextifier_new.handlers.csv.preprocessor import CsvParsedData

if TYPE_CHECKING:
    from contextifier_new.services.table_service import TableService


class CsvContentExtractor(BaseContentExtractor):
    """
    Content extractor for CSV and TSV files.

    Converts parsed row data into a formatted table string and
    structured TableData. No images or charts for CSV files.
    """

    def extract_text(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> str:
        """
        Format CSV data as a table string.

        Uses TableService to render the table according to
        the configured output format (HTML, Markdown, or text).

        Args:
            preprocessed: Output from CsvPreprocessor.
            **kwargs: Ignored.

        Returns:
            Formatted table string.
        """
        parsed = self._get_parsed_data(preprocessed)
        if parsed is None or not parsed.rows:
            return ""

        table_data = _build_table_data(parsed.rows, parsed.has_header)

        if self._table_service is not None:
            return self._table_service.format_table(table_data)

        # Fallback: tab-separated text if no service available
        self._logger.warning("No TableService — falling back to plain text")
        return _fallback_format(parsed.rows)

    def extract_tables(
        self,
        preprocessed: PreprocessedData,
        **kwargs: Any,
    ) -> List[TableData]:
        """
        Return the CSV data as a structured TableData.

        This provides programmatic access to the table structure
        (cells, headers, spans) independent of text formatting.

        Args:
            preprocessed: Output from CsvPreprocessor.
            **kwargs: Ignored.

        Returns:
            List containing one TableData (the entire CSV).
        """
        parsed = self._get_parsed_data(preprocessed)
        if parsed is None or not parsed.rows:
            return []

        table_data = _build_table_data(parsed.rows, parsed.has_header)
        return [table_data]

    def get_format_name(self) -> str:
        return "csv"

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_parsed_data(preprocessed: PreprocessedData) -> Optional[CsvParsedData]:
        """Safely extract CsvParsedData from preprocessed content."""
        content = preprocessed.content
        if isinstance(content, CsvParsedData):
            return content
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Table Building (module-level for testability)
# ═══════════════════════════════════════════════════════════════════════════

_logger = logging.getLogger("contextifier.csv.content_extractor")


def _build_table_data(
    rows: List[List[str]],
    has_header: bool,
) -> TableData:
    """
    Convert parsed CSV rows into a TableData with merged cell support.

    If empty cells are detected in a merge pattern (adjacent empties
    suggest colspan or rowspan), the corresponding TableCell objects
    get their span attributes set accordingly.

    Non-merged CSVs produce simple 1×1 cells.

    Args:
        rows: Parsed row data (list of string lists).
        has_header: Whether the first row is a header.

    Returns:
        Fully populated TableData.
    """
    if not rows:
        return TableData()

    col_count = max(len(r) for r in rows)

    if _has_merged_cells(rows):
        return _build_with_merges(rows, has_header, col_count)
    return _build_simple(rows, has_header, col_count)


def _build_simple(
    rows: List[List[str]],
    has_header: bool,
    col_count: int,
) -> TableData:
    """Build TableData without merge analysis (fast path)."""
    table_rows: List[List[TableCell]] = []

    for row_idx, row in enumerate(rows):
        cells: List[TableCell] = []
        is_header_row = has_header and row_idx == 0

        for col_idx in range(col_count):
            content = row[col_idx].strip() if col_idx < len(row) else ""
            cells.append(TableCell(
                content=content,
                row_span=1,
                col_span=1,
                is_header=is_header_row,
                row_index=row_idx,
                col_index=col_idx,
            ))
        table_rows.append(cells)

    return TableData(
        rows=table_rows,
        num_rows=len(rows),
        num_cols=col_count,
        has_header=has_header,
    )


def _build_with_merges(
    rows: List[List[str]],
    has_header: bool,
    col_count: int,
) -> TableData:
    """
    Build TableData with colspan/rowspan from empty cell patterns.

    Algorithm (matches v1.0 analyze_merge_info):
    1. Initialize all cells with value, colspan=1, rowspan=1, skip=False.
    2. Horizontal pass: non-empty cell followed by empty cells → colspan.
    3. Vertical pass: non-empty cell followed by empty cells below → rowspan.
    4. Skipped cells are excluded from the final TableData rows.
    """
    row_count = len(rows)

    # Initialize merge info grid
    grid: List[List[Dict[str, Any]]] = []
    for r_idx, row in enumerate(rows):
        row_info: List[Dict[str, Any]] = []
        for c_idx in range(col_count):
            value = row[c_idx].strip() if c_idx < len(row) else ""
            row_info.append({
                "value": value,
                "colspan": 1,
                "rowspan": 1,
                "skip": False,
            })
        grid.append(row_info)

    # Pass 1: Horizontal merge (colspan)
    for r_idx in range(row_count):
        c_idx = 0
        while c_idx < col_count:
            cell = grid[r_idx][c_idx]
            if cell["skip"] or not cell["value"]:
                c_idx += 1
                continue

            colspan = 1
            next_c = c_idx + 1
            while next_c < col_count:
                next_cell = grid[r_idx][next_c]
                if not next_cell["value"] and not next_cell["skip"]:
                    colspan += 1
                    next_cell["skip"] = True
                    next_c += 1
                else:
                    break

            cell["colspan"] = colspan
            c_idx = next_c

    # Pass 2: Vertical merge (rowspan)
    for c_idx in range(col_count):
        r_idx = 0
        while r_idx < row_count:
            cell = grid[r_idx][c_idx]
            if cell["skip"] or not cell["value"]:
                r_idx += 1
                continue

            rowspan = 1
            next_r = r_idx + 1
            while next_r < row_count:
                next_cell = grid[next_r][c_idx]
                if not next_cell["value"] and not next_cell["skip"]:
                    rowspan += 1
                    next_cell["skip"] = True
                    next_r += 1
                else:
                    break

            cell["rowspan"] = rowspan
            r_idx = next_r

    # Build TableData from grid (exclude skipped cells)
    table_rows: List[List[TableCell]] = []
    for r_idx in range(row_count):
        cells: List[TableCell] = []
        is_header_row = has_header and r_idx == 0

        for c_idx in range(col_count):
            cell = grid[r_idx][c_idx]
            if cell["skip"]:
                continue

            cells.append(TableCell(
                content=cell["value"],
                row_span=cell["rowspan"],
                col_span=cell["colspan"],
                is_header=is_header_row,
                row_index=r_idx,
                col_index=c_idx,
            ))
        table_rows.append(cells)

    return TableData(
        rows=table_rows,
        num_rows=row_count,
        num_cols=col_count,
        has_header=has_header,
    )


def _has_merged_cells(rows: List[List[str]]) -> bool:
    """
    Detect whether empty cells suggest merged regions.

    Merge patterns:
    - Vertical: empty cell in first column with non-empty above.
    - Horizontal: empty cell right-adjacent to a non-empty cell.

    Args:
        rows: Parsed row data.

    Returns:
        True if merge patterns are detected.
    """
    if not rows or len(rows) < 2:
        return False

    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            cell_value = cell.strip() if cell else ""
            if cell_value:
                continue

            # Vertical merge indicator: empty first column, non-first row
            if row_idx > 0 and col_idx == 0:
                return True

            # Horizontal merge indicator: previous cell is non-empty
            if col_idx > 0:
                prev = row[col_idx - 1].strip() if col_idx - 1 < len(row) else ""
                if prev:
                    return True

    return False


def _fallback_format(rows: List[List[str]]) -> str:
    """Simple tab-separated fallback when TableService is unavailable."""
    lines: List[str] = []
    for row in rows:
        lines.append("\t".join(cell.strip() for cell in row))
    return "\n".join(lines)


__all__ = ["CsvContentExtractor"]
