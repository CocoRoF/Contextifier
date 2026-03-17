# contextifier_new/handlers/rtf/_table_parser.py
"""
RTF Table Parser — internal module for extracting tables from RTF content.

RTF encodes tables with these control words:
- ``\\trowd``     : Table row definition start
- ``\\cellxN``    : Cell right boundary (N in twips)
- ``\\clmgf``     : Horizontal merge — first cell
- ``\\clmrg``     : Horizontal merge — continuation
- ``\\clvmgf``    : Vertical merge — first cell
- ``\\clvmrg``    : Vertical merge — continuation
- ``\\intbl``     : Paragraph belongs to a table cell
- ``\\cell``      : Cell content end
- ``\\row``       : Row end

This module parses RTF table structures and produces standard
``TableData`` / ``TableCell`` objects for the pipeline.

Ported from v1.0 rtf_table_extractor.py with:
- Output is TableData/TableCell instead of RTFTable.to_html()
- Uses shared _cleaner and _decoder utilities
- find_excluded_regions integrated from _cleaner module
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

from contextifier_new.types import TableCell, TableData
from contextifier_new.handlers.rtf._decoder import decode_hex_escapes
from contextifier_new.handlers.rtf._cleaner import (
    clean_rtf_text,
    find_excluded_regions,
    is_in_excluded_region,
)

_logger = logging.getLogger("contextifier.rtf.table")

# Row proximity threshold: consecutive rows within this char distance
# are grouped into the same table.
_ROW_GAP_THRESHOLD = 150


# ═══════════════════════════════════════════════════════════════════════════
# Internal Data Structures
# ═══════════════════════════════════════════════════════════════════════════

class _CellDef(NamedTuple):
    """Cell definition parsed from row header (before \\cell content)."""
    h_merge_first: bool   # \clmgf — start horizontal merge
    h_merge_cont: bool    # \clmrg — continue horizontal merge
    v_merge_first: bool   # \clvmgf — start vertical merge
    v_merge_cont: bool    # \clvmrg — continue vertical merge
    right_boundary: int   # \cellxN — right boundary in twips


class _ParsedCell(NamedTuple):
    """Cell with content + merge info, prior to TableCell conversion."""
    text: str
    h_merge_first: bool
    h_merge_cont: bool
    v_merge_first: bool
    v_merge_cont: bool


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def extract_tables(
    content: str,
    encoding: str = "cp949",
) -> List[TableData]:
    """
    Extract all tables from RTF content.

    Args:
        content: Decoded RTF string.
        encoding: Source encoding for hex escape decoding.

    Returns:
        List of TableData.
    """
    _, regions = extract_tables_with_positions(content, encoding)
    return [table for _, _, table in regions]


def extract_tables_with_positions(
    content: str,
    encoding: str = "cp949",
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, TableData]]]:
    """
    Extract tables from RTF content with position information.

    Position info is needed by the ContentExtractor to interleave
    table references with surrounding inline text.

    Args:
        content: Decoded RTF string.
        encoding: Source encoding for hex escape decoding.

    Returns:
        Tuple of:
        - position_ranges: List of (start, end) character positions
        - table_regions: List of (start, end, TableData)
    """
    excluded_regions = find_excluded_regions(content)

    # Find all \row end positions
    row_ends: List[int] = [
        m.end()
        for m in re.finditer(r"\\row(?![a-z])", content)
    ]
    if not row_ends:
        return [], []

    # Find \trowd before each \row → (start, end, text) triples
    raw_rows: List[Tuple[int, int, str]] = []
    for i, row_end in enumerate(row_ends):
        search_start = row_ends[i - 1] if i > 0 else 0
        segment = content[search_start:row_end]
        trowd_match = re.search(r"\\trowd", segment)

        if trowd_match is None:
            continue

        row_start = search_start + trowd_match.start()

        if is_in_excluded_region(row_start, excluded_regions):
            _logger.debug(
                "Skipping table row at %d (header/footer/footnote)", row_start,
            )
            continue

        row_text = content[row_start:row_end]
        raw_rows.append((row_start, row_end, row_text))

    if not raw_rows:
        return [], []

    # Group consecutive rows into tables
    table_groups = _group_rows_into_tables(raw_rows)
    _logger.info("Found %d table groups from %d rows", len(table_groups), len(raw_rows))

    # Parse each group → TableData
    position_ranges: List[Tuple[int, int]] = []
    table_regions: List[Tuple[int, int, TableData]] = []

    for start_pos, end_pos, row_texts in table_groups:
        table = _parse_table(row_texts, encoding)
        if table is not None and table.rows:
            position_ranges.append((start_pos, end_pos))
            table_regions.append((start_pos, end_pos, table))

    _logger.info("Extracted %d valid tables", len(table_regions))
    return position_ranges, table_regions


# ═══════════════════════════════════════════════════════════════════════════
# Row Grouping
# ═══════════════════════════════════════════════════════════════════════════

def _group_rows_into_tables(
    raw_rows: List[Tuple[int, int, str]],
) -> List[Tuple[int, int, List[str]]]:
    """
    Group consecutive rows into tables.

    Rows that are within ``_ROW_GAP_THRESHOLD`` characters
    of each other belong to the same table.

    Returns:
        List of (start_pos, end_pos, list_of_row_texts).
    """
    groups: List[Tuple[int, int, List[str]]] = []
    current_texts: List[str] = []
    current_start = -1
    current_end = -1
    prev_end = -1

    for row_start, row_end, row_text in raw_rows:
        if prev_end == -1 or (row_start - prev_end) < _ROW_GAP_THRESHOLD:
            if current_start == -1:
                current_start = row_start
            current_texts.append(row_text)
            current_end = row_end
        else:
            if current_texts:
                groups.append((current_start, current_end, current_texts))
            current_texts = [row_text]
            current_start = row_start
            current_end = row_end
        prev_end = row_end

    if current_texts:
        groups.append((current_start, current_end, current_texts))

    return groups


# ═══════════════════════════════════════════════════════════════════════════
# Table Parsing (Rows → TableData)
# ═══════════════════════════════════════════════════════════════════════════

def _parse_table(
    row_texts: List[str],
    encoding: str,
) -> Optional[TableData]:
    """
    Parse a list of RTF row strings into a TableData.

    Steps:
    1. For each row: parse cell definitions + cell text content
    2. Match definitions to content → List[_ParsedCell] per row
    3. Determine if it is a real table (≥ 2 effective columns)
    4. Calculate colspan/rowspan from merge flags
    5. Build TableData with TableCell instances
    """
    parsed_rows: List[List[_ParsedCell]] = []
    for row_text in row_texts:
        cells = _extract_cells_with_merge(row_text, encoding)
        if cells:
            parsed_rows.append(cells)

    if not parsed_rows:
        return None

    # Check if this is a real table (not a list disguised as table)
    if not _is_real_table(parsed_rows):
        return None

    return _build_table_data(parsed_rows)


def _is_real_table(parsed_rows: List[List[_ParsedCell]]) -> bool:
    """
    Determine if parsed rows represent a real table (≥ 2 effective columns).

    A single-column structure (Nx1) is considered a list, not a table.
    """
    if not parsed_rows:
        return False

    effective_counts: List[int] = []
    for row in parsed_rows:
        non_empty = [
            i for i, cell in enumerate(row)
            if not cell.h_merge_cont and (cell.text.strip() or cell.v_merge_first)
        ]
        if non_empty:
            effective_counts.append(max(non_empty) + 1)

    return max(effective_counts, default=0) >= 2


def _build_table_data(
    parsed_rows: List[List[_ParsedCell]],
) -> TableData:
    """
    Convert parsed cells (with merge flags) into standard TableData.

    Merge flag semantics:
    - h_merge_first + h_merge_cont → colspan
    - v_merge_first + v_merge_cont → rowspan
    """
    num_rows = len(parsed_rows)
    max_cols = max(len(row) for row in parsed_rows) if parsed_rows else 0

    # Initialize merge map: (colspan, rowspan) per cell
    merge: List[List[Tuple[int, int]]] = [
        [(1, 1) for _ in range(max_cols)] for _ in range(num_rows)
    ]

    # ── Horizontal merge (colspan) ─────────────────────────────────────
    for r_idx, row in enumerate(parsed_rows):
        c_idx = 0
        while c_idx < len(row):
            cell = row[c_idx]
            if cell.h_merge_first:
                colspan = 1
                for nc in range(c_idx + 1, len(row)):
                    if row[nc].h_merge_cont:
                        colspan += 1
                        merge[r_idx][nc] = (0, 0)  # consumed
                    else:
                        break
                merge[r_idx][c_idx] = (colspan, 1)
            c_idx += 1

    # ── Vertical merge (rowspan) ───────────────────────────────────────
    for c_idx in range(max_cols):
        r_idx = 0
        while r_idx < num_rows:
            if c_idx >= len(parsed_rows[r_idx]):
                r_idx += 1
                continue
            cell = parsed_rows[r_idx][c_idx]

            if cell.v_merge_first:
                rowspan = 1
                for nr in range(r_idx + 1, num_rows):
                    if c_idx < len(parsed_rows[nr]) and parsed_rows[nr][c_idx].v_merge_cont:
                        rowspan += 1
                        merge[nr][c_idx] = (0, 0)  # consumed
                    else:
                        break
                # Preserve existing colspan
                existing_cs = merge[r_idx][c_idx][0]
                merge[r_idx][c_idx] = (existing_cs, rowspan)
                r_idx += rowspan
            elif cell.v_merge_cont:
                merge[r_idx][c_idx] = (0, 0)
                r_idx += 1
            else:
                r_idx += 1

    # ── Build TableCell rows ───────────────────────────────────────────
    table_rows: List[List[TableCell]] = []
    for r_idx, row in enumerate(parsed_rows):
        table_cells: List[TableCell] = []
        for c_idx, cell in enumerate(row):
            if c_idx < max_cols:
                cs, rs = merge[r_idx][c_idx]
                if cs == 0 or rs == 0:
                    # Merged-away cell → skip (not added to output)
                    continue
            else:
                cs, rs = 1, 1

            text = re.sub(r"\s+", " ", cell.text).strip()
            table_cells.append(TableCell(
                content=text,
                row_span=rs,
                col_span=cs,
                is_header=False,
                row_index=r_idx,
                col_index=c_idx,
            ))
        table_rows.append(table_cells)

    return TableData(
        rows=table_rows,
        num_rows=num_rows,
        num_cols=max_cols,
        has_header=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cell Extraction (single row)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_cells_with_merge(
    row_text: str,
    encoding: str,
) -> List[_ParsedCell]:
    """
    Extract cell content and merge information from a single RTF table row.

    1. Parse cell attribute definitions (before first \\cell)
    2. Extract cell text segments (between \\cell delimiters)
    3. Match definitions to content
    """
    # ── Step 1: Parse cell definitions ─────────────────────────────────
    first_cell_pos = _find_first_cell_pos(row_text)
    def_part = row_text[:first_cell_pos]

    cell_defs = _parse_cell_definitions(def_part)

    # ── Step 2: Extract cell texts ─────────────────────────────────────
    cell_texts = _extract_cell_texts(row_text, encoding)

    # ── Step 3: Match definitions to content ───────────────────────────
    cells: List[_ParsedCell] = []
    for i, text in enumerate(cell_texts):
        if i < len(cell_defs):
            d = cell_defs[i]
            cells.append(_ParsedCell(
                text=text,
                h_merge_first=d.h_merge_first,
                h_merge_cont=d.h_merge_cont,
                v_merge_first=d.v_merge_first,
                v_merge_cont=d.v_merge_cont,
            ))
        else:
            cells.append(_ParsedCell(
                text=text,
                h_merge_first=False,
                h_merge_cont=False,
                v_merge_first=False,
                v_merge_cont=False,
            ))

    return cells


def _find_first_cell_pos(row_text: str) -> int:
    """
    Find the position of the first ``\\cell`` (not ``\\cellx``).

    ``\\cellx`` defines a cell boundary (definition part).
    ``\\cell`` ends a cell's content.
    """
    pos = 0
    while True:
        idx = row_text.find("\\cell", pos)
        if idx == -1:
            return len(row_text)
        # Check it's not \cellx
        after = idx + 5
        if after < len(row_text) and row_text[after] == "x":
            pos = idx + 1
            continue
        return idx


_CELL_DEF_PATTERN = re.compile(r"\\cl(?:mgf|mrg|vmgf|vmrg)|\\cellx(-?\d+)")


def _parse_cell_definitions(def_part: str) -> List[_CellDef]:
    """
    Parse cell attribute definitions from the row header.

    Scans for ``\\clmgf``, ``\\clmrg``, ``\\clvmgf``, ``\\clvmrg``
    and ``\\cellxN``. Each ``\\cellx`` terminates one cell definition.
    """
    defs: List[_CellDef] = []

    h_first = False
    h_cont = False
    v_first = False
    v_cont = False

    for match in _CELL_DEF_PATTERN.finditer(def_part):
        token = match.group()
        if token == "\\clmgf":
            h_first = True
        elif token == "\\clmrg":
            h_cont = True
        elif token == "\\clvmgf":
            v_first = True
        elif token == "\\clvmrg":
            v_cont = True
        elif token.startswith("\\cellx"):
            boundary = int(match.group(1)) if match.group(1) else 0
            defs.append(_CellDef(
                h_merge_first=h_first,
                h_merge_cont=h_cont,
                v_merge_first=v_first,
                v_merge_cont=v_cont,
                right_boundary=boundary,
            ))
            # Reset for next cell definition
            h_first = h_cont = v_first = v_cont = False

    return defs


def _extract_cell_texts(
    row_text: str,
    encoding: str,
) -> List[str]:
    """
    Extract the text content of each cell in a row.

    Cells are delimited by ``\\cell`` (not ``\\cellx``).
    Content between the last ``\\cellx`` and first ``\\cell``
    is the first cell's content.
    """
    # Find all \cell positions (excluding \cellx)
    cell_positions: List[int] = []
    pos = 0
    while True:
        idx = row_text.find("\\cell", pos)
        if idx == -1:
            break
        after = idx + 5
        if after < len(row_text) and row_text[after] == "x":
            pos = idx + 1
            continue
        cell_positions.append(idx)
        pos = idx + 1

    if not cell_positions:
        return []

    # Find end of last \cellx before first \cell (= content start)
    first_cell_pos = cell_positions[0]
    def_part = row_text[:first_cell_pos]
    last_cellx_end = 0
    for m in re.finditer(r"\\cellx-?\d+", def_part):
        last_cellx_end = m.end()

    # Extract text between boundaries
    texts: List[str] = []
    prev_end = last_cellx_end
    for cell_end in cell_positions:
        raw = row_text[prev_end:cell_end]
        decoded = decode_hex_escapes(raw, encoding)
        cleaned = clean_rtf_text(decoded, encoding)
        texts.append(cleaned)
        prev_end = cell_end + 5  # len("\\cell") = 5

    return texts


# ═══════════════════════════════════════════════════════════════════════════
# Utility: Convert single-column "table" to text
# ═══════════════════════════════════════════════════════════════════════════

def single_column_to_text(
    row_texts: List[str],
    encoding: str,
) -> str:
    """
    Convert a single-column table group to plain text.

    Used when _is_real_table() returns False — the content is
    rendered as paragraph text rather than a table structure.

    Args:
        row_texts: RTF row strings.
        encoding: Encoding for decoding.

    Returns:
        Concatenated cell text.
    """
    lines: List[str] = []
    for row_text in row_texts:
        cells = _extract_cells_with_merge(row_text, encoding)
        for cell in cells:
            text = cell.text.strip()
            if text:
                lines.append(text)
    return "\n\n".join(lines)


__all__ = [
    "extract_tables",
    "extract_tables_with_positions",
    "single_column_to_text",
]
