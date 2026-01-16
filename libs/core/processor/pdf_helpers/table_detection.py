"""
Table Detection Engine for PDF Handler

Detects tables using multiple strategies and selects the best results.
Includes graphic region exclusion and fake table filtering capabilities.
Improved cell extraction accuracy.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Set

import fitz
import pdfplumber

from libs.core.processor.pdf_helpers.types import (
    PDFConfig,
    TableDetectionStrategy,
    GridInfo,
    CellInfo,
    TableCandidate,
)
from libs.core.processor.pdf_helpers.line_analysis import LineAnalysisEngine
from libs.core.processor.pdf_helpers.graphic_detector import GraphicRegionDetector
from libs.core.processor.pdf_helpers.table_validator import TableQualityValidator

logger = logging.getLogger(__name__)


# ============================================================================
# Table Detection Engine
# ============================================================================

class TableDetectionEngine:
    """
    Table Detection Engine

    Detects tables using multiple strategies and selects the best results.

    Features:
        - GraphicRegionDetector integration to exclude vector graphic regions
        - TableQualityValidator integration to filter fake tables

    Supported Strategies:
        1. PyMuPDF find_tables() - Most accurate, preferred
        2. pdfplumber - Line-based detection
        3. Line-based - Direct line analysis
    """

    # Configuration constants
    CONFIDENCE_THRESHOLD = getattr(PDFConfig, 'CONFIDENCE_THRESHOLD', 0.5)
    MIN_TABLE_ROWS = getattr(PDFConfig, 'MIN_TABLE_ROWS', 2)
    MIN_TABLE_COLS = getattr(PDFConfig, 'MIN_TABLE_COLS', 2)

    def __init__(self, page, page_num: int, file_path: str):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            file_path: PDF file path
        """
        self.page = page
        self.page_num = page_num
        self.file_path = file_path
        self.page_width = page.rect.width
        self.page_height = page.rect.height

        # Line analysis engine
        self.line_engine = LineAnalysisEngine(page, self.page_width, self.page_height)
        self.h_lines, self.v_lines = self.line_engine.analyze()

        # Graphic region detector
        self.graphic_detector = GraphicRegionDetector(page, page_num)
        self.graphic_regions = self.graphic_detector.detect()

        # Table quality validator
        self.quality_validator = TableQualityValidator(page, self.graphic_detector)

    def detect_tables(self) -> List[TableCandidate]:
        """
        Detect tables using all strategies.

        Returns:
            List of table candidates sorted by confidence.
        """
        candidates: List[TableCandidate] = []

        # Strategy 1: PyMuPDF
        pymupdf_candidates = self._detect_with_pymupdf()

        # Pre-merge adjacent header-data tables (before validation)
        pymupdf_candidates = self._merge_header_data_tables(pymupdf_candidates)
        candidates.extend(pymupdf_candidates)

        # Strategy 2: pdfplumber
        pdfplumber_candidates = self._detect_with_pdfplumber()
        pdfplumber_candidates = self._merge_header_data_tables(pdfplumber_candidates)
        candidates.extend(pdfplumber_candidates)

        # Strategy 3: Line-based (HYBRID_ANALYSIS)
        # Used only when PyMuPDF and pdfplumber don't find tables
        # Or used additionally with stricter validation
        line_candidates = self._detect_with_lines()

        # Enhanced cross-validation for HYBRID results
        if line_candidates and not pymupdf_candidates:
            # When PyMuPDF didn't find tables but HYBRID did
            # Apply higher confidence threshold (0.65 or above)
            line_candidates = [
                c for c in line_candidates
                if c.confidence >= 0.65
            ]
            logger.debug(f"[TableDetection] HYBRID-only detection: "
                        f"{len(line_candidates)} candidates passed stricter threshold (0.65)")

        candidates.extend(line_candidates)

        # Filter fake tables through quality validation
        validated_candidates = self._validate_candidates(candidates)

        # Select best candidates based on confidence
        selected = self._select_best_candidates(validated_candidates)

        return selected

    def _merge_header_data_tables(self, candidates: List[TableCandidate]) -> List[TableCandidate]:
        """
        Merge adjacent header-data tables.

        Conditions:
            1. First table has 1-2 rows (assumed to be header)
            2. Second table is directly below (Y gap < 30pt)
            3. X range is similar (80% or more overlap)
            4. Column count relationship: header columns <= data columns
        """
        if len(candidates) < 2:
            return candidates

        # Sort by Y position
        sorted_candidates = sorted(candidates, key=lambda c: c.bbox[1])
        merged = []
        skip_indices = set()

        for i, header_cand in enumerate(sorted_candidates):
            if i in skip_indices:
                continue

            # Check header candidate condition (1-2 rows)
            if len(header_cand.data) > 2:
                merged.append(header_cand)
                continue

            # Check if can merge with next table
            merged_cand = header_cand
            for j in range(i + 1, len(sorted_candidates)):
                if j in skip_indices:
                    continue

                data_cand = sorted_candidates[j]

                if self._can_merge_header_data(merged_cand, data_cand):
                    merged_cand = self._do_merge_header_data(merged_cand, data_cand)
                    skip_indices.add(j)
                    logger.debug(f"[TableDetection] Merged header with data table: "
                               f"header rows={len(header_cand.data)}, "
                               f"data rows={len(data_cand.data)}")
                else:
                    break

            merged.append(merged_cand)

        return merged

    def _can_merge_header_data(self, header: TableCandidate, data: TableCandidate) -> bool:
        """Determine if header and data tables can be merged."""
        # Check Y gap
        y_gap = data.bbox[1] - header.bbox[3]
        if y_gap < -5 or y_gap > 40:  # Allow slight overlap, max 40pt gap
            return False

        # Check X range overlap
        x_overlap_start = max(header.bbox[0], data.bbox[0])
        x_overlap_end = min(header.bbox[2], data.bbox[2])
        x_overlap = max(0, x_overlap_end - x_overlap_start)

        header_width = header.bbox[2] - header.bbox[0]
        data_width = data.bbox[2] - data.bbox[0]
        max_width = max(header_width, data_width)

        if max_width > 0 and x_overlap / max_width < 0.7:
            return False

        # Check column count relationship
        header_cols = max(len(row) for row in header.data) if header.data else 0
        data_cols = max(len(row) for row in data.data) if data.data else 0

        # Don't merge if header has more columns than data
        if header_cols > data_cols + 1:
            return False

        return True

    def _do_merge_header_data(self, header: TableCandidate, data: TableCandidate) -> TableCandidate:
        """Perform header and data table merge (includes subheader detection)."""
        # New bbox
        merged_bbox = (
            min(header.bbox[0], data.bbox[0]),
            header.bbox[1],
            max(header.bbox[2], data.bbox[2]),
            data.bbox[3]
        )

        # Determine column count
        header_cols = max(len(row) for row in header.data) if header.data else 0
        data_cols = max(len(row) for row in data.data) if data.data else 0
        merged_cols = max(header_cols, data_cols)

        # Detect subheader between header and data
        subheader_row = self._detect_subheader_between(header, data, merged_cols)

        # Merge data
        merged_data = []
        merged_cells = []

        # Process header rows
        for row_idx, row in enumerate(header.data):
            if len(row) < merged_cols:
                # Apply colspan if header has fewer columns
                adjusted_row = list(row)
                col_diff = merged_cols - len(row)

                # Apply colspan to second column
                if len(row) >= 2 and col_diff > 0:
                    # Store colspan info
                    merged_cells.append({
                        'row': row_idx,
                        'col': 1,
                        'rowspan': 1,
                        'colspan': 1 + col_diff,
                        'bbox': None
                    })
                    # Add empty columns
                    for _ in range(col_diff):
                        adjusted_row.insert(2, '')
                else:
                    adjusted_row.extend([''] * col_diff)

                merged_data.append(adjusted_row)
            else:
                merged_data.append(list(row))

        # Insert subheader row (header cell info)
        header_row_count = len(header.data)
        if subheader_row:
            merged_data.append(subheader_row)
            # Add cell info for subheader row (each cell has colspan=1)
            subheader_row_idx = header_row_count  # Row after header
            for col_idx, cell_value in enumerate(subheader_row):
                merged_cells.append({
                    'row': subheader_row_idx,
                    'col': col_idx,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })
            header_row_count += 1
            logger.debug(f"[TableDetection] Added subheader row with cell info: {subheader_row}")

        # Header cell info
        if header.cells:
            for cell in header.cells:
                if not any(c['row'] == cell.row and c['col'] == cell.col for c in merged_cells):
                    merged_cells.append({
                        'row': cell.row,
                        'col': cell.col,
                        'rowspan': cell.rowspan,
                        'colspan': cell.colspan,
                        'bbox': cell.bbox
                    })

        # Process data rows
        for row_idx, row in enumerate(data.data):
            if len(row) < merged_cols:
                adjusted_row = list(row) + [''] * (merged_cols - len(row))
            else:
                adjusted_row = list(row)
            merged_data.append(adjusted_row)

        # Data cell info (apply row offset)
        if data.cells:
            for cell in data.cells:
                merged_cells.append({
                    'row': cell.row + header_row_count,
                    'col': cell.col,
                    'rowspan': cell.rowspan,
                    'colspan': cell.colspan,
                    'bbox': cell.bbox
                })

        # Convert cell info to CellInfo objects
        cell_objects = [
            CellInfo(
                row=c['row'],
                col=c['col'],
                rowspan=c.get('rowspan', 1),
                colspan=c.get('colspan', 1),
                # Use default value if bbox is None or missing
                bbox=c.get('bbox') or (0, 0, 0, 0)
            )
            for c in merged_cells
        ]

        return TableCandidate(
            strategy=header.strategy,
            confidence=max(header.confidence, data.confidence),
            bbox=merged_bbox,
            grid=header.grid or data.grid,
            cells=cell_objects,
            data=merged_data,
            raw_table=None
        )

    def _detect_subheader_between(self, header: TableCandidate, data: TableCandidate,
                                   num_cols: int) -> Optional[List[str]]:
        """
        Detect subheader row between header and data tables.

        Example: Sub-column headers like (A), (B), etc.
        """
        header_bottom = header.bbox[3]
        data_top = data.bbox[1]

        # Must have sufficient gap between header and data
        gap = data_top - header_bottom
        if gap < 5 or gap > 50:
            return None

        # Extract text from the region on the page
        page_dict = self.page.get_text("dict", sort=True)

        subheader_texts = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_bbox = line.get("bbox", (0, 0, 0, 0))
                line_y = (line_bbox[1] + line_bbox[3]) / 2

                # Check if located between header and data
                if header_bottom - 5 <= line_y <= data_top + 5:
                    # Check if within table X range
                    if line_bbox[0] >= header.bbox[0] - 10 and line_bbox[2] <= data.bbox[2] + 10:
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            span_bbox = span.get("bbox", (0, 0, 0, 0))
                            if text and text not in [' ', '']:
                                subheader_texts.append({
                                    'text': text,
                                    'x0': span_bbox[0],
                                    'x1': span_bbox[2]
                                })

        if not subheader_texts:
            return None

        # Check subheader pattern: (A), (B), etc.
        has_subheader_pattern = any('(' in t['text'] and ')' in t['text'] for t in subheader_texts)
        if not has_subheader_pattern:
            return None

        # Construct subheader row
        table_left = min(header.bbox[0], data.bbox[0])
        table_width = max(header.bbox[2], data.bbox[2]) - table_left
        col_width = table_width / num_cols

        subheader_row = [''] * num_cols
        for item in sorted(subheader_texts, key=lambda x: x['x0']):
            relative_x = item['x0'] - table_left
            col_idx = min(int(relative_x / col_width), num_cols - 1)
            col_idx = max(0, col_idx)

            if subheader_row[col_idx]:
                subheader_row[col_idx] += ' ' + item['text']
            else:
                subheader_row[col_idx] = item['text']

        # Validate subheader (must have at least one (A), (B) pattern)
        valid_count = sum(1 for s in subheader_row if '(' in s and ')' in s)
        if valid_count < 1:
            return None

        return subheader_row

    def _validate_candidates(self, candidates: List[TableCandidate]) -> List[TableCandidate]:
        """
        Validate table candidates for quality.

        Validation criteria:
            1. Not overlapping with graphic regions (except PyMuPDF - text-based, high reliability)
            2. Sufficient filled cell ratio
            3. Has meaningful data

        Tables detected with PyMuPDF strategy skip graphic region check.
        Reason: PyMuPDF detects tables based on text, so it accurately recognizes
        tables even when cells with background colors are mistaken as graphics.
        """
        validated = []

        for candidate in candidates:
            # PyMuPDF strategy skips graphic region check
            skip_graphic_check = (candidate.strategy == TableDetectionStrategy.PYMUPDF_NATIVE)

            is_valid, new_confidence, reason = self.quality_validator.validate(
                data=candidate.data,
                bbox=candidate.bbox,
                cells_info=candidate.cells,
                skip_graphic_check=skip_graphic_check  # New parameter
            )

            if is_valid:
                # Adjust confidence based on validation result
                adjusted_confidence = min(candidate.confidence, new_confidence)

                validated.append(TableCandidate(
                    strategy=candidate.strategy,
                    confidence=adjusted_confidence,
                    bbox=candidate.bbox,
                    grid=candidate.grid,
                    cells=candidate.cells,
                    data=candidate.data,
                    raw_table=candidate.raw_table
                ))
            else:
                logger.debug(f"[TableDetection] Filtered out candidate: page={self.page_num+1}, "
                           f"bbox={candidate.bbox}, reason={reason}")

        return validated

    def _detect_with_pymupdf(self) -> List[TableCandidate]:
        """Use PyMuPDF find_tables() (tolerance settings to resolve double-line issues)."""
        candidates = []

        if not hasattr(self.page, 'find_tables'):
            return candidates

        try:
            # Apply same tolerance settings as pdf_handler.py
            # Resolves fake column creation due to double/triple line borders
            # snap_tolerance: Snaps nearby coordinates together
            # join_tolerance: Joins nearby lines together
            # edge_min_length: Ignores short lines (border lines)
            # intersection_tolerance: Intersection detection tolerance
            tabs = self.page.find_tables(
                snap_tolerance=7,
                join_tolerance=7,
                edge_min_length=10,
                intersection_tolerance=7,
            )

            for table_idx, table in enumerate(tabs.tables):
                try:
                    table_data = table.extract()

                    if not table_data or not any(any(cell for cell in row if cell) for row in table_data):
                        continue

                    # Narrow column merge processing
                    merged_data, col_mapping = self._merge_narrow_columns(
                        table_data, table.cells if hasattr(table, 'cells') else None
                    )

                    # Calculate confidence (with merged data)
                    confidence = self._calculate_pymupdf_confidence(table, merged_data)

                    if confidence < self.CONFIDENCE_THRESHOLD:
                        continue

                    # Extract cell info (apply col_mapping)
                    cells = self._extract_cells_from_pymupdf_with_mapping(table, col_mapping)

                    candidates.append(TableCandidate(
                        strategy=TableDetectionStrategy.PYMUPDF_NATIVE,
                        confidence=confidence,
                        bbox=table.bbox,
                        grid=None,
                        cells=cells,
                        data=merged_data,
                        raw_table=table
                    ))

                except Exception as e:
                    logger.debug(f"[PDF] PyMuPDF table extraction error: {e}")
                    continue

        except Exception as e:
            logger.debug(f"[PDF] PyMuPDF find_tables error: {e}")

        return candidates

    def _merge_narrow_columns(
        self,
        data: List[List],
        cells: List[Tuple] = None,
        min_col_width: float = 15.0
    ) -> Tuple[List[List[str]], Dict[int, int]]:
        """
        Merge narrow columns with adjacent columns.

        Removes fake columns generated by double/triple line borders in PDF.

        Args:
            data: Table data
            cells: PyMuPDF cell bbox list
            min_col_width: Minimum column width (pt)

        Returns:
            (merged data, original column -> new column mapping)
        """
        if not data or not data[0]:
            return data, {}

        num_cols = max(len(row) for row in data)

        # Analyze columns based on text if no cell info
        if not cells:
            return self._merge_columns_by_content(data)

        # Calculate width per column
        col_widths = self._calculate_column_widths(cells, num_cols)

        # Determine column groups to merge
        col_groups = self._determine_column_groups(col_widths, min_col_width)

        if len(col_groups) == num_cols:
            # No merge needed
            return data, {i: i for i in range(num_cols)}

        # Create column mapping
        col_mapping = {}
        for new_idx, group in enumerate(col_groups):
            for old_idx in group:
                col_mapping[old_idx] = new_idx

        # Merge data
        merged_data = []
        for row in data:
            new_row = [''] * len(col_groups)
            for old_idx, cell_val in enumerate(row):
                if old_idx in col_mapping:
                    new_idx = col_mapping[old_idx]
                    if cell_val and str(cell_val).strip():
                        if new_row[new_idx]:
                            new_row[new_idx] += str(cell_val).strip()
                        else:
                            new_row[new_idx] = str(cell_val).strip()
            merged_data.append(new_row)

        logger.debug(f"[TableDetection] Merged {num_cols} columns -> {len(col_groups)} columns")

        return merged_data, col_mapping

    def _calculate_column_widths(self, cells: List[Tuple], num_cols: int) -> List[float]:
        """Calculate column widths from cell bbox."""
        if not cells:
            return [0.0] * num_cols

        # Collect X coordinates
        x_coords = sorted(set([c[0] for c in cells if c] + [c[2] for c in cells if c]))

        if len(x_coords) < 2:
            return [0.0] * num_cols

        # Calculate column widths
        widths = []
        for i in range(len(x_coords) - 1):
            widths.append(x_coords[i + 1] - x_coords[i])

        # Match num_cols
        if len(widths) < num_cols:
            widths.extend([0.0] * (num_cols - len(widths)))
        elif len(widths) > num_cols:
            widths = widths[:num_cols]

        return widths

    def _determine_column_groups(
        self,
        col_widths: List[float],
        min_width: float
    ) -> List[List[int]]:
        """
        Determine column groups to merge based on column widths.

        Narrow columns are merged with the next wider column.
        """
        groups = []
        current_group = []

        for idx, width in enumerate(col_widths):
            current_group.append(idx)

            # Finalize group when total width meets minimum
            group_width = sum(col_widths[i] for i in current_group)

            if group_width >= min_width:
                groups.append(current_group)
                current_group = []

        # Handle last group
        if current_group:
            if groups:
                # Merge with previous group
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)

        return groups

    def _merge_columns_by_content(self, data: List[List]) -> Tuple[List[List[str]], Dict[int, int]]:
        """
        Merge empty columns based on text content.

        Columns empty in most rows are merged with adjacent columns.
        """
        if not data or not data[0]:
            return data, {}

        num_cols = max(len(row) for row in data)
        num_rows = len(data)

        # Calculate "emptiness" ratio for each column
        empty_ratios = []
        for col_idx in range(num_cols):
            empty_count = 0
            for row in data:
                if col_idx >= len(row) or not row[col_idx] or not str(row[col_idx]).strip():
                    empty_count += 1
            empty_ratios.append(empty_count / num_rows if num_rows > 0 else 1.0)

        # Find columns with 90%+ empty ratio and merge with adjacent
        groups = []
        current_group = []

        for col_idx, empty_ratio in enumerate(empty_ratios):
            current_group.append(col_idx)

            # Finalize group for non-empty columns
            if empty_ratio < 0.9:
                groups.append(current_group)
                current_group = []

        # Handle last group
        if current_group:
            if groups:
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)

        if len(groups) == num_cols:
            return data, {i: i for i in range(num_cols)}

        # Create column mapping
        col_mapping = {}
        for new_idx, group in enumerate(groups):
            for old_idx in group:
                col_mapping[old_idx] = new_idx

        # Merge data
        merged_data = []
        for row in data:
            new_row = [''] * len(groups)
            for old_idx, cell_val in enumerate(row):
                if old_idx in col_mapping:
                    new_idx = col_mapping[old_idx]
                    if cell_val and str(cell_val).strip():
                        if new_row[new_idx]:
                            new_row[new_idx] += str(cell_val).strip()
                        else:
                            new_row[new_idx] = str(cell_val).strip()
            merged_data.append(new_row)

        logger.debug(f"[TableDetection] Content-based merge: {num_cols} -> {len(groups)} columns")

        return merged_data, col_mapping

    def _extract_cells_from_pymupdf_with_mapping(
        self,
        table,
        col_mapping: Dict[int, int]
    ) -> List[CellInfo]:
        """
        Extract cell info with column mapping applied.
        """
        if not col_mapping:
            return self._extract_cells_from_pymupdf(table)

        cells = self._extract_cells_from_pymupdf(table)

        if not cells:
            return cells

        # Calculate mapped column count
        new_col_count = max(col_mapping.values()) + 1 if col_mapping else 0

        # Remap cell info
        remapped_cells = []
        processed_positions = set()

        for cell in cells:
            old_col = cell.col
            new_col = col_mapping.get(old_col, old_col)

            # Skip if cell already exists at same position
            if (cell.row, new_col) in processed_positions:
                continue

            # Recalculate colspan: consider merged columns
            new_colspan = 1
            for c in range(cell.col, cell.col + cell.colspan):
                mapped_c = col_mapping.get(c, c)
                if mapped_c != new_col:
                    new_colspan = max(new_colspan, mapped_c - new_col + 1)

            new_colspan = min(new_colspan, new_col_count - new_col)

            remapped_cells.append(CellInfo(
                row=cell.row,
                col=new_col,
                rowspan=cell.rowspan,
                colspan=max(1, new_colspan),
                bbox=cell.bbox
            ))

            # Record covered positions
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(new_col, new_col + max(1, new_colspan)):
                    processed_positions.add((r, c))

        return remapped_cells

    def _calculate_pymupdf_confidence(self, table, data: List[List]) -> float:
        """
        Calculate PyMuPDF result confidence.

        Features:
            - Higher base score (trusting PyMuPDF results)
            - Relaxed penalties
            - Stronger cell info bonus
        """
        score = 0.0

        # Higher base score (PyMuPDF is highly reliable)
        score += 0.6

        # Score based on row/column count
        num_rows = len(data)
        if num_rows >= self.MIN_TABLE_ROWS:
            score += 0.1
        if table.col_count >= self.MIN_TABLE_COLS:
            score += 0.1

        # Score based on data density (relaxed penalties)
        total_cells = sum(len(row) for row in data)
        filled_cells = sum(1 for row in data for cell in row if cell and str(cell).strip())

        if total_cells > 0:
            density = filled_cells / total_cells

            if density < 0.05:
                # Penalty only for very low density
                score -= 0.2
            elif density < 0.1:
                score -= 0.1
            else:
                score += density * 0.15

        # Additional score for cell info (stronger bonus)
        if hasattr(table, 'cells') and table.cells:
            score += 0.15

        # Check meaningful cell count (relaxed penalty)
        meaningful_count = sum(
            1 for row in data for cell in row
            if cell and len(str(cell).strip()) >= 2
        )

        if meaningful_count < 2:
            score -= 0.1

        # Check valid row count (relaxed penalty)
        valid_rows = sum(1 for row in data if any(cell and str(cell).strip() for cell in row))
        if valid_rows <= 1:
            score -= 0.1

        # Check graphic region overlap (relaxed penalty)
        if self.graphic_detector:
            if self.graphic_detector.is_bbox_in_graphic_region(table.bbox, threshold=0.5):
                score -= 0.15

        return max(0.0, min(1.0, score))

    def _extract_cells_from_pymupdf(self, table) -> List[CellInfo]:
        """
        Extract cell info from PyMuPDF table.

        Applies logic from pdf_handler_default's _extract_cell_spans_from_table():
            1. Extract physical bbox for each cell from table.cells
            2. Map Y coordinates to row indices, X coordinates to column indices
            3. Calculate rowspan/colspan if cell bbox spans multiple grid cells
        """
        cells = []

        if not hasattr(table, 'cells') or not table.cells:
            # Return empty list if no cell info (handled by CellAnalysisEngine)
            return cells

        raw_cells = table.cells
        if not raw_cells:
            return cells

        # Extract X, Y boundary lines (same approach as pdf_handler_default)
        x_coords = sorted(set([c[0] for c in raw_cells if c] + [c[2] for c in raw_cells if c]))
        y_coords = sorted(set([c[1] for c in raw_cells if c] + [c[3] for c in raw_cells if c]))

        if len(x_coords) < 2 or len(y_coords) < 2:
            # Return basic cell info if grid cannot be constructed
            for idx, cell_bbox in enumerate(raw_cells):
                if cell_bbox is None:
                    continue
                num_rows = len(table.extract()) if hasattr(table, 'extract') else 0
                row_idx = idx // max(1, table.col_count) if hasattr(table, 'col_count') else 0
                col_idx = idx % max(1, table.col_count) if hasattr(table, 'col_count') else 0
                cells.append(CellInfo(
                    row=row_idx,
                    col=col_idx,
                    rowspan=1,
                    colspan=1,
                    bbox=cell_bbox
                ))
            return cells

        # Function to map coordinates to grid indices (same as pdf_handler_default)
        def coord_to_index(coord: float, coords: List[float], tolerance: float = 3.0) -> int:
            for i, c in enumerate(coords):
                if abs(coord - c) <= tolerance:
                    return i
            # Return closest index
            return min(range(len(coords)), key=lambda i: abs(coords[i] - coord))

        # Track processed grid positions
        processed_positions: Set[Tuple[int, int]] = set()

        for cell_bbox in raw_cells:
            if cell_bbox is None:
                continue

            x0, y0, x1, y1 = cell_bbox[:4]

            col_start = coord_to_index(x0, x_coords)
            col_end = coord_to_index(x1, x_coords)
            row_start = coord_to_index(y0, y_coords)
            row_end = coord_to_index(y1, y_coords)

            colspan = max(1, col_end - col_start)
            rowspan = max(1, row_end - row_start)

            if (row_start, col_start) in processed_positions:
                continue

            processed_positions.add((row_start, col_start))

            cells.append(CellInfo(
                row=row_start,
                col=col_start,
                rowspan=rowspan,
                colspan=colspan,
                bbox=cell_bbox
            ))

            # Mark other cells in merged area
            for r in range(row_start, row_start + rowspan):
                for c in range(col_start, col_start + colspan):
                    if (r, c) != (row_start, col_start):
                        processed_positions.add((r, c))

        return cells

    def _cluster_grid_positions(self, positions: List[float], tolerance: float = 3.0) -> List[float]:
        """
        Cluster grid positions.

        Merge nearby lines into one.
        """
        if not positions:
            return []

        sorted_pos = sorted(set(positions))
        if len(sorted_pos) == 0:
            return []

        clusters: List[List[float]] = [[sorted_pos[0]]]

        for pos in sorted_pos[1:]:
            if pos - clusters[-1][-1] <= tolerance:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])

        # Return average value of each cluster
        return [sum(c) / len(c) for c in clusters]

    def _find_grid_index_v2(self, value: float, grid_lines: List[float],
                            tolerance: float = 5.0) -> Optional[int]:
        """
        Find index of value in grid lines (improved version).

        If exact matching fails, select the closest line.
        """
        if not grid_lines:
            return None

        # Try exact matching
        for i, line in enumerate(grid_lines):
            if abs(value - line) <= tolerance:
                return i

        # Find closest line
        min_diff = float('inf')
        closest_idx = 0

        for i, line in enumerate(grid_lines):
            diff = abs(value - line)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        # Return if within 3x tolerance
        if min_diff <= tolerance * 3:
            return closest_idx

        return None

    def _find_grid_index(self, value: float, grid_lines: List[float], tolerance: float = 3.0) -> Optional[int]:
        """Find index of value in grid lines (for compatibility)."""
        return self._find_grid_index_v2(value, grid_lines, tolerance)

    def _detect_with_pdfplumber(self) -> List[TableCandidate]:
        """Use pdfplumber for table detection."""
        candidates = []

        try:
            with pdfplumber.open(self.file_path) as pdf:
                if self.page_num >= len(pdf.pages):
                    return candidates

                plumber_page = pdf.pages[self.page_num]

                # Table settings
                settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                }

                tables = plumber_page.extract_tables(settings)

                for table_idx, table_data in enumerate(tables):
                    if not table_data or not any(any(cell for cell in row if cell) for row in table_data):
                        continue

                    # Estimate bbox
                    bbox = self._estimate_table_bbox_pdfplumber(plumber_page, table_data)

                    if not bbox:
                        continue

                    confidence = self._calculate_pdfplumber_confidence(table_data)

                    if confidence < self.CONFIDENCE_THRESHOLD:
                        continue

                    candidates.append(TableCandidate(
                        strategy=TableDetectionStrategy.PDFPLUMBER_LINES,
                        confidence=confidence,
                        bbox=bbox,
                        grid=None,
                        cells=[],
                        data=table_data,
                        raw_table=None
                    ))

        except Exception as e:
            logger.debug(f"[PDF] pdfplumber error: {e}")

        return candidates

    def _estimate_table_bbox_pdfplumber(self, page, data: List[List]) -> Optional[Tuple[float, float, float, float]]:
        """Estimate pdfplumber table bbox."""
        try:
            words = page.extract_words()
            if not words:
                return None

            table_texts = set()
            for row in data:
                for cell in row:
                    if cell and str(cell).strip():
                        table_texts.add(str(cell).strip()[:20])

            matching_words = []
            for word in words:
                if any(word['text'] in text or text in word['text'] for text in table_texts):
                    matching_words.append(word)

            if not matching_words:
                return None

            x0 = min(w['x0'] for w in matching_words)
            y0 = min(w['top'] for w in matching_words)
            x1 = max(w['x1'] for w in matching_words)
            y1 = max(w['bottom'] for w in matching_words)

            margin = 5
            return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)

        except Exception:
            return None

    def _calculate_pdfplumber_confidence(self, data: List[List]) -> float:
        """Calculate pdfplumber result confidence."""
        score = 0.0

        # Base score (slightly lower than PyMuPDF)
        score += 0.4

        num_rows = len(data)
        col_count = max(len(row) for row in data) if data else 0

        if num_rows >= self.MIN_TABLE_ROWS:
            score += 0.1
        if col_count >= self.MIN_TABLE_COLS:
            score += 0.1

        # Data density
        total_cells = sum(len(row) for row in data)
        filled_cells = sum(1 for row in data for cell in row if cell and str(cell).strip())

        if total_cells > 0:
            density = filled_cells / total_cells

            if density < 0.1:
                score -= 0.5
            elif density < 0.2:
                score -= 0.3
            else:
                score += density * 0.2

        # Meaningful cell count
        meaningful_count = sum(
            1 for row in data for cell in row
            if cell and len(str(cell).strip()) >= 2
        )

        if meaningful_count < 2:
            score -= 0.3

        # Valid row count
        valid_rows = sum(1 for row in data if any(cell and str(cell).strip() for cell in row))
        if valid_rows <= 1:
            score -= 0.2

        # Empty row ratio
        empty_rows = num_rows - valid_rows
        if num_rows > 0 and empty_rows / num_rows > 0.5:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _detect_with_lines(self) -> List[TableCandidate]:
        """Line analysis based table detection."""
        candidates = []

        # Build grid
        grid = self.line_engine.build_grid()

        if not grid:
            return candidates

        # Recover incomplete border
        if not grid.is_complete:
            grid = self.line_engine.reconstruct_incomplete_border(grid)
            if not grid.is_complete:
                return candidates

        # Check if grid is valid
        if grid.row_count < self.MIN_TABLE_ROWS or grid.col_count < self.MIN_TABLE_COLS:
            return candidates

        # Extract text from cells
        data = self._extract_text_from_grid(grid)

        if not data or not any(any(cell for cell in row if cell) for row in data):
            return candidates

        # Create cell info
        cells = self._create_cells_from_grid(grid)

        # Calculate confidence
        confidence = self._calculate_line_based_confidence(grid, data)

        if confidence < self.CONFIDENCE_THRESHOLD:
            return candidates

        candidates.append(TableCandidate(
            strategy=TableDetectionStrategy.HYBRID_ANALYSIS,
            confidence=confidence,
            bbox=grid.bbox,
            grid=grid,
            cells=cells,
            data=data,
            raw_table=None
        ))

        return candidates

    def _extract_text_from_grid(self, grid: GridInfo) -> List[List[Optional[str]]]:
        """Extract text from grid cells."""
        data = []

        page_dict = self.page.get_text("dict", sort=True)

        for row_idx in range(grid.row_count):
            row_data = []
            y0 = grid.h_lines[row_idx]
            y1 = grid.h_lines[row_idx + 1]

            for col_idx in range(grid.col_count):
                x0 = grid.v_lines[col_idx]
                x1 = grid.v_lines[col_idx + 1]

                cell_bbox = (x0, y0, x1, y1)
                cell_text = self._get_text_in_bbox(page_dict, cell_bbox)
                row_data.append(cell_text)

            data.append(row_data)

        return data

    def _get_text_in_bbox(self, page_dict: dict, bbox: Tuple[float, float, float, float]) -> str:
        """Extract text within bbox."""
        x0, y0, x1, y1 = bbox
        texts = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_bbox = line.get("bbox", (0, 0, 0, 0))

                if self._bbox_overlaps(line_bbox, bbox):
                    line_text = ""
                    for span in line.get("spans", []):
                        span_bbox = span.get("bbox", (0, 0, 0, 0))
                        if self._bbox_overlaps(span_bbox, bbox):
                            line_text += span.get("text", "")

                    if line_text.strip():
                        texts.append(line_text.strip())

        return " ".join(texts)

    def _bbox_overlaps(self, bbox1: Tuple, bbox2: Tuple, threshold: float = 0.3) -> bool:
        """Check if two bboxes overlap."""
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])

        if x1 <= x0 or y1 <= y0:
            return False

        overlap_area = (x1 - x0) * (y1 - y0)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

        if bbox1_area <= 0:
            return False

        return overlap_area / bbox1_area >= threshold

    def _create_cells_from_grid(self, grid: GridInfo) -> List[CellInfo]:
        """Create cell info from grid."""
        cells = []

        for row_idx in range(grid.row_count):
            y0 = grid.h_lines[row_idx]
            y1 = grid.h_lines[row_idx + 1]

            for col_idx in range(grid.col_count):
                x0 = grid.v_lines[col_idx]
                x1 = grid.v_lines[col_idx + 1]

                cells.append(CellInfo(
                    row=row_idx,
                    col=col_idx,
                    rowspan=1,
                    colspan=1,
                    bbox=(x0, y0, x1, y1)
                ))

        return cells

    def _calculate_line_based_confidence(self, grid: GridInfo, data: List[List]) -> float:
        """Calculate line-based result confidence."""
        score = 0.0

        # Base score (lower than other strategies)
        score += 0.3

        # Grid completeness
        if grid.is_complete:
            score += 0.2
        elif grid.reconstructed:
            score += 0.1

        # Row/column count
        if grid.row_count >= self.MIN_TABLE_ROWS:
            score += 0.1
        if grid.col_count >= self.MIN_TABLE_COLS:
            score += 0.1

        # Data density
        total_cells = sum(len(row) for row in data)
        filled_cells = sum(1 for row in data for cell in row if cell and str(cell).strip())

        if total_cells > 0:
            density = filled_cells / total_cells

            if density < 0.1:
                score -= 0.4
            elif density < 0.2:
                score -= 0.2
            else:
                score += density * 0.2

        # Meaningful cell count
        meaningful_count = sum(
            1 for row in data for cell in row
            if cell and len(str(cell).strip()) >= 2
        )

        if meaningful_count < 2:
            score -= 0.2

        # Valid row count
        valid_rows = sum(1 for row in data if any(cell and str(cell).strip() for cell in row))
        if valid_rows <= 1:
            score -= 0.2

        # Check graphic region overlap
        if self.graphic_detector:
            if self.graphic_detector.is_bbox_in_graphic_region(grid.bbox, threshold=0.3):
                score -= 0.3

        return max(0.0, min(1.0, score))

    def _select_best_candidates(self, candidates: List[TableCandidate]) -> List[TableCandidate]:
        """
        Select best table candidates.

        Strongly reflects strategy priority:
            - PyMuPDF is most accurate, so PyMuPDF results are preferred in the same region
            - If confidence difference is less than 0.2, select by strategy priority
        """
        if not candidates:
            return []

        # Strategy priority: PYMUPDF > PDFPLUMBER > HYBRID
        priority_order = {
            TableDetectionStrategy.PYMUPDF_NATIVE: 0,
            TableDetectionStrategy.PDFPLUMBER_LINES: 1,
            TableDetectionStrategy.HYBRID_ANALYSIS: 2,
            TableDetectionStrategy.BORDERLESS_HEURISTIC: 3,
        }

        # Changed sort key - prioritize strategy order more
        # If confidence difference is not large, decide by strategy priority
        def sort_key(c):
            # Subtract strategy priority * 0.15 from confidence
            # This makes PyMuPDF (priority=0) more favorable than pdfplumber (priority=1)
            adjusted_confidence = c.confidence - (priority_order.get(c.strategy, 99) * 0.15)
            return (-adjusted_confidence, priority_order.get(c.strategy, 99))

        candidates_sorted = sorted(candidates, key=sort_key)

        selected = []

        for candidate in candidates_sorted:
            overlaps = False

            for selected_candidate in selected:
                if self._tables_overlap_any(candidate.bbox, selected_candidate.bbox):
                    overlaps = True
                    break

            if not overlaps:
                selected.append(candidate)

        return selected

    def _tables_overlap_any(self, bbox1: Tuple, bbox2: Tuple, threshold: float = 0.3) -> bool:
        """
        Check if two tables overlap (improved version).

        Returns True if either one is covered by the other by threshold or more.
        """
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])

        if x1 <= x0 or y1 <= y0:
            return False

        overlap_area = (x1 - x0) * (y1 - y0)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if bbox1_area <= 0 or bbox2_area <= 0:
            return False

        # Consider overlapping if either side is covered by threshold or more
        ratio1 = overlap_area / bbox1_area
        ratio2 = overlap_area / bbox2_area

        return ratio1 >= threshold or ratio2 >= threshold


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'TableDetectionEngine',
]
