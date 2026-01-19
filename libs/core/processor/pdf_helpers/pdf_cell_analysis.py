# service/document_processor/processor/pdf_helpers/cell_analysis.py
"""
Cell Analysis Engine

Analyzes physical cell information and text positions to calculate accurate rowspan/colspan.

- Precise grid analysis based on bbox
- Accurate distinction between merged cells and empty cells
- Enhanced merge validation based on text position
- Improved span inference through adjacent cell analysis
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


class CellAnalysisEngine:
    """
    Cell Analysis Engine.

    Analyzes physical cell information and text positions to calculate accurate rowspan/colspan.

    - Precise bbox-based analysis when PyMuPDF cell info is available
    - Cell position recalculation based on grid lines
    - Accurate distinction between empty cells and merged cells
    """

    # Tolerance constants
    GRID_TOLERANCE = 5.0  # Grid line matching tolerance (pt)
    OVERLAP_THRESHOLD = 0.3  # Bbox overlap threshold

    def __init__(self, table_info: Any, page: Any):
        """
        Args:
            table_info: TableInfo object (requires data, cells_info, bbox attributes)
            page: PyMuPDF page object
        """
        self.table_info = table_info
        self.page = page
        self.data = table_info.data or []
        self.cells_info = table_info.cells_info or []
        self.table_bbox = getattr(table_info, 'bbox', None)

        # Grid line cache
        self._h_grid_lines: List[float] = []
        self._v_grid_lines: List[float] = []

    def analyze(self) -> List[Dict]:
        """
        Perform cell analysis.

        Returns:
            List of cell info (row, col, rowspan, colspan, bbox)

            - Uses existing rowspan/colspan info from TableDetectionEngine if available
            - Improves accuracy by avoiding unnecessary recalculation
        """
        num_rows = len(self.data)
        num_cols = max(len(row) for row in self.data) if self.data else 0

        if num_rows == 0 or num_cols == 0:
            return []

        # Use existing cells with validation if valid rowspan/colspan info exists
        if self.cells_info and self._has_valid_span_info():
            result = self._use_existing_cells_with_validation(num_rows, num_cols)
            if result:
                return result

        # 1. If cell info exists, perform precise bbox-based analysis
        if self.cells_info and any(c.get('bbox') for c in self.cells_info):
            result = self._analyze_with_bbox_grid()
            if result:
                return result

        # 2. If cell info exists but no bbox, validate existing info
        if self.cells_info:
            result = self._validate_and_enhance_cells()
            if result:
                return result

        # 3. If no cell info, create default cells based on data
        return self._create_default_cells(num_rows, num_cols)

    def _has_valid_span_info(self) -> bool:
        """Check if cell info has valid rowspan/colspan.

        Conditions:
        - Two or more cells have rowspan > 1 or colspan > 1
        - Or all cells have row, col information
        """
        if not self.cells_info:
            return False

        has_span = False
        has_position = True

        for cell in self.cells_info:
            rowspan = cell.get('rowspan', 1)
            colspan = cell.get('colspan', 1)

            if rowspan > 1 or colspan > 1:
                has_span = True

            if cell.get('row') is None or cell.get('col') is None:
                has_position = False

        return has_span or has_position

    def _use_existing_cells_with_validation(self, num_rows: int, num_cols: int) -> List[Dict]:
        """Use existing cell info after validation.

        Uses already correctly calculated rowspan/colspan from TableDetectionEngine
        without recalculating, only validates the range.
        """
        validated_cells: List[Dict] = []
        covered_positions: Set[Tuple[int, int]] = set()

        for cell in self.cells_info:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            rowspan = max(1, cell.get('rowspan', 1))
            colspan = max(1, cell.get('colspan', 1))
            bbox = cell.get('bbox')

            # Validate data range
            if row >= num_rows or col >= num_cols:
                continue

            # Adjust span to fit within data range
            rowspan = min(rowspan, num_rows - row)
            colspan = min(colspan, num_cols - col)

            # Check if position is already covered
            if (row, col) in covered_positions:
                continue

            validated_cells.append({
                'row': row,
                'col': col,
                'rowspan': rowspan,
                'colspan': colspan,
                'bbox': bbox
            })

            # Record covered positions
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    covered_positions.add((r, c))

        # Add missing cells (positions not covered by span)
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                if (row_idx, col_idx) not in covered_positions:
                    validated_cells.append({
                        'row': row_idx,
                        'col': col_idx,
                        'rowspan': 1,
                        'colspan': 1,
                        'bbox': None
                    })

        return validated_cells

    def _analyze_with_bbox_grid(self) -> List[Dict]:
        """
        Perform precise grid analysis using bbox information.

        Algorithm:
        1. Extract grid lines from all cell bboxes
        2. Calculate how many grid cells each cell's bbox covers
        3. Determine rowspan/colspan
        """
        # Extract grid lines
        h_lines: Set[float] = set()
        v_lines: Set[float] = set()

        for cell in self.cells_info:
            bbox = cell.get('bbox')
            if bbox and len(bbox) >= 4:
                # Y coordinates (horizontal lines)
                h_lines.add(round(bbox[1], 1))
                h_lines.add(round(bbox[3], 1))
                # X coordinates (vertical lines)
                v_lines.add(round(bbox[0], 1))
                v_lines.add(round(bbox[2], 1))

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        # Sort and cluster
        self._h_grid_lines = self._cluster_and_sort(list(h_lines))
        self._v_grid_lines = self._cluster_and_sort(list(v_lines))

        grid_rows = len(self._h_grid_lines) - 1
        grid_cols = len(self._v_grid_lines) - 1

        if grid_rows < 1 or grid_cols < 1:
            return []

        # Calculate grid position and span for each cell
        analyzed_cells: List[Dict] = []
        covered_positions: Set[Tuple[int, int]] = set()

        # Process cells with bbox
        cells_with_bbox = [c for c in self.cells_info if c.get('bbox')]

        for cell in cells_with_bbox:
            bbox = cell.get('bbox')
            orig_row = cell.get('row', 0)
            orig_col = cell.get('col', 0)

            # Determine grid position from bbox
            row_start = self._find_grid_index(bbox[1], self._h_grid_lines)
            row_end = self._find_grid_index(bbox[3], self._h_grid_lines)
            col_start = self._find_grid_index(bbox[0], self._v_grid_lines)
            col_end = self._find_grid_index(bbox[2], self._v_grid_lines)

            if row_start is None or col_start is None:
                # Use original values if grid matching fails
                row_start = orig_row
                row_end = orig_row + cell.get('rowspan', 1)
                col_start = orig_col
                col_end = orig_col + cell.get('colspan', 1)
            else:
                # If end index is less than or equal to start, span is 1
                if row_end is None or row_end <= row_start:
                    row_end = row_start + 1
                if col_end is None or col_end <= col_start:
                    col_end = col_start + 1

            rowspan = max(1, row_end - row_start)
            colspan = max(1, col_end - col_start)

            # Check and adjust data range
            num_data_rows = len(self.data)
            num_data_cols = max(len(row) for row in self.data) if self.data else 0

            # Grid rows/cols may differ from data rows/cols
            # Map to data index
            data_row = min(row_start, num_data_rows - 1) if num_data_rows > 0 else 0
            data_col = min(col_start, num_data_cols - 1) if num_data_cols > 0 else 0

            # Adjust span to data range
            rowspan = min(rowspan, num_data_rows - data_row)
            colspan = min(colspan, num_data_cols - data_col)

            # Check if position is already covered
            if (data_row, data_col) in covered_positions:
                continue

            analyzed_cells.append({
                'row': data_row,
                'col': data_col,
                'rowspan': max(1, rowspan),
                'colspan': max(1, colspan),
                'bbox': bbox
            })

            # Record covered positions
            for r in range(data_row, min(data_row + rowspan, num_data_rows)):
                for c in range(data_col, min(data_col + colspan, num_data_cols)):
                    covered_positions.add((r, c))

        # Add default cells for uncovered positions
        num_data_rows = len(self.data)
        num_data_cols = max(len(row) for row in self.data) if self.data else 0

        for row_idx in range(num_data_rows):
            for col_idx in range(num_data_cols):
                if (row_idx, col_idx) not in covered_positions:
                    analyzed_cells.append({
                        'row': row_idx,
                        'col': col_idx,
                        'rowspan': 1,
                        'colspan': 1,
                        'bbox': None
                    })

        return analyzed_cells

    def _cluster_and_sort(self, values: List[float], tolerance: float = None) -> List[float]:
        """Cluster and sort values."""
        if not values:
            return []

        if tolerance is None:
            tolerance = self.GRID_TOLERANCE

        sorted_vals = sorted(values)
        clusters: List[List[float]] = [[sorted_vals[0]]]

        for val in sorted_vals[1:]:
            if val - clusters[-1][-1] <= tolerance:
                clusters[-1].append(val)
            else:
                clusters.append([val])

        # Return average value of each cluster
        return [sum(c) / len(c) for c in clusters]

    def _find_grid_index(self, value: float, grid_lines: List[float],
                         tolerance: float = None) -> Optional[int]:
        """Find grid index corresponding to the value."""
        if tolerance is None:
            tolerance = self.GRID_TOLERANCE

        for i, line in enumerate(grid_lines):
            if abs(value - line) <= tolerance:
                return i

        # If no exact match, find the closest line
        if grid_lines:
            closest_idx = 0
            min_diff = abs(value - grid_lines[0])

            for i, line in enumerate(grid_lines[1:], 1):
                diff = abs(value - line)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i

            # Return if within 2x tolerance
            if min_diff <= tolerance * 2:
                return closest_idx

        return None

    def _validate_and_enhance_cells(self) -> List[Dict]:
        """
        Validate and enhance existing cell info.

        - Fix spans that exceed data range
        - Remove duplicate cell info
        - Add missing cells
        """
        num_rows = len(self.data)
        num_cols = max(len(row) for row in self.data) if self.data else 0

        enhanced_cells: List[Dict] = []
        covered_positions: Set[Tuple[int, int]] = set()

        # Process existing cell info
        for cell in self.cells_info:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            rowspan = cell.get('rowspan', 1)
            colspan = cell.get('colspan', 1)
            bbox = cell.get('bbox')

            # Validate and adjust range
            if row >= num_rows or col >= num_cols:
                continue

            rowspan = min(rowspan, num_rows - row)
            colspan = min(colspan, num_cols - col)

            # Check if position is already covered
            if (row, col) in covered_positions:
                continue

            # Text-based span verification (when bbox exists)
            if bbox and self.data:
                verified_rowspan, verified_colspan = self._verify_span_with_text_v2(
                    row, col, rowspan, colspan, bbox
                )
                rowspan = max(rowspan, verified_rowspan)
                colspan = max(colspan, verified_colspan)

            enhanced_cells.append({
                'row': row,
                'col': col,
                'rowspan': max(1, rowspan),
                'colspan': max(1, colspan),
                'bbox': bbox
            })

            # Record covered positions
            for r in range(row, min(row + rowspan, num_rows)):
                for c in range(col, min(col + colspan, num_cols)):
                    covered_positions.add((r, c))

        # Add missing cells
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                if (row_idx, col_idx) not in covered_positions:
                    enhanced_cells.append({
                        'row': row_idx,
                        'col': col_idx,
                        'rowspan': 1,
                        'colspan': 1,
                        'bbox': None
                    })

        return enhanced_cells

    def _verify_span_with_text_v2(
        self,
        row: int,
        col: int,
        rowspan: int,
        colspan: int,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[int, int]:
        """
        Verify span using text position.

        Logic:
        - If current cell has text
        - And adjacent cell is empty
        - And is contained within bbox
        - Extend span
        """
        num_rows = len(self.data)
        num_cols = max(len(row) for row in self.data) if self.data else 0

        # Check current cell value
        current_value = ""
        if row < len(self.data) and col < len(self.data[row]):
            current_value = str(self.data[row][col] or "").strip()

        if not current_value:
            return rowspan, colspan

        verified_rowspan = rowspan
        verified_colspan = colspan

        # Colspan verification: check empty cells to the right in same row
        for c in range(col + colspan, num_cols):
            if c >= len(self.data[row]):
                break
            next_val = str(self.data[row][c] or "").strip()
            if not next_val:
                # Empty cell -> check merge possibility
                # Hard to verify if current bbox extends to that column
                # But if consecutive empty cells, increase colspan
                verified_colspan += 1
            else:
                break

        # Rowspan verification: check empty cells below in same column
        for r in range(row + rowspan, num_rows):
            if col >= len(self.data[r]):
                break
            next_val = str(self.data[r][col] or "").strip()
            if not next_val:
                # Check if other cells in same row have values
                has_value_in_row = any(
                    str(self.data[r][c] or "").strip()
                    for c in range(len(self.data[r]))
                    if c != col
                )
                if has_value_in_row:
                    # If other columns have values, increase rowspan
                    verified_rowspan += 1
                else:
                    break
            else:
                break

        return verified_rowspan, verified_colspan

    def _create_default_cells(self, num_rows: int, num_cols: int) -> List[Dict]:
        """
        Create default cell info. Creates all cells as 1x1 without value-based inference.
        Value-based inference is disabled due to high error rates,
        prioritizing PyMuPDF's physical cell information instead.

        Empty cells are rendered as empty <td> elements in HTML generation.
        (Having empty cells is normal in table structures)
        """
        cells = []

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                cells.append({
                    'row': row_idx,
                    'col': col_idx,
                    'rowspan': 1,
                    'colspan': 1,
                    'bbox': None
                })

        return cells

# ============================================================================
# Export
# ============================================================================

__all__ = [
    'CellAnalysisEngine',
]
