# service/document_processor/processor/pdf_helpers/cell_analysis.py
"""
Cell Analysis Engine - V3.3 고도화

물리적 셀 정보와 텍스트 위치를 분석하여 정확한 rowspan/colspan을 계산합니다.

V3.3 개선사항:
- bbox 기반 정밀 그리드 분석
- 병합셀과 빈 셀의 정확한 구분
- 텍스트 위치 기반 병합 검증 강화
- 인접 셀 분석을 통한 span 추론 개선
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class CellAnalysisEngine:
    """
    셀 분석 엔진 (V3.3 고도화)

    물리적 셀 정보와 텍스트 위치를 분석하여 정확한 rowspan/colspan을 계산합니다.

    V3.3 개선사항:
    - PyMuPDF 셀 정보가 있으면 bbox 기반 정밀 분석
    - 그리드 라인 기반 셀 위치 재계산
    - 빈 셀과 병합셀의 정확한 구분
    """

    # 허용 오차 상수
    GRID_TOLERANCE = 5.0  # 그리드 라인 매칭 허용 오차 (pt)
    OVERLAP_THRESHOLD = 0.3  # bbox 겹침 판단 임계값

    def __init__(self, table_info: Any, page: Any):
        """
        Args:
            table_info: TableInfo 객체 (data, cells_info, bbox 속성 필요)
            page: PyMuPDF 페이지 객체
        """
        self.table_info = table_info
        self.page = page
        self.data = table_info.data or []
        self.cells_info = table_info.cells_info or []
        self.table_bbox = getattr(table_info, 'bbox', None)

        # 그리드 라인 캐시
        self._h_grid_lines: List[float] = []
        self._v_grid_lines: List[float] = []

    def analyze(self) -> List[Dict]:
        """
        셀 분석 수행

        Returns:
            셀 정보 리스트 (row, col, rowspan, colspan, bbox)

        V3.4 개선:
        - TableDetectionEngine에서 이미 계산된 rowspan/colspan 정보가 있으면 그대로 사용
        - 불필요한 재계산 방지로 정확도 향상
        """
        num_rows = len(self.data)
        num_cols = max(len(row) for row in self.data) if self.data else 0

        if num_rows == 0 or num_cols == 0:
            return []

        # V3.4: 셀 정보에 이미 유효한 rowspan/colspan이 있으면 검증 후 바로 사용
        if self.cells_info and self._has_valid_span_info():
            result = self._use_existing_cells_with_validation(num_rows, num_cols)
            if result:
                return result

        # 1. 셀 정보가 있으면 bbox 기반 정밀 분석
        if self.cells_info and any(c.get('bbox') for c in self.cells_info):
            result = self._analyze_with_bbox_grid()
            if result:
                return result

        # 2. 셀 정보가 있지만 bbox가 없으면 기존 정보 검증
        if self.cells_info:
            result = self._validate_and_enhance_cells()
            if result:
                return result

        # 3. 셀 정보가 없으면 데이터 기반 기본 셀 생성
        return self._create_default_cells(num_rows, num_cols)

    def _has_valid_span_info(self) -> bool:
        """
        V3.4: 셀 정보에 유효한 rowspan/colspan이 있는지 확인

        조건:
        - 2개 이상의 셀에서 rowspan > 1 또는 colspan > 1인 경우
        - 또는 모든 셀에 row, col 정보가 있는 경우
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
        """
        V3.4: 기존 셀 정보를 검증 후 그대로 사용

        TableDetectionEngine에서 이미 올바르게 계산된 rowspan/colspan을
        다시 계산하지 않고 범위만 검증하여 사용
        """
        validated_cells: List[Dict] = []
        covered_positions: Set[Tuple[int, int]] = set()

        for cell in self.cells_info:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            rowspan = max(1, cell.get('rowspan', 1))
            colspan = max(1, cell.get('colspan', 1))
            bbox = cell.get('bbox')

            # 데이터 범위 검증
            if row >= num_rows or col >= num_cols:
                continue

            # span을 데이터 범위 내로 조정
            rowspan = min(rowspan, num_rows - row)
            colspan = min(colspan, num_cols - col)

            # 이미 커버된 위치인지 확인
            if (row, col) in covered_positions:
                continue

            validated_cells.append({
                'row': row,
                'col': col,
                'rowspan': rowspan,
                'colspan': colspan,
                'bbox': bbox
            })

            # 커버된 위치 기록
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    covered_positions.add((r, c))

        # 누락된 셀 추가 (span으로 커버되지 않은 위치)
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
        bbox 정보를 사용하여 정밀 그리드 분석 수행.

        알고리즘:
        1. 모든 셀 bbox에서 그리드 라인 추출
        2. 각 셀의 bbox가 몇 개의 그리드 셀을 커버하는지 계산
        3. rowspan/colspan 결정
        """
        # 그리드 라인 추출
        h_lines: Set[float] = set()
        v_lines: Set[float] = set()

        for cell in self.cells_info:
            bbox = cell.get('bbox')
            if bbox and len(bbox) >= 4:
                # Y 좌표 (수평 라인)
                h_lines.add(round(bbox[1], 1))
                h_lines.add(round(bbox[3], 1))
                # X 좌표 (수직 라인)
                v_lines.add(round(bbox[0], 1))
                v_lines.add(round(bbox[2], 1))

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        # 정렬 및 클러스터링
        self._h_grid_lines = self._cluster_and_sort(list(h_lines))
        self._v_grid_lines = self._cluster_and_sort(list(v_lines))

        grid_rows = len(self._h_grid_lines) - 1
        grid_cols = len(self._v_grid_lines) - 1

        if grid_rows < 1 or grid_cols < 1:
            return []

        # 각 셀에 대해 그리드 위치와 span 계산
        analyzed_cells: List[Dict] = []
        covered_positions: Set[Tuple[int, int]] = set()

        # bbox가 있는 셀 처리
        cells_with_bbox = [c for c in self.cells_info if c.get('bbox')]

        for cell in cells_with_bbox:
            bbox = cell.get('bbox')
            orig_row = cell.get('row', 0)
            orig_col = cell.get('col', 0)

            # bbox로 그리드 위치 결정
            row_start = self._find_grid_index(bbox[1], self._h_grid_lines)
            row_end = self._find_grid_index(bbox[3], self._h_grid_lines)
            col_start = self._find_grid_index(bbox[0], self._v_grid_lines)
            col_end = self._find_grid_index(bbox[2], self._v_grid_lines)

            if row_start is None or col_start is None:
                # 그리드 매칭 실패 시 원래 값 사용
                row_start = orig_row
                row_end = orig_row + cell.get('rowspan', 1)
                col_start = orig_col
                col_end = orig_col + cell.get('colspan', 1)
            else:
                # 종료 인덱스가 시작보다 작거나 같으면 1 span
                if row_end is None or row_end <= row_start:
                    row_end = row_start + 1
                if col_end is None or col_end <= col_start:
                    col_end = col_start + 1

            rowspan = max(1, row_end - row_start)
            colspan = max(1, col_end - col_start)

            # 데이터 범위 확인 및 조정
            num_data_rows = len(self.data)
            num_data_cols = max(len(row) for row in self.data) if self.data else 0

            # 그리드 행/열이 데이터 행/열과 다를 수 있음
            # 데이터 인덱스로 매핑
            data_row = min(row_start, num_data_rows - 1) if num_data_rows > 0 else 0
            data_col = min(col_start, num_data_cols - 1) if num_data_cols > 0 else 0

            # span도 데이터 범위로 조정
            rowspan = min(rowspan, num_data_rows - data_row)
            colspan = min(colspan, num_data_cols - data_col)

            # 이미 커버된 위치인지 확인
            if (data_row, data_col) in covered_positions:
                continue

            analyzed_cells.append({
                'row': data_row,
                'col': data_col,
                'rowspan': max(1, rowspan),
                'colspan': max(1, colspan),
                'bbox': bbox
            })

            # 커버된 위치 기록
            for r in range(data_row, min(data_row + rowspan, num_data_rows)):
                for c in range(data_col, min(data_col + colspan, num_data_cols)):
                    covered_positions.add((r, c))

        # 커버되지 않은 셀에 대해 기본 셀 추가
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
        """값들을 클러스터링하고 정렬"""
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

        # 각 클러스터의 평균값 반환
        return [sum(c) / len(c) for c in clusters]

    def _find_grid_index(self, value: float, grid_lines: List[float],
                         tolerance: float = None) -> Optional[int]:
        """값에 해당하는 그리드 인덱스 찾기"""
        if tolerance is None:
            tolerance = self.GRID_TOLERANCE

        for i, line in enumerate(grid_lines):
            if abs(value - line) <= tolerance:
                return i

        # 정확히 일치하지 않으면 가장 가까운 라인 찾기
        if grid_lines:
            closest_idx = 0
            min_diff = abs(value - grid_lines[0])

            for i, line in enumerate(grid_lines[1:], 1):
                diff = abs(value - line)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i

            # 허용 오차의 2배 이내면 반환
            if min_diff <= tolerance * 2:
                return closest_idx

        return None

    def _validate_and_enhance_cells(self) -> List[Dict]:
        """
        기존 셀 정보 검증 및 보완.

        V3.3 개선:
        - 데이터 범위를 벗어나는 span 수정
        - 중복 셀 정보 제거
        - 누락된 셀 추가
        """
        num_rows = len(self.data)
        num_cols = max(len(row) for row in self.data) if self.data else 0

        enhanced_cells: List[Dict] = []
        covered_positions: Set[Tuple[int, int]] = set()

        # 기존 셀 정보 처리
        for cell in self.cells_info:
            row = cell.get('row', 0)
            col = cell.get('col', 0)
            rowspan = cell.get('rowspan', 1)
            colspan = cell.get('colspan', 1)
            bbox = cell.get('bbox')

            # 범위 검증 및 조정
            if row >= num_rows or col >= num_cols:
                continue

            rowspan = min(rowspan, num_rows - row)
            colspan = min(colspan, num_cols - col)

            # 이미 커버된 위치인지 확인
            if (row, col) in covered_positions:
                continue

            # 텍스트 기반 span 검증 (bbox가 있는 경우)
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

            # 커버된 위치 기록
            for r in range(row, min(row + rowspan, num_rows)):
                for c in range(col, min(col + colspan, num_cols)):
                    covered_positions.add((r, c))

        # 누락된 셀 추가
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
        텍스트 위치로 span 검증 (V3.3 강화).

        로직:
        - 현재 셀에 텍스트가 있고
        - 인접 셀이 비어있으면서
        - bbox 내에 포함되면
        - span 확장
        """
        num_rows = len(self.data)
        num_cols = max(len(row) for row in self.data) if self.data else 0

        # 현재 셀 값 확인
        current_value = ""
        if row < len(self.data) and col < len(self.data[row]):
            current_value = str(self.data[row][col] or "").strip()

        if not current_value:
            return rowspan, colspan

        verified_rowspan = rowspan
        verified_colspan = colspan

        # colspan 검증: 같은 행에서 오른쪽으로 빈 셀 확인
        for c in range(col + colspan, num_cols):
            if c >= len(self.data[row]):
                break
            next_val = str(self.data[row][c] or "").strip()
            if not next_val:
                # 빈 셀 → 병합 가능성 확인
                # 현재 bbox가 해당 열까지 확장되어 있는지 확인은 어렵지만
                # 연속된 빈 셀이면 colspan 증가
                verified_colspan += 1
            else:
                break

        # rowspan 검증: 같은 열에서 아래로 빈 셀 확인
        for r in range(row + rowspan, num_rows):
            if col >= len(self.data[r]):
                break
            next_val = str(self.data[r][col] or "").strip()
            if not next_val:
                # 같은 열의 다른 셀에도 값이 있는지 확인
                has_value_in_row = any(
                    str(self.data[r][c] or "").strip()
                    for c in range(len(self.data[r]))
                    if c != col
                )
                if has_value_in_row:
                    # 다른 열에 값이 있으면 rowspan 증가
                    verified_rowspan += 1
                else:
                    break
            else:
                break

        return verified_rowspan, verified_colspan

    def _create_default_cells(self, num_rows: int, num_cols: int) -> List[Dict]:
        """
        기본 셀 정보 생성.

        V3.3: 값 기반 추론을 제거하고 모든 셀을 1x1로 생성.
        값 기반 추론은 오류가 많아 비활성화하고,
        PyMuPDF의 물리적 셀 정보를 우선 사용합니다.

        빈 셀은 HTML 생성 시 그대로 빈 <td>로 표현됩니다.
        (빈 셀이 있는 것은 정상적인 테이블 구조임)
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

    def _infer_cells_from_data_v2(self) -> List[Dict]:
        """
        ★ V3.3 비활성화: 값 기반 병합셀 추론

        이 메서드는 잘못된 추론으로 인해 문제를 일으킬 수 있어
        기본적으로 비활성화됩니다.

        원래 로직의 문제점:
        1. 빈 셀 = 병합셀이라는 잘못된 가정
        2. 실제로 비어있는 데이터 셀과 병합셀 구분 불가
        3. 표 구조를 왜곡시킴

        PyMuPDF의 물리적 셀 정보가 없는 경우,
        모든 셀을 1x1로 처리하는 것이 더 안전합니다.
        """
        return self._create_default_cells(
            len(self.data),
            max(len(row) for row in self.data) if self.data else 0
        )


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'CellAnalysisEngine',
]
