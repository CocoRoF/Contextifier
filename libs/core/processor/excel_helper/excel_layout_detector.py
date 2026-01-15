"""
Excel 레이아웃 및 객체 감지 모듈

엑셀 시트에서 실제 데이터가 있는 영역(layout)을 감지합니다.
개별 객체(테이블) 감지:
1. 테두리가 있는 영역을 먼저 개별 개체로 인식
2. 완전히 붙어있는 인접 개체들을 병합
3. 각 개체를 사각형 영역으로 반환
"""

import logging
from typing import Tuple, Optional, List, Set, Dict
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("document-processor")


@dataclass
class LayoutRange:
    """레이아웃 범위 정보"""
    min_row: int  # 시작 행 (1-based)
    max_row: int  # 끝 행 (1-based)
    min_col: int  # 시작 열 (1-based)
    max_col: int  # 끝 열 (1-based)
    
    def is_valid(self) -> bool:
        """유효한 범위인지 확인"""
        return (self.min_row > 0 and self.max_row > 0 and 
                self.min_col > 0 and self.max_col > 0 and
                self.min_row <= self.max_row and 
                self.min_col <= self.max_col)
    
    def row_count(self) -> int:
        """행 개수"""
        return self.max_row - self.min_row + 1
    
    def col_count(self) -> int:
        """열 개수"""
        return self.max_col - self.min_col + 1
    
    def cell_count(self) -> int:
        """셀 개수"""
        return self.row_count() * self.col_count()
    
    def is_adjacent(self, other: 'LayoutRange') -> bool:
        """다른 LayoutRange와 완전히 인접해 있는지 확인 (변이 맞닿아 있음)"""
        # 수평으로 인접 (같은 행 범위에서 열이 맞닿음)
        if self.min_row <= other.max_row and self.max_row >= other.min_row:
            if self.max_col + 1 == other.min_col or other.max_col + 1 == self.min_col:
                return True
        # 수직으로 인접 (같은 열 범위에서 행이 맞닿음)
        if self.min_col <= other.max_col and self.max_col >= other.min_col:
            if self.max_row + 1 == other.min_row or other.max_row + 1 == self.min_row:
                return True
        return False
    
    def merge_with(self, other: 'LayoutRange') -> 'LayoutRange':
        """다른 LayoutRange와 병합하여 새로운 범위 반환"""
        return LayoutRange(
            min_row=min(self.min_row, other.min_row),
            max_row=max(self.max_row, other.max_row),
            min_col=min(self.min_col, other.min_col),
            max_col=max(self.max_col, other.max_col)
        )
    
    def overlaps(self, other: 'LayoutRange') -> bool:
        """다른 LayoutRange와 겹치는지 확인"""
        return not (self.max_row < other.min_row or 
                    self.min_row > other.max_row or
                    self.max_col < other.min_col or 
                    self.min_col > other.max_col)


def layout_detect_range_xlsx(ws) -> Optional[LayoutRange]:
    """
    XLSX 워크시트에서 실제 데이터가 있는 영역을 감지합니다.
    
    Args:
        ws: openpyxl Worksheet 객체
    
    Returns:
        LayoutRange 객체 또는 데이터가 없으면 None
    """
    try:
        if ws.max_row is None or ws.max_row == 0:
            return None
        
        sheet_max_row = min(ws.max_row, 1000)
        sheet_max_col = min(ws.max_column, 100) if ws.max_column else 100
        
        min_row = None
        max_row = None
        min_col = None
        max_col = None
        
        # 왼쪽→오른쪽으로 첫 번째 데이터 열 찾기
        for col_idx in range(1, sheet_max_col + 1):
            for row_idx in range(1, sheet_max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value is not None and str(cell.value).strip():
                    min_col = col_idx
                    break
            if min_col is not None:
                break
        
        if min_col is None:
            return None
        
        # 위→아래로 첫 번째 데이터 행 찾기
        for row_idx in range(1, sheet_max_row + 1):
            for col_idx in range(min_col, sheet_max_col + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value is not None and str(cell.value).strip():
                    min_row = row_idx
                    break
            if min_row is not None:
                break
        
        if min_row is None:
            return None
        
        # 오른쪽→왼쪽으로 마지막 데이터 열 찾기
        for col_idx in range(sheet_max_col, min_col - 1, -1):
            for row_idx in range(min_row, sheet_max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value is not None and str(cell.value).strip():
                    max_col = col_idx
                    break
            if max_col is not None:
                break
        
        if max_col is None:
            max_col = min_col
        
        # 아래→위로 마지막 데이터 행 찾기
        for row_idx in range(sheet_max_row, min_row - 1, -1):
            for col_idx in range(min_col, max_col + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value is not None and str(cell.value).strip():
                    max_row = row_idx
                    break
            if max_row is not None:
                break
        
        if max_row is None:
            max_row = min_row
        
        layout = LayoutRange(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col)
        logger.debug(f"Layout detected: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
        return layout
        
    except Exception as e:
        logger.warning(f"Error detecting layout range: {e}")
        return None


def layout_detect_range_xls(sheet) -> Optional[LayoutRange]:
    """
    XLS 시트에서 실제 데이터가 있는 영역을 감지합니다.
    
    Args:
        sheet: xlrd Sheet 객체
    
    Returns:
        LayoutRange 객체 또는 데이터가 없으면 None
    """
    try:
        if sheet.nrows == 0 or sheet.ncols == 0:
            return None
        
        sheet_max_row = min(sheet.nrows, 1000)
        sheet_max_col = min(sheet.ncols, 100)
        
        min_row = None
        max_row = None
        min_col = None
        max_col = None
        
        # 왼쪽→오른쪽으로 첫 번째 데이터 열 찾기 (0-based)
        for col_idx in range(sheet_max_col):
            for row_idx in range(sheet_max_row):
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value is not None and str(value).strip():
                        min_col = col_idx + 1  # 1-based
                        break
                except Exception:
                    pass
            if min_col is not None:
                break
        
        if min_col is None:
            return None
        
        # 위→아래로 첫 번째 데이터 행 찾기
        for row_idx in range(sheet_max_row):
            for col_idx in range(min_col - 1, sheet_max_col):
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value is not None and str(value).strip():
                        min_row = row_idx + 1  # 1-based
                        break
                except Exception:
                    pass
            if min_row is not None:
                break
        
        if min_row is None:
            return None
        
        # 오른쪽→왼쪽으로 마지막 데이터 열 찾기
        for col_idx in range(sheet_max_col - 1, min_col - 2, -1):
            for row_idx in range(min_row - 1, sheet_max_row):
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value is not None and str(value).strip():
                        max_col = col_idx + 1  # 1-based
                        break
                except Exception:
                    pass
            if max_col is not None:
                break
        
        if max_col is None:
            max_col = min_col
        
        # 아래→위로 마지막 데이터 행 찾기
        for row_idx in range(sheet_max_row - 1, min_row - 2, -1):
            for col_idx in range(min_col - 1, max_col):
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value is not None and str(value).strip():
                        max_row = row_idx + 1  # 1-based
                        break
                except Exception:
                    pass
            if max_row is not None:
                break
        
        if max_row is None:
            max_row = min_row
        
        layout = LayoutRange(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col)
        logger.debug(f"XLS Layout detected: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
        return layout
        
    except Exception as e:
        logger.warning(f"Error detecting XLS layout range: {e}")
        return None


def _has_border_xlsx(cell) -> bool:
    """XLSX 셀에 테두리가 있는지 확인 (상하좌우 중 하나라도)"""
    try:
        border = cell.border
        if border is None:
            return False
        
        sides = [border.top, border.bottom, border.left, border.right]
        for side in sides:
            if side is not None and side.style is not None and side.style != 'none':
                return True
        return False
    except Exception:
        return False


def _detect_bordered_regions_xlsx(ws, layout: LayoutRange) -> List[LayoutRange]:
    """
    XLSX 워크시트에서 테두리가 있는 영역들을 감지합니다.
    테두리가 있는 셀들을 BFS로 그룹화하여 사각형 영역으로 반환합니다.
    
    Args:
        ws: openpyxl Worksheet 객체
        layout: 탐색할 레이아웃 범위
    
    Returns:
        테두리 영역 목록
    """
    # 테두리가 있는 셀 좌표 수집
    bordered_cells: Set[Tuple[int, int]] = set()
    
    for row_idx in range(layout.min_row, layout.max_row + 1):
        for col_idx in range(layout.min_col, layout.max_col + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            if _has_border_xlsx(cell):
                bordered_cells.add((row_idx, col_idx))
    
    if not bordered_cells:
        return []
    
    # BFS로 인접한 테두리 셀들을 그룹화
    visited: Set[Tuple[int, int]] = set()
    regions: List[LayoutRange] = []
    
    # 위→아래, 왼쪽→오른쪽 순서로 정렬
    sorted_cells = sorted(bordered_cells, key=lambda x: (x[0], x[1]))
    
    for start_cell in sorted_cells:
        if start_cell in visited:
            continue
        
        # BFS
        group: Set[Tuple[int, int]] = set()
        queue = deque([start_cell])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            group.add(current)
            
            row, col = current
            # 상하좌우 인접 셀
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            
            for neighbor in neighbors:
                if neighbor in bordered_cells and neighbor not in visited:
                    queue.append(neighbor)
        
        # 그룹에서 사각형 영역 계산
        if group:
            min_r = min(r for r, c in group)
            max_r = max(r for r, c in group)
            min_c = min(c for r, c in group)
            max_c = max(c for r, c in group)
            regions.append(LayoutRange(min_row=min_r, max_row=max_r, min_col=min_c, max_col=max_c))
    
    return regions


def _detect_value_regions_xlsx(ws, layout: LayoutRange, exclude_regions: List[LayoutRange]) -> List[LayoutRange]:
    """
    XLSX 워크시트에서 값이 있는 영역들을 감지합니다 (테두리 영역 제외).
    병합 셀의 경우, 병합 영역의 일부가 layout에 포함되면 전체 영역을 감지합니다.
    
    Args:
        ws: openpyxl Worksheet 객체
        layout: 탐색할 레이아웃 범위
        exclude_regions: 제외할 영역 목록 (이미 감지된 테두리 영역)
    
    Returns:
        값이 있는 영역 목록
    """
    # 이미 감지된 영역에 포함된 셀인지 확인하는 함수
    def is_in_excluded(row: int, col: int) -> bool:
        for region in exclude_regions:
            if (region.min_row <= row <= region.max_row and
                region.min_col <= col <= region.max_col):
                return True
        return False
    
    # 병합 셀 정보 수집: 각 셀이 어떤 병합 영역에 속하는지
    merged_cell_map: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}  # (row, col) -> (min_row, max_row, min_col, max_col)
    for merged_range in ws.merged_cells.ranges:
        mr_min_row, mr_min_col = merged_range.min_row, merged_range.min_col
        mr_max_row, mr_max_col = merged_range.max_row, merged_range.max_col
        for r in range(mr_min_row, mr_max_row + 1):
            for c in range(mr_min_col, mr_max_col + 1):
                merged_cell_map[(r, c)] = (mr_min_row, mr_max_row, mr_min_col, mr_max_col)
    
    # 값이 있는 셀 좌표 수집 (제외 영역 외)
    value_cells: Set[Tuple[int, int]] = set()
    
    for row_idx in range(layout.min_row, layout.max_row + 1):
        for col_idx in range(layout.min_col, layout.max_col + 1):
            if is_in_excluded(row_idx, col_idx):
                continue
            
            cell = ws.cell(row=row_idx, column=col_idx)
            
            # 일반 셀: 값이 있으면 추가
            if cell.value is not None and str(cell.value).strip():
                value_cells.add((row_idx, col_idx))
            # 병합 셀의 일부인 경우: 병합 셀의 첫 번째 셀에 값이 있으면 이 셀도 추가
            elif (row_idx, col_idx) in merged_cell_map:
                mr_min_row, mr_max_row, mr_min_col, mr_max_col = merged_cell_map[(row_idx, col_idx)]
                # 병합 셀의 첫 번째 셀 값 확인
                first_cell = ws.cell(row=mr_min_row, column=mr_min_col)
                if first_cell.value is not None and str(first_cell.value).strip():
                    value_cells.add((row_idx, col_idx))
    
    if not value_cells:
        return []
    
    # BFS로 인접한 값 셀들을 그룹화
    visited: Set[Tuple[int, int]] = set()
    regions: List[LayoutRange] = []
    
    sorted_cells = sorted(value_cells, key=lambda x: (x[0], x[1]))
    
    for start_cell in sorted_cells:
        if start_cell in visited:
            continue
        
        group: Set[Tuple[int, int]] = set()
        queue = deque([start_cell])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            group.add(current)
            
            row, col = current
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            
            for neighbor in neighbors:
                if neighbor in value_cells and neighbor not in visited:
                    queue.append(neighbor)
        
        if group:
            min_r = min(r for r, c in group)
            max_r = max(r for r, c in group)
            min_c = min(c for r, c in group)
            max_c = max(c for r, c in group)
            regions.append(LayoutRange(min_row=min_r, max_row=max_r, min_col=min_c, max_col=max_c))
    
    return regions


def _merge_adjacent_regions(regions: List[LayoutRange]) -> List[LayoutRange]:
    """
    완전히 인접한 영역들을 병합합니다.
    반복적으로 인접한 영역을 찾아 병합합니다.
    
    Args:
        regions: 영역 목록
    
    Returns:
        병합된 영역 목록
    """
    if len(regions) <= 1:
        return regions
    
    merged = True
    current_regions = list(regions)
    
    while merged:
        merged = False
        new_regions = []
        used = set()
        
        for i, region_a in enumerate(current_regions):
            if i in used:
                continue
            
            merged_region = region_a
            
            for j, region_b in enumerate(current_regions):
                if j <= i or j in used:
                    continue
                
                if merged_region.is_adjacent(region_b):
                    merged_region = merged_region.merge_with(region_b)
                    used.add(j)
                    merged = True
            
            new_regions.append(merged_region)
            used.add(i)
        
        current_regions = new_regions
    
    return current_regions


def object_detect_xlsx(ws, layout: Optional[LayoutRange] = None) -> List[LayoutRange]:
    """
    XLSX 워크시트에서 개별 객체(테이블/데이터 블록)를 감지합니다.
    
    알고리즘:
    1. 테두리가 있는 영역을 먼저 개별 개체로 인식
    2. 테두리가 없는 값 영역을 감지
    3. 완전히 인접한 개체들을 병합
    4. 위→아래, 왼쪽→오른쪽 순서로 정렬하여 반환
    
    Args:
        ws: openpyxl Worksheet 객체
        layout: 탐색할 레이아웃 범위 (None이면 자동 감지)
    
    Returns:
        개별 객체 영역 목록
    """
    try:
        if layout is None:
            layout = layout_detect_range_xlsx(ws)
            if layout is None:
                return []
        
        # 1. 테두리 영역 감지
        bordered_regions = _detect_bordered_regions_xlsx(ws, layout)
        logger.debug(f"Detected {len(bordered_regions)} bordered regions")
        
        # 2. 값 영역 감지 (테두리 영역 제외)
        value_regions = _detect_value_regions_xlsx(ws, layout, bordered_regions)
        logger.debug(f"Detected {len(value_regions)} value regions (excluding bordered)")
        
        # 3. 모든 영역 합치기
        all_regions = bordered_regions + value_regions
        
        if not all_regions:
            return []
        
        # 4. 인접 영역 병합
        merged_regions = _merge_adjacent_regions(all_regions)
        logger.debug(f"After merging: {len(merged_regions)} regions")
        
        # 5. 위→아래, 왼쪽→오른쪽 순서로 정렬
        sorted_regions = sorted(merged_regions, key=lambda r: (r.min_row, r.min_col))
        
        for i, obj in enumerate(sorted_regions):
            logger.debug(
                f"  Object {i+1}: rows {obj.min_row}-{obj.max_row}, "
                f"cols {obj.min_col}-{obj.max_col} ({obj.cell_count()} cells)"
            )
        
        return sorted_regions
        
    except Exception as e:
        logger.warning(f"Error detecting objects in XLSX: {e}")
        return []


def _has_border_xls(sheet, wb, row_idx: int, col_idx: int) -> bool:
    """XLS 셀에 테두리가 있는지 확인 (0-based 인덱스)"""
    try:
        xf_index = sheet.cell_xf_index(row_idx, col_idx)
        xf = wb.xf_list[xf_index]
        
        # 테두리 인덱스 확인
        borders = [
            xf.border.top_line_style,
            xf.border.bottom_line_style,
            xf.border.left_line_style,
            xf.border.right_line_style
        ]
        
        for border_style in borders:
            if border_style and border_style > 0:
                return True
        return False
    except Exception:
        return False


def _detect_bordered_regions_xls(sheet, wb, layout: LayoutRange) -> List[LayoutRange]:
    """
    XLS 시트에서 테두리가 있는 영역들을 감지합니다.
    
    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 탐색할 레이아웃 범위 (1-based)
    
    Returns:
        테두리 영역 목록 (1-based)
    """
    bordered_cells: Set[Tuple[int, int]] = set()
    
    for row_idx in range(layout.min_row, layout.max_row + 1):
        for col_idx in range(layout.min_col, layout.max_col + 1):
            # XLS는 0-based
            if _has_border_xls(sheet, wb, row_idx - 1, col_idx - 1):
                bordered_cells.add((row_idx, col_idx))
    
    if not bordered_cells:
        return []
    
    visited: Set[Tuple[int, int]] = set()
    regions: List[LayoutRange] = []
    
    sorted_cells = sorted(bordered_cells, key=lambda x: (x[0], x[1]))
    
    for start_cell in sorted_cells:
        if start_cell in visited:
            continue
        
        group: Set[Tuple[int, int]] = set()
        queue = deque([start_cell])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            group.add(current)
            
            row, col = current
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            
            for neighbor in neighbors:
                if neighbor in bordered_cells and neighbor not in visited:
                    queue.append(neighbor)
        
        if group:
            min_r = min(r for r, c in group)
            max_r = max(r for r, c in group)
            min_c = min(c for r, c in group)
            max_c = max(c for r, c in group)
            regions.append(LayoutRange(min_row=min_r, max_row=max_r, min_col=min_c, max_col=max_c))
    
    return regions


def _detect_value_regions_xls(sheet, layout: LayoutRange, exclude_regions: List[LayoutRange]) -> List[LayoutRange]:
    """
    XLS 시트에서 값이 있는 영역들을 감지합니다 (테두리 영역 제외).
    
    Args:
        sheet: xlrd Sheet 객체
        layout: 탐색할 레이아웃 범위 (1-based)
        exclude_regions: 제외할 영역 목록
    
    Returns:
        값이 있는 영역 목록 (1-based)
    """
    def is_in_excluded(row: int, col: int) -> bool:
        for region in exclude_regions:
            if (region.min_row <= row <= region.max_row and
                region.min_col <= col <= region.max_col):
                return True
        return False
    
    value_cells: Set[Tuple[int, int]] = set()
    
    for row_idx in range(layout.min_row, layout.max_row + 1):
        for col_idx in range(layout.min_col, layout.max_col + 1):
            if is_in_excluded(row_idx, col_idx):
                continue
            try:
                # XLS는 0-based
                value = sheet.cell_value(row_idx - 1, col_idx - 1)
                if value is not None and str(value).strip():
                    value_cells.add((row_idx, col_idx))
            except Exception:
                pass
    
    if not value_cells:
        return []
    
    visited: Set[Tuple[int, int]] = set()
    regions: List[LayoutRange] = []
    
    sorted_cells = sorted(value_cells, key=lambda x: (x[0], x[1]))
    
    for start_cell in sorted_cells:
        if start_cell in visited:
            continue
        
        group: Set[Tuple[int, int]] = set()
        queue = deque([start_cell])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            group.add(current)
            
            row, col = current
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            
            for neighbor in neighbors:
                if neighbor in value_cells and neighbor not in visited:
                    queue.append(neighbor)
        
        if group:
            min_r = min(r for r, c in group)
            max_r = max(r for r, c in group)
            min_c = min(c for r, c in group)
            max_c = max(c for r, c in group)
            regions.append(LayoutRange(min_row=min_r, max_row=max_r, min_col=min_c, max_col=max_c))
    
    return regions


def object_detect_xls(sheet, wb, layout: Optional[LayoutRange] = None) -> List[LayoutRange]:
    """
    XLS 시트에서 개별 객체(테이블/데이터 블록)를 감지합니다.
    
    알고리즘:
    1. 테두리가 있는 영역을 먼저 개별 개체로 인식
    2. 테두리가 없는 값 영역을 감지
    3. 완전히 인접한 개체들을 병합
    4. 위→아래, 왼쪽→오른쪽 순서로 정렬하여 반환
    
    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 탐색할 레이아웃 범위 (None이면 자동 감지)
    
    Returns:
        개별 객체 영역 목록 (1-based 좌표)
    """
    try:
        if layout is None:
            layout = layout_detect_range_xls(sheet)
            if layout is None:
                return []
        
        # 1. 테두리 영역 감지
        bordered_regions = _detect_bordered_regions_xls(sheet, wb, layout)
        logger.debug(f"XLS: Detected {len(bordered_regions)} bordered regions")
        
        # 2. 값 영역 감지 (테두리 영역 제외)
        value_regions = _detect_value_regions_xls(sheet, layout, bordered_regions)
        logger.debug(f"XLS: Detected {len(value_regions)} value regions (excluding bordered)")
        
        # 3. 모든 영역 합치기
        all_regions = bordered_regions + value_regions
        
        if not all_regions:
            return []
        
        # 4. 인접 영역 병합
        merged_regions = _merge_adjacent_regions(all_regions)
        logger.debug(f"XLS: After merging: {len(merged_regions)} regions")
        
        # 5. 위→아래, 왼쪽→오른쪽 순서로 정렬
        sorted_regions = sorted(merged_regions, key=lambda r: (r.min_row, r.min_col))
        
        for i, obj in enumerate(sorted_regions):
            logger.debug(
                f"  XLS Object {i+1}: rows {obj.min_row}-{obj.max_row}, "
                f"cols {obj.min_col}-{obj.max_col} ({obj.cell_count()} cells)"
            )
        
        return sorted_regions
        
    except Exception as e:
        logger.warning(f"Error detecting objects in XLS: {e}")
        return []
