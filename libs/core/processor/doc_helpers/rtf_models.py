# service/document_processor/processor/doc_helpers/rtf_models.py
"""
RTF Parser 데이터 모델

RTF 파싱에 사용되는 데이터 클래스들을 정의합니다.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, NamedTuple, Tuple


class RTFCellInfo(NamedTuple):
    """RTF 셀 정보 (병합 정보 포함)"""
    text: str           # 셀 텍스트
    h_merge_first: bool  # 수평 병합 첫 번째 셀 (clmgf)
    h_merge_cont: bool   # 수평 병합 연속 셀 (clmrg)
    v_merge_first: bool  # 수직 병합 첫 번째 셀 (clvmgf)
    v_merge_cont: bool   # 수직 병합 연속 셀 (clvmrg)
    right_boundary: int  # 셀 오른쪽 경계 (twips)


@dataclass
class RTFTable:
    """RTF 테이블 구조 (병합 셀 지원)"""
    rows: List[List[RTFCellInfo]] = field(default_factory=list)
    col_count: int = 0
    position: int = 0  # 문서 내 시작 위치
    end_position: int = 0  # 문서 내 종료 위치
    _logical_cells: List[List[Optional[RTFCellInfo]]] = field(default_factory=list, repr=False)
    
    def get_effective_col_count(self) -> int:
        """
        실제 유효한 열 수를 계산합니다.
        빈 셀만 있는 열은 제외합니다.
        
        Returns:
            실제 내용이 있는 최대 열 수
        """
        if not self.rows:
            return 0
        
        effective_counts = []
        for row in self.rows:
            # 빈 셀과 병합된 셀을 제외한 유효 셀 수 계산
            non_empty_cells = []
            for i, cell in enumerate(row):
                # 병합으로 건너뛰는 셀 제외
                if cell.h_merge_cont:
                    continue
                # 내용이 있거나 수직 병합 시작인 경우 유효
                if cell.text.strip() or cell.v_merge_first:
                    non_empty_cells.append(i)
            
            if non_empty_cells:
                # 마지막 유효 셀의 인덱스 + 1
                effective_counts.append(max(non_empty_cells) + 1)
        
        return max(effective_counts) if effective_counts else 0
    
    def is_real_table(self) -> bool:
        """
        실제 테이블인지 판단합니다.
        
        n rows × 1 column 형태는 테이블이 아닌 단순 리스트로 간주합니다.
        빈 셀만 있는 열은 열 수에서 제외합니다.
        
        Returns:
            True if 실제 테이블 (유효 열이 2개 이상), False otherwise
        """
        if not self.rows:
            return False
        
        # 유효 열 수로 판단
        effective_cols = self.get_effective_col_count()
        return effective_cols >= 2
    
    def _calculate_merge_info(self) -> List[List[Tuple[int, int]]]:
        """
        각 셀의 colspan, rowspan을 계산합니다.
        
        RTF의 병합 처리:
        1. 명시적 병합 플래그 (clmgf/clmrg, clvmgf/clvmrg) 사용
        2. 열 경계(cellx) 값을 기반으로 암시적 colspan 계산
           - 테이블 전체의 고유 열 경계를 수집
           - 각 행의 셀이 몇 개의 논리적 열을 차지하는지 계산
        
        Returns:
            각 셀별 (colspan, rowspan) 정보 2D 리스트
            (0, 0)은 이 셀이 다른 셀에 병합되어 건너뛰어야 함을 의미
        """
        if not self.rows:
            return []
        
        num_rows = len(self.rows)
        
        # 1단계: 전체 테이블의 고유 열 경계 수집
        all_boundaries = set()
        for row in self.rows:
            for cell in row:
                if cell.right_boundary > 0:
                    all_boundaries.add(cell.right_boundary)
        
        # 정렬된 열 경계 리스트
        sorted_boundaries = sorted(all_boundaries)
        total_logical_cols = len(sorted_boundaries)
        
        if total_logical_cols == 0:
            # 열 경계 정보가 없으면 기본 처리
            max_cols = max(len(row) for row in self.rows) if self.rows else 0
            return [[(1, 1) for _ in range(max_cols)] for _ in range(num_rows)]
        
        # 경계값 -> 논리적 열 인덱스 매핑
        boundary_to_col = {b: i for i, b in enumerate(sorted_boundaries)}
        
        # 2단계: 각 행별로 셀의 colspan 계산
        # merge_info[row][logical_col] = (colspan, rowspan) 또는 (0, 0)
        merge_info = [[None for _ in range(total_logical_cols)] for _ in range(num_rows)]
        
        for row_idx, row in enumerate(self.rows):
            prev_boundary = 0
            for cell in row:
                if cell.right_boundary <= 0:
                    continue
                
                # 이 셀이 차지하는 논리적 열 범위 계산
                start_col = 0
                for i, b in enumerate(sorted_boundaries):
                    if b <= prev_boundary:
                        start_col = i + 1
                    else:
                        break
                
                end_col = boundary_to_col[cell.right_boundary]
                colspan = end_col - start_col + 1
                
                if colspan <= 0:
                    colspan = 1
                
                # 시작 열에 셀 정보 기록
                if start_col < total_logical_cols:
                    merge_info[row_idx][start_col] = (colspan, 1, cell)
                    # 병합된 열들은 (0, 0)으로 표시
                    for col in range(start_col + 1, start_col + colspan):
                        if col < total_logical_cols:
                            merge_info[row_idx][col] = (0, 0, None)
                
                prev_boundary = cell.right_boundary
        
        # 3단계: 수직 병합 (rowspan) 처리
        for col_idx in range(total_logical_cols):
            row_idx = 0
            while row_idx < num_rows:
                info = merge_info[row_idx][col_idx]
                if info is None or len(info) < 3 or info[2] is None:
                    row_idx += 1
                    continue
                
                colspan, _, cell = info
                if colspan == 0:
                    row_idx += 1
                    continue
                
                if cell.v_merge_first:
                    # 수직 병합 시작
                    rowspan = 1
                    for next_row in range(row_idx + 1, num_rows):
                        next_info = merge_info[next_row][col_idx]
                        if next_info is None or len(next_info) < 3 or next_info[2] is None:
                            break
                        _, _, next_cell = next_info
                        if next_cell.v_merge_cont:
                            rowspan += 1
                            merge_info[next_row][col_idx] = (0, 0, None)
                        else:
                            break
                    
                    merge_info[row_idx][col_idx] = (colspan, rowspan, cell)
                    row_idx += rowspan
                elif cell.v_merge_cont:
                    merge_info[row_idx][col_idx] = (0, 0, None)
                    row_idx += 1
                else:
                    row_idx += 1
        
        # 4단계: 최종 결과 (colspan, rowspan)만 반환
        result = []
        for row_idx in range(num_rows):
            row_result = []
            for col_idx in range(total_logical_cols):
                info = merge_info[row_idx][col_idx]
                if info is None:
                    row_result.append((1, 1))
                elif len(info) >= 2:
                    row_result.append((info[0], info[1]))
                else:
                    row_result.append((1, 1))
            result.append(row_result)
        
        # 실제 셀 데이터도 저장 (to_html에서 사용)
        self._logical_cells = []
        for row_idx in range(num_rows):
            row_cells = []
            for col_idx in range(total_logical_cols):
                info = merge_info[row_idx][col_idx]
                if info is not None and len(info) >= 3 and info[2] is not None:
                    row_cells.append(info[2])
                else:
                    row_cells.append(None)
            self._logical_cells.append(row_cells)
        
        return result
    
    def to_html(self) -> str:
        """테이블을 HTML로 변환 (병합 셀 지원)"""
        if not self.rows:
            return ""
        
        merge_info = self._calculate_merge_info()
        
        # _logical_cells가 없으면 기존 방식 사용
        if not hasattr(self, '_logical_cells') or not self._logical_cells:
            return self._to_html_legacy(merge_info)
        
        html_parts = ['<table border="1">']
        
        for row_idx, row_merge in enumerate(merge_info):
            html_parts.append('<tr>')
            
            for col_idx, (colspan, rowspan) in enumerate(row_merge):
                if colspan == 0 or rowspan == 0:
                    continue
                
                cell = self._logical_cells[row_idx][col_idx] if col_idx < len(self._logical_cells[row_idx]) else None
                cell_text = cell.text if cell and cell.text else ''
                
                attrs = []
                if colspan > 1:
                    attrs.append(f'colspan="{colspan}"')
                if rowspan > 1:
                    attrs.append(f'rowspan="{rowspan}"')
                
                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                html_parts.append(f'<td{attr_str}>{cell_text}</td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        return '\n'.join(html_parts)
    
    def _to_html_legacy(self, merge_info: List[List[Tuple[int, int]]]) -> str:
        """기존 HTML 변환 (열 경계 정보 없을 때)"""
        html_parts = ['<table border="1">']
        
        for row_idx, row in enumerate(self.rows):
            html_parts.append('<tr>')
            
            for col_idx, cell in enumerate(row):
                # 병합 정보 확인
                if col_idx < len(merge_info[row_idx]) and merge_info[row_idx][col_idx]:
                    colspan, rowspan = merge_info[row_idx][col_idx]
                    
                    if colspan == 0 and rowspan == 0:
                        # 이 셀은 다른 셀에 병합됨 - 건너뜀
                        continue
                    
                    # 셀 내용 정리
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    
                    # 속성 생성
                    attrs = []
                    if colspan > 1:
                        attrs.append(f'colspan="{colspan}"')
                    if rowspan > 1:
                        attrs.append(f'rowspan="{rowspan}"')
                    
                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    html_parts.append(f'<td{attr_str}>{cell_text}</td>')
                else:
                    # 병합 정보 없음 - 일반 셀
                    cell_text = re.sub(r'\s+', ' ', cell.text).strip()
                    html_parts.append(f'<td>{cell_text}</td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        return '\n'.join(html_parts)
    
    def to_text_list(self) -> str:
        """
        1열 테이블을 텍스트 리스트로 변환합니다.
        
        - 1×1 테이블: 셀 내용만 반환 (컨테이너 테이블)
        - n×1 테이블: 각 행을 빈 줄로 구분하여 반환
        
        Returns:
            텍스트 형식의 문자열
        """
        if not self.rows:
            return ""
        
        # 1×1 테이블: 셀 내용만 반환 (컨테이너 테이블)
        if len(self.rows) == 1 and len(self.rows[0]) == 1:
            return self.rows[0][0].text
        
        lines = []
        for row in self.rows:
            if row:
                # 첫 번째 셀만 사용 (1열 테이블)
                cell_text = row[0].text
                if cell_text:
                    lines.append(cell_text)
        
        # 빈 줄로 구분
        return '\n\n'.join(lines)


@dataclass
class RTFContentPart:
    """문서 내 콘텐츠 조각 (텍스트 또는 테이블)"""
    content_type: str  # "text" 또는 "table"
    position: int      # 원본 문서 내 위치
    text: str = ""     # content_type이 "text"인 경우
    table: Optional['RTFTable'] = None  # content_type이 "table"인 경우


@dataclass
class RTFDocument:
    """RTF 문서 구조"""
    text_content: str = ""
    tables: List[RTFTable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[bytes] = field(default_factory=list)
    image_tags: List[str] = field(default_factory=list)  # v3: MinIO 업로드된 이미지 태그
    encoding: str = "cp949"
    # v2: 인라인 콘텐츠 - 원래 순서대로 정렬된 콘텐츠 조각들
    content_parts: List[RTFContentPart] = field(default_factory=list)
    
    def get_inline_content(self) -> str:
        """
        테이블이 원래 위치에 인라인으로 배치된 전체 콘텐츠를 반환합니다.
        
        Returns:
            인라인 배치된 전체 텍스트
        """
        if not self.content_parts:
            # 호환성: content_parts가 없으면 기존 방식으로 반환
            return self.text_content
        
        # 위치순 정렬
        sorted_parts = sorted(self.content_parts, key=lambda p: p.position)
        
        result_parts = []
        for part in sorted_parts:
            if part.content_type == "text" and part.text.strip():
                result_parts.append(part.text)
            elif part.content_type == "table" and part.table:
                if part.table.is_real_table():
                    result_parts.append(part.table.to_html())
                else:
                    text_list = part.table.to_text_list()
                    if text_list:
                        result_parts.append(text_list)
        
        return '\n\n'.join(result_parts)
