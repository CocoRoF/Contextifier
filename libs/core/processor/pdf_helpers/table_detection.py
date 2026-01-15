"""
Table Detection Engine for PDF Handler V3

다중 전략을 사용하여 테이블을 감지하고 최선의 결과를 선택합니다.
V3.1 고도화로 그래픽 영역 제외 및 가짜 테이블 필터링 기능 포함.
V3.3 셀 추출 정확도 개선.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Set

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from .v3_types import (
    V3Config,
    TableDetectionStrategy,
    GridInfo,
    CellInfo,
    TableCandidate,
)
from .line_analysis import LineAnalysisEngine
from .graphic_detector import GraphicRegionDetector
from .table_validator import TableQualityValidator

logger = logging.getLogger(__name__)


# ============================================================================
# Table Detection Engine
# ============================================================================

class TableDetectionEngine:
    """
    테이블 감지 엔진 (V3.1 고도화)

    다중 전략을 사용하여 테이블을 감지하고 최선의 결과를 선택합니다.

    V3.1 개선사항:
    - GraphicRegionDetector 통합으로 벡터 그래픽 영역 제외
    - TableQualityValidator 통합으로 가짜 테이블 필터링

    지원 전략:
    1. PyMuPDF find_tables() - 가장 정확, 우선 사용
    2. pdfplumber - 선 기반 감지
    3. Line-based - 직접 선 분석
    """

    # 설정 상수
    CONFIDENCE_THRESHOLD = getattr(V3Config, 'CONFIDENCE_THRESHOLD', 0.5)
    MIN_TABLE_ROWS = getattr(V3Config, 'MIN_TABLE_ROWS', 2)
    MIN_TABLE_COLS = getattr(V3Config, 'MIN_TABLE_COLS', 2)

    def __init__(self, page, page_num: int, file_path: str):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed)
            file_path: PDF 파일 경로
        """
        self.page = page
        self.page_num = page_num
        self.file_path = file_path
        self.page_width = page.rect.width
        self.page_height = page.rect.height

        # 선 분석 엔진
        self.line_engine = LineAnalysisEngine(page, self.page_width, self.page_height)
        self.h_lines, self.v_lines = self.line_engine.analyze()

        # V3.1: 그래픽 영역 감지기
        self.graphic_detector = GraphicRegionDetector(page, page_num)
        self.graphic_regions = self.graphic_detector.detect()

        # V3.1: 테이블 품질 검증기
        self.quality_validator = TableQualityValidator(page, self.graphic_detector)

    def detect_tables(self) -> List[TableCandidate]:
        """
        모든 전략을 사용하여 테이블 감지

        Returns:
            신뢰도 순으로 정렬된 테이블 후보 목록
        """
        candidates: List[TableCandidate] = []

        # Strategy 1: PyMuPDF
        pymupdf_candidates = self._detect_with_pymupdf()

        # V3.2: 인접 헤더-데이터 테이블 사전 병합 (검증 전)
        pymupdf_candidates = self._merge_header_data_tables(pymupdf_candidates)
        candidates.extend(pymupdf_candidates)

        # Strategy 2: pdfplumber
        if PDFPLUMBER_AVAILABLE:
            pdfplumber_candidates = self._detect_with_pdfplumber()
            pdfplumber_candidates = self._merge_header_data_tables(pdfplumber_candidates)
            candidates.extend(pdfplumber_candidates)

        # Strategy 3: Line-based (HYBRID_ANALYSIS)
        # V3.2: PyMuPDF와 pdfplumber가 테이블을 찾지 못한 경우에만 사용
        # 또는 찾았더라도 추가로 사용하되, 더 엄격한 검증 적용
        line_candidates = self._detect_with_lines()

        # V3.2: HYBRID 결과에 대한 교차 검증 강화
        if line_candidates and not pymupdf_candidates:
            # PyMuPDF가 테이블을 찾지 못했는데 HYBRID가 찾은 경우
            # 더 높은 신뢰도 임계값 적용 (0.65 이상)
            line_candidates = [
                c for c in line_candidates
                if c.confidence >= 0.65
            ]
            logger.debug(f"[TableDetection] HYBRID-only detection: "
                        f"{len(line_candidates)} candidates passed stricter threshold (0.65)")

        candidates.extend(line_candidates)

        # V3.1: 품질 검증을 통한 가짜 테이블 필터링
        validated_candidates = self._validate_candidates(candidates)

        # 신뢰도 기반 최선의 후보 선택
        selected = self._select_best_candidates(validated_candidates)

        return selected

    def _merge_header_data_tables(self, candidates: List[TableCandidate]) -> List[TableCandidate]:
        """
        V3.2: 인접한 헤더-데이터 테이블을 병합합니다.

        조건:
        1. 첫 번째 테이블이 1-2행 (헤더로 추정)
        2. 두 번째 테이블이 바로 아래에 있음 (Y gap < 30pt)
        3. X 범위가 유사함 (80% 이상 겹침)
        4. 열 수 관계: 헤더 열 수 <= 데이터 열 수
        """
        if len(candidates) < 2:
            return candidates

        # Y 위치로 정렬
        sorted_candidates = sorted(candidates, key=lambda c: c.bbox[1])
        merged = []
        skip_indices = set()

        for i, header_cand in enumerate(sorted_candidates):
            if i in skip_indices:
                continue

            # 헤더 후보 조건 확인 (1-2행)
            if len(header_cand.data) > 2:
                merged.append(header_cand)
                continue

            # 다음 테이블과 병합 가능한지 확인
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
        """헤더와 데이터 테이블이 병합 가능한지 판단"""
        # Y gap 확인
        y_gap = data.bbox[1] - header.bbox[3]
        if y_gap < -5 or y_gap > 40:  # 약간의 겹침 허용, 최대 40pt 간격
            return False

        # X 범위 겹침 확인
        x_overlap_start = max(header.bbox[0], data.bbox[0])
        x_overlap_end = min(header.bbox[2], data.bbox[2])
        x_overlap = max(0, x_overlap_end - x_overlap_start)

        header_width = header.bbox[2] - header.bbox[0]
        data_width = data.bbox[2] - data.bbox[0]
        max_width = max(header_width, data_width)

        if max_width > 0 and x_overlap / max_width < 0.7:
            return False

        # 열 수 관계 확인
        header_cols = max(len(row) for row in header.data) if header.data else 0
        data_cols = max(len(row) for row in data.data) if data.data else 0

        # 헤더 열 수가 데이터 열 수보다 많으면 병합하지 않음
        if header_cols > data_cols + 1:
            return False

        return True

    def _do_merge_header_data(self, header: TableCandidate, data: TableCandidate) -> TableCandidate:
        """헤더와 데이터 테이블 병합 수행 (서브헤더 감지 포함)"""
        # 새 bbox
        merged_bbox = (
            min(header.bbox[0], data.bbox[0]),
            header.bbox[1],
            max(header.bbox[2], data.bbox[2]),
            data.bbox[3]
        )

        # 열 수 결정
        header_cols = max(len(row) for row in header.data) if header.data else 0
        data_cols = max(len(row) for row in data.data) if data.data else 0
        merged_cols = max(header_cols, data_cols)

        # 헤더와 데이터 사이의 서브헤더 감지
        subheader_row = self._detect_subheader_between(header, data, merged_cols)

        # 데이터 병합
        merged_data = []
        merged_cells = []

        # 헤더 행 처리
        for row_idx, row in enumerate(header.data):
            if len(row) < merged_cols:
                # 헤더의 열이 적으면 colspan 처리
                adjusted_row = list(row)
                col_diff = merged_cols - len(row)

                # 두 번째 열(시험결과)에 colspan 적용
                if len(row) >= 2 and col_diff > 0:
                    # colspan 정보 저장
                    merged_cells.append({
                        'row': row_idx,
                        'col': 1,
                        'rowspan': 1,
                        'colspan': 1 + col_diff,
                        'bbox': None
                    })
                    # 빈 열 추가
                    for _ in range(col_diff):
                        adjusted_row.insert(2, '')
                else:
                    adjusted_row.extend([''] * col_diff)

                merged_data.append(adjusted_row)
            else:
                merged_data.append(list(row))

        # 서브헤더 행 삽입 (헤더 셀 정보)
        header_row_count = len(header.data)
        if subheader_row:
            merged_data.append(subheader_row)
            # 서브헤더 행에 대한 셀 정보 추가 (각 셀은 colspan=1)
            subheader_row_idx = header_row_count  # 헤더 다음 행
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

        # 헤더 셀 정보
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

        # 데이터 행 처리
        for row_idx, row in enumerate(data.data):
            if len(row) < merged_cols:
                adjusted_row = list(row) + [''] * (merged_cols - len(row))
            else:
                adjusted_row = list(row)
            merged_data.append(adjusted_row)

        # 데이터 셀 정보 (행 offset 적용)
        if data.cells:
            for cell in data.cells:
                merged_cells.append({
                    'row': cell.row + header_row_count,
                    'col': cell.col,
                    'rowspan': cell.rowspan,
                    'colspan': cell.colspan,
                    'bbox': cell.bbox
                })

        # 셀 정보를 CellInfo 객체로 변환
        cell_objects = [
            CellInfo(
                row=c['row'],
                col=c['col'],
                rowspan=c.get('rowspan', 1),
                colspan=c.get('colspan', 1),
                # bbox가 None이거나 없을 경우 기본값 사용
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
        헤더와 데이터 테이블 사이의 서브헤더 행을 감지합니다.

        예: (A), (B) 등의 하위 컬럼 헤더
        """
        header_bottom = header.bbox[3]
        data_top = data.bbox[1]

        # 헤더와 데이터 사이에 충분한 간격이 있어야 함
        gap = data_top - header_bottom
        if gap < 5 or gap > 50:
            return None

        # 페이지에서 해당 영역의 텍스트 추출
        page_dict = self.page.get_text("dict", sort=True)

        subheader_texts = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_bbox = line.get("bbox", (0, 0, 0, 0))
                line_y = (line_bbox[1] + line_bbox[3]) / 2

                # 헤더와 데이터 사이에 위치하는지 확인
                if header_bottom - 5 <= line_y <= data_top + 5:
                    # 테이블 X 범위 내에 있는지 확인
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

        # 서브헤더 패턴 확인: (A), (B) 등
        has_subheader_pattern = any('(' in t['text'] and ')' in t['text'] for t in subheader_texts)
        if not has_subheader_pattern:
            return None

        # 서브헤더 행 구성
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

        # 서브헤더가 유효한지 확인 (최소 하나의 (A), (B) 패턴이 있어야 함)
        valid_count = sum(1 for s in subheader_row if '(' in s and ')' in s)
        if valid_count < 1:
            return None

        return subheader_row

    def _validate_candidates(self, candidates: List[TableCandidate]) -> List[TableCandidate]:
        """
        V3.1: 테이블 후보들을 품질 검증합니다.

        검증 기준:
        1. 그래픽 영역과 겹치지 않는지 (PyMuPDF 제외 - 텍스트 기반으로 신뢰도 높음)
        2. 채워진 셀 비율이 충분한지
        3. 의미 있는 데이터가 있는지

        V3.2: PyMuPDF 전략으로 감지된 테이블은 그래픽 영역 체크를 건너뜁니다.
        이유: PyMuPDF는 텍스트 기반으로 테이블을 감지하므로, 배경색이 있는 셀이
        그래픽으로 오인되는 경우에도 정확하게 테이블을 인식합니다.
        """
        validated = []

        for candidate in candidates:
            # V3.2: PyMuPDF 전략은 그래픽 영역 체크를 건너뜀
            skip_graphic_check = (candidate.strategy == TableDetectionStrategy.PYMUPDF_NATIVE)

            is_valid, new_confidence, reason = self.quality_validator.validate(
                data=candidate.data,
                bbox=candidate.bbox,
                cells_info=candidate.cells,
                skip_graphic_check=skip_graphic_check  # V3.2: 새 파라미터
            )

            if is_valid:
                # 검증 결과로 신뢰도 조정
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
        """PyMuPDF find_tables() 사용 (V3.5: tolerance 설정으로 이중선 문제 해결)"""
        candidates = []

        if not hasattr(self.page, 'find_tables'):
            return candidates

        try:
            # V3.5: pdf_handler.py와 동일한 tolerance 설정 적용
            # 이중선/삼중선 테두리로 인한 가짜 열 생성 문제를 해결
            # snap_tolerance: 근접한 좌표들을 스냅하여 하나로 처리
            # join_tolerance: 근접한 선들을 하나로 합침
            # edge_min_length: 짧은 선(테두리 선)을 무시
            # intersection_tolerance: 교차점 판단 허용 오차
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

                    # V3.4: 좁은 열 병합 처리
                    merged_data, col_mapping = self._merge_narrow_columns(
                        table_data, table.cells if hasattr(table, 'cells') else None
                    )

                    # 신뢰도 계산 (병합된 데이터로)
                    confidence = self._calculate_pymupdf_confidence(table, merged_data)

                    if confidence < self.CONFIDENCE_THRESHOLD:
                        continue

                    # 셀 정보 추출 (col_mapping 적용)
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
                    logger.debug(f"[PDF-V3] PyMuPDF table extraction error: {e}")
                    continue

        except Exception as e:
            logger.debug(f"[PDF-V3] PyMuPDF find_tables error: {e}")

        return candidates

    def _merge_narrow_columns(
        self,
        data: List[List],
        cells: List[Tuple] = None,
        min_col_width: float = 15.0
    ) -> Tuple[List[List[str]], Dict[int, int]]:
        """
        V3.4: 좁은 열들을 인접 열과 병합합니다.

        PDF의 이중선/삼중선 테두리로 인해 생성된 가짜 열들을 제거합니다.

        Args:
            data: 테이블 데이터
            cells: PyMuPDF 셀 bbox 리스트
            min_col_width: 최소 열 너비 (pt)

        Returns:
            (병합된 데이터, 원본 열 -> 새 열 매핑)
        """
        if not data or not data[0]:
            return data, {}

        num_cols = max(len(row) for row in data)

        # 셀 정보가 없으면 텍스트 기반으로 열 분석
        if not cells:
            return self._merge_columns_by_content(data)

        # 열별 너비 계산
        col_widths = self._calculate_column_widths(cells, num_cols)

        # 병합할 열 그룹 결정
        col_groups = self._determine_column_groups(col_widths, min_col_width)

        if len(col_groups) == num_cols:
            # 병합 필요 없음
            return data, {i: i for i in range(num_cols)}

        # 열 매핑 생성
        col_mapping = {}
        for new_idx, group in enumerate(col_groups):
            for old_idx in group:
                col_mapping[old_idx] = new_idx

        # 데이터 병합
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
        """셀 bbox에서 열별 너비 계산"""
        if not cells:
            return [0.0] * num_cols

        # X 좌표 수집
        x_coords = sorted(set([c[0] for c in cells if c] + [c[2] for c in cells if c]))

        if len(x_coords) < 2:
            return [0.0] * num_cols

        # 열 너비 계산
        widths = []
        for i in range(len(x_coords) - 1):
            widths.append(x_coords[i + 1] - x_coords[i])

        # num_cols와 맞추기
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
        열 너비를 기반으로 병합할 열 그룹을 결정합니다.

        좁은 열들은 다음 넓은 열과 병합됩니다.
        """
        groups = []
        current_group = []

        for idx, width in enumerate(col_widths):
            current_group.append(idx)

            # 현재 그룹의 총 너비가 최소 너비 이상이면 그룹 확정
            group_width = sum(col_widths[i] for i in current_group)

            if group_width >= min_width:
                groups.append(current_group)
                current_group = []

        # 마지막 그룹 처리
        if current_group:
            if groups:
                # 이전 그룹에 합치기
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)

        return groups

    def _merge_columns_by_content(self, data: List[List]) -> Tuple[List[List[str]], Dict[int, int]]:
        """
        텍스트 내용을 기반으로 빈 열들을 병합합니다.

        대부분의 행에서 빈 열은 인접 열과 병합됩니다.
        """
        if not data or not data[0]:
            return data, {}

        num_cols = max(len(row) for row in data)
        num_rows = len(data)

        # 각 열의 "비어있음" 비율 계산
        empty_ratios = []
        for col_idx in range(num_cols):
            empty_count = 0
            for row in data:
                if col_idx >= len(row) or not row[col_idx] or not str(row[col_idx]).strip():
                    empty_count += 1
            empty_ratios.append(empty_count / num_rows if num_rows > 0 else 1.0)

        # 빈 비율이 90% 이상인 열을 찾아 인접 열과 병합
        groups = []
        current_group = []

        for col_idx, empty_ratio in enumerate(empty_ratios):
            current_group.append(col_idx)

            # 비어있지 않은 열이면 그룹 확정
            if empty_ratio < 0.9:
                groups.append(current_group)
                current_group = []

        # 마지막 그룹 처리
        if current_group:
            if groups:
                groups[-1].extend(current_group)
            else:
                groups.append(current_group)

        if len(groups) == num_cols:
            return data, {i: i for i in range(num_cols)}

        # 열 매핑 생성
        col_mapping = {}
        for new_idx, group in enumerate(groups):
            for old_idx in group:
                col_mapping[old_idx] = new_idx

        # 데이터 병합
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
        V3.4: 열 매핑을 적용하여 셀 정보 추출
        """
        if not col_mapping:
            return self._extract_cells_from_pymupdf(table)

        cells = self._extract_cells_from_pymupdf(table)

        if not cells:
            return cells

        # 매핑된 열 수 계산
        new_col_count = max(col_mapping.values()) + 1 if col_mapping else 0

        # 셀 정보 재매핑
        remapped_cells = []
        processed_positions = set()

        for cell in cells:
            old_col = cell.col
            new_col = col_mapping.get(old_col, old_col)

            # 같은 위치에 이미 셀이 있으면 colspan 확장
            if (cell.row, new_col) in processed_positions:
                continue

            # colspan 재계산: 병합된 열들을 고려
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

            # 커버된 위치 기록
            for r in range(cell.row, cell.row + cell.rowspan):
                for c in range(new_col, new_col + max(1, new_colspan)):
                    processed_positions.add((r, c))

        return remapped_cells

    def _calculate_pymupdf_confidence(self, table, data: List[List]) -> float:
        """
        PyMuPDF 결과 신뢰도 계산 (V3.3 개선)

        V3.3 개선:
        - 기본 점수 상향 (PyMuPDF 결과 신뢰)
        - 패널티 완화
        - 셀 정보 보너스 강화
        """
        score = 0.0

        # 기본 점수 상향 (PyMuPDF는 신뢰도 높음)
        score += 0.6

        # 행/열 수에 따른 점수
        num_rows = len(data)
        if num_rows >= self.MIN_TABLE_ROWS:
            score += 0.1
        if table.col_count >= self.MIN_TABLE_COLS:
            score += 0.1

        # 데이터 밀도에 따른 점수 (V3.3: 패널티 완화)
        total_cells = sum(len(row) for row in data)
        filled_cells = sum(1 for row in data for cell in row if cell and str(cell).strip())

        if total_cells > 0:
            density = filled_cells / total_cells

            if density < 0.05:
                # 매우 낮은 밀도에서만 패널티
                score -= 0.2
            elif density < 0.1:
                score -= 0.1
            else:
                score += density * 0.15

        # 셀 정보가 있으면 추가 점수 (V3.3: 보너스 강화)
        if hasattr(table, 'cells') and table.cells:
            score += 0.15

        # 의미 있는 셀 수 체크 (V3.3: 패널티 완화)
        meaningful_count = sum(
            1 for row in data for cell in row
            if cell and len(str(cell).strip()) >= 2
        )

        if meaningful_count < 2:
            score -= 0.1

        # 유효 행 수 체크 (V3.3: 패널티 완화)
        valid_rows = sum(1 for row in data if any(cell and str(cell).strip() for cell in row))
        if valid_rows <= 1:
            score -= 0.1

        # 그래픽 영역 겹침 확인 (V3.3: 패널티 완화)
        if self.graphic_detector:
            if self.graphic_detector.is_bbox_in_graphic_region(table.bbox, threshold=0.5):
                score -= 0.15

        return max(0.0, min(1.0, score))

    def _extract_cells_from_pymupdf(self, table) -> List[CellInfo]:
        """
        PyMuPDF 테이블에서 셀 정보 추출 (V3.4 개선)

        pdf_handler_default의 _extract_cell_spans_from_table() 로직 적용:
        1. table.cells에서 각 셀의 물리적 bbox 추출
        2. Y 좌표를 행 인덱스로, X 좌표를 열 인덱스로 매핑
        3. 셀 bbox가 여러 그리드 셀을 차지하면 rowspan/colspan 계산
        """
        cells = []

        if not hasattr(table, 'cells') or not table.cells:
            # 셀 정보가 없으면 빈 리스트 반환 (CellAnalysisEngine에서 처리)
            return cells

        raw_cells = table.cells
        if not raw_cells:
            return cells

        # X, Y 경계선 추출 (pdf_handler_default와 동일한 방식)
        x_coords = sorted(set([c[0] for c in raw_cells if c] + [c[2] for c in raw_cells if c]))
        y_coords = sorted(set([c[1] for c in raw_cells if c] + [c[3] for c in raw_cells if c]))

        if len(x_coords) < 2 or len(y_coords) < 2:
            # 그리드를 구성할 수 없으면 기본 셀 정보 반환
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

        # 좌표를 그리드 인덱스로 매핑하는 함수 (pdf_handler_default와 동일)
        def coord_to_index(coord: float, coords: List[float], tolerance: float = 3.0) -> int:
            for i, c in enumerate(coords):
                if abs(coord - c) <= tolerance:
                    return i
            # 가장 가까운 인덱스 반환
            return min(range(len(coords)), key=lambda i: abs(coords[i] - coord))

        # 처리된 그리드 위치 추적
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

            # 병합된 영역의 다른 셀들 마킹
            for r in range(row_start, row_start + rowspan):
                for c in range(col_start, col_start + colspan):
                    if (r, c) != (row_start, col_start):
                        processed_positions.add((r, c))

        return cells

    def _cluster_grid_positions(self, positions: List[float], tolerance: float = 3.0) -> List[float]:
        """
        V3.3: 그리드 위치 클러스터링

        근접한 라인들을 하나로 병합합니다.
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

        # 각 클러스터의 평균값 반환
        return [sum(c) / len(c) for c in clusters]

    def _find_grid_index_v2(self, value: float, grid_lines: List[float],
                            tolerance: float = 5.0) -> Optional[int]:
        """
        V3.3: 그리드 라인에서 값의 인덱스 찾기 (개선 버전)

        정확한 매칭이 실패하면 가장 가까운 라인 선택
        """
        if not grid_lines:
            return None

        # 정확한 매칭 시도
        for i, line in enumerate(grid_lines):
            if abs(value - line) <= tolerance:
                return i

        # 가장 가까운 라인 찾기
        min_diff = float('inf')
        closest_idx = 0

        for i, line in enumerate(grid_lines):
            diff = abs(value - line)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        # 허용 오차의 3배 이내면 반환
        if min_diff <= tolerance * 3:
            return closest_idx

        return None

    def _find_grid_index(self, value: float, grid_lines: List[float], tolerance: float = 3.0) -> Optional[int]:
        """그리드 라인에서 값의 인덱스 찾기 (호환성 유지)"""
        return self._find_grid_index_v2(value, grid_lines, tolerance)

    def _detect_with_pdfplumber(self) -> List[TableCandidate]:
        """pdfplumber 사용"""
        candidates = []

        try:
            with pdfplumber.open(self.file_path) as pdf:
                if self.page_num >= len(pdf.pages):
                    return candidates

                plumber_page = pdf.pages[self.page_num]

                # 테이블 설정
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

                    # bbox 추정
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
            logger.debug(f"[PDF-V3] pdfplumber error: {e}")

        return candidates

    def _estimate_table_bbox_pdfplumber(self, page, data: List[List]) -> Optional[Tuple[float, float, float, float]]:
        """pdfplumber 테이블 bbox 추정"""
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
        """pdfplumber 결과 신뢰도 계산"""
        score = 0.0

        # 기본 점수 (PyMuPDF보다 약간 낮음)
        score += 0.4

        num_rows = len(data)
        col_count = max(len(row) for row in data) if data else 0

        if num_rows >= self.MIN_TABLE_ROWS:
            score += 0.1
        if col_count >= self.MIN_TABLE_COLS:
            score += 0.1

        # 데이터 밀도
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

        # 의미 있는 셀 수
        meaningful_count = sum(
            1 for row in data for cell in row
            if cell and len(str(cell).strip()) >= 2
        )

        if meaningful_count < 2:
            score -= 0.3

        # 유효 행 수
        valid_rows = sum(1 for row in data if any(cell and str(cell).strip() for cell in row))
        if valid_rows <= 1:
            score -= 0.2

        # 빈 행 비율
        empty_rows = num_rows - valid_rows
        if num_rows > 0 and empty_rows / num_rows > 0.5:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _detect_with_lines(self) -> List[TableCandidate]:
        """선 분석 기반 테이블 감지"""
        candidates = []

        # 그리드 구성
        grid = self.line_engine.build_grid()

        if not grid:
            return candidates

        # 불완전 테두리 복구
        if not grid.is_complete:
            grid = self.line_engine.reconstruct_incomplete_border(grid)
            if not grid.is_complete:
                return candidates

        # 그리드가 유효한지 확인
        if grid.row_count < self.MIN_TABLE_ROWS or grid.col_count < self.MIN_TABLE_COLS:
            return candidates

        # 셀 내 텍스트 추출
        data = self._extract_text_from_grid(grid)

        if not data or not any(any(cell for cell in row if cell) for row in data):
            return candidates

        # 셀 정보 생성
        cells = self._create_cells_from_grid(grid)

        # 신뢰도 계산
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
        """그리드 셀에서 텍스트 추출"""
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
        """bbox 내의 텍스트 추출"""
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
        """두 bbox가 겹치는지 확인"""
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
        """그리드에서 셀 정보 생성"""
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
        """선 기반 결과 신뢰도 계산"""
        score = 0.0

        # 기본 점수 (다른 전략보다 낮음)
        score += 0.3

        # 그리드 완전성
        if grid.is_complete:
            score += 0.2
        elif grid.reconstructed:
            score += 0.1

        # 행/열 수
        if grid.row_count >= self.MIN_TABLE_ROWS:
            score += 0.1
        if grid.col_count >= self.MIN_TABLE_COLS:
            score += 0.1

        # 데이터 밀도
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

        # 의미 있는 셀 수
        meaningful_count = sum(
            1 for row in data for cell in row
            if cell and len(str(cell).strip()) >= 2
        )

        if meaningful_count < 2:
            score -= 0.2

        # 유효 행 수
        valid_rows = sum(1 for row in data if any(cell and str(cell).strip() for cell in row))
        if valid_rows <= 1:
            score -= 0.2

        # 그래픽 영역 겹침 확인
        if self.graphic_detector:
            if self.graphic_detector.is_bbox_in_graphic_region(grid.bbox, threshold=0.3):
                score -= 0.3

        return max(0.0, min(1.0, score))

    def _select_best_candidates(self, candidates: List[TableCandidate]) -> List[TableCandidate]:
        """
        최선의 테이블 후보 선택

        V3.2: 전략 우선순위를 더 강하게 반영
        - PyMuPDF가 가장 정확하므로, 같은 영역에서 PyMuPDF 결과 우선
        - 신뢰도 차이가 0.2 미만이면 전략 우선순위로 선택
        """
        if not candidates:
            return []

        # 전략 우선순위: PYMUPDF > PDFPLUMBER > HYBRID
        priority_order = {
            TableDetectionStrategy.PYMUPDF_NATIVE: 0,
            TableDetectionStrategy.PDFPLUMBER_LINES: 1,
            TableDetectionStrategy.HYBRID_ANALYSIS: 2,
            TableDetectionStrategy.BORDERLESS_HEURISTIC: 3,
        }

        # V3.2: 정렬 키 변경 - 전략 우선순위를 더 중요하게
        # 신뢰도 차이가 크지 않으면 전략 우선순위로 결정
        def sort_key(c):
            # 전략 우선순위 * 0.15를 신뢰도에서 뺌
            # 이렇게 하면 PyMuPDF(priority=0)가 pdfplumber(priority=1)보다 유리해짐
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
        두 테이블이 겹치는지 확인 (개선된 버전).

        둘 중 하나라도 상대방에 의해 threshold 이상 커버되면 True 반환.
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

        # 어느 한 쪽이라도 상대에 의해 threshold 이상 커버되면 겹침으로 판단
        ratio1 = overlap_area / bbox1_area
        ratio2 = overlap_area / bbox2_area

        return ratio1 >= threshold or ratio2 >= threshold


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'TableDetectionEngine',
]
