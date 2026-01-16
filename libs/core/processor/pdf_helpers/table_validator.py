"""
Table Quality Validator for PDF Handler

Validates whether detected table candidates are actual tables.
Prevents graphic regions from being misidentified as tables.
"""

import logging
from typing import List, Tuple, Optional

from libs.core.processor.pdf_helpers.types import PDFConfig
from libs.core.processor.pdf_helpers.graphic_detector import GraphicRegionDetector

logger = logging.getLogger(__name__)


# ============================================================================
# Table Quality Validator
# ============================================================================

class TableQualityValidator:
    """
    Table Quality Validator
    
    Validates whether detected table candidates are actual tables.
    
    Validation Criteria:
    1. Filled cell ratio (too low indicates fake table)
    2. Empty row/column ratio
    3. Text density
    4. Data validity (meaningful text)
    5. Grid regularity
    6. Long text cell detection (text blocks misidentified as tables)
    7. Paragraph text detection (body text misidentified as tables)
    8. Two-column table special validation (body text easily misidentified as table)
    """
    
    def __init__(self, page, graphic_detector: Optional[GraphicRegionDetector] = None):
        """
        Args:
            page: PyMuPDF page object
            graphic_detector: Graphic region detector (optional)
        """
        self.page = page
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.graphic_detector = graphic_detector
    
    def validate(self, 
                 data: List[List[Optional[str]]], 
                 bbox: Tuple[float, float, float, float],
                 cells_info: Optional[List] = None,
                 skip_graphic_check: bool = False) -> Tuple[bool, float, str]:
        """
        테이블 후보를 검증합니다.
        
                - 패널티 누적 완화
        - 일반 테이블 필터링 방지
        - PyMuPDF 결과 신뢰도 강화
        
        Args:
            data: 테이블 데이터 (2D 리스트)
            bbox: 테이블 영역
            cells_info: 셀 정보 (선택)
            skip_graphic_check: 그래픽 영역 체크 건너뛰기 
                                PyMuPDF 전략은 텍스트 기반이므로 신뢰도가 높음
            
        Returns:
            (is_valid, confidence, reason) 튜플
        """
        reasons = []
        penalties = []
        is_valid = True
        confidence = 1.0
        
        # PyMuPDF가 셀 정보를 제공했으면 기본 신뢰도 상향
        if cells_info and len(cells_info) > 0:
            confidence = 1.1  # 약간의 보너스
        
        # 0. 그래픽 영역 체크 (skip_graphic_check 옵션 추가)
        if not skip_graphic_check:
            if self.graphic_detector and self.graphic_detector.is_bbox_in_graphic_region(bbox, threshold=0.5):
                return False, 0.0, "in_graphic_region"
        
        # 1. 기본 데이터 검증
        if not data or len(data) == 0:
            return False, 0.0, "empty_data"
        
        num_rows = len(data)
        num_cols = max(len(row) for row in data) if data else 0
        
        if num_rows < PDFConfig.MIN_TABLE_ROWS:
            return False, 0.0, f"too_few_rows({num_rows})"
        
        if num_cols < PDFConfig.MIN_TABLE_COLS:
            return False, 0.0, f"too_few_cols({num_cols})"
        
        # 2. 채워진 셀 비율 검증
        total_cells = sum(len(row) for row in data)
        filled_cells = sum(1 for row in data for cell in row 
                          if cell and str(cell).strip())
        filled_ratio = filled_cells / total_cells if total_cells > 0 else 0
        
        # 채워진 비율에 따른 점진적 패널티
        if filled_ratio < PDFConfig.TABLE_MIN_FILLED_CELL_RATIO:
            if filled_ratio < 0.05:
                penalties.append(f"very_low_fill_ratio({filled_ratio:.2f})")
                confidence -= 0.3
            else:
                penalties.append(f"low_fill_ratio({filled_ratio:.2f})")
                confidence -= 0.15
        
        # 3. 빈 행 비율 검증
        empty_rows = sum(1 for row in data 
                        if not any(cell and str(cell).strip() for cell in row))
        empty_row_ratio = empty_rows / num_rows if num_rows > 0 else 1.0
        
        if empty_row_ratio >= PDFConfig.TABLE_MAX_EMPTY_ROW_RATIO:
            penalties.append(f"too_many_empty_rows({empty_row_ratio:.2f})")
            confidence -= 0.15
        
        # 4. 의미 있는 셀 수 검증
        meaningful_cells = self._count_meaningful_cells(data)
        if meaningful_cells < PDFConfig.TABLE_MIN_MEANINGFUL_CELLS:
            penalties.append(f"few_meaningful_cells({meaningful_cells})")
            confidence -= 0.15
        
        # 5. 유효 행 수 검증 (빈 행이 아닌 행)
        valid_rows = sum(1 for row in data 
                        if any(cell and str(cell).strip() for cell in row))
        if valid_rows < PDFConfig.TABLE_MIN_VALID_ROWS:
            penalties.append(f"few_valid_rows({valid_rows})")
            confidence -= 0.15
        
        # 6. 텍스트 밀도 검증
        text_density = self._calculate_text_density(data, bbox)
        if text_density < PDFConfig.TABLE_MIN_TEXT_DENSITY:
            penalties.append(f"low_text_density({text_density:.3f})")
            confidence -= 0.1
        
        # 7. 단일 행/열 테이블 특별 검증
        if num_rows == 1 or num_cols == 1:
            # 1행 또는 1열인 경우 더 엄격한 검증
            if filled_ratio < 0.5:
                penalties.append("single_row_col_low_fill")
                confidence -= 0.2
        
        # 8. 비정상적인 행/열 비율 검증
        if num_cols > num_rows * 5:  # 열이 행보다 5배 이상 많음
            penalties.append(f"abnormal_ratio(cols/rows={num_cols}/{num_rows})")
            confidence -= 0.1
        
        # 9. 긴 텍스트 셀 감지 (텍스트 블록이 테이블로 오인된 경우)
        long_cell_count, extreme_cell_count = self._analyze_cell_lengths(data)
        
        # 극단적으로 긴 셀이 있으면 즉시 실패
        if extreme_cell_count > 0:
            return False, 0.0, f"extreme_long_cell({extreme_cell_count})"
        
        # 긴 텍스트 셀 비율 검사 (더 관대함)
        if filled_cells > 0:
            long_cell_ratio = long_cell_count / filled_cells
            if long_cell_ratio > PDFConfig.TABLE_MAX_LONG_CELLS_RATIO:
                penalties.append(f"too_many_long_cells({long_cell_ratio:.2f})")
                confidence -= 0.2
        
        # 10. 문단 텍스트 감지 (본문 텍스트가 테이블로 오인된 경우)
        paragraph_count = self._count_paragraph_cells(data)
        if paragraph_count > 0:
            # 문단 형태의 텍스트가 있으면 테이블이 아닐 가능성 높음
            paragraph_ratio = paragraph_count / max(1, filled_cells)
            if paragraph_ratio > 0.25:  # 15% → 25%로 완화
                return False, 0.0, f"contains_paragraph_text({paragraph_count})"
            elif paragraph_ratio > 0.1:  # 5% → 10%로 완화
                penalties.append(f"has_paragraph_cells({paragraph_count})")
                confidence -= 0.15
        
        # 11. 2열 테이블 특별 검증 (본문이 테이블로 오인되기 쉬움)
        if num_cols == 2:
            is_valid_2col, reason_2col = self._validate_two_column_table(data, bbox)
            if not is_valid_2col:
                return False, 0.0, f"invalid_2col_table({reason_2col})"
        
        # 12. 테이블 bbox가 페이지의 큰 부분을 차지하면서 행이 많으면 의심
        # 더 관대한 조건
        bbox_height = bbox[3] - bbox[1]
        page_coverage = bbox_height / self.page_height if self.page_height > 0 else 0
        if page_coverage > 0.7 and num_rows > 15 and num_cols == 2:  # 조건 완화
            # 페이지 70% 이상 차지하고, 15행 초과, 2열이면 본문 텍스트일 가능성 높음
            penalties.append(f"suspicious_large_2col(coverage={page_coverage:.2f}, rows={num_rows})")
            confidence -= 0.15
        
        # 최종 판단
        # 신뢰도 하한선 조정 (0.4로 낮춤)
        confidence = max(0.0, min(1.0, confidence))
        
        # CONFIDENCE_THRESHOLD 대신 더 낮은 임계값 사용
        min_threshold = 0.35  # 기존 0.5에서 낮춤
        if confidence < min_threshold:
            is_valid = False
        
        reason = ", ".join(penalties) if penalties else "valid"
        
        if not is_valid:
            logger.debug(f"[TableValidator] Rejected: {bbox}, reason={reason}, conf={confidence:.2f}")
        
        return is_valid, confidence, reason
    
    def _analyze_cell_lengths(self, data: List[List[Optional[str]]]) -> Tuple[int, int]:
        """
        셀 텍스트 길이를 분석합니다.
        
        Returns:
            (long_cell_count, extreme_cell_count) 튜플
            - long_cell_count: TABLE_MAX_CELL_TEXT_LENGTH 초과 셀 수
            - extreme_cell_count: TABLE_EXTREME_CELL_LENGTH 초과 셀 수
        """
        long_count = 0
        extreme_count = 0
        
        for row in data:
            for cell in row:
                if cell:
                    text = str(cell).strip()
                    text_len = len(text)
                    
                    if text_len > PDFConfig.TABLE_EXTREME_CELL_LENGTH:
                        extreme_count += 1
                        long_count += 1  # 극단적으로 긴 것도 긴 셀에 포함
                    elif text_len > PDFConfig.TABLE_MAX_CELL_TEXT_LENGTH:
                        long_count += 1
        
        return long_count, extreme_count
    
    def _count_meaningful_cells(self, data: List[List[Optional[str]]]) -> int:
        """
        의미 있는 셀 수를 계산합니다.
        
        의미 있는 셀:
        - 2자 이상의 텍스트
        - 단순 기호가 아닌 것
        """
        count = 0
        simple_symbols = {'', '-', '–', '—', '.', ':', ';', '|', '/', '\\', 
                          '*', '#', '@', '!', '?', ',', ' '}
        
        for row in data:
            for cell in row:
                if cell:
                    text = str(cell).strip()
                    if len(text) >= 2 and text not in simple_symbols:
                        count += 1
        
        return count
    
    def _calculate_text_density(self, 
                                 data: List[List[Optional[str]]], 
                                 bbox: Tuple[float, float, float, float]) -> float:
        """
        영역 대비 텍스트 밀도를 계산합니다.
        """
        # 총 텍스트 길이
        total_text_len = sum(
            len(str(cell).strip()) 
            for row in data 
            for cell in row 
            if cell
        )
        
        # 영역 넓이
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if area <= 0:
            return 0.0
        
        # 대략적인 글자당 면적 (10pt 폰트 기준 약 50 pt²)
        estimated_text_area = total_text_len * 50
        
        return estimated_text_area / area

    def _count_paragraph_cells(self, data: List[List[Optional[str]]]) -> int:
        """
        문단 형태의 텍스트를 포함하는 셀 수를 계산합니다.
        
        문단 판단 기준:
        - 50자 이상의 텍스트
        - 문장 부호(마침표, 쉼표 등) 포함
        - 공백으로 구분된 단어가 5개 이상
        
        이런 셀이 많으면 본문 텍스트가 테이블로 오인된 것입니다.
        """
        paragraph_count = 0
        
        for row in data:
            for cell in row:
                if not cell:
                    continue
                    
                text = str(cell).strip()
                text_len = len(text)
                
                # 기본 조건: 50자 이상
                if text_len < 50:
                    continue
                
                # 단어 수 계산
                words = text.split()
                word_count = len(words)
                
                # 문장 부호 확인
                has_sentence_marks = any(p in text for p in ['.', '。', '?', '!', ',', '、'])
                
                # 문단 판단
                is_paragraph = False
                
                # Case 1: 긴 텍스트 + 여러 단어 + 문장 부호
                if text_len >= 100 and word_count >= 8 and has_sentence_marks:
                    is_paragraph = True
                
                # Case 2: 매우 긴 텍스트 + 문장 부호
                elif text_len >= 150 and has_sentence_marks:
                    is_paragraph = True
                
                # Case 3: 괄호 안의 긴 설명 (예: 논문, 보고서 등의 주석)
                elif text_len >= 80 and word_count >= 10:
                    is_paragraph = True
                
                if is_paragraph:
                    paragraph_count += 1
        
        return paragraph_count
    
    def _validate_two_column_table(self, data: List[List[Optional[str]]], 
                                    bbox: Tuple[float, float, float, float]) -> Tuple[bool, str]:
        """
        2열 테이블의 유효성을 검증합니다.
        
        2열 테이블은 본문 텍스트가 테이블로 오인되기 쉽습니다.
        예: 차트의 Y축 레이블 + 본문이 2열 테이블로 감지될 수 있음
        
        Returns:
            (is_valid, reason) 튜플
        """
        num_rows = len(data)
        
        # 1. 첫 번째 열이 대부분 빈 셀이거나 짧은 텍스트인지 확인
        col1_empty_count = 0
        col1_short_count = 0
        col2_long_count = 0
        col2_has_paragraphs = 0
        
        for row in data:
            if len(row) < 2:
                continue
            
            col1 = str(row[0]).strip() if row[0] else ""
            col2 = str(row[1]).strip() if row[1] else ""
            
            # 첫 번째 열 분석
            if not col1:
                col1_empty_count += 1
            elif len(col1) <= 10:
                col1_short_count += 1
            
            # 두 번째 열 분석
            if len(col2) > 80:
                col2_long_count += 1
                # 문장 형태 확인
                if any(p in col2 for p in ['.', '。', ',', '、']) and len(col2.split()) >= 5:
                    col2_has_paragraphs += 1
        
        # 패턴 1: 첫 열이 대부분 빈칸이고 두 번째 열에 긴 텍스트
        if num_rows > 0:
            col1_empty_ratio = col1_empty_count / num_rows
            col2_long_ratio = col2_long_count / num_rows
            
            # 첫 열 60% 이상 빈칸 + 두 번째 열 30% 이상 긴 텍스트 = 본문 텍스트
            if col1_empty_ratio >= 0.6 and col2_long_ratio >= 0.3:
                return False, f"col1_empty({col1_empty_ratio:.0%})_col2_long({col2_long_ratio:.0%})"
        
        # 패턴 2: 두 번째 열에 문단 형태가 많음
        if num_rows > 5 and col2_has_paragraphs >= 2:
            return False, f"col2_paragraphs({col2_has_paragraphs})"
        
        # 패턴 3: 전체적으로 첫 열이 짧고 두 번째 열이 길면 키-값이 아닌 본문일 가능성
        if num_rows > 10:
            col1_short_ratio = (col1_empty_count + col1_short_count) / num_rows
            if col1_short_ratio >= 0.8 and col2_long_count >= 5:
                return False, f"asymmetric_cols(short1={col1_short_ratio:.0%}, long2={col2_long_count})"
        
        return True, "valid"


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'TableQualityValidator',
]
