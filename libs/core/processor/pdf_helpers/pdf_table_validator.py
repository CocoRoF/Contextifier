"""
Table Quality Validator for PDF Handler

Validates whether detected table candidates are actual tables.
Prevents graphic regions from being misidentified as tables.
"""

import logging
from typing import List, Tuple, Optional

from libs.core.processor.pdf_helpers.types import PDFConfig
from libs.core.processor.pdf_helpers.pdf_graphic_detector import GraphicRegionDetector

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
        Validates a table candidate.
        
        Features:
        - Relaxed penalty accumulation
        - Prevents filtering of normal tables
        - Enhanced PyMuPDF result reliability
        
        Args:
            data: Table data (2D list)
            bbox: Table bounding box
            cells_info: Cell information (optional)
            skip_graphic_check: Skip graphic region check.
                                PyMuPDF strategy is text-based, so it has high reliability.
            
        Returns:
            Tuple of (is_valid, confidence, reason)
        """
        reasons = []
        penalties = []
        is_valid = True
        confidence = 1.0
        
        # If PyMuPDF provided cell information, increase base confidence
        if cells_info and len(cells_info) > 0:
            confidence = 1.1  # Slight bonus
        
        # 0. Graphic region check (skip_graphic_check option added)
        if not skip_graphic_check:
            if self.graphic_detector and self.graphic_detector.is_bbox_in_graphic_region(bbox, threshold=0.5):
                return False, 0.0, "in_graphic_region"
        
        # 1. Basic data validation
        if not data or len(data) == 0:
            return False, 0.0, "empty_data"
        
        num_rows = len(data)
        num_cols = max(len(row) for row in data) if data else 0
        
        if num_rows < PDFConfig.MIN_TABLE_ROWS:
            return False, 0.0, f"too_few_rows({num_rows})"
        
        if num_cols < PDFConfig.MIN_TABLE_COLS:
            return False, 0.0, f"too_few_cols({num_cols})"
        
        # 2. Filled cell ratio validation
        total_cells = sum(len(row) for row in data)
        filled_cells = sum(1 for row in data for cell in row 
                          if cell and str(cell).strip())
        filled_ratio = filled_cells / total_cells if total_cells > 0 else 0
        
        # Progressive penalty based on fill ratio
        if filled_ratio < PDFConfig.TABLE_MIN_FILLED_CELL_RATIO:
            if filled_ratio < 0.05:
                penalties.append(f"very_low_fill_ratio({filled_ratio:.2f})")
                confidence -= 0.3
            else:
                penalties.append(f"low_fill_ratio({filled_ratio:.2f})")
                confidence -= 0.15
        
        # 3. Empty row ratio validation
        empty_rows = sum(1 for row in data 
                        if not any(cell and str(cell).strip() for cell in row))
        empty_row_ratio = empty_rows / num_rows if num_rows > 0 else 1.0
        
        if empty_row_ratio >= PDFConfig.TABLE_MAX_EMPTY_ROW_RATIO:
            penalties.append(f"too_many_empty_rows({empty_row_ratio:.2f})")
            confidence -= 0.15
        
        # 4. Meaningful cell count validation
        meaningful_cells = self._count_meaningful_cells(data)
        if meaningful_cells < PDFConfig.TABLE_MIN_MEANINGFUL_CELLS:
            penalties.append(f"few_meaningful_cells({meaningful_cells})")
            confidence -= 0.15
        
        # 5. Valid row count validation (rows that are not empty)
        valid_rows = sum(1 for row in data 
                        if any(cell and str(cell).strip() for cell in row))
        if valid_rows < PDFConfig.TABLE_MIN_VALID_ROWS:
            penalties.append(f"few_valid_rows({valid_rows})")
            confidence -= 0.15
        
        # 6. Text density validation
        text_density = self._calculate_text_density(data, bbox)
        if text_density < PDFConfig.TABLE_MIN_TEXT_DENSITY:
            penalties.append(f"low_text_density({text_density:.3f})")
            confidence -= 0.1
        
        # 7. Single row/column table special validation
        if num_rows == 1 or num_cols == 1:
            # More strict validation for 1 row or 1 column tables
            if filled_ratio < 0.5:
                penalties.append("single_row_col_low_fill")
                confidence -= 0.2
        
        # 8. Abnormal row/column ratio validation
        if num_cols > num_rows * 5:  # More than 5 times as many columns as rows
            penalties.append(f"abnormal_ratio(cols/rows={num_cols}/{num_rows})")
            confidence -= 0.1
        
        # 9. Long text cell detection (text blocks misidentified as tables)
        long_cell_count, extreme_cell_count = self._analyze_cell_lengths(data)
        
        # Fail immediately if there are extremely long cells
        if extreme_cell_count > 0:
            return False, 0.0, f"extreme_long_cell({extreme_cell_count})"
        
        # Long text cell ratio check (more lenient)
        if filled_cells > 0:
            long_cell_ratio = long_cell_count / filled_cells
            if long_cell_ratio > PDFConfig.TABLE_MAX_LONG_CELLS_RATIO:
                penalties.append(f"too_many_long_cells({long_cell_ratio:.2f})")
                confidence -= 0.2
        
        # 10. Paragraph text detection (body text misidentified as tables)
        paragraph_count = self._count_paragraph_cells(data)
        if paragraph_count > 0:
            # High probability of not being a table if paragraph-style text exists
            paragraph_ratio = paragraph_count / max(1, filled_cells)
            if paragraph_ratio > 0.25:  # Relaxed from 15% to 25%
                return False, 0.0, f"contains_paragraph_text({paragraph_count})"
            elif paragraph_ratio > 0.1:  # Relaxed from 5% to 10%
                penalties.append(f"has_paragraph_cells({paragraph_count})")
                confidence -= 0.15
        
        # 11. Two-column table special validation (body text easily misidentified as table)
        if num_cols == 2:
            is_valid_2col, reason_2col = self._validate_two_column_table(data, bbox)
            if not is_valid_2col:
                return False, 0.0, f"invalid_2col_table({reason_2col})"
        
        # 12. Suspicious if table bbox covers large portion of page with many rows
        # More lenient conditions
        bbox_height = bbox[3] - bbox[1]
        page_coverage = bbox_height / self.page_height if self.page_height > 0 else 0
        if page_coverage > 0.7 and num_rows > 15 and num_cols == 2:  # Relaxed conditions
            # High probability of body text if covering 70%+ of page, 15+ rows, and 2 columns
            penalties.append(f"suspicious_large_2col(coverage={page_coverage:.2f}, rows={num_rows})")
            confidence -= 0.15
        
        # Final judgment
        # Confidence floor adjustment (lowered to 0.4)
        confidence = max(0.0, min(1.0, confidence))
        
        # Using lower threshold instead of CONFIDENCE_THRESHOLD
        min_threshold = 0.35  # Lowered from 0.5
        if confidence < min_threshold:
            is_valid = False
        
        reason = ", ".join(penalties) if penalties else "valid"
        
        if not is_valid:
            logger.debug(f"[TableValidator] Rejected: {bbox}, reason={reason}, conf={confidence:.2f}")
        
        return is_valid, confidence, reason
    
    def _analyze_cell_lengths(self, data: List[List[Optional[str]]]) -> Tuple[int, int]:
        """
        Analyzes cell text lengths.
        
        Returns:
            Tuple of (long_cell_count, extreme_cell_count)
            - long_cell_count: Number of cells exceeding TABLE_MAX_CELL_TEXT_LENGTH
            - extreme_cell_count: Number of cells exceeding TABLE_EXTREME_CELL_LENGTH
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
                        long_count += 1  # Extremely long cells are also included in long cells
                    elif text_len > PDFConfig.TABLE_MAX_CELL_TEXT_LENGTH:
                        long_count += 1
        
        return long_count, extreme_count
    
    def _count_meaningful_cells(self, data: List[List[Optional[str]]]) -> int:
        """
        Counts the number of meaningful cells.
        
        Meaningful cells:
        - Text with 2 or more characters
        - Not simple symbols
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
        Calculates text density relative to the region area.
        """
        # Total text length
        total_text_len = sum(
            len(str(cell).strip()) 
            for row in data 
            for cell in row 
            if cell
        )
        
        # Region area
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if area <= 0:
            return 0.0
        
        # Approximate area per character (approximately 50 pt² for 10pt font)
        estimated_text_area = total_text_len * 50
        
        return estimated_text_area / area

    def _count_paragraph_cells(self, data: List[List[Optional[str]]]) -> int:
        """
        Counts cells containing paragraph-style text.
        
        Paragraph detection criteria:
        - Text with 50 or more characters
        - Contains sentence punctuation (periods, commas, etc.)
        - 5 or more words separated by spaces
        
        If many such cells exist, body text has likely been misidentified as a table.
        """
        paragraph_count = 0
        
        for row in data:
            for cell in row:
                if not cell:
                    continue
                    
                text = str(cell).strip()
                text_len = len(text)
                
                # Base condition: 50 characters or more
                if text_len < 50:
                    continue
                
                # Calculate word count
                words = text.split()
                word_count = len(words)
                
                # Check for sentence punctuation
                has_sentence_marks = any(p in text for p in ['.', '。', '?', '!', ',', '、'])
                
                # Paragraph determination
                is_paragraph = False
                
                # Case 1: Long text + multiple words + sentence punctuation
                if text_len >= 100 and word_count >= 8 and has_sentence_marks:
                    is_paragraph = True
                
                # Case 2: Very long text + sentence punctuation
                elif text_len >= 150 and has_sentence_marks:
                    is_paragraph = True
                
                # Case 3: Long description in parentheses (e.g., annotations in papers, reports)
                elif text_len >= 80 and word_count >= 10:
                    is_paragraph = True
                
                if is_paragraph:
                    paragraph_count += 1
        
        return paragraph_count
    
    def _validate_two_column_table(self, data: List[List[Optional[str]]], 
                                    bbox: Tuple[float, float, float, float]) -> Tuple[bool, str]:
        """
        Validates the validity of a two-column table.
        
        Two-column tables are easily misidentified from body text.
        Example: Chart Y-axis labels + body text can be detected as a 2-column table.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        num_rows = len(data)
        
        # 1. Check if first column is mostly empty cells or short text
        col1_empty_count = 0
        col1_short_count = 0
        col2_long_count = 0
        col2_has_paragraphs = 0
        
        for row in data:
            if len(row) < 2:
                continue
            
            col1 = str(row[0]).strip() if row[0] else ""
            col2 = str(row[1]).strip() if row[1] else ""
            
            # First column analysis
            if not col1:
                col1_empty_count += 1
            elif len(col1) <= 10:
                col1_short_count += 1
            
            # Second column analysis
            if len(col2) > 80:
                col2_long_count += 1
                # Check for sentence structure
                if any(p in col2 for p in ['.', '。', ',', '、']) and len(col2.split()) >= 5:
                    col2_has_paragraphs += 1
        
        # Pattern 1: First column mostly empty + second column has long text
        if num_rows > 0:
            col1_empty_ratio = col1_empty_count / num_rows
            col2_long_ratio = col2_long_count / num_rows
            
            # First column 60%+ empty + second column 30%+ long text = body text
            if col1_empty_ratio >= 0.6 and col2_long_ratio >= 0.3:
                return False, f"col1_empty({col1_empty_ratio:.0%})_col2_long({col2_long_ratio:.0%})"
        
        # Pattern 2: Many paragraph-style entries in second column
        if num_rows > 5 and col2_has_paragraphs >= 2:
            return False, f"col2_paragraphs({col2_has_paragraphs})"
        
        # Pattern 3: If first column is short and second is long overall, likely body text not key-value
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
