# contextifier_new/handlers/pdf_plus/_table_validator.py
"""
PDF Plus — Table Quality Validator (12-point system).

Validates whether a detected table candidate is a *real* table (not a
graphic misidentified as one, or body text in columns).

The 12 validation criteria are:
  0. Graphic-region overlap (bypass-able for PyMuPDF results)
  1. Basic size (min rows / cols)
  2. Filled-cell ratio
  3. Empty-row ratio
  4. Meaningful cells
  5. Valid rows
  6. Text density
  7. Single row/col special
  8. Abnormal row-col ratio
  9. Long / extreme text cells
 10. Paragraph-text cells
 11. Two-column table special
 12. Large 2-col page-coverage heuristic
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from contextifier_new.handlers.pdf_plus._types import PdfPlusConfig
from contextifier_new.handlers.pdf_plus._graphic_detector import GraphicRegionDetector

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class TableQualityValidator:
    """Perform 12-point quality validation on a table candidate."""

    def __init__(
        self,
        page: Any,
        graphic_detector: Optional[GraphicRegionDetector] = None,
    ) -> None:
        self.page = page
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height
        self.graphic_detector = graphic_detector

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def validate(
        self,
        data: List[List[Optional[str]]],
        bbox: Tuple[float, float, float, float],
        cells_info: Optional[list] = None,
        skip_graphic_check: bool = False,
    ) -> Tuple[bool, float, str]:
        """
        Returns *(is_valid, confidence, reason_string)*.

        *skip_graphic_check* should be ``True`` for PyMuPDF candidates
        (text-based detection is reliable even in coloured-cell tables).
        """
        penalties: list[str] = []
        confidence = 1.0

        # Bonus for cell-info
        if cells_info:
            confidence = 1.1

        # --- 0. graphic overlap ---
        if not skip_graphic_check and self.graphic_detector:
            if self.graphic_detector.is_bbox_in_graphic_region(bbox, threshold=0.5):
                return False, 0.0, "in_graphic_region"

        # --- 1. basic data ---
        if not data:
            return False, 0.0, "empty_data"
        num_rows = len(data)
        num_cols = max((len(r) for r in data), default=0)
        if num_rows < CFG.MIN_TABLE_ROWS:
            return False, 0.0, f"too_few_rows({num_rows})"
        if num_cols < CFG.MIN_TABLE_COLS:
            return False, 0.0, f"too_few_cols({num_cols})"

        total_cells = sum(len(r) for r in data)
        filled_cells = sum(
            1 for row in data for cell in row if cell and str(cell).strip()
        )
        filled_ratio = filled_cells / total_cells if total_cells else 0.0

        # --- 2. filled-cell ratio ---
        if filled_ratio < CFG.TABLE_MIN_FILLED_CELL_RATIO:
            if filled_ratio < 0.05:
                penalties.append(f"very_low_fill({filled_ratio:.2f})")
                confidence -= 0.3
            else:
                penalties.append(f"low_fill({filled_ratio:.2f})")
                confidence -= 0.15

        # --- 3. empty-row ratio ---
        empty_rows = sum(
            1 for row in data if not any(c and str(c).strip() for c in row)
        )
        empty_ratio = empty_rows / num_rows if num_rows else 1.0
        if empty_ratio >= CFG.TABLE_MAX_EMPTY_ROW_RATIO:
            penalties.append(f"empty_rows({empty_ratio:.2f})")
            confidence -= 0.15

        # --- 4. meaningful cells ---
        meaningful = self._count_meaningful(data)
        if meaningful < CFG.TABLE_MIN_MEANINGFUL_CELLS:
            penalties.append(f"few_meaningful({meaningful})")
            confidence -= 0.15

        # --- 5. valid rows ---
        valid_rows = sum(
            1 for row in data if any(c and str(c).strip() for c in row)
        )
        if valid_rows < CFG.TABLE_MIN_VALID_ROWS:
            penalties.append(f"few_valid_rows({valid_rows})")
            confidence -= 0.15

        # --- 6. text density ---
        density = self._text_density(data, bbox)
        if density < CFG.TABLE_MIN_TEXT_DENSITY:
            penalties.append(f"low_density({density:.3f})")
            confidence -= 0.1

        # --- 7. single row/col ---
        if num_rows == 1 or num_cols == 1:
            if filled_ratio < 0.5:
                penalties.append("single_rc_low_fill")
                confidence -= 0.2

        # --- 8. abnormal ratio ---
        if num_cols > num_rows * 5:
            penalties.append(f"abnormal_ratio(c/r={num_cols}/{num_rows})")
            confidence -= 0.1

        # --- 9. long cells ---
        long_cnt, extreme_cnt = self._cell_lengths(data)
        if extreme_cnt > 0:
            return False, 0.0, f"extreme_long_cell({extreme_cnt})"
        if filled_cells > 0 and long_cnt / filled_cells > CFG.TABLE_MAX_LONG_CELLS_RATIO:
            penalties.append(f"long_cells({long_cnt / filled_cells:.2f})")
            confidence -= 0.2

        # --- 10. paragraph text ---
        para_cnt = self._paragraph_cells(data)
        if para_cnt and filled_cells:
            pratio = para_cnt / filled_cells
            if pratio > 0.25:
                return False, 0.0, f"paragraph_text({para_cnt})"
            if pratio > 0.1:
                penalties.append(f"para_cells({para_cnt})")
                confidence -= 0.15

        # --- 11. two-column special ---
        if num_cols == 2:
            ok, reason2 = self._validate_two_col(data, bbox)
            if not ok:
                return False, 0.0, f"invalid_2col({reason2})"

        # --- 12. large 2-col page coverage ---
        page_cov = (bbox[3] - bbox[1]) / self.page_height if self.page_height else 0
        if page_cov > 0.7 and num_rows > 15 and num_cols == 2:
            penalties.append(f"large_2col(cov={page_cov:.2f})")
            confidence -= 0.15

        # --- final ---
        confidence = max(0.0, min(1.0, confidence))
        is_valid = confidence >= 0.35
        reason = ", ".join(penalties) if penalties else "valid"
        if not is_valid:
            logger.debug(
                "[TableValidator] Rejected: %s  reason=%s  conf=%.2f",
                bbox, reason, confidence,
            )
        return is_valid, confidence, reason

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _SIMPLE_SYMS = frozenset(["", "-", "–", "—", ".", ":", ";", "|", "/",
                               "\\", "*", "#", "@", "!", "?", ",", " "])

    def _count_meaningful(self, data: list) -> int:
        return sum(
            1 for row in data for cell in row
            if cell and len(str(cell).strip()) >= 2
            and str(cell).strip() not in self._SIMPLE_SYMS
        )

    @staticmethod
    def _text_density(data: list, bbox: tuple) -> float:
        total = sum(len(str(c).strip()) for row in data for c in row if c)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area <= 0:
            return 0.0
        return (total * 50) / area

    @staticmethod
    def _cell_lengths(data: list) -> Tuple[int, int]:
        long_c = extreme_c = 0
        for row in data:
            for cell in row:
                if cell:
                    n = len(str(cell).strip())
                    if n > CFG.TABLE_EXTREME_CELL_LENGTH:
                        extreme_c += 1
                        long_c += 1
                    elif n > CFG.TABLE_MAX_CELL_TEXT_LENGTH:
                        long_c += 1
        return long_c, extreme_c

    @staticmethod
    def _paragraph_cells(data: list) -> int:
        cnt = 0
        for row in data:
            for cell in row:
                if not cell:
                    continue
                t = str(cell).strip()
                tl = len(t)
                if tl < 50:
                    continue
                wc = len(t.split())
                has_punct = any(p in t for p in [".", "。", "?", "!", ",", "、"])
                if tl >= 100 and wc >= 8 and has_punct:
                    cnt += 1
                elif tl >= 150 and has_punct:
                    cnt += 1
                elif tl >= 80 and wc >= 10:
                    cnt += 1
        return cnt

    def _validate_two_col(self, data: list, bbox: tuple) -> Tuple[bool, str]:
        nr = len(data)
        c1_empty = c1_short = c2_long = c2_para = 0
        for row in data:
            if len(row) < 2:
                continue
            c1 = str(row[0]).strip() if row[0] else ""
            c2 = str(row[1]).strip() if row[1] else ""
            if not c1:
                c1_empty += 1
            elif len(c1) <= 10:
                c1_short += 1
            if len(c2) > 80:
                c2_long += 1
                if any(p in c2 for p in [".", "。", ",", "、"]) and len(c2.split()) >= 5:
                    c2_para += 1
        if nr > 0:
            if c1_empty / nr >= 0.6 and c2_long / nr >= 0.3:
                return False, f"empty_col1({c1_empty / nr:.0%})_long_col2({c2_long / nr:.0%})"
        if nr > 5 and c2_para >= 2:
            return False, f"col2_paragraphs({c2_para})"
        if nr > 10 and (c1_empty + c1_short) / nr >= 0.8 and c2_long >= 5:
            return False, f"asymmetric({(c1_empty + c1_short) / nr:.0%},{c2_long})"
        return True, "valid"


__all__ = ["TableQualityValidator"]
