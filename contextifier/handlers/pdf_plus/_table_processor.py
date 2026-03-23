# contextifier/handlers/pdf_plus/_table_processor.py
"""
PDF Plus — Table Processor (orchestrator).

Responsibilities
~~~~~~~~~~~~~~~~
1. Run ``TableDetectionEngine`` on each page.
2. Compute accurate rowspan / colspan via ``CellAnalysisEngine``.
3. Detect **annotations** (footnotes / sub-headers) near tables.
4. Handle **cross-page table continuity**.
5. Convert table data + cell info into an HTML ``<table>`` with merged cells.

The public facade is :class:`TableProcessor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from contextifier.handlers.pdf_plus._types import (
    AnnotationInfo,
    CellInfo,
    PdfPlusConfig,
    TableCandidate,
)
from contextifier.handlers.pdf_plus._table_detection import TableDetectionEngine
from contextifier.handlers.pdf_plus._cell_analysis import CellAnalysisEngine
from contextifier.handlers.pdf_plus._utils import escape_html

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


# ──────────────────────────────────────────────────────────────────────
# Data containers for processed tables
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ProcessedTable:
    """One fully-processed table."""
    page_num: int
    bbox: Tuple[float, float, float, float]
    data: List[List[str]]
    cells: List[Dict]
    html: str
    annotations: List[AnnotationInfo] = field(default_factory=list)
    is_continuation: bool = False
    confidence: float = 1.0


# ──────────────────────────────────────────────────────────────────────
# TableProcessor
# ──────────────────────────────────────────────────────────────────────

class TableProcessor:
    """
    High-level table-processing orchestrator for a *single page*.

    Typical usage::

        tp = TableProcessor(page, page_num, file_data)
        tables = tp.process()           # List[ProcessedTable]
        tp.check_continuity(previous)   # mutates tables
    """

    def __init__(
        self,
        page: Any,
        page_num: int,
        file_data: bytes,
    ) -> None:
        self.page = page
        self.page_num = page_num
        self.file_data = file_data
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self) -> List[ProcessedTable]:
        """Detect → analyse → annotate → convert to HTML."""
        engine = TableDetectionEngine(self.page, self.page_num, self.file_data)
        candidates = engine.detect_tables()

        results: list[ProcessedTable] = []
        for cand in candidates:
            cells = self._analyse_cells(cand)
            annotations = self._detect_annotations(cand)
            html = self._to_html(cand.data, cells, annotations)
            results.append(ProcessedTable(
                page_num=self.page_num,
                bbox=cand.bbox,
                data=cand.data,
                cells=cells,
                html=html,
                annotations=annotations,
                confidence=cand.confidence,
            ))
        return results

    def check_continuity(
        self,
        prev_tables: List[ProcessedTable],
        cur_tables: List[ProcessedTable],
    ) -> None:
        """
        Mark *cur_tables* as continuations of *prev_tables* when:
          - previous table ends near the bottom of its page,
          - current table starts near the top of its page,
          - column counts are compatible.
        """
        if not prev_tables or not cur_tables:
            return
        for pt in prev_tables:
            pb = pt.bbox
            if pb[3] < self.page_height * CFG.TABLE_CONTINUITY_BOTTOM_RATIO:
                continue  # didn't reach bottom
            for ct in cur_tables:
                cb = ct.bbox
                if cb[1] > self.page_height * CFG.TABLE_CONTINUITY_TOP_RATIO:
                    continue  # doesn't start near top
                pcols = max((len(r) for r in pt.data), default=0)
                ccols = max((len(r) for r in ct.data), default=0)
                if abs(pcols - ccols) <= 1:
                    ct.is_continuation = True
                    logger.debug(
                        "[TableProc] page %d table is continuation (prev cols=%d, cur cols=%d)",
                        ct.page_num + 1, pcols, ccols,
                    )

    # ------------------------------------------------------------------
    # Cell analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _analyse_cells(cand: TableCandidate) -> List[Dict]:
        engine = CellAnalysisEngine(
            data=cand.data,
            cells_info=cand.cells,
            bbox=cand.bbox,
        )
        return engine.analyze()

    # ------------------------------------------------------------------
    # Annotation detection
    # ------------------------------------------------------------------

    def _detect_annotations(self, cand: TableCandidate) -> List[AnnotationInfo]:
        """Detect footnote-like text below a table (within 30 pt)."""
        annotations: list[AnnotationInfo] = []
        tb = cand.bbox
        search_y_top = tb[3]
        search_y_bot = tb[3] + CFG.TABLE_ANNOTATION_GAP

        pd = self.page.get_text("dict", sort=True)
        for blk in pd.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                lb = ln.get("bbox", (0, 0, 0, 0))
                mid_y = (lb[1] + lb[3]) / 2
                if not (search_y_top <= mid_y <= search_y_bot):
                    continue
                if lb[0] < tb[0] - 10 or lb[2] > tb[2] + 10:
                    continue
                text = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                if not text:
                    continue
                atype = self._classify_annotation(text)
                annotations.append(AnnotationInfo(
                    text=text,
                    bbox=lb,
                    annotation_type=atype,
                ))
        return annotations

    @staticmethod
    def _classify_annotation(text: str) -> str:
        t = text.strip()
        if t.startswith("*") or t.startswith("※") or t.startswith("주"):
            return "footnote"
        if t.startswith("(") and t.endswith(")"):
            return "subheader"
        if any(t.startswith(p) for p in ["출처", "자료", "Source", "Note"]):
            return "source"
        return "annotation"

    # ------------------------------------------------------------------
    # HTML conversion
    # ------------------------------------------------------------------

    def _to_html(
        self,
        data: List[List],
        cells: List[Dict],
        annotations: List[AnnotationInfo],
    ) -> str:
        """Build an HTML ``<table>`` with rowspan / colspan attributes."""
        nr = len(data)
        nc = max((len(r) for r in data), default=0)
        if not nr or not nc:
            return ""

        # Build a cell lookup: (row, col) → cell_dict
        cell_map: dict[tuple[int, int], dict] = {}
        covered: set[tuple[int, int]] = set()
        for c in cells:
            r, co = c["row"], c["col"]
            cell_map[(r, co)] = c
            for ri in range(r, r + c.get("rowspan", 1)):
                for ci in range(co, co + c.get("colspan", 1)):
                    if (ri, ci) != (r, co):
                        covered.add((ri, ci))

        parts: list[str] = ["<table>"]
        for ri in range(nr):
            parts.append("<tr>")
            for ci in range(nc):
                if (ri, ci) in covered:
                    continue
                cell_dict = cell_map.get((ri, ci))
                rs = cell_dict["rowspan"] if cell_dict else 1
                cs = cell_dict["colspan"] if cell_dict else 1
                val = data[ri][ci] if ci < len(data[ri]) else ""
                val_esc = escape_html(str(val).strip()) if val else ""

                tag = "th" if ri == 0 else "td"
                attrs = ""
                if rs > 1:
                    attrs += f' rowspan="{rs}"'
                if cs > 1:
                    attrs += f' colspan="{cs}"'
                parts.append(f"<{tag}{attrs}>{val_esc}</{tag}>")
            parts.append("</tr>")
        parts.append("</table>")

        # Append annotations as <p> below table
        for ann in annotations:
            parts.append(f'<p class="table-annotation">{escape_html(ann.text)}</p>')

        return "\n".join(parts)


__all__ = ["TableProcessor", "ProcessedTable"]
