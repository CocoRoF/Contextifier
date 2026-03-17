# contextifier_new/handlers/pdf_plus/_table_detection.py
"""
PDF Plus — Multi-strategy Table Detection Engine.

Detects tables on a single PDF page using three cascading strategies:

  1. **PyMuPDF ``find_tables()``** — highest reliability, text-based.
  2. **pdfplumber** — line-based cross-validation.
  3. **Line Analysis** (HYBRID) — custom grid reconstruction
     (stricter threshold when used alone, ≥ 0.65).

Post-processing:
  - Header–data merging (1-2 row headers glued to adjacent tables)
  - Narrow-column merging (fake columns from double/triple borders)
  - 12-point quality validation via :class:`TableQualityValidator`
  - Strategy-priority best-candidate selection
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier.handlers.pdf_plus._types import (
    CellInfo,
    GridInfo,
    PdfPlusConfig,
    TableCandidate,
    TableDetectionStrategy,
)
from contextifier.handlers.pdf_plus._line_analysis import LineAnalysisEngine
from contextifier.handlers.pdf_plus._graphic_detector import GraphicRegionDetector
from contextifier.handlers.pdf_plus._table_validator import TableQualityValidator

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


class TableDetectionEngine:
    """Three-strategy table detection with narrow-column & header–data merging."""

    CONFIDENCE_THRESHOLD: float = CFG.TABLE_CONFIDENCE_THRESHOLD
    MIN_TABLE_ROWS: int = CFG.MIN_TABLE_ROWS
    MIN_TABLE_COLS: int = CFG.MIN_TABLE_COLS

    def __init__(self, page: Any, page_num: int, file_data: bytes) -> None:
        self.page = page
        self.page_num = page_num
        self.file_data = file_data          # raw PDF bytes (for pdfplumber)
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height

        self.line_engine = LineAnalysisEngine(page, self.page_width, self.page_height)
        self.h_lines, self.v_lines = self.line_engine.analyze()

        self.graphic_detector = GraphicRegionDetector(page, page_num)
        self.graphic_regions = self.graphic_detector.detect()

        self.quality_validator = TableQualityValidator(page, self.graphic_detector)

    # ==================================================================
    # Public
    # ==================================================================

    def detect_tables(self) -> List[TableCandidate]:
        """Detect, merge, validate, and return best table candidates."""
        candidates: list[TableCandidate] = []

        # Strategy 1 — PyMuPDF
        py_cands = self._merge_header_data(self._detect_pymupdf())
        candidates.extend(py_cands)

        # Strategy 2 — pdfplumber
        pl_cands = self._merge_header_data(self._detect_pdfplumber())
        candidates.extend(pl_cands)

        # Strategy 3 — line analysis (HYBRID)
        line_cands = self._detect_lines()
        if line_cands and not py_cands:
            line_cands = [c for c in line_cands if c.confidence >= 0.65]
        candidates.extend(line_cands)

        validated = self._validate(candidates)
        return self._select_best(validated)

    # ==================================================================
    # Strategy 1 — PyMuPDF
    # ==================================================================

    def _detect_pymupdf(self) -> list[TableCandidate]:
        cands: list[TableCandidate] = []
        if not hasattr(self.page, "find_tables"):
            return cands
        try:
            tabs = self.page.find_tables(
                snap_tolerance=CFG.PYMUPDF_SNAP_TOLERANCE,
                join_tolerance=CFG.PYMUPDF_JOIN_TOLERANCE,
                edge_min_length=CFG.PYMUPDF_EDGE_MIN_LENGTH,
                intersection_tolerance=CFG.PYMUPDF_INTERSECTION_TOLERANCE,
            )
            for table in tabs.tables:
                try:
                    data = table.extract()
                    if not data or not any(any(c for c in r if c) for r in data):
                        continue
                    merged_data, col_map = self._merge_narrow_cols(
                        data,
                        table.cells if hasattr(table, "cells") else None,
                    )
                    conf = self._pymupdf_confidence(table, merged_data)
                    if conf < self.CONFIDENCE_THRESHOLD:
                        continue
                    cells = self._cells_from_pymupdf(table, col_map)
                    cands.append(TableCandidate(
                        strategy=TableDetectionStrategy.PYMUPDF_NATIVE,
                        confidence=conf, bbox=table.bbox, grid=None,
                        cells=cells, data=merged_data, raw_table=table,
                    ))
                except Exception as exc:
                    logger.debug("[TableDet] pymupdf table err: %s", exc)
        except Exception as exc:
            logger.debug("[TableDet] pymupdf find_tables err: %s", exc)
        return cands

    # ==================================================================
    # Strategy 2 — pdfplumber
    # ==================================================================

    def _detect_pdfplumber(self) -> list[TableCandidate]:
        cands: list[TableCandidate] = []
        try:
            import pdfplumber
            from io import BytesIO

            with pdfplumber.open(BytesIO(self.file_data)) as pdf:
                if self.page_num >= len(pdf.pages):
                    return cands
                pp = pdf.pages[self.page_num]
                settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                }
                tables = pp.extract_tables(settings)
                for tdata in tables:
                    if not tdata or not any(any(c for c in r if c) for r in tdata):
                        continue
                    bbox = self._estimate_bbox_plumber(pp, tdata)
                    if not bbox:
                        continue
                    conf = self._plumber_confidence(tdata)
                    if conf < self.CONFIDENCE_THRESHOLD:
                        continue
                    cands.append(TableCandidate(
                        strategy=TableDetectionStrategy.PDFPLUMBER_LINES,
                        confidence=conf, bbox=bbox, grid=None,
                        cells=[], data=tdata, raw_table=None,
                    ))
        except ImportError:
            logger.debug("[TableDet] pdfplumber not installed — skipping")
        except Exception as exc:
            logger.debug("[TableDet] pdfplumber err: %s", exc)
        return cands

    # ==================================================================
    # Strategy 3 — line analysis
    # ==================================================================

    def _detect_lines(self) -> list[TableCandidate]:
        grid = self.line_engine.build_grid()
        if not grid:
            return []
        if not grid.is_complete:
            grid = self.line_engine.reconstruct_incomplete_border(grid)
            if not grid.is_complete:
                return []
        if grid.row_count < self.MIN_TABLE_ROWS or grid.col_count < self.MIN_TABLE_COLS:
            return []
        data = self._text_from_grid(grid)
        if not data or not any(any(c for c in r if c) for r in data):
            return []
        cells = self._cells_from_grid(grid)
        conf = self._line_confidence(grid, data)
        if conf < self.CONFIDENCE_THRESHOLD:
            return []
        return [TableCandidate(
            strategy=TableDetectionStrategy.HYBRID_ANALYSIS,
            confidence=conf, bbox=grid.bbox, grid=grid,
            cells=cells, data=data, raw_table=None,
        )]

    # ==================================================================
    # Header–data merging
    # ==================================================================

    def _merge_header_data(self, cands: list[TableCandidate]) -> list[TableCandidate]:
        if len(cands) < 2:
            return cands
        ordered = sorted(cands, key=lambda c: c.bbox[1])
        merged: list[TableCandidate] = []
        skip: set[int] = set()
        for i, hd in enumerate(ordered):
            if i in skip:
                continue
            if len(hd.data) > 2:
                merged.append(hd)
                continue
            cur = hd
            for j in range(i + 1, len(ordered)):
                if j in skip:
                    continue
                nxt = ordered[j]
                if self._can_merge_hd(cur, nxt):
                    cur = self._do_merge_hd(cur, nxt)
                    skip.add(j)
                else:
                    break
            merged.append(cur)
        return merged

    def _can_merge_hd(self, hd: TableCandidate, dt: TableCandidate) -> bool:
        ygap = dt.bbox[1] - hd.bbox[3]
        if ygap < -5 or ygap > 40:
            return False
        xo_start = max(hd.bbox[0], dt.bbox[0])
        xo_end = min(hd.bbox[2], dt.bbox[2])
        mw = max(hd.bbox[2] - hd.bbox[0], dt.bbox[2] - dt.bbox[0])
        if mw > 0 and max(0, xo_end - xo_start) / mw < 0.7:
            return False
        hcols = max((len(r) for r in hd.data), default=0)
        dcols = max((len(r) for r in dt.data), default=0)
        return hcols <= dcols + 1

    def _do_merge_hd(self, hd: TableCandidate, dt: TableCandidate) -> TableCandidate:
        bbox = (
            min(hd.bbox[0], dt.bbox[0]), hd.bbox[1],
            max(hd.bbox[2], dt.bbox[2]), dt.bbox[3],
        )
        hcols = max((len(r) for r in hd.data), default=0)
        dcols = max((len(r) for r in dt.data), default=0)
        ncols = max(hcols, dcols)
        merged_data: list[list] = []
        merged_cells: list[CellInfo] = []
        # Header rows
        for ri, row in enumerate(hd.data):
            if len(row) < ncols:
                adj = list(row) + [""] * (ncols - len(row))
            else:
                adj = list(row)
            merged_data.append(adj)
        hrc = len(hd.data)
        # Header cell info
        merged_cells.extend(hd.cells)
        # Data rows
        for ri, row in enumerate(dt.data):
            adj = list(row) + ([""] * (ncols - len(row))) if len(row) < ncols else list(row)
            merged_data.append(adj)
        for c in dt.cells:
            merged_cells.append(CellInfo(
                row=c.row + hrc, col=c.col,
                rowspan=c.rowspan, colspan=c.colspan, bbox=c.bbox,
            ))
        return TableCandidate(
            strategy=hd.strategy,
            confidence=max(hd.confidence, dt.confidence),
            bbox=bbox, grid=hd.grid or dt.grid,
            cells=merged_cells, data=merged_data, raw_table=None,
        )

    # ==================================================================
    # Narrow-column merge
    # ==================================================================

    def _merge_narrow_cols(
        self,
        data: list[list],
        cells: Any = None,
        min_width: float = 15.0,
    ) -> Tuple[list[list[str]], dict[int, int]]:
        if not data or not data[0]:
            return data, {}
        ncols = max(len(r) for r in data)
        if not cells:
            return self._merge_cols_by_content(data)
        widths = self._col_widths(cells, ncols)
        groups = self._col_groups(widths, min_width)
        if len(groups) == ncols:
            return data, {i: i for i in range(ncols)}
        col_map: dict[int, int] = {}
        for ni, grp in enumerate(groups):
            for oi in grp:
                col_map[oi] = ni
        merged: list[list[str]] = []
        for row in data:
            nr = [""] * len(groups)
            for oi, val in enumerate(row):
                if oi in col_map and val and str(val).strip():
                    ni = col_map[oi]
                    nr[ni] = (nr[ni] + str(val).strip()) if nr[ni] else str(val).strip()
            merged.append(nr)
        return merged, col_map

    @staticmethod
    def _col_widths(cells: Any, ncols: int) -> list[float]:
        if not cells:
            return [0.0] * ncols
        xs = sorted({c[0] for c in cells if c} | {c[2] for c in cells if c})
        if len(xs) < 2:
            return [0.0] * ncols
        ws = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        while len(ws) < ncols:
            ws.append(0.0)
        return ws[:ncols]

    @staticmethod
    def _col_groups(widths: list[float], min_w: float) -> list[list[int]]:
        groups: list[list[int]] = []
        cur: list[int] = []
        for i, w in enumerate(widths):
            cur.append(i)
            if sum(widths[j] for j in cur) >= min_w:
                groups.append(cur)
                cur = []
        if cur:
            (groups[-1] if groups else groups.append(cur) or groups).extend([] if not groups else [])
            if cur and groups and cur is not groups[-1]:
                groups[-1].extend(cur)
        return groups

    def _merge_cols_by_content(self, data: list[list]) -> Tuple[list[list[str]], dict[int, int]]:
        ncols = max(len(r) for r in data)
        nrows = len(data)
        ratios = []
        for ci in range(ncols):
            empty = sum(1 for r in data if ci >= len(r) or not r[ci] or not str(r[ci]).strip())
            ratios.append(empty / nrows if nrows else 1.0)
        groups: list[list[int]] = []
        cur: list[int] = []
        for ci, er in enumerate(ratios):
            cur.append(ci)
            if er < 0.9:
                groups.append(cur)
                cur = []
        if cur:
            if groups:
                groups[-1].extend(cur)
            else:
                groups.append(cur)
        if len(groups) == ncols:
            return data, {i: i for i in range(ncols)}
        col_map = {oi: ni for ni, grp in enumerate(groups) for oi in grp}
        merged = []
        for row in data:
            nr = [""] * len(groups)
            for oi, val in enumerate(row):
                if oi in col_map and val and str(val).strip():
                    ni = col_map[oi]
                    nr[ni] = (nr[ni] + str(val).strip()) if nr[ni] else str(val).strip()
            merged.append(nr)
        return merged, col_map

    # ==================================================================
    # PyMuPDF cell extraction
    # ==================================================================

    def _cells_from_pymupdf(self, table: Any, col_map: dict[int, int]) -> list[CellInfo]:
        raw = self._extract_raw_cells(table)
        if not col_map or not raw:
            return raw
        new_nc = max(col_map.values()) + 1 if col_map else 0
        seen: set[tuple[int, int]] = set()
        out: list[CellInfo] = []
        for c in raw:
            nc = col_map.get(c.col, c.col)
            if (c.row, nc) in seen:
                continue
            ncs = 1
            for oc in range(c.col, c.col + c.colspan):
                mc = col_map.get(oc, oc)
                if mc != nc:
                    ncs = max(ncs, mc - nc + 1)
            ncs = min(ncs, new_nc - nc)
            out.append(CellInfo(row=c.row, col=nc, rowspan=c.rowspan, colspan=max(1, ncs), bbox=c.bbox))
            for ri in range(c.row, c.row + c.rowspan):
                for ci in range(nc, nc + max(1, ncs)):
                    seen.add((ri, ci))
        return out

    @staticmethod
    def _extract_raw_cells(table: Any) -> list[CellInfo]:
        cells: list[CellInfo] = []
        if not hasattr(table, "cells") or not table.cells:
            return cells
        raw = table.cells
        xs = sorted({c[0] for c in raw if c} | {c[2] for c in raw if c})
        ys = sorted({c[1] for c in raw if c} | {c[3] for c in raw if c})
        if len(xs) < 2 or len(ys) < 2:
            return cells

        def idx(val: float, coords: list[float], tol: float = 3.0) -> int:
            for i, c in enumerate(coords):
                if abs(val - c) <= tol:
                    return i
            return min(range(len(coords)), key=lambda i: abs(coords[i] - val))

        seen: set[tuple[int, int]] = set()
        for cb in raw:
            if cb is None:
                continue
            cs = idx(cb[0], xs)
            ce = idx(cb[2], xs)
            rs = idx(cb[1], ys)
            re = idx(cb[3], ys)
            if (rs, cs) in seen:
                continue
            seen.add((rs, cs))
            cells.append(CellInfo(
                row=rs, col=cs,
                rowspan=max(1, re - rs), colspan=max(1, ce - cs),
                bbox=cb,
            ))
            for r in range(rs, rs + max(1, re - rs)):
                for c in range(cs, cs + max(1, ce - cs)):
                    if (r, c) != (rs, cs):
                        seen.add((r, c))
        return cells

    # ==================================================================
    # Confidence scoring
    # ==================================================================

    def _pymupdf_confidence(self, table: Any, data: list[list]) -> float:
        sc = CFG.PYMUPDF_CONFIDENCE_BASE
        nr = len(data)
        if nr >= self.MIN_TABLE_ROWS:
            sc += 0.1
        if hasattr(table, "col_count") and table.col_count >= self.MIN_TABLE_COLS:
            sc += 0.1
        tc = sum(len(r) for r in data)
        fc = sum(1 for r in data for c in r if c and str(c).strip())
        if tc:
            d = fc / tc
            if d < 0.05:
                sc -= 0.2
            elif d < 0.1:
                sc -= 0.1
            else:
                sc += d * 0.15
        if hasattr(table, "cells") and table.cells:
            sc += 0.15
        mc = sum(1 for r in data for c in r if c and len(str(c).strip()) >= 2)
        if mc < 2:
            sc -= 0.1
        vr = sum(1 for r in data if any(c and str(c).strip() for c in r))
        if vr <= 1:
            sc -= 0.1
        if self.graphic_detector.is_bbox_in_graphic_region(table.bbox, 0.5):
            sc -= 0.15
        return max(0.0, min(1.0, sc))

    def _plumber_confidence(self, data: list[list]) -> float:
        sc = CFG.PDFPLUMBER_CONFIDENCE_BASE
        nr = len(data)
        nc = max((len(r) for r in data), default=0)
        if nr >= self.MIN_TABLE_ROWS:
            sc += 0.1
        if nc >= self.MIN_TABLE_COLS:
            sc += 0.1
        tc = sum(len(r) for r in data)
        fc = sum(1 for r in data for c in r if c and str(c).strip())
        if tc:
            d = fc / tc
            if d < 0.1:
                sc -= 0.5
            elif d < 0.2:
                sc -= 0.3
            else:
                sc += d * 0.2
        mc = sum(1 for r in data for c in r if c and len(str(c).strip()) >= 2)
        if mc < 2:
            sc -= 0.3
        vr = sum(1 for r in data if any(c and str(c).strip() for c in r))
        if vr <= 1:
            sc -= 0.2
        er = nr - vr
        if nr and er / nr > 0.5:
            sc -= 0.2
        return max(0.0, min(1.0, sc))

    def _line_confidence(self, grid: GridInfo, data: list[list]) -> float:
        sc = CFG.LINE_CONFIDENCE_BASE
        if grid.is_complete:
            sc += 0.2
        elif grid.reconstructed:
            sc += 0.1
        if grid.row_count >= self.MIN_TABLE_ROWS:
            sc += 0.1
        if grid.col_count >= self.MIN_TABLE_COLS:
            sc += 0.1
        tc = sum(len(r) for r in data)
        fc = sum(1 for r in data for c in r if c and str(c).strip())
        if tc:
            d = fc / tc
            if d < 0.1:
                sc -= 0.4
            elif d < 0.2:
                sc -= 0.2
            else:
                sc += d * 0.2
        mc = sum(1 for r in data for c in r if c and len(str(c).strip()) >= 2)
        if mc < 2:
            sc -= 0.2
        vr = sum(1 for r in data if any(c and str(c).strip() for c in r))
        if vr <= 1:
            sc -= 0.2
        if self.graphic_detector.is_bbox_in_graphic_region(grid.bbox, 0.3):
            sc -= 0.3
        return max(0.0, min(1.0, sc))

    # ==================================================================
    # Validation & selection
    # ==================================================================

    def _validate(self, cands: list[TableCandidate]) -> list[TableCandidate]:
        out: list[TableCandidate] = []
        for c in cands:
            skip_gfx = c.strategy == TableDetectionStrategy.PYMUPDF_NATIVE
            ok, conf, reason = self.quality_validator.validate(
                data=c.data, bbox=c.bbox,
                cells_info=[
                    {"row": ci.row, "col": ci.col, "rowspan": ci.rowspan,
                     "colspan": ci.colspan, "bbox": ci.bbox}
                    for ci in c.cells
                ] if c.cells else None,
                skip_graphic_check=skip_gfx,
            )
            if ok:
                out.append(TableCandidate(
                    strategy=c.strategy,
                    confidence=min(c.confidence, conf),
                    bbox=c.bbox, grid=c.grid,
                    cells=c.cells, data=c.data, raw_table=c.raw_table,
                ))
            else:
                logger.debug("[TableDet] filtered: page=%d bbox=%s reason=%s",
                             self.page_num + 1, c.bbox, reason)
        return out

    def _select_best(self, cands: list[TableCandidate]) -> list[TableCandidate]:
        if not cands:
            return []
        priority = {
            TableDetectionStrategy.PYMUPDF_NATIVE: 0,
            TableDetectionStrategy.PDFPLUMBER_LINES: 1,
            TableDetectionStrategy.HYBRID_ANALYSIS: 2,
            TableDetectionStrategy.BORDERLESS_HEURISTIC: 3,
        }

        def key(c: TableCandidate) -> tuple:
            adj = c.confidence - priority.get(c.strategy, 99) * 0.15
            return (-adj, priority.get(c.strategy, 99))

        ordered = sorted(cands, key=key)
        selected: list[TableCandidate] = []
        for c in ordered:
            if not any(self._overlap(c.bbox, s.bbox) for s in selected):
                selected.append(c)
        return selected

    @staticmethod
    def _overlap(a: tuple, b: tuple, threshold: float = 0.3) -> bool:
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return False
        oa = (x1 - x0) * (y1 - y0)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        if aa <= 0 or ab <= 0:
            return False
        return oa / aa >= threshold or oa / ab >= threshold

    # ==================================================================
    # Grid text extraction helpers
    # ==================================================================

    def _text_from_grid(self, grid: GridInfo) -> list[list]:
        pd = self.page.get_text("dict", sort=True)
        data: list[list] = []
        for ri in range(grid.row_count):
            row: list[str] = []
            y0, y1 = grid.h_lines[ri], grid.h_lines[ri + 1]
            for ci in range(grid.col_count):
                x0, x1 = grid.v_lines[ci], grid.v_lines[ci + 1]
                row.append(self._text_in_bbox(pd, (x0, y0, x1, y1)))
            data.append(row)
        return data

    @staticmethod
    def _text_in_bbox(pd: dict, bbox: tuple) -> str:
        texts: list[str] = []
        for blk in pd.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for ln in blk.get("lines", []):
                lb = ln.get("bbox", (0, 0, 0, 0))
                if not (max(lb[0], bbox[0]) < min(lb[2], bbox[2])
                        and max(lb[1], bbox[1]) < min(lb[3], bbox[3])):
                    continue
                for sp in ln.get("spans", []):
                    sb = sp.get("bbox", (0, 0, 0, 0))
                    if max(sb[0], bbox[0]) < min(sb[2], bbox[2]) \
                            and max(sb[1], bbox[1]) < min(sb[3], bbox[3]):
                        t = sp.get("text", "").strip()
                        if t:
                            texts.append(t)
        return " ".join(texts)

    @staticmethod
    def _cells_from_grid(grid: GridInfo) -> list[CellInfo]:
        return [
            CellInfo(row=ri, col=ci, rowspan=1, colspan=1,
                     bbox=(grid.v_lines[ci], grid.h_lines[ri],
                           grid.v_lines[ci + 1], grid.h_lines[ri + 1]))
            for ri in range(grid.row_count)
            for ci in range(grid.col_count)
        ]

    # ==================================================================
    # pdfplumber bbox estimation
    # ==================================================================

    @staticmethod
    def _estimate_bbox_plumber(pp: Any, data: list[list]) -> Optional[tuple]:
        try:
            words = pp.extract_words()
            if not words:
                return None
            table_texts = {str(c).strip()[:20] for r in data for c in r if c and str(c).strip()}
            hits = [w for w in words if any(w["text"] in t or t in w["text"] for t in table_texts)]
            if not hits:
                return None
            m = 5
            return (
                min(w["x0"] for w in hits) - m,
                min(w["top"] for w in hits) - m,
                max(w["x1"] for w in hits) + m,
                max(w["bottom"] for w in hits) + m,
            )
        except Exception:
            return None


__all__ = ["TableDetectionEngine"]
