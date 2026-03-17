# contextifier_new/handlers/pdf_plus/_layout_block_detector.py
"""
PDF Plus — Layout Block Detector.

Divides complex multi-column pages (newspapers, magazines, brochures)
into **semantic layout blocks** so the block-image engine can render
them individually at high resolution.

Algorithm (6 phases):
    1. Extract content elements (text, images, drawings)
    2. Detect column structure (X-clustering)
    3. Detect header / footer regions
    4. Cluster elements into semantic blocks (separator + proximity)
    5. Classify block types (article, image, sidebar, …)
    6. Optimize & determine reading order
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier.handlers.pdf_plus._types import (
    LayoutAnalysisResult,
    LayoutBlock,
    LayoutBlockType,
    PdfPlusConfig,
)

logger = logging.getLogger(__name__)

CFG = PdfPlusConfig


# ─────────────────────────────────────────────────────────────────────────────
# Internal data-classes (not exported)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _ContentElement:
    """One content element on a page."""
    element_type: str            # 'text', 'image', 'drawing'
    bbox: Tuple[float, float, float, float]
    content: Optional[str] = None
    font_size: float = 0.0
    is_bold: bool = False
    text_length: int = 0
    image_area: float = 0.0


@dataclass
class _ColumnInfo:
    """Detected column region."""
    index: int
    x_start: float
    x_end: float

    @property
    def width(self) -> float:
        return self.x_end - self.x_start


# ─────────────────────────────────────────────────────────────────────────────
# Layout Block Detector
# ─────────────────────────────────────────────────────────────────────────────


class LayoutBlockDetector:
    """
    Detect semantic layout blocks on a complex PDF page.

    Typical usage::

        detector = LayoutBlockDetector(page, page_num)
        result = detector.detect()
        for block in result.blocks:
            ...
    """

    # ── configuration shortcuts ──────────────────────────────────────────
    _GAP = CFG.LBD_GAP_THRESHOLD            # 20 pt column gap
    _HEADER_RATIO = CFG.LBD_HEADER_FOOTER_RATIO  # 10 % margin
    _HEADER_MAX_H = CFG.LBD_HEADER_MAX_HEIGHT    # 60 pt
    _VERT_DIST = CFG.LBD_VERT_CLUSTER_DIST       # 40 pt merge distance
    _HORIZ_DIST = CFG.LBD_HORIZ_CLUSTER_DIST     # 15 pt
    _MIN_BOX_AREA = CFG.LBD_MIN_BOX_AREA         # 10 000 pt²
    _MAX_BLOCKS = CFG.LBD_MAX_BLOCKS_BEFORE_MERGE  # 15

    _HEADLINE_MIN_SIZE = 14.0
    _HEADLINE_FONT_RATIO = 1.3
    _CAPTION_MAX_DIST = 30.0
    _CAPTION_MAX_H = 50.0
    _SEPARATOR_LENGTH_RATIO = 0.30
    _SEPARATOR_MAX_THICK = 3.0
    _MIN_BLOCK_W = CFG.BLOCK_MIN_REGION_WIDTH   # 80 pt
    _MIN_BLOCK_H = CFG.BLOCK_MIN_REGION_HEIGHT  # 60 pt
    _MIN_BLOCK_AREA = CFG.BLOCK_MIN_AREA        # 15 000 pt²
    _TARGET_MAX_BLOCKS = 10
    _X_CLUSTER_TOL = CFG.COLUMN_CLUSTER_TOLERANCE  # 50 pt

    def __init__(self, page: Any, page_num: int) -> None:
        self.page = page
        self.page_num = page_num
        self.page_width: float = page.rect.width
        self.page_height: float = page.rect.height

        # internal caches
        self._text_dict: Optional[Dict] = None
        self._drawings: Optional[list] = None
        self._images: Optional[list] = None

        self._elements: List[_ContentElement] = []
        self._separators: List[Tuple[float, float, float, float]] = []
        self._boxes: List[Tuple[float, float, float, float]] = []

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def detect(self) -> LayoutAnalysisResult:
        """Run full 6-phase layout detection and return the result."""
        columns: List[_ColumnInfo] = [
            _ColumnInfo(index=0, x_start=0, x_end=self.page_width),
        ]
        header_region: Optional[Tuple[float, float, float, float]] = None
        footer_region: Optional[Tuple[float, float, float, float]] = None
        blocks: List[LayoutBlock] = []

        try:
            # Phase 1 — basic analysis
            self._extract_elements()
            self._extract_separators_and_boxes()

            # Phase 2 — column detection
            columns = self._detect_columns()

            # Phase 3 — header/footer
            header_region, footer_region = self._detect_header_footer()

            # Phase 4 — semantic clustering
            try:
                blocks = self._cluster_into_blocks(
                    columns, header_region, footer_region,
                )
            except Exception:
                blocks = self._create_column_fallback(columns)

            # Phase 5 — classify block types
            self._classify_blocks(blocks)

            # Phase 6 — optimise & reading order
            blocks = self._optimize_and_sort(blocks, columns)

        except Exception as exc:
            logger.error(
                "[LayoutBlockDetector] Critical error on page %d: %s",
                self.page_num + 1, exc,
            )
            blocks = [
                LayoutBlock(
                    block_type=LayoutBlockType.UNKNOWN,
                    bbox=(0, 0, self.page_width, self.page_height),
                    elements=[],
                    confidence=0.1,
                    column_index=0,
                ),
            ]

        # assemble result
        col_tuples = [(c.x_start, c.x_end) for c in columns]
        return LayoutAnalysisResult(
            blocks=blocks,
            columns=col_tuples,
            has_header=header_region is not None,
            has_footer=footer_region is not None,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Phase 1 — basic analysis
    # ─────────────────────────────────────────────────────────────────────

    def _extract_elements(self) -> None:
        self._elements.clear()
        text_dict = self._get_text_dict()
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
            max_fs = 0.0
            bold = False
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    fs = span.get("size", 0.0)
                    if fs > max_fs:
                        max_fs = fs
                    if span.get("flags", 0) & (1 << 4):
                        bold = True
                    text += span.get("text", "")
            stripped = text.strip()
            if stripped:
                self._elements.append(_ContentElement(
                    element_type="text",
                    bbox=bbox,
                    content=stripped,
                    font_size=max_fs,
                    is_bold=bold,
                    text_length=len(stripped),
                ))

        for img_info in self._get_images():
            xref = img_info[0]
            try:
                for rect in self.page.get_image_rects(xref):
                    b = (rect.x0, rect.y0, rect.x1, rect.y1)
                    area = (b[2] - b[0]) * (b[3] - b[1])
                    self._elements.append(_ContentElement(
                        element_type="image",
                        bbox=b,
                        image_area=area,
                    ))
                    break  # first occurrence only
            except Exception:
                pass

    def _extract_separators_and_boxes(self) -> None:
        self._separators.clear()
        self._boxes.clear()
        for drawing in self._get_drawings():
            try:
                rect = drawing.get("rect")
                if rect is None:
                    continue
                try:
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                    w, h = rect.width, rect.height
                except (AttributeError, TypeError):
                    if isinstance(rect, (list, tuple)) and len(rect) >= 4:
                        x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
                        w, h = x1 - x0, y1 - y0
                    else:
                        continue

                # horizontal separator
                if (h <= self._SEPARATOR_MAX_THICK
                        and w >= self.page_width * self._SEPARATOR_LENGTH_RATIO):
                    self._separators.append((x0, y0, x1, y1))
                # vertical separator
                elif (w <= self._SEPARATOR_MAX_THICK
                      and h >= self.page_height * self._SEPARATOR_LENGTH_RATIO * 0.5):
                    self._separators.append((x0, y0, x1, y1))
                # box (advertisement / infobox candidate)
                elif w > 50 and h > 50 and w * h >= self._MIN_BOX_AREA:
                    so = drawing.get("stroke_opacity")
                    has_stroke = drawing.get("color") or (so is not None and so > 0)
                    if has_stroke:
                        self._boxes.append((x0, y0, x1, y1))
            except Exception:
                continue

    # ─────────────────────────────────────────────────────────────────────
    # Phase 2 — column detection
    # ─────────────────────────────────────────────────────────────────────

    def _detect_columns(self) -> List[_ColumnInfo]:
        x_starts = [
            e.bbox[0]
            for e in self._elements
            if e.element_type == "text" and e.text_length > 20
        ]
        if not x_starts:
            return [_ColumnInfo(0, 0, self.page_width)]

        x_starts.sort()
        clusters = self._cluster_x_positions(x_starts)
        if len(clusters) <= 1:
            return [_ColumnInfo(0, 0, self.page_width)]

        centers = [sum(c) / len(c) for c in clusters]
        boundaries = [0.0]
        for i in range(len(centers) - 1):
            if centers[i + 1] - centers[i] >= self._GAP:
                boundaries.append((centers[i] + centers[i + 1]) / 2)
        boundaries.append(self.page_width)

        return [
            _ColumnInfo(index=i, x_start=boundaries[i], x_end=boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        ]

    def _cluster_x_positions(self, xs: List[float]) -> List[List[float]]:
        if not xs:
            return []
        clusters: List[List[float]] = []
        cur = [xs[0]]
        for x in xs[1:]:
            if x - cur[-1] <= self._X_CLUSTER_TOL:
                cur.append(x)
            else:
                if len(cur) >= 3:
                    clusters.append(cur)
                cur = [x]
        if len(cur) >= 3:
            clusters.append(cur)
        return clusters

    # ─────────────────────────────────────────────────────────────────────
    # Phase 3 — header / footer
    # ─────────────────────────────────────────────────────────────────────

    def _detect_header_footer(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float, float]],
               Optional[Tuple[float, float, float, float]]]:
        h_bound = self.page_height * self._HEADER_RATIO
        f_bound = self.page_height * (1 - self._HEADER_RATIO)

        header = self._region_for_band(0, h_bound)
        footer = self._region_for_band(f_bound, self.page_height)
        return header, footer

    def _region_for_band(
        self, y_min: float, y_max: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        elems = [
            e for e in self._elements
            if e.element_type == "text"
            and e.bbox[1] >= y_min - 5 and e.bbox[3] <= y_max + 5
        ]
        if not elems:
            return None
        min_y = min(e.bbox[1] for e in elems)
        max_y = max(e.bbox[3] for e in elems)
        if max_y - min_y <= self._HEADER_MAX_H:
            return (0, min_y, self.page_width, max_y)
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Phase 4 — semantic block clustering
    # ─────────────────────────────────────────────────────────────────────

    def _cluster_into_blocks(
        self,
        columns: List[_ColumnInfo],
        header: Optional[Tuple[float, float, float, float]],
        footer: Optional[Tuple[float, float, float, float]],
    ) -> List[LayoutBlock]:
        blocks: List[LayoutBlock] = []

        # partition elements
        main_elems: List[_ContentElement] = []
        header_elems: List[_ContentElement] = []
        footer_elems: List[_ContentElement] = []
        for e in self._elements:
            if header and self._is_inside(e.bbox, header):
                header_elems.append(e)
            elif footer and self._is_inside(e.bbox, footer):
                footer_elems.append(e)
            else:
                main_elems.append(e)

        if header_elems:
            blocks.append(LayoutBlock(
                block_type=LayoutBlockType.HEADER,
                bbox=self._merge_bboxes([e.bbox for e in header_elems]),
                elements=[],
                confidence=0.9,
            ))

        # process per column
        for col in columns:
            col_elems = [e for e in main_elems if self._elem_in_col(e, col)]
            if not col_elems:
                continue
            vert_groups = self._split_by_separators(col_elems, col)
            for group in vert_groups:
                if not group:
                    continue
                for cluster in self._cluster_adjacent(group):
                    if not cluster:
                        continue
                    bbox = self._merge_bboxes([e.bbox for e in cluster])
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self._MIN_BLOCK_W or h < self._MIN_BLOCK_H:
                        continue
                    blocks.append(LayoutBlock(
                        block_type=LayoutBlockType.UNKNOWN,
                        bbox=bbox,
                        elements=[],
                        confidence=0.7,
                        column_index=col.index,
                    ))

        if footer_elems:
            blocks.append(LayoutBlock(
                block_type=LayoutBlockType.FOOTER,
                bbox=self._merge_bboxes([e.bbox for e in footer_elems]),
                elements=[],
                confidence=0.9,
            ))

        return blocks

    def _create_column_fallback(
        self, columns: List[_ColumnInfo],
    ) -> List[LayoutBlock]:
        blocks: List[LayoutBlock] = []
        for col in columns:
            col_elems = [e for e in self._elements if self._elem_in_col(e, col)]
            if col_elems:
                blocks.append(LayoutBlock(
                    block_type=LayoutBlockType.COLUMN_BLOCK,
                    bbox=self._merge_bboxes([e.bbox for e in col_elems]),
                    elements=[],
                    confidence=0.5,
                    column_index=col.index,
                ))
        if not blocks:
            blocks.append(LayoutBlock(
                block_type=LayoutBlockType.UNKNOWN,
                bbox=(0, 0, self.page_width, self.page_height),
                elements=[],
                confidence=0.1,
            ))
        return blocks

    # ── splitting / clustering helpers ───────────────────────────────────

    def _split_by_separators(
        self,
        elements: List[_ContentElement],
        col: _ColumnInfo,
    ) -> List[List[_ContentElement]]:
        sep_ys = []
        for sep in self._separators:
            is_h = abs(sep[3] - sep[1]) < 5
            if is_h and sep[0] <= col.x_end and sep[2] >= col.x_start:
                sep_ys.append(sep[1])
        if not sep_ys:
            return [elements]
        sep_ys.sort()
        bounds = [0.0] + sep_ys + [self.page_height]
        groups: List[List[_ContentElement]] = []
        for i in range(len(bounds) - 1):
            g = [
                e for e in elements
                if e.bbox[1] >= bounds[i] - 5 and e.bbox[3] <= bounds[i + 1] + 5
            ]
            if g:
                groups.append(g)
        return groups or [elements]

    def _cluster_adjacent(
        self, elements: List[_ContentElement],
    ) -> List[List[_ContentElement]]:
        if len(elements) <= 1:
            return [elements]
        sorted_elems = sorted(elements, key=lambda e: (e.bbox[1], e.bbox[0]))
        clusters: List[List[_ContentElement]] = []
        used: Set[int] = set()
        for elem in sorted_elems:
            eid = id(elem)
            if eid in used:
                continue
            cluster = [elem]
            used.add(eid)
            queue = [elem]
            while queue:
                cur = queue.pop(0)
                for other in sorted_elems:
                    oid = id(other)
                    if oid in used:
                        continue
                    if self._are_adjacent(cur, other):
                        cluster.append(other)
                        used.add(oid)
                        queue.append(other)
            clusters.append(cluster)
        return clusters

    def _are_adjacent(self, e1: _ContentElement, e2: _ContentElement) -> bool:
        vgap = max(0.0, e2.bbox[1] - e1.bbox[3], e1.bbox[1] - e2.bbox[3])
        xo_s = max(e1.bbox[0], e2.bbox[0])
        xo_e = min(e1.bbox[2], e2.bbox[2])
        if xo_s < xo_e and vgap <= self._VERT_DIST:
            return True
        hgap = max(0.0, e2.bbox[0] - e1.bbox[2], e1.bbox[0] - e2.bbox[2])
        yo_s = max(e1.bbox[1], e2.bbox[1])
        yo_e = min(e1.bbox[3], e2.bbox[3])
        if yo_s < yo_e and hgap <= self._HORIZ_DIST:
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Phase 5 — classify
    # ─────────────────────────────────────────────────────────────────────

    def _classify_blocks(self, blocks: List[LayoutBlock]) -> None:
        for block in blocks:
            if block.block_type in (LayoutBlockType.HEADER, LayoutBlockType.FOOTER):
                continue
            block.block_type = self._determine_block_type(block)

    def _determine_block_type(self, block: LayoutBlock) -> LayoutBlockType:
        # re-collect elements that fall inside the block bbox
        texts = [
            e for e in self._elements
            if e.element_type == "text" and self._is_inside(e.bbox, block.bbox, 5)
        ]
        images = [
            e for e in self._elements
            if e.element_type == "image" and self._is_inside(e.bbox, block.bbox, 5)
        ]

        has_text = bool(texts)
        has_image = bool(images)

        if has_image and has_text:
            return LayoutBlockType.IMAGE_WITH_CAPTION
        if has_image:
            return LayoutBlockType.STANDALONE_IMAGE
        if has_text:
            max_fs = max((t.font_size for t in texts), default=0)
            avg_fs = (
                sum(t.font_size for t in texts) / len(texts) if texts else 0
            )
            if max_fs >= self._HEADLINE_MIN_SIZE and max_fs >= avg_fs * self._HEADLINE_FONT_RATIO:
                return LayoutBlockType.ARTICLE
            if self._bbox_in_box(block.bbox):
                total_len = sum(t.text_length for t in texts)
                return (
                    LayoutBlockType.ADVERTISEMENT
                    if total_len < 200
                    else LayoutBlockType.SIDEBAR
                )
            return LayoutBlockType.ARTICLE
        return LayoutBlockType.UNKNOWN

    # ─────────────────────────────────────────────────────────────────────
    # Phase 6 — optimise & sort
    # ─────────────────────────────────────────────────────────────────────

    def _optimize_and_sort(
        self,
        blocks: List[LayoutBlock],
        columns: List[_ColumnInfo],
    ) -> List[LayoutBlock]:
        if not blocks:
            return blocks

        # merge small blocks
        blocks = self._merge_small(blocks)

        # reading order: header → column-L-to-R + top-to-bottom → footer
        headers = [b for b in blocks if b.block_type == LayoutBlockType.HEADER]
        footers = [b for b in blocks if b.block_type == LayoutBlockType.FOOTER]
        main = [
            b for b in blocks
            if b.block_type not in (LayoutBlockType.HEADER, LayoutBlockType.FOOTER)
        ]

        col_groups: Dict[int, List[LayoutBlock]] = defaultdict(list)
        for b in main:
            col_groups[b.column_index].append(b)
        for cg in col_groups.values():
            cg.sort(key=lambda b: b.bbox[1])

        ordered: List[LayoutBlock] = list(headers)
        for ci in sorted(col_groups):
            ordered.extend(col_groups[ci])
        ordered.extend(footers)
        return ordered

    def _merge_small(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        if len(blocks) <= self._TARGET_MAX_BLOCKS:
            return blocks

        aggressive = len(blocks) > self._MAX_BLOCKS
        min_area = self._MIN_BLOCK_AREA * (2 if aggressive else 1)

        result: List[LayoutBlock] = []
        skip: Set[int] = set()

        for i, block in enumerate(blocks):
            if i in skip:
                continue
            area = (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])
            if area >= min_area:
                result.append(block)
                continue
            merged = False
            for j, other in enumerate(blocks):
                if j == i or j in skip:
                    continue
                if self._should_merge(block, other, aggressive):
                    other.bbox = self._merge_bboxes([block.bbox, other.bbox])
                    skip.add(i)
                    merged = True
                    break
            if not merged:
                result.append(block)

        # if still over target, force-merge within columns
        if len(result) > self._TARGET_MAX_BLOCKS:
            result = self._force_merge(result)
        return result

    def _should_merge(
        self, b1: LayoutBlock, b2: LayoutBlock, aggressive: bool,
    ) -> bool:
        if not aggressive and b1.column_index != b2.column_index:
            return False
        if aggressive and abs(b1.column_index - b2.column_index) > 1:
            return False
        vgap = max(
            0.0,
            b2.bbox[1] - b1.bbox[3],
            b1.bbox[1] - b2.bbox[3],
        )
        threshold = self._VERT_DIST * (3 if aggressive else 2)
        return vgap <= threshold

    def _force_merge(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        col_groups: Dict[int, List[LayoutBlock]] = defaultdict(list)
        for b in blocks:
            col_groups[b.column_index].append(b)
        result: List[LayoutBlock] = []
        for ci in sorted(col_groups):
            cbs = sorted(col_groups[ci], key=lambda b: b.bbox[1])
            if len(cbs) <= 2:
                result.extend(cbs)
                continue
            target_groups = max(2, min(3, len(cbs) // 2))
            per = max(1, len(cbs) // target_groups)
            group: List[LayoutBlock] = []
            for i, b in enumerate(cbs):
                group.append(b)
                if len(group) >= per and len(result) < len(col_groups) * target_groups - 1:
                    result.append(self._merge_group(group))
                    group = []
            if group:
                result.append(self._merge_group(group))
        return result

    @staticmethod
    def _merge_group(blocks: List[LayoutBlock]) -> LayoutBlock:
        if len(blocks) == 1:
            return blocks[0]
        bboxes = [b.bbox for b in blocks]
        merged_bbox = (
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        )
        return LayoutBlock(
            block_type=blocks[0].block_type,
            bbox=merged_bbox,
            elements=[],
            confidence=min(b.confidence for b in blocks),
            column_index=blocks[0].column_index,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────────────────────────────────

    def _elem_in_col(self, elem: _ContentElement, col: _ColumnInfo) -> bool:
        cx = (elem.bbox[0] + elem.bbox[2]) / 2
        return col.x_start <= cx <= col.x_end

    @staticmethod
    def _is_inside(
        inner: Tuple[float, float, float, float],
        outer: Tuple[float, float, float, float],
        margin: float = 0,
    ) -> bool:
        return (
            inner[0] >= outer[0] - margin
            and inner[1] >= outer[1] - margin
            and inner[2] <= outer[2] + margin
            and inner[3] <= outer[3] + margin
        )

    def _bbox_in_box(self, bbox: Tuple[float, float, float, float]) -> bool:
        return any(self._is_inside(bbox, box, 10) for box in self._boxes)

    @staticmethod
    def _merge_bboxes(
        bboxes: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float, float, float]:
        if not bboxes:
            return (0, 0, 0, 0)
        return (
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Caching
    # ─────────────────────────────────────────────────────────────────────

    def _get_text_dict(self) -> Dict:
        if self._text_dict is None:
            self._text_dict = self.page.get_text("dict", sort=True)
        return self._text_dict

    def _get_drawings(self) -> list:
        if self._drawings is None:
            self._drawings = self.page.get_drawings()
        return self._drawings

    def _get_images(self) -> list:
        if self._images is None:
            self._images = self.page.get_images()
        return self._images


__all__ = ["LayoutBlockDetector"]
