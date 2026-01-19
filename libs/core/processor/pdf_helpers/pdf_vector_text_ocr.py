"""
Vector Text OCR Engine for PDF Handler

Detects regions in PDFs where text is rendered as vector curves (Bézier curves)
rather than font glyphs, and extracts text using OCR.
"""

import io
import logging
from typing import List, Dict, Tuple, Optional

import fitz
from PIL import Image
import pytesseract

from libs.core.processor.pdf_helpers.types import VectorTextRegion

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration for Vector Text OCR
# ============================================================================

class VectorTextConfig:
    """Vector text OCR configuration settings."""
    MAX_HEIGHT = 50.0           # Maximum height for vector text regions
    MIN_ITEMS = 5               # Minimum number of drawing items
    OCR_SCALE = 3.0             # Rendering scale factor for OCR
    OCR_LANG = 'kor+eng'        # OCR language


# ============================================================================
# Vector Text OCR Engine
# ============================================================================

class VectorTextOCREngine:
    """
    Vector Text OCR Engine
    
    Detects regions in PDFs where text is rendered as vector curves (Bézier curves)
    rather than font glyphs, and extracts text using OCR.
    
    Why is this needed?
    - Some PDFs convert text to outlines to avoid font embedding issues
    - Design programs (Illustrator, InDesign, etc.) apply "Create Outlines"
    - In these cases, regular text extraction cannot retrieve the content
    """
    
    def __init__(self, page, page_num: int):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
        """
        self.page = page
        self.page_num = page_num
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.vector_regions: List[VectorTextRegion] = []
        
    def detect_and_extract(self) -> List[VectorTextRegion]:
        """
        Detect vector text regions and extract using OCR.
        
        Returns:
            List of VectorTextRegion (including OCR text)
        """
        # 1. Detect vector text regions
        self._detect_vector_text_regions()
        
        if not self.vector_regions:
            return []
        
        logger.info(f"[VectorTextOCR] Page {self.page_num + 1}: Found {len(self.vector_regions)} vector text regions")
        
        # 2. Perform OCR for each region
        for region in self.vector_regions:
            self._ocr_region(region)
        
        # 3. Return only regions with OCR results
        valid_regions = [r for r in self.vector_regions if r.ocr_text.strip()]
        logger.info(f"[VectorTextOCR] Page {self.page_num + 1}: Extracted text from {len(valid_regions)} regions")
        
        return valid_regions
    
    def _detect_vector_text_regions(self):
        """
        Detect vector text regions.
        
        Characteristics of vector text:
        1. Many items in drawings (each character stroke is a path)
        2. Relatively narrow height (text height level)
        3. No or very little actual text in that region
        """
        drawings = self.page.get_drawings()
        if not drawings:
            return
        
        # Collect text block areas (for comparing vector text vs actual text)
        text_dict = self.page.get_text("dict")
        text_blocks = text_dict.get("blocks", [])
        text_bboxes = []
        for block in text_blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text and len(text) > 1:  # Meaningful text
                            text_bboxes.append((span.get("bbox"), text))
        
        # Group drawings (merge adjacent drawings into one region)
        potential_regions: List[Dict] = []
        
        for drawing in drawings:
            rect = drawing.get("rect")
            items = drawing.get("items", [])
            
            if not rect or not items:
                continue
            
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            height = y1 - y0
            width = x1 - x0
            item_count = len(items)
            
            # Count curves
            curve_count = sum(1 for item in items if item[0] == 'c')
            fill = drawing.get("fill")
            
            # Vector text conditions:
            # 1. Height at text level (below VectorTextConfig.MAX_HEIGHT)
            # 2. Many items (character strokes)
            # 3. Small height relative to width (text line shape)
            if (height <= VectorTextConfig.MAX_HEIGHT and 
                item_count >= VectorTextConfig.MIN_ITEMS and
                width > height * 2):
                
                # Check if actual text exists in this region
                has_real_text = self._has_text_in_region((x0, y0, x1, y1), text_bboxes)
                
                if not has_real_text:
                    potential_regions.append({
                        'bbox': (x0, y0, x1, y1),
                        'item_count': item_count,
                        'curve_count': curve_count,
                        'fill_count': 1 if fill else 0
                    })
        
        # Merge adjacent regions
        merged_regions = self._merge_adjacent_regions(potential_regions)
        
        for region_data in merged_regions:
            self.vector_regions.append(VectorTextRegion(
                bbox=region_data['bbox'],
                drawing_count=region_data.get('drawing_count', 1),
                curve_count=region_data.get('curve_count', 0),
                fill_count=region_data.get('fill_count', 0)
            ))
    
    def _has_text_in_region(self, bbox: Tuple[float, float, float, float], 
                           text_bboxes: List[Tuple]) -> bool:
        """Check if actual text exists in the specified region."""
        x0, y0, x1, y1 = bbox
        
        for text_bbox, text in text_bboxes:
            if not text_bbox:
                continue
            tx0, ty0, tx1, ty1 = text_bbox
            
            # Check region overlap
            if (x0 <= tx1 and x1 >= tx0 and y0 <= ty1 and y1 >= ty0):
                # True if there is sufficient text
                if len(text) >= 3:
                    return True
        
        return False
    
    def _merge_adjacent_regions(self, regions: List[Dict]) -> List[Dict]:
        """Merge adjacent vector text regions."""
        if not regions:
            return []
        
        # Sort by Y coordinate
        sorted_regions = sorted(regions, key=lambda r: (r['bbox'][1], r['bbox'][0]))
        
        merged = []
        current = None
        
        for region in sorted_regions:
            if current is None:
                current = {
                    'bbox': list(region['bbox']),
                    'item_count': region['item_count'],
                    'curve_count': region.get('curve_count', 0),
                    'fill_count': region.get('fill_count', 0),
                    'drawing_count': 1
                }
            else:
                # Merge if on the same line and adjacent
                c_x0, c_y0, c_x1, c_y1 = current['bbox']
                r_x0, r_y0, r_x1, r_y1 = region['bbox']
                
                # Similar Y coordinates (same line) and adjacent X
                y_overlap = abs(c_y0 - r_y0) < 5 and abs(c_y1 - r_y1) < 5
                x_adjacent = r_x0 - c_x1 < 20  # Adjacent if within 20pt
                
                if y_overlap and x_adjacent:
                    # Merge
                    current['bbox'][0] = min(c_x0, r_x0)
                    current['bbox'][2] = max(c_x1, r_x1)
                    current['bbox'][1] = min(c_y0, r_y0)
                    current['bbox'][3] = max(c_y1, r_y1)
                    current['item_count'] += region['item_count']
                    current['curve_count'] += region.get('curve_count', 0)
                    current['fill_count'] += region.get('fill_count', 0)
                    current['drawing_count'] += 1
                else:
                    # New region
                    merged.append({
                        'bbox': tuple(current['bbox']), 
                        'item_count': current['item_count'],
                        'curve_count': current['curve_count'],
                        'fill_count': current['fill_count'],
                        'drawing_count': current['drawing_count']
                    })
                    current = {
                        'bbox': list(region['bbox']),
                        'item_count': region['item_count'],
                        'curve_count': region.get('curve_count', 0),
                        'fill_count': region.get('fill_count', 0),
                        'drawing_count': 1
                    }
        
        if current:
            merged.append({
                'bbox': tuple(current['bbox']), 
                'item_count': current['item_count'],
                'curve_count': current['curve_count'],
                'fill_count': current['fill_count'],
                'drawing_count': current['drawing_count']
            })
        
        return merged
    
    def _ocr_region(self, region: VectorTextRegion):
        """Perform OCR on a specific region."""
        try:
            x0, y0, x1, y1 = region.bbox
            
            # Add slight padding
            padding = 5
            clip = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(self.page_width, x1 + padding),
                min(self.page_height, y1 + padding)
            )
            
            # Render at high resolution
            mat = fitz.Matrix(VectorTextConfig.OCR_SCALE, VectorTextConfig.OCR_SCALE)
            pix = self.page.get_pixmap(matrix=mat, clip=clip)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            ocr_config = '--psm 7'  # Treat as single text line
            text = pytesseract.image_to_string(
                img, 
                lang=VectorTextConfig.OCR_LANG,
                config=ocr_config
            )
            
            region.ocr_text = text.strip()
            
            # Calculate confidence (simple heuristic)
            if region.ocr_text:
                # Estimate confidence by Korean/English character ratio
                def is_korean(c: str) -> bool:
                    return '가' <= c <= '힣' or 'ㄱ' <= c <= 'ㅎ' or 'ㅏ' <= c <= 'ㅣ'
                valid_chars = sum(1 for c in region.ocr_text if c.isalnum() or is_korean(c))
                total_chars = len(region.ocr_text)
                region.confidence = valid_chars / total_chars if total_chars > 0 else 0.0
            
            logger.debug(f"[VectorTextOCR] Region {region.bbox}: OCR='{region.ocr_text[:50]}...' conf={region.confidence:.2f}")
            
        except Exception as e:
            logger.warning(f"[VectorTextOCR] OCR failed for region {region.bbox}: {e}")
            region.ocr_text = ""
            region.confidence = 0.0


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'VectorTextConfig',
    'VectorTextOCREngine',
]
