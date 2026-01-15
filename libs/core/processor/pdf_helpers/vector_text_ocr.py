"""
Vector Text OCR Engine for PDF Handler V3

PDF에서 텍스트가 폰트 글리프가 아닌 벡터 곡선(Bézier curves)으로
렌더링된 영역을 감지하고 OCR로 텍스트를 추출합니다.
"""

import io
import logging
from typing import List, Dict, Tuple, Optional

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from .v3_types import VectorTextRegion

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration for Vector Text OCR
# ============================================================================

class VectorTextConfig:
    """벡터 텍스트 OCR 설정"""
    MAX_HEIGHT = 50.0           # 벡터 텍스트 영역 최대 높이
    MIN_ITEMS = 5               # 최소 드로잉 아이템 수
    OCR_SCALE = 3.0             # OCR용 렌더링 배율
    OCR_LANG = 'kor+eng'        # OCR 언어


# ============================================================================
# Vector Text OCR Engine
# ============================================================================

class VectorTextOCREngine:
    """
    벡터 텍스트 OCR 엔진
    
    PDF에서 텍스트가 폰트 글리프가 아닌 벡터 곡선(Bézier curves)으로
    렌더링된 영역을 감지하고 OCR로 텍스트를 추출합니다.
    
    왜 필요한가?
    - 일부 PDF는 폰트 임베딩 문제를 피하기 위해 텍스트를 아웃라인으로 변환
    - 디자인 프로그램(Illustrator, InDesign 등)에서 "Create Outlines" 적용
    - 이 경우 일반 텍스트 추출로는 내용을 얻을 수 없음
    """
    
    def __init__(self, page, page_num: int):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed)
        """
        self.page = page
        self.page_num = page_num
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.vector_regions: List[VectorTextRegion] = []
        
    def detect_and_extract(self) -> List[VectorTextRegion]:
        """
        벡터 텍스트 영역을 감지하고 OCR로 추출
        
        Returns:
            VectorTextRegion 목록 (OCR 텍스트 포함)
        """
        if not PYTESSERACT_AVAILABLE:
            logger.debug("[VectorTextOCR] pytesseract not available, skipping")
            return []
        
        if not PYMUPDF_AVAILABLE:
            logger.debug("[VectorTextOCR] PyMuPDF not available, skipping")
            return []
        
        # 1. 벡터 텍스트 영역 감지
        self._detect_vector_text_regions()
        
        if not self.vector_regions:
            return []
        
        logger.info(f"[VectorTextOCR] Page {self.page_num + 1}: Found {len(self.vector_regions)} vector text regions")
        
        # 2. 각 영역에 대해 OCR 수행
        for region in self.vector_regions:
            self._ocr_region(region)
        
        # 3. OCR 결과가 있는 영역만 반환
        valid_regions = [r for r in self.vector_regions if r.ocr_text.strip()]
        logger.info(f"[VectorTextOCR] Page {self.page_num + 1}: Extracted text from {len(valid_regions)} regions")
        
        return valid_regions
    
    def _detect_vector_text_regions(self):
        """
        벡터 텍스트 영역 감지
        
        벡터 텍스트의 특징:
        1. drawings에 많은 수의 items (글자 획 하나하나가 path)
        2. 비교적 좁은 높이 (텍스트 높이 수준)
        3. 해당 영역에 실제 텍스트가 없거나 매우 적음
        """
        drawings = self.page.get_drawings()
        if not drawings:
            return
        
        # 텍스트 블록 영역 수집 (벡터 텍스트 vs 실제 텍스트 비교용)
        text_dict = self.page.get_text("dict")
        text_blocks = text_dict.get("blocks", [])
        text_bboxes = []
        for block in text_blocks:
            if block.get("type") == 0:  # 텍스트 블록
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text and len(text) > 1:  # 의미있는 텍스트
                            text_bboxes.append((span.get("bbox"), text))
        
        # Drawing 그룹화 (인접한 drawing을 하나의 영역으로)
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
            
            # 곡선 수 계산
            curve_count = sum(1 for item in items if item[0] == 'c')
            fill = drawing.get("fill")
            
            # 벡터 텍스트 조건:
            # 1. 높이가 텍스트 수준 (VectorTextConfig.MAX_HEIGHT 이하)
            # 2. items 수가 많음 (글자 획)
            # 3. 너비 대비 높이가 작음 (텍스트 라인 형태)
            if (height <= VectorTextConfig.MAX_HEIGHT and 
                item_count >= VectorTextConfig.MIN_ITEMS and
                width > height * 2):
                
                # 해당 영역에 실제 텍스트가 있는지 확인
                has_real_text = self._has_text_in_region((x0, y0, x1, y1), text_bboxes)
                
                if not has_real_text:
                    potential_regions.append({
                        'bbox': (x0, y0, x1, y1),
                        'item_count': item_count,
                        'curve_count': curve_count,
                        'fill_count': 1 if fill else 0
                    })
        
        # 인접 영역 병합
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
        """해당 영역에 실제 텍스트가 있는지 확인"""
        x0, y0, x1, y1 = bbox
        
        for text_bbox, text in text_bboxes:
            if not text_bbox:
                continue
            tx0, ty0, tx1, ty1 = text_bbox
            
            # 영역 겹침 확인
            if (x0 <= tx1 and x1 >= tx0 and y0 <= ty1 and y1 >= ty0):
                # 충분한 텍스트가 있으면 True
                if len(text) >= 3:
                    return True
        
        return False
    
    def _merge_adjacent_regions(self, regions: List[Dict]) -> List[Dict]:
        """인접한 벡터 텍스트 영역 병합"""
        if not regions:
            return []
        
        # Y 좌표로 정렬
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
                # 같은 라인에 있고 인접한 경우 병합
                c_x0, c_y0, c_x1, c_y1 = current['bbox']
                r_x0, r_y0, r_x1, r_y1 = region['bbox']
                
                # Y 좌표가 비슷하고 (같은 라인) X가 인접한 경우
                y_overlap = abs(c_y0 - r_y0) < 5 and abs(c_y1 - r_y1) < 5
                x_adjacent = r_x0 - c_x1 < 20  # 20pt 이내면 인접
                
                if y_overlap and x_adjacent:
                    # 병합
                    current['bbox'][0] = min(c_x0, r_x0)
                    current['bbox'][2] = max(c_x1, r_x1)
                    current['bbox'][1] = min(c_y0, r_y0)
                    current['bbox'][3] = max(c_y1, r_y1)
                    current['item_count'] += region['item_count']
                    current['curve_count'] += region.get('curve_count', 0)
                    current['fill_count'] += region.get('fill_count', 0)
                    current['drawing_count'] += 1
                else:
                    # 새 영역
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
        """특정 영역에 대해 OCR 수행"""
        try:
            x0, y0, x1, y1 = region.bbox
            
            # 약간의 패딩 추가
            padding = 5
            clip = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(self.page_width, x1 + padding),
                min(self.page_height, y1 + padding)
            )
            
            # 고해상도로 렌더링
            mat = fitz.Matrix(VectorTextConfig.OCR_SCALE, VectorTextConfig.OCR_SCALE)
            pix = self.page.get_pixmap(matrix=mat, clip=clip)
            
            # PIL Image로 변환
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR 수행
            ocr_config = '--psm 7'  # 단일 텍스트 라인으로 처리
            text = pytesseract.image_to_string(
                img, 
                lang=VectorTextConfig.OCR_LANG,
                config=ocr_config
            )
            
            region.ocr_text = text.strip()
            
            # 신뢰도 계산 (간단한 휴리스틱)
            if region.ocr_text:
                # 한글/영문 비율로 신뢰도 추정
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
