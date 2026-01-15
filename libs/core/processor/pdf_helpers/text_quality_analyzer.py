"""
Text Quality Analyzer for PDF Handler V3

PDF에서 추출된 텍스트의 품질을 분석하고, 깨진 텍스트(인코딩 문제, 
ToUnicode CMap 누락 등)를 감지하여 OCR 폴백 필요 여부를 판단합니다.

=============================================================================
깨진 텍스트의 특징:
=============================================================================
1. Private Use Area (PUA) 문자 다수 포함: U+E000 ~ U+F8FF
2. 대체 문자 (Replacement Character): U+FFFD (�)
3. 한글 조합 불가능한 문자 조합 (자음/모음만 연속)
4. 의미없는 한글 음절 연속 (실제 단어가 아닌 무작위 조합)
5. CJK 문자와 PUA/제어문자 혼합

=============================================================================
해결 전략:
=============================================================================
1. 텍스트 품질 점수 계산 (0.0 ~ 1.0)
2. 품질이 임계값 이하이면 OCR 폴백 수행
3. 페이지 전체 또는 특정 영역에 대해 OCR 적용
"""

import logging
import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

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

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class TextQualityConfig:
    """텍스트 품질 분석 설정"""
    
    # 품질 판단 임계값
    QUALITY_THRESHOLD = 0.7           # 이 값 이하면 OCR 폴백 (0.5 → 0.7로 상향)
    MIN_TEXT_LENGTH = 10              # 품질 분석을 위한 최소 텍스트 길이
    
    # PUA 기반 임계값 (PUA 비율이 이 이상이면 무조건 OCR)
    PUA_RATIO_THRESHOLD = 0.1         # 10% 이상이면 OCR
    
    # PUA (Private Use Area) 범위
    PUA_RANGES = [
        (0xE000, 0xF8FF),     # BMP Private Use Area
        (0xF0000, 0xFFFFD),   # Supplementary PUA-A
        (0x100000, 0x10FFFD), # Supplementary PUA-B
    ]
    
    # 제어 문자 및 특수 문자
    CONTROL_RANGES = [
        (0x0000, 0x001F),     # C0 controls
        (0x007F, 0x009F),     # C1 controls
        (0xFFF0, 0xFFFF),     # Specials
    ]
    
    # OCR 설정
    OCR_LANG = 'kor+eng'
    OCR_DPI = 300
    OCR_SCALE = 3.0
    
    # 한글 음절 범위
    HANGUL_SYLLABLE_RANGE = (0xAC00, 0xD7A3)
    HANGUL_JAMO_RANGE = (0x1100, 0x11FF)
    HANGUL_COMPAT_JAMO_RANGE = (0x3130, 0x318F)
    
    # 품질 분석용 가중치
    WEIGHT_PUA = 0.4              # PUA 문자 비율 가중치
    WEIGHT_REPLACEMENT = 0.3      # 대체 문자 가중치
    WEIGHT_VALID_RATIO = 0.3      # 유효 문자 비율 가중치


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TextQualityResult:
    """텍스트 품질 분석 결과"""
    quality_score: float          # 0.0 ~ 1.0 (높을수록 품질 좋음)
    total_chars: int              # 전체 문자 수
    pua_count: int                # PUA 문자 수
    replacement_count: int        # 대체 문자 수
    valid_chars: int              # 유효 문자 수 (한글, 영문, 숫자)
    control_chars: int            # 제어 문자 수
    needs_ocr: bool               # OCR 필요 여부
    details: Dict                 # 상세 정보


@dataclass
class PageTextAnalysis:
    """페이지 텍스트 분석 결과"""
    page_num: int
    quality_result: TextQualityResult
    text_blocks: List[Dict]       # 개별 텍스트 블록 정보
    problem_regions: List[Tuple[float, float, float, float]]  # 문제 있는 영역 bbox
    ocr_text: Optional[str] = None  # OCR 결과 (수행된 경우)


# ============================================================================
# Text Quality Analyzer
# ============================================================================

class TextQualityAnalyzer:
    """
    텍스트 품질 분석기
    
    PDF에서 추출된 텍스트의 품질을 분석하고,
    깨진 텍스트를 감지하여 OCR 폴백 필요 여부를 판단합니다.
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
        
    def analyze_page(self) -> PageTextAnalysis:
        """
        페이지 전체 텍스트 품질 분석
        
        Returns:
            PageTextAnalysis 객체
        """
        # 텍스트 딕셔너리 추출
        text_dict = self.page.get_text("dict", sort=True)
        blocks = text_dict.get("blocks", [])
        
        all_text = []
        text_blocks = []
        problem_regions = []
        
        for block in blocks:
            if block.get("type") != 0:  # 텍스트 블록만
                continue
            
            block_bbox = block.get("bbox", (0, 0, 0, 0))
            block_text = []
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text:
                        block_text.append(text)
                        all_text.append(text)
            
            if block_text:
                combined_text = " ".join(block_text)
                quality = self.analyze_text(combined_text)
                
                text_blocks.append({
                    'bbox': block_bbox,
                    'text': combined_text,
                    'quality': quality
                })
                
                # 품질이 낮은 영역 기록
                if quality.needs_ocr:
                    problem_regions.append(block_bbox)
        
        # 전체 텍스트 품질 분석
        full_text = " ".join(all_text)
        overall_quality = self.analyze_text(full_text)
        
        return PageTextAnalysis(
            page_num=self.page_num,
            quality_result=overall_quality,
            text_blocks=text_blocks,
            problem_regions=problem_regions
        )
    
    def analyze_text(self, text: str) -> TextQualityResult:
        """
        텍스트 품질 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            TextQualityResult 객체
        """
        if not text or len(text) < TextQualityConfig.MIN_TEXT_LENGTH:
            return TextQualityResult(
                quality_score=1.0,  # 텍스트가 없거나 너무 짧으면 OK로 처리
                total_chars=len(text),
                pua_count=0,
                replacement_count=0,
                valid_chars=len(text),
                control_chars=0,
                needs_ocr=False,
                details={'reason': 'text_too_short'}
            )
        
        total_chars = len(text)
        pua_count = 0
        replacement_count = 0
        control_count = 0
        valid_chars = 0  # 한글, 영문, 숫자, 공백, 기본 구두점
        
        # 문자별 분석
        for char in text:
            code = ord(char)
            
            # PUA 체크
            if self._is_pua(code):
                pua_count += 1
                continue
            
            # 대체 문자 체크
            if code == 0xFFFD:
                replacement_count += 1
                continue
            
            # 제어 문자 체크
            if self._is_control(code):
                control_count += 1
                continue
            
            # 유효 문자 체크
            if self._is_valid_char(char, code):
                valid_chars += 1
        
        # 품질 점수 계산
        quality_score = self._calculate_quality_score(
            total_chars=total_chars,
            pua_count=pua_count,
            replacement_count=replacement_count,
            valid_chars=valid_chars
        )
        
        # OCR 필요 여부 판단
        pua_ratio = pua_count / total_chars if total_chars > 0 else 0
        needs_ocr = (
            quality_score < TextQualityConfig.QUALITY_THRESHOLD or
            pua_ratio >= TextQualityConfig.PUA_RATIO_THRESHOLD
        )
        
        return TextQualityResult(
            quality_score=quality_score,
            total_chars=total_chars,
            pua_count=pua_count,
            replacement_count=replacement_count,
            valid_chars=valid_chars,
            control_chars=control_count,
            needs_ocr=needs_ocr,
            details={
                'pua_ratio': pua_count / total_chars if total_chars > 0 else 0,
                'replacement_ratio': replacement_count / total_chars if total_chars > 0 else 0,
                'valid_ratio': valid_chars / total_chars if total_chars > 0 else 0,
            }
        )
    
    def _is_pua(self, code: int) -> bool:
        """Private Use Area 문자 여부"""
        for start, end in TextQualityConfig.PUA_RANGES:
            if start <= code <= end:
                return True
        return False
    
    def _is_control(self, code: int) -> bool:
        """제어 문자 여부"""
        for start, end in TextQualityConfig.CONTROL_RANGES:
            if start <= code <= end:
                return True
        return False
    
    def _is_valid_char(self, char: str, code: int) -> bool:
        """유효 문자 여부 (한글, 영문, 숫자, 공백, 기본 구두점)"""
        # 공백
        if char.isspace():
            return True
        
        # 영문, 숫자
        if char.isalnum() and code < 128:
            return True
        
        # 한글 음절
        if TextQualityConfig.HANGUL_SYLLABLE_RANGE[0] <= code <= TextQualityConfig.HANGUL_SYLLABLE_RANGE[1]:
            return True
        
        # 한글 자모
        if TextQualityConfig.HANGUL_JAMO_RANGE[0] <= code <= TextQualityConfig.HANGUL_JAMO_RANGE[1]:
            return True
        
        # 한글 호환 자모
        if TextQualityConfig.HANGUL_COMPAT_JAMO_RANGE[0] <= code <= TextQualityConfig.HANGUL_COMPAT_JAMO_RANGE[1]:
            return True
        
        # 기본 구두점
        if char in '.,!?;:\'"()[]{}-–—…·•':
            return True
        
        # CJK 문자 (중국어, 일본어)
        if 0x4E00 <= code <= 0x9FFF:  # CJK Unified Ideographs
            return True
        
        # 일본어 히라가나/가타카나
        if 0x3040 <= code <= 0x30FF:
            return True
        
        return False
    
    def _calculate_quality_score(
        self,
        total_chars: int,
        pua_count: int,
        replacement_count: int,
        valid_chars: int
    ) -> float:
        """품질 점수 계산 (0.0 ~ 1.0)"""
        if total_chars == 0:
            return 1.0
        
        # 각 비율 계산
        pua_ratio = pua_count / total_chars
        replacement_ratio = replacement_count / total_chars
        valid_ratio = valid_chars / total_chars
        
        # 가중 점수 계산
        # PUA가 많을수록, 대체 문자가 많을수록, 유효 비율이 낮을수록 점수 하락
        score = 1.0
        
        # PUA 문자 페널티 (많을수록 감점)
        score -= pua_ratio * TextQualityConfig.WEIGHT_PUA * 2
        
        # 대체 문자 페널티
        score -= replacement_ratio * TextQualityConfig.WEIGHT_REPLACEMENT * 3
        
        # 유효 문자 비율 보정
        score = score * (0.5 + valid_ratio * 0.5)
        
        return max(0.0, min(1.0, score))


# ============================================================================
# Page OCR Fallback Engine
# ============================================================================

class PageOCRFallbackEngine:
    """
    페이지 OCR 폴백 엔진
    
    텍스트 품질이 낮은 페이지에 대해 전체 페이지 또는
    특정 영역에 대해 OCR을 수행합니다.
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
    
    def ocr_full_page(self) -> str:
        """
        전체 페이지 OCR 수행
        
        Returns:
            OCR로 추출된 텍스트
        """
        if not PYTESSERACT_AVAILABLE:
            logger.warning("[PageOCR] pytesseract not available")
            return ""
        
        try:
            # 고해상도로 페이지 렌더링
            mat = fitz.Matrix(TextQualityConfig.OCR_SCALE, TextQualityConfig.OCR_SCALE)
            pix = self.page.get_pixmap(matrix=mat)
            
            # PIL Image로 변환
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR 수행 (한국어 우선)
            ocr_config = '--psm 3 --oem 3'  # 자동 페이지 분할 + LSTM OCR
            text = pytesseract.image_to_string(
                img,
                lang=TextQualityConfig.OCR_LANG,
                config=ocr_config
            )
            
            # OCR 후처리: 노이즈 제거
            text = self._postprocess_ocr_text(text)
            
            logger.info(f"[PageOCR] Page {self.page_num + 1}: OCR extracted {len(text)} chars")
            return text.strip()
            
        except Exception as e:
            logger.error(f"[PageOCR] Page {self.page_num + 1} OCR failed: {e}")
            return ""
    
    def _postprocess_ocr_text(self, text: str) -> str:
        """
        OCR 결과 후처리
        
        - 특수 기호로만 이루어진 라인 제거
        - 너무 짧은 무의미한 라인 제거
        - 반복 문자 정리
        - OCR 노이즈 패턴 제거
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # OCR 노이즈 패턴 (배경 그래픽에서 잘못 인식된 텍스트)
        noise_patterns = [
            r'^[ri\-—maOANIUTLOG\s]+$',  # 배경 원형 그래픽에서 인식된 노이즈
            r'^[0-9"\'\[\]\(\)°\s]{1,5}$',  # 짧은 숫자/기호 조합
            r'^[A-Za-z\-—\s]{3,}$',  # 의미없는 영문 조합 (한글이 없는 경우)
            r'^‥+\s*$',  # 점선만
            r'^\s*[°·•○●□■◇◆△▲▽▼]+\s*$',  # 기호만
        ]
        
        import re
        
        for line in lines:
            line = line.strip()
            
            # 빈 라인 스킵
            if not line:
                continue
            
            # 특수 기호로만 이루어진 라인 제거
            if all(c in '.,;:!?@#$%^&*()[]{}|\\/<>~`\'"-_+=°·•○●□■◇◆△▲▽▼' or c.isspace() for c in line):
                continue
            
            # 노이즈 패턴 체크
            is_noise = False
            for pattern in noise_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_noise = True
                    break
            if is_noise:
                continue
            
            # 한글이 포함된 라인은 우선 유지
            korean_count = sum(1 for c in line if '가' <= c <= '힣')
            if korean_count > 0:
                cleaned_lines.append(line)
                continue
            
            # 영문만 있는 경우 의미있는지 확인
            alpha_count = sum(1 for c in line if c.isalpha())
            total_len = len(line.replace(' ', ''))
            
            if total_len > 0:
                meaningful_ratio = alpha_count / total_len
                # 의미있는 문자가 50% 이상이고 3글자 이상인 경우만 유지
                if meaningful_ratio >= 0.5 and alpha_count >= 3:
                    # 대문자 연속인 약어(PLATEER, IDT 등)는 유지
                    if line.isupper() or any(word.isupper() and len(word) >= 2 for word in line.split()):
                        cleaned_lines.append(line)
                    # 일반 영문 텍스트 (Insight Report 등)
                    elif any(c.islower() for c in line):
                        cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def ocr_region(self, bbox: Tuple[float, float, float, float]) -> str:
        """
        특정 영역 OCR 수행
        
        Args:
            bbox: 영역 (x0, y0, x1, y1)
            
        Returns:
            OCR로 추출된 텍스트
        """
        if not PYTESSERACT_AVAILABLE:
            return ""
        
        try:
            x0, y0, x1, y1 = bbox
            
            # 패딩 추가
            padding = 10
            clip = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(self.page_width, x1 + padding),
                min(self.page_height, y1 + padding)
            )
            
            # 고해상도로 영역 렌더링
            mat = fitz.Matrix(TextQualityConfig.OCR_SCALE, TextQualityConfig.OCR_SCALE)
            pix = self.page.get_pixmap(matrix=mat, clip=clip)
            
            # PIL Image로 변환
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # OCR 수행
            ocr_config = '--psm 6 --oem 3'  # 균일한 텍스트 블록 + LSTM
            text = pytesseract.image_to_string(
                img,
                lang=TextQualityConfig.OCR_LANG,
                config=ocr_config
            )
            
            # OCR 후처리
            text = self._postprocess_ocr_text(text)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"[PageOCR] Region OCR failed for {bbox}: {e}")
            return ""
    
    def ocr_problem_regions(
        self,
        problem_regions: List[Tuple[float, float, float, float]]
    ) -> Dict[Tuple, str]:
        """
        문제 영역들에 대해 OCR 수행
        
        Args:
            problem_regions: 문제 있는 영역 bbox 목록
            
        Returns:
            {bbox: ocr_text} 딕셔너리
        """
        results = {}
        
        for bbox in problem_regions:
            text = self.ocr_region(bbox)
            if text:
                results[bbox] = text
        
        return results


# ============================================================================
# Integrated Text Extractor with Quality Check
# ============================================================================

class QualityAwareTextExtractor:
    """
    품질 인식 텍스트 추출기
    
    텍스트 품질을 분석하고, 필요시 OCR 폴백을 수행하여
    항상 고품질의 텍스트를 추출합니다.
    """
    
    def __init__(self, page, page_num: int, quality_threshold: float = None):
        """
        Args:
            page: PyMuPDF page 객체
            page_num: 페이지 번호 (0-indexed)
            quality_threshold: 품질 임계값 (기본값: TextQualityConfig.QUALITY_THRESHOLD)
        """
        self.page = page
        self.page_num = page_num
        self.quality_threshold = quality_threshold or TextQualityConfig.QUALITY_THRESHOLD
        
        self.analyzer = TextQualityAnalyzer(page, page_num)
        self.ocr_engine = PageOCRFallbackEngine(page, page_num)
    
    def extract(self) -> Tuple[str, PageTextAnalysis]:
        """
        품질을 고려한 텍스트 추출
        
        Returns:
            (추출된 텍스트, 분석 결과) 튜플
        """
        # 1. 페이지 텍스트 품질 분석
        analysis = self.analyzer.analyze_page()
        
        logger.debug(
            f"[QualityAware] Page {self.page_num + 1}: "
            f"quality={analysis.quality_result.quality_score:.2f}, "
            f"pua={analysis.quality_result.pua_count}, "
            f"valid={analysis.quality_result.valid_chars}"
        )
        
        # 2. 품질이 좋으면 기존 텍스트 반환
        if not analysis.quality_result.needs_ocr:
            # 기존 방식으로 텍스트 추출
            text = self.page.get_text("text")
            return text, analysis
        
        # 3. 품질이 낮으면 OCR 폴백
        logger.info(
            f"[QualityAware] Page {self.page_num + 1}: "
            f"Quality too low ({analysis.quality_result.quality_score:.2f}), "
            f"falling back to OCR"
        )
        
        # 문제 영역이 적으면 해당 영역만 OCR
        if len(analysis.problem_regions) <= 3 and len(analysis.problem_regions) > 0:
            # 문제 영역만 OCR
            ocr_results = self.ocr_engine.ocr_problem_regions(analysis.problem_regions)
            
            # 기존 텍스트에서 문제 영역 텍스트를 OCR 결과로 대체
            text = self._merge_ocr_results(analysis, ocr_results)
            analysis.ocr_text = str(ocr_results)
        else:
            # 전체 페이지 OCR
            text = self.ocr_engine.ocr_full_page()
            analysis.ocr_text = text
        
        return text, analysis
    
    def _merge_ocr_results(
        self,
        analysis: PageTextAnalysis,
        ocr_results: Dict[Tuple, str]
    ) -> str:
        """
        기존 텍스트와 OCR 결과 병합
        
        좋은 품질의 블록은 그대로 사용하고,
        문제 있는 블록은 OCR 결과로 대체합니다.
        """
        merged_parts = []
        
        for block in analysis.text_blocks:
            bbox = tuple(block['bbox'])
            quality = block['quality']
            
            if quality.needs_ocr and bbox in ocr_results:
                # OCR 결과 사용
                merged_parts.append(ocr_results[bbox])
            else:
                # 기존 텍스트 사용
                merged_parts.append(block['text'])
        
        return "\n".join(merged_parts)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'TextQualityConfig',
    'TextQualityResult',
    'PageTextAnalysis',
    'TextQualityAnalyzer',
    'PageOCRFallbackEngine',
    'QualityAwareTextExtractor',
]
