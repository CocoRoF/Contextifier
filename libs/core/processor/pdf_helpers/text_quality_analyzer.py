"""
Text Quality Analyzer for PDF Handler

Analyzes the quality of text extracted from PDF and detects broken text
(encoding issues, missing ToUnicode CMap, etc.) to determine whether OCR fallback is needed.

=============================================================================
Characteristics of Broken Text:
=============================================================================
1. Contains many Private Use Area (PUA) characters: U+E000 ~ U+F8FF
2. Replacement Character: U+FFFD (�)
3. Invalid Korean character combinations (only consonants/vowels in sequence)
4. Meaningless Korean syllable sequences (random combinations, not real words)
5. Mixture of CJK characters with PUA/control characters

=============================================================================
Resolution Strategy:
=============================================================================
1. Calculate text quality score (0.0 ~ 1.0)
2. Perform OCR fallback if quality is below threshold
3. Apply OCR to entire page or specific regions
"""

import logging
import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

import fitz
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class TextQualityConfig:
    """Text quality analysis configuration."""
    
    # Quality threshold
    QUALITY_THRESHOLD = 0.7           # OCR fallback if below this value (raised from 0.5 to 0.7)
    MIN_TEXT_LENGTH = 10              # Minimum text length for quality analysis
    
    # PUA-based threshold (force OCR if PUA ratio is above this)
    PUA_RATIO_THRESHOLD = 0.1         # 10% or more triggers OCR
    
    # PUA (Private Use Area) ranges
    PUA_RANGES = [
        (0xE000, 0xF8FF),     # BMP Private Use Area
        (0xF0000, 0xFFFFD),   # Supplementary PUA-A
        (0x100000, 0x10FFFD), # Supplementary PUA-B
    ]
    
    # Control characters and special characters
    CONTROL_RANGES = [
        (0x0000, 0x001F),     # C0 controls
        (0x007F, 0x009F),     # C1 controls
        (0xFFF0, 0xFFFF),     # Specials
    ]
    
    # OCR settings
    OCR_LANG = 'kor+eng'
    OCR_DPI = 300
    OCR_SCALE = 3.0
    
    # Korean syllable ranges
    HANGUL_SYLLABLE_RANGE = (0xAC00, 0xD7A3)
    HANGUL_JAMO_RANGE = (0x1100, 0x11FF)
    HANGUL_COMPAT_JAMO_RANGE = (0x3130, 0x318F)
    
    # Quality analysis weights
    WEIGHT_PUA = 0.4              # PUA character ratio weight
    WEIGHT_REPLACEMENT = 0.3      # Replacement character weight
    WEIGHT_VALID_RATIO = 0.3      # Valid character ratio weight


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TextQualityResult:
    """Text quality analysis result."""
    quality_score: float          # 0.0 ~ 1.0 (higher is better quality)
    total_chars: int              # Total character count
    pua_count: int                # PUA character count
    replacement_count: int        # Replacement character count
    valid_chars: int              # Valid character count (Korean, English, digits)
    control_chars: int            # Control character count
    needs_ocr: bool               # Whether OCR is needed
    details: Dict                 # Detailed information


@dataclass
class PageTextAnalysis:
    """Page text analysis result."""
    page_num: int
    quality_result: TextQualityResult
    text_blocks: List[Dict]       # Individual text block information
    problem_regions: List[Tuple[float, float, float, float]]  # Bounding boxes of problematic regions
    ocr_text: Optional[str] = None  # OCR result (if performed)


# ============================================================================
# Text Quality Analyzer
# ============================================================================

class TextQualityAnalyzer:
    """
    Text Quality Analyzer.
    
    Analyzes the quality of text extracted from PDF and
    detects broken text to determine whether OCR fallback is needed.
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
        
    def analyze_page(self) -> PageTextAnalysis:
        """
        Analyze text quality for the entire page.
        
        Returns:
            PageTextAnalysis object
        """
        # Extract text dictionary
        text_dict = self.page.get_text("dict", sort=True)
        blocks = text_dict.get("blocks", [])
        
        all_text = []
        text_blocks = []
        problem_regions = []
        
        for block in blocks:
            if block.get("type") != 0:  # Text blocks only
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
                
                # Record low quality regions
                if quality.needs_ocr:
                    problem_regions.append(block_bbox)
        
        # Analyze overall text quality
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
        Analyze text quality.
        
        Args:
            text: Text to analyze
            
        Returns:
            TextQualityResult object
        """
        if not text or len(text) < TextQualityConfig.MIN_TEXT_LENGTH:
            return TextQualityResult(
                quality_score=1.0,  # Treat as OK if text is empty or too short
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
        valid_chars = 0  # Korean, English, digits, spaces, basic punctuation
        
        # Character-by-character analysis
        for char in text:
            code = ord(char)
            
            # PUA check
            if self._is_pua(code):
                pua_count += 1
                continue
            
            # Replacement character check
            if code == 0xFFFD:
                replacement_count += 1
                continue
            
            # Control character check
            if self._is_control(code):
                control_count += 1
                continue
            
            # Valid character check
            if self._is_valid_char(char, code):
                valid_chars += 1
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            total_chars=total_chars,
            pua_count=pua_count,
            replacement_count=replacement_count,
            valid_chars=valid_chars
        )
        
        # Determine OCR necessity
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
        """Check if character is in Private Use Area."""
        for start, end in TextQualityConfig.PUA_RANGES:
            if start <= code <= end:
                return True
        return False
    
    def _is_control(self, code: int) -> bool:
        """Check if character is a control character."""
        for start, end in TextQualityConfig.CONTROL_RANGES:
            if start <= code <= end:
                return True
        return False
    
    def _is_valid_char(self, char: str, code: int) -> bool:
        """Check if character is valid (Korean, English, digits, spaces, basic punctuation)."""
        # Whitespace
        if char.isspace():
            return True
        
        # ASCII alphanumeric
        if char.isalnum() and code < 128:
            return True
        
        # Korean syllables
        if TextQualityConfig.HANGUL_SYLLABLE_RANGE[0] <= code <= TextQualityConfig.HANGUL_SYLLABLE_RANGE[1]:
            return True
        
        # Korean Jamo
        if TextQualityConfig.HANGUL_JAMO_RANGE[0] <= code <= TextQualityConfig.HANGUL_JAMO_RANGE[1]:
            return True
        
        # Korean compatibility Jamo
        if TextQualityConfig.HANGUL_COMPAT_JAMO_RANGE[0] <= code <= TextQualityConfig.HANGUL_COMPAT_JAMO_RANGE[1]:
            return True
        
        # Basic punctuation
        if char in '.,!?;:\'"()[]{}-–—…·•':
            return True
        
        # CJK characters (Chinese, Japanese)
        if 0x4E00 <= code <= 0x9FFF:  # CJK Unified Ideographs
            return True
        
        # Japanese Hiragana/Katakana
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
        """Calculate quality score (0.0 ~ 1.0)."""
        if total_chars == 0:
            return 1.0
        
        # Calculate ratios
        pua_ratio = pua_count / total_chars
        replacement_ratio = replacement_count / total_chars
        valid_ratio = valid_chars / total_chars
        
        # Calculate weighted score
        # Score decreases with more PUA chars, more replacement chars, lower valid ratio
        score = 1.0
        
        # PUA character penalty (more = lower score)
        score -= pua_ratio * TextQualityConfig.WEIGHT_PUA * 2
        
        # Replacement character penalty
        score -= replacement_ratio * TextQualityConfig.WEIGHT_REPLACEMENT * 3
        
        # Valid character ratio adjustment
        score = score * (0.5 + valid_ratio * 0.5)
        
        return max(0.0, min(1.0, score))


# ============================================================================
# Page OCR Fallback Engine
# ============================================================================

class PageOCRFallbackEngine:
    """
    Page OCR Fallback Engine.
    
    Performs OCR on the entire page or specific regions
    for pages with low text quality.
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
    
    def ocr_full_page(self) -> str:
        """
        Perform OCR on the entire page.
        
        Returns:
            Text extracted via OCR
        """
        try:
            # Render page at high resolution
            mat = fitz.Matrix(TextQualityConfig.OCR_SCALE, TextQualityConfig.OCR_SCALE)
            pix = self.page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR (Korean priority)
            ocr_config = '--psm 3 --oem 3'  # Automatic page segmentation + LSTM OCR
            text = pytesseract.image_to_string(
                img,
                lang=TextQualityConfig.OCR_LANG,
                config=ocr_config
            )
            
            # OCR post-processing: noise removal
            text = self._postprocess_ocr_text(text)
            
            logger.info(f"[PageOCR] Page {self.page_num + 1}: OCR extracted {len(text)} chars")
            return text.strip()
            
        except Exception as e:
            logger.error(f"[PageOCR] Page {self.page_num + 1} OCR failed: {e}")
            return ""
    
    def _postprocess_ocr_text(self, text: str) -> str:
        """
        Post-process OCR results.
        
        - Remove lines consisting only of special symbols
        - Remove meaningless short lines
        - Clean up repeated characters
        - Remove OCR noise patterns
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        # OCR noise patterns (text incorrectly recognized from background graphics)
        noise_patterns = [
            r'^[ri\-—maOANIUTLOG\s]+$',  # Noise from circular background graphics
            r'^[0-9"\'\[\]\(\)°\s]{1,5}$',  # Short number/symbol combinations
            r'^[A-Za-z\-—\s]{3,}$',  # Meaningless English combinations (when no Korean)
            r'^‥+\s*$',  # Only dotted lines
            r'^\s*[°·•○●□■◇◆△▲▽▼]+\s*$',  # Only symbols
        ]
        
        import re
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove lines consisting only of special symbols
            if all(c in '.,;:!?@#$%^&*()[]{}|\\/<>~`\'"-_+=°·•○●□■◇◆△▲▽▼' or c.isspace() for c in line):
                continue
            
            # Check noise patterns
            is_noise = False
            for pattern in noise_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_noise = True
                    break
            if is_noise:
                continue
            
            # Prioritize keeping lines with Korean
            korean_count = sum(1 for c in line if '가' <= c <= '힣')
            if korean_count > 0:
                cleaned_lines.append(line)
                continue
            
            # For English-only lines, check if meaningful
            alpha_count = sum(1 for c in line if c.isalpha())
            total_len = len(line.replace(' ', ''))
            
            if total_len > 0:
                meaningful_ratio = alpha_count / total_len
                # Keep only if meaningful characters >= 50% and at least 3 characters
                if meaningful_ratio >= 0.5 and alpha_count >= 3:
                    # Keep uppercase abbreviations (PLATEER, IDT, etc.)
                    if line.isupper() or any(word.isupper() and len(word) >= 2 for word in line.split()):
                        cleaned_lines.append(line)
                    # Regular English text (Insight Report, etc.)
                    elif any(c.islower() for c in line):
                        cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def ocr_region(self, bbox: Tuple[float, float, float, float]) -> str:
        """
        Perform OCR on a specific region.
        
        Args:
            bbox: Region coordinates (x0, y0, x1, y1)
            
        Returns:
            Text extracted via OCR
        """
        try:
            x0, y0, x1, y1 = bbox
            
            # Add padding
            padding = 10
            clip = fitz.Rect(
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(self.page_width, x1 + padding),
                min(self.page_height, y1 + padding)
            )
            
            # Render region at high resolution
            mat = fitz.Matrix(TextQualityConfig.OCR_SCALE, TextQualityConfig.OCR_SCALE)
            pix = self.page.get_pixmap(matrix=mat, clip=clip)
            
            # Convert to PIL Image
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            ocr_config = '--psm 6 --oem 3'  # Uniform text block + LSTM
            text = pytesseract.image_to_string(
                img,
                lang=TextQualityConfig.OCR_LANG,
                config=ocr_config
            )
            
            # OCR post-processing
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
        Perform OCR on problematic regions.
        
        Args:
            problem_regions: List of bounding boxes for problematic regions
            
        Returns:
            Dictionary mapping {bbox: ocr_text}
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
    Quality-Aware Text Extractor.
    
    Analyzes text quality and performs OCR fallback when necessary
    to always extract high-quality text.
    """
    
    def __init__(self, page, page_num: int, quality_threshold: float = None):
        """
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            quality_threshold: Quality threshold (default: TextQualityConfig.QUALITY_THRESHOLD)
        """
        self.page = page
        self.page_num = page_num
        self.quality_threshold = quality_threshold or TextQualityConfig.QUALITY_THRESHOLD
        
        self.analyzer = TextQualityAnalyzer(page, page_num)
        self.ocr_engine = PageOCRFallbackEngine(page, page_num)
    
    def extract(self) -> Tuple[str, PageTextAnalysis]:
        """
        Extract text with quality consideration.
        
        Returns:
            Tuple of (extracted text, analysis result)
        """
        # 1. Analyze page text quality
        analysis = self.analyzer.analyze_page()
        
        logger.debug(
            f"[QualityAware] Page {self.page_num + 1}: "
            f"quality={analysis.quality_result.quality_score:.2f}, "
            f"pua={analysis.quality_result.pua_count}, "
            f"valid={analysis.quality_result.valid_chars}"
        )
        
        # 2. Return existing text if quality is good
        if not analysis.quality_result.needs_ocr:
            # Extract text using standard method
            text = self.page.get_text("text")
            return text, analysis
        
        # 3. OCR fallback if quality is low
        logger.info(
            f"[QualityAware] Page {self.page_num + 1}: "
            f"Quality too low ({analysis.quality_result.quality_score:.2f}), "
            f"falling back to OCR"
        )
        
        # If few problem regions, OCR only those regions
        if len(analysis.problem_regions) <= 3 and len(analysis.problem_regions) > 0:
            # OCR only problem regions
            ocr_results = self.ocr_engine.ocr_problem_regions(analysis.problem_regions)
            
            # Replace problem region text with OCR results
            text = self._merge_ocr_results(analysis, ocr_results)
            analysis.ocr_text = str(ocr_results)
        else:
            # Full page OCR
            text = self.ocr_engine.ocr_full_page()
            analysis.ocr_text = text
        
        return text, analysis
    
    def _merge_ocr_results(
        self,
        analysis: PageTextAnalysis,
        ocr_results: Dict[Tuple, str]
    ) -> str:
        """
        Merge existing text with OCR results.
        
        Uses existing text for good quality blocks,
        replaces problematic blocks with OCR results.
        """
        merged_parts = []
        
        for block in analysis.text_blocks:
            bbox = tuple(block['bbox'])
            quality = block['quality']
            
            if quality.needs_ocr and bbox in ocr_results:
                # Use OCR result
                merged_parts.append(ocr_results[bbox])
            else:
                # Use existing text
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
