# contextifier/core/processor/rtf_helper/rtf_preprocessor.py
"""
RTF Preprocessor

Preprocesses RTF binary data before conversion:
- \\binN tag processing (skip N bytes of raw binary data)
- \\pict group image extraction
- Image saving and tag generation
- Encoding detection

Implements BasePreprocessor interface.
"""
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from contextifier.core.functions.preprocessor import (
    BasePreprocessor,
    PreprocessedData,
)
from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend
from contextifier.core.processor.rtf_helper.rtf_decoder import (
    detect_encoding,
)

logger = logging.getLogger("contextify.rtf.preprocessor")


# Image format magic numbers
IMAGE_SIGNATURES = {
    b'\xff\xd8\xff': 'jpeg',
    b'\x89PNG\r\n\x1a\n': 'png',
    b'GIF87a': 'gif',
    b'GIF89a': 'gif',
    b'BM': 'bmp',
    b'\xd7\xcd\xc6\x9a': 'wmf',
    b'\x01\x00\x09\x00': 'wmf',
    b'\x01\x00\x00\x00': 'emf',
}

# RTF image type mapping
RTF_IMAGE_TYPES = {
    'jpegblip': 'jpeg',
    'pngblip': 'png',
    'wmetafile': 'wmf',
    'emfblip': 'emf',
    'dibitmap': 'bmp',
    'wbitmap': 'bmp',
}

# Supported image formats for saving
SUPPORTED_IMAGE_FORMATS = {'jpeg', 'png', 'gif', 'bmp'}


@dataclass
class RTFBinaryRegion:
    """RTF binary data region information."""
    start_pos: int
    end_pos: int
    bin_type: str  # "bin" or "pict"
    data_size: int
    image_format: str = ""
    image_data: bytes = b""


class RTFPreprocessor(BasePreprocessor):
    """
    RTF-specific preprocessor.
    
    Handles RTF binary preprocessing:
    - Removes \\bin tag binary data
    - Extracts embedded images
    - Detects encoding
    - Returns clean content ready for parsing
    
    Usage:
        preprocessor = RTFPreprocessor(image_processor=img_proc)
        result = preprocessor.preprocess(rtf_bytes)
        
        # result.clean_content - bytes ready for parsing
        # result.encoding - detected encoding
        # result.extracted_resources["image_tags"] - list of image tags
    """
    
    RTF_MAGIC = b'{\\rtf'
    
    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        processed_images: Optional[Set[str]] = None,
    ):
        """
        Initialize RTFPreprocessor.
        
        Args:
            image_processor: Image processor for saving images
            processed_images: Set of already processed image hashes
        """
        self._image_processor = image_processor
        self._processed_images = processed_images if processed_images is not None else set()
    
    def preprocess(
        self,
        file_data: bytes,
        **kwargs
    ) -> PreprocessedData:
        """
        Preprocess RTF binary data.
        
        Args:
            file_data: Raw RTF binary data
            **kwargs: Additional options
            
        Returns:
            PreprocessedData with clean content, encoding, and image tags
        """
        if not file_data:
            return PreprocessedData(
                raw_content=b"",
                clean_content=b"",
                encoding="cp949",
            )
        
        # Get options from kwargs
        image_processor = kwargs.get('image_processor', self._image_processor)
        processed_images = kwargs.get('processed_images', self._processed_images)
        
        # Detect encoding
        detected_encoding = detect_encoding(file_data, "cp949")
        
        # Process binary data (extract images, clean content)
        clean_content, image_tags = self._process_binary_content(
            file_data,
            image_processor,
            processed_images
        )
        
        # Filter valid image tags
        valid_tags = [
            tag for tag in image_tags
            if tag and tag.strip() and '/uploads/.' not in tag
        ]
        
        return PreprocessedData(
            raw_content=file_data,
            clean_content=clean_content,
            encoding=detected_encoding,
            extracted_resources={
                "image_tags": valid_tags,
            }
        )
    
    def get_format_name(self) -> str:
        """Return format name."""
        return "RTF Preprocessor"
    
    def validate(self, file_data: bytes) -> bool:
        """Validate if data is a valid RTF file."""
        if not file_data or len(file_data) < 5:
            return False
        return file_data[:5] == self.RTF_MAGIC
    
    def _process_binary_content(
        self,
        content: bytes,
        image_processor: Optional[ImageProcessor],
        processed_images: Set[str]
    ) -> Tuple[bytes, List[str]]:
        """
        Process RTF binary content.
        
        Args:
            content: RTF binary content
            image_processor: Image processor instance
            processed_images: Set of processed image hashes
            
        Returns:
            Tuple of (clean_content, list of image tags)
        """
        image_tags: Dict[int, str] = {}
        
        # Find \bin tag regions
        bin_regions = self._find_bin_regions(content)
        
        # Find \pict regions (excluding bin regions)
        pict_regions = self._find_pict_regions(content, bin_regions)
        
        # Merge and sort all regions
        all_regions = bin_regions + pict_regions
        all_regions.sort(key=lambda r: r.start_pos)
        
        # Process images and generate tags
        for region in all_regions:
            if not region.image_data:
                continue
            
            # Check for duplicates
            image_hash = hashlib.md5(region.image_data).hexdigest()
            if image_hash in processed_images:
                image_tags[region.start_pos] = ""
                continue
            
            processed_images.add(image_hash)
            
            if region.image_format in SUPPORTED_IMAGE_FORMATS and image_processor:
                tag = image_processor.save_image(region.image_data)
                if tag:
                    image_tags[region.start_pos] = f"\n{tag}\n"
                    logger.info(
                        f"Saved RTF image: {tag} "
                        f"(format={region.image_format}, size={region.data_size})"
                    )
                else:
                    image_tags[region.start_pos] = ""
            else:
                image_tags[region.start_pos] = ""
        
        # Remove binary data from content
        clean_content = self._remove_binary_data(content, all_regions, image_tags)
        
        # Collect all image tags as list
        tag_list = [tag for tag in image_tags.values() if tag and tag.strip()]
        
        return clean_content, tag_list
    
    def _find_bin_regions(self, content: bytes) -> List[RTFBinaryRegion]:
        """Find \\binN tags and identify binary regions."""
        regions = []
        pattern = rb'\\bin(\d+)'
        
        for match in re.finditer(pattern, content):
            try:
                bin_size = int(match.group(1))
                bin_tag_start = match.start()
                bin_tag_end = match.end()
                
                data_start = bin_tag_end
                if data_start < len(content) and content[data_start:data_start+1] == b' ':
                    data_start += 1
                
                data_end = data_start + bin_size
                
                if data_end <= len(content):
                    binary_data = content[data_start:data_end]
                    image_format = self._detect_image_format(binary_data)
                    
                    # Find parent \shppict group
                    group_start = bin_tag_start
                    group_end = data_end
                    
                    search_start = max(0, bin_tag_start - 500)
                    search_area = content[search_start:bin_tag_start]
                    
                    shppict_pos = search_area.rfind(b'\\shppict')
                    if shppict_pos != -1:
                        abs_pos = search_start + shppict_pos
                        brace_pos = abs_pos
                        while brace_pos > 0 and content[brace_pos:brace_pos+1] != b'{':
                            brace_pos -= 1
                        group_start = brace_pos
                        
                        depth = 1
                        j = data_end
                        while j < len(content) and depth > 0:
                            if content[j:j+1] == b'{':
                                depth += 1
                            elif content[j:j+1] == b'}':
                                depth -= 1
                            j += 1
                        group_end = j
                    
                    regions.append(RTFBinaryRegion(
                        start_pos=group_start,
                        end_pos=group_end,
                        bin_type="bin",
                        data_size=bin_size,
                        image_format=image_format,
                        image_data=binary_data
                    ))
            except (ValueError, IndexError):
                continue
        
        return regions
    
    def _find_pict_regions(
        self,
        content: bytes,
        exclude_regions: List[RTFBinaryRegion]
    ) -> List[RTFBinaryRegion]:
        """Find hex-encoded \\pict regions."""
        regions = []
        
        bin_tag_positions = {r.start_pos for r in exclude_regions if r.bin_type == "bin"}
        excluded_ranges = [(r.start_pos, r.end_pos) for r in exclude_regions]
        
        def is_excluded(pos: int) -> bool:
            return any(start <= pos < end for start, end in excluded_ranges)
        
        def has_bin_nearby(pict_pos: int) -> bool:
            return any(pict_pos < bp < pict_pos + 200 for bp in bin_tag_positions)
        
        try:
            text_content = content.decode('cp1252', errors='replace')
            pict_pattern = r'\\pict\s*((?:\\[a-zA-Z]+\d*\s*)*)'
            
            for match in re.finditer(pict_pattern, text_content):
                start_pos = match.start()
                
                if is_excluded(start_pos) or has_bin_nearby(start_pos):
                    continue
                
                attrs = match.group(1)
                image_format = ""
                for rtf_type, fmt in RTF_IMAGE_TYPES.items():
                    if rtf_type in attrs:
                        image_format = fmt
                        break
                
                # Extract hex data
                hex_start = match.end()
                hex_data = []
                i = hex_start
                
                while i < len(text_content):
                    ch = text_content[i]
                    if ch in '0123456789abcdefABCDEF':
                        hex_data.append(ch)
                    elif ch in ' \t\r\n':
                        pass
                    elif ch == '}':
                        break
                    elif ch == '\\':
                        if text_content[i:i+4] == '\\bin':
                            hex_data = []
                            break
                        while i < len(text_content) and text_content[i] not in ' \t\r\n}':
                            i += 1
                        continue
                    else:
                        break
                    i += 1
                
                hex_str = ''.join(hex_data)
                
                if len(hex_str) >= 32:
                    try:
                        image_data = bytes.fromhex(hex_str)
                        if not image_format:
                            image_format = self._detect_image_format(image_data)
                        
                        if image_format:
                            regions.append(RTFBinaryRegion(
                                start_pos=start_pos,
                                end_pos=i,
                                bin_type="pict",
                                data_size=len(image_data),
                                image_format=image_format,
                                image_data=image_data
                            ))
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"Error finding pict regions: {e}")
        
        return regions
    
    def _detect_image_format(self, data: bytes) -> str:
        """Detect image format from binary data."""
        if not data or len(data) < 4:
            return ""
        
        for signature, format_name in IMAGE_SIGNATURES.items():
            if data.startswith(signature):
                return format_name
        
        if len(data) >= 2 and data[0:2] == b'\xff\xd8':
            return 'jpeg'
        
        return ""
    
    def _remove_binary_data(
        self,
        content: bytes,
        regions: List[RTFBinaryRegion],
        image_tags: Dict[int, str]
    ) -> bytes:
        """Remove binary data regions from content."""
        if not regions:
            return content
        
        sorted_regions = sorted(regions, key=lambda r: r.start_pos, reverse=True)
        result = bytearray(content)
        
        for region in sorted_regions:
            replacement = b''
            if region.start_pos in image_tags:
                tag = image_tags[region.start_pos]
                if tag:
                    replacement = tag.encode('ascii', errors='replace')
            result[region.start_pos:region.end_pos] = replacement
        
        return bytes(result)


__all__ = ['RTFPreprocessor', 'RTFBinaryRegion']
