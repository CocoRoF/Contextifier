# contextifier/core/processor/hwp5_helper/hwp5_image_processor.py
"""
HWP 5.0 Image Processor

Provides HWP 5.0-specific image processing that inherits from ImageProcessor.
Handles BinData stream images and embedded images in HWP 5.0 OLE format.

Features:
- zlib decompression for compressed images
- BinData stream finding and extraction
- OLE storage image processing
"""
import io
import os
import zlib
import struct
import logging
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from PIL import Image

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.storage_backend import BaseStorageBackend

if TYPE_CHECKING:
    import olefile

logger = logging.getLogger("document-processor.HWP5")


class HWP5ImageProcessor(ImageProcessor):
    """
    HWP 5.0 Image Processor.
    
    Handles:
    - BinData stream images (compressed/uncompressed)
    - Embedded OLE images
    - Image decompression (zlib)
    
    Usage:
        processor = HWP5ImageProcessor()
        
        # Process BinData image
        tag = processor.process_image(image_data, bindata_id="BIN0001")
        
        # Process from OLE stream
        tag = processor.process_bindata_stream(ole, stream_path)
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize HWP5ImageProcessor.
        
        Args:
            directory_path: Image save directory
            tag_prefix: Tag prefix for image references
            tag_suffix: Tag suffix for image references
            storage_backend: Storage backend for saving images
        """
        super().__init__(
            directory_path=directory_path,
            tag_prefix=tag_prefix,
            tag_suffix=tag_suffix,
            storage_backend=storage_backend,
        )
    
    def process_image(
        self,
        image_data: bytes,
        bindata_id: Optional[str] = None,
        image_index: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save HWP image data.
        
        Args:
            image_data: Raw image binary data
            bindata_id: BinData ID (e.g., "BIN0001")
            image_index: Image index (for naming)
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = None
        if bindata_id is not None:
            custom_name = f"hwp5_{bindata_id}"
        elif image_index is not None:
            custom_name = f"hwp5_image_{image_index}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def process_bindata_stream(
        self,
        ole: "olefile.OleFileIO",
        stream_path: str,
        is_compressed: bool = True,
    ) -> Optional[str]:
        """
        Process image from HWP BinData OLE stream.
        
        Args:
            ole: OleFileIO object
            stream_path: Path to BinData stream
            is_compressed: Whether data is zlib compressed
            
        Returns:
            Image tag string, or None on failure
        """
        try:
            stream_data = ole.openstream(stream_path).read()
            
            if is_compressed:
                try:
                    image_data = zlib.decompress(stream_data, -15)
                except zlib.error:
                    try:
                        image_data = zlib.decompress(stream_data)
                    except zlib.error:
                        image_data = stream_data
            else:
                image_data = stream_data
            
            # Extract bindata ID from path
            bindata_id = stream_path.split('/')[-1] if '/' in stream_path else stream_path
            
            return self.process_image(image_data, bindata_id=bindata_id)
            
        except Exception as e:
            self._logger.warning(f"Failed to process BinData stream {stream_path}: {e}")
            return None
    
    def process_embedded_image(
        self,
        image_data: bytes,
        image_name: Optional[str] = None,
        bindata_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process embedded HWP image.
        
        Args:
            image_data: Image binary data
            image_name: Original image filename
            bindata_id: BinData ID
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = image_name
        if custom_name is None and bindata_id is not None:
            custom_name = f"hwp5_embed_{bindata_id}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def decompress_and_process(
        self,
        compressed_data: bytes,
        bindata_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Decompress and process zlib-compressed image data.
        
        Args:
            compressed_data: zlib compressed image data
            bindata_id: BinData ID
            
        Returns:
            Image tag string, or None on failure
        """
        image_data = self.try_decompress_image(compressed_data)
        return self.process_image(image_data, bindata_id=bindata_id)
    
    @staticmethod
    def try_decompress_image(data: bytes) -> bytes:
        """
        Attempt to decompress HWP image data.
        
        Args:
            data: Original image data (possibly compressed)
            
        Returns:
            Decompressed image data (or original if not compressed)
        """
        # 1. Try zlib decompression if zlib header present
        if data.startswith(b'\x78'):
            try:
                return zlib.decompress(data)
            except Exception:
                pass

        # 2. Check if already a valid image
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.verify()
            return data
        except Exception:
            pass

        # 3. Try raw deflate (no header)
        try:
            return zlib.decompress(data, -15)
        except Exception:
            pass

        return data
    
    @staticmethod
    def find_bindata_stream(ole: "olefile.OleFileIO", storage_id: int, ext: str) -> Optional[List[str]]:
        """
        Find BinData stream in OLE container by storage_id and extension.
        
        Args:
            ole: OLE file object
            storage_id: BinData storage ID
            ext: File extension
            
        Returns:
            Stream path if found, None otherwise
        """
        ole_dirs = ole.listdir()

        candidates = [
            f"BinData/BIN{storage_id:04X}.{ext}",
            f"BinData/BIN{storage_id:04x}.{ext}",
            f"BinData/Bin{storage_id:04X}.{ext}",
            f"BinData/Bin{storage_id:04x}.{ext}",
            f"BinData/BIN{storage_id:04X}.{ext.lower()}",
            f"BinData/BIN{storage_id:04x}.{ext.lower()}",
        ]

        # Pattern matching
        for entry in ole_dirs:
            if entry[0] == "BinData" and len(entry) > 1:
                fname = entry[1].lower()
                expected_patterns = [
                    f"bin{storage_id:04x}",
                    f"bin{storage_id:04X}",
                ]
                for pattern in expected_patterns:
                    if pattern.lower() in fname.lower():
                        logger.debug(f"Found stream by pattern match: {entry}")
                        return entry

        # Exact path matching
        for candidate in candidates:
            candidate_parts = candidate.split('/')
            if candidate_parts in ole_dirs:
                return candidate_parts

        # Case-insensitive matching
        for entry in ole_dirs:
            if entry[0] == "BinData" and len(entry) > 1:
                fname = entry[1]
                for candidate in candidates:
                    if fname.lower() == candidate.split('/')[-1].lower():
                        return entry

        return None
    
    @staticmethod
    def extract_bindata_index(payload: bytes, bin_data_list_len: int) -> Optional[int]:
        """
        Extract BinData index from SHAPE_COMPONENT_PICTURE payload.
        
        Args:
            payload: SHAPE_COMPONENT_PICTURE record payload
            bin_data_list_len: Length of bin_data_list
            
        Returns:
            BinData index (1-based) or None
        """
        if bin_data_list_len == 0:
            return None

        # Strategy 1: Offset 79 (HWP 5.0.3.x+)
        if len(payload) >= 81:
            test_id = struct.unpack('<H', payload[79:81])[0]
            if 0 < test_id <= bin_data_list_len:
                logger.debug(f"Found BinData index at offset 79: {test_id}")
                return test_id

        # Strategy 2: Offset 8 (older versions)
        if len(payload) >= 10:
            test_id = struct.unpack('<H', payload[8:10])[0]
            if 0 < test_id <= bin_data_list_len:
                logger.debug(f"Found BinData index at offset 8: {test_id}")
                return test_id

        # Strategy 3: Common offset scan
        for offset in [4, 6, 10, 12, 14, 16, 18, 20, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]:
            if len(payload) >= offset + 2:
                test_id = struct.unpack('<H', payload[offset:offset+2])[0]
                if 0 < test_id <= bin_data_list_len:
                    logger.debug(f"Found potential BinData index at offset {offset}: {test_id}")
                    return test_id

        # Strategy 4: Full scan for first valid value
        for i in range(0, min(len(payload) - 1, 100), 2):
            test_id = struct.unpack('<H', payload[i:i+2])[0]
            if 0 < test_id <= bin_data_list_len:
                logger.debug(f"Found BinData index by scanning at offset {i}: {test_id}")
                return test_id

        return None
    
    def extract_and_save_image(
        self,
        ole: "olefile.OleFileIO",
        target_stream: List[str],
        processed_images: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Extract image from OLE stream and save.
        
        Args:
            ole: OLE file object
            target_stream: Stream path
            processed_images: Set of processed image paths
            
        Returns:
            Image tag string or None
        """
        try:
            stream = ole.openstream(target_stream)
            image_data = stream.read()
            image_data = self.try_decompress_image(image_data)

            bindata_id = target_stream[-1] if target_stream else None
            image_tag = self.process_image(image_data, bindata_id=bindata_id)
            
            if image_tag:
                if processed_images is not None:
                    processed_images.add("/".join(target_stream))
                logger.info(f"Successfully extracted inline image: {image_tag}")
                return f"\n{image_tag}\n"
        except Exception as e:
            logger.warning(f"Failed to process inline HWP image {target_stream}: {e}")

        return None
    
    def process_images_from_bindata(
        self,
        ole: "olefile.OleFileIO",
        processed_images: Optional[Set[str]] = None,
    ) -> str:
        """
        Extract all images from BinData storage.
        
        Args:
            ole: OLE file object
            processed_images: Set of already processed paths (to skip)
            
        Returns:
            Combined image tag strings
        """
        results = []

        try:
            bindata_streams = [
                entry for entry in ole.listdir()
                if entry[0] == "BinData"
            ]

            for stream_path in bindata_streams:
                if processed_images and "/".join(stream_path) in processed_images:
                    continue

                stream_name = stream_path[-1]
                ext = os.path.splitext(stream_name)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    stream = ole.openstream(stream_path)
                    image_data = stream.read()
                    image_data = self.try_decompress_image(image_data)

                    bindata_id = stream_name
                    image_tag = self.process_image(image_data, bindata_id=bindata_id)
                    if image_tag:
                        results.append(image_tag)

        except Exception as e:
            logger.warning(f"Error processing HWP images: {e}")

        return "\n\n".join(results)


__all__ = ["HWP5ImageProcessor"]
