# libs/core/functions/img_processor.py
"""
Image Processing Module

Provides functionality to save image data to the local file system and convert to tag format.
A general-purpose image processing module that replaces the existing image upload functions.

Main Features:
- Save image data to a specified directory
- Return saved path in custom tag format
- Duplicate image detection and handling
- Support for various image formats

Usage Example:
    from contextifier.core.functions.img_processor import ImageProcessor

    # Use with default settings
    processor = ImageProcessor()
    tag = processor.save_image(image_bytes)
    # Result: "[Image:temp/abc123.png]"

    # Custom settings
    processor = ImageProcessor(
        directory_path="output/images",
        tag_prefix="<img src='",
        tag_suffix="'>"
    )
    tag = processor.save_image(image_bytes)
    # Result: "<img src='output/images/abc123.png'>"
"""
import hashlib
import io
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("document-processor")


class ImageFormat(Enum):
    """Supported image formats"""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    GIF = "gif"
    BMP = "bmp"
    WEBP = "webp"
    TIFF = "tiff"
    UNKNOWN = "unknown"


class NamingStrategy(Enum):
    """Image file naming strategies"""
    HASH = "hash"           # Content-based hash (prevents duplicates)
    UUID = "uuid"           # Unique UUID
    SEQUENTIAL = "sequential"  # Sequential numbering
    TIMESTAMP = "timestamp"    # Timestamp-based


@dataclass
class ImageProcessorConfig:
    """
    ImageProcessor Configuration

    Attributes:
        directory_path: Directory path to save images
        tag_prefix: Tag prefix (e.g., "[Image:")
        tag_suffix: Tag suffix (e.g., "]")
        naming_strategy: File naming strategy
        default_format: Default image format
        create_directory: Auto-create directory if not exists
        use_absolute_path: Use absolute path in tags
        hash_algorithm: Hash algorithm (for hash strategy)
        max_filename_length: Maximum filename length
    """
    directory_path: str = "temp"
    tag_prefix: str = "[Image:"
    tag_suffix: str = "]"
    naming_strategy: NamingStrategy = NamingStrategy.HASH
    default_format: ImageFormat = ImageFormat.PNG
    create_directory: bool = True
    use_absolute_path: bool = False
    hash_algorithm: str = "sha256"
    max_filename_length: int = 64


class ImageProcessor:
    """
    Image Processing Class

    Saves image data to the local file system and returns
    the saved path in the specified tag format.

    Args:
        directory_path: Image save directory (default: "temp")
        tag_prefix: Tag prefix (default: "[Image:")
        tag_suffix: Tag suffix (default: "]")
        naming_strategy: File naming strategy (default: HASH)
        config: ImageProcessorConfig object (takes precedence over individual parameters)

    Examples:
        >>> processor = ImageProcessor()
        >>> tag = processor.save_image(image_bytes)
        "[Image:temp/a1b2c3d4.png]"

        >>> processor = ImageProcessor(
        ...     directory_path="images",
        ...     tag_prefix="![image](",
        ...     tag_suffix=")"
        ... )
        >>> tag = processor.save_image(image_bytes)
        "![image](images/a1b2c3d4.png)"
    """

    def __init__(
        self,
        directory_path: str = "temp",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        naming_strategy: Union[NamingStrategy, str] = NamingStrategy.HASH,
        config: Optional[ImageProcessorConfig] = None,
    ):
        if config:
            self.config = config
        else:
            # Convert string to Enum if needed
            if isinstance(naming_strategy, str):
                naming_strategy = NamingStrategy(naming_strategy.lower())

            self.config = ImageProcessorConfig(
                directory_path=directory_path,
                tag_prefix=tag_prefix,
                tag_suffix=tag_suffix,
                naming_strategy=naming_strategy,
            )

        # Track processed image hashes (for duplicate prevention)
        self._processed_hashes: Dict[str, str] = {}

        # Sequential counter (for sequential strategy)
        self._sequential_counter: int = 0

        # Create directory
        if self.config.create_directory:
            self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Check if directory exists and create if not"""
        path = Path(self.config.directory_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")

    def _compute_hash(self, data: bytes) -> str:
        """Compute hash of image data"""
        hasher = hashlib.new(self.config.hash_algorithm)
        hasher.update(data)
        return hasher.hexdigest()[:32]  # Use first 32 characters

    def _detect_format(self, data: bytes) -> ImageFormat:
        """Detect format from image data"""
        # Detect format using magic bytes
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return ImageFormat.PNG
        elif data[:2] == b'\xff\xd8':
            return ImageFormat.JPEG
        elif data[:6] in (b'GIF87a', b'GIF89a'):
            return ImageFormat.GIF
        elif data[:2] == b'BM':
            return ImageFormat.BMP
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return ImageFormat.WEBP
        elif data[:4] in (b'II*\x00', b'MM\x00*'):
            return ImageFormat.TIFF
        else:
            return ImageFormat.UNKNOWN

    def _generate_filename(
        self,
        data: bytes,
        image_format: ImageFormat,
        custom_name: Optional[str] = None
    ) -> str:
        """Generate filename"""
        if custom_name:
            # Add extension
            if not any(custom_name.lower().endswith(f".{fmt.value}") for fmt in ImageFormat if fmt != ImageFormat.UNKNOWN):
                ext = image_format.value if image_format != ImageFormat.UNKNOWN else self.config.default_format.value
                return f"{custom_name}.{ext}"
            return custom_name

        ext = image_format.value if image_format != ImageFormat.UNKNOWN else self.config.default_format.value

        strategy = self.config.naming_strategy

        if strategy == NamingStrategy.HASH:
            base = self._compute_hash(data)
        elif strategy == NamingStrategy.UUID:
            base = str(uuid.uuid4())[:16]
        elif strategy == NamingStrategy.SEQUENTIAL:
            self._sequential_counter += 1
            base = f"image_{self._sequential_counter:06d}"
        elif strategy == NamingStrategy.TIMESTAMP:
            import time
            base = f"img_{int(time.time() * 1000)}"
        else:
            base = self._compute_hash(data)

        filename = f"{base}.{ext}"

        # Limit filename length
        if len(filename) > self.config.max_filename_length:
            max_base_len = self.config.max_filename_length - len(ext) - 1
            filename = f"{base[:max_base_len]}.{ext}"

        return filename

    def _build_tag(self, file_path: str) -> str:
        """Build tag from saved file path"""
        if self.config.use_absolute_path:
            path_str = str(Path(file_path).absolute())
        else:
            path_str = file_path

        # Normalize path separators (Windows -> Unix style)
        path_str = path_str.replace("\\", "/")

        return f"{self.config.tag_prefix}{path_str}{self.config.tag_suffix}"

    def save_image(
        self,
        image_data: bytes,
        custom_name: Optional[str] = None,
        processed_images: Optional[Set[str]] = None,
        skip_duplicate: bool = True,
    ) -> Optional[str]:
        """
        Save image data to file and return tag.

        Args:
            image_data: Image binary data
            custom_name: Custom filename (extension optional)
            processed_images: Set of processed image paths (for external duplicate tracking)
            skip_duplicate: If True, skip saving duplicate images (return existing path)

        Returns:
            Image tag string, or None on failure

        Examples:
            >>> processor = ImageProcessor()
            >>> tag = processor.save_image(png_bytes)
            "[Image:temp/abc123.png]"
        """
        if not image_data:
            logger.warning("Empty image data provided")
            return None

        try:
            # Detect image format
            image_format = self._detect_format(image_data)

            # Compute hash (for duplicate check)
            image_hash = self._compute_hash(image_data)

            # Check for duplicates
            if skip_duplicate and image_hash in self._processed_hashes:
                existing_path = self._processed_hashes[image_hash]
                logger.debug(f"Duplicate image detected, returning existing: {existing_path}")
                return self._build_tag(existing_path)

            # Generate filename
            filename = self._generate_filename(image_data, image_format, custom_name)

            # Full path
            file_path = os.path.join(self.config.directory_path, filename)

            # Check external duplicate tracking
            if processed_images is not None and file_path in processed_images:
                logger.debug(f"Image already processed externally: {file_path}")
                return self._build_tag(file_path)

            # Ensure directory exists
            self._ensure_directory_exists()

            # Save file
            with open(file_path, 'wb') as f:
                f.write(image_data)

            logger.debug(f"Image saved: {file_path}")

            # Update internal duplicate tracking
            self._processed_hashes[image_hash] = file_path

            # Update external duplicate tracking
            if processed_images is not None:
                processed_images.add(file_path)

            return self._build_tag(file_path)

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return None

    def save_image_from_pil(
        self,
        pil_image,
        image_format: Optional[ImageFormat] = None,
        custom_name: Optional[str] = None,
        processed_images: Optional[Set[str]] = None,
        quality: int = 95,
    ) -> Optional[str]:
        """
        Save PIL Image object to file and return tag.

        Args:
            pil_image: PIL Image object
            image_format: Image format to save (None keeps original or uses default)
            custom_name: Custom filename
            processed_images: Set of processed image paths
            quality: JPEG quality (1-100)

        Returns:
            Image tag string, or None on failure
        """
        try:
            from PIL import Image

            if not isinstance(pil_image, Image.Image):
                logger.error("Invalid PIL Image object")
                return None

            # Determine format
            fmt = image_format or ImageFormat.PNG
            if fmt == ImageFormat.UNKNOWN:
                fmt = self.config.default_format

            # Convert to bytes
            buffer = io.BytesIO()
            save_format = fmt.value.upper()
            if save_format == "JPG":
                save_format = "JPEG"

            save_kwargs = {}
            if save_format == "JPEG":
                save_kwargs["quality"] = quality
            elif save_format == "PNG":
                save_kwargs["compress_level"] = 6

            pil_image.save(buffer, format=save_format, **save_kwargs)
            image_data = buffer.getvalue()

            return self.save_image(image_data, custom_name, processed_images)

        except Exception as e:
            logger.error(f"Failed to save PIL image: {e}")
            return None

    def get_processed_count(self) -> int:
        """Return number of processed images"""
        return len(self._processed_hashes)

    def get_processed_paths(self) -> List[str]:
        """Return all processed image paths"""
        return list(self._processed_hashes.values())

    def clear_cache(self) -> None:
        """Clear internal duplicate tracking cache"""
        self._processed_hashes.clear()
        self._sequential_counter = 0

    def cleanup(self, delete_files: bool = False) -> int:
        """
        Clean up resources.

        Args:
            delete_files: If True, also delete saved files

        Returns:
            Number of deleted files
        """
        deleted = 0

        if delete_files:
            for path in self._processed_hashes.values():
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete file {path}: {e}")

        self.clear_cache()
        return deleted

    def get_pattern_string(self) -> str:
        """
        Get regex pattern string for matching image tags.

        Returns a regex pattern that matches the image tag format used by this processor.
        The pattern captures the image path as group 1.

        Returns:
            Regex pattern string for matching image tags

        Examples:
            >>> processor = ImageProcessor()  # default: [Image:...]
            >>> processor.get_pattern_string()
            '\\[Image:([^\\]]+)\\]'

            >>> processor = ImageProcessor(tag_prefix="<img src='", tag_suffix="'/>")
            >>> processor.get_pattern_string()
            "<img src='([^']+)'/>"
        """
        import re
        prefix = re.escape(self.config.tag_prefix)
        suffix = re.escape(self.config.tag_suffix)

        # Determine the capture group pattern based on suffix
        # If suffix is empty, capture everything until whitespace or end
        if not self.config.tag_suffix:
            capture = r'(\S+)'
        else:
            # Use negated character class based on first char of suffix
            first_char = self.config.tag_suffix[0]
            capture = f'([^{re.escape(first_char)}]+)'

        return f'{prefix}{capture}{suffix}'


# ============================================================================
# Config-based ImageProcessor Access
# ============================================================================

# Default configuration values
DEFAULT_IMAGE_CONFIG = {
    "directory_path": "temp/images",
    "tag_prefix": "[Image:",
    "tag_suffix": "]",
    "naming_strategy": NamingStrategy.HASH,
}


def create_image_processor(
    directory_path: Optional[str] = None,
    tag_prefix: Optional[str] = None,
    tag_suffix: Optional[str] = None,
    naming_strategy: Optional[Union[NamingStrategy, str]] = None,
) -> ImageProcessor:
    """
    Create a new ImageProcessor instance.

    Args:
        directory_path: Image save directory (default: "temp/images")
        tag_prefix: Tag prefix (default: "[Image:")
        tag_suffix: Tag suffix (default: "]")
        naming_strategy: File naming strategy (default: HASH)

    Returns:
        New ImageProcessor instance

    Examples:
        >>> processor = create_image_processor(
        ...     directory_path="output/images",
        ...     tag_prefix="<img src='",
        ...     tag_suffix="'/>"
        ... )
    """
    if naming_strategy is not None and isinstance(naming_strategy, str):
        naming_strategy = NamingStrategy(naming_strategy.lower())

    return ImageProcessor(
        directory_path=directory_path or DEFAULT_IMAGE_CONFIG["directory_path"],
        tag_prefix=tag_prefix or DEFAULT_IMAGE_CONFIG["tag_prefix"],
        tag_suffix=tag_suffix or DEFAULT_IMAGE_CONFIG["tag_suffix"],
        naming_strategy=naming_strategy or DEFAULT_IMAGE_CONFIG["naming_strategy"],
    )


def save_image_to_file(
    image_data: bytes,
    directory_path: str = "temp",
    tag_prefix: str = "[Image:",
    tag_suffix: str = "]",
    processed_images: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Save image to file and return tag.

    A simple function that replaces the existing image upload functions.

    Args:
        image_data: Image binary data
        directory_path: Save directory
        tag_prefix: Tag prefix
        tag_suffix: Tag suffix
        processed_images: Set for duplicate tracking

    Returns:
        Image tag string, or None on failure

    Examples:
        >>> tag = save_image_to_file(image_bytes)
        "[Image:temp/abc123.png]"

        >>> tag = save_image_to_file(
        ...     image_bytes,
        ...     directory_path="output",
        ...     tag_prefix="<img src='",
        ...     tag_suffix="'/>"
        ... )
        "<img src='output/abc123.png'/>"
    """
    processor = ImageProcessor(
        directory_path=directory_path,
        tag_prefix=tag_prefix,
        tag_suffix=tag_suffix,
    )

    return processor.save_image(image_data, processed_images=processed_images)


__all__ = [
    # Classes
    "ImageProcessor",
    "ImageProcessorConfig",
    "ImageFormat",
    "NamingStrategy",
    # Factory function
    "create_image_processor",
    "DEFAULT_IMAGE_CONFIG",
    # Convenience function
    "save_image_to_file",
]
