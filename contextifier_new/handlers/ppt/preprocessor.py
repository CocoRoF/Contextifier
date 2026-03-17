"""
PptPreprocessor — Stage 2: OLE2 data → PreprocessedData.

Extracts the ``PowerPoint Document`` stream and any available
metadata streams from the OLE2 file. Stores them in
``PreprocessedData`` for downstream extraction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.types import PreprocessedData
from contextifier_new.errors import PreprocessingError

from contextifier_new.handlers.ppt._constants import PP_DOCUMENT_STREAM
from contextifier_new.handlers.ppt.converter import PptConvertedData

logger = logging.getLogger(__name__)


class PptPreprocessor(BasePreprocessor):
    """
    Preprocessor for genuine OLE2 PPT files.

    Reads the ``PowerPoint Document`` stream and any embedded
    image data from the OLE compound file.
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """
        Extract PowerPoint data streams from the OLE2 file.

        Args:
            converted_data: ``PptConvertedData`` from the Converter.

        Returns:
            PreprocessedData with:
            - ``content``: the OLE object
            - ``raw_content``: the OLE object
            - ``resources["pp_stream"]``: raw PowerPoint Document stream bytes
            - ``resources["image_streams"]``: list of embedded image bytes
            - ``properties``: stream_size, has_pp_stream
        """
        if converted_data is None:
            raise PreprocessingError(
                "Received None as converted data",
                stage="preprocess",
                handler="ppt",
            )

        if isinstance(converted_data, PptConvertedData):
            ole = converted_data.ole
        elif hasattr(converted_data, "ole"):
            ole = converted_data.ole
        else:
            ole = converted_data

        # Read PowerPoint Document stream
        pp_stream: Optional[bytes] = None
        stream_size = 0
        has_pp_stream = False

        try:
            if ole.exists(PP_DOCUMENT_STREAM):
                pp_stream = ole.openstream(PP_DOCUMENT_STREAM).read()
                stream_size = len(pp_stream)
                has_pp_stream = True
        except Exception as exc:
            logger.debug("Failed to read PowerPoint Document stream: %s", exc)

        # Collect embedded images (stored in Pictures stream or as OLE parts)
        image_streams: List[bytes] = []
        try:
            if ole.exists("Pictures"):
                image_data = ole.openstream("Pictures").read()
                if image_data:
                    image_streams = _extract_images_from_pictures_stream(image_data)
        except Exception as exc:
            logger.debug("Failed to read Pictures stream: %s", exc)

        return PreprocessedData(
            content=ole,
            raw_content=ole,
            encoding="utf-8",
            resources={
                "pp_stream": pp_stream,
                "image_streams": image_streams,
            },
            properties={
                "stream_size": stream_size,
                "has_pp_stream": has_pp_stream,
                "image_count": len(image_streams),
            },
        )

    def get_format_name(self) -> str:
        return "ppt"


def _extract_images_from_pictures_stream(data: bytes) -> List[bytes]:
    """
    Extract individual images from the PPT Pictures stream.

    The Pictures stream contains image records with headers.
    Each record starts with a record header (8 bytes):
    - recVer/recInstance (2 bytes)
    - recType (2 bytes)
    - recLen (4 bytes, little-endian)

    Known image types:
    - 0xF01A: EMF
    - 0xF01B: WMF
    - 0xF01C: PICT
    - 0xF01D: JPEG
    - 0xF01E: PNG
    - 0xF01F: DIB
    - 0xF029: TIFF
    """
    images: List[bytes] = []
    offset = 0

    while offset + 8 <= len(data):
        # Read record header
        # rec_type = 2 bytes at offset+2 (little-endian)
        rec_type = int.from_bytes(data[offset + 2 : offset + 4], "little")
        rec_len = int.from_bytes(data[offset + 4 : offset + 8], "little")

        if rec_len <= 0 or offset + 8 + rec_len > len(data):
            break

        # Image record types
        if rec_type in (0xF01A, 0xF01B, 0xF01C, 0xF01D, 0xF01E, 0xF01F, 0xF029):
            # Skip 17 or 25 byte image header (UID + tag)
            # The exact header size depends on the record instance
            rec_instance = int.from_bytes(data[offset : offset + 2], "little") >> 4
            # If instance is one more than the base, there's an extra 8 bytes
            header_size = 25 if (rec_instance & 1) else 17
            if header_size < rec_len:
                img_data = data[offset + 8 + header_size : offset + 8 + rec_len]
                if img_data:
                    images.append(img_data)

        offset += 8 + rec_len

    return images


__all__ = ["PptPreprocessor"]
