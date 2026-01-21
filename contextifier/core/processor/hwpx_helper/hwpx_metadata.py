# contextifier/core/processor/hwpx_helper/hwpx_metadata.py
"""
HWPX Metadata Extraction Module

Provides HWPXMetadataExtractor class for extracting metadata from HWPX files.
Implements BaseMetadataExtractor interface.

Metadata locations in HWPX:
- version.xml: Document version information
- META-INF/container.xml: Container information
- Contents/header.xml: Document properties (author, date, etc.)

Note: HWPX is a Korean-native document format, so Korean metadata labels
are preserved in output for proper display.
"""
import logging
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)
from contextifier.core.processor.hwpx_helper.hwpx_constants import HWPX_NAMESPACES, HEADER_FILE_PATHS

logger = logging.getLogger("document-processor")


class HWPXMetadataExtractor(BaseMetadataExtractor):
    """
    HWPX Metadata Extractor.
    
    Extracts HWPX metadata from zipfile.ZipFile objects.
    
    Supported fields:
    - Standard fields: title, subject, author, keywords, comments, etc.
    - HWPX-specific: version, media_type, etc. (stored in custom fields)
    
    Usage:
        extractor = HWPXMetadataExtractor()
        metadata = extractor.extract(zip_file)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: zipfile.ZipFile) -> DocumentMetadata:
        """
        Extract metadata from HWPX file.
        
        Args:
            source: Open zipfile.ZipFile object
            
        Returns:
            DocumentMetadata instance containing extracted metadata.
        """
        raw_metadata: Dict[str, Any] = {}

        try:
            # Try to read header.xml for document properties
            for header_path in HEADER_FILE_PATHS:
                if header_path in source.namelist():
                    with source.open(header_path) as f:
                        header_content = f.read()
                        header_root = ET.fromstring(header_content)

                        # Try to find document properties
                        # <hh:docInfo> contains metadata
                        doc_info = header_root.find('.//hh:docInfo', HWPX_NAMESPACES)
                        if doc_info is not None:
                            # Get properties
                            for prop in doc_info:
                                tag = prop.tag.split('}')[-1] if '}' in prop.tag else prop.tag
                                if prop.text:
                                    raw_metadata[tag.lower()] = prop.text
                    break

            # Try to read version.xml
            if 'version.xml' in source.namelist():
                with source.open('version.xml') as f:
                    version_content = f.read()
                    version_root = ET.fromstring(version_content)

                    # Get version info
                    if version_root.text:
                        raw_metadata['version'] = version_root.text
                    for attr in version_root.attrib:
                        raw_metadata[f'version_{attr}'] = version_root.get(attr)

            # Try to read META-INF/manifest.xml for additional info
            if 'META-INF/manifest.xml' in source.namelist():
                with source.open('META-INF/manifest.xml') as f:
                    manifest_content = f.read()
                    manifest_root = ET.fromstring(manifest_content)

                    # Get mimetype and other info
                    for child in manifest_root:
                        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        if tag == 'file-entry':
                            full_path = child.get('full-path', child.get('{urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}full-path', ''))
                            if full_path == '/':
                                media_type = child.get('media-type', child.get('{urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}media-type', ''))
                                if media_type:
                                    raw_metadata['media_type'] = media_type

            self.logger.debug(f"Extracted HWPX metadata: {list(raw_metadata.keys())}")

        except Exception as e:
            self.logger.warning(f"Failed to extract HWPX metadata: {e}")
        
        # Separate standard fields from custom fields
        standard_fields = {'title', 'subject', 'author', 'keywords', 'comments', 
                          'last_saved_by', 'create_time', 'last_saved_time'}
        custom_fields = {k: v for k, v in raw_metadata.items() if k not in standard_fields}
        
        return DocumentMetadata(
            title=raw_metadata.get('title'),
            subject=raw_metadata.get('subject'),
            author=raw_metadata.get('author'),
            keywords=raw_metadata.get('keywords'),
            comments=raw_metadata.get('comments'),
            last_saved_by=raw_metadata.get('last_saved_by'),
            create_time=raw_metadata.get('create_time'),
            last_saved_time=raw_metadata.get('last_saved_time'),
            custom=custom_fields,
        )


def parse_bin_item_map(zf: zipfile.ZipFile) -> Dict[str, str]:
    """
    Parse content.hpf file to create BinItem ID to file path mapping.

    Args:
        zf: Open ZipFile object

    Returns:
        Dictionary mapping BinItem ID to file path.
    """
    from .hwpx_constants import HPF_PATH, OPF_NAMESPACES

    bin_item_map = {}

    try:
        if HPF_PATH in zf.namelist():
            with zf.open(HPF_PATH) as f:
                hpf_content = f.read()
                hpf_root = ET.fromstring(hpf_content)

                for item in hpf_root.findall('.//opf:item', OPF_NAMESPACES):
                    item_id = item.get('id')
                    href = item.get('href')
                    if item_id and href:
                        bin_item_map[item_id] = href

    except Exception as e:
        logger.warning(f"Failed to parse content.hpf: {e}")

    return bin_item_map


__all__ = [
    'HWPXMetadataExtractor',
    'parse_bin_item_map',
]
