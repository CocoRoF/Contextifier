# hwpx_helper/hwpx_metadata.py
"""
HWPX 메타데이터 추출

HWPX 파일에서 메타데이터를 추출합니다.
메타데이터는 다음 파일에 저장됩니다:
- version.xml: 문서 버전 정보
- META-INF/container.xml: 컨테이너 정보
- Contents/header.xml: 문서 속성 (작성자, 날짜 등)
"""
import logging
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict

from .hwpx_constants import HWPX_NAMESPACES, HEADER_FILE_PATHS

logger = logging.getLogger("document-processor")


def extract_hwpx_metadata(zf: zipfile.ZipFile) -> Dict[str, Any]:
    """
    HWPX 파일에서 메타데이터를 추출합니다.

    HWPX stores metadata in:
    - version.xml: Document version info
    - META-INF/container.xml: Container info
    - Contents/header.xml: Document properties (작성자, 날짜 등)

    Args:
        zf: 열린 ZipFile 객체

    Returns:
        추출된 메타데이터 딕셔너리
    """
    metadata = {}

    try:
        # Try to read header.xml for document properties
        for header_path in HEADER_FILE_PATHS:
            if header_path in zf.namelist():
                with zf.open(header_path) as f:
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
                                metadata[tag.lower()] = prop.text
                break

        # Try to read version.xml
        if 'version.xml' in zf.namelist():
            with zf.open('version.xml') as f:
                version_content = f.read()
                version_root = ET.fromstring(version_content)

                # Get version info
                if version_root.text:
                    metadata['version'] = version_root.text
                for attr in version_root.attrib:
                    metadata[f'version_{attr}'] = version_root.get(attr)

        # Try to read META-INF/manifest.xml for additional info
        if 'META-INF/manifest.xml' in zf.namelist():
            with zf.open('META-INF/manifest.xml') as f:
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
                                metadata['media_type'] = media_type

        logger.info(f"Extracted HWPX metadata: {metadata}")

    except Exception as e:
        logger.warning(f"Failed to extract HWPX metadata: {e}")

    return metadata


def parse_bin_item_map(zf: zipfile.ZipFile) -> Dict[str, str]:
    """
    content.hpf 파일을 파싱하여 BinItem ID와 파일 경로 매핑을 생성합니다.

    Args:
        zf: 열린 ZipFile 객체

    Returns:
        BinItem ID -> 파일 경로 매핑 딕셔너리
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
