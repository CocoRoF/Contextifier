# hwpx_helper/hwpx_section.py
"""
HWPX 섹션 파싱

HWPX 문서의 섹션 XML을 파싱하여 텍스트, 테이블, 이미지를 추출합니다.
"""
import logging
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, Set, Optional

from contextifier.core.processor.hwpx_helper.hwpx_constants import HWPX_NAMESPACES
from contextifier.core.processor.hwpx_helper.hwpx_table_extractor import parse_hwpx_table

from contextifier.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")


def parse_hwpx_section(
    xml_content: bytes,
    zf: zipfile.ZipFile = None,
    bin_item_map: Dict[str, str] = None,
    processed_images: Set[str] = None,
    image_processor: ImageProcessor = None
) -> str:
    """
    HWPX 섹션 XML을 파싱합니다.

    문단, 테이블, 인라인 이미지를 처리합니다.

    HWPX structure:
    - <hs:sec> -> <hp:p> (최상위 문단)
    - <hp:p> -> <hp:run> -> <hp:t> (Text)
    - <hp:p> -> <hp:run> -> <hp:tbl> (Table)
    - <hp:p> -> <hp:run> -> <hp:ctrl> -> <hc:pic> (Image)

    Args:
        xml_content: 섹션 XML 바이너리 데이터
        zf: ZipFile 객체 (이미지 추출용)
        bin_item_map: BinItem ID -> 파일 경로 매핑
        processed_images: 처리된 이미지 경로 집합 (중복 방지)

    Returns:
        추출된 텍스트 문자열
    """
    try:
        root = ET.fromstring(xml_content)
        ns = HWPX_NAMESPACES

        text_parts = []

        # 최상위 레벨의 hp:p만 처리 (테이블 내부의 hp:p는 테이블 파서에서 처리)
        for p in root.findall('hp:p', ns):
            p_text = []
            for run in p.findall('hp:run', ns):
                # Text
                t = run.find('hp:t', ns)
                if t is not None and t.text:
                    p_text.append(t.text)

                # Table (직접 hp:run 안에 hp:tbl로 존재!)
                table = run.find('hp:tbl', ns)
                if table is not None:
                    table_html = parse_hwpx_table(table, ns)
                    if table_html:
                        p_text.append(f"\n{table_html}\n")

                # Ctrl (Image 등)
                ctrl = run.find('hp:ctrl', ns)
                if ctrl is not None:
                    # 혹시 ctrl 안에 테이블이 있는 경우도 처리
                    ctrl_table = ctrl.find('hp:tbl', ns)
                    if ctrl_table is not None:
                        table_html = parse_hwpx_table(ctrl_table, ns)
                        if table_html:
                            p_text.append(f"\n{table_html}\n")

                    # Image (hc:pic)
                    pic = ctrl.find('hc:pic', ns)
                    if pic is not None and zf and bin_item_map:
                        image_text = _process_inline_image(
                            pic, zf, bin_item_map, processed_images, image_processor
                        )
                        if image_text:
                            p_text.append(image_text)

            if p_text:
                text_parts.append("".join(p_text))

        return "\n".join(text_parts)

    except Exception as e:
        logger.error(f"Error parsing HWPX XML: {e}")
        return ""


def _process_inline_image(
    pic: ET.Element,
    zf: zipfile.ZipFile,
    bin_item_map: Dict[str, str],
    processed_images: Optional[Set[str]],
    image_processor: ImageProcessor
) -> str:
    """
    인라인 이미지를 처리합니다.

    Args:
        pic: hc:pic 요소
        zf: ZipFile 객체
        bin_item_map: BinItem ID -> 파일 경로 매핑
        processed_images: 처리된 이미지 경로 집합
        image_processor: 이미지 프로세서 인스턴스

    Returns:
        이미지 태그 문자열 또는 빈 문자열
    """
    try:
        bin_item_id = pic.get('BinItem')
        if bin_item_id not in bin_item_map:
            return ""

        img_path = bin_item_map[bin_item_id]

        # HWPX href might be relative. Usually "BinData/xxx.png"
        full_path = img_path
        if full_path not in zf.namelist():
            if f"Contents/{img_path}" in zf.namelist():
                full_path = f"Contents/{img_path}"

        if full_path not in zf.namelist():
            return ""

        with zf.open(full_path) as f:
            image_data = f.read()

        image_tag = image_processor.save_image(image_data)
        if image_tag:
            if processed_images is not None:
                processed_images.add(full_path)
            return f"\n{image_tag}\n"

    except Exception as e:
        logger.warning(f"Failed to process inline image: {e}")

    return ""
