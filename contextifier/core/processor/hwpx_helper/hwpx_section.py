# hwpx_helper/hwpx_section.py
"""
HWPX 섹션 파싱

HWPX 문서의 섹션 XML을 파싱하여 텍스트, 테이블, 이미지, 차트를 추출합니다.

테이블 처리:
- HWPXTableExtractor: hp:tbl 요소 → TableData 변환
- HWPXTableProcessor: TableData → HTML/Markdown/Text 출력

차트 처리:
- hp:chart 요소 발견 시 chart_callback 호출
- 원본 문서 순서대로 차트가 삽입됨
"""
import logging
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, Set, Optional, Callable

from contextifier.core.processor.hwpx_helper.hwpx_constants import HWPX_NAMESPACES
from contextifier.core.processor.hwpx_helper.hwpx_table_extractor import (
    HWPXTableExtractor,
)
from contextifier.core.processor.hwpx_helper.hwpx_table_processor import (
    HWPXTableProcessor,
)

from contextifier.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")

# Module-level instances (lazy initialized)
_table_extractor: Optional[HWPXTableExtractor] = None
_table_processor: Optional[HWPXTableProcessor] = None


def _get_table_extractor() -> HWPXTableExtractor:
    """Get or create the module-level table extractor."""
    global _table_extractor
    if _table_extractor is None:
        _table_extractor = HWPXTableExtractor()
    return _table_extractor


def _get_table_processor() -> HWPXTableProcessor:
    """Get or create the module-level table processor."""
    global _table_processor
    if _table_processor is None:
        _table_processor = HWPXTableProcessor()
    return _table_processor


def _process_table(table_element: ET.Element, ns: Dict[str, str]) -> str:
    """Process a table element and return formatted output.
    
    Uses HWPXTableExtractor to convert XML to TableData,
    then HWPXTableProcessor to format as HTML.
    
    Args:
        table_element: hp:tbl XML element
        ns: Namespace dictionary
        
    Returns:
        Formatted table string (HTML)
    """
    extractor = _get_table_extractor()
    processor = _get_table_processor()
    
    table_data = extractor.extract_table(table_element, ns)
    if table_data:
        return processor.format_table(table_data)
    return ""


def parse_hwpx_section(
    xml_content: bytes,
    zf: zipfile.ZipFile = None,
    bin_item_map: Dict[str, str] = None,
    processed_images: Set[str] = None,
    image_processor: ImageProcessor = None,
    chart_callback: Optional[Callable[[str], str]] = None
) -> str:
    """
    HWPX 섹션 XML을 파싱합니다.

    문단, 테이블, 인라인 이미지, 차트를 원본 문서 순서대로 처리합니다.

    HWPX structure:
    - <hs:sec> -> <hp:p> (최상위 문단)
    - <hp:p> -> <hp:run> -> <hp:t> (Text)
    - <hp:p> -> <hp:run> -> <hp:tbl> (Table)
    - <hp:p> -> <hp:run> -> <hp:ctrl> -> <hc:pic> (Image)
    - <hp:p> -> <hp:run> -> <hp:switch> -> <hp:case> -> <hp:chart> (Chart)
    - <hp:p> -> <hp:run> -> <hp:pic> (Direct Image)

    Args:
        xml_content: 섹션 XML 바이너리 데이터
        zf: ZipFile 객체 (이미지 추출용)
        bin_item_map: BinItem ID -> 파일 경로 매핑
        processed_images: 처리된 이미지 경로 집합 (중복 방지)
        image_processor: 이미지 프로세서 인스턴스
        chart_callback: 차트 참조 발견 시 호출할 콜백 함수
                       chartIDRef (예: "Chart/chart1.xml")를 받아 포맷된 차트 텍스트 반환

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
                    table_html = _process_table(table, ns)
                    if table_html:
                        p_text.append(f"\n{table_html}\n")

                # Chart in switch/case (hp:switch > hp:case > hp:chart)
                switch = run.find('hp:switch', ns)
                if switch is not None:
                    case = switch.find('hp:case', ns)
                    if case is not None:
                        chart = case.find('hp:chart', ns)
                        if chart is not None and chart_callback:
                            chart_id_ref = chart.get('chartIDRef')
                            if chart_id_ref:
                                chart_text = chart_callback(chart_id_ref)
                                if chart_text:
                                    p_text.append(f"\n{chart_text}\n")

                # Direct Image (hp:pic directly in hp:run)
                pic = run.find('hp:pic', ns)
                if pic is not None and zf and bin_item_map:
                    image_text = _process_inline_image(
                        pic, zf, bin_item_map, processed_images, image_processor
                    )
                    if image_text:
                        p_text.append(image_text)

                # Ctrl (Image 등)
                ctrl = run.find('hp:ctrl', ns)
                if ctrl is not None:
                    # 혹시 ctrl 안에 테이블이 있는 경우도 처리
                    ctrl_table = ctrl.find('hp:tbl', ns)
                    if ctrl_table is not None:
                        table_html = _process_table(ctrl_table, ns)
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

    HWPX 이미지 구조:
    - <hp:pic> 또는 <hc:pic>
      - <hc:img binaryItemIDRef="image3">
    
    Args:
        pic: hp:pic 또는 hc:pic 요소
        zf: ZipFile 객체
        bin_item_map: BinItem ID -> 파일 경로 매핑
        processed_images: 처리된 이미지 경로 집합
        image_processor: 이미지 프로세서 인스턴스

    Returns:
        이미지 태그 문자열 또는 빈 문자열
    """
    ns = HWPX_NAMESPACES
    
    try:
        # Try to find binaryItemIDRef from nested hc:img element
        img_elem = pic.find('hc:img', ns)
        if img_elem is not None:
            bin_item_id = img_elem.get('binaryItemIDRef')
        else:
            # Fallback: try direct BinItem attribute
            bin_item_id = pic.get('BinItem')
        
        if not bin_item_id or bin_item_id not in bin_item_map:
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
