# service/document_processor/processor/docx_helper/docx_drawing.py
"""
DOCX Drawing 요소 처리 유틸리티

DOCX 문서의 Drawing 요소(이미지, 차트, 다이어그램)를 처리합니다.
- process_drawing_element: Drawing 요소 처리 (이미지/차트/다이어그램 분기)
- extract_chart_from_drawing: Drawing에서 차트 추출
- parse_chart_data_enhanced: 차트 파트에서 데이터 파싱
- extract_diagram_from_drawing: Drawing에서 다이어그램 추출
"""
import logging
from typing import Any, Optional, Set, Tuple

from docx import Document
from docx.oxml.ns import qn

from libs.core.processor.docx_helper.docx_constants import ElementType, NAMESPACES
from libs.core.processor.docx_helper.docx_image import extract_image_from_drawing
from libs.core.processor.docx_helper.docx_chart import parse_ooxml_chart_xml, format_chart_data, parse_chart_data_basic
from libs.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")


def process_drawing_element(
    drawing_elem,
    doc: Document,
    processed_images: Set[str],
    file_path: str = None,
    image_processor: Optional[ImageProcessor] = None
) -> Tuple[str, Optional[ElementType]]:
    """
    Drawing 요소를 처리합니다 (이미지, 차트, 다이어그램).

    Args:
        drawing_elem: drawing XML 요소
        doc: python-docx Document 객체
        processed_images: 처리된 이미지 경로 집합 (중복 방지)
        file_path: 원본 파일 경로
        image_processor: ImageProcessor 인스턴스

    Returns:
        (content, element_type) 튜플
    """
    try:
        # inline 또는 anchor 확인
        inline = drawing_elem.find('.//wp:inline', NAMESPACES)
        anchor = drawing_elem.find('.//wp:anchor', NAMESPACES)

        container = inline if inline is not None else anchor
        if container is None:
            return "", None

        # graphic 데이터 확인
        graphic = container.find('.//a:graphic', NAMESPACES)
        if graphic is None:
            return "", None

        graphic_data = graphic.find('a:graphicData', NAMESPACES)
        if graphic_data is None:
            return "", None

        uri = graphic_data.get('uri', '')

        # 이미지인 경우
        if 'picture' in uri.lower():
            return extract_image_from_drawing(graphic_data, doc, processed_images, image_processor)

        # 차트인 경우
        if 'chart' in uri.lower():
            return extract_chart_from_drawing(graphic_data, doc, file_path)

        # 다이어그램인 경우
        if 'diagram' in uri.lower():
            return extract_diagram_from_drawing(graphic_data, doc)

        # 기타 drawing
        return "", None

    except Exception as e:
        logger.warning(f"Error processing drawing element: {e}")
        return "", None


def extract_chart_from_drawing(graphic_data, doc: Document, file_path: str = None) -> Tuple[str, Optional[ElementType]]:
    """
    Drawing에서 차트 정보를 추출합니다.

    DOCX의 차트는 OOXML DrawingML Chart 형식(ISO/IEC 29500)으로 저장됩니다.
    차트 데이터는 word/charts/chart*.xml 파일에 저장됩니다.

    Args:
        graphic_data: graphicData XML 요소
        doc: python-docx Document 객체
        file_path: 원본 파일 경로

    Returns:
        (content, element_type) 튜플
    """
    try:
        # 차트 참조 찾기
        chart_ref = graphic_data.find('.//c:chart', NAMESPACES)
        if chart_ref is None:
            # 다른 namespace 시도
            for elem in graphic_data.iter():
                if 'chart' in elem.tag.lower():
                    chart_ref = elem
                    break

        if chart_ref is None:
            return "[차트]", ElementType.CHART

        # Relationship ID
        rId = chart_ref.get(qn('r:id'))
        if not rId:
            return "[차트]", ElementType.CHART

        try:
            rel = doc.part.rels.get(rId)
            if rel is None:
                return "[차트]", ElementType.CHART

            # 차트 파트에서 데이터 추출 시도
            if hasattr(rel, 'target_part'):
                chart_part = rel.target_part
                chart_content = parse_chart_data_enhanced(chart_part)
                if chart_content:
                    return chart_content, ElementType.CHART

            return "[차트]", ElementType.CHART

        except Exception as e:
            logger.warning(f"Error extracting chart data: {e}")
            return "[차트]", ElementType.CHART

    except Exception as e:
        logger.warning(f"Error extracting chart from drawing: {e}")
        return "[차트]", ElementType.CHART


def parse_chart_data_enhanced(chart_part) -> str:
    """
    차트 파트에서 OOXML 차트 데이터를 파싱합니다.

    DrawingML Chart 사양에 따라 다음을 추출합니다:
    - 차트 제목
    - 차트 유형
    - 카테고리 (X축 레이블)
    - 시리즈 이름 및 값

    Args:
        chart_part: 차트 파트 객체

    Returns:
        포맷된 차트 데이터 문자열
    """
    try:
        if not hasattr(chart_part, 'blob'):
            return ""

        chart_xml = chart_part.blob

        # OOXML 파서 사용
        chart_info = parse_ooxml_chart_xml(chart_xml)

        if chart_info and chart_info.get('series'):
            return format_chart_data(chart_info)

        # 폴백: 기본 정보 추출
        return parse_chart_data_basic(chart_xml)

    except Exception as e:
        logger.debug(f"Error parsing chart data: {e}")
        return ""


def extract_diagram_from_drawing(graphic_data, doc: Document) -> Tuple[str, Optional[ElementType]]:
    """
    Drawing에서 다이어그램 정보를 추출합니다.

    Args:
        graphic_data: graphicData XML 요소
        doc: python-docx Document 객체

    Returns:
        (content, element_type) 튜플
    """
    try:
        # 다이어그램의 텍스트 추출 시도
        texts = []
        for t_elem in graphic_data.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
            if t_elem.text:
                texts.append(t_elem.text.strip())

        if texts:
            return f"[다이어그램: {' / '.join(texts)}]", ElementType.DIAGRAM

        return "[다이어그램]", ElementType.DIAGRAM

    except Exception as e:
        logger.warning(f"Error extracting diagram from drawing: {e}")
        return "[다이어그램]", ElementType.DIAGRAM


__all__ = [
    'process_drawing_element',
    'extract_chart_from_drawing',
    'parse_chart_data_enhanced',
    'extract_diagram_from_drawing',
]
