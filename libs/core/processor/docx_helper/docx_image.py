# libs/core/processor/docx_helper/docx_image.py
"""
DOCX 이미지 추출 유틸리티

DOCX 문서에서 이미지를 추출하고 로컬에 저장합니다.
- extract_image_from_drawing: Drawing 요소에서 이미지 추출
- process_pict_element: 레거시 VML pict 요소 처리
"""
import logging
from typing import Optional, Set, Tuple

from docx import Document
from docx.oxml.ns import qn

from libs.core.functions.img_processor import ImageProcessor
from libs.core.processor.docx_helper.docx_constants import ElementType, NAMESPACES

logger = logging.getLogger("document-processor")


def extract_image_from_drawing(
    graphic_data,
    doc: Document,
    processed_images: Set[str],
    image_processor: ImageProcessor
) -> Tuple[str, Optional[ElementType]]:
    """
    Drawing에서 이미지를 추출합니다.

    Args:
        graphic_data: graphicData XML 요소
        doc: python-docx Document 객체
        processed_images: 처리된 이미지 경로 집합 (중복 방지)
        image_processor: ImageProcessor 인스턴스

    Returns:
        (content, element_type) 튜플
    """
    
    try:
        # blip 요소 찾기 (이미지 참조)
        blip = graphic_data.find('.//a:blip', NAMESPACES)
        if blip is None:
            return "", None

        # Relationship ID
        r_embed = blip.get(qn('r:embed'))
        r_link = blip.get(qn('r:link'))

        rId = r_embed or r_link
        if not rId:
            return "", None

        # Relationship에서 이미지 파트 찾기
        try:
            rel = doc.part.rels.get(rId)
            if rel is None:
                return "", None

            # 이미지 데이터 추출
            if hasattr(rel, 'target_part') and hasattr(rel.target_part, 'blob'):
                image_data = rel.target_part.blob

                # 로컬에 저장
                image_tag = image_processor.save_image(image_data, processed_images=processed_images)

                if image_tag:
                    return f"\n{image_tag}\n", ElementType.IMAGE

            return "[이미지]", ElementType.IMAGE

        except Exception as e:
            logger.warning(f"Error extracting image from relationship: {e}")
            return "[이미지]", ElementType.IMAGE

    except Exception as e:
        logger.warning(f"Error extracting image from drawing: {e}")
        return "", None


def process_pict_element(
    pict_elem,
    doc: Document,
    processed_images: Set[str],
    image_processor: ImageProcessor
) -> str:
    """
    레거시 VML pict 요소를 처리합니다.

    Args:
        pict_elem: pict XML 요소
        doc: python-docx Document 객체
        processed_images: 처리된 이미지 경로 집합 (중복 방지)
        image_processor: ImageProcessor 인스턴스

    Returns:
        이미지 마크업 문자열
    """
    
    try:
        # VML imagedata 찾기
        ns_v = 'urn:schemas-microsoft-com:vml'
        ns_r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

        imagedata = pict_elem.find('.//{%s}imagedata' % ns_v)
        if imagedata is None:
            return "[이미지]"

        rId = imagedata.get('{%s}id' % ns_r)
        if not rId:
            return "[이미지]"

        try:
            rel = doc.part.rels.get(rId)
            if rel and hasattr(rel, 'target_part') and hasattr(rel.target_part, 'blob'):
                image_data = rel.target_part.blob
                image_tag = image_processor.save_image(image_data, processed_images=processed_images)
                if image_tag:
                    return f"\n{image_tag}\n"
        except Exception:
            pass

        return "[이미지]"

    except Exception as e:
        logger.warning(f"Error processing pict element: {e}")
        return ""


__all__ = [
    'extract_image_from_drawing',
    'process_pict_element',
]
