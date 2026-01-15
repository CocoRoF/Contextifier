# service/document_processor/processor/docx_helper/docx_paragraph.py
"""
DOCX Paragraph 처리 유틸리티

DOCX 문서의 Paragraph 요소를 처리합니다.
- process_paragraph_element: Paragraph 요소 처리
- has_page_break_element: 페이지 브레이크 확인
"""
import logging
from typing import Set, Tuple

from docx import Document

from .docx_constants import ElementType, NAMESPACES
from .docx_drawing import process_drawing_element
from .docx_image import process_pict_element

logger = logging.getLogger("document-processor")


def process_paragraph_element(
    para_elem,
    doc: Document,
    app_db,
    processed_images: Set[str],
    file_path: str = None
) -> Tuple[str, bool, int, int]:
    """
    Paragraph 요소를 처리합니다.

    텍스트, 이미지, 차트를 추출하고 페이지 브레이크를 감지합니다.

    Args:
        para_elem: paragraph XML 요소
        doc: python-docx Document 객체
        app_db: 데이터베이스 연결
        processed_images: 처리된 이미지 경로 집합 (중복 방지)
        file_path: 원본 파일 경로

    Returns:
        (content, has_page_break, image_count, chart_count) 튜플
    """
    content_parts = []
    has_page_break = False
    image_count = 0
    chart_count = 0

    try:
        # 페이지 브레이크 확인
        has_page_break = has_page_break_element(para_elem)

        # Run 요소들 순회
        for run_elem in para_elem.findall('.//w:r', NAMESPACES):
            # 텍스트 추출
            for t_elem in run_elem.findall('w:t', NAMESPACES):
                if t_elem.text:
                    content_parts.append(t_elem.text)

            # Drawing (이미지/차트/다이어그램) 처리
            for drawing_elem in run_elem.findall('w:drawing', NAMESPACES):
                drawing_content, drawing_type = process_drawing_element(
                    drawing_elem, doc, app_db, processed_images, file_path
                )
                if drawing_content:
                    content_parts.append(drawing_content)
                    if drawing_type == ElementType.IMAGE:
                        image_count += 1
                    elif drawing_type == ElementType.CHART:
                        chart_count += 1

            # pict 요소 (레거시 VML 이미지) 처리
            for pict_elem in run_elem.findall('w:pict', NAMESPACES):
                pict_content = process_pict_element(pict_elem, doc, app_db, processed_images)
                if pict_content:
                    content_parts.append(pict_content)
                    image_count += 1

    except Exception as e:
        logger.warning(f"Error processing paragraph: {e}")
        # 폴백: 단순 텍스트 추출
        try:
            texts = para_elem.findall('.//w:t', NAMESPACES)
            content_parts = [t.text or '' for t in texts]
        except:
            pass

    return ''.join(content_parts), has_page_break, image_count, chart_count


def has_page_break_element(element) -> bool:
    """
    요소에 페이지 브레이크가 있는지 확인합니다.

    Args:
        element: XML 요소

    Returns:
        페이지 브레이크 존재 여부
    """
    try:
        # 명시적 페이지 브레이크
        if element.findall('.//w:br[@w:type="page"]', NAMESPACES):
            return True
        # 렌더링된 페이지 브레이크
        if element.findall('.//w:lastRenderedPageBreak', NAMESPACES):
            return True
        return False
    except Exception:
        return False


__all__ = [
    'process_paragraph_element',
    'has_page_break_element',
]
