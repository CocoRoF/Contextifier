"""
PPT 슬라이드 처리 모듈

포함 함수:
- extract_slide_notes(): 슬라이드 노트 추출
- merge_slide_elements(): 슬라이드 요소들을 병합하여 최종 텍스트 생성
"""
import logging
from typing import List, Optional

from .ppt_constants import ElementType, SlideElement

logger = logging.getLogger("document-processor")


def extract_slide_notes(slide) -> Optional[str]:
    """
    슬라이드 노트를 추출합니다.
    
    Args:
        slide: python-pptx Slide 객체
        
    Returns:
        노트 텍스트 또는 None
    """
    try:
        if hasattr(slide, "notes_slide") and slide.notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame:
                notes_text = notes_frame.text.strip()
                if notes_text:
                    return notes_text
    except Exception:
        pass
    return None


def merge_slide_elements(elements: List[SlideElement]) -> str:
    """
    슬라이드 요소들을 병합하여 최종 텍스트를 생성합니다.
    
    각 요소 타입에 맞게 적절한 포맷팅을 적용합니다:
    - TABLE: 앞뒤 줄바꿈 추가
    - IMAGE: 그대로 출력 (이미 줄바꿈 포함)
    - CHART: 앞뒤 줄바꿈 추가
    - TEXT: 뒤에 줄바꿈 추가
    
    Args:
        elements: SlideElement 리스트 (위치 기준 정렬된 상태)
        
    Returns:
        병합된 텍스트
    """
    if not elements:
        return ""

    result_parts = []

    for element in elements:
        if element.element_type == ElementType.TABLE:
            result_parts.append("\n" + element.content + "\n")
        elif element.element_type == ElementType.IMAGE:
            result_parts.append(element.content)
        elif element.element_type == ElementType.CHART:
            result_parts.append("\n" + element.content + "\n")
        elif element.element_type == ElementType.TEXT:
            result_parts.append(element.content + "\n")

    return "".join(result_parts)
