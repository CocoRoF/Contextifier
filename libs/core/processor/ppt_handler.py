# your_package/document_processor/ppt_handler.py
"""
PPT Handler - PPT/PPTX 문서 처리기

리팩터링된 구조:
- 기능별 로직은 ppt_helper/ 모듈로 분리
- 이 파일은 조합 및 조율 역할

주요 기능:
- 메타데이터 추출 (제목, 작성자, 생성일, 수정일 등)
- 텍스트 추출 (python-pptx를 통한 직접 파싱)
- 테이블 추출 (HTML 형식 보존, rowspan/colspan 지원)
- 인라인 이미지 추출 및 로컬 저장
- 차트 데이터 추출 (카테고리, 시리즈, 값)

모든 처리는 python-pptx를 통한 직접 Binary 파싱으로 수행됩니다.
"""
import logging
from typing import Any, Dict, List

from pptx import Presentation

from libs.core.processor.ppt_helper import (
    # Constants & Types
    ElementType,
    SlideElement,
    # Metadata
    extract_ppt_metadata,
    format_metadata,
    # Bullet
    extract_text_with_bullets,
    # Table
    is_simple_table,
    extract_simple_table_as_text,
    convert_table_to_html,
    extract_table_as_text,
    # Chart
    extract_chart_data,
    # Shape
    get_shape_position,
    is_picture_shape,
    process_image_shape,
    process_group_shape,
    # Slide
    extract_slide_notes,
    merge_slide_elements,
)

logger = logging.getLogger("document-processor")


# === 메인 함수 ===

def extract_text_from_ppt(
    file_path: str,
    current_config: Dict[str, Any] = None,
    extract_default_metadata: bool = True
) -> str:
    """
    PPT/PPTX 파일에서 텍스트를 추출합니다.

    python-pptx를 사용한 직접 Binary 파싱으로 처리합니다.

    Args:
        file_path: PPT/PPTX 파일 경로
        current_config: 설정 딕셔너리 (현재 사용하지 않음)
        extract_default_metadata: 기본 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (인라인 이미지 태그, 테이블 HTML 포함)
    """
    logger.info(f"PPT processing: {file_path}")

    # enhanced가 기본이며, 모든 경우에 고도화된 파싱 사용
    return _extract_ppt_enhanced(file_path, extract_default_metadata)


# === 고도화된 PPT 처리 함수들 ===

def _extract_ppt_enhanced(
    file_path: str,
    extract_default_metadata: bool = True
) -> str:
    """
    고도화된 PPT 처리.

    - 메타데이터 추출 (제목, 작성자, 생성일 등)
    - 인라인 이미지 추출 및 로컬 저장
    - 테이블 HTML 형식 보존 (셀 병합 지원)
    - 차트 데이터 추출 (카테고리, 시리즈, 값)
    - 슬라이드 요소 위치 기반 정렬

    Args:
        file_path: PPT/PPTX 파일 경로
        extract_default_metadata: 기본 메타데이터 추출 여부 (기본값: True)
    """
    logger.info(f"Enhanced PPT processing: {file_path}")

    try:
        prs = Presentation(file_path)
        result_parts = []
        processed_images = set()
        total_tables = 0
        total_images = 0

        # 메타데이터 추출 및 추가 (using helper)
        if extract_default_metadata:
            metadata = extract_ppt_metadata(prs)
            metadata_text = format_metadata(metadata)
            if metadata_text:
                result_parts.append(metadata_text)
                result_parts.append("")  # 빈 줄

        for slide_idx, slide in enumerate(prs.slides):
            # 슬라이드 헤더
            result_parts.append(f"\n=== 슬라이드 {slide_idx + 1} ===\n")

            # 슬라이드의 모든 요소 수집
            elements: List[SlideElement] = []

            for shape in slide.shapes:
                try:
                    # 위치 정보 (using helper)
                    position = get_shape_position(shape)
                    shape_id = shape.shape_id if hasattr(shape, 'shape_id') else id(shape)

                    # 테이블 처리 (using helper)
                    if shape.has_table:
                        # 단순 표(1xN, Nx1, 2x2 이하)는 텍스트로 처리
                        if is_simple_table(shape.table):
                            simple_text = extract_simple_table_as_text(shape.table)
                            if simple_text:
                                elements.append(SlideElement(
                                    element_type=ElementType.TEXT,
                                    content=simple_text,
                                    position=position,
                                    shape_id=shape_id
                                ))
                        else:
                            # 일반 표는 HTML로 처리
                            table_html = convert_table_to_html(shape.table)
                            if table_html:
                                total_tables += 1
                                elements.append(SlideElement(
                                    element_type=ElementType.TABLE,
                                    content=table_html,
                                    position=position,
                                    shape_id=shape_id
                                ))

                    # 이미지 처리 (using helper)
                    elif is_picture_shape(shape):
                        image_tag = process_image_shape(shape, processed_images)
                        if image_tag:
                            total_images += 1
                            elements.append(SlideElement(
                                element_type=ElementType.IMAGE,
                                content=image_tag,
                                position=position,
                                shape_id=shape_id
                            ))

                    # 차트 처리 (using helper)
                    elif shape.has_chart:
                        chart_text = extract_chart_data(shape.chart)
                        if chart_text:
                            elements.append(SlideElement(
                                element_type=ElementType.CHART,
                                content=chart_text,
                                position=position,
                                shape_id=shape_id
                            ))

                    # 텍스트 처리 - 목록 정보 포함 (using helper)
                    elif hasattr(shape, "text_frame") and shape.text_frame:
                        text_content = extract_text_with_bullets(shape.text_frame)
                        if text_content:
                            elements.append(SlideElement(
                                element_type=ElementType.TEXT,
                                content=text_content,
                                position=position,
                                shape_id=shape_id
                            ))

                    # 기존 text 속성만 있는 경우 (폴백)
                    elif hasattr(shape, "text") and shape.text.strip():
                        elements.append(SlideElement(
                            element_type=ElementType.TEXT,
                            content=shape.text.strip(),
                            position=position,
                            shape_id=shape_id
                        ))

                    # 그룹 Shape 처리 (using helper)
                    elif hasattr(shape, "shapes"):
                        group_elements = process_group_shape(shape, processed_images)
                        elements.extend(group_elements)

                except Exception as shape_e:
                    logger.warning(f"Error processing shape in slide {slide_idx + 1}: {shape_e}")
                    continue

            # 요소들을 위치 기준으로 정렬
            elements.sort(key=lambda e: e.sort_key)

            # 요소들을 결합 (using helper)
            slide_content = merge_slide_elements(elements)

            if slide_content.strip():
                result_parts.append(slide_content)
            else:
                result_parts.append("[빈 슬라이드]\n")

            # 슬라이드 노트 추가 (using helper)
            notes_text = extract_slide_notes(slide)
            if notes_text:
                result_parts.append(f"\n[슬라이드 노트]\n{notes_text}\n")

        result = "".join(result_parts)
        logger.info(f"Enhanced PPT processing completed: {len(prs.slides)} slides, {total_tables} tables, {total_images} images")

        return result

    except Exception as e:
        logger.error(f"Error in enhanced PPT processing: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        # 폴백: 간단한 텍스트 추출 (무한 재귀 방지)
        return _extract_ppt_simple_text(file_path)


def _extract_ppt_simple_text(file_path: str) -> str:
    """
    간단한 텍스트 추출 (폴백용).
    에러 발생 시에도 안정적으로 동작합니다.
    """
    try:
        prs = Presentation(file_path)
        result_parts = []

        for slide_idx, slide in enumerate(prs.slides):
            result_parts.append(f"\n=== 슬라이드 {slide_idx + 1} ===\n")

            slide_texts = []
            for shape in slide.shapes:
                try:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_texts.append(shape.text.strip())
                    elif hasattr(shape, "table"):
                        # 테이블은 평문으로 추출 (using helper)
                        table = shape.table
                        table_text = extract_table_as_text(table)
                        if table_text:
                            slide_texts.append(table_text)
                except Exception:
                    continue

            if slide_texts:
                result_parts.append("\n".join(slide_texts) + "\n")
            else:
                result_parts.append("[빈 슬라이드]\n")

        return "".join(result_parts)

    except Exception as e:
        logger.error(f"Error in simple PPT text extraction: {e}")
        return f"[PPT 파일 처리 실패: {str(e)}]"


def _extract_ppt_text_only(file_path: str) -> str:
    """
    텍스트만 추출합니다 (이미지/테이블 태그 없음).
    """
    logger.info(f"Text-only PPT processing: {file_path}")

    try:
        prs = Presentation(file_path)
        result_parts = []

        for slide_idx, slide in enumerate(prs.slides):
            result_parts.append(f"\n=== 슬라이드 {slide_idx + 1} ===\n")

            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
                elif shape.has_table:
                    # 테이블은 평문으로 추출 (using helper)
                    table_text = extract_table_as_text(shape.table)
                    if table_text:
                        slide_texts.append(table_text)

            if slide_texts:
                result_parts.append("\n".join(slide_texts) + "\n")
            else:
                result_parts.append("[빈 슬라이드]\n")

        return "".join(result_parts)

    except Exception as e:
        logger.error(f"Error in text-only PPT processing: {e}")
        return _extract_ppt_simple_text(file_path)
