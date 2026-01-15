"""
PPT Helper 모듈

PPT/PPTX 문서 처리를 위한 헬퍼 함수 모음.

모듈 구성:
- ppt_constants: 상수, 매핑 테이블, 타입 정의
- ppt_metadata: 메타데이터 추출/포맷팅
- ppt_bullet: 목록(Bullet/Numbering) 처리
- ppt_table: 테이블 처리 (HTML 변환, 병합)
- ppt_chart: 차트 데이터 추출
- ppt_shape: Shape 처리 (위치, 이미지, 그룹)
- ppt_slide: 슬라이드 처리 (노트, 요소 병합)
"""

# === Constants ===
from .ppt_constants import (
    WINGDINGS_MAPPING,
    WINGDINGS_CHAR_MAPPING,
    SYMBOL_MAPPING,
    ElementType,
    SlideElement,
)

# === Metadata ===
from .ppt_metadata import (
    extract_ppt_metadata,
    format_metadata,
)

# === Bullet/Numbering ===
from .ppt_bullet import (
    extract_text_with_bullets,
    extract_bullet_info,
    convert_special_font_char,
)

# === Table ===
from .ppt_table import (
    is_simple_table,
    extract_simple_table_as_text,
    convert_table_to_html,
    extract_table_as_text,
    debug_table_structure,
)

# === Chart ===
from .ppt_chart import (
    extract_chart_data,
)

# === Shape ===
from .ppt_shape import (
    get_shape_position,
    is_picture_shape,
    process_image_shape,
    process_group_shape,
)

# === Slide ===
from .ppt_slide import (
    extract_slide_notes,
    merge_slide_elements,
)


__all__ = [
    # Constants
    "WINGDINGS_MAPPING",
    "WINGDINGS_CHAR_MAPPING",
    "SYMBOL_MAPPING",
    "ElementType",
    "SlideElement",
    # Metadata
    "extract_ppt_metadata",
    "format_metadata",
    # Bullet
    "extract_text_with_bullets",
    "extract_bullet_info",
    "convert_special_font_char",
    # Table
    "is_simple_table",
    "extract_simple_table_as_text",
    "convert_table_to_html",
    "extract_table_as_text",
    "debug_table_structure",
    # Chart
    "extract_chart_data",
    # Shape
    "get_shape_position",
    "is_picture_shape",
    "process_image_shape",
    "process_group_shape",
    # Slide
    "extract_slide_notes",
    "merge_slide_elements",
]
