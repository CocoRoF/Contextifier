# service/document_processor/processor/hwp_helper/hwp_constants.py
"""
HWP/HWPX 공통 상수 정의

HWP 5.0 OLE 형식의 레코드 태그 ID, 차트 타입 코드 등을 정의합니다.
"""

# ==========================================================================
# HWP 5.0 Tag Constants
# ==========================================================================

HWPTAG_BEGIN = 0x10

# DocInfo 관련
HWPTAG_BIN_DATA = HWPTAG_BEGIN + 2               # 18 - Binary data info in DocInfo

# Section/Paragraph 관련
HWPTAG_PARA_HEADER = HWPTAG_BEGIN + 50           # 66 - Paragraph header
HWPTAG_PARA_TEXT = HWPTAG_BEGIN + 51             # 67 - Paragraph text

# Control/Shape 관련
HWPTAG_CTRL_HEADER = HWPTAG_BEGIN + 55           # 71 - Control header
HWPTAG_LIST_HEADER = HWPTAG_BEGIN + 56           # 72 - List header (table cells)
HWPTAG_SHAPE_COMPONENT = HWPTAG_BEGIN + 60       # 76 - Shape component (container)
HWPTAG_TABLE = HWPTAG_BEGIN + 61                 # 77 - Table properties
HWPTAG_SHAPE_COMPONENT_OLE = HWPTAG_BEGIN + 63   # 79 - OLE object (charts are OLE)
HWPTAG_SHAPE_COMPONENT_PICTURE = HWPTAG_BEGIN + 69  # 85 - Picture shape

# Chart 관련
HWPTAG_CHART_DATA = HWPTAG_BEGIN + 118           # 134 - Chart data


# ==========================================================================
# Chart Type Constants
# ==========================================================================

# HWP Chart specification에서 정의된 차트 타입 코드
CHART_TYPES = {
    0: '3D 막대', 1: '2D 막대', 2: '3D 선', 3: '2D 선',
    4: '3D 영역', 5: '2D 영역', 6: '3D 계단', 7: '2D 계단',
    8: '3D 조합', 9: '2D 조합', 10: '3D 가로 막대', 11: '2D 가로 막대',
    12: '3D 클러스터 막대', 13: '3D 파이', 14: '2D 파이', 15: '2D 도넛',
    16: '2D XY', 17: '2D 원추', 18: '2D 방사', 19: '2D 풍선',
    20: '2D Hi-Lo', 21: '2D 간트', 22: '3D 간트', 23: '3D 평면',
    24: '2D 등고선', 25: '3D 산포', 26: '3D XYZ'
}


# ==========================================================================
# Control Character Codes
# ==========================================================================

# PARA_TEXT에서 사용되는 컨트롤 문자 코드
CTRL_CHAR_DRAWING_TABLE_OBJECT = 0x0B  # Extended control for GSO (images, tables, etc.)


# ==========================================================================
# Export List
# ==========================================================================

__all__ = [
    # Tag IDs
    'HWPTAG_BEGIN',
    'HWPTAG_BIN_DATA',
    'HWPTAG_PARA_HEADER',
    'HWPTAG_PARA_TEXT',
    'HWPTAG_CTRL_HEADER',
    'HWPTAG_LIST_HEADER',
    'HWPTAG_SHAPE_COMPONENT',
    'HWPTAG_SHAPE_COMPONENT_PICTURE',
    'HWPTAG_TABLE',
    'HWPTAG_SHAPE_COMPONENT_OLE',
    'HWPTAG_CHART_DATA',
    # Chart types
    'CHART_TYPES',
    # Control chars
    'CTRL_CHAR_DRAWING_TABLE_OBJECT',
]
