# contextifier/core/processor/hwp5_helper/hwp5_constants.py
"""
HWP 5.0 OLE Format Constants

Defines record tag IDs, chart type codes, and control character codes
for HWP 5.0 OLE format processing.

Record Structure:
- Header (4 bytes): TagID (10 bits) | Level (10 bits) | Size (12 bits)
- If Size == 0xFFF, next 4 bytes contain actual size
- Payload: Variable length data

Reference: HWP 5.0 File Format Specification (한글과컴퓨터)
"""

# ==========================================================================
# HWP 5.0 Tag Constants
# ==========================================================================

HWPTAG_BEGIN = 0x10

# DocInfo 관련
HWPTAG_DOCUMENT_PROPERTIES = HWPTAG_BEGIN + 0   # 16 - Document properties
HWPTAG_ID_MAPPINGS = HWPTAG_BEGIN + 1           # 17 - ID mappings
HWPTAG_BIN_DATA = HWPTAG_BEGIN + 2               # 18 - Binary data info in DocInfo
HWPTAG_FACE_NAME = HWPTAG_BEGIN + 3              # 19 - Font face name
HWPTAG_BORDER_FILL = HWPTAG_BEGIN + 4            # 20 - Border/fill style
HWPTAG_CHAR_SHAPE = HWPTAG_BEGIN + 5             # 21 - Character shape
HWPTAG_TAB_DEF = HWPTAG_BEGIN + 6                # 22 - Tab definition
HWPTAG_NUMBERING = HWPTAG_BEGIN + 7              # 23 - Numbering
HWPTAG_BULLET = HWPTAG_BEGIN + 8                 # 24 - Bullet
HWPTAG_PARA_SHAPE = HWPTAG_BEGIN + 9             # 25 - Paragraph shape
HWPTAG_STYLE = HWPTAG_BEGIN + 10                 # 26 - Style

# Section/Paragraph 관련
HWPTAG_PARA_HEADER = HWPTAG_BEGIN + 50           # 66 - Paragraph header
HWPTAG_PARA_TEXT = HWPTAG_BEGIN + 51             # 67 - Paragraph text
HWPTAG_PARA_CHAR_SHAPE = HWPTAG_BEGIN + 52       # 68 - Paragraph character shape
HWPTAG_PARA_LINE_SEG = HWPTAG_BEGIN + 53         # 69 - Paragraph line segment
HWPTAG_PARA_RANGE_TAG = HWPTAG_BEGIN + 54        # 70 - Paragraph range tag

# Control/Shape 관련
HWPTAG_CTRL_HEADER = HWPTAG_BEGIN + 55           # 71 - Control header
HWPTAG_LIST_HEADER = HWPTAG_BEGIN + 56           # 72 - List header (table cells)
HWPTAG_PAGE_DEF = HWPTAG_BEGIN + 57              # 73 - Page definition
HWPTAG_FOOTNOTE_SHAPE = HWPTAG_BEGIN + 58        # 74 - Footnote shape
HWPTAG_PAGE_BORDER_FILL = HWPTAG_BEGIN + 59      # 75 - Page border fill
HWPTAG_SHAPE_COMPONENT = HWPTAG_BEGIN + 60       # 76 - Shape component (container)
HWPTAG_TABLE = HWPTAG_BEGIN + 61                 # 77 - Table properties
HWPTAG_SHAPE_COMPONENT_LINE = HWPTAG_BEGIN + 62  # 78 - Line shape
HWPTAG_SHAPE_COMPONENT_OLE = HWPTAG_BEGIN + 63   # 79 - OLE object (charts are OLE)
HWPTAG_SHAPE_COMPONENT_RECTANGLE = HWPTAG_BEGIN + 64  # 80 - Rectangle
HWPTAG_SHAPE_COMPONENT_ELLIPSE = HWPTAG_BEGIN + 65    # 81 - Ellipse
HWPTAG_SHAPE_COMPONENT_ARC = HWPTAG_BEGIN + 66        # 82 - Arc
HWPTAG_SHAPE_COMPONENT_POLYGON = HWPTAG_BEGIN + 67    # 83 - Polygon
HWPTAG_SHAPE_COMPONENT_CURVE = HWPTAG_BEGIN + 68      # 84 - Curve
HWPTAG_SHAPE_COMPONENT_PICTURE = HWPTAG_BEGIN + 69    # 85 - Picture shape
HWPTAG_SHAPE_COMPONENT_TEXTART = HWPTAG_BEGIN + 70    # 86 - TextArt
HWPTAG_SHAPE_COMPONENT_CONTAINER = HWPTAG_BEGIN + 71  # 87 - Container

# Memo/Annotation
HWPTAG_MEMO_SHAPE = HWPTAG_BEGIN + 72            # 88 - Memo shape
HWPTAG_MEMO_LIST = HWPTAG_BEGIN + 73             # 89 - Memo list

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
CTRL_CHAR_PARA_BREAK = 0x0D            # Paragraph break
CTRL_CHAR_LINE_BREAK = 0x0A            # Line break
CTRL_CHAR_TAB = 0x09                   # Tab


# ==========================================================================
# Control IDs (reverse byte order)
# ==========================================================================

CTRL_ID_TABLE = b'tbl '      # Table control
CTRL_ID_GSO = b'gso '        # Generic Shape Object
CTRL_ID_SECTION = b'secd'    # Section definition
CTRL_ID_COLUMN = b'cold'     # Column definition
CTRL_ID_HEADER = b'head'     # Header
CTRL_ID_FOOTER = b'foot'     # Footer
CTRL_ID_FOOTNOTE = b'fn  '   # Footnote
CTRL_ID_ENDNOTE = b'en  '    # Endnote
CTRL_ID_AUTO_NUM = b'atno'   # Auto number
CTRL_ID_NEW_NUM = b'nwno'    # New number
CTRL_ID_FIELD = b'%unk'      # Field (unknown type marker)


# ==========================================================================
# File Signatures
# ==========================================================================

# OLE Compound Document signature
OLE_MAGIC = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

# HWP file signature (in FileHeader stream)
HWP_SIGNATURE = b'HWP Document File'


# ==========================================================================
# BinData Storage Types
# ==========================================================================

BINDATA_LINK = 0        # External link
BINDATA_EMBEDDING = 1   # Embedded in DocInfo
BINDATA_STORAGE = 2     # Stored in BinData folder


# ==========================================================================
# Export List
# ==========================================================================

__all__ = [
    # Tag IDs
    'HWPTAG_BEGIN',
    'HWPTAG_DOCUMENT_PROPERTIES',
    'HWPTAG_ID_MAPPINGS',
    'HWPTAG_BIN_DATA',
    'HWPTAG_FACE_NAME',
    'HWPTAG_BORDER_FILL',
    'HWPTAG_CHAR_SHAPE',
    'HWPTAG_TAB_DEF',
    'HWPTAG_NUMBERING',
    'HWPTAG_BULLET',
    'HWPTAG_PARA_SHAPE',
    'HWPTAG_STYLE',
    'HWPTAG_PARA_HEADER',
    'HWPTAG_PARA_TEXT',
    'HWPTAG_PARA_CHAR_SHAPE',
    'HWPTAG_PARA_LINE_SEG',
    'HWPTAG_PARA_RANGE_TAG',
    'HWPTAG_CTRL_HEADER',
    'HWPTAG_LIST_HEADER',
    'HWPTAG_PAGE_DEF',
    'HWPTAG_FOOTNOTE_SHAPE',
    'HWPTAG_PAGE_BORDER_FILL',
    'HWPTAG_SHAPE_COMPONENT',
    'HWPTAG_TABLE',
    'HWPTAG_SHAPE_COMPONENT_LINE',
    'HWPTAG_SHAPE_COMPONENT_OLE',
    'HWPTAG_SHAPE_COMPONENT_PICTURE',
    'HWPTAG_CHART_DATA',
    # Chart types
    'CHART_TYPES',
    # Control chars
    'CTRL_CHAR_DRAWING_TABLE_OBJECT',
    'CTRL_CHAR_PARA_BREAK',
    'CTRL_CHAR_LINE_BREAK',
    'CTRL_CHAR_TAB',
    # Control IDs
    'CTRL_ID_TABLE',
    'CTRL_ID_GSO',
    'CTRL_ID_SECTION',
    'CTRL_ID_COLUMN',
    'CTRL_ID_HEADER',
    'CTRL_ID_FOOTER',
    'CTRL_ID_FOOTNOTE',
    'CTRL_ID_ENDNOTE',
    'CTRL_ID_AUTO_NUM',
    'CTRL_ID_NEW_NUM',
    # File signatures
    'OLE_MAGIC',
    'HWP_SIGNATURE',
    # BinData types
    'BINDATA_LINK',
    'BINDATA_EMBEDDING',
    'BINDATA_STORAGE',
]
