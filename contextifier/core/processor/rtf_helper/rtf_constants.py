# contextifier/core/processor/rtf_helper/rtf_constants.py
"""
RTF Constants

Constants used for RTF parsing.
"""

# Shape property names (to be removed)
SHAPE_PROPERTY_NAMES = [
    'shapeType', 'fFlipH', 'fFlipV', 'rotation',
    'posh', 'posrelh', 'posv', 'posrelv',
    'fLayoutInCell', 'fAllowOverlap', 'fBehindDocument',
    'fPseudoInline', 'fLockAnchor', 'fLockPosition',
    'fLockAspectRatio', 'fLockRotation', 'fLockAgainstSelect',
    'fLockCropping', 'fLockVerticies', 'fLockText',
    'fLockAdjustHandles', 'fLockAgainstGrouping',
    'geoLeft', 'geoTop', 'geoRight', 'geoBottom',
    'shapePath', 'pWrapPolygonVertices', 'dxWrapDistLeft',
    'dyWrapDistTop', 'dxWrapDistRight', 'dyWrapDistBottom',
    'fLine', 'fFilled', 'fillType', 'fillColor',
    'fillOpacity', 'fillBackColor', 'fillBackOpacity',
    'lineColor', 'lineOpacity', 'lineWidth', 'lineStyle',
    'lineDashing', 'lineStartArrowhead', 'lineStartArrowWidth',
    'lineStartArrowLength', 'lineEndArrowhead', 'lineEndArrowWidth',
    'lineEndArrowLength', 'shadowType', 'shadowColor',
    'shadowOpacity', 'shadowOffsetX', 'shadowOffsetY',
]

# RTF destination 키워드 (제외 대상)
EXCLUDE_DESTINATION_KEYWORDS = [
    'fonttbl', 'colortbl', 'stylesheet', 'listtable',
    'listoverridetable', 'revtbl', 'rsidtbl', 'generator',
    'info', 'xmlnstbl', 'mmathPr', 'themedata', 'colorschememapping',
    'datastore', 'latentstyles', 'pgptbl', 'protusertbl',
]

# RTF skip destinations
SKIP_DESTINATIONS = {
    'fonttbl', 'colortbl', 'stylesheet', 'listtable',
    'listoverridetable', 'revtbl', 'rsidtbl', 'generator',
    'xmlnstbl', 'mmathPr', 'themedata', 'colorschememapping',
    'datastore', 'latentstyles', 'pgptbl', 'protusertbl',
    'bookmarkstart', 'bookmarkend', 'bkmkstart', 'bkmkend',
    'fldinst', 'fldrslt',  # field instructions and results
}

# Image-related destinations
IMAGE_DESTINATIONS = {
    'pict', 'shppict', 'nonshppict', 'blipuid',
}

# Codepage to encoding mapping
CODEPAGE_ENCODING_MAP = {
    437: 'cp437',
    850: 'cp850',
    852: 'cp852',
    855: 'cp855',
    857: 'cp857',
    860: 'cp860',
    861: 'cp861',
    863: 'cp863',
    865: 'cp865',
    866: 'cp866',
    869: 'cp869',
    874: 'cp874',
    932: 'cp932',     # Japanese
    936: 'gb2312',    # Simplified Chinese
    949: 'cp949',     # Korean
    950: 'big5',      # Traditional Chinese
    1250: 'cp1250',   # Central European
    1251: 'cp1251',   # Cyrillic
    1252: 'cp1252',   # Western European
    1253: 'cp1253',   # Greek
    1254: 'cp1254',   # Turkish
    1255: 'cp1255',   # Hebrew
    1256: 'cp1256',   # Arabic
    1257: 'cp1257',   # Baltic
    1258: 'cp1258',   # Vietnamese
    10000: 'mac_roman',
    65001: 'utf-8',
}

# Default encodings to try
DEFAULT_ENCODINGS = ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']


__all__ = [
    'SHAPE_PROPERTY_NAMES',
    'EXCLUDE_DESTINATION_KEYWORDS',
    'SKIP_DESTINATIONS',
    'IMAGE_DESTINATIONS',
    'CODEPAGE_ENCODING_MAP',
    'DEFAULT_ENCODINGS',
]
