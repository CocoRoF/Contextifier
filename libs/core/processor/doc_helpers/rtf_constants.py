# service/document_processor/processor/doc_helpers/rtf_constants.py
"""
RTF Parser 상수 정의

RTF 파싱에 사용되는 상수들을 정의합니다.
"""

# Shape 속성 이름들 (\sn으로 시작하는 속성들) - 텍스트에서 제거해야 함
SHAPE_PROPERTY_NAMES = {
    'shapeType', 'fFlipH', 'fFlipV', 'txflTextFlow', 'fFilled', 'fLine',
    'dxTextLeft', 'dxTextRight', 'dyTextTop', 'dyTextBottom',
    'posrelh', 'posrelv', 'fBehindDocument', 'fLayoutInCell', 'fAllowOverlap',
    'fillColor', 'fillBackColor', 'fNoFillHitTest', 'lineColor', 'lineWidth',
    'posh', 'posv', 'fLockAnchor', 'fLockPosition', 'fLockAspectRatio',
    'fLockRotation', 'fLockCropping', 'fLockAgainstGrouping', 'fNoLineDrawDash',
    'wzName', 'wzDescription', 'pWrapPolygonVertices', 'dxWrapDistLeft',
    'dxWrapDistRight', 'dyWrapDistTop', 'dyWrapDistBottom', 'lidRegroup',
    'fEditedWrap', 'fBehindDocument', 'fOnDblClickNotify', 'fIsButton',
    'fOneD', 'fHidden', 'fPrint', 'geoLeft', 'geoTop', 'geoRight', 'geoBottom',
    'shapePath', 'pSegmentInfo', 'pVertices', 'fFillOK', 'fFillShadeShapeOK',
    'fGtextOK', 'fLineOK', 'f3DOK', 'fShadowOK', 'fArrowheadsOK',
}

# 제외할 destination 키워드들 (본문이 아닌 영역)
EXCLUDE_DESTINATION_KEYWORDS = [
    r'\\header(?:f|l|r)?\b',      # 헤더
    r'\\footer(?:f|l|r)?\b',      # 푸터
    r'\\footnote\b',               # 각주
    r'\\ftnsep\b', r'\\ftnsepc\b',  # 각주 구분선
    r'\\aftncn\b', r'\\aftnsep\b', r'\\aftnsepc\b',  # 미주
    r'\\pntext\b', r'\\pntxta\b', r'\\pntxtb\b',  # 번호 매기기
]

# 제거할 destination 패턴들
SKIP_DESTINATIONS = [
    'themedata', 'colorschememapping', 'latentstyles', 'datastore',
    'xmlnstbl', 'wgrffmtfilter', 'generator', 'mmathPr', 'xmlopen',
    'background', 'pgptbl', 'listpicture', 'pnseclvl', 'revtbl',
    'bkmkstart', 'bkmkend', 'fldinst', 'objdata', 'objclass',
    'objemb', 'result', 'category', 'comment', 'company', 'creatim',
    'doccomm', 'hlinkbase', 'keywords', 'manager', 'operator',
    'revtim', 'subject', 'title', 'userprops',
    'nonshppict', 'blipuid', 'picprop',
]

# 이미지 관련 destination
IMAGE_DESTINATIONS = ['shppict']

# 코드 페이지 -> 인코딩 매핑
CODEPAGE_ENCODING_MAP = {
    949: 'cp949',
    932: 'cp932',
    936: 'gb2312',
    950: 'big5',
    1252: 'cp1252',
    65001: 'utf-8',
}

# 기본 인코딩 시도 순서
DEFAULT_ENCODINGS = ['cp949', 'utf-8', 'cp1252', 'latin-1']
