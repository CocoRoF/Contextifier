# hwpx_helper/hwpx_constants.py
"""
HWPX Handler 상수 및 네임스페이스 정의

HWPX (ZIP/XML 기반 한글 문서) 처리에 필요한 상수와 네임스페이스를 정의합니다.
"""

# HWPX XML 네임스페이스
HWPX_NAMESPACES = {
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hc': 'http://www.hancom.co.kr/hwpml/2011/core',
    'hh': 'http://www.hancom.co.kr/hwpml/2011/head',
}

# OPF 네임스페이스 (content.hpf 파싱용)
OPF_NAMESPACES = {
    'opf': 'http://www.idpf.org/2007/opf',
}

# 지원하는 이미지 확장자
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# 건너뛸 이미지 확장자 (차트 추출 시)
SKIP_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.wmf', '.emf']

# HWPX 메타데이터 파일 경로 후보
HEADER_FILE_PATHS = ['Contents/header.xml', 'header.xml']

# HWPX 콘텐츠 파일 경로
HPF_PATH = "Contents/content.hpf"
