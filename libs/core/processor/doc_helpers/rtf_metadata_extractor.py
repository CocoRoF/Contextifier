# service/document_processor/processor/doc_helpers/rtf_metadata_extractor.py
"""
RTF 메타데이터 추출기

RTF 문서에서 메타데이터를 추출하는 기능을 제공합니다.
"""
import logging
import re
from datetime import datetime
from typing import Any, Dict

from libs.core.processor.doc_helpers.rtf_decoder import (
    decode_hex_escapes,
)
from libs.core.processor.doc_helpers.rtf_text_cleaner import (
    clean_rtf_text,
)

logger = logging.getLogger("document-processor")


def extract_metadata(content: str, encoding: str = "cp949") -> Dict[str, Any]:
    """
    RTF 콘텐츠에서 메타데이터를 추출합니다.

    Args:
        content: RTF 문자열 콘텐츠
        encoding: 사용할 인코딩

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    # \info 그룹 찾기
    info_match = re.search(r'\\info\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', content)
    if info_match:
        info_content = info_match.group(1)

        # 각 메타데이터 필드 추출
        field_patterns = {
            'title': r'\\title\s*\{([^}]*)\}',
            'subject': r'\\subject\s*\{([^}]*)\}',
            'author': r'\\author\s*\{([^}]*)\}',
            'keywords': r'\\keywords\s*\{([^}]*)\}',
            'comments': r'\\doccomm\s*\{([^}]*)\}',
            'last_saved_by': r'\\operator\s*\{([^}]*)\}',
        }

        for key, pattern in field_patterns.items():
            match = re.search(pattern, info_content)
            if match:
                value = decode_hex_escapes(match.group(1), encoding)
                value = clean_rtf_text(value, encoding)
                if value:
                    metadata[key] = value

        # 날짜 추출
        date_patterns = {
            'create_time': r'\\creatim\\yr(\d+)\\mo(\d+)\\dy(\d+)(?:\\hr(\d+))?(?:\\min(\d+))?',
            'last_saved_time': r'\\revtim\\yr(\d+)\\mo(\d+)\\dy(\d+)(?:\\hr(\d+))?(?:\\min(\d+))?',
        }

        for key, pattern in date_patterns.items():
            match = re.search(pattern, content)
            if match:
                try:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))
                    hour = int(match.group(4)) if match.group(4) else 0
                    minute = int(match.group(5)) if match.group(5) else 0
                    metadata[key] = datetime(year, month, day, hour, minute)
                except (ValueError, TypeError):
                    pass

    logger.debug(f"Extracted metadata: {list(metadata.keys())}")
    return metadata
