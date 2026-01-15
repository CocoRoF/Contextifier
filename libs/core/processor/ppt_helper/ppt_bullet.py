"""
PPT 목록(Bullet/Numbering) 처리 모듈

포함 함수:
- extract_text_with_bullets(): TextFrame에서 목록 기호 포함 텍스트 추출
- extract_bullet_info(): Paragraph에서 목록 정보 추출
- convert_special_font_char(): 특수 폰트 문자 변환

지원하는 목록 스타일:
- Bullet: •, ○, ■, □, ✓, ➢ 등 모든 Unicode 문자
- Numbering: 1., I., i., A., a., (1), 1) 등
"""
import logging
from typing import Any, Dict

from .ppt_constants import WINGDINGS_MAPPING, WINGDINGS_CHAR_MAPPING, SYMBOL_MAPPING

logger = logging.getLogger("document-processor")


def extract_text_with_bullets(text_frame) -> str:
    """
    TextFrame에서 목록 기호/번호를 포함한 텍스트를 추출합니다.
    
    지원하는 목록 스타일:
    - Bullet: •, ○, ■, □, ✓, ➢ 등 모든 Unicode 문자
    - Numbering: 1., I., i., A., a., (1), 1) 등
    
    Args:
        text_frame: Shape의 text_frame 객체
        
    Returns:
        목록 기호가 포함된 텍스트
    """
    if not text_frame:
        return ""
    
    result_lines = []
    numbering_state = {}  # 레벨별 번호 상태 추적
    
    try:
        for paragraph in text_frame.paragraphs:
            para_text = paragraph.text.strip()
            
            if not para_text:
                result_lines.append("")
                continue
            
            # 들여쓰기 레벨 (0-8)
            level = paragraph.level if hasattr(paragraph, 'level') else 0
            indent = "  " * level  # 2칸씩 들여쓰기
            
            # 목록 정보 추출
            bullet_info = extract_bullet_info(paragraph)
            
            if bullet_info['type'] == 'numbered':
                # 번호 목록 처리
                num_format = bullet_info['format']
                current_num = _get_or_increment_number(numbering_state, level, bullet_info)
                
                # 번호 포맷팅
                formatted_num = _format_number(current_num, num_format)
                result_lines.append(f"{indent}{formatted_num} {para_text}")
                
            elif bullet_info['type'] == 'bulleted':
                # Bullet 목록 처리
                bullet_char = bullet_info['char']
                result_lines.append(f"{indent}{bullet_char} {para_text}")
                
            else:
                # 목록이 아닌 일반 텍스트
                # 목록이 끝나면 번호 상태 초기화
                if numbering_state:
                    numbering_state.clear()
                
                if level > 0:
                    result_lines.append(f"{indent}{para_text}")
                else:
                    result_lines.append(para_text)
    
    except Exception as e:
        logger.warning(f"Error extracting text with bullets: {e}")
        # 폴백: 기본 텍스트만 추출
        return text_frame.text.strip() if text_frame.text else ""
    
    return "\n".join(result_lines)


def extract_bullet_info(paragraph) -> Dict[str, Any]:
    """
    Paragraph에서 목록(bullet/numbering) 정보를 추출합니다.
    특수 폰트(Wingdings, Symbol 등)의 문자를 올바른 Unicode로 변환합니다.
    
    Args:
        paragraph: python-pptx Paragraph 객체
    
    Returns:
        {
            'type': 'none' | 'bulleted' | 'numbered',
            'char': str,           # bullet 문자 (type='bulleted'인 경우)
            'format': str,         # 번호 포맷 (type='numbered'인 경우)
            'start_at': int        # 시작 번호
        }
    """
    result = {
        'type': 'none',
        'char': None,
        'format': None,
        'start_at': 1
    }
    
    try:
        # XML 요소 접근
        pPr = paragraph._element.pPr
        
        if pPr is None:
            return result
        
        # namespace
        ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
        
        # buNone 확인 (목록 비활성화)
        buNone = pPr.find('.//a:buNone', namespaces=ns)
        if buNone is not None:
            return result
        
        # Bullet 폰트 확인 (특수 폰트 여부)
        buFont = pPr.find('.//a:buFont', namespaces=ns)
        font_typeface = None
        if buFont is not None:
            font_typeface = buFont.get('typeface', '').lower()
        
        # Bullet 문자 확인
        buChar = pPr.find('.//a:buChar', namespaces=ns)
        if buChar is not None:
            result['type'] = 'bulleted'
            raw_char = buChar.get('char', '•')
            
            # 특수 폰트인 경우 문자 변환
            if font_typeface:
                converted_char = convert_special_font_char(raw_char, font_typeface)
                result['char'] = converted_char
            else:
                result['char'] = raw_char
            
            return result
        
        # 자동 번호 확인
        buAutoNum = pPr.find('.//a:buAutoNum', namespaces=ns)
        if buAutoNum is not None:
            result['type'] = 'numbered'
            result['format'] = buAutoNum.get('type', 'arabicPeriod')
            result['start_at'] = int(buAutoNum.get('startAt', '1'))
            return result
        
        # Font bullet만 있고 buChar가 없는 경우 (기본 bullet)
        if buFont is not None:
            result['type'] = 'bulleted'
            result['char'] = '•'
            return result
    
    except Exception as e:
        logger.debug(f"Error extracting bullet info: {e}")
    
    return result


def convert_special_font_char(char: str, font_typeface: str) -> str:
    """
    특수 폰트(Wingdings, Symbol 등)의 문자를 일반 Unicode로 변환합니다.
    
    Args:
        char: 원본 문자
        font_typeface: 폰트 이름 (소문자)
    
    Returns:
        변환된 Unicode 문자
    """
    if not char:
        return '•'
    
    try:
        # 먼저 문자 기반 매핑 시도 (가장 정확)
        if 'wingdings' in font_typeface:
            # 문자 자체로 매핑 시도
            if char in WINGDINGS_CHAR_MAPPING:
                return WINGDINGS_CHAR_MAPPING[char]
            
            # 문자 코드로 매핑 시도
            char_code = ord(char[0]) if len(char) > 0 else 0
            if char_code in WINGDINGS_MAPPING:
                return WINGDINGS_MAPPING[char_code]
            
            # 매핑되지 않은 경우 로그 출력 (디버깅용)
            logger.debug(f"Unmapped Wingdings char: '{char}' (code: {char_code}, hex: 0x{char_code:02X})")
            return '•'  # 기본값
        
        # Symbol 폰트
        elif 'symbol' in font_typeface:
            char_code = ord(char[0]) if len(char) > 0 else 0
            if char_code in SYMBOL_MAPPING:
                return SYMBOL_MAPPING[char_code]
            return char
        
        # Webdings 폰트 (필요시 매핑 추가)
        elif 'webdings' in font_typeface:
            return '•'  # 기본값
        
        # 일반 폰트는 그대로 반환
        else:
            return char
    
    except Exception as e:
        logger.debug(f"Error converting special font char: {e}")
        return '•'


def _get_or_increment_number(numbering_state: Dict, level: int, bullet_info: Dict) -> int:
    """
    레벨별 번호를 추적하고 증가시킵니다.
    
    Args:
        numbering_state: 레벨별 번호 상태 딕셔너리
        level: 현재 들여쓰기 레벨
        bullet_info: 목록 정보
        
    Returns:
        현재 번호
    """
    # 새로운 번호 시퀀스 시작
    if level not in numbering_state:
        numbering_state[level] = bullet_info['start_at']
    else:
        numbering_state[level] += 1
    
    # 하위 레벨 초기화
    for l in list(numbering_state.keys()):
        if l > level:
            del numbering_state[l]
    
    return numbering_state[level]


def _format_number(num: int, format_type: str) -> str:
    """
    번호를 지정된 포맷으로 변환합니다.
    
    지원 포맷:
    - arabicPeriod: 1.
    - arabicParenR: 1)
    - arabicParenBoth: (1)
    - romanUcPeriod: I.
    - romanLcPeriod: i.
    - alphaUcPeriod: A.
    - alphaLcPeriod: a.
    - alphaUcParenR: A)
    - alphaLcParenR: a)
    등등...
    
    Args:
        num: 번호
        format_type: 포맷 타입 문자열
    
    Returns:
        포맷팅된 번호 문자열
    """
    # 번호 변환
    if 'roman' in format_type.lower():
        num_str = _to_roman(num)
        if 'Lc' in format_type:  # 소문자
            num_str = num_str.lower()
    elif 'alpha' in format_type.lower():
        num_str = _to_alpha(num)
        if 'Lc' in format_type:  # 소문자
            num_str = num_str.lower()
    else:
        num_str = str(num)
    
    # 구분자 추가
    if 'Period' in format_type:
        return f"{num_str}."
    elif 'ParenBoth' in format_type:
        return f"({num_str})"
    elif 'ParenR' in format_type:
        return f"{num_str})"
    elif 'ParenL' in format_type:
        return f"({num_str}"
    elif 'Plain' in format_type:
        return num_str
    else:
        # 기본값
        return f"{num_str}."


def _to_roman(num: int) -> str:
    """
    숫자를 로마 숫자로 변환합니다.
    
    Args:
        num: 1-3999 범위의 정수
        
    Returns:
        로마 숫자 문자열 (예: 1→I, 4→IV, 9→IX)
    """
    val_map = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    result = []
    for value, letter in val_map:
        count, num = divmod(num, value)
        result.append(letter * count)
    return ''.join(result)


def _to_alpha(num: int) -> str:
    """
    숫자를 알파벳으로 변환합니다.
    
    Args:
        num: 양의 정수
        
    Returns:
        알파벳 문자열 (예: 1→A, 2→B, 26→Z, 27→AA)
    """
    result = []
    while num > 0:
        num -= 1
        result.append(chr(65 + (num % 26)))
        num //= 26
    return ''.join(reversed(result))
