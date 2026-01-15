"""
PPT 상수 및 타입 정의 모듈

포함 내용:
- Wingdings/Symbol 폰트 매핑 테이블
- ElementType Enum
- SlideElement dataclass
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


# === Wingdings/Symbol 폰트 매핑 테이블 ===
# PPT에서 특수 폰트(Wingdings, Symbol 등)로 표시되는 목록 기호를
# 올바른 Unicode 문자로 변환하기 위한 매핑 테이블

WINGDINGS_MAPPING = {
    # 기본 도형
    0x6C: '●',   # 'l' -> 검정 원 (filled circle)
    0x6D: '○',   # 'm' -> 빈 원 (empty circle)  
    0x6E: '■',   # 'n' -> 검정 사각형 (filled square)
    0x6F: '□',   # 'o' -> 빈 사각형 (empty square)
    0x70: '◆',   # 'p' -> 검정 마름모 (filled diamond)
    0x71: '◇',   # 'q' -> 빈 마름모 (empty diamond)
    0x75: '◆',   # 'u' -> 마름모
    0x76: '❖',   # 'v' -> 마름모 변형
    
    # 체크마크/X 마크
    0xFC: '✓',   # 체크마크
    0xFB: '✓',   # 체크마크 변형
    0xFD: '✗',   # X 마크
    0xFE: '✘',   # Heavy X
    
    # 화살표
    0xD8: '➢',   # Ø -> 3D 입체 화살표 (가장 많이 사용)
    0xE0: '➢',   # 오른쪽 화살표
    0xE1: '⬅',   # 왼쪽 화살표
    0xE2: '⬆',   # 위쪽 화살표
    0xE3: '⬇',   # 아래쪽 화살표
    0xE4: '⬌',   # 양방향 화살표
    0xE8: '➢',   # 화살표 (è)
    0xE9: '➣',   # 화살표 변형
    0xEA: '➤',   # 삼각 화살표
    0xF0: '➢',   # 화살표
    0xD0: '➢',   # 화살표
    
    # 손가락 포인터
    0x46: '☞',   # 'F' -> 오른쪽 손가락
    0x47: '☜',   # 'G' -> 왼쪽 손가락
    
    # 별/특수 기호
    0xAB: '★',   # 검정 별
    0xAC: '☆',   # 빈 별
    0xA7: '§',   # Section -> 네모로 변환
    
    # 숫자 원
    0x31: '①',   # '1'
    0x32: '②',   # '2'
    0x33: '③',   # '3'
    0x34: '④',   # '4'
    0x35: '⑤',   # '5'
    0x36: '⑥',   # '6'
    0x37: '⑦',   # '7'
    0x38: '⑧',   # '8'
    0x39: '⑨',   # '9'
    0x30: '⓪',   # '0'
}

# 특정 문자에서 Unicode로 직접 매핑 (문자 기반)
WINGDINGS_CHAR_MAPPING = {
    '§': '■',    # Section sign -> 검정 사각형
    'Ø': '➢',    # 3D 입체 화살표 (가장 많이 사용)
    'ü': '✓',    # 체크마크
    'u': '◆',    # 마름모
    'n': '■',    # 검정 사각형
    'l': '●',    # 검정 원
    'o': '□',    # 빈 사각형
    'q': '◇',    # 빈 마름모
    'v': '❖',    # 마름모 변형
    'F': '☞',    # 오른쪽 손가락
    'ð': '➢',    # 화살표
    'Ð': '➢',    # 화살표
    'à': '➢',    # 화살표
    'è': '➢',    # 화살표 (0xE8)
    'ê': '➤',    # 삼각 화살표
}

SYMBOL_MAPPING = {
    0xB7: '•',   # Bullet
    0xD7: '×',   # Multiplication
    0xF7: '÷',   # Division
    0xA5: '∞',   # Infinity
    0xB1: '±',   # Plus-minus
}


# === 슬라이드 요소 타입 정의 ===

class ElementType(Enum):
    """슬라이드 요소 타입"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"


@dataclass
class SlideElement:
    """슬라이드 내 요소를 나타내는 데이터 클래스"""
    element_type: ElementType
    content: str
    position: Tuple[int, int, int, int]  # (left, top, width, height) in EMU
    shape_id: int

    @property
    def sort_key(self) -> Tuple[int, int]:
        """정렬 키: (top, left) - 위에서 아래, 왼쪽에서 오른쪽"""
        return (self.position[1], self.position[0])
