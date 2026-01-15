# service/document_processor/processor/docx_helper/docx_constants.py
"""
DOCX 상수 및 타입 정의

DOCX 문서 처리에 필요한 상수, Enum, 데이터클래스를 정의합니다.
- ElementType: 문서 요소 타입 (텍스트, 이미지, 테이블 등)
- DocxElement: 문서 요소 데이터 클래스
- NAMESPACES: OOXML 네임스페이스
- CHART_TYPE_MAP: 차트 타입 매핑
"""
from dataclasses import dataclass
from enum import Enum


# === 문서 요소 타입 정의 ===

class ElementType(Enum):
    """문서 요소 타입"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    PAGE_BREAK = "page_break"


@dataclass
class DocxElement:
    """문서 내 요소를 나타내는 데이터 클래스"""
    element_type: ElementType
    content: str
    element_index: int  # 문서 내 순서


# === OOXML 네임스페이스 ===

NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'dgm': 'http://schemas.openxmlformats.org/drawingml/2006/diagram',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'wps': 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
}

# OOXML 차트 타입 맵핑
CHART_TYPE_MAP = {
    'barChart': '막대 차트',
    'bar3DChart': '3D 막대 차트',
    'lineChart': '선 차트',
    'line3DChart': '3D 선 차트',
    'pieChart': '파이 차트',
    'pie3DChart': '3D 파이 차트',
    'doughnutChart': '도넛 차트',
    'areaChart': '영역 차트',
    'area3DChart': '3D 영역 차트',
    'scatterChart': '분산형 차트',
    'radarChart': '방사형 차트',
    'bubbleChart': '거품형 차트',
    'stockChart': '주식형 차트',
    'surfaceChart': '표면 차트',
    'surface3DChart': '3D 표면 차트',
    'ofPieChart': '분리형 파이 차트',
}


__all__ = [
    'ElementType',
    'DocxElement',
    'NAMESPACES',
    'CHART_TYPE_MAP',
]
