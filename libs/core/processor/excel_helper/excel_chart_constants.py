"""
Excel 차트 상수 모듈

OOXML 차트 타입 맵핑 및 관련 상수 정의
"""

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

# OOXML 네임스페이스
CHART_NAMESPACES = {
    'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
}
