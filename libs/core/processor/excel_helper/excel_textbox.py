"""
XLSX 텍스트박스 추출 모듈

XLSX 파일의 DrawingML에서 텍스트박스 내용을 추출합니다.
텍스트박스는 xl/drawings/drawing*.xml에 <xdr:sp> 요소로 저장됩니다.
"""

import os
import zipfile
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# DrawingML 네임스페이스
NAMESPACES = {
    'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'pkg': 'http://schemas.openxmlformats.org/package/2006/relationships',
    'ss': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
}

# 네임스페이스 URI 상수
NS_XDR = '{http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing}'
NS_A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
NS_R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
NS_PKG = '{http://schemas.openxmlformats.org/package/2006/relationships}'
NS_SS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def extract_textboxes_from_xlsx(file_path: str) -> Dict[str, List[str]]:
    """
    XLSX 파일에서 텍스트박스를 추출합니다.

    XLSX의 텍스트박스는 xl/drawings/drawing*.xml 파일에 저장됩니다.
    DrawingML 형식으로 <xdr:sp> (shape) 요소 내 <xdr:txBody>에 텍스트가 포함됩니다.

    Args:
        file_path: XLSX 파일 경로

    Returns:
        {시트명: [텍스트박스 내용 리스트]} 형태의 딕셔너리
    """
    textboxes_by_sheet: Dict[str, List[str]] = {}

    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # 시트와 drawing 관계 매핑 구축
            sheet_drawing_map = _get_sheet_drawing_mapping(zf)
            logger.debug(f"Sheet-Drawing mapping: {sheet_drawing_map}")

            # 모든 drawing 파일 처리
            for name in zf.namelist():
                if name.startswith('xl/drawings/drawing') and name.endswith('.xml'):
                    try:
                        drawing_xml = zf.read(name)
                        textboxes = _parse_drawing_textboxes(drawing_xml)

                        if textboxes:
                            # drawing 파일에 해당하는 시트 찾기
                            drawing_name = os.path.basename(name)
                            sheet_name = sheet_drawing_map.get(drawing_name, f"Sheet ({drawing_name})")

                            if sheet_name not in textboxes_by_sheet:
                                textboxes_by_sheet[sheet_name] = []
                            textboxes_by_sheet[sheet_name].extend(textboxes)

                            logger.info(f"Extracted {len(textboxes)} textboxes from {name} -> {sheet_name}")

                    except Exception as e:
                        logger.warning(f"Error parsing textboxes from {name}: {e}")

        total_textboxes = sum(len(tb) for tb in textboxes_by_sheet.values())
        if total_textboxes > 0:
            logger.info(f"Total extracted {total_textboxes} textboxes from XLSX")

    except Exception as e:
        logger.warning(f"Error extracting textboxes from XLSX: {e}")

    return textboxes_by_sheet


def _get_sheet_drawing_mapping(zf: zipfile.ZipFile) -> Dict[str, str]:
    """
    XLSX 내부 관계를 파싱하여 drawing 파일과 시트 이름의 매핑을 구축합니다.

    Args:
        zf: ZipFile 객체

    Returns:
        {drawing 파일명: 시트명} 매핑
    """
    drawing_to_sheet: Dict[str, str] = {}
    sheet_rid_map: Dict[str, str] = {}  # rId -> sheet_name
    rid_to_sheet_file: Dict[str, str] = {}  # rId -> sheet파일경로

    try:
        # 1. workbook.xml에서 시트 정보 추출 (rId -> sheet_name)
        if 'xl/workbook.xml' in zf.namelist():
            workbook_xml = zf.read('xl/workbook.xml')
            wb_root = ET.fromstring(workbook_xml)

            for sheet_elem in wb_root.findall(f'.//{NS_SS}sheet'):
                sheet_name = sheet_elem.get('name', '')
                r_id = sheet_elem.get(f'{NS_R}id', '')
                if sheet_name and r_id:
                    sheet_rid_map[r_id] = sheet_name

        # 2. workbook.xml.rels에서 rId -> sheet*.xml 매핑
        if 'xl/_rels/workbook.xml.rels' in zf.namelist():
            rels_xml = zf.read('xl/_rels/workbook.xml.rels')
            rels_root = ET.fromstring(rels_xml)

            for rel_elem in rels_root.findall(f'.//{NS_PKG}Relationship'):
                r_id = rel_elem.get('Id', '')
                target = rel_elem.get('Target', '')
                if 'worksheets/sheet' in target:
                    rid_to_sheet_file[r_id] = target

        # 3. sheet파일 -> sheet_name 매핑
        sheet_file_to_name: Dict[str, str] = {}
        for r_id, sheet_name in sheet_rid_map.items():
            if r_id in rid_to_sheet_file:
                sheet_file = rid_to_sheet_file[r_id]
                # worksheets/sheet1.xml -> sheet1.xml
                sheet_file_base = os.path.basename(sheet_file)
                sheet_file_to_name[sheet_file_base] = sheet_name

        # 4. 각 sheet*.xml.rels에서 drawing 관계 찾기
        for name in zf.namelist():
            if name.startswith('xl/worksheets/_rels/sheet') and name.endswith('.xml.rels'):
                try:
                    rels_xml = zf.read(name)
                    rels_root = ET.fromstring(rels_xml)

                    # sheet*.xml.rels -> sheet*.xml
                    sheet_file = os.path.basename(name).replace('.rels', '')
                    sheet_name = sheet_file_to_name.get(sheet_file, sheet_file)

                    for rel_elem in rels_root.findall(f'.//{NS_PKG}Relationship'):
                        target = rel_elem.get('Target', '')
                        if 'drawings/drawing' in target:
                            # ../drawings/drawing1.xml -> drawing1.xml
                            drawing_file = os.path.basename(target)
                            drawing_to_sheet[drawing_file] = sheet_name
                            logger.debug(f"Mapped {drawing_file} -> {sheet_name}")

                except Exception as e:
                    logger.debug(f"Error parsing sheet rels {name}: {e}")

    except Exception as e:
        logger.debug(f"Error building sheet-drawing mapping: {e}")

    return drawing_to_sheet


def _parse_drawing_textboxes(drawing_xml: bytes) -> List[str]:
    """
    DrawingML XML에서 텍스트박스 내용을 추출합니다.

    Args:
        drawing_xml: drawing XML 바이트

    Returns:
        텍스트박스 내용 리스트
    """
    textboxes: List[str] = []

    try:
        # XML 파싱
        try:
            root = ET.fromstring(drawing_xml)
        except ET.ParseError:
            # BOM 제거 후 재시도
            drawing_str = drawing_xml.decode('utf-8-sig', errors='ignore')
            root = ET.fromstring(drawing_str)

        # 모든 shape 요소 직접 찾기 (<xdr:sp>)
        # 전체 문서에서 모든 sp 요소 탐색
        sp_elems = root.findall(f'.//{NS_XDR}sp')
        logger.debug(f"Found {len(sp_elems)} shape elements in drawing")

        for sp in sp_elems:
            textbox_content = _extract_textbox_content(sp)
            if textbox_content:
                textboxes.append(textbox_content)
                logger.debug(f"Extracted textbox: {textbox_content[:50]}...")

    except Exception as e:
        logger.warning(f"Error parsing drawing textboxes: {e}")

    return textboxes


def _extract_textbox_content(sp_elem) -> Optional[str]:
    """
    Shape 요소에서 텍스트박스 내용을 추출합니다.

    XLSX의 텍스트박스 구조:
    <xdr:sp>
        <xdr:nvSpPr>...</xdr:nvSpPr>
        <xdr:spPr>...</xdr:spPr>
        <xdr:txBody>  <-- 직접 자식! (.//가 아님)
            <a:p>
                <a:r>
                    <a:t>텍스트</a:t>
                </a:r>
            </a:p>
        </xdr:txBody>
    </xdr:sp>

    Args:
        sp_elem: shape XML 요소

    Returns:
        텍스트박스 내용 (없으면 None)
    """
    try:
        # txBody 요소 찾기 - xdr 네임스페이스의 직접 자식으로 찾기
        txBody = sp_elem.find(f'{NS_XDR}txBody')

        if txBody is None:
            return None

        # 모든 텍스트 추출
        text_parts: List[str] = []

        # 각 paragraph (a:p) 처리
        paragraphs = txBody.findall(f'.//{NS_A}p')

        for p_elem in paragraphs:
            para_texts: List[str] = []

            # 각 run (a:r) 내의 텍스트 (a:t) 찾기
            runs = p_elem.findall(f'.//{NS_A}r')

            for r_elem in runs:
                # a:t는 a:r의 직접 자식
                t_elem = r_elem.find(f'{NS_A}t')

                if t_elem is not None and t_elem.text:
                    para_texts.append(t_elem.text)

            # run 없이 직접 a:t가 있는 경우도 처리
            if not para_texts:
                t_elems = p_elem.findall(f'.//{NS_A}t')
                for t_elem in t_elems:
                    if t_elem is not None and t_elem.text:
                        para_texts.append(t_elem.text)

            if para_texts:
                text_parts.append(''.join(para_texts))

        if text_parts:
            # 줄바꿈으로 문단 구분
            full_text = '\n'.join(text_parts).strip()
            if full_text:
                return full_text

        return None

    except Exception as e:
        logger.debug(f"Error extracting textbox content: {e}")
        return None
