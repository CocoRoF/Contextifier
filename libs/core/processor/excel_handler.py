# your_package/document_processor/excel_handler.py
"""
Excel Handler - Excel 문서 처리기 (XLSX/XLS)

주요 기능:
- 메타데이터 추출 (제목, 작성자, 주제, 키워드, 작성일, 수정일 등)
- 텍스트 추출 (openpyxl/xlrd를 통한 직접 파싱)
- 테이블 추출 (병합셀 유무에 따라 Markdown 또는 HTML 변환)
- 인라인 이미지 추출 및 MinIO 업로드
- 차트 처리 (1순위: 테이블로 변환, 2순위: matplotlib 이미지)
- 다중 시트 지원

세부 로직은 excel_helper 모듈에서 분리 관리됩니다.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Set

from libs.core.functions.utils import upload_image_to_minio

if TYPE_CHECKING:
    from openpyxl.workbook import Workbook
    from openpyxl.worksheet.worksheet import Worksheet
from libs.core.processor.excel_helper import (
    # Textbox
    extract_textboxes_from_xlsx,
    # Metadata
    extract_xlsx_metadata,
    extract_xls_metadata,
    format_metadata,
    # Chart
    extract_charts_from_xlsx,
    process_chart,
    extract_chart_info_basic,
    # Image
    extract_images_from_xlsx,
    get_sheet_images,
    # Table
    convert_xlsx_sheet_to_table,
    convert_xls_sheet_to_table,
    # Object Detection (개별 테이블 청킹)
    convert_xlsx_objects_to_tables,
    convert_xls_objects_to_tables,
)

logger = logging.getLogger("document-processor")

# openpyxl (xlsx)
try:
    from openpyxl import load_workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# xlrd (xls)
try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False


# === 메인 함수 ===

async def extract_text_from_excel(
    file_path: str,
    current_config: Dict[str, Any] = None,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    Excel 파일에서 텍스트를 추출합니다.

    Args:
        file_path: Excel 파일 경로
        current_config: 설정 딕셔너리
        app_db: 데이터베이스 연결 (이미지 메타데이터 저장용)
        extract_default_metadata: 기본 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (테이블 HTML, 이미지 태그, 차트 정보 포함)
    """
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"Excel processing: {file_path}, ext: {ext}")

    if ext == '.xlsx':
        if not OPENPYXL_AVAILABLE:
            raise ImportError("openpyxl이 설치되어야 .xlsx 파일을 처리할 수 있습니다.")
        return await _extract_xlsx(file_path, app_db, extract_default_metadata)

    elif ext == '.xls':
        if not XLRD_AVAILABLE:
            raise ImportError("xlrd가 설치되어야 .xls 파일을 처리할 수 있습니다.")
        return await _extract_xls(file_path, app_db, extract_default_metadata)

    else:
        raise ValueError(f"지원하지 않는 Excel 형식입니다: {ext}")


# === XLSX 처리 ===

async def _extract_xlsx(
    file_path: str,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """XLSX 파일 처리 메인 함수."""
    logger.info(f"XLSX processing: {file_path}")

    try:
        wb = load_workbook(file_path, data_only=True)

        # 전처리: 파일 레벨 데이터 추출
        preload = _preload_xlsx_data(file_path, wb, extract_default_metadata)

        # 시트별 처리
        result_parts = [preload["metadata_str"]] if preload["metadata_str"] else []
        processed_images: Set[str] = set()
        stats = {"charts": 0, "images": 0, "textboxes": 0}

        for sheet_name in wb.sheetnames:
            sheet_result = _process_xlsx_sheet(
                wb[sheet_name],
                sheet_name,
                preload,
                app_db,
                processed_images,
                stats
            )
            result_parts.append(sheet_result)

        # 남은 차트 처리 (시트에 연결되지 않은 차트)
        remaining = _process_remaining_charts(
            preload["charts"],
            preload["chart_idx"],
            app_db,
            processed_images,
            stats
        )
        if remaining:
            result_parts.append(remaining)

        result = "".join(result_parts)
        logger.info(
            f"XLSX processing completed: {len(wb.sheetnames)} sheets, "
            f"{stats['charts']} charts, {stats['images']} images, {stats['textboxes']} textboxes"
        )
        return result

    except Exception as e:
        logger.error(f"Error in XLSX processing: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


def _preload_xlsx_data(
    file_path: str,
    wb: Workbook,
    extract_default_metadata: bool
) -> Dict[str, Any]:
    """XLSX 파일에서 전처리 데이터를 추출합니다."""
    result = {
        "metadata_str": "",
        "charts": [],
        "images_data": [],
        "textboxes_by_sheet": {},
        "chart_idx": 0,
    }

    # 메타데이터
    if extract_default_metadata:
        metadata = extract_xlsx_metadata(wb)
        result["metadata_str"] = format_metadata(metadata)
        if result["metadata_str"]:
            result["metadata_str"] += "\n\n"
            logger.info(f"XLSX metadata extracted: {list(metadata.keys())}")

    # 차트 (OOXML 파싱)
    result["charts"] = extract_charts_from_xlsx(file_path)

    # 이미지
    result["images_data"] = extract_images_from_xlsx(file_path)
    logger.info(f"Found {len(result['images_data'])} images in XLSX")

    # 텍스트박스
    result["textboxes_by_sheet"] = extract_textboxes_from_xlsx(file_path)
    total_textboxes = sum(len(tb) for tb in result["textboxes_by_sheet"].values())
    logger.info(f"Found {total_textboxes} textboxes in XLSX")

    return result


def _process_xlsx_sheet(
    ws: Worksheet,
    sheet_name: str,
    preload: Dict[str, Any],
    app_db,
    processed_images: Set[str],
    stats: Dict[str, int]
) -> str:
    """
    XLSX 시트 하나를 처리합니다.
    테두리 기반 개별 객체 감지 후 인접 객체 병합 후 청킹합니다.
    """
    parts = [f"\n=== 시트: {sheet_name} ===\n"]

    # 1. 테이블 추출 (개별 객체별 청킹)
    table_contents = convert_xlsx_objects_to_tables(ws)
    if table_contents:
        for i, table_content in enumerate(table_contents, 1):
            if len(table_contents) > 1:
                parts.append(f"\n[테이블 {i}]\n{table_content}\n")
            else:
                parts.append(f"\n{table_content}\n")
        logger.debug(f"Sheet '{sheet_name}': {len(table_contents)} table objects extracted")

    # 2. 차트 처리
    chart_result = _process_sheet_charts(
        ws, preload, app_db, processed_images, stats
    )
    if chart_result:
        parts.append(chart_result)

    # 3. 이미지 처리
    image_result = _process_sheet_images(
        ws, preload, app_db, processed_images, stats
    )
    if image_result:
        parts.append(image_result)

    # 4. 텍스트박스 처리
    textbox_result = _process_sheet_textboxes(sheet_name, preload, stats)
    if textbox_result:
        parts.append(textbox_result)

    return "".join(parts)


def _process_sheet_charts(
    ws: Worksheet,
    preload: Dict[str, Any],
    app_db,
    processed_images: Set[str],
    stats: Dict[str, int]
) -> str:
    """시트의 차트를 처리합니다."""
    if not hasattr(ws, '_charts') or not ws._charts:
        return ""

    parts = []
    charts = preload["charts"]

    for chart in ws._charts:
        if preload["chart_idx"] < len(charts):
            chart_text = process_chart(
                charts[preload["chart_idx"]],
                app_db,
                processed_images,
                upload_func=upload_image_to_minio
            )
            preload["chart_idx"] += 1
        else:
            chart_text = extract_chart_info_basic(chart, ws)

        if chart_text:
            parts.append(f"\n{chart_text}\n")
            stats["charts"] += 1

    return "".join(parts)


def _process_sheet_images(
    ws: Worksheet,
    preload: Dict[str, Any],
    app_db,
    processed_images: Set[str],
    stats: Dict[str, int]
) -> str:
    """시트의 이미지를 처리합니다."""
    parts = []
    sheet_images = get_sheet_images(ws, preload["images_data"], "")

    for img_data, img_anchor in sheet_images:
        minio_path = upload_image_to_minio(
            img_data, app_db=app_db, processed_images=processed_images
        )
        if minio_path:
            parts.append(f"\n[image:{minio_path}]\n")
            stats["images"] += 1

    return "".join(parts)


def _process_sheet_textboxes(
    sheet_name: str,
    preload: Dict[str, Any],
    stats: Dict[str, int]
) -> str:
    """시트의 텍스트박스를 처리합니다."""
    textboxes = preload["textboxes_by_sheet"].get(sheet_name, [])
    if not textboxes:
        return ""

    parts = []
    for textbox_content in textboxes:
        parts.append(f"\n[textbox]\n{textbox_content}\n[/textbox]\n")
        stats["textboxes"] += 1

    return "".join(parts)


def _process_remaining_charts(
    charts: List[Dict],
    chart_idx: int,
    app_db,
    processed_images: Set[str],
    stats: Dict[str, int]
) -> str:
    """시트에 연결되지 않은 나머지 차트를 처리합니다."""
    parts = []

    while chart_idx < len(charts):
        chart_text = process_chart(
            charts[chart_idx],
            app_db,
            processed_images,
            upload_func=upload_image_to_minio
        )
        if chart_text:
            parts.append(f"\n{chart_text}\n")
            stats["charts"] += 1
        chart_idx += 1

    return "".join(parts)


# === XLS 처리 ===

async def _extract_xls(
    file_path: str,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """XLS 파일 처리 메인 함수."""
    logger.info(f"XLS processing: {file_path}")

    try:
        wb = xlrd.open_workbook(file_path, formatting_info=True)
        result_parts = []

        # 메타데이터 추출
        if extract_default_metadata:
            metadata = extract_xls_metadata(wb)
            metadata_str = format_metadata(metadata)
            if metadata_str:
                result_parts.append(f"{metadata_str}\n\n")
                logger.info(f"XLS metadata extracted: {list(metadata.keys())}")

        # 시트별 처리
        for sheet_idx in range(wb.nsheets):
            sheet = wb.sheet_by_index(sheet_idx)
            sheet_result = _process_xls_sheet(sheet, wb)
            result_parts.append(sheet_result)

        result = "".join(result_parts)
        logger.info(f"XLS processing completed: {wb.nsheets} sheets")
        return result

    except Exception as e:
        logger.error(f"Error in XLS processing: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


def _process_xls_sheet(sheet, wb) -> str:
    """
    XLS 시트 하나를 처리합니다.
    테두리 기반 개별 객체 감지 후 인접 객체 병합 후 청킹합니다.
    """
    parts = [f"\n=== 시트: {sheet.name} ===\n"]

    # 개별 객체별 테이블 추출
    table_contents = convert_xls_objects_to_tables(sheet, wb)
    if table_contents:
        for i, table_content in enumerate(table_contents, 1):
            if len(table_contents) > 1:
                parts.append(f"\n[테이블 {i}]\n{table_content}\n")
            else:
                parts.append(f"\n{table_content}\n")
        logger.debug(f"Sheet '{sheet.name}': {len(table_contents)} table objects extracted")

    return "".join(parts)
