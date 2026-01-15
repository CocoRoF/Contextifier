# service/document_processor/processor/pdf_helper.py
"""
PDF 처리 공통 헬퍼 모듈

모든 PDF 핸들러(v1, v2, v3 등)에서 공통으로 사용하는 유틸리티 함수들을 정의합니다.
"""
import logging
import os
import io
import tempfile
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from service.storage.minio_client import (
    get_minio_client,
    upload_file,
    ensure_bucket_exists,
    DEFAULT_BUCKET_NAME
)

# 공통 upload_image_to_minio 함수 import
from libs.core.functions.utils import upload_image_to_minio

logger = logging.getLogger("document-processor")


# ============================================================================
# PDF 메타데이터 추출
# ============================================================================

def extract_pdf_metadata(doc) -> Dict[str, Any]:
    """
    PDF 문서에서 메타데이터를 추출합니다.

    Args:
        doc: PyMuPDF 문서 객체

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        pdf_meta = doc.metadata
        if not pdf_meta:
            return metadata

        if pdf_meta.get('title'):
            metadata['title'] = pdf_meta['title'].strip()

        if pdf_meta.get('subject'):
            metadata['subject'] = pdf_meta['subject'].strip()

        if pdf_meta.get('author'):
            metadata['author'] = pdf_meta['author'].strip()

        if pdf_meta.get('keywords'):
            metadata['keywords'] = pdf_meta['keywords'].strip()

        if pdf_meta.get('creationDate'):
            create_time = parse_pdf_date(pdf_meta['creationDate'])
            if create_time:
                metadata['create_time'] = create_time

        if pdf_meta.get('modDate'):
            mod_time = parse_pdf_date(pdf_meta['modDate'])
            if mod_time:
                metadata['last_saved_time'] = mod_time

    except Exception as e:
        logger.debug(f"[PDF] Error extracting metadata: {e}")

    return metadata


def parse_pdf_date(date_str: str) -> Optional[datetime]:
    """
    PDF 날짜 문자열을 datetime으로 변환합니다.

    Args:
        date_str: PDF 날짜 문자열 (예: "D:20231215120000")

    Returns:
        datetime 객체 또는 None
    """
    if not date_str:
        return None

    try:
        if date_str.startswith("D:"):
            date_str = date_str[2:]

        if len(date_str) >= 14:
            return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        elif len(date_str) >= 8:
            return datetime.strptime(date_str[:8], "%Y%m%d")

    except Exception as e:
        logger.debug(f"[PDF] Error parsing date '{date_str}': {e}")

    return None


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터를 문자열로 포맷합니다.

    Args:
        metadata: 메타데이터 딕셔너리

    Returns:
        포맷된 메타데이터 문자열
    """
    if not metadata:
        return ""

    lines = ["<Document-Metadata>"]

    field_names = {
        'title': '제목',
        'subject': '주제',
        'author': '작성자',
        'keywords': '키워드',
        'create_time': '작성일',
        'last_saved_time': '마지막 수정일'
    }

    for key, label in field_names.items():
        value = metadata.get(key)
        if value:
            if isinstance(value, datetime):
                value = value.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"  {label}: {value}")

    lines.append("</Document-Metadata>\n")

    return "\n".join(lines)


# ============================================================================
# HTML 이스케이프
# ============================================================================

def escape_html(text: str) -> str:
    """
    HTML 특수문자를 이스케이프합니다.

    Args:
        text: 원본 텍스트

    Returns:
        이스케이프된 텍스트
    """
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ============================================================================
# bbox 관련 유틸리티
# ============================================================================

def calculate_overlap_ratio(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    두 bbox의 겹침 비율을 계산합니다.

    Args:
        bbox1: 첫 번째 bbox (x0, y0, x1, y1)
        bbox2: 두 번째 bbox (x0, y0, x1, y1)

    Returns:
        bbox1 기준 겹침 비율 (0.0 ~ 1.0)
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    overlap_area = (x1 - x0) * (y1 - y0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    if bbox1_area <= 0:
        return 0.0

    return overlap_area / bbox1_area


def is_inside_any_bbox(
    bbox: Tuple[float, float, float, float],
    bbox_list: List[Tuple[float, float, float, float]],
    threshold: float = 0.5
) -> bool:
    """
    bbox가 bbox_list 중 하나에 포함되는지 확인합니다.

    Args:
        bbox: 확인할 bbox
        bbox_list: bbox 목록
        threshold: 겹침 비율 임계값

    Returns:
        포함 여부
    """
    for target_bbox in bbox_list:
        overlap = calculate_overlap_ratio(bbox, target_bbox)
        if overlap > threshold:
            return True
    return False


def bboxes_overlap(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
    threshold: float = 0.5
) -> bool:
    """
    두 bbox가 겹치는지 확인합니다.

    Args:
        bbox1: 첫 번째 bbox
        bbox2: 두 번째 bbox
        threshold: 겹침 비율 임계값

    Returns:
        겹침 여부
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x1 <= x0 or y1 <= y0:
        return False

    overlap_area = (x1 - x0) * (y1 - y0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    min_area = min(bbox1_area, bbox2_area)
    if min_area <= 0:
        return False

    return overlap_area / min_area >= threshold


# ============================================================================
# 이미지 위치 찾기
# ============================================================================

def find_image_position(page, xref: int) -> Optional[Tuple[float, float, float, float]]:
    """
    이미지의 페이지 내 위치를 찾습니다.

    Args:
        page: PyMuPDF 페이지 객체
        xref: 이미지 xref

    Returns:
        bbox 또는 None
    """
    try:
        image_list = page.get_image_info(xrefs=True)

        for img_info in image_list:
            if img_info.get("xref") == xref:
                bbox = img_info.get("bbox")
                if bbox:
                    return tuple(bbox)

        return None

    except Exception as e:
        logger.debug(f"[PDF] Error finding image position: {e}")
        return None


# ============================================================================
# 텍스트 라인 추출
# ============================================================================

def get_text_lines_with_positions(page) -> List[Dict]:
    """
    페이지에서 텍스트 라인과 위치 정보를 추출합니다.

    Args:
        page: PyMuPDF 페이지 객체

    Returns:
        텍스트 라인 정보 목록
    """
    lines = []
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            line_bbox = line.get("bbox", (0, 0, 0, 0))
            text_parts = []

            for span in line.get("spans", []):
                text_parts.append(span.get("text", ""))

            full_text = "".join(text_parts).strip()
            if full_text:
                lines.append({
                    'text': full_text,
                    'y0': line_bbox[1],
                    'y1': line_bbox[3],
                    'x0': line_bbox[0],
                    'x1': line_bbox[2]
                })

    return lines
