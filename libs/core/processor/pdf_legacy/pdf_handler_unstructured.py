# your_package/document_processor/pdf_handler_unstructured.py
"""
PDF Handler (Unstructured) - pdfminer.six + Unstructured Cleaning 기반 PDF 처리기

Python 3.14에서 unstructured-inference가 지원되지 않아 partition_pdf를 직접 사용할 수 없음.
대안으로 pdfminer.six를 사용하여 텍스트 추출 후 unstructured의 cleaning 기능만 활용.

주요 기능:
- pdfminer.six를 통한 페이지별 텍스트 추출
- unstructured의 cleaning 기능 (clean, group_broken_paragraphs)
- 메타데이터 추출 (PyMuPDF 사용)
- 페이지 마커 추가
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# pdfminer.six imports
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LAParams

# unstructured cleaning imports
from unstructured.cleaners.core import clean, group_broken_paragraphs

logger = logging.getLogger("document-processor")

# PyMuPDF import (메타데이터 추출용)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available for metadata extraction")


async def extract_text_from_pdf_unstructured(
    file_path: str,
    current_config: Dict[str, Any] = None,
    app_db=None,
    extract_default_metadata: bool = True
) -> str:
    """
    pdfminer.six + Unstructured Cleaning을 사용한 PDF 텍스트 추출.

    Args:
        file_path: PDF 파일 경로
        current_config: 설정 딕셔너리 (현재 미사용)
        app_db: 데이터베이스 연결 (현재 미사용)
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (메타데이터, 페이지 마커 포함)
    """
    logger.info(f"PDF processing with pdfminer + unstructured cleaning: {file_path}")
    
    result_parts = []
    
    # 1. 메타데이터 추출 (PyMuPDF 사용) - extract_default_metadata가 True인 경우에만
    if extract_default_metadata:
        metadata_str = _extract_metadata_with_pymupdf(file_path)
        if metadata_str:
            result_parts.append(metadata_str)
    
    # 2. pdfminer.six로 페이지별 텍스트 추출
    pages_content = _extract_pages_with_pdfminer(file_path)
    
    if not pages_content:
        logger.warning(f"No content extracted from PDF: {file_path}")
        return result_parts[0] if result_parts else ""
    
    # 3. 각 페이지 텍스트에 cleaning 적용 및 페이지 마커 추가
    for page in pages_content:
        page_num = page["page_number"]
        page_text = page["text"]
        
        if not page_text.strip():
            continue
        
        # unstructured cleaning 적용
        cleaned_text = _apply_unstructured_cleaning(page_text)
        
        if cleaned_text.strip():
            # 페이지 마커 추가
            page_content = f"<페이지 번호> {page_num} </페이지 번호>\n{cleaned_text}"
            result_parts.append(page_content)
    
    result = "\n\n".join(result_parts)
    
    logger.info(f"PDF extraction complete: {len(pages_content)} pages, {len(result)} chars")
    
    return result


def _extract_metadata_with_pymupdf(file_path: str) -> str:
    """
    PyMuPDF를 사용하여 PDF 메타데이터를 추출합니다.
    
    Args:
        file_path: PDF 파일 경로
        
    Returns:
        포맷된 메타데이터 문자열
    """
    if not PYMUPDF_AVAILABLE:
        return ""
    
    try:
        doc = fitz.open(file_path)
        raw_metadata = doc.metadata
        doc.close()
        
        if not raw_metadata:
            return ""
        
        metadata = {}
        
        # 제목
        if raw_metadata.get("title"):
            metadata["title"] = raw_metadata["title"]
        
        # 작성자
        if raw_metadata.get("author"):
            metadata["author"] = raw_metadata["author"]
        
        # 주제
        if raw_metadata.get("subject"):
            metadata["subject"] = raw_metadata["subject"]
        
        # 키워드
        if raw_metadata.get("keywords"):
            metadata["keywords"] = raw_metadata["keywords"]
        
        # 작성일
        if raw_metadata.get("creationDate"):
            create_time = _parse_pdf_date(raw_metadata["creationDate"])
            if create_time:
                metadata["create_time"] = create_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 수정일
        if raw_metadata.get("modDate"):
            mod_time = _parse_pdf_date(raw_metadata["modDate"])
            if mod_time:
                metadata["last_saved_time"] = mod_time.strftime("%Y-%m-%d %H:%M:%S")
        
        return _format_metadata(metadata)
        
    except Exception as e:
        logger.warning(f"Failed to extract PDF metadata: {e}")
        return ""


def _parse_pdf_date(date_str: str) -> Optional[datetime]:
    """
    PDF 날짜 문자열을 datetime 객체로 변환합니다.
    
    PDF 날짜 형식: D:YYYYMMDDHHmmSSOHH'mm'
    """
    if not date_str:
        return None
    
    try:
        # D: 접두사 제거
        if date_str.startswith("D:"):
            date_str = date_str[2:]
        
        # 타임존 정보 제거
        if "+" in date_str:
            date_str = date_str.split("+")[0]
        elif "-" in date_str and len(date_str) > 14:
            date_str = date_str[:14]
        
        # 기본 형식으로 파싱
        if len(date_str) >= 14:
            return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        elif len(date_str) >= 8:
            return datetime.strptime(date_str[:8], "%Y%m%d")
            
    except Exception:
        pass
    
    return None


def _format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터를 사람이 읽기 쉬운 형식의 문자열로 포맷합니다.
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
            lines.append(f"  {label}: {value}")
    
    lines.append("</Document-Metadata>\n")
    
    return "\n".join(lines)


def _extract_pages_with_pdfminer(file_path: str) -> List[Dict[str, Any]]:
    """
    pdfminer.six를 사용하여 PDF에서 페이지별 텍스트를 추출합니다.
    
    Args:
        file_path: PDF 파일 경로
        
    Returns:
        [{"page_number": int, "text": str}, ...]
    """
    pages_content = []
    
    try:
        # LAParams 설정 (레이아웃 분석 파라미터)
        laparams = LAParams(
            line_margin=0.5,      # 줄 간격 임계값
            word_margin=0.1,      # 단어 간격 임계값
            char_margin=2.0,      # 문자 간격 임계값
            boxes_flow=0.5,       # 텍스트 박스 흐름 방향
            detect_vertical=True  # 세로 텍스트 감지
        )
        
        for page_num, page_layout in enumerate(extract_pages(file_path, laparams=laparams), start=1):
            page_text = ""
            
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text += element.get_text()
            
            pages_content.append({
                "page_number": page_num,
                "text": page_text.strip()
            })
        
        logger.debug(f"Extracted {len(pages_content)} pages from PDF")
        
    except Exception as e:
        logger.error(f"Failed to extract pages with pdfminer: {e}")
    
    return pages_content


def _apply_unstructured_cleaning(text: str) -> str:
    """
    unstructured의 cleaning 기능을 적용합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        정제된 텍스트
    """
    if not text or not text.strip():
        return ""
    
    try:
        # 1. 기본 cleaning (여분의 공백, 대시, 불릿 정리)
        cleaned = clean(
            text,
            extra_whitespace=True,
            dashes=True,
            bullets=True,
            trailing_punctuation=False  # 문장 끝 구두점은 유지
        )
        
        # 2. 끊어진 문단 그룹화
        cleaned = group_broken_paragraphs(cleaned)
        
        return cleaned
        
    except Exception as e:
        logger.warning(f"Unstructured cleaning failed, returning original text: {e}")
        return text
