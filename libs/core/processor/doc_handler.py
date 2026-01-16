# service/document_processor/processor/doc_handler.py
"""
DOC Handler - 구형 Microsoft Word 문서 처리기

주요 기능:
- RTF (Rich Text Format) 파일 처리 (striprtf 사용)
- 구형 OLE Compound Document (.doc) 처리 (olefile + LibreOffice 변환)
- HTML로 저장된 DOC 파일 처리 (BeautifulSoup 사용)
- 메타데이터 추출 (제목, 작성자, 주제, 키워드, 작성일, 수정일 등)
- 텍스트 추출 및 정리
- 테이블 추출 (HTML 형식 보존)
- 인라인 이미지 추출 및 로컬 저장
- LibreOffice 변환 폴백 (모든 형식 지원)

구형 DOC 파일 유형:
1. RTF (Rich Text Format) - .doc 확장자로 저장되는 경우가 많음
2. OLE Compound Document - 진짜 구형 Microsoft Word 바이너리 형식
3. HTML - Word가 웹 페이지로 저장할 때 .doc 확장자 사용
4. DOCX (ZIP) - 잘못된 확장자로 저장된 경우

이 핸들러는 파일의 실제 형식을 자동 감지하여 적절한 방법으로 처리합니다.
"""
import logging
import os
import re
import shutil
import tempfile
import subprocess
import struct
import traceback
import base64
from datetime import datetime
from typing import Any, Dict, List, Set
from enum import Enum
import zipfile

import olefile
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text

# 커스텀 RTF 파서 (바이너리 직접 분석)
from libs.core.processor.doc_helpers.rtf_parser import parse_rtf, RTFDocument
from libs.core.functions.img_processor import ImageProcessor

# 모듈 레벨 이미지 프로세서
_image_processor = ImageProcessor(
    directory_path="temp/images",
    tag_prefix="[image:",
    tag_suffix="]"
)

logger = logging.getLogger("document-processor")


# === 문서 형식 타입 정의 ===

class DocFormat(Enum):
    """DOC 파일의 실제 형식"""
    RTF = "rtf"                 # Rich Text Format
    OLE = "ole"                 # OLE Compound Document (진짜 구형 DOC)
    HTML = "html"               # HTML로 저장된 DOC
    DOCX = "docx"               # 잘못된 확장자의 DOCX (ZIP)
    UNKNOWN = "unknown"         # 알 수 없는 형식


# === 파일 매직 넘버 ===

MAGIC_NUMBERS = {
    'RTF': b'{\\rtf',
    'OLE': b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',
    'ZIP': b'PK\x03\x04',
}


# === 메타데이터 필드 매핑 ===

METADATA_FIELD_NAMES = {
    'title': '제목',
    'subject': '주제',
    'author': '작성자',
    'keywords': '키워드',
    'comments': '설명',
    'last_saved_by': '마지막 저장자',
    'create_time': '작성일',
    'last_saved_time': '수정일',
}


# === 파일 형식 감지 ===

def _detect_doc_format(file_path: str) -> DocFormat:
    """
    DOC 파일의 실제 형식을 감지합니다.

    파일의 매직 넘버(시그니처)를 확인하여 실제 형식을 판별합니다.

    Args:
        file_path: DOC 파일 경로

    Returns:
        감지된 파일 형식
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(32)

        if not header:
            return DocFormat.UNKNOWN

        # RTF 확인 - {\\rtf로 시작
        if header.startswith(MAGIC_NUMBERS['RTF']):
            logger.info(f"Detected RTF format: {file_path}")
            return DocFormat.RTF

        # OLE Compound Document 확인
        if header.startswith(MAGIC_NUMBERS['OLE']):
            logger.info(f"Detected OLE Compound Document format: {file_path}")
            return DocFormat.OLE

        # ZIP (DOCX) 확인
        if header.startswith(MAGIC_NUMBERS['ZIP']):
            # DOCX인지 확인
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    if '[Content_Types].xml' in zf.namelist():
                        logger.info(f"Detected DOCX (ZIP) format with .doc extension: {file_path}")
                        return DocFormat.DOCX
            except zipfile.BadZipFile:
                pass

        # HTML 확인
        header_lower = header.lower()
        if (header_lower.startswith(b'<!doctype') or
            header_lower.startswith(b'<html') or
            b'<html' in header_lower[:100]):
            logger.info(f"Detected HTML format: {file_path}")
            return DocFormat.HTML

        # 텍스트 기반 확인 (BOM 포함 가능)
        try:
            # UTF-8 BOM 확인
            if header.startswith(b'\xef\xbb\xbf'):
                text_header = header[3:].decode('utf-8', errors='ignore').lower()
            else:
                text_header = header.decode('utf-8', errors='ignore').lower()

            if text_header.startswith('{\\rtf'):
                logger.info(f"Detected RTF format (text check): {file_path}")
                return DocFormat.RTF
            if text_header.startswith('<!doctype') or text_header.startswith('<html'):
                logger.info(f"Detected HTML format (text check): {file_path}")
                return DocFormat.HTML
        except:
            pass

        logger.warning(f"Unknown DOC format, will try LibreOffice conversion: {file_path}")
        return DocFormat.UNKNOWN

    except Exception as e:
        logger.error(f"Error detecting DOC format: {e}")
        return DocFormat.UNKNOWN


# === 메타데이터 포맷팅 ===

def _format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터 딕셔너리를 읽기 쉬운 문자열로 변환합니다.

    Args:
        metadata: 메타데이터 딕셔너리

    Returns:
        포맷된 메타데이터 문자열
    """
    if not metadata:
        return ""

    lines = ["<Document-Metadata>"]

    for key, label in METADATA_FIELD_NAMES.items():
        if key in metadata and metadata[key]:
            value = metadata[key]

            # datetime 객체 포맷팅
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')

            lines.append(f"  {label}: {value}")

    lines.append("</Document-Metadata>")

    return "\n".join(lines)


# === RTF 처리 ===

def _extract_rtf_metadata(content: str) -> Dict[str, Any]:
    """
    RTF 콘텐츠에서 메타데이터를 추출합니다.

    RTF 메타데이터는 \\info 그룹 내에 저장됩니다:
    - \\title: 제목
    - \\subject: 주제
    - \\author: 작성자
    - \\keywords: 키워드
    - \\doccomm: 설명
    - \\operator: 마지막 저장자
    - \\creatim: 작성일
    - \\revtim: 수정일

    Args:
        content: RTF 파일 콘텐츠 (문자열)

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        # RTF 메타데이터 패턴
        patterns = {
            'title': r'\\title\s*\{([^}]*)\}',
            'subject': r'\\subject\s*\{([^}]*)\}',
            'author': r'\\author\s*\{([^}]*)\}',
            'keywords': r'\\keywords\s*\{([^}]*)\}',
            'comments': r'\\doccomm\s*\{([^}]*)\}',
            'last_saved_by': r'\\operator\s*\{([^}]*)\}',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value:
                    metadata[key] = value

        # 날짜 추출 (\\creatim, \\revtim)
        date_patterns = {
            'create_time': r'\\creatim\\yr(\d+)\\mo(\d+)\\dy(\d+)(?:\\hr(\d+)\\min(\d+))?',
            'last_saved_time': r'\\revtim\\yr(\d+)\\mo(\d+)\\dy(\d+)(?:\\hr(\d+)\\min(\d+))?',
        }

        for key, pattern in date_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
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

        logger.debug(f"Extracted RTF metadata: {list(metadata.keys())}")

    except Exception as e:
        logger.warning(f"Failed to extract RTF metadata: {e}")

    return metadata


def _clean_rtf_text(text: str) -> str:
    """
    RTF 제어 문자를 제거하고 순수 텍스트만 추출합니다.

    Args:
        text: RTF 형식의 텍스트

    Returns:
        정리된 텍스트
    """
    try:
        # 제어 시퀀스 제거
        # \\controlword, \\', \\'XX (hex escape) 등
        text = re.sub(r'\\[a-z]+\d*\s?', '', text)
        text = re.sub(r"\\'[0-9a-fA-F]{2}", '', text)
        text = re.sub(r'\\[{}\\]', '', text)
        text = re.sub(r'[{}]', '', text)

        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    except:
        return text


def _extract_text_from_rtf(
    file_path: str,
    extract_default_metadata: bool = True
) -> str:
    """
    RTF 파일에서 텍스트를 추출합니다.

    커스텀 RTF 파서를 사용하여 바이너리를 직접 분석합니다.
    LibreOffice 없이 순수 Python으로 처리합니다.

    v3 개선사항:
    - \\bin 태그의 바이너리 데이터 처리 (이미지 로컬 저장)
    - 테이블이 원래 위치에 인라인으로 배치됨
    - 병합 셀 처리 (colspan/rowspan)
    - 내용 유실 방지
    - 유니코드 음수 이스케이프 완벽 처리

    Args:
        file_path: RTF 파일 경로
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트
    """
    logger.info(f"Processing RTF file with native parser v3: {file_path}")

    try:
        # 파일 읽기
        with open(file_path, 'rb') as f:
            content = f.read()

        # 중복 이미지 방지용 집합
        processed_images: Set[str] = set()

        # RTF 파서로 처리 (v3: 바이너리 이미지 자동 처리)
        doc = parse_rtf(content, processed_images=processed_images)

        result_parts = []

        # 메타데이터 추출 (extract_default_metadata가 True인 경우에만)
        if extract_default_metadata:
            metadata_str = _format_metadata(doc.metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")
                logger.info(f"RTF metadata extracted: {list(doc.metadata.keys())}")

        # 페이지 1 시작
        result_parts.append("<페이지 번호> 1 </페이지 번호>\n")

        # v3: 인라인 콘텐츠 사용 (테이블이 원래 위치에 배치됨)
        inline_content = doc.get_inline_content()
        if inline_content:
            result_parts.append(inline_content)

            # 테이블 개수 계산 (로깅용)
            real_tables = sum(1 for t in doc.tables if t.is_real_table())
            text_lists = sum(1 for t in doc.tables if not t.is_real_table())

            logger.info(f"RTF v3: Inline content with {real_tables} tables, {text_lists} text lists")
        else:
            # 폴백: 기존 방식으로 처리
            if doc.text_content:
                result_parts.append(doc.text_content)

            # 테이블 처리: 2열 이상은 HTML, 1열은 텍스트 리스트
            real_tables = []
            text_lists = []

            for table in doc.tables:
                if not table.rows:
                    continue

                if table.is_real_table():
                    # 2열 이상: HTML 테이블
                    real_tables.append(table.to_html())
                else:
                    # 1열: 텍스트 리스트
                    text_lists.append(table.to_text_list())

            # 1열 테이블 (텍스트 리스트) 추가
            if text_lists:
                for text_list in text_lists:
                    if text_list:
                        result_parts.append("\n" + text_list + "\n")

            # 실제 테이블 (2열 이상) HTML 추가
            if real_tables:
                result_parts.append("\n\n<!-- Tables -->\n")
                for html in real_tables:
                    result_parts.append("\n" + html + "\n")
                logger.info(f"Extracted {len(real_tables)} real tables from RTF (legacy)")

        result = "\n".join(result_parts)

        # 잘못된 이미지 태그 제거 (해시가 없는 경로)
        # [image:documents/uploads/.png] 같은 패턴 제거
        result = re.sub(r'\[image:[^\]]*uploads/\.[^\]]*\]', '', result)

        logger.info(f"RTF processing completed: {len(result)} chars")

        return result

    except Exception as e:
        logger.error(f"Error processing RTF file: {e}")
        logger.debug(traceback.format_exc())

        # 폴백: striprtf 라이브러리 사용
        try:
            return _extract_text_from_rtf_fallback(file_path)
        except Exception as e2:
            logger.error(f"striprtf fallback also failed: {e2}")

        # 최종 폴백: LibreOffice 변환
        return _convert_with_libreoffice(file_path)


def _extract_text_from_rtf_fallback(
    file_path: str,
    extract_default_metadata: bool = True
) -> str:
    """
    RTF 파일에서 텍스트를 추출합니다 (폴백: striprtf 사용).

    Args:
        file_path: RTF 파일 경로
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트
    """
    logger.info(f"Processing RTF file with striprtf fallback: {file_path}")

    # 파일 읽기 (여러 인코딩 시도)
    content = None
    encodings = ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
        content = raw_content.decode('cp1252', errors='replace')

    result_parts = []
    processed_images: Set[str] = set()

    # 메타데이터 추출 (extract_default_metadata가 True인 경우에만)
    if extract_default_metadata:
        metadata = _extract_rtf_metadata(content)
        metadata_str = _format_metadata(metadata)
        if metadata_str:
            result_parts.append(metadata_str + "\n\n")

    # 페이지 1 시작
    result_parts.append("<페이지 번호> 1 </페이지 번호>\n")

    # 텍스트 추출
    try:
        text = rtf_to_text(content)
    except Exception as e:
        logger.warning(f"striprtf failed: {e}")
        text = _clean_rtf_text(content)

    if text:
        text = re.sub(r'\n{3,}', '\n\n', text)
        result_parts.append(text.strip())

    return "\n".join(result_parts)


# === OLE Compound Document 처리 ===

def _extract_ole_metadata(ole: olefile.OleFileIO) -> Dict[str, Any]:
    """
    OLE Compound Document에서 메타데이터를 추출합니다.

    Args:
        ole: olefile.OleFileIO 객체

    Returns:
        메타데이터 딕셔너리
    """
    metadata = {}

    try:
        # olefile의 get_metadata() 사용
        ole_meta = ole.get_metadata()

        if ole_meta:
            if ole_meta.title:
                metadata['title'] = _decode_ole_string(ole_meta.title)
            if ole_meta.subject:
                metadata['subject'] = _decode_ole_string(ole_meta.subject)
            if ole_meta.author:
                metadata['author'] = _decode_ole_string(ole_meta.author)
            if ole_meta.keywords:
                metadata['keywords'] = _decode_ole_string(ole_meta.keywords)
            if ole_meta.comments:
                metadata['comments'] = _decode_ole_string(ole_meta.comments)
            if ole_meta.last_saved_by:
                metadata['last_saved_by'] = _decode_ole_string(ole_meta.last_saved_by)
            if ole_meta.create_time:
                metadata['create_time'] = ole_meta.create_time
            if ole_meta.last_saved_time:
                metadata['last_saved_time'] = ole_meta.last_saved_time

        logger.debug(f"Extracted OLE metadata: {list(metadata.keys())}")

    except Exception as e:
        logger.warning(f"Failed to extract OLE metadata: {e}")

    return metadata


def _decode_ole_string(value) -> str:
    """
    OLE 메타데이터 문자열을 디코딩합니다.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        # 여러 인코딩 시도
        for encoding in ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
            try:
                return value.decode(encoding).strip()
            except (UnicodeDecodeError, UnicodeError):
                continue
        return value.decode('utf-8', errors='replace').strip()
    return str(value).strip()


def _extract_ole_text(ole: olefile.OleFileIO) -> str:
    """
    OLE Compound Document에서 텍스트를 직접 추출합니다.

    Word Binary Format (.doc)의 텍스트는 'WordDocument' 스트림에 저장됩니다.
    복잡한 구조이므로 기본적인 텍스트만 추출합니다.

    Args:
        ole: olefile.OleFileIO 객체

    Returns:
        추출된 텍스트
    """
    text_parts = []

    try:
        # WordDocument 스트림 확인
        if ole.exists('WordDocument'):
            logger.debug("Found WordDocument stream")

            # Word Binary Format은 매우 복잡하므로
            # 여기서는 간단한 텍스트 추출만 시도
            stream = ole.openstream('WordDocument')
            data = stream.read()

            # 기본적인 텍스트 추출 시도
            # Word 문서의 텍스트는 유니코드 또는 CP1252로 인코딩됨
            text = _extract_text_from_word_binary(data)
            if text:
                text_parts.append(text)

        # 1Table 또는 0Table 스트림 (텍스트 위치 정보)
        # 이것은 복잡한 파싱이 필요하므로 LibreOffice 변환을 권장

    except Exception as e:
        logger.debug(f"Error extracting text from OLE: {e}")

    return "\n".join(text_parts)


def _extract_text_from_word_binary(data: bytes) -> str:
    """
    Word Binary Format에서 텍스트를 추출합니다.

    이것은 간소화된 버전입니다. 완전한 파싱을 위해서는
    LibreOffice 변환을 사용하는 것이 좋습니다.

    Args:
        data: WordDocument 스트림 데이터

    Returns:
        추출된 텍스트
    """
    try:
        # FIB (File Information Block) 파싱
        if len(data) < 32:
            return ""

        # wIdent (매직 넘버) 확인: 0xA5EC 또는 0xA5DC
        wIdent = struct.unpack('<H', data[0:2])[0]
        if wIdent not in (0xA5EC, 0xA5DC):
            return ""

        # 복잡한 구조 파싱은 생략하고
        # 단순히 유니코드 문자열 추출 시도
        text_parts = []

        # 연속된 ASCII/유니코드 텍스트 찾기
        i = 0
        current_text = []

        while i < len(data) - 1:
            # UTF-16LE 문자 시도
            char_code = struct.unpack('<H', data[i:i+2])[0]

            if 32 <= char_code < 127 or char_code in (9, 10, 13):
                # 일반 ASCII 범위
                current_text.append(chr(char_code))
                i += 2
            elif 0xAC00 <= char_code <= 0xD7A3:
                # 한글 범위
                current_text.append(chr(char_code))
                i += 2
            elif 0x3000 <= char_code <= 0x9FFF:
                # CJK 문자 범위
                current_text.append(chr(char_code))
                i += 2
            else:
                if len(current_text) > 5:
                    text_parts.append(''.join(current_text))
                current_text = []
                i += 2

        if len(current_text) > 5:
            text_parts.append(''.join(current_text))

        return '\n'.join(text_parts)

    except Exception as e:
        logger.debug(f"Error extracting text from Word binary: {e}")
        return ""


def _extract_ole_images(ole: olefile.OleFileIO, processed_images: Set[str]) -> List[str]:
    """
    OLE Compound Document에서 이미지를 추출합니다.

    이미지는 'Data' 또는 'Pictures' 스트림에 저장될 수 있습니다.

    Args:
        ole: olefile.OleFileIO 객체
        processed_images: 처리된 이미지 해시 집합

    Returns:
        이미지 태그 리스트
    """
    images = []

    try:
        # 이미지 스트림 찾기
        for entry in ole.listdir():
            entry_path = '/'.join(entry)

            # 이미지 관련 스트림
            if any(x.lower() in ['pictures', 'data', 'object', 'oleobject']
                   for x in entry):
                try:
                    stream = ole.openstream(entry)
                    data = stream.read()

                    # 이미지 시그니처 확인
                    if data[:8] == b'\x89PNG\r\n\x1a\n':
                        ext = 'png'
                    elif data[:2] == b'\xff\xd8':
                        ext = 'jpg'
                    elif data[:6] in (b'GIF87a', b'GIF89a'):
                        ext = 'gif'
                    elif data[:2] == b'BM':
                        ext = 'bmp'
                    else:
                        continue

                    # 로컬에 저장
                    image_tag = _image_processor.save_image(data)

                    if image_tag:
                        images.append(f"\n{image_tag}\n")
                    else:
                        images.append("[이미지]")

                except Exception as e:
                    logger.debug(f"Error extracting image from {entry_path}: {e}")
                    continue

    except Exception as e:
        logger.warning(f"Failed to extract OLE images: {e}")

    return images


def _extract_text_from_ole(
    file_path: str,
    extract_default_metadata: bool = True
) -> str:
    """
    OLE Compound Document (.doc)에서 텍스트를 추출합니다.

    구형 Word Binary Format은 매우 복잡하므로,
    메타데이터를 직접 추출하고 텍스트는 LibreOffice로 변환합니다.

    Args:
        file_path: DOC 파일 경로
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트
    """
    logger.info(f"Processing OLE Compound Document: {file_path}")

    result_parts = []
    processed_images: Set[str] = set()

    try:
        with olefile.OleFileIO(file_path) as ole:
            # 메타데이터 추출 (extract_default_metadata가 True인 경우에만)
            if extract_default_metadata:
                metadata = _extract_ole_metadata(ole)
                metadata_str = _format_metadata(metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")
                    logger.info(f"OLE metadata extracted: {list(metadata.keys())}")

            # 이미지 추출
            images = _extract_ole_images(ole, processed_images)

    except Exception as e:
        logger.warning(f"Error reading OLE file: {e}")
        metadata_str = ""
        images = []

    # LibreOffice로 텍스트 변환
    try:
        converted_text = _convert_with_libreoffice(file_path, skip_metadata=True, extract_default_metadata=extract_default_metadata)

        if converted_text:
            result_parts.append("<페이지 번호> 1 </페이지 번호>\n")
            result_parts.append(converted_text)

        # 이미지 태그 추가
        for img_tag in images:
            result_parts.append(img_tag)

        result = "\n".join(result_parts)
        logger.info(f"OLE processing completed: {len(result)} chars")

        return result

    except Exception as e:
        logger.error(f"Error in OLE processing: {e}")

        # 직접 텍스트 추출 시도 (폴백)
        try:
            with olefile.OleFileIO(file_path) as ole:
                text = _extract_ole_text(ole)
                if text:
                    result_parts.append("<페이지 번호> 1 </페이지 번호>\n")
                    result_parts.append(text)
        except:
            pass

        if result_parts:
            return "\n".join(result_parts)

        return f"[DOC 파일 처리 실패: {str(e)}]"


# === HTML 처리 ===

def _extract_text_from_html_doc(
    file_path: str,
    extract_default_metadata: bool = True
) -> str:
    """
    HTML로 저장된 DOC 파일에서 텍스트를 추출합니다.

    Word에서 HTML로 저장하면 특수한 Microsoft Office 네임스페이스와
    스타일이 포함됩니다.

    Args:
        file_path: DOC 파일 경로
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트
    """
    logger.info(f"Processing HTML-format DOC: {file_path}")

    try:
        # 파일 읽기 (여러 인코딩 시도)
        content = None
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'cp1252', 'latin-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace')

        result_parts = []
        processed_images: Set[str] = set()

        # BeautifulSoup으로 파싱
        soup = BeautifulSoup(content, 'html.parser')

        # 메타데이터 추출 (extract_default_metadata가 True인 경우에만)
        if extract_default_metadata:
            metadata = _extract_html_metadata(soup)
            metadata_str = _format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")

        # 페이지 1 시작
        result_parts.append("<페이지 번호> 1 </페이지 번호>\n")

        # 스크립트, 스타일 태그 제거
        for tag in soup(['script', 'style', 'meta', 'link', 'head']):
            tag.decompose()

        # 텍스트 추출
        text = soup.get_text(separator='\n', strip=True)

        # 연속된 빈 줄 정리
        text = re.sub(r'\n{3,}', '\n\n', text)

        if text:
            result_parts.append(text)

        # 테이블 추출 (HTML 형식 보존)
        tables = soup.find_all('table')
        if tables:
            for table in tables:
                # 테이블 HTML 정리
                table_html = str(table)
                # Word의 복잡한 스타일 제거
                table_html = re.sub(r'\s+style="[^"]*"', '', table_html)
                table_html = re.sub(r'\s+class="[^"]*"', '', table_html)
                result_parts.append("\n" + table_html + "\n")

        # 이미지 추출
        imgs = soup.find_all('img')
        for img in imgs:
            src = img.get('src', '')
            if src:
                # base64 인코딩된 이미지 처리
                if src.startswith('data:image'):
                    try:
                        # data:image/png;base64,... 형식
                        match = re.match(r'data:image/(\w+);base64,(.+)', src)
                        if match:
                            image_data = base64.b64decode(match.group(2))
                            image_tag = _image_processor.save_image(image_data)
                            if image_tag:
                                result_parts.append(f"\n{image_tag}\n")
                            else:
                                result_parts.append("[이미지]")
                    except Exception as e:
                        logger.debug(f"Error processing base64 image: {e}")
                        result_parts.append("[이미지]")
                else:
                    result_parts.append(f"[이미지: {src}]")

        result = "\n".join(result_parts)
        logger.info(f"HTML DOC processing completed: {len(result)} chars")

        return result

    except Exception as e:
        logger.error(f"Error processing HTML DOC: {e}")
        logger.debug(traceback.format_exc())
        # 폴백: LibreOffice 변환
        return _convert_with_libreoffice(file_path)


def _extract_html_metadata(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    HTML에서 메타데이터를 추출합니다.

    Word에서 생성한 HTML에는 meta 태그에 문서 정보가 포함됩니다.
    """
    metadata = {}

    try:
        # <title> 태그
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            metadata['title'] = title_tag.string.strip()

        # meta 태그들
        meta_mappings = {
            'author': 'author',
            'description': 'comments',
            'keywords': 'keywords',
            'subject': 'subject',
            'creator': 'author',
            'producer': 'last_saved_by',
        }

        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')

            if name in meta_mappings and content:
                metadata[meta_mappings[name]] = content.strip()

    except Exception as e:
        logger.debug(f"Error extracting HTML metadata: {e}")

    return metadata


# === LibreOffice 변환 ===

def _convert_with_libreoffice(
    file_path: str,
    skip_metadata: bool = False,
    extract_default_metadata: bool = True
) -> str:
    """
    LibreOffice를 사용하여 DOC 파일을 텍스트로 변환합니다.

    LibreOffice의 headless 모드를 사용하여 문서를 텍스트 또는 HTML로 변환합니다.

    Args:
        file_path: DOC 파일 경로
        skip_metadata: 메타데이터 추출 건너뛰기 (이미 추출된 경우)
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        변환된 텍스트
    """
    logger.info(f"Converting DOC with LibreOffice: {file_path}")

    # LibreOffice 확인
    libreoffice_path = None
    for path in ['/usr/bin/libreoffice', '/usr/bin/soffice', 'libreoffice', 'soffice']:
        try:
            result = subprocess.run([path, '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                libreoffice_path = path
                break
        except:
            continue

    if libreoffice_path is None:
        logger.error("LibreOffice not found")
        return "[DOC 파일 처리 실패: LibreOffice가 설치되어 있지 않습니다]"

    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 파일을 임시 디렉토리로 복사 (한글 파일명 문제 방지)
            temp_input = os.path.join(temp_dir, "input.doc")
            shutil.copy2(file_path, temp_input)

            # LibreOffice로 HTML 변환 (테이블 보존을 위해)
            cmd = [
                libreoffice_path,
                '--headless',
                '--convert-to', 'html:HTML:EmbedImages',
                '--outdir', temp_dir,
                temp_input
            ]

            logger.debug(f"Running LibreOffice: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120,
                env={**os.environ, 'HOME': temp_dir}  # HOME 설정
            )

            if result.returncode != 0:
                logger.warning(f"LibreOffice conversion warning: {result.stderr.decode('utf-8', errors='replace')}")

            # 변환된 파일 찾기
            html_file = os.path.join(temp_dir, "input.html")
            txt_file = os.path.join(temp_dir, "input.txt")

            result_parts = []
            processed_images: Set[str] = set()

            if os.path.exists(html_file):
                # HTML 파일 처리
                with open(html_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                soup = BeautifulSoup(content, 'html.parser')

                # 메타데이터 추출 (skip하지 않고 extract_default_metadata가 True인 경우)
                if not skip_metadata and extract_default_metadata:
                    metadata = _extract_html_metadata(soup)
                    metadata_str = _format_metadata(metadata)
                    if metadata_str:
                        result_parts.append(metadata_str + "\n\n")

                # 스크립트, 스타일 제거
                for tag in soup(['script', 'style', 'meta', 'link']):
                    tag.decompose()

                # 텍스트 추출
                text = soup.get_text(separator='\n', strip=True)
                text = re.sub(r'\n{3,}', '\n\n', text)

                if text:
                    result_parts.append(text)

                # 테이블 처리
                tables = soup.find_all('table')
                for table in tables:
                    table_html = str(table)
                    table_html = re.sub(r'\s+style="[^"]*"', '', table_html)
                    table_html = re.sub(r'\s+class="[^"]*"', '', table_html)
                    result_parts.append("\n" + table_html + "\n")

                # 이미지 처리
                imgs = soup.find_all('img')
                for img in imgs:
                    src = img.get('src', '')
                    if src and src.startswith('data:image'):
                        try:
                            match = re.match(r'data:image/(\w+);base64,(.+)', src)
                            if match:
                                image_data = base64.b64decode(match.group(2))
                                image_tag = _image_processor.save_image(image_data)
                                if image_tag:
                                    result_parts.append(f"\n{image_tag}\n")
                        except:
                            pass

            elif os.path.exists(txt_file):
                # 텍스트 파일 처리
                with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()

                if text:
                    result_parts.append(text)

            else:
                # 변환 실패
                logger.error(f"LibreOffice conversion failed - no output file found")
                return "[DOC 파일 처리 실패: LibreOffice 변환 실패]"

            result = "\n".join(result_parts)
            logger.info(f"LibreOffice conversion completed: {len(result)} chars")

            return result

        except subprocess.TimeoutExpired:
            logger.error("LibreOffice conversion timed out")
            return "[DOC 파일 처리 실패: 변환 시간 초과]"

        except Exception as e:
            logger.error(f"LibreOffice conversion error: {e}")
            logger.debug(traceback.format_exc())
            return f"[DOC 파일 처리 실패: {str(e)}]"


# === DOCX 처리 (잘못된 확장자) ===

def _extract_text_from_docx_misnamed(
    file_path: str,
    extract_default_metadata: bool = True
) -> str:
    """
    .doc 확장자로 저장된 DOCX 파일을 처리합니다.

    파일을 임시 .docx 파일로 복사하고 docx_handler를 사용합니다.

    Args:
        file_path: DOC 파일 경로
        extract_default_metadata: 기본 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트
    """
    logger.info(f"Processing misnamed DOCX file: {file_path}")

    try:
        # docx_handler import
        from libs.core.processor.docx_handler import extract_text_from_docx

        # 임시 파일로 복사 (확장자를 .docx로)
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            shutil.copy2(file_path, tmp.name)
            temp_path = tmp.name

        try:
            # docx_handler로 처리
            result = extract_text_from_docx(temp_path, None, extract_default_metadata)
            logger.info(f"Misnamed DOCX processing completed")
            return result
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error processing misnamed DOCX: {e}")
        logger.debug(traceback.format_exc())
        return f"[DOC 파일 처리 실패: {str(e)}]"


# === 메인 함수 ===

def extract_text_from_doc(
    file_path: str,
    current_config: Dict[str, Any] = None,
    extract_default_metadata: bool = True
) -> str:
    """
    구형 DOC 파일에서 텍스트를 추출합니다.

    파일의 실제 형식(RTF, OLE, HTML, DOCX)을 자동 감지하여
    적절한 방법으로 처리합니다.

    Args:
        file_path: DOC 파일 경로
        current_config: 설정 딕셔너리
        extract_default_metadata: 메타데이터 추출 여부 (기본값: True)

    Returns:
        추출된 텍스트 (메타데이터, 테이블 HTML, 이미지 태그 포함)
    """
    if current_config is None:
        current_config = {}

    logger.info(f"DOC processing: {file_path}")

    # 파일 존재 확인
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return f"[DOC 파일을 찾을 수 없습니다: {file_path}]"

    # 파일 형식 감지
    doc_format = _detect_doc_format(file_path)

    try:
        if doc_format == DocFormat.RTF:
            # RTF 형식 처리
            return _extract_text_from_rtf(file_path, extract_default_metadata)

        elif doc_format == DocFormat.OLE:
            # OLE Compound Document 처리
            return _extract_text_from_ole(file_path, extract_default_metadata)

        elif doc_format == DocFormat.HTML:
            # HTML 형식 처리
            return _extract_text_from_html_doc(file_path, extract_default_metadata)

        elif doc_format == DocFormat.DOCX:
            # 잘못된 확장자의 DOCX 처리
            return _extract_text_from_docx_misnamed(file_path, extract_default_metadata)

        else:
            # 알 수 없는 형식 - LibreOffice 변환 시도
            logger.warning(f"Unknown DOC format, trying LibreOffice conversion")
            return _convert_with_libreoffice(file_path, extract_default_metadata)

    except Exception as e:
        logger.error(f"Error in DOC processing: {e}")
        logger.debug(traceback.format_exc())

        # 최종 폴백: LibreOffice 변환
        try:
            return _convert_with_libreoffice(file_path, extract_default_metadata)
        except Exception as e2:
            logger.error(f"LibreOffice fallback also failed: {e2}")
            return f"[DOC 파일 처리 실패: {str(e)}]"
