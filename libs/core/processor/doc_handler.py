# libs/core/processor/doc_handler.py
"""
DOC Handler - Legacy Microsoft Word Document Processor

Class-based handler for DOC files inheriting from BaseHandler.
Automatically detects file format (RTF, OLE, HTML, DOCX) and processes accordingly.
"""
import io
import logging
import os
import re
import shutil
import tempfile
import struct
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from enum import Enum
import zipfile

import olefile
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text

from libs.core.processor.doc_helpers.rtf_parser import parse_rtf, RTFDocument
from libs.core.processor.base_handler import BaseHandler
from libs.core.functions.img_processor import ImageProcessor

if TYPE_CHECKING:
    from libs.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


class DocFormat(Enum):
    """DOC 파일의 실제 형식"""
    RTF = "rtf"
    OLE = "ole"
    HTML = "html"
    DOCX = "docx"
    UNKNOWN = "unknown"


MAGIC_NUMBERS = {
    'RTF': b'{\\rtf',
    'OLE': b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',
    'ZIP': b'PK\x03\x04',
}

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


class DOCHandler(BaseHandler):
    """DOC file processing handler class."""
    
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """Extract text from DOC file."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"DOC processing: {file_path}")
        
        if not file_data:
            self.logger.error(f"Empty file data: {file_path}")
            return f"[DOC file is empty: {file_path}]"
        
        doc_format = self._detect_format_from_bytes(file_data)
        
        try:
            if doc_format == DocFormat.RTF:
                return self._extract_from_rtf(current_file, extract_metadata)
            elif doc_format == DocFormat.OLE:
                return self._extract_from_ole(current_file, extract_metadata)
            elif doc_format == DocFormat.HTML:
                return self._extract_from_html(current_file, extract_metadata)
            elif doc_format == DocFormat.DOCX:
                return self._extract_from_docx_misnamed(current_file, extract_metadata)
            else:
                self.logger.warning(f"Unknown DOC format, trying OLE: {file_path}")
                return self._extract_from_ole(current_file, extract_metadata)
        except Exception as e:
            self.logger.error(f"Error in DOC processing: {e}")
            return f"[DOC file processing failed: {str(e)}]"
    
    def _detect_format_from_bytes(self, file_data: bytes) -> DocFormat:
        """Detect file format from binary data."""
        try:
            header = file_data[:32] if len(file_data) >= 32 else file_data
            
            if not header:
                return DocFormat.UNKNOWN
            
            if header.startswith(MAGIC_NUMBERS['RTF']):
                return DocFormat.RTF
            
            if header.startswith(MAGIC_NUMBERS['OLE']):
                return DocFormat.OLE
            
            if header.startswith(MAGIC_NUMBERS['ZIP']):
                try:
                    file_stream = io.BytesIO(file_data)
                    with zipfile.ZipFile(file_stream, 'r') as zf:
                        if '[Content_Types].xml' in zf.namelist():
                            return DocFormat.DOCX
                except zipfile.BadZipFile:
                    pass
            
            header_lower = header.lower()
            if header_lower.startswith(b'<!doctype') or header_lower.startswith(b'<html') or b'<html' in header_lower[:100]:
                return DocFormat.HTML
            
            try:
                if header.startswith(b'\xef\xbb\xbf'):
                    text_header = header[3:].decode('utf-8', errors='ignore').lower()
                else:
                    text_header = header.decode('utf-8', errors='ignore').lower()
                
                if text_header.startswith('{\\rtf'):
                    return DocFormat.RTF
                if text_header.startswith('<!doctype') or text_header.startswith('<html'):
                    return DocFormat.HTML
            except:
                pass
            
            return DocFormat.UNKNOWN
        except Exception as e:
            self.logger.error(f"Error detecting format: {e}")
            return DocFormat.UNKNOWN
    
    def _extract_from_rtf(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """RTF file processing."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"Processing RTF: {file_path}")
        
        try:
            content = file_data
            
            processed_images: Set[str] = set()
            doc = parse_rtf(content, processed_images=processed_images, image_processor=self.image_processor)
            
            result_parts = []
            
            if extract_metadata:
                metadata_str = self._format_metadata(doc.metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")
            
            page_tag = self.create_page_tag(1)
            result_parts.append(f"{page_tag}\n")
            
            inline_content = doc.get_inline_content()
            if inline_content:
                result_parts.append(inline_content)
            else:
                if doc.text_content:
                    result_parts.append(doc.text_content)
                
                for table in doc.tables:
                    if not table.rows:
                        continue
                    if table.is_real_table():
                        result_parts.append("\n" + table.to_html() + "\n")
                    else:
                        result_parts.append("\n" + table.to_text_list() + "\n")
            
            result = "\n".join(result_parts)
            result = re.sub(r'\[image:[^\]]*uploads/\.[^\]]*\]', '', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"RTF processing error: {e}")
            return self._extract_rtf_fallback(current_file, extract_metadata)
    
    def _extract_rtf_fallback(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """RTF fallback (striprtf)."""
        file_data = current_file.get("file_data", b"")
        
        content = None
        for encoding in ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
            try:
                content = file_data.decode(encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            content = file_data.decode('cp1252', errors='replace')
        
        result_parts = []
        
        if extract_metadata:
            metadata = self._extract_rtf_metadata(content)
            metadata_str = self._format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")
        
        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")
        
        try:
            text = rtf_to_text(content)
        except:
            text = re.sub(r'\\[a-z]+\d*\s?', '', content)
            text = re.sub(r"\\'[0-9a-fA-F]{2}", '', text)
            text = re.sub(r'[{}]', '', text)
        
        if text:
            text = re.sub(r'\n{3,}', '\n\n', text)
            result_parts.append(text.strip())
        
        return "\n".join(result_parts)
    
    def _extract_rtf_metadata(self, content: str) -> Dict[str, Any]:
        """RTF metadata extraction."""
        metadata = {}
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
        
        return metadata
    
    def _extract_from_ole(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """OLE Compound Document processing - extract text directly from WordDocument stream."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"Processing OLE: {file_path}")
        
        result_parts = []
        processed_images: Set[str] = set()
        
        try:
            file_stream = io.BytesIO(file_data)
            with olefile.OleFileIO(file_stream) as ole:
                # Metadata extraction
                if extract_metadata:
                    metadata = self._extract_ole_metadata(ole)
                    metadata_str = self._format_metadata(metadata)
                    if metadata_str:
                        result_parts.append(metadata_str + "\n\n")
                
                page_tag = self.create_page_tag(1)
                result_parts.append(f"{page_tag}\n")
                
                # Extract text from WordDocument stream
                text = self._extract_ole_text(ole)
                if text:
                    result_parts.append(text)
                
                # Extract images
                images = self._extract_ole_images(ole, processed_images)
                for img_tag in images:
                    result_parts.append(img_tag)
                
        except Exception as e:
            self.logger.error(f"OLE processing error: {e}")
            return f"[DOC file processing failed: {str(e)}]"
        
        return "\n".join(result_parts)
    
    def _extract_ole_metadata(self, ole: olefile.OleFileIO) -> Dict[str, Any]:
        """OLE 메타데이터 추출"""
        metadata = {}
        try:
            ole_meta = ole.get_metadata()
            if ole_meta:
                if ole_meta.title:
                    metadata['title'] = self._decode_ole_string(ole_meta.title)
                if ole_meta.subject:
                    metadata['subject'] = self._decode_ole_string(ole_meta.subject)
                if ole_meta.author:
                    metadata['author'] = self._decode_ole_string(ole_meta.author)
                if ole_meta.keywords:
                    metadata['keywords'] = self._decode_ole_string(ole_meta.keywords)
                if ole_meta.comments:
                    metadata['comments'] = self._decode_ole_string(ole_meta.comments)
                if ole_meta.last_saved_by:
                    metadata['last_saved_by'] = self._decode_ole_string(ole_meta.last_saved_by)
                if ole_meta.create_time:
                    metadata['create_time'] = ole_meta.create_time
                if ole_meta.last_saved_time:
                    metadata['last_saved_time'] = ole_meta.last_saved_time
        except Exception as e:
            self.logger.warning(f"Error extracting OLE metadata: {e}")
        return metadata
    
    def _decode_ole_string(self, value) -> str:
        """OLE 문자열 디코딩"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bytes):
            for encoding in ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
                try:
                    return value.decode(encoding).strip()
                except (UnicodeDecodeError, UnicodeError):
                    continue
            return value.decode('utf-8', errors='replace').strip()
        return str(value).strip()
    
    def _extract_ole_images(self, ole: olefile.OleFileIO, processed_images: Set[str]) -> List[str]:
        """OLE에서 이미지 추출"""
        images = []
        try:
            for entry in ole.listdir():
                if any(x.lower() in ['pictures', 'data', 'object', 'oleobject'] for x in entry):
                    try:
                        stream = ole.openstream(entry)
                        data = stream.read()
                        
                        if data[:8] == b'\x89PNG\r\n\x1a\n' or data[:2] == b'\xff\xd8' or \
                           data[:6] in (b'GIF87a', b'GIF89a') or data[:2] == b'BM':
                            image_tag = self.image_processor.save_image(data)
                            if image_tag:
                                images.append(f"\n{image_tag}\n")
                    except:
                        continue
        except Exception as e:
            self.logger.warning(f"Error extracting OLE images: {e}")
        return images
    
    def _extract_from_html(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """HTML DOC processing."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")
        
        self.logger.info(f"Processing HTML DOC: {file_path}")
        
        content = None
        for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
            try:
                content = file_data.decode(encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            content = file_data.decode('utf-8', errors='replace')
        
        result_parts = []
        soup = BeautifulSoup(content, 'html.parser')
        
        if extract_metadata:
            metadata = self._extract_html_metadata(soup)
            metadata_str = self._format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")
        
        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")
        
        for tag in soup(['script', 'style', 'meta', 'link', 'head']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        if text:
            result_parts.append(text)
        
        for table in soup.find_all('table'):
            table_html = str(table)
            table_html = re.sub(r'\s+style="[^"]*"', '', table_html)
            table_html = re.sub(r'\s+class="[^"]*"', '', table_html)
            result_parts.append("\n" + table_html + "\n")
        
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and src.startswith('data:image'):
                try:
                    match = re.match(r'data:image/(\w+);base64,(.+)', src)
                    if match:
                        image_data = base64.b64decode(match.group(2))
                        image_tag = self.image_processor.save_image(image_data)
                        if image_tag:
                            result_parts.append(f"\n{image_tag}\n")
                except:
                    pass
        
        return "\n".join(result_parts)
    
    def _extract_html_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """HTML metadata extraction."""
        metadata = {}
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            metadata['title'] = title_tag.string.strip()
        
        meta_mappings = {
            'author': 'author', 'description': 'comments', 'keywords': 'keywords',
            'subject': 'subject', 'creator': 'author', 'producer': 'last_saved_by',
        }
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            if name in meta_mappings and content:
                metadata[meta_mappings[name]] = content.strip()
        
        return metadata
    
    def _extract_from_docx_misnamed(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """Process misnamed DOCX file."""
        file_path = current_file.get("file_path", "unknown")
        
        self.logger.info(f"Processing misnamed DOCX: {file_path}")
        
        try:
            from libs.core.processor.docx_handler import DOCXHandler
            
            # Pass current_file directly - DOCXHandler now accepts CurrentFile
            docx_handler = DOCXHandler(config=self.config, image_processor=self.image_processor)
            return docx_handler.extract_text(current_file, extract_metadata=extract_metadata)
        except Exception as e:
            self.logger.error(f"Error processing misnamed DOCX: {e}")
            return f"[DOC file processing failed: {str(e)}]"
    
    def _extract_ole_text(self, ole: olefile.OleFileIO) -> str:
        """Extract text from OLE WordDocument stream."""
        try:
            # Check WordDocument stream
            if not ole.exists('WordDocument'):
                self.logger.warning("WordDocument stream not found")
                return ""
            
            # Read Word Document stream
            word_stream = ole.openstream('WordDocument')
            word_data = word_stream.read()
            
            if len(word_data) < 12:
                return ""
            
            # FIB (File Information Block) parsing
            # Check magic number (0xA5EC or 0xA5DC)
            magic = struct.unpack('<H', word_data[0:2])[0]
            if magic not in (0xA5EC, 0xA5DC):
                self.logger.warning(f"Invalid Word magic number: {hex(magic)}")
                return ""
            
            # Text extraction attempt
            text_parts = []
            
            # 1. Table 스트림에서 텍스트 조각 찾기 시도
            table_stream_name = None
            if ole.exists('1Table'):
                table_stream_name = '1Table'
            elif ole.exists('0Table'):
                table_stream_name = '0Table'
            
            # 2. 간단한 방식: 유니코드/ASCII 텍스트 직접 추출
            # Word 97-2003은 대부분 유니코드 텍스트를 포함
            extracted_text = self._extract_text_from_word_stream(word_data)
            if extracted_text:
                text_parts.append(extracted_text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            self.logger.warning(f"Error extracting OLE text: {e}")
            return ""
    
    def _extract_text_from_word_stream(self, data: bytes) -> str:
        """Word 스트림에서 텍스트 추출 (휴리스틱 방식)"""
        text_parts = []
        
        # 방법 1: UTF-16LE 유니코드 텍스트 추출
        try:
            # 연속된 유니코드 문자열 찾기
            i = 0
            while i < len(data) - 1:
                # 유니코드 텍스트 시작점 찾기 (printable 문자)
                if 0x20 <= data[i] <= 0x7E and data[i+1] == 0x00:
                    # 유니코드 문자열 수집
                    unicode_bytes = []
                    j = i
                    while j < len(data) - 1:
                        char = data[j]
                        next_byte = data[j+1]
                        
                        # ASCII 범위 유니코드 문자 또는 한글
                        if next_byte == 0x00 and (0x20 <= char <= 0x7E or char in (0x0D, 0x0A, 0x09)):
                            unicode_bytes.extend([char, next_byte])
                            j += 2
                        elif 0xAC <= next_byte <= 0xD7:  # 한글 유니코드 범위 (AC00-D7AF)
                            unicode_bytes.extend([char, next_byte])
                            j += 2
                        elif next_byte in range(0x30, 0x4E):  # CJK 범위 일부
                            unicode_bytes.extend([char, next_byte])
                            j += 2
                        else:
                            break
                    
                    if len(unicode_bytes) >= 8:  # 최소 4자 이상
                        try:
                            text = bytes(unicode_bytes).decode('utf-16-le', errors='ignore')
                            text = text.strip()
                            if len(text) >= 4 and not text.startswith('\\'):
                                # 제어 문자 정리
                                text = text.replace('\r\n', '\n').replace('\r', '\n')
                                text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
                                if text:
                                    text_parts.append(text)
                        except:
                            pass
                    i = j
                else:
                    i += 1
        except Exception as e:
            self.logger.debug(f"Unicode extraction error: {e}")
        
        # 결과 정리
        if text_parts:
            # 중복 제거 및 연결
            seen = set()
            unique_parts = []
            for part in text_parts:
                if part not in seen and len(part) > 3:
                    seen.add(part)
                    unique_parts.append(part)
            
            result = '\n'.join(unique_parts)
            # 과도한 줄바꿈 정리
            result = re.sub(r'\n{3,}', '\n\n', result)
            return result.strip()
        
        return ""
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """메타데이터 포맷팅"""
        if not metadata:
            return ""
        
        lines = ["<Document-Metadata>"]
        for key, label in METADATA_FIELD_NAMES.items():
            if key in metadata and metadata[key]:
                value = metadata[key]
                if isinstance(value, datetime):
                    value = value.strftime('%Y-%m-%d %H:%M:%S')
                lines.append(f"  {label}: {value}")
        lines.append("</Document-Metadata>")
        
        return "\n".join(lines)
