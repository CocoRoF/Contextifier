# libs/core/processor/doc_handler.py
"""
DOC Handler - Legacy Microsoft Word Document Processor

Class-based handler for DOC files inheriting from BaseHandler.
Automatically detects file format (RTF, OLE, HTML, DOCX) and processes accordingly.
RTF processing is delegated to RTFHandler.
"""
import io
import logging
import os
import re
import struct
import base64
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from enum import Enum
import zipfile

import olefile
from bs4 import BeautifulSoup

from contextifier.core.processor.base_handler import BaseHandler
from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.chart_extractor import BaseChartExtractor, NullChartExtractor
from contextifier.core.processor.doc_helpers.doc_image_processor import DOCImageProcessor

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile

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


class DOCHandler(BaseHandler):
    """DOC file processing handler class."""
    
    def _create_file_converter(self):
        """Create DOC-specific file converter."""
        from contextifier.core.processor.doc_helpers.doc_file_converter import DOCFileConverter
        return DOCFileConverter()
    
    def _create_chart_extractor(self) -> BaseChartExtractor:
        """DOC files chart extraction not yet implemented. Return NullChartExtractor."""
        return NullChartExtractor(self._chart_processor)
    
    def _create_metadata_extractor(self):
        """DOC metadata extraction not yet implemented. Return None to use NullMetadataExtractor."""
        return None
    
    def _create_format_image_processor(self) -> ImageProcessor:
        """Create DOC-specific image processor."""
        return DOCImageProcessor()
    
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
        
        try:
            # Use file_converter to detect format and convert
            converted_obj, doc_format = self.file_converter.convert(file_data)
            
            if doc_format == DocFormat.RTF:
                # Delegate to RTFHandler for RTF processing
                return self._delegate_to_rtf_handler(converted_obj, current_file, extract_metadata)
            elif doc_format == DocFormat.OLE:
                return self._extract_from_ole_obj(converted_obj, current_file, extract_metadata)
            elif doc_format == DocFormat.HTML:
                return self._extract_from_html_obj(converted_obj, current_file, extract_metadata)
            elif doc_format == DocFormat.DOCX:
                return self._extract_from_docx_obj(converted_obj, current_file, extract_metadata)
            else:
                self.logger.warning(f"Unknown DOC format, trying OLE fallback: {file_path}")
                return self._extract_from_ole(current_file, extract_metadata)
        except Exception as e:
            self.logger.error(f"Error in DOC processing: {e}")
            return f"[DOC file processing failed: {str(e)}]"
    
    def _delegate_to_rtf_handler(self, rtf_doc, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """
        Delegate RTF processing to RTFHandler.
        
        Args:
            rtf_doc: Pre-converted RTFDocument object
            current_file: CurrentFile dict
            extract_metadata: Whether to extract metadata
            
        Returns:
            Extracted text
        """
        from contextifier.core.processor.rtf_handler import RTFHandler
        
        rtf_handler = RTFHandler(
            config=self.config,
            image_processor=self._image_processor,
            page_tag_processor=self._page_tag_processor,
            chart_processor=self._chart_processor
        )
        
        return rtf_handler.extract_from_rtf_document(rtf_doc, current_file, extract_metadata)
    
    def _extract_from_ole_obj(self, ole, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """OLE Compound Document processing using pre-converted OLE object."""
        file_path = current_file.get("file_path", "unknown")
        
        self.logger.info(f"Processing OLE: {file_path}")
        
        result_parts = []
        processed_images: Set[str] = set()
        
        try:
            # Metadata extraction
            if extract_metadata:
                metadata = self._extract_ole_metadata(ole)
                metadata_str = self.extract_and_format_metadata(metadata)
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
        finally:
            # Close the OLE object
            self.file_converter.close(ole)
        
        return "\n".join(result_parts)
    
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
                    metadata_str = self.extract_and_format_metadata(metadata)
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
                            image_tag = self.format_image_processor.save_image(data)
                            if image_tag:
                                images.append(f"\n{image_tag}\n")
                    except:
                        continue
        except Exception as e:
            self.logger.warning(f"Error extracting OLE images: {e}")
        return images
    
    def _extract_from_html_obj(self, soup, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """HTML DOC processing using pre-converted BeautifulSoup object."""
        file_path = current_file.get("file_path", "unknown")
        
        self.logger.info(f"Processing HTML DOC: {file_path}")
        
        result_parts = []
        
        if extract_metadata:
            metadata = self._extract_html_metadata(soup)
            metadata_str = self.extract_and_format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")
        
        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")
        
        # Copy soup to avoid modifying the original
        soup_copy = BeautifulSoup(str(soup), 'html.parser')
        
        for tag in soup_copy(['script', 'style', 'meta', 'link', 'head']):
            tag.decompose()
        
        text = soup_copy.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        if text:
            result_parts.append(text)
        
        for table in soup_copy.find_all('table'):
            table_html = str(table)
            table_html = re.sub(r'\s+style="[^"]*"', '', table_html)
            table_html = re.sub(r'\s+class="[^"]*"', '', table_html)
            result_parts.append("\n" + table_html + "\n")
        
        for img in soup_copy.find_all('img'):
            src = img.get('src', '')
            if src and src.startswith('data:image'):
                try:
                    match = re.match(r'data:image/(\w+);base64,(.+)', src)
                    if match:
                        image_data = base64.b64decode(match.group(2))
                        image_tag = self.format_image_processor.save_image(image_data)
                        if image_tag:
                            result_parts.append(f"\n{image_tag}\n")
                except:
                    pass
        
        return "\n".join(result_parts)
    
    def _extract_from_docx_obj(self, doc, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """Extract from misnamed DOCX using pre-converted Document object."""
        file_path = current_file.get("file_path", "unknown")
        
        self.logger.info(f"Processing misnamed DOCX: {file_path}")
        
        try:
            result_parts = []
            
            if extract_metadata:
                # Basic metadata from docx Document
                if hasattr(doc, 'core_properties'):
                    metadata = {
                        'title': doc.core_properties.title or '',
                        'author': doc.core_properties.author or '',
                        'subject': doc.core_properties.subject or '',
                        'keywords': doc.core_properties.keywords or '',
                    }
                    metadata = {k: v for k, v in metadata.items() if v}
                    metadata_str = self.extract_and_format_metadata(metadata)
                    if metadata_str:
                        result_parts.append(metadata_str + "\n\n")
            
            page_tag = self.create_page_tag(1)
            result_parts.append(f"{page_tag}\n")
            
            for para in doc.paragraphs:
                if para.text.strip():
                    result_parts.append(para.text)
            
            for table in doc.tables:
                for row in table.rows:
                    row_texts = []
                    for cell in row.cells:
                        row_texts.append(cell.text.strip())
                    if any(t for t in row_texts):
                        result_parts.append(" | ".join(row_texts))
            
            return "\n".join(result_parts)
            
        except Exception as e:
            self.logger.error(f"Error processing misnamed DOCX: {e}")
            return f"[DOCX processing failed: {str(e)}]"
    
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
            metadata_str = self.extract_and_format_metadata(metadata)
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
                        image_tag = self.format_image_processor.save_image(image_data)
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
            from contextifier.core.processor.docx_handler import DOCXHandler
            
            # Pass current_file directly - DOCXHandler now accepts CurrentFile
            docx_handler = DOCXHandler(config=self.config, image_processor=self.format_image_processor)
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
