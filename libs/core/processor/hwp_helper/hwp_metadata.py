# service/document_processor/processor/hwp_helper/hwp_metadata.py
"""
HWP 메타데이터 추출 유틸리티

HWP 5.0 OLE 파일에서 메타데이터를 추출합니다.
- extract_metadata: OLE 표준 메타데이터 + HwpSummaryInformation 추출
- parse_hwp_summary_information: HWP 고유 Property Set 파싱
- format_metadata: 메타데이터를 문자열로 포맷팅
"""
import struct
import logging
from datetime import datetime
from typing import Dict, Any

import olefile

logger = logging.getLogger("document-processor")


def extract_metadata(ole: olefile.OleFileIO) -> Dict[str, Any]:
    """
    HWP 파일의 메타데이터를 추출합니다.
    
    두 가지 방법으로 메타데이터를 추출합니다:
    1. olefile의 get_metadata() - OLE 표준 메타데이터
    2. HwpSummaryInformation 스트림 직접 파싱 - HWP 고유 메타데이터
    
    Args:
        ole: OLE 파일 객체
        
    Returns:
        추출된 메타데이터 딕셔너리
    """
    metadata = {}
    
    # Method 1: olefile의 get_metadata() 사용
    try:
        ole_meta = ole.get_metadata()
        
        if ole_meta:
            if ole_meta.title:
                metadata['title'] = ole_meta.title
            if ole_meta.subject:
                metadata['subject'] = ole_meta.subject
            if ole_meta.author:
                metadata['author'] = ole_meta.author
            if ole_meta.keywords:
                metadata['keywords'] = ole_meta.keywords
            if ole_meta.comments:
                metadata['comments'] = ole_meta.comments
            if ole_meta.last_saved_by:
                metadata['last_saved_by'] = ole_meta.last_saved_by
            if ole_meta.create_time:
                metadata['create_time'] = ole_meta.create_time
            if ole_meta.last_saved_time:
                metadata['last_saved_time'] = ole_meta.last_saved_time
        
        logger.info(f"Extracted OLE metadata: {metadata}")
        
    except Exception as e:
        logger.warning(f"Failed to extract OLE metadata: {e}")
    
    # Method 2: HwpSummaryInformation 스트림 직접 파싱
    try:
        hwp_summary_stream = '\x05HwpSummaryInformation'
        if ole.exists(hwp_summary_stream):
            logger.debug("Found HwpSummaryInformation stream, attempting to parse...")
            stream = ole.openstream(hwp_summary_stream)
            data = stream.read()
            hwp_meta = parse_hwp_summary_information(data)
            
            # HWP 특화 메타데이터가 우선
            for key, value in hwp_meta.items():
                if value:
                    metadata[key] = value
                    
    except Exception as e:
        logger.debug(f"Failed to parse HwpSummaryInformation: {e}")
    
    return metadata


def parse_hwp_summary_information(data: bytes) -> Dict[str, Any]:
    """
    HwpSummaryInformation 스트림을 파싱합니다. (OLE Property Set 형식)
    
    OLE Property Set 구조:
    - Header (28 bytes)
    - Section(s) containing property ID/offset pairs
    - Property values (string, datetime 등)
    
    Args:
        data: HwpSummaryInformation 스트림 바이너리 데이터
        
    Returns:
        파싱된 메타데이터 딕셔너리
    """
    metadata = {}
    
    try:
        if len(data) < 28:
            return metadata
        
        pos = 0
        _byte_order = struct.unpack('<H', data[pos:pos+2])[0]  # noqa: F841
        pos = 28  # 헤더 스킵
        
        if len(data) < pos + 20:
            return metadata
        
        # Section Header: FMTID (16 bytes) + Offset (4 bytes)
        section_offset = struct.unpack('<I', data[pos+16:pos+20])[0]
        
        if section_offset >= len(data):
            return metadata
        
        # Section 파싱
        pos = section_offset
        if len(data) < pos + 8:
            return metadata
        
        _section_size = struct.unpack('<I', data[pos:pos+4])[0]  # noqa: F841
        num_properties = struct.unpack('<I', data[pos+4:pos+8])[0]
        pos += 8
        
        # Property ID/Offset 쌍 읽기
        properties = []
        for _ in range(min(num_properties, 50)):
            if len(data) < pos + 8:
                break
            prop_id = struct.unpack('<I', data[pos:pos+4])[0]
            prop_offset = struct.unpack('<I', data[pos+4:pos+8])[0]
            properties.append((prop_id, prop_offset))
            pos += 8
        
        # Property 값 읽기
        for prop_id, prop_offset in properties:
            abs_offset = section_offset + prop_offset
            if abs_offset + 4 >= len(data):
                continue
            
            prop_type = struct.unpack('<I', data[abs_offset:abs_offset+4])[0]
            value_offset = abs_offset + 4
            
            value = None
            
            if prop_type == 0x1E:  # ANSI String
                if value_offset + 4 < len(data):
                    str_len = struct.unpack('<I', data[value_offset:value_offset+4])[0]
                    if str_len > 0 and value_offset + 4 + str_len <= len(data):
                        try:
                            value = data[value_offset+4:value_offset+4+str_len].decode('cp949', errors='ignore').rstrip('\x00')
                        except Exception:
                            value = data[value_offset+4:value_offset+4+str_len].decode('utf-8', errors='ignore').rstrip('\x00')
            
            elif prop_type == 0x1F:  # Unicode String
                if value_offset + 4 < len(data):
                    str_len = struct.unpack('<I', data[value_offset:value_offset+4])[0]
                    byte_len = str_len * 2
                    if str_len > 0 and value_offset + 4 + byte_len <= len(data):
                        value = data[value_offset+4:value_offset+4+byte_len].decode('utf-16le', errors='ignore').rstrip('\x00')
            
            elif prop_type == 0x40:  # FILETIME
                if value_offset + 8 <= len(data):
                    filetime = struct.unpack('<Q', data[value_offset:value_offset+8])[0]
                    if filetime > 0:
                        try:
                            seconds = filetime / 10000000
                            epoch_diff = 11644473600
                            unix_time = seconds - epoch_diff
                            if 0 < unix_time < 2000000000:
                                value = datetime.fromtimestamp(unix_time)
                        except Exception:
                            pass
            
            # Property ID 매핑
            if value:
                if prop_id == 0x02:
                    metadata['title'] = value
                elif prop_id == 0x03:
                    metadata['subject'] = value
                elif prop_id == 0x04:
                    metadata['author'] = value
                elif prop_id == 0x05:
                    metadata['keywords'] = value
                elif prop_id == 0x06:
                    metadata['comments'] = value
                elif prop_id == 0x08:
                    metadata['last_saved_by'] = value
                elif prop_id == 0x0C:
                    metadata['create_time'] = value
                elif prop_id == 0x0D:
                    metadata['last_saved_time'] = value
    
    except Exception as e:
        logger.debug(f"Error parsing HWP summary information: {e}")
    
    return metadata


def format_metadata(metadata: Dict[str, Any]) -> str:
    """
    메타데이터 딕셔너리를 읽기 쉬운 문자열로 포맷팅합니다.
    
    Args:
        metadata: 메타데이터 딕셔너리
        
    Returns:
        포맷팅된 메타데이터 문자열
    """
    if not metadata:
        return ""
    
    lines = ["<Document-Metadata>"]
    
    field_names = {
        'title': '제목',
        'subject': '주제',
        'author': '작성자',
        'keywords': '키워드',
        'comments': '설명',
        'last_saved_by': '마지막 저장자',
        'create_time': '작성일',
        'last_saved_time': '수정일',
    }
    
    for key, label in field_names.items():
        if key in metadata and metadata[key]:
            value = metadata[key]
            
            # Format datetime objects
            if isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            
            lines.append(f"  {label}: {value}")
    
    lines.append("</Document-Metadata>")
    
    return "\n".join(lines)


# 하위 호환성을 위한 클래스 래퍼
class MetadataHelper:
    """메타데이터 처리 관련 유틸리티 (하위 호환성)"""
    
    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> str:
        return format_metadata(metadata)


__all__ = [
    'extract_metadata',
    'parse_hwp_summary_information',
    'format_metadata',
    'MetadataHelper',
]
