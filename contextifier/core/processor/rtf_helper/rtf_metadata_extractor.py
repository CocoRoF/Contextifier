# contextifier/core/processor/rtf_helper/rtf_metadata_extractor.py
"""
RTF Metadata Extractor

Extracts metadata from RTF content.
Implements BaseMetadataExtractor interface.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Union

from contextifier.core.functions.metadata_extractor import (
    BaseMetadataExtractor,
    DocumentMetadata,
)
from contextifier.core.processor.rtf_helper.rtf_decoder import (
    decode_hex_escapes,
)
from contextifier.core.processor.rtf_helper.rtf_text_cleaner import (
    clean_rtf_text,
)

logger = logging.getLogger("contextify.rtf.metadata")


@dataclass
class RTFSourceInfo:
    """
    Source information for RTF metadata extraction.
    
    Container for data passed to RTFMetadataExtractor.extract().
    """
    content: str
    encoding: str = "cp949"


class RTFMetadataExtractor(BaseMetadataExtractor):
    """
    RTF Metadata Extractor.
    
    Extracts metadata from RTF content.
    
    Supported fields:
    - title, subject, author, keywords, comments
    - last_saved_by, create_time, last_saved_time
    
    Usage:
        extractor = RTFMetadataExtractor()
        source = RTFSourceInfo(content=rtf_content, encoding="cp949")
        metadata = extractor.extract(source)
        text = extractor.format(metadata)
    """
    
    def extract(self, source: Union[RTFSourceInfo, Dict[str, Any]]) -> DocumentMetadata:
        """
        Extract metadata from RTF content.
        
        Args:
            source: RTFSourceInfo object (content string and encoding)
                    OR Dict[str, Any] (pre-parsed metadata)
            
        Returns:
            DocumentMetadata instance
        """
        if isinstance(source, dict):
            return self._from_dict(source)
        
        content = source.content
        encoding = source.encoding
        
        title = None
        subject = None
        author = None
        keywords = None
        comments = None
        last_saved_by = None
        create_time = None
        last_saved_time = None
        
        # Find \info group
        info_match = re.search(r'\\info\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', content)
        if info_match:
            info_content = info_match.group(1)
            
            # Extract each metadata field
            field_patterns = {
                'title': r'\\title\s*\{([^}]*)\}',
                'subject': r'\\subject\s*\{([^}]*)\}',
                'author': r'\\author\s*\{([^}]*)\}',
                'keywords': r'\\keywords\s*\{([^}]*)\}',
                'comments': r'\\doccomm\s*\{([^}]*)\}',
                'last_saved_by': r'\\operator\s*\{([^}]*)\}',
            }
            
            for key, pattern in field_patterns.items():
                match = re.search(pattern, info_content)
                if match:
                    value = decode_hex_escapes(match.group(1), encoding)
                    value = clean_rtf_text(value, encoding)
                    if value:
                        if key == 'title':
                            title = value
                        elif key == 'subject':
                            subject = value
                        elif key == 'author':
                            author = value
                        elif key == 'keywords':
                            keywords = value
                        elif key == 'comments':
                            comments = value
                        elif key == 'last_saved_by':
                            last_saved_by = value
            
            # Extract dates
            create_time = self._extract_date(
                content, 
                r'\\creatim\\yr(\d+)\\mo(\d+)\\dy(\d+)(?:\\hr(\d+))?(?:\\min(\d+))?'
            )
            last_saved_time = self._extract_date(
                content,
                r'\\revtim\\yr(\d+)\\mo(\d+)\\dy(\d+)(?:\\hr(\d+))?(?:\\min(\d+))?'
            )
        
        self.logger.debug("Extracted RTF metadata fields")
        
        return DocumentMetadata(
            title=title,
            subject=subject,
            author=author,
            keywords=keywords,
            comments=comments,
            last_saved_by=last_saved_by,
            create_time=create_time,
            last_saved_time=last_saved_time,
        )
    
    def _extract_date(self, content: str, pattern: str) -> Optional[datetime]:
        """Extract datetime from RTF date pattern."""
        match = re.search(pattern, content)
        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                hour = int(match.group(4)) if match.group(4) else 0
                minute = int(match.group(5)) if match.group(5) else 0
                return datetime(year, month, day, hour, minute)
            except (ValueError, TypeError):
                pass
        return None
    
    def _from_dict(self, metadata: Dict[str, Any]) -> DocumentMetadata:
        """
        Convert pre-parsed metadata dict to DocumentMetadata.
        
        Args:
            metadata: Pre-parsed metadata dict
            
        Returns:
            DocumentMetadata instance
        """
        return DocumentMetadata(
            title=metadata.get('title'),
            subject=metadata.get('subject'),
            author=metadata.get('author'),
            keywords=metadata.get('keywords'),
            comments=metadata.get('comments'),
            last_saved_by=metadata.get('last_saved_by'),
            create_time=metadata.get('create_time'),
            last_saved_time=metadata.get('last_saved_time'),
        )


__all__ = [
    'RTFMetadataExtractor',
    'RTFSourceInfo',
]
