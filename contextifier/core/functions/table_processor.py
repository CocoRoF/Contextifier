# libs/core/functions/table_processor.py
"""
Table Processor - Common Table Processing Module

Provides common table processing utilities for formatting tables.
This module handles HTML, Markdown, and Text conversion of TableData.

================================================================================
TABLE PROCESSOR ARCHITECTURE
================================================================================

Main Entry Point:
    format_table(table: TableData) → str

Internal Processing Functions (called from format_table):
    ├─ format_table_as_html()     - HTML 변환 (rowspan/colspan 지원)
    ├─ format_table_as_markdown() - Markdown 변환 (간단한 표)
    └─ format_table_as_text()     - Text 변환 (순수 텍스트)

Common Utility:
    └─ _clean_cell_content()      - 셀 내용 정리 (공백 처리)

================================================================================
PROCESSING FLOW
================================================================================

format_table(table)  ← Main Entry Point
│
├─ config.output_format == HTML?
│   └─► format_table_as_html(table)
│       ├─ colgroup 생성 (열 너비)
│       ├─ 행/셀 순회
│       │   ├─ rowspan/colspan 처리
│       │   ├─ nested_table 재귀 처리
│       │   └─ _clean_cell_content()
│       └─ HTML 문자열 반환
│
├─ config.output_format == MARKDOWN?
│   └─► format_table_as_markdown(table)
│       ├─ 행/셀 순회
│       │   └─ _clean_cell_content()
│       ├─ 헤더 구분선 추가
│       └─ Markdown 문자열 반환
│
└─ config.output_format == TEXT?
    └─► format_table_as_text(table)
        ├─ 행/셀 순회
        │   └─ _clean_cell_content()
        └─ 탭 구분 문자열 반환

================================================================================
OUTPUT FORMAT COMPARISON
================================================================================

| Format   | Use Case                    | Merge Support | Structure |
|----------|-----------------------------|--------------|-----------| 
| HTML     | 웹 렌더링, 문서 변환         | ✅ 완전 지원  | ✅ 보존   |
| Markdown | GitHub, 노션, 문서 편집기    | ❌ 미지원     | ⚠️ 단순화 |
| Text     | 검색 인덱싱, 로깅, 디버깅    | ❌ 미지원     | ❌ 없음   |

================================================================================
"""
import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from contextifier.core.functions.table_extractor import TableData, TableCell

logger = logging.getLogger("document-processor")


class TableOutputFormat(Enum):
    """Table output format options."""
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass
class TableProcessorConfig:
    """Configuration for table processing."""
    output_format: TableOutputFormat = TableOutputFormat.HTML
    clean_whitespace: bool = True
    preserve_merged_cells: bool = True


class TableProcessor:
    """
    Main table processing class.
    
    ============================================================================
    CLASS STRUCTURE
    ============================================================================
    
    Public Methods:
        format_table()            ← Main Entry Point (config.output_format으로 분기)
        format_table_as_html()    ← HTML 변환 (내부: format_table에서 호출)
        format_table_as_markdown()← Markdown 변환 (내부: format_table에서 호출)
        format_table_as_text()    ← Text 변환 (내부: format_table에서 호출)
    
    Private Methods:
        _clean_cell_content()     ← 공통 유틸 (셀 내용 정리)
    
    ============================================================================
    """
    
    def __init__(self, config: Optional[TableProcessorConfig] = None):
        self.config = config or TableProcessorConfig()
        self.logger = logging.getLogger("document-processor")
    
    # ==========================================================================
    # format_table() - Main Entry Point
    # ==========================================================================
    #
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │ format_table(table)                                                 │
    # │                                                                     │
    # │   config.output_format 확인                                         │
    # │       │                                                             │
    # │       ├─ HTML ────────► format_table_as_html()                     │
    # │       │                     ├─ colgroup 생성 (열 너비)              │
    # │       │                     ├─ 행/셀 순회                           │
    # │       │                     │   ├─ rowspan/colspan                 │
    # │       │                     │   ├─ nested_table (재귀)             │
    # │       │                     │   └─ _clean_cell_content()           │
    # │       │                     └─ HTML 반환                            │
    # │       │                                                             │
    # │       ├─ MARKDOWN ──► format_table_as_markdown()                   │
    # │       │                     ├─ 행/셀 순회                           │
    # │       │                     │   └─ _clean_cell_content()           │
    # │       │                     ├─ 헤더 구분선                          │
    # │       │                     └─ Markdown 반환                        │
    # │       │                                                             │
    # │       └─ TEXT ──────► format_table_as_text()                       │
    # │                             ├─ 행/셀 순회                           │
    # │                             │   └─ _clean_cell_content()           │
    # │                             └─ Text 반환                            │
    # └─────────────────────────────────────────────────────────────────────┘
    #
    # ==========================================================================
    
    def format_table(self, table: TableData) -> str:
        """
        Main entry point for table formatting.
        
        Routes to appropriate format handler based on config.output_format.
        
        Args:
            table: TableData from table extractor
            
        Returns:
            Formatted string (HTML/Markdown/Text)
        """
        if self.config.output_format == TableOutputFormat.HTML:
            return self.format_table_as_html(table)
        elif self.config.output_format == TableOutputFormat.MARKDOWN:
            return self.format_table_as_markdown(table)
        else:
            return self.format_table_as_text(table)
    
    # ==========================================================================
    # format_table_as_html() - HTML 변환 (format_table에서 호출)
    # ==========================================================================
    
    def format_table_as_html(self, table: TableData) -> str:
        """
        Convert TableData to HTML string.
        
        Called from format_table() when output_format == HTML.
        
        Features:
        - colgroup for column widths
        - rowspan/colspan for merged cells
        - nested_table support (recursive)
        """
        if not table.rows:
            return ""
        
        html_parts = ["<table>"]
        
        if table.col_widths_percent:
            html_parts.append("  <colgroup>")
            for width_pct in table.col_widths_percent:
                html_parts.append(f'    <col style="width: {width_pct:.1f}%">')
            html_parts.append("  </colgroup>")
        
        for row_idx, row in enumerate(table.rows):
            html_parts.append("  <tr>")
            
            for cell in row:
                tag = "th" if cell.is_header else "td"
                attrs = []
                if self.config.preserve_merged_cells:
                    if cell.row_span > 1:
                        attrs.append(f'rowspan="{cell.row_span}"')
                    if cell.col_span > 1:
                        attrs.append(f'colspan="{cell.col_span}"')
                
                attr_str = " " + " ".join(attrs) if attrs else ""
                
                if cell.nested_table:
                    nested_html = self.format_table_as_html(cell.nested_table)
                    html_parts.append(f"    <{tag}{attr_str}>{nested_html}</{tag}>")
                else:
                    content = self._clean_cell_content(cell.content)
                    html_parts.append(f"    <{tag}{attr_str}>{content}</{tag}>")
            
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        return "\n".join(html_parts)
    
    # ==========================================================================
    # format_table_as_markdown() - Markdown 변환 (format_table에서 호출)
    # ==========================================================================
    
    def format_table_as_markdown(self, table: TableData) -> str:
        """
        Convert TableData to Markdown string.
        
        Called from format_table() when output_format == MARKDOWN.
        
        Note: Markdown does NOT support rowspan/colspan.
        """
        if not table.rows:
            return ""
        
        lines = []
        for row_idx, row in enumerate(table.rows):
            cells = [self._clean_cell_content(cell.content) for cell in row]
            line = "| " + " | ".join(cells) + " |"
            lines.append(line)
            
            if row_idx == 0 and table.has_header:
                separator = "| " + " | ".join(["---"] * len(row)) + " |"
                lines.append(separator)
        
        return "\n".join(lines)
    
    # ==========================================================================
    # format_table_as_text() - Text 변환 (format_table에서 호출)
    # ==========================================================================
    
    def format_table_as_text(self, table: TableData) -> str:
        """
        Convert TableData to plain text string.
        
        Called from format_table() when output_format == TEXT.
        
        Note: No table structure preserved. Useful for search indexing.
        """
        if not table.rows:
            return ""
        
        lines = []
        for row in table.rows:
            cells = [self._clean_cell_content(cell.content) for cell in row]
            lines.append("\t".join(cells))
        
        return "\n".join(lines)
    
    # ==========================================================================
    # _clean_cell_content() - 공통 유틸리티 (모든 format 함수에서 호출)
    # ==========================================================================
    
    def _clean_cell_content(self, content: str) -> str:
        """
        Clean cell content (whitespace normalization).
        
        Called from all format_table_as_* methods.
        """
        if not content:
            return ""
        
        if self.config.clean_whitespace:
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
        
        return content


def create_table_processor(config: Optional[TableProcessorConfig] = None) -> TableProcessor:
    """
    Factory function to create a TableProcessor.
    
    Args:
        config: Table processing configuration
        
    Returns:
        Configured TableProcessor instance
    """
    return TableProcessor(config)


# Default configuration
DEFAULT_PROCESSOR_CONFIG = TableProcessorConfig()
