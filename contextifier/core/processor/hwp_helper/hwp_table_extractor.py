# contextifier/core/processor/hwp_helper/hwp_table_extractor.py
"""
HWP Legacy (2.0/3.0) Table Extractor

Implements table extraction for HWP 2.0/3.0 legacy format files.
Follows BaseTableExtractor interface from table_extractor.py.

HWP 2.0/3.0 Table Detection:
- No explicit table metadata in file format
- Uses heuristic pattern recognition
- Table cells are stored as separate sections
- Header patterns: 구분, 내용, 비고, 항목, 설명

2-Pass Approach:
1. Pass 1: Detect table regions from section sequences
2. Pass 2: Extract content and build TableData objects

Usage:
    from contextifier.core.processor.hwp_helper.hwp_table_extractor import (
        HWPLegacyTableExtractor,
    )

    extractor = HWPLegacyTableExtractor()
    
    # From parsed sections
    regions = extractor.detect_table_regions(sections)
    for region in regions:
        table = extractor.extract_table_from_region(sections, region)
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from contextifier.core.functions.table_extractor import (
    BaseTableExtractor,
    TableCell,
    TableData,
    TableRegion,
    TableExtractorConfig,
)

logger = logging.getLogger("document-processor.HWP-Legacy")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HWPLegacyTableExtractorConfig(TableExtractorConfig):
    """Configuration specific to HWP legacy table extraction.
    
    Attributes:
        min_cells_for_table: Minimum number of cells to consider as table
        detect_header_patterns: Whether to use header pattern detection
        header_keywords: Keywords that indicate table headers
    """
    min_cells_for_table: int = 3
    detect_header_patterns: bool = True
    header_keywords: tuple = ('구분', '내용', '비고', '항목', '설명', '분류')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HWPLegacySection:
    """Represents a section in HWP 2.0/3.0 file.
    
    This mirrors the HWPSection from hwp_legacy_parser.py.
    """
    index: int
    section_type: str  # 'TEXT_CELL', 'FONT_DEF', 'STRUCT_MARKER', 'OTHER'
    text: str
    raw_data: bytes = b''
    length: int = 0


@dataclass
class HWPLegacyTableRegionInfo:
    """Additional information for HWP legacy table region.
    
    Stores reference to the section sequence and parsed metadata
    for use in Pass 2 extraction.
    
    Attributes:
        sections: List of HWPLegacySection objects forming this table
        cell_texts: Extracted cell text contents
        detected_cols: Detected column count
    """
    sections: List[HWPLegacySection] = field(default_factory=list)
    cell_texts: List[str] = field(default_factory=list)
    detected_cols: int = 0


# ============================================================================
# HWP Legacy Table Extractor Class
# ============================================================================

class HWPLegacyTableExtractor(BaseTableExtractor):
    """HWP legacy (2.0/3.0) format-specific table extractor.
    
    Extracts tables from HWP 2.0/3.0 files using heuristic pattern recognition.
    Implements BaseTableExtractor interface from table_extractor.py.
    
    HWP 2.0/3.0 files do not have explicit table structure metadata.
    Tables are detected by analyzing section sequences and patterns:
    - Short text cells appearing in sequence
    - Header patterns (구분, 내용, 비고)
    - Regular interval patterns
    
    Supports:
    - Heuristic table detection
    - Header row detection
    - Column count estimation
    """
    
    def __init__(self, config: Optional[HWPLegacyTableExtractorConfig] = None):
        """Initialize HWP legacy table extractor.
        
        Args:
            config: HWP legacy table extraction configuration
        """
        self._config = config or HWPLegacyTableExtractorConfig()
        super().__init__(self._config)
        # Cache for region info (index -> HWPLegacyTableRegionInfo)
        self._region_info_cache: Dict[int, HWPLegacyTableRegionInfo] = {}
    
    # ========================================================================
    # Pass 1: Region Detection
    # ========================================================================
    
    def detect_table_regions(
        self,
        content: Any,
        **kwargs
    ) -> List[TableRegion]:
        """Detect table regions in HWP legacy section list.
        
        Analyzes section sequences to find table-like patterns.
        
        Args:
            content: List of HWPLegacySection or similar section objects
            **kwargs: Additional options
            
        Returns:
            List of TableRegion objects for detected tables
        """
        regions = []
        self._region_info_cache.clear()
        
        if not isinstance(content, list):
            return regions
        
        # Convert to HWPLegacySection if needed
        sections = self._normalize_sections(content)
        
        # Find potential table cell sequences
        cell_sequences = self._find_cell_sequences(sections)
        
        # Build table regions from sequences
        for idx, sequence in enumerate(cell_sequences):
            region_info = self._analyze_sequence(sequence)
            
            if region_info.detected_cols < 2:
                continue
            
            num_rows = (len(region_info.cell_texts) + region_info.detected_cols - 1) // region_info.detected_cols
            
            # Calculate confidence
            confidence = self._calculate_confidence(region_info)
            
            region = TableRegion(
                start_offset=sequence[0].index,
                end_offset=sequence[-1].index,
                row_count=num_rows,
                col_count=region_info.detected_cols,
                confidence=confidence,
                metadata={
                    'cell_count': len(region_info.cell_texts),
                    'has_header': self._has_header_pattern(region_info.cell_texts),
                }
            )
            regions.append(region)
            
            # Cache for Pass 2
            self._region_info_cache[idx] = region_info
        
        logger.debug(f"Detected {len(regions)} table regions in HWP legacy file")
        return regions
    
    def _normalize_sections(self, sections: List[Any]) -> List[HWPLegacySection]:
        """Convert sections to HWPLegacySection objects.
        
        Args:
            sections: List of section objects
            
        Returns:
            List of HWPLegacySection objects
        """
        result = []
        for i, section in enumerate(sections):
            if isinstance(section, HWPLegacySection):
                result.append(section)
            elif hasattr(section, 'index') and hasattr(section, 'section_type'):
                # Convert from hwp_legacy_parser.HWPSection
                result.append(HWPLegacySection(
                    index=section.index,
                    section_type=section.section_type,
                    text=getattr(section, 'text', ''),
                    raw_data=getattr(section, 'raw_data', b''),
                    length=getattr(section, 'length', 0),
                ))
            else:
                # Create from dict or simple object
                result.append(HWPLegacySection(
                    index=i,
                    section_type='OTHER',
                    text=str(section) if section else '',
                ))
        return result
    
    def _find_cell_sequences(self, sections: List[HWPLegacySection]) -> List[List[HWPLegacySection]]:
        """Find sequences of sections that look like table cells.
        
        Args:
            sections: List of sections
            
        Returns:
            List of section sequences
        """
        sequences = []
        current_sequence = []
        
        for section in sections:
            if section.section_type == 'TEXT_CELL' and section.text:
                if self._is_table_cell_candidate(section.text):
                    current_sequence.append(section)
                else:
                    if len(current_sequence) >= self._config.min_cells_for_table:
                        sequences.append(current_sequence)
                    current_sequence = []
        
        if len(current_sequence) >= self._config.min_cells_for_table:
            sequences.append(current_sequence)
        
        return sequences
    
    def _is_table_cell_candidate(self, text: str) -> bool:
        """Check if text looks like a table cell.
        
        Args:
            text: Cell text content
            
        Returns:
            True if likely a table cell
        """
        if not text:
            return False
        
        # Table cells are typically short
        is_short = len(text) < 150
        
        # Don't start with bullet points
        no_bullet = not text.startswith(('-', 'l', '•', '※', '*'))
        
        # Don't start with numbering
        no_numbering = not re.match(r'^\d+[\.\)]\s', text)
        
        # Special handling for bracket-starting texts
        no_bracket_start = not text.startswith('[') or any(k in text for k in ['구매', '지원', '항목'])
        
        return is_short and no_bullet and no_numbering and no_bracket_start
    
    def _analyze_sequence(self, sequence: List[HWPLegacySection]) -> HWPLegacyTableRegionInfo:
        """Analyze a sequence to determine table structure.
        
        Args:
            sequence: List of sections
            
        Returns:
            HWPLegacyTableRegionInfo with analysis results
        """
        cell_texts = [s.text for s in sequence]
        num_cols = self._detect_column_count(cell_texts)
        
        return HWPLegacyTableRegionInfo(
            sections=sequence,
            cell_texts=cell_texts,
            detected_cols=num_cols,
        )
    
    def _detect_column_count(self, cells: List[str]) -> int:
        """Detect the number of columns in a table.
        
        Args:
            cells: List of cell texts
            
        Returns:
            Estimated column count
        """
        if len(cells) < 3:
            return 0
        
        # Check for common header patterns
        header_patterns = [
            ['구분', '내용', '비고'],
            ['구분', '내용'],
            ['항목', '내용', '비고'],
            ['항목', '설명'],
            ['분류', '내용'],
        ]
        
        for pattern in header_patterns:
            if len(cells) >= len(pattern):
                match_count = sum(
                    1 for i, p in enumerate(pattern) 
                    if i < len(cells) and p in cells[i]
                )
                if match_count >= len(pattern) - 1:
                    return len(pattern)
        
        # Analyze cell lengths to detect pattern
        short_cells_indices = [i for i, c in enumerate(cells[:10]) if len(c) < 20]
        
        if len(short_cells_indices) >= 3:
            diffs = [
                short_cells_indices[i+1] - short_cells_indices[i] 
                for i in range(len(short_cells_indices)-1)
            ]
            if diffs and all(d == diffs[0] for d in diffs):
                return diffs[0]
        
        # Default to 3 columns if header-like patterns found
        first_cells = cells[:5]
        has_header_words = any(
            any(h in c for h in self._config.header_keywords)
            for c in first_cells
        )
        
        if has_header_words:
            return 3
        
        return 0
    
    def _has_header_pattern(self, cells: List[str]) -> bool:
        """Check if cells contain header pattern.
        
        Args:
            cells: List of cell texts
            
        Returns:
            True if header pattern detected
        """
        if not cells:
            return False
        
        first_cells = cells[:5]
        return any(
            any(h in c for h in self._config.header_keywords)
            for c in first_cells
        )
    
    def _calculate_confidence(self, info: HWPLegacyTableRegionInfo) -> float:
        """Calculate confidence score for table detection.
        
        Args:
            info: Table region info
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Bonus for header patterns
        if self._has_header_pattern(info.cell_texts):
            confidence += 0.3
        
        # Bonus for regular column count
        if info.detected_cols >= 2:
            confidence += 0.1
        
        # Bonus for more cells
        if len(info.cell_texts) >= 6:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    # ========================================================================
    # Pass 2: Table Extraction
    # ========================================================================
    
    def extract_table_from_region(
        self,
        content: Any,
        region: TableRegion,
        **kwargs
    ) -> Optional[TableData]:
        """Extract table content from detected region.
        
        Args:
            content: List of sections (not used, info is cached)
            region: TableRegion from Pass 1
            **kwargs: Additional options
            
        Returns:
            TableData object with extracted content, or None
        """
        # Find cached info by matching region
        info = None
        for idx, cached_info in self._region_info_cache.items():
            if (cached_info.sections and 
                cached_info.sections[0].index == region.start_offset):
                info = cached_info
                break
        
        if info is None:
            logger.warning(f"Region info not found for offset {region.start_offset}")
            return None
        
        cells = info.cell_texts
        num_cols = info.detected_cols
        num_rows = (len(cells) + num_cols - 1) // num_cols
        
        # Build TableData
        rows = []
        for r in range(num_rows):
            row_cells = []
            for c in range(num_cols):
                cell_idx = r * num_cols + c
                if cell_idx < len(cells):
                    cell = TableCell(
                        content=cells[cell_idx],
                        row_span=1,
                        col_span=1,
                        is_header=(r == 0),
                    )
                else:
                    cell = TableCell(content="")
                row_cells.append(cell)
            rows.append(row_cells)
        
        return TableData(
            rows=rows,
            num_rows=num_rows,
            num_cols=num_cols,
            has_header=self._has_header_pattern(cells),
            metadata={
                'source': 'hwp_legacy',
                'cell_count': len(cells),
            }
        )


# ============================================================================
# Convenience Aliases
# ============================================================================

# Alias for backward compatibility
HWPTableExtractor = HWPLegacyTableExtractor
HWPTableExtractorConfig = HWPLegacyTableExtractorConfig


__all__ = [
    'HWPLegacyTableExtractor',
    'HWPLegacyTableExtractorConfig',
    'HWPLegacySection',
    'HWPLegacyTableRegionInfo',
    # Aliases
    'HWPTableExtractor',
    'HWPTableExtractorConfig',
]
