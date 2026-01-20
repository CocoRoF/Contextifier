# libs/core/functions/page_tag_processor.py
"""
Page Tag Processor Module

Provides functionality for generating and parsing page/slide/sheet markers in extracted text.
This module standardizes page numbering format across all document handlers.

=== Architecture Overview ===

1. Creation:
   - PageTagProcessor instance is created when DocumentProcessor is initialized.
   - Created via DocumentProcessor.__init__() calling _create_page_tag_processor() method.

2. Propagation:
   - The created PageTagProcessor is passed to ALL handlers.
   - In DocumentProcessor._get_handler_registry(), each handler is created with
     page_tag_processor=self._page_tag_processor parameter.
   - Even handlers that don't use page tags receive it for consistency.

3. Access from Handlers:
   - Each Handler inherits from BaseHandler and can access via self.page_tag_processor.
   - Convenience methods: self.create_page_tag(n), self.create_slide_tag(n), self.create_sheet_tag(name)

4. Components:
   - PageTagConfig: Dataclass holding tag prefix/suffix settings
   - PageTagProcessor: Main class for tag generation and parsing
   - PageTagType: Enum distinguishing PAGE, SLIDE, SHEET types

=== Usage Examples ===

    # Custom settings at DocumentProcessor level
    from libs.core.document_processor import DocumentProcessor
    
    processor = DocumentProcessor(
        page_tag_prefix="<page>",
        page_tag_suffix="</page>",
        slide_tag_prefix="<slide>",
        slide_tag_suffix="</slide>"
    )
    
    # Usage inside Handler (BaseHandler subclass)
    class MyHandler(BaseHandler):
        def extract_text(self, ...):
            tag = self.create_page_tag(1)  # "[Page Number: 1]" or custom format
            slide_tag = self.create_slide_tag(1)  # "[Slide Number: 1]"
            sheet_tag = self.create_sheet_tag("Sheet1")  # "[Sheet: Sheet1]"

=== Default Tag Formats ===

    - Page: [Page Number: 1]
    - Slide: [Slide Number: 1]
    - Sheet: [Sheet: Sheet1]

=== Supported Handlers ===

    - PDFHandler: Uses create_page_tag()
    - DOCXHandler: Uses create_page_tag()
    - DOCHandler: Uses create_page_tag()
    - PPTHandler: Uses create_slide_tag()
    - ExcelHandler: Uses create_sheet_tag()
    - HWPHandler, HWPXHandler, CSVHandler, TextHandler: Propagated but not used

"""
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Pattern, Tuple

logger = logging.getLogger("document-processor")


class PageTagType(Enum):
    """Type of page tag for different document formats."""
    PAGE = "page"       # PDF, DOCX, DOC, HWP
    SLIDE = "slide"     # PPT, PPTX
    SHEET = "sheet"     # Excel (XLSX, XLS)


@dataclass
class PageTagConfig:
    """
    PageTagProcessor configuration.

    Attributes:
        tag_prefix: Tag prefix (e.g., "[Page Number: ")
        tag_suffix: Tag suffix (e.g., "]")
        slide_prefix: Slide tag prefix for presentations (e.g., "[Slide Number: ")
        slide_suffix: Slide tag suffix (e.g., "]")
        sheet_prefix: Sheet tag prefix for spreadsheets (e.g., "[Sheet: ")
        sheet_suffix: Sheet tag suffix (e.g., "]")
    """
    tag_prefix: str = "[Page Number: "
    tag_suffix: str = "]"
    slide_prefix: str = "[Slide Number: "
    slide_suffix: str = "]"
    sheet_prefix: str = "[Sheet: "
    sheet_suffix: str = "]"


class PageTagProcessor:
    """
    Page Tag Processor Class

    Generates and parses page/slide/sheet markers for document text extraction.
    Provides a standardized interface for all document handlers.

    Args:
        tag_prefix: Page tag prefix (default: "[Page Number: ")
        tag_suffix: Page tag suffix (default: "]")
        slide_prefix: Slide tag prefix (default: "[Slide Number: ")
        slide_suffix: Slide tag suffix (default: "]")
        sheet_prefix: Sheet tag prefix (default: "[Sheet: ")
        sheet_suffix: Sheet tag suffix (default: "]")
        config: PageTagConfig instance (overrides individual parameters)
    """

    def __init__(
        self,
        tag_prefix: Optional[str] = None,
        tag_suffix: Optional[str] = None,
        slide_prefix: Optional[str] = None,
        slide_suffix: Optional[str] = None,
        sheet_prefix: Optional[str] = None,
        sheet_suffix: Optional[str] = None,
        config: Optional[PageTagConfig] = None
    ):
        """Initialize PageTagProcessor with configuration."""
        if config is not None:
            self._config = config
        else:
            self._config = PageTagConfig(
                tag_prefix=tag_prefix if tag_prefix is not None else PageTagConfig.tag_prefix,
                tag_suffix=tag_suffix if tag_suffix is not None else PageTagConfig.tag_suffix,
                slide_prefix=slide_prefix if slide_prefix is not None else PageTagConfig.slide_prefix,
                slide_suffix=slide_suffix if slide_suffix is not None else PageTagConfig.slide_suffix,
                sheet_prefix=sheet_prefix if sheet_prefix is not None else PageTagConfig.sheet_prefix,
                sheet_suffix=sheet_suffix if sheet_suffix is not None else PageTagConfig.sheet_suffix,
            )

        # Pre-compile regex patterns for parsing
        self._page_pattern: Optional[Pattern] = None
        self._slide_pattern: Optional[Pattern] = None
        self._sheet_pattern: Optional[Pattern] = None

    @property
    def config(self) -> PageTagConfig:
        """Current configuration."""
        return self._config

    @property
    def page_pattern(self) -> Pattern:
        """Compiled regex pattern for matching page tags."""
        if self._page_pattern is None:
            escaped_prefix = re.escape(self._config.tag_prefix)
            escaped_suffix = re.escape(self._config.tag_suffix)
            self._page_pattern = re.compile(
                f'{escaped_prefix}(\\d+){escaped_suffix}',
                re.IGNORECASE
            )
        return self._page_pattern

    @property
    def slide_pattern(self) -> Pattern:
        """Compiled regex pattern for matching slide tags."""
        if self._slide_pattern is None:
            escaped_prefix = re.escape(self._config.slide_prefix)
            escaped_suffix = re.escape(self._config.slide_suffix)
            self._slide_pattern = re.compile(
                f'{escaped_prefix}(\\d+){escaped_suffix}',
                re.IGNORECASE
            )
        return self._slide_pattern

    @property
    def sheet_pattern(self) -> Pattern:
        """Compiled regex pattern for matching sheet tags."""
        if self._sheet_pattern is None:
            escaped_prefix = re.escape(self._config.sheet_prefix)
            escaped_suffix = re.escape(self._config.sheet_suffix)
            self._sheet_pattern = re.compile(
                f'{escaped_prefix}([^\\]]+){escaped_suffix}',
                re.IGNORECASE
            )
        return self._sheet_pattern

    def create_tag(self, page_number: int, tag_type: PageTagType = PageTagType.PAGE) -> str:
        """
        Create a page/slide/sheet tag.

        Args:
            page_number: Page, slide, or sheet number
            tag_type: Type of tag (PAGE, SLIDE, SHEET)

        Returns:
            Formatted tag string

        Example:
            >>> processor = PageTagProcessor()
            >>> processor.create_tag(1)
            '[Page Number: 1]'
            >>> processor.create_tag(1, PageTagType.SLIDE)
            '[Slide Number: 1]'
        """
        if tag_type == PageTagType.SLIDE:
            return f"{self._config.slide_prefix}{page_number}{self._config.slide_suffix}"
        elif tag_type == PageTagType.SHEET:
            return f"{self._config.sheet_prefix}{page_number}{self._config.sheet_suffix}"
        else:
            return f"{self._config.tag_prefix}{page_number}{self._config.tag_suffix}"

    def create_page_tag(self, page_number: int) -> str:
        """
        Create a page tag (convenience method).

        Args:
            page_number: Page number

        Returns:
            Formatted page tag string
        """
        return self.create_tag(page_number, PageTagType.PAGE)

    def create_slide_tag(self, slide_number: int) -> str:
        """
        Create a slide tag (convenience method).

        Args:
            slide_number: Slide number

        Returns:
            Formatted slide tag string
        """
        return self.create_tag(slide_number, PageTagType.SLIDE)

    def create_sheet_tag(self, sheet_name: str) -> str:
        """
        Create a sheet tag with name.

        Args:
            sheet_name: Sheet name

        Returns:
            Formatted sheet tag string
        """
        return f"{self._config.sheet_prefix}{sheet_name}{self._config.sheet_suffix}"

    def find_page_numbers(self, text: str) -> List[Tuple[int, int, int]]:
        """
        Find all page numbers in text.

        Args:
            text: Text to search

        Returns:
            List of tuples: (page_number, start_pos, end_pos)
        """
        results = []
        for match in self.page_pattern.finditer(text):
            page_num = int(match.group(1))
            results.append((page_num, match.start(), match.end()))
        return results

    def find_slide_numbers(self, text: str) -> List[Tuple[int, int, int]]:
        """
        Find all slide numbers in text.

        Args:
            text: Text to search

        Returns:
            List of tuples: (slide_number, start_pos, end_pos)
        """
        results = []
        for match in self.slide_pattern.finditer(text):
            slide_num = int(match.group(1))
            results.append((slide_num, match.start(), match.end()))
        return results

    def has_page_markers(self, text: str) -> bool:
        """
        Check if text contains page markers.

        Args:
            text: Text to check

        Returns:
            True if page markers found
        """
        return bool(self.page_pattern.search(text))

    def has_slide_markers(self, text: str) -> bool:
        """
        Check if text contains slide markers.

        Args:
            text: Text to check

        Returns:
            True if slide markers found
        """
        return bool(self.slide_pattern.search(text))

    def get_pattern_string(self, tag_type: PageTagType = PageTagType.PAGE) -> str:
        """
        Get the regex pattern string for the specified tag type.

        Args:
            tag_type: Type of tag

        Returns:
            Regex pattern string
        """
        if tag_type == PageTagType.SLIDE:
            escaped_prefix = re.escape(self._config.slide_prefix)
            escaped_suffix = re.escape(self._config.slide_suffix)
        elif tag_type == PageTagType.SHEET:
            escaped_prefix = re.escape(self._config.sheet_prefix)
            escaped_suffix = re.escape(self._config.sheet_suffix)
        else:
            escaped_prefix = re.escape(self._config.tag_prefix)
            escaped_suffix = re.escape(self._config.tag_suffix)

        return f'{escaped_prefix}(\\d+){escaped_suffix}'

    def remove_page_markers(self, text: str) -> str:
        """
        Remove all page markers from text.

        Args:
            text: Text with page markers

        Returns:
            Text with page markers removed
        """
        text = self.page_pattern.sub('', text)
        text = self.slide_pattern.sub('', text)
        return text

    def __repr__(self) -> str:
        return (
            f"PageTagProcessor(tag_prefix={self._config.tag_prefix!r}, "
            f"tag_suffix={self._config.tag_suffix!r})"
        )


# Default instance for convenience
_default_processor: Optional[PageTagProcessor] = None


def get_default_page_tag_processor() -> PageTagProcessor:
    """Get the default PageTagProcessor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = PageTagProcessor()
    return _default_processor


def create_page_tag(page_number: int) -> str:
    """
    Create a page tag using the default processor.

    Args:
        page_number: Page number

    Returns:
        Formatted page tag string
    """
    return get_default_page_tag_processor().create_page_tag(page_number)


def create_slide_tag(slide_number: int) -> str:
    """
    Create a slide tag using the default processor.

    Args:
        slide_number: Slide number

    Returns:
        Formatted slide tag string
    """
    return get_default_page_tag_processor().create_slide_tag(slide_number)


__all__ = [
    "PageTagType",
    "PageTagConfig",
    "PageTagProcessor",
    "get_default_page_tag_processor",
    "create_page_tag",
    "create_slide_tag",
]
