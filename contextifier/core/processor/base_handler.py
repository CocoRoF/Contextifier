# libs/core/processor/base_handler.py
"""
BaseHandler - Abstract base class for document processing handlers

Defines the base interface for all document handlers.
Manages config, ImageProcessor, PageTagProcessor, and ChartProcessor passed from 
DocumentProcessor at instance level for reuse by internal methods.

Each handler should override _create_chart_extractor() to provide a format-specific
chart extractor implementation.

Usage Example:
    class PDFHandler(BaseHandler):
        def extract_text(self, current_file: CurrentFile, extract_metadata: bool = True) -> str:
            # Access self.config, self.image_processor, self.page_tag_processor
            # Use self.chart_extractor.process(chart_element) for chart extraction
            ...
"""
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from contextifier.core.functions.img_processor import ImageProcessor
from contextifier.core.functions.page_tag_processor import PageTagProcessor
from contextifier.core.functions.chart_processor import ChartProcessor
from contextifier.core.functions.chart_extractor import BaseChartExtractor, NullChartExtractor

if TYPE_CHECKING:
    from contextifier.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


class BaseHandler(ABC):
    """
    Abstract base class for document handlers.
    
    All handlers inherit from this class.
    config, image_processor, page_tag_processor, and chart_processor are passed 
    at creation and stored as instance variables.
    
    Each handler should override _create_chart_extractor() to provide a
    format-specific chart extractor. The chart_extractor is lazy-initialized
    on first access.
    
    Attributes:
        config: Configuration dictionary passed from DocumentProcessor
        image_processor: ImageProcessor instance passed from DocumentProcessor
        page_tag_processor: PageTagProcessor instance passed from DocumentProcessor
        chart_processor: ChartProcessor instance passed from DocumentProcessor
        chart_extractor: Format-specific chart extractor instance
        logger: Logging instance
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        image_processor: Optional[ImageProcessor] = None,
        page_tag_processor: Optional[PageTagProcessor] = None,
        chart_processor: Optional[ChartProcessor] = None
    ):
        """
        Initialize BaseHandler.
        
        Args:
            config: Configuration dictionary (passed from DocumentProcessor)
            image_processor: ImageProcessor instance (passed from DocumentProcessor)
            page_tag_processor: PageTagProcessor instance (passed from DocumentProcessor)
            chart_processor: ChartProcessor instance (passed from DocumentProcessor)
        """
        self._config = config or {}
        self._image_processor = image_processor or ImageProcessor()
        self._page_tag_processor = page_tag_processor or self._get_page_tag_processor_from_config()
        self._chart_processor = chart_processor or self._get_chart_processor_from_config()
        self._chart_extractor: Optional[BaseChartExtractor] = None
        self._logger = logging.getLogger(f"document-processor.{self.__class__.__name__}")
    
    def _get_page_tag_processor_from_config(self) -> PageTagProcessor:
        """Get PageTagProcessor from config or create default."""
        if self._config and "page_tag_processor" in self._config:
            return self._config["page_tag_processor"]
        return PageTagProcessor()
    
    def _get_chart_processor_from_config(self) -> ChartProcessor:
        """Get ChartProcessor from config or create default."""
        if self._config and "chart_processor" in self._config:
            return self._config["chart_processor"]
        return ChartProcessor()
    
    def _create_chart_extractor(self) -> BaseChartExtractor:
        """
        Create format-specific chart extractor.
        
        Override this method in subclasses to provide the appropriate
        chart extractor for the file format.
        
        Returns:
            BaseChartExtractor subclass instance
        """
        return NullChartExtractor(self._chart_processor)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Configuration dictionary."""
        return self._config
    
    @property
    def image_processor(self) -> ImageProcessor:
        """ImageProcessor instance."""
        return self._image_processor
    
    @property
    def page_tag_processor(self) -> PageTagProcessor:
        """PageTagProcessor instance."""
        return self._page_tag_processor
    
    @property
    def chart_processor(self) -> ChartProcessor:
        """ChartProcessor instance."""
        return self._chart_processor
    
    @property
    def chart_extractor(self) -> BaseChartExtractor:
        """
        Format-specific chart extractor (lazy-initialized).
        
        Returns the chart extractor for this handler's file format.
        """
        if self._chart_extractor is None:
            self._chart_extractor = self._create_chart_extractor()
        return self._chart_extractor
    
    @property
    def logger(self) -> logging.Logger:
        """Logger instance."""
        return self._logger
    
    @abstractmethod
    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """
        Extract text from file.
        
        Args:
            current_file: CurrentFile dict containing file info and binary data
            extract_metadata: Whether to extract metadata
            **kwargs: Additional options
            
        Returns:
            Extracted text
        """
        pass
    
    def get_file_stream(self, current_file: "CurrentFile") -> io.BytesIO:
        """
        Get a fresh BytesIO stream from current_file.
        
        Resets the stream position to the beginning for reuse.
        
        Args:
            current_file: CurrentFile dict
            
        Returns:
            BytesIO stream ready for reading
        """
        stream = current_file.get("file_stream")
        if stream is not None:
            stream.seek(0)
            return stream
        # Fallback: create new stream from file_data
        return io.BytesIO(current_file.get("file_data", b""))
    
    def save_image(self, image_data: bytes, processed_images: Optional[set] = None) -> Optional[str]:
        """
        Save image and return tag.
        
        Convenience method that wraps self.image_processor.save_image().
        
        Args:
            image_data: Image binary data
            processed_images: Set of processed image hashes (for deduplication)
            
        Returns:
            Image tag string or None
        """
        return self._image_processor.save_image(image_data, processed_images=processed_images)

    def create_page_tag(self, page_number: int) -> str:
        """
        Create a page number tag.
        
        Convenience method that wraps self.page_tag_processor.create_page_tag().
        
        Args:
            page_number: Page number
            
        Returns:
            Page tag string (e.g., "[Page Number: 1]")
        """
        return self._page_tag_processor.create_page_tag(page_number)

    def create_slide_tag(self, slide_number: int) -> str:
        """
        Create a slide number tag.
        
        Convenience method that wraps self.page_tag_processor.create_slide_tag().
        
        Args:
            slide_number: Slide number
            
        Returns:
            Slide tag string (e.g., "[Slide Number: 1]")
        """
        return self._page_tag_processor.create_slide_tag(slide_number)

    def create_sheet_tag(self, sheet_name: str) -> str:
        """
        Create a sheet name tag.
        
        Convenience method that wraps self.page_tag_processor.create_sheet_tag().
        
        Args:
            sheet_name: Sheet name
            
        Returns:
            Sheet tag string (e.g., "[Sheet: Sheet1]")
        """
        return self._page_tag_processor.create_sheet_tag(sheet_name)

    def process_chart(self, chart_element: Any) -> str:
        """
        Process chart element using the format-specific chart extractor.
        
        This is the main method for chart processing. It uses the chart_extractor
        to extract data from the format-specific chart element and formats it
        using ChartProcessor.
        
        Args:
            chart_element: Format-specific chart object/element
            
        Returns:
            Formatted chart text with tags
        """
        return self.chart_extractor.process(chart_element)


__all__ = ["BaseHandler"]
