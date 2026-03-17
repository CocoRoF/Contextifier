# contextifier_new/pipeline/preprocessor.py
"""
BasePreprocessor — Stage 2: Clean / Transform

Responsible for:
- Taking the converted format object from Stage 1
- Cleaning, normalizing, or transforming it
- Extracting embedded resources (images, charts) if needed
- Producing a PreprocessedData result

Examples per format:
- PDF: page analysis, complexity scoring
- DOCX: namespace cleanup, style normalization
- RTF: control code removal, image extraction
- Text: encoding detection, BOM removal
- CSV: delimiter detection, encoding normalization

Contract:
- preprocess() is the ONLY abstract method
- validate() pre-checks input (optional override)
- get_format_name() identifies the format (abstract)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from contextifier_new.types import PreprocessedData
from contextifier_new.errors import PreprocessingError


class BasePreprocessor(ABC):
    """
    Abstract base for all format preprocessors.

    Subclasses implement preprocess() and get_format_name().
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(
            f"contextifier.preprocessor.{self.__class__.__name__}"
        )

    @abstractmethod
    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """
        Preprocess the converted format object.

        This is the SECOND stage of the pipeline:
        1. Converter.convert() → format object
        2. Preprocessor.preprocess() → PreprocessedData   ← THIS
        3. MetadataExtractor.extract()
        4. ContentExtractor.extract()
        5. Postprocessor.postprocess()

        Args:
            converted_data: The object returned by Converter.convert().
                            Type depends on format (Document, Workbook, bytes, etc.)
            **kwargs: Format-specific preprocessing options.

        Returns:
            PreprocessedData containing cleaned content and extracted resources.

        Raises:
            PreprocessingError: If preprocessing fails.
        """
        ...

    @abstractmethod
    def get_format_name(self) -> str:
        """Return the canonical format name."""
        ...

    def validate(self, data: Any) -> bool:
        """
        Pre-check whether the data is suitable for preprocessing.

        Default: always returns True.

        Args:
            data: The converted data to validate.

        Returns:
            True if data can be preprocessed.
        """
        return data is not None


class NullPreprocessor(BasePreprocessor):
    """
    Null preprocessor — wraps input as-is into PreprocessedData.

    Used for formats where the converted object needs no cleaning.
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """Pass through unchanged."""
        return PreprocessedData(
            content=converted_data,
            raw_content=converted_data,
        )

    def get_format_name(self) -> str:
        return "null"


__all__ = ["BasePreprocessor", "NullPreprocessor"]
