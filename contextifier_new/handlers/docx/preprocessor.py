"""
DocxPreprocessor — Stage 2: python-docx Document → PreprocessedData.

DOCX documents are fully parsed by python-docx, so the preprocessor
is lightweight. It wraps the ``Document`` object in ``PreprocessedData``
and computes summary statistics (paragraph count, table count, etc.)
that downstream stages can use for diagnostics or optimisation.

If charts exist in the DOCX, they are pre-extracted here (from ZIP-level
``word/charts/*.xml``) and stored in ``resources["charts"]`` so that
the ContentExtractor can match them to inline chart references in
document order.
"""

from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional

from contextifier_new.pipeline.preprocessor import BasePreprocessor
from contextifier_new.types import PreprocessedData
from contextifier_new.errors import PreprocessingError

from contextifier_new.handlers.docx._constants import CHART_TYPE_MAP

logger = logging.getLogger(__name__)

# OOXML chart namespace
_NS_C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


class DocxPreprocessor(BasePreprocessor):
    """
    Preprocessor for DOCX files.

    Wraps the python-docx Document in ``PreprocessedData``, computes
    summary properties, and pre-extracts chart XML from the ZIP archive.
    """

    def preprocess(self, converted_data: Any, **kwargs: Any) -> PreprocessedData:
        """
        Wrap python-docx Document in PreprocessedData.

        Args:
            converted_data: ``python_docx.Document`` from the Converter.

        Returns:
            PreprocessedData with:
            - ``content``: the Document object
            - ``raw_content``: the Document object (same)
            - ``resources["charts"]``: list of pre-extracted chart strings
            - ``properties``: paragraph_count, table_count, chart_count
        """
        doc = converted_data
        if doc is None:
            raise PreprocessingError(
                "Received None as converted data",
                stage="preprocess",
                handler="docx",
            )

        # Summary statistics
        paragraph_count = len(doc.paragraphs) if hasattr(doc, "paragraphs") else 0
        table_count = len(doc.tables) if hasattr(doc, "tables") else 0

        # Pre-extract charts from ZIP
        charts: List[str] = []
        try:
            charts = self._extract_charts_from_zip(doc)
        except Exception as exc:
            logger.debug("Chart pre-extraction failed: %s", exc)

        return PreprocessedData(
            content=doc,
            raw_content=doc,
            encoding="utf-8",
            resources={
                "charts": charts,
            },
            properties={
                "paragraph_count": paragraph_count,
                "table_count": table_count,
                "chart_count": len(charts),
            },
        )

    def get_format_name(self) -> str:
        return "docx"

    # ── Chart pre-extraction ──────────────────────────────────────────────

    def _extract_charts_from_zip(self, doc: Any) -> List[str]:
        """
        Extract chart data from word/charts/*.xml inside the DOCX ZIP.

        Returns a list of formatted chart strings in file order
        (chart1.xml, chart2.xml, ...).  These will be consumed by
        the ContentExtractor in document order.
        """
        # Access the underlying ZIP via python-docx's package
        part = doc.part
        if not hasattr(part, "package") or not hasattr(part.package, "part_related_by"):
            return []

        # Try to read charts from the ZIP directly via the package blob
        zip_stream = self._get_zip_stream(doc)
        if zip_stream is None:
            return []

        charts: List[str] = []
        try:
            with zipfile.ZipFile(zip_stream, "r") as zf:
                chart_files = sorted(
                    name
                    for name in zf.namelist()
                    if name.startswith("word/charts/chart")
                    and name.endswith(".xml")
                )

                for chart_file in chart_files:
                    try:
                        chart_xml = zf.read(chart_file)
                        formatted = self._parse_chart_xml(chart_xml)
                        charts.append(formatted)
                    except Exception as exc:
                        logger.debug("Failed to parse %s: %s", chart_file, exc)
                        charts.append("[Chart]")
        except Exception as exc:
            logger.debug("Failed to read ZIP for charts: %s", exc)

        return charts

    @staticmethod
    def _get_zip_stream(doc: Any) -> Optional[io.BytesIO]:
        """
        Get a seekable BytesIO stream of the DOCX ZIP from the Document.

        python-docx stores the original blob in the package's
        PartFactory; we try multiple access paths.
        """
        # Path 1: via doc.part.package.blob (not always available)
        try:
            blob = doc.part.package.blob
            if blob:
                return io.BytesIO(blob)
        except Exception:
            pass

        # Path 2: via doc.element to reconstruct — not feasible
        # Path 3: return None and skip chart extraction
        return None

    @staticmethod
    def _parse_chart_xml(chart_xml: bytes) -> str:
        """
        Parse a single chart XML and return a formatted string.

        Format: [Chart: <type> - <title>] or [Chart: <type>] or [Chart]
        """
        try:
            root = ET.fromstring(chart_xml)
        except ET.ParseError:
            try:
                text = chart_xml.decode("utf-8-sig", errors="ignore")
                root = ET.fromstring(text)
            except Exception:
                return "[Chart]"

        ns = {"c": _NS_C, "a": _NS_A}

        # Find chart element
        chart_elem = root.find(".//c:chart", ns)
        if chart_elem is None:
            return "[Chart]"

        # Extract title
        title_elem = chart_elem.find(".//c:title//c:tx//c:rich//a:t", ns)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else None

        # Detect chart type
        chart_type = "Chart"
        plot_area = chart_elem.find(".//c:plotArea", ns)
        if plot_area is not None:
            for tag, name in CHART_TYPE_MAP.items():
                if plot_area.find(f".//c:{tag}", ns) is not None:
                    chart_type = name
                    break

        if title:
            return f"[Chart: {chart_type} - {title}]"
        return f"[Chart: {chart_type}]"


__all__ = ["DocxPreprocessor"]
